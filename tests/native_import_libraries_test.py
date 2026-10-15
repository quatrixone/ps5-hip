"""Offline integration test of the pinned converter patch against its unmodified source.

Run after the source-only public HIP build. No console or firmware files.
"""
import base64
import hashlib
import json
import pathlib
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build/public-hip"


def read_imports(path, verify_hash=False):
    data = path.read_bytes()
    phoff = struct.unpack_from("<Q", data, 32)[0]
    stride, count = struct.unpack_from("<HH", data, 54)
    headers = [struct.unpack_from("<IIQQQQQQ", data, phoff + i * stride) for i in range(count)]

    def offset(address):
        for kind, flags, start, va, pa, size, memsize, align in headers:
            if kind == 1 and va <= address < va + size:
                return start + address - va
        raise ValueError("unmapped dynamic address")

    dynamic = next(h for h in headers if h[0] == 2)
    entries = [struct.unpack_from("<QQ", data, p)
               for p in range(dynamic[2], dynamic[2] + dynamic[5], 16)]
    tags = dict(entries)
    strings = offset(tags[5])

    def string(index):
        return data[strings + index:data.index(b"\0", strings + index)].decode()

    modules, libraries, symbols = {}, {}, {}
    for tag, value in entries:
        if tag == 0x61000045:
            modules[value >> 48] = string(value & 0xffffffff)
        if tag == 0x61000049:
            libraries[value >> 48] = string(value & 0xffffffff)
    start = offset(tags[6])
    decoded = {}
    for p in range(start + 24, start + tags[0x6100003f], 24):
        name = string(struct.unpack_from("<I", data, p)[0])
        hashed, library, module = name.split("#")
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
        def number(text):
            value = 0
            for char in text:
                value = value * 64 + alphabet.index(char)
            return value
        symbols[hashed] = (modules[number(module)], libraries[number(library)])
        decoded[(p - start) // 24] = hashed + '#' + libraries[number(library)] + '#' + modules[number(module)]
    if verify_hash:
        table = offset(tags[4])
        buckets, chains = struct.unpack_from('<II', data, table)
        if not buckets or chains != tags[0x6100003f] // 24:
            raise ValueError('invalid import hash dimensions')
        for index, identity in decoded.items():
            hashed = 0
            for byte in identity.encode():
                hashed = ((hashed << 4) + byte) & 0xffffffff
                high = hashed & 0xf0000000
                if high:
                    hashed ^= high >> 24
                hashed &= ~high
            current = struct.unpack_from('<I', data, table + 8 + (hashed % buckets) * 4)[0]
            visited = set()
            while current and current != index:
                if current >= chains or current in visited:
                    raise ValueError('invalid import hash chain')
                visited.add(current)
                current = struct.unpack_from('<I', data, table + 8 + buckets * 4 + current * 4)[0]
            if current != index:
                raise ValueError('serialized import identity absent from hash bucket: ' + identity)
    return modules, libraries, symbols


def nid(symbol):
    suffix = bytes.fromhex("518d64a635ded8c1e6b039b1c3e55230")
    return base64.b64encode(hashlib.sha1(symbol.encode() + suffix).digest()[:8][::-1]).decode().rstrip("=").replace("/", "-")


class ImportLibraries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="ps5hip-imports-")
        cls.work = pathlib.Path(cls.temporary.name)
        lock = json.loads((ROOT / "dependencies.json").read_text())
        native = ROOT / ".deps/sources/native-tools/tooling/native"
        original = cls.work / "original"
        shutil.copytree(native, original)
        zlib = ROOT / ".deps" / ("host-zlib-" + lock["host_zlib"]["version"]) / "root"
        cls.original_tool = cls.work / "original-tool"
        patched = cls.work / 'patched'
        shutil.copytree(original, patched)
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-d', str(patched),
                        '-i', str(ROOT / 'patches/native-import-libraries.patch')], check=True, capture_output=True)
        cls.patched_tool = cls.work / 'patched-tool'
        for source, target in ((original, cls.original_tool), (patched, cls.patched_tool)):
            subprocess.run(["clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                        *[str(source / name) for name in ("native_app_builder.cpp", "elf_object.cpp",
                                                           "sce_module_writer.cpp", "self_container.cpp")],
                        "-I", str(zlib / "include"), str(zlib / "lib/libz.a"), "-o", str(target)],
                       check=True, capture_output=True)
        cls.base = cls.convert(tool=cls.original_tool)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def convert(cls, *bindings, tool=None, success=True, pie=None):
        path = cls.work / "output.elf"
        path.unlink(missing_ok=True)
        command = [str(tool or cls.patched_tool), "link", "--in", str(pie or BUILD / "title.elf"),
                   "--out", str(path), "--stub-dir", str(ROOT / ".deps/sdk/target/lib"),
                   "--stub", str(BUILD / "libSceAgc.so"), "--stub", str(BUILD / "libSceAgcDriver.so"),
                   "--file-name", "eboot.elf"]
        for binding in bindings:
            command += ["--import-library", binding]
        result = subprocess.run(command, capture_output=True, text=True)
        if success:
            if result.returncode:
                raise AssertionError(result.stderr)
            return path.read_bytes()
        if result.returncode == 0 or path.exists():
            raise AssertionError("invalid mapping was accepted or published output")

    def inspect(self, *bindings):
        data = self.convert(*bindings)
        path = self.work / "inspect.elf"
        path.write_bytes(data)
        return read_imports(path, verify_hash=True)

    def test_no_override_is_byte_identical_to_unpatched_converter(self):
        self.assertEqual(self.convert(), self.base)

    def test_serialized_hash_lookup_rejects_missing_bucket(self):
        data = bytearray(self.convert('libSceLibcInternal:exit:libExitProbe'))
        phoff = struct.unpack_from('<Q', data, 32)[0]
        count = struct.unpack_from('<H', data, 56)[0]
        headers = [struct.unpack_from('<IIQQQQQQ', data, phoff + i * 56) for i in range(count)]
        dynamic = next(row for row in headers if row[0] == 2)
        tags = dict(struct.unpack_from('<QQ', data, p)
                    for p in range(dynamic[2], dynamic[2] + dynamic[5], 16))
        load = next(row for row in headers if row[0] == 1 and row[3] <= tags[4] < row[3] + row[5])
        table = load[2] + tags[4] - load[3]
        buckets = struct.unpack_from('<I', data, table)[0]
        data[table + 8:table + 8 + buckets * 4] = bytes(buckets * 4)
        path = self.work / 'corrupt-hash.elf'
        path.write_bytes(data)
        # Symbol/module/library decoding alone still succeeds, so this isolates
        # the hash consistency check from the original import-identity check.
        read_imports(path)
        with self.assertRaisesRegex(ValueError, 'absent from hash bucket'):
            read_imports(path, verify_hash=True)

    def test_multiple_libraries_and_shared_override(self):
        modules, libraries, symbols = self.inspect(
            "libSceLibcInternal:exit:libTestTermination",
            "libSceLibcInternal:atexit:libTestTermination")
        self.assertEqual(symbols[nid("exit")], ("libSceLibcInternal", "libTestTermination"))
        self.assertEqual(symbols[nid("atexit")], symbols[nid("exit")])
        self.assertEqual(symbols[nid("__cxa_atexit")], ("libSceLibcInternal", "libSceLibcInternal"))
        self.assertEqual(list(libraries.values()).count("libTestTermination"), 1)
        self.assertEqual(list(modules.values()).count("libSceLibcInternal"), 1)

    def test_two_distinct_overrides_and_default_reuse(self):
        _, libraries, symbols = self.inspect("libSceLibcInternal:exit:libExitProbe",
                                             "libSceLibcInternal:atexit:libCallbackProbe")
        self.assertEqual(symbols[nid("exit")][1], "libExitProbe")
        self.assertEqual(symbols[nid("atexit")][1], "libCallbackProbe")
        self.assertNotEqual(symbols[nid("exit")], symbols[nid("atexit")])
        self.assertEqual(self.convert("libSceLibcInternal:exit:libSceLibcInternal"), self.base)

    def test_invalid_unknown_wrong_module_and_duplicate_bindings(self):
        for binding in ("exit", "libSceLibcInternal::library", "a:b:c:d", "a:b:bad/name",
                        "libSceLibcInternal:missing_symbol:libX", "libkernel:exit:libX",
                        "libMissing:exit:libX", "libSceLibcInternal:exit:" + "x" * 129):
            with self.subTest(binding=binding):
                self.convert(binding, success=False)
        for second in ("libX", "libY"):
            self.convert("libSceLibcInternal:exit:libX",
                         "libSceLibcInternal:exit:" + second, success=False)

    def test_real_shell_service_import_uses_distinct_library(self):
        # Link-only fixture: no runtime startup, signing, packaging or execution.
        source = self.work / "shell-import.cpp"
        source.write_text('extern "C" int sceShellCoreUtilExitApp();\n'
                          'extern "C" int sceSystemServiceHideSplashScreen();\n'
                          'volatile int status = 1;\n'
                          'extern "C" void _start() {\n'
                          ' status = sceSystemServiceHideSplashScreen();\n'
                          ' status = sceShellCoreUtilExitApp();\n}\n')
        obj, pie = self.work / "shell-import.o", self.work / "shell-import.elf"
        lock = json.loads((ROOT / "dependencies.json").read_text())
        subprocess.run([str(ROOT / "tools/ps5-clangxx"), "-O2", "-fPIC", "-fvisibility=hidden",
                        "-fno-stack-protector", "-c", str(source), "-o", str(obj)], check=True, capture_output=True)
        subprocess.run(["ld.lld-18", "-pie", "-T", str(ROOT / "ports/clr/ps5-pie.ld"), "-z", "max-page-size=0x4000",
                        "--version-script", str(BUILD / "symbols.map"), "-e", "_start", "-o", str(pie),
                        str(obj), str(ROOT / ".deps/sdk/target/lib/libSceSystemService.so")],
                       check=True, capture_output=True)
        data = self.convert("libSceSystemService:sceShellCoreUtilExitApp:libSceShellCoreUtil", pie=pie)
        path = self.work / "shell-native.elf"
        path.write_bytes(data)
        modules, libraries, symbols = read_imports(path, verify_hash=True)
        self.assertEqual(symbols[nid("sceShellCoreUtilExitApp")],
                         ("libSceSystemService", "libSceShellCoreUtil"))
        self.assertEqual(symbols[nid("sceSystemServiceHideSplashScreen")],
                         ("libSceSystemService", "libSceSystemService"))
        self.assertEqual(list(modules.values()), ["libSceSystemService"])
        self.assertCountEqual(libraries.values(), ["libSceSystemService", "libSceShellCoreUtil"])


if __name__ == "__main__":
    unittest.main()
