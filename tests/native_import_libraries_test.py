"""Offline integration test of the pinned converter patch against its unmodified source.

Run after build_native_canary.py for gfx1013/runtime. No console or firmware files.
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
BUILD = ROOT / "build/native/gfx1013-runtime"


def read_imports(path):
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
        native = ROOT / ".deps" / ("native-tools-" + lock["native_tools"]["revision"])
        original = cls.work / "original"
        shutil.copytree(native, original)
        subprocess.run(["patch", "--batch", "--fuzz=0", "-R", "-p1", "-d", str(original),
                        "-i", str(ROOT / "patches/native-import-libraries.patch")], check=True, capture_output=True)
        zlib = ROOT / ".deps" / ("host-zlib-" + lock["host_zlib"]["version"]) / "root"
        cls.original_tool = cls.work / "original-tool"
        subprocess.run(["clang++-18", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                        *[str(original / name) for name in ("native_app_builder.cpp", "elf_object.cpp",
                                                           "sce_module_writer.cpp", "self_container.cpp")],
                        "-I", str(zlib / "include"), str(zlib / "lib/libz.a"), "-o", str(cls.original_tool)],
                       check=True, capture_output=True)
        cls.base = cls.convert(tool=cls.original_tool)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def convert(cls, *bindings, tool=None, success=True):
        path = cls.work / "output.elf"
        path.unlink(missing_ok=True)
        command = [str(tool or BUILD / "ps5-native-tool"), "link", "--in", str(BUILD / "llvm-pie.elf"),
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
        return read_imports(path)

    def test_no_override_is_byte_identical_to_unpatched_converter(self):
        self.assertEqual(self.convert(), self.base)

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


if __name__ == "__main__":
    unittest.main()
