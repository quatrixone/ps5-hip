"""Boundary tests for the genuine HIP title recipe and frozen staging identity."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import struct
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_native_vector_title as recipe
import native_vector_title as profile
import stage_native_title as stager
import verify_native_case as verifier
arguments = argparse.ArgumentParser(add_help=False)
arguments.add_argument("--native-build", type=Path)
options, remaining = arguments.parse_known_args()
sys.argv = [sys.argv[0], *remaining]


def digest(data):
    return hashlib.sha256(data).hexdigest()


class VectorTitleTests(unittest.TestCase):
    def setUp(self):
        # These staging fixtures test the caller contract, not ELF qualification.
        self.validator = mock.patch.object(profile, "validate_build", side_effect=lambda r, s, *args: profile.validate_profile(r, s))
        self.validator.start()
        self.addCleanup(self.validator.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "build"
        self.candidate = self.build / "candidate"
        self.candidate.mkdir(parents=True)
        required = ("tools/build_native_vector_title.py", "tools/native_vector_title.py", "tools/hip_vector_title.cpp",
                    "tools/hip_vector_control.cpp", "tests/hip_vector_client.cpp", "tools/native_shell_exit_probe.cpp",
                    "tools/native_runtime.h", "evidence/offline-compatible-hip-headers.json", "assets/icon0.png")
        for name in required:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        artifacts = {}
        for name in ("eboot.bin", "lapy.elf", "sce_module/libc.prx"):
            path = self.candidate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("synthetic " + name).encode())
            artifacts[name] = digest(path.read_bytes())
        self.receipt = {"target": "gfx1013", "kernel": "hip_vector", "shutdown_mode": "shell-request-probe",
            "source_commit": "a" * 40, "source_tree_clean": True,
            "hardware_validated": False, "hip_vector_profile": copy.deepcopy(profile.PROFILE), "artifacts": artifacts,
            "source_sha256": {n: digest((self.root / n).read_bytes()) for n in required if n != "assets/icon0.png"},
            "vector_audits": {"raw_pie": {"bundle_sha256": profile.PROFILE["bundle_sha256"]},
                "converted_eboot": {"bundle_sha256": profile.PROFILE["bundle_sha256"]},
                "startup": {"install_order": ["shell", "vector_finalizer", "_init", "main"], "compiler_atexit_unregister": True},
                "fself_roundtrip": {"matches_except_unmapped_sie_note": True}},
                "hip_vector_provenance": {n: digest(b"synthetic provenance") for n in profile.PROVENANCE_FILES}}
        (self.build / "provenance").mkdir()
        for n in profile.PROVENANCE_FILES:
            (self.build / "provenance" / n).write_bytes(b"synthetic provenance")

    def test_invented_audits_do_not_authorize_arbitrary_signed_bytes(self):
        self.validator.stop()
        with self.assertRaisesRegex(ValueError, "provenance missing"):
            profile.validate_build(self.receipt, self.root)

    def test_signed_payload_is_bound_to_all_audited_elf_segments(self):
        elf = bytearray(256)
        elf[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<Q", elf, 32, 64)
        struct.pack_into("<HH", elf, 54, 56, 1)
        struct.pack_into("<IIQQQQQQ", elf, 64, 1, 5, 128, 0x4000, 0, 128, 128, 0x4000)
        elf[128:] = bytes(range(128))
        container = bytearray(512)
        container[:4] = bytes.fromhex("5414f5ee")
        struct.pack_into("<H", container, 24, 2)
        struct.pack_into("<QQQQ", container, 32, 0x110004, 256, 32, 32)
        struct.pack_into("<QQQQ", container, 64, 0x2804, 320, 128, 128)
        container[96:216] = elf[:120]
        container[320:448] = elf[128:]
        profile.validate_fself(container, elf)
        for offset in (96, 160, 320, 447):
            bad = bytearray(container)
            bad[offset] ^= 1
            with self.assertRaisesRegex(ValueError, "signed/ELF binding"):
                profile.validate_fself(bad, elf)
        for size in (0, 31, 100, 447):
            with self.assertRaises(ValueError):
                profile.validate_fself(container[:size], elf)

    def test_only_final_unmapped_sie_note_may_reconstruct_zero(self):
        raw = bytearray(320)
        raw[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<Q", raw, 32, 64)
        struct.pack_into("<HH", raw, 54, 56, 2)
        struct.pack_into("<IIQQQQQQ", raw, 64, 1, 5, 176, 0x4000, 0, 120, 120, 0x4000)
        struct.pack_into("<IIQQQQQQ", raw, 120, 4, 0, 296, 0, 0, 24, 0, 4)
        raw[296:] = bytes.fromhex("040000000800000003000000534945000102030405060708")
        extracted = raw[:296] + bytes(24)
        result = profile.audit_roundtrip(raw, extracted)
        self.assertFalse(result["matches_converted_elf"])
        self.assertEqual(result["omitted_note"]["offset"], 296)
        for offset in (0, 200, 295, 296, 319):
            changed = bytearray(extracted)
            changed[offset] ^= 1
            with self.assertRaises(ValueError):
                profile.audit_roundtrip(raw, changed)
        for offset in (120, 128, 144, 160, 308):
            changed = bytearray(raw)
            changed[offset] ^= 1
            with self.assertRaises(ValueError):
                profile.audit_roundtrip(changed, extracted)

    def stage(self):
        (self.build / "receipt.json").write_text(json.dumps(self.receipt))
        return stager.stage(self.root, self.build, self.root / "stage",
            {"title_id": "PPSA99014", "title_name": "ps5-hip Compute Lab"}, "a" * 40, "/chosen/results")

    def test_genuine_profile_embeds_bundle_without_external_canary(self):
        manifest = self.stage()
        self.assertEqual(manifest["hip_vector_profile"], profile.PROFILE)
        self.assertEqual(manifest["expected_log"], "/chosen/results/ps5-hip-gfx1013-hip_vector.log")
        self.assertNotIn("canary.hsaco", manifest["files"])
        self.assertFalse((self.root / "stage/app/canary.hsaco").exists())
        frozen = (self.root / "stage/manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.stage()
        self.assertEqual((self.root / "stage/manifest.json").read_bytes(), frozen)

    def test_profile_mutations_fail_before_any_stage(self):
        for field, bad in (("target", "gfx1030"), ("kernel_name", "vector_add"),
                           ("combined_object_sha256", "0" * 64), ("bundle_sha256", "0" * 64),
                           ("code_sha256", "0" * 64), ("elements", 64), ("dispatches", 1),
                           ("exact_values", 73), ("kernarg_bytes", 32), ("data_logical_bytes", 292),
                           ("interior_offset_bytes", 0), ("elements", 73.0)):
            with self.subTest(field=field, bad=bad):
                original = self.receipt["hip_vector_profile"][field]
                self.receipt["hip_vector_profile"][field] = bad
                with self.assertRaises(ValueError):
                    self.stage()
                self.assertFalse((self.root / "stage").exists())
                self.receipt["hip_vector_profile"][field] = original

    def test_wrong_receipt_or_bundle_audit_cannot_stage(self):
        for key, bad in (("target", "gfx1030"), ("shutdown_mode", "platform-exit"), ("hardware_validated", True)):
            old = self.receipt[key]
            self.receipt[key] = bad
            with self.assertRaises(ValueError):
                self.stage()
            self.receipt[key] = old
        self.receipt["vector_audits"]["converted_eboot"]["bundle_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "embedded bundle"):
            self.stage()

    def test_external_canary_and_missing_source_rejected(self):
        self.receipt["artifacts"]["canary.hsaco"] = digest(b"foreign")
        with self.assertRaisesRegex(ValueError, "invalid offline"):
            self.stage()
        del self.receipt["artifacts"]["canary.hsaco"]
        del self.receipt["source_sha256"]["tools/hip_vector_control.cpp"]
        with self.assertRaisesRegex(ValueError, "source provenance"):
            self.stage()

    def test_frozen_verifier_checks_matching_profile(self):
        manifest = self.stage()
        stage = self.root / "stage"
        image = stage / "PPSA99014.ffpkg"
        image.write_bytes(b"synthetic UFS image; no hardware")
        package = {"title_id": "PPSA99014", "manifest_sha256": digest((stage / "manifest.json").read_bytes()),
                   "image": {"file": image.name, **verifier.fingerprint(image)}}
        (stage / "package-receipt.json").write_text(json.dumps(package))
        controllers = self.root / "controllers"
        controllers.mkdir()
        for name in ("launch.elf", "close.elf"):
            (controllers / name).write_bytes(name.encode())
        control = {"title_id": "PPSA99014", "artifacts": {n: verifier.fingerprint(controllers / n) for n in ("launch.elf", "close.elf")}}
        (controllers / "receipt.json").write_text(json.dumps(control))
        packages = {"candidates": [{"manifest": manifest, "package": package}]}
        old = verifier.ROOT
        verifier.ROOT = self.root
        self.addCleanup(setattr, verifier, "ROOT", old)
        result = verifier.verify_case(stage, controllers, packages, {"controller_build": control}, "PPSA99014")
        self.assertFalse(result["hardware_gate_passed"])
        self.assertEqual(result["hip_vector_profile"]["exact_values"], 146)
        manifest["hip_vector_profile"]["dispatches"] = 1
        (stage / "manifest.json").write_text(json.dumps(manifest))
        package["manifest_sha256"] = digest((stage / "manifest.json").read_bytes())
        (stage / "package-receipt.json").write_text(json.dumps(package))
        with self.assertRaisesRegex(ValueError, "manifest/build profile"):
            verifier.verify_case(stage, controllers, packages, {"controller_build": control}, "PPSA99014")

    def test_startup_patching_is_narrow_and_fail_closed(self):
        source = 'extern "C" void ps5hip_install_shell_exit_probe();\n    ps5hip_install_shell_exit_probe();\n#endif\n    _init();'
        patched = recipe.patch_startup(source)
        self.assertIn('extern "C" void ps5hip_install_vector_title_finalizer();', patched)
        self.assertLess(patched.index("    ps5hip_install_shell_exit_probe();"), patched.index("    ps5hip_install_vector_title_finalizer();"))
        self.assertLess(patched.index("    ps5hip_install_vector_title_finalizer();"), patched.index("    _init();"))
        for invalid in (source.replace("_init();", "_init(0);"), source + source, patched):
            with self.assertRaises(RuntimeError):
                recipe.patch_startup(invalid)

    def test_linked_startup_order_and_actual_unregister_required(self):
        text = '''00000000 <_start>:
  call <ps5hip_install_shell_exit_probe>
  call <ps5hip_install_vector_title_finalizer>
  call <_init>
  call <main>
00000100 <__hip_module_ctor>:
  call <atexit>
00000200 <__hip_module_dtor>:
  call <__hipUnregisterFatBinary>
00000300 <ps5hip_install_vector_title_finalizer>:
  call <atexit>
'''
        self.assertTrue(recipe.audit_startup(text)["compiler_atexit_unregister"])
        for invalid in (text.replace("<__hipUnregisterFatBinary>", "<free>"),
                        text.replace("<ps5hip_install_vector_title_finalizer>\n  call <_init>", "<_init>\n  call <ps5hip_install_vector_title_finalizer>"),
                        text.replace("call <atexit>", "call <abort>"), text.replace("<main>", "<other>")):
            with self.assertRaises(RuntimeError):
                recipe.audit_startup(invalid)


@unittest.skipUnless(options.native_build, "requires a completed native title build")
class ActualVectorTitleTests(unittest.TestCase):
    def test_actual_linked_artifacts_and_stale_candidate_rejection(self):
        build = options.native_build.resolve()
        receipt = json.loads((build / "receipt.json").read_text())
        signed = (build / "candidate/eboot.bin").read_bytes()
        profile.validate_build(receipt, ROOT, build / "provenance", signed)
        for name, expected in receipt["raw_artifacts"].items():
            self.assertEqual(digest((build / name).read_bytes()), expected, name)
        changed = bytearray(signed)
        changed[-1] ^= 1
        replacement = copy.deepcopy(receipt)
        replacement["artifacts"]["eboot.bin"] = digest(changed)
        with self.assertRaisesRegex(ValueError, "digest relation"):
            profile.validate_build(replacement, ROOT, build / "provenance", changed)
        for name in ("signed_sha256", "raw_pie_sha256", "app_crt_sha256", "linker_sha256"):
            replacement = copy.deepcopy(receipt)
            del replacement["vector_audits"]["fself_roundtrip"][name]
            with self.assertRaisesRegex(ValueError, "digest relation"):
                profile.validate_build(replacement, ROOT, build / "provenance", signed)
        disassembly = (build / "startup-disassembly.stdout").read_text()
        good = recipe.audit_startup(disassembly, build / "llvm-pie.elf")
        self.assertEqual(good, receipt["vector_audits"]["startup"])
        for wrong in ("ps5hip_install_vector_title_finalizer", "__hipUnregisterFatBinary", "__hip_module_dtor"):
            with self.assertRaises(RuntimeError):
                recipe.audit_startup(disassembly.replace("<" + wrong + ">", "<foreign>"), build / "llvm-pie.elf")
        wrong_slot = disassembly.replace("# 0x" + format(good["atexit_got"][0], "x") + " ", "# 0xdeadbeef ")
        with self.assertRaises(RuntimeError):
            recipe.audit_startup(wrong_slot, build / "llvm-pie.elf")


if __name__ == "__main__":
    unittest.main()
