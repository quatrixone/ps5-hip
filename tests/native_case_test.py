import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("verify_native_case", pathlib.Path(__file__).resolve().parents[1] / "tools/verify_native_case.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.stage, self.controllers = self.root / "stage", self.root / "controllers"
        (self.stage / "app").mkdir(parents=True)
        self.controllers.mkdir()
        self.title = "PPSA99012"
        for path, data in ((self.stage / "app/eboot.bin", b"app"),
                           (self.stage / "build-receipt.json", b"{}"),
                           (self.stage / (self.title + ".ffpkg"), b"image"),
                           (self.controllers / "launch.elf", b"launch"),
                           (self.controllers / "close.elf", b"close")):
            path.write_bytes(data)
        manifest = {"title_id": self.title, "target": "gfx1013", "kernel": "runtime",
                    "source_commit": "fixture", "expected_log": "/download0/fixture.log",
                    "build_receipt_sha256": module.fingerprint(self.stage / "build-receipt.json")["sha256"],
                    "files": {"eboot.bin": module.fingerprint(self.stage / "app/eboot.bin")}}
        self.write(self.stage / "manifest.json", manifest)
        package = {"title_id": self.title,
                   "manifest_sha256": module.fingerprint(self.stage / "manifest.json")["sha256"],
                   "image": {"file": self.title + ".ffpkg", **module.fingerprint(self.stage / (self.title + ".ffpkg"))}}
        self.write(self.stage / "package-receipt.json", package)
        control = {"title_id": self.title, "artifacts": {
            name: module.fingerprint(self.controllers / name) for name in ("launch.elf", "close.elf")}}
        self.write(self.controllers / "receipt.json", control)
        self.packages = {"candidates": [{"manifest": manifest, "package": package}]}
        self.controls = {"controller_build": control}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def verify(self):
        return module.verify_case(self.stage, self.controllers, self.packages, self.controls, self.title)

    def test_valid_local_bytes_do_not_qualify_hardware(self):
        result = self.verify()
        self.assertTrue(result["local_artifacts_verified"])
        self.assertFalse(result["hardware_gate_passed"])

    def test_changed_artifacts_and_receipts_are_rejected(self):
        for path in (self.stage / "app/eboot.bin", self.stage / (self.title + ".ffpkg"),
                     self.stage / "build-receipt.json", self.controllers / "launch.elf",
                     self.controllers / "close.elf", self.stage / "manifest.json",
                     self.stage / "package-receipt.json", self.controllers / "receipt.json"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(original)

    def test_extra_staged_file_is_rejected(self):
        (self.stage / "app/extra").write_bytes(b"extra")
        with self.assertRaises(ValueError):
            self.verify()

    def test_rehashed_adjacent_receipt_cannot_override_frozen_evidence(self):
        (self.controllers / "launch.elf").write_bytes(b"replacement")
        receipt = copy.deepcopy(self.controls["controller_build"])
        receipt["artifacts"]["launch.elf"] = module.fingerprint(self.controllers / "launch.elf")
        self.write(self.controllers / "receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "frozen evidence"):
            self.verify()

    def test_wrong_title_is_rejected(self):
        self.title = "PPSA99999"
        with self.assertRaisesRegex(ValueError, "title mismatch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
