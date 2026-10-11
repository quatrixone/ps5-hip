import importlib.util
import json
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("stage_native_title", pathlib.Path(__file__).resolve().parents[1] / "tools/stage_native_title.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pack_spec = importlib.util.spec_from_file_location("package_native_title", pathlib.Path(__file__).resolve().parents[1] / "tools/package_native_title.py")
pack = importlib.util.module_from_spec(pack_spec)
pack_spec.loader.exec_module(pack)


class StageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.build, self.output = self.root / "build", self.root / "out"
        (self.root / "assets").mkdir()
        (self.root / "assets/icon0.png").write_bytes((module.ROOT / "assets/icon0.png").read_bytes())
        (self.build / "candidate").mkdir(parents=True)
        (self.root / "source.cpp").write_bytes(b"fixture source")
        (self.build / "candidate/eboot.bin").write_bytes(b"fixture container")
        (self.build / "candidate/lapy.elf").write_bytes(b"fixture helper")
        (self.build / "candidate/sce_module").mkdir()
        (self.build / "candidate/sce_module/libc.prx").write_bytes(b"fixture companion")
        self.receipt = {"target": "gfx1013", "kernel": "runtime", "hardware_validated": False,
                        "source_sha256": {"source.cpp": module.digest(b"fixture source")},
                        "artifacts": {"eboot.bin": module.digest(b"fixture container"),
                                      "lapy.elf": module.digest(b"fixture helper"),
                                      "sce_module/libc.prx": module.digest(b"fixture companion")}}
        self.identity = {"title_id": "PPSA99012", "title_name": "ps5-hip Compute Lab"}

    def run_stage(self):
        (self.build / "receipt.json").write_text(json.dumps(self.receipt))
        return module.stage(self.root, self.build, self.output, self.identity, "a" * 40, "/chosen/results")

    def test_runtime_and_immutable_stage(self):
        manifest = self.run_stage()
        metadata = json.loads((self.output / "app/sce_sys/param.json").read_text())
        self.assertEqual(metadata["titleId"], "PPSA99012")
        self.assertEqual(metadata["conceptId"], "99012")
        self.assertRegex(metadata["contentId"], r"^UP9000-PPSA99012_00-[A-Z0-9]{16}$")
        self.assertEqual(manifest["expected_log"], "/chosen/results/ps5-hip-gfx1013-runtime.log")
        self.assertEqual((self.output / "app/results-dir.txt").read_bytes(), b"/chosen/results")
        self.assertEqual(metadata["downloadDataSize"], 0)
        self.assertEqual((self.output / "app/sce_sys/icon0.png").read_bytes(),
                         (self.root / "assets/icon0.png").read_bytes())
        self.assertFalse((self.output / "app/canary.hsaco").exists())
        for name, entry in manifest["files"].items():
            data = (self.output / "app" / name).read_bytes()
            self.assertEqual(entry, {"sha256": module.digest(data), "bytes": len(data)})
        frozen = (self.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_stage()
        self.assertEqual((self.output / "manifest.json").read_bytes(), frozen)

    def test_gpu_fixture(self):
        self.receipt["kernel"] = "indexed_store"
        self.receipt["artifacts"]["canary.hsaco"] = module.digest(b"fixture shader")
        (self.build / "candidate/canary.hsaco").write_bytes(b"fixture shader")
        self.run_stage()
        self.assertEqual((self.output / "app/canary.hsaco").read_bytes(), b"fixture shader")

    def test_wave_and_spill_require_verified_shader(self):
        for kernel in ("wave_full", "wave_partial", "register_spill"):
            self.receipt["kernel"] = kernel
            self.output = self.root / kernel
            self.receipt["artifacts"].pop("canary.hsaco", None)
            with self.assertRaisesRegex(ValueError, "invalid offline candidate receipt"):
                self.run_stage()
            self.assertFalse(self.output.exists())
            self.receipt["artifacts"]["canary.hsaco"] = module.digest(b"wave shader")
            shader = self.build / "candidate/canary.hsaco"
            shader.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "candidate digest"):
                self.run_stage()
            self.assertFalse(self.output.exists())
            shader.write_bytes(b"wave shader")
            manifest = self.run_stage()
            self.assertEqual(manifest["expected_log"], f"/chosen/results/ps5-hip-gfx1013-{kernel}.log")
            self.assertEqual((self.output / "app/canary.hsaco").read_bytes(), b"wave shader")

    def test_two_dispatch_profile_requires_exact_count(self):
        self.receipt["kernel"] = "vector_add_twice"
        self.receipt["artifacts"]["canary.hsaco"] = module.digest(b"fixture shader")
        (self.build / "candidate/canary.hsaco").write_bytes(b"fixture shader")
        for invalid in (None, 0, 1, 3, True, 2.0, "2"):
            self.receipt["dispatches"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "two dispatches"):
                self.run_stage()
            self.assertFalse(self.output.exists())
        self.receipt["dispatches"] = 2
        manifest = self.run_stage()
        self.assertEqual(manifest["dispatches"], 2)
        self.assertEqual(manifest["expected_log"], "/chosen/results/ps5-hip-gfx1013-vector_add_twice.log")

    def test_package_inventory(self):
        manifest = self.run_stage()
        app = self.output / "app"
        pack.verify_tree(app, manifest["files"])
        (app / "unexpected").write_bytes(b"extra")
        with self.assertRaises(ValueError):
            pack.verify_tree(app, manifest["files"])
        (app / "unexpected").unlink()
        (app / "eboot.bin").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            pack.verify_tree(app, manifest["files"])

    def test_tampered_binary(self):
        (self.build / "candidate/eboot.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "candidate digest"):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_changed_source(self):
        (self.root / "source.cpp").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_invalid_identity(self):
        self.identity["title_id"] = "../PPSA99012"
        with self.assertRaisesRegex(ValueError, "identity"):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_missing_or_invalid_icon_prevents_staging(self):
        icon = self.root / "assets/icon0.png"
        icon.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_stage()
        self.assertFalse(self.output.exists())
        icon.write_bytes(b"not png")
        with self.assertRaisesRegex(ValueError, "512x512 PNG"):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_missing_loader_companion_prevents_staging(self):
        del self.receipt["artifacts"]["sce_module/libc.prx"]
        with self.assertRaisesRegex(ValueError, "invalid offline candidate receipt"):
            self.run_stage()
        self.assertFalse(self.output.exists())

    def test_results_directory_contract(self):
        for path in ("/", "relative", "/a/", "/a//b", "/a/../b", "/a/./b", "/a\n", "/a\\b", "/" + "x" * 240):
            with self.subTest(path=path), self.assertRaises(ValueError):
                module.validate_results_directory(path)
        self.assertEqual(module.validate_results_directory("/chosen/results"), "/chosen/results")

    def test_invalid_receipt(self):
        for field, value in (("target", "gfx9999"), ("kernel", "other"),
                             ("hardware_validated", True), ("artifacts", {}),
                             ("source_sha256", {"../escape": "0" * 64})):
            with self.subTest(field=field):
                old = self.receipt[field]
                self.receipt[field] = value
                with self.assertRaises(ValueError):
                    self.run_stage()
                self.receipt[field] = old
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
