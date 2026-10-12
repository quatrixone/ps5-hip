import copy
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import private_buffer_fixture as fixture
import stage_native_title as stager
import verify_native_case as verifier
import check_canary_log as checker


class PrivateBufferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.output = self.root / "compiler"
        self.output.mkdir()
        self.profiles = []
        artifacts = {}
        for target in ("gfx1013", "gfx1030"):
            for opt in ("O1", "O2"):
                stem = target + "-" + opt
                data = (stem + " accepted ordinary shader").encode()
                ir = (stem + " original private5 IR").encode()
                (self.output / (stem + ".hsaco")).write_bytes(data)
                (self.output / (stem + ".ll")).write_bytes(ir)
                artifacts[stem + ".hsaco"] = fixture.digest(data)
                artifacts[stem + ".ll"] = fixture.digest(ir)
                self.profiles.append(dict(target=target, optimization=opt, input_sha256=fixture.digest(ir),
                                          code_sha256=fixture.digest(data), bytes=len(data), path=stem + ".hsaco"))
        (self.output / "audit.json").write_bytes(b"accepted graph audit")
        self.receipt = dict(schema=1, selected_recipe_reproduction_passed=True,
                            counts=copy.deepcopy(fixture.COUNTS), selected_profiles=self.profiles,
                            artifacts=artifacts, receipts={"audit.json": fixture.digest(b"accepted graph audit")})
        (self.root / "experimental").mkdir()
        (self.root / "tools").mkdir()
        (self.root / "tools/private_buffer_fixture.py").write_bytes((ROOT / "tools/private_buffer_fixture.py").read_bytes())
        self.freeze()

    def freeze(self):
        raw = json.dumps(self.receipt).encode()
        (self.output / "reproduction-receipt.json").write_bytes(raw)
        lock = dict(schema=1, profile="private_calls_buffer", reproduction_receipt_sha256=fixture.digest(raw),
                    selected_profiles=self.profiles)
        (self.root / fixture.LOCK).write_text(json.dumps(lock))

    def load(self):
        return fixture.verified_private_buffer_object(self.root, self.output, "gfx1013")

    def test_exact_copy_and_receipt_substitution(self):
        data, report = self.load()
        self.assertEqual(data, (self.output / "gfx1013-O2.hsaco").read_bytes())
        self.assertEqual(report, fixture.binding(self.root, "gfx1013"))
        raw = (self.output / "reproduction-receipt.json").read_bytes()
        (self.output / "reproduction-receipt.json").write_bytes(raw + b" ")
        with self.assertRaisesRegex(ValueError, "receipt digest"):
            self.load()

    def test_strict_counts_success_and_profile_set(self):
        original = copy.deepcopy(self.receipt)
        for key, bad in (("schema", True), ("selected_recipe_reproduction_passed", 1),
                         ("selected_recipe_reproduction_passed", False)):
            self.receipt = copy.deepcopy(original); self.receipt[key] = bad; self.freeze()
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError): self.load()
        for key in fixture.COUNTS:
            for bad in (0, float(fixture.COUNTS[key]), True):
                self.receipt = copy.deepcopy(original); self.receipt["counts"][key] = bad; self.freeze()
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError): self.load()
        self.receipt = copy.deepcopy(original)
        self.receipt["selected_profiles"] = self.profiles[:-1] + [self.profiles[0]]
        self.freeze()
        with self.assertRaises(ValueError): self.load()

    def test_every_artifact_and_input_are_bound(self):
        for name in list(self.receipt["artifacts"]) + list(self.receipt["receipts"]):
            path = self.output / name; raw = path.read_bytes(); path.write_bytes(raw + b"altered")
            with self.subTest(name=name), self.assertRaises(ValueError): self.load()
            path.write_bytes(raw)
        self.receipt["artifacts"].pop("gfx1013-O1.ll"); self.freeze()
        with self.assertRaisesRegex(ValueError, "input mismatch"): self.load()

    def test_path_and_symlink_escape(self):
        for name in ("../outside", "/absolute", "a/../outside", "a//b", "./x", "C:/x", "a\\b"):
            self.receipt["receipts"] = {name: "0" * 64}; self.freeze()
            with self.subTest(name=name), self.assertRaises(ValueError): self.load()
        outside = self.root / "outside"; outside.write_bytes(b"outside")
        (self.output / "escape").symlink_to(outside)
        self.receipt["receipts"] = {"escape": fixture.digest(b"outside")}; self.freeze()
        with self.assertRaisesRegex(ValueError, "outside"): self.load()

    def test_wrong_profiles_and_bytes(self):
        original = copy.deepcopy(self.receipt)
        for field, value in (("optimization", "O0"), ("target", "gfx900"), ("bytes", True),
                             ("bytes", 999), ("input_sha256", "f" * 64), ("code_sha256", "f" * 64),
                             ("path", "generic-volatile-proof-off.hsaco")):
            self.receipt = copy.deepcopy(original)
            self.profiles = copy.deepcopy(original["selected_profiles"])
            self.profiles[1][field] = value; self.receipt["selected_profiles"] = self.profiles; self.freeze()
            with self.subTest(field=field), self.assertRaises((ValueError, FileNotFoundError)): self.load()

    def build_fixture(self):
        data, report = self.load()
        build = self.root / "build"; (build / "candidate/sce_module").mkdir(parents=True)
        (self.root / "assets").mkdir(); (self.root / "assets/icon0.png").write_bytes((ROOT / "assets/icon0.png").read_bytes())
        files = {"eboot.bin": b"native", "lapy.elf": b"helper", "sce_module/libc.prx": b"companion", "canary.hsaco": data}
        for name, content in files.items(): (build / "candidate" / name).write_bytes(content)
        receipt = dict(target="gfx1013", kernel="private_calls_buffer", hardware_validated=False,
                       device_build=report, private_buffer=copy.deepcopy(fixture.CONTRACT),
                       artifacts={k: fixture.digest(v) for k, v in files.items()},
                       source_sha256={k: fixture.digest((self.root / k).read_bytes()) for k in
                                      (fixture.LOCK, "tools/private_buffer_fixture.py")})
        return build, receipt

    def test_stage_binding_and_immutable_output(self):
        build, receipt = self.build_fixture()
        stage = self.root / "stage"
        identity = {"title_id": "PPSA99014", "title_name": "test"}
        def run():
            (build / "receipt.json").write_text(json.dumps(receipt))
            return stager.stage(self.root, build, stage, identity, "a" * 40, "/results")
        original = copy.deepcopy(receipt)
        for field in ("private_buffer", "device_build", "source_sha256"):
            receipt[field] = {}
            with self.subTest(field=field), self.assertRaises(ValueError): run()
            self.assertFalse(stage.exists()); receipt = copy.deepcopy(original)
        for key in fixture.CONTRACT:
            receipt["private_buffer"][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError): run()
            receipt = copy.deepcopy(original)
        manifest = run()
        self.assertEqual(manifest["private_buffer"], fixture.CONTRACT)
        self.assertEqual(manifest["private_buffer_compiler"], receipt["device_build"])
        with self.assertRaises(FileExistsError): run()
        # Feed this actual staged tree to the existing frozen-package verifier.
        controls = self.root / "controllers"; controls.mkdir()
        for name in ("launch.elf", "close.elf"): (controls / name).write_bytes(name.encode())
        control = {"title_id": "PPSA99014", "artifacts": {n: verifier.fingerprint(controls / n) for n in ("launch.elf", "close.elf")}}
        (controls / "receipt.json").write_text(json.dumps(control))
        image = stage / "PPSA99014.ffpkg"; image.write_bytes(b"synthetic package, no executable")
        package = {"title_id": "PPSA99014", "manifest_sha256": verifier.fingerprint(stage / "manifest.json")["sha256"],
                   "image": {"file": image.name, **verifier.fingerprint(image)}}
        (stage / "package-receipt.json").write_text(json.dumps(package))
        packages = {"candidates": [{"manifest": manifest, "package": package}]}
        def verify():
            with mock.patch.object(verifier, "ROOT", self.root):
                return verifier.verify_case(stage, controls, packages, {"controller_build": control}, "PPSA99014")
        self.assertEqual(verify()["private_buffer"], fixture.CONTRACT)
        for field in ("private_buffer", "private_buffer_compiler"):
            saved = manifest[field]; manifest[field] = {}
            (stage / "manifest.json").write_text(json.dumps(manifest))
            package["manifest_sha256"] = verifier.fingerprint(stage / "manifest.json")["sha256"]
            (stage / "package-receipt.json").write_text(json.dumps(package))
            with self.subTest(field=field), self.assertRaises(ValueError): verify()
            manifest[field] = saved

    def test_cli_binding_required_only_for_new_profile(self):
        base = [sys.executable, str(ROOT / "tools/build_native_canary.py"), "--native-source", ".", "--target", "gfx1013"]
        for extra in (["--kernel", "private_calls_buffer"], ["--kernel", "runtime", "--private-buffer-compiler", "."]):
            result = subprocess.run(base + extra, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("required only for private_calls_buffer", result.stderr)

    def test_exact_log_contract(self):
        for target in ("gfx1013", "gfx1030"):
            lines = [f"PS5HIP START target={target} kernel=private_calls_buffer",
                     "PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass",
                     f"PS5HIP SUBMIT target={target} kernel=private_calls_buffer invocations=16448 groups=257 user_sgprs=8 private_bytes=304 scratch_bytes=13107200 timeout_ms=2000",
                     "PS5HIP SCRATCH payload=modified guards=unchanged",
                     "PS5HIP PASS kernel=private_calls_buffer exact_values=16448 guards=unchanged allocations=closed"]
            data = ("\n".join(lines) + "\n").encode()
            def result(d): return checker.check_log(d, target, "private_calls_buffer")["application_result"]
            self.assertEqual(result(data), "application-pass")
            for old, new in ((b"304", b"288"), (b"13107200", b"11796480"), (b"private_calls_buffer", b"nested_private"),
                             (b"16448", b"16447"), (b"modified", b"untouched")):
                self.assertEqual(result(data.replace(old, new)), "incomplete-or-unexpected")
            for i in range(len(lines)):
                self.assertEqual(result(("\n".join(lines[:i] + lines[i+1:]) + "\n").encode()), "incomplete-or-unexpected")
            self.assertEqual(result(data + b"PS5HIP FAIL: mismatch\n"), "failed")
            self.assertEqual(result(data + b"PS5HIP QUARANTINED: GPU ownership unresolved\n"), "quarantined")
            self.assertEqual(result(data + data), "incomplete-or-unexpected")


if __name__ == "__main__":
    unittest.main()
