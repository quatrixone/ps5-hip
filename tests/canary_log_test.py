import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("check_canary_log", pathlib.Path(__file__).resolve().parents[1] / "tools/check_canary_log.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LogTests(unittest.TestCase):
    runtime = (b"PS5HIP START target=gfx1013 kernel=runtime\n"
               b"PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n"
               b"PS5HIP PASS kernel=runtime gpu_api_calls=0\n")

    def test_exact_application_pass_is_not_hardware_pass(self):
        result = module.check_log(self.runtime, "gfx1013", "runtime")
        self.assertEqual(result["application_result"], "application-pass")
        self.assertFalse(result["hardware_gate_passed"])
        self.assertIn("log freshness", result["unproven"])

    def test_reject_stale_duplicate_truncated_or_mismatched_records(self):
        for data in (b"", self.runtime[:-1], self.runtime * 2, self.runtime.replace(b"\n", b"\r\n"),
                     self.runtime.replace(b"gfx1013", b"gfx1030"),
                     self.runtime + b"unexpected cleanup failure\n"):
            self.assertEqual(module.check_log(data, "gfx1013", "runtime")["application_result"],
                             "incomplete-or-unexpected")

    def test_failure_overrides_pass(self):
        for suffix, result in ((b"PS5HIP FAIL: cleanup\n", "failed"),
                               (b"PS5HIP QUARANTINED: unresolved\n", "quarantined")):
            self.assertEqual(module.check_log(self.runtime + suffix, "gfx1013", "runtime")["application_result"], result)

    def test_gpu_contract(self):
        for target in ("gfx1013", "gfx1030"):
            for kernel in ("indexed_store", "vector_add"):
                data = (f"PS5HIP START target={target} kernel={kernel}\n"
                        "PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n"
                        f"PS5HIP SUBMIT target={target} kernel={kernel} count=73 groups=2 timeout_ms=2000\n"
                        f"PS5HIP PASS kernel={kernel} exact_values=73 guards=unchanged allocations=closed\n").encode()
                self.assertEqual(module.check_log(data, target, kernel)["application_result"], "application-pass")
                self.assertNotEqual(module.check_log(data.replace(b"values=73", b"values=72"), target, kernel)["application_result"], "application-pass")

    def test_input_bounds(self):
        with self.assertRaises(ValueError):
            module.check_log(b"x" * 65537, "gfx1013", "runtime")
        with self.assertRaises(UnicodeDecodeError):
            module.check_log(b"\xff", "gfx1013", "runtime")


if __name__ == "__main__":
    unittest.main()
