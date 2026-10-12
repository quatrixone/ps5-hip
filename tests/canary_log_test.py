import importlib.util
import pathlib
import unittest
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

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

    def test_submission_exception_remains_quarantined(self):
        for reason in (b"monotonic clock failed", b"unknown exception"):
            data = (b"PS5HIP FAIL: submission exception: " + reason + b"\n"
                    b"PS5HIP QUARANTINED: GPU ownership unresolved; no retry or local teardown\n")
            self.assertEqual(module.check_log(data, "gfx1013", "indexed_store")["application_result"], "quarantined")

    def test_gpu_contract(self):
        for target in ("gfx1013", "gfx1030"):
            for kernel in ("indexed_store", "vector_add"):
                data = (f"PS5HIP START target={target} kernel={kernel}\n"
                        "PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n"
                        f"PS5HIP SUBMIT target={target} kernel={kernel} count=73 groups=2 timeout_ms=2000\n"
                        f"PS5HIP PASS kernel={kernel} exact_values=73 guards=unchanged allocations=closed\n").encode()
                self.assertEqual(module.check_log(data, target, kernel)["application_result"], "application-pass")
                self.assertNotEqual(module.check_log(data.replace(b"values=73", b"values=72"), target, kernel)["application_result"], "application-pass")
                suffix = (b"PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\n"
                          b"PS5HIP SHUTDOWN static-dtor=observed\n"
                          b"PS5HIP SHELL_EXIT application-fini=completed\n"
                          b"PS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0\n")
                complete = data + suffix
                result = module.check_log(complete, target, kernel, shell_exit_probe=True)
                self.assertEqual(result["application_result"], "application-pass")
                self.assertFalse(result["hardware_gate_passed"])
                for bad in (data, suffix + data, complete[:-1],
                            complete.replace(b"allocations=closed", b"allocations=retained"),
                            complete.replace(b"request-return=0", b"request-return=-1"),
                            complete + b"PS5HIP FAIL: cleanup\n",
                            complete + b"PS5HIP QUARANTINED: unresolved\n"):
                    self.assertNotEqual(module.check_log(bad, target, kernel, shell_exit_probe=True)
                                        ["application_result"], "application-pass")

    def test_fixed_two_dispatch_order_and_summary(self):
        suffix = ("PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\n"
                  "PS5HIP SHUTDOWN static-dtor=observed\nPS5HIP SHELL_EXIT application-fini=completed\n"
                  "PS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0\n")
        for target in ("gfx1013", "gfx1030"):
            for profile, first in (("arithmetic_suite", "indexed_store"), ("vector_add_twice", "vector_add")):
                start = f"PS5HIP START target={target} kernel={profile}\n" + self.runtime.decode().splitlines()[1] + "\n"
                dispatches = [f"PS5HIP DISPATCH index={i}/2 kernel={kernel}\n"
                              f"PS5HIP SUBMIT target={target} kernel={kernel} count=73 groups=2 timeout_ms=2000\n"
                              f"PS5HIP PASS kernel={kernel} exact_values=73 guards=unchanged allocations=closed\n"
                              for i, kernel in enumerate((first, "vector_add"), 1)]
                summary = f"PS5HIP PASS kernel={profile} dispatches=2 exact_values=146 guards=unchanged allocations=closed\n"
                good = start + "".join(dispatches) + summary
                for shell in (False, True):
                    complete = good + (suffix if shell else "")
                    result = module.check_log(complete.encode(), target, profile, shell_exit_probe=shell)
                    self.assertEqual(result["application_result"], "application-pass")
                    self.assertFalse(result["hardware_gate_passed"])
                    for bad in (complete.replace("exact_values=146", "exact_values=219"),
                                complete.replace("index=2/2", "index=1/2"),
                                complete.replace(dispatches[0], ""), complete.replace(dispatches[1], ""),
                                complete.replace("guards=unchanged", "guards=modified"),
                                complete.replace("allocations=closed", "allocations=retained"),
                                complete.replace("".join(dispatches), "".join(reversed(dispatches))),
                                complete + "PS5HIP FAIL: cleanup\n", complete + "PS5HIP QUARANTINED: ownership\n",
                                complete.replace("request-return=0", "request-return=-1") if shell else complete[:-1]):
                        self.assertNotEqual(module.check_log(bad.encode(), target, profile, shell_exit_probe=shell)
                                            ["application_result"], "application-pass")

    def test_input_bounds(self):
        with self.assertRaises(ValueError):
            module.check_log(b"x" * 65537, "gfx1013", "runtime")
        with self.assertRaises(UnicodeDecodeError):
            module.check_log(b"\xff", "gfx1013", "runtime")

    def test_shell_exit_requires_returned_success_and_does_not_prove_exit(self):
        suffix = (b"PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\n"
                  b"PS5HIP SHUTDOWN static-dtor=observed\n"
                  b"PS5HIP SHELL_EXIT application-fini=completed\n"
                  b"PS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0\n")
        data = self.runtime + suffix
        result = module.check_log(data, "gfx1013", "runtime", shell_exit_probe=True)
        self.assertEqual(result["application_result"], "application-pass")
        self.assertFalse(result["hardware_gate_passed"])
        for bad in (data.rsplit(b"PS5HIP SHELL_EXIT request-return=0", 1)[0],
                    data.replace(b"request-return=0", b"request-return=-1"),
                    data.replace(b"PS5HIP SHELL_EXIT application-fini=completed\n", b""),
                    data + b"PS5HIP FAIL: shell-exit timeout\n"):
            self.assertNotEqual(module.check_log(bad, "gfx1013", "runtime", shell_exit_probe=True)
                                ["application_result"], "application-pass")

    def test_shutdown_probe_requires_all_checkpoints_but_never_certifies_exit(self):
        checkpoints = [b"PS5HIP SHUTDOWN main-return=0\n", b"PS5HIP SHUTDOWN atexit=observed\n",
                       b"PS5HIP SHUTDOWN static-dtor=observed\n"]
        data = self.runtime + b"".join(checkpoints)
        result = module.check_log(data, "gfx1013", "runtime", True)
        self.assertEqual(result["application_result"], "application-pass")
        self.assertFalse(result["hardware_gate_passed"])
        for partial in (self.runtime, data[:-1], data + checkpoints[-1],
                        self.runtime + b"".join(reversed(checkpoints))):
            self.assertNotEqual(module.check_log(partial, "gfx1013", "runtime", True)["application_result"],
                                "application-pass")
        with self.assertRaises(ValueError):
            module.check_log(data, "gfx1013", "vector_add", True)


if __name__ == "__main__":
    unittest.main()
