import importlib.util
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import unittest

spec = importlib.util.spec_from_file_location("check_canary_log", pathlib.Path(__file__).resolve().parents[1] / "tools/check_canary_log.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LogTests(unittest.TestCase):
    def test_routes_title_requires_all_routes_and_finalization(self):
        lines = ["PS5HIP START target=gfx1013 kernel=hip_routes",
                 "HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=4 constructor_gpu_calls=0",
                 "PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass"]
        names = ("integer", "lds", "spill", "geometry")
        for sequence in range(8):
            lines += [f"RUNTIME_COMPLETE sequence={sequence+1} kernel=routes_{names[sequence%4]} blocking=true",
                      f"ROUTES_COMPLETE round={sequence//4+1} route={sequence%4} blocking=true full_guards=true inputs=unchanged"]
        lines += ["ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true",
                  "HIP_TITLE COMPUTE complete=true dispatches=8 word_comparisons=393216 finalization=pending",
                  "PS5HIP SHUTDOWN main-return=0", "PS5HIP SHUTDOWN atexit=observed",
                  "PS5HIP SHUTDOWN static-dtor=observed", "VECTOR_UNREGISTER after_main=true active=0",
                  "VECTOR_FINAL registered=1 removed=1 completed=8 active=0 terminal=0 main_finished=1 hardware_validated=false",
                  "HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed dispatches=8",
                  "PS5HIP SHELL_EXIT application-fini=completed", "PS5HIP SHELL_EXIT request=begin",
                  "PS5HIP SHELL_EXIT request-return=0"]
        data = ("\n".join(lines) + "\n").encode()
        def check(raw):
            return module.check_log(raw, "gfx1013", "hip_routes", shell_exit_probe=True)
        self.assertEqual(check(data)["application_result"], "application-pass")
        self.assertFalse(check(data)["hardware_gate_passed"])
        for index in range(len(lines)):
            missing = ("\n".join(lines[:index] + lines[index+1:]) + "\n").encode()
            self.assertNotEqual(check(missing)["application_result"], "application-pass")
        for before, after in ((b"routes_spill", b"routes_lds"), (b"round=2", b"round=1"),
                              (b"completed=8", b"completed=7"), (b"functions=4", b"functions=1"),
                              (b"393216", b"393215"), (b"removed=1", b"removed=0")):
            self.assertNotEqual(check(data.replace(before, after))["application_result"], "application-pass")
        for bad in (data*2, data[:-1], data.replace(b"\n", b"\r\n")):
            self.assertNotEqual(check(bad)["application_result"], "application-pass")
        self.assertEqual(check(data+b"ROUTES_CLIENT_REJECT failure\n")["application_result"], "failed")
        self.assertEqual(check(data+b"HIP_TITLE QUARANTINED retained\n")["application_result"], "quarantined")
        with self.assertRaises(ValueError):
            module.check_log(data,"gfx1030","hip_routes",shell_exit_probe=True)

    hip_title = (
        b"PS5HIP START target=gfx1013 kernel=hip_vector\n"
        b"HIP_TITLE REGISTRATION observed_after_redirect=true modules=1 functions=1 constructor_gpu_calls=0\n"
        b"PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n"
        b"VECTOR_COMPLETE sequence=1 n=73 blocking=true\n"
        b"VECTOR_COMPLETE sequence=2 n=73 blocking=true\n"
        b"VECTOR_CLIENT_PASS exact_values=146 immutable_inputs=true full_guards=true blocking=true hardware_validated=false\n"
        b"HIP_TITLE COMPUTE complete=true dispatches=2 exact_values=146 finalization=pending\n"
        b"PS5HIP SHUTDOWN main-return=0\n"
        b"PS5HIP SHUTDOWN atexit=observed\n"
        b"PS5HIP SHUTDOWN static-dtor=observed\n"
        b"VECTOR_UNREGISTER after_main=true active=0\n"
        b"VECTOR_FINAL registered=1 removed=1 completed=2 active=0 terminal=0 main_finished=1 hardware_validated=false\n"
        b"HIP_TITLE FINAL compiler_unregister=observed state_destroyed=true allocations=closed exact_values=146\n"
        b"PS5HIP SHELL_EXIT application-fini=completed\n"
        b"PS5HIP SHELL_EXIT request=begin\n"
        b"PS5HIP SHELL_EXIT request-return=0\n")

    def test_hip_title_requires_complete_finalization(self):
        def check(data):
            return module.check_log(data, "gfx1013", "hip_vector", shell_exit_probe=True)
        result = check(self.hip_title)
        self.assertEqual(result["application_result"], "application-pass")
        self.assertFalse(result["hardware_gate_passed"])
        lines = self.hip_title.splitlines(keepends=True)
        for index in range(len(lines)):
            with self.subTest(missing=index):
                self.assertNotEqual(check(b"".join(lines[:index] + lines[index+1:]))["application_result"], "application-pass")
        # Computational success cannot substitute for genuine compiler teardown.
        for data in (b"".join(lines[:7]), self.hip_title * 2, self.hip_title[:-1],
                     self.hip_title.replace(b"\n", b"\r\n"),
                     b"".join(lines[:10] + [lines[12], lines[11], lines[10]] + lines[13:])):
            self.assertNotEqual(check(data)["application_result"], "application-pass")
        for before, after in ((b"removed=1", b"removed=0"), (b"active=0", b"active=1"),
                              (b"terminal=0", b"terminal=1"), (b"state_destroyed=true", b"state_destroyed=false"),
                              (b"allocations=closed", b"allocations=retained"), (b"sequence=2", b"sequence=1"),
                              (b"values=146", b"values=73"), (b"request-return=0", b"request-return=-1")):
            self.assertNotEqual(check(self.hip_title.replace(before, after))["application_result"], "application-pass")
        for suffix in (b"VECTOR_ERROR cleanup\n", b"VECTOR_CLIENT_REJECT bad output\n", b"HIP_TITLE FAIL finalization\n"):
            self.assertEqual(check(self.hip_title + suffix)["application_result"], "failed")
        self.assertEqual(check(self.hip_title + b"HIP_TITLE QUARANTINED retained\n")["application_result"], "quarantined")
        with self.assertRaises(ValueError):
            module.check_log(self.hip_title, "gfx1030", "hip_vector", shell_exit_probe=True)
        with self.assertRaises(ValueError):
            module.check_log(self.hip_title, "gfx1013", "hip_vector")

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

    def test_input_bounds(self):
        with self.assertRaises(ValueError):
            module.check_log(b"x" * 65537, "gfx1013", "runtime")
        with self.assertRaises(UnicodeDecodeError):
            module.check_log(b"\xff", "gfx1013", "runtime")

    def test_two_dispatch_vector_contract(self):
        start = (b"PS5HIP START target=gfx1013 kernel=vector_add_twice\n"
                 b"PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n")
        def dispatch(index):
            return (f"PS5HIP DISPATCH index={index}/2 kernel=vector_add\n"
                    "PS5HIP SUBMIT target=gfx1013 kernel=vector_add count=73 groups=2 timeout_ms=2000\n"
                    "PS5HIP PASS kernel=vector_add exact_values=73 guards=unchanged allocations=closed\n").encode()
        summary = b"PS5HIP PASS kernel=vector_add_twice dispatches=2 exact_values=146 guards=unchanged allocations=closed\n"
        valid = start + dispatch(1) + dispatch(2) + summary
        self.assertEqual(module.check_log(valid, "gfx1013", "vector_add_twice")["application_result"], "application-pass")
        for bad in (start + dispatch(1) + summary, start + dispatch(2) + dispatch(1) + summary,
                    start + dispatch(1) * 2 + summary, valid + dispatch(3), valid[:-1],
                    valid.replace(b"exact_values=146", b"exact_values=73"),
                    valid.replace(b"index=2/2", b"index=3/2"),
                    valid.replace(b"allocations=closed", b"allocations=retained"),
                    valid + b"PS5HIP QUARANTINED: unresolved\n"):
            self.assertNotEqual(module.check_log(bad, "gfx1013", "vector_add_twice")["application_result"], "application-pass")
        self.assertNotEqual(module.check_log(valid, "gfx1013", "vector_add")["application_result"], "application-pass")
        self.assertNotEqual(module.check_log(valid, "gfx1013", "vector_add_twice", shell_exit_probe=True)
                            ["application_result"], "application-pass")
        suffix = (b"PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\n"
                  b"PS5HIP SHUTDOWN static-dtor=observed\nPS5HIP SHELL_EXIT application-fini=completed\n"
                  b"PS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0\n")
        self.assertEqual(module.check_log(valid + suffix, "gfx1013", "vector_add_twice", shell_exit_probe=True)
                         ["application_result"], "application-pass")

    def test_wave_contract(self):
        for target in ("gfx1013", "gfx1030"):
            for kernel in ("wave_full", "wave_partial"):
                data = (f"PS5HIP START target={target} kernel={kernel}\n"
                        "PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n"
                        f"PS5HIP SUBMIT target={target} kernel={kernel} invocations=64 output_words=256 groups=1 timeout_ms=2000\n"
                        f"PS5HIP PASS kernel={kernel} exact_values=256 guards=unchanged allocations=closed\n").encode()
                self.assertEqual(module.check_log(data, target, kernel)["application_result"], "application-pass")
                for bad in (data.replace(b"invocations=64", b"invocations=32"),
                            data.replace(b"output_words=256", b"output_words=128"),
                            data.replace(b"groups=1", b"groups=2"),
                            data.replace(b"exact_values=256", b"exact_values=128"),
                            data.replace(b"guards=unchanged", b"guards=changed"), data[:-1],
                            data.replace(kernel.encode(), b"indexed_store"),
                            data + b"PS5HIP QUARANTINED: unresolved\n"):
                    self.assertNotEqual(module.check_log(bad, target, kernel)["application_result"], "application-pass")

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
