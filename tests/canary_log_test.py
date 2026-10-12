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

    def test_private_aperture_observations_and_corruption(self):
        for target in ("gfx1013", "gfx1030"):
            for across in ("equal", "different"):
                # Zero and descending/wrapped values remain raw observations.
                records = [f"PS5HIP APERTURE lane={i} base=0x{(0 if across == 'equal' or i < 32 else 0xffffffffffffffff):016x} "
                           f"limit=0x0000000000000000 sentinel=0x{i ^ 0x6b18d3a5:08x}" for i in range(64)]
                lines = [f"PS5HIP START target={target} kernel=private_aperture",
                         self.runtime.decode().splitlines()[1],
                         f"PS5HIP SUBMIT target={target} kernel=private_aperture count=64 groups=1 timeout_ms=2000",
                         *records,
                         f"PS5HIP APERTURE waves=2 within_wave=consistent across_waves={across} interpretation=raw-only",
                         "PS5HIP PASS kernel=private_aperture exact_values=128 raw_words=256 guards=unchanged allocations=closed"]
                good = "\n".join(lines) + "\n"
                result = module.check_log(good.encode(), target, "private_aperture")
                self.assertEqual(result["application_result"], "application-pass")
                self.assertTrue(result["aperture_observation_only"])
                self.assertEqual(len(result["aperture_records"]), 64)
                self.assertFalse(result["hardware_gate_passed"])
                self.assertIn("private aperture routing", result["unproven"])
                bad_logs = [good[:-1], good + records[0] + "\n", good.replace(records[1] + "\n", ""),
                            good.replace(records[1], records[0]), good.replace("exact_values=128", "exact_values=384"),
                            good.replace("raw_words=256", "raw_words=128"), good.replace("guards=unchanged", "guards=modified"),
                            good.replace("allocations=closed", "allocations=retained"),
                            good.replace("interpretation=raw-only", "interpretation=valid"),
                            good.replace("across_waves=" + across, "across_waves=" + ("different" if across == "equal" else "equal")),
                            good.replace(records[1], records[1].replace("base=0x0000000000000000", "base=0x0000000000000001")),
                            good.replace("sentinel=0x6b18d3a5", "sentinel=0x00000000"),
                            good.replace("base=0x0000000000000000", "base=0x0"), good.replace("\n", "\r\n"),
                            good.replace("count=64", "count=384"), good + "PS5HIP FAIL: cleanup\n",
                            good + "PS5HIP QUARANTINED: ownership\n"]
                for bad in bad_logs:
                    failed = module.check_log(bad.encode(), target, "private_aperture")
                    self.assertNotEqual(failed["application_result"], "application-pass")
                    self.assertEqual(failed["aperture_records"], [])
                suffix = ("PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\n"
                          "PS5HIP SHUTDOWN static-dtor=observed\nPS5HIP SHELL_EXIT application-fini=completed\n"
                          "PS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0\n")
                self.assertEqual(module.check_log((good + suffix).encode(), target, "private_aperture", shell_exit_probe=True)
                                 ["application_result"], "application-pass")
                self.assertNotEqual(module.check_log(good.encode(), target, "private_aperture", shell_exit_probe=True)
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
