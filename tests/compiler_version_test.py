import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("builder", pathlib.Path(__file__).resolve().parents[1] / "tools/build_canaries.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class VersionTests(unittest.TestCase):
    def test_pinned_compiler_linker_and_inspectors(self):
        builder.require_tool_versions({"clang": "Ubuntu clang version 18.1.8 (20ubuntu8)",
                                       "lld": "Ubuntu LLD 18.1.8 (compatible with GNU linkers)",
                                       "readobj": "Ubuntu LLVM version 18.1.8\n  Optimized build."}, "18.1.8")

    def test_mixed_unrecognized_and_nearby_versions_fail(self):
        for text in ("", "18.1.8", "clang version 18.1.7", "LLD 19.1.8",
                     "LLVM version 18.1.80", "LLVM version 18.1.8custom", "LLVM version 18.1.8-rc1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                builder.require_tool_versions({"clang": "clang version 18.1.8", "other": text}, "18.1.8")


if __name__ == "__main__":
    unittest.main()
