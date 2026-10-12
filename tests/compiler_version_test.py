import importlib.util
import pathlib
import hashlib
import json
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

spec = importlib.util.spec_from_file_location("builder", pathlib.Path(__file__).resolve().parents[1] / "tools/build_canaries.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
sys.modules['build_canaries'] = builder
native_spec = importlib.util.spec_from_file_location("native_builder", pathlib.Path(__file__).resolve().parents[1] / "tools/build_native_canary.py")
native = importlib.util.module_from_spec(native_spec)
native_spec.loader.exec_module(native)
comparison_spec = importlib.util.spec_from_file_location("comparison", pathlib.Path(__file__).resolve().parents[1] / "tools/compare_device_compiler.py")
comparison = importlib.util.module_from_spec(comparison_spec)
comparison_spec.loader.exec_module(comparison)


class VersionTests(unittest.TestCase):
    def test_comparison_preserves_all_linker_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name in ('tools/build_canaries.py', 'build/host/ps5hip-inspect',
                         'bin/clang++', 'bin/lld', 'bin/llvm-readobj', 'bin/llvm-objdump'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            for fixture in ('canary', 'resources'):
                source = root / 'kernels' / (fixture + '.hip')
                source.parent.mkdir(exist_ok=True)
                source.write_bytes(b'fixture')
                profiles = []
                for target in ('gfx1013', 'gfx1030'):
                    code = root / 'build/canaries' / target / ('canary.hsaco' if fixture == 'canary' else 'resources/resources.hsaco')
                    code.parent.mkdir(parents=True, exist_ok=True)
                    code.write_bytes(b'fixture')
                    profiles.append({'target': target, 'code_sha256': comparison.sha(code),
                                     'compile_command': ['clang++', '-B/old', '-c', str(source), '-o', str(code)]})
                report = {'source_sha256': comparison.sha(source),
                          'recipe_sha256': comparison.sha(root / 'tools/build_canaries.py'), 'profiles': profiles}
                (root / 'build/canaries' / ('build.json' if fixture == 'canary' else 'resources-build.json')).write_text(json.dumps(report))

            def capture(command, output):
                self.assertTrue('--version' in command or '-###' in command, 'rejected linker must never compile')
                stderr = '\"/wrong/lld\" \"elf64_amdgpu\" \"--whole-archive\"' if '-###' in command else ''
                output.write_text('test version\n' + stderr)
                return subprocess.CompletedProcess(command, 0, 'test version\n', stderr)

            output = root / 'output'
            with patch.object(comparison, 'ROOT', root), patch.object(comparison, 'capture', side_effect=capture), \
                    patch.object(sys, 'argv', ['comparison', '--compiler-bin', str(root / 'bin'), '--output', str(output)]), \
                    patch('builtins.print'):
                self.assertEqual(comparison.main(), 1)
            cases = json.loads((output / 'comparison.json').read_text())['cases']
            self.assertEqual(len(cases), 4)
            self.assertTrue(all(not case['compiled'] and 'linker differs' in case['failure'] for case in cases))

    def test_native_build_rejects_stale_device_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            files = {'tools/build_canaries.py': b'recipe', 'kernels/canary.hip': b'kernel',
                     'build/canaries/gfx1013/canary.hsaco': b'object',
                     'build/canaries/gfx1013/driver.txt': b'"/pinned/lld" "elf64_amdgpu" "--whole-archive"'}
            for name, data in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            sha = lambda name: hashlib.sha256(files[name]).hexdigest()
            report = {'recipe_sha256': sha('tools/build_canaries.py'),
                      'source_sha256': sha('kernels/canary.hip'),
                      'tool_versions': {'lld': 'LLD 18.1.8'},
                      'profiles': [{'target': 'gfx1013', 'bytes': 6,
                                    'code_sha256': sha('build/canaries/gfx1013/canary.hsaco'),
                                    'driver_sha256': sha('build/canaries/gfx1013/driver.txt')}]}
            (root / 'build/canaries/build.json').write_text(json.dumps(report))
            with patch.object(native.shutil, 'which', return_value='/pinned/lld'):
                self.assertEqual(native.verified_device_object(root, 'gfx1013', '18.1.8')[0], b'object')
                for name, data in files.items():
                    (root / name).write_bytes(data + b'changed')
                    with self.assertRaises(ValueError):
                        native.verified_device_object(root, 'gfx1013', '18.1.8')
                    (root / name).write_bytes(data)
                del report['recipe_sha256']
                (root / 'build/canaries/build.json').write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    native.verified_device_object(root, 'gfx1013', '18.1.8')

    def test_actual_device_linker_must_match(self):
        command = ' "/pinned/lld" "-flavor" "gnu" "-m" "elf64_amdgpu" "--whole-archive" "input.o"'
        builder.require_device_linker(command, '/pinned/lld')
        for trace in ('', command.replace('/pinned/lld', '/other/lld'), command + '\n' + command):
            with self.assertRaises(ValueError):
                builder.require_device_linker(trace, '/pinned/lld')

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
