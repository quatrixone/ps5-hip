import importlib.util
import pathlib
import hashlib
import json
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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

    def test_native_resource_build_rejects_stale_or_ambiguous_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            files = {'tools/build_canaries.py': b'recipe', 'kernels/resources.hip': b'resource kernel',
                     'build/canaries/gfx1013/resources/resources.hsaco': b'resource object',
                     'build/canaries/gfx1013/resources/driver.txt': b'"/pinned/lld" "elf64_amdgpu" "--whole-archive"'}
            for name, data in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            sha = lambda name: hashlib.sha256(files[name]).hexdigest()
            profile = {'target': 'gfx1013', 'bytes': len(b'resource object'),
                       'code_sha256': sha('build/canaries/gfx1013/resources/resources.hsaco'),
                       'driver_sha256': sha('build/canaries/gfx1013/resources/driver.txt')}
            report = {'recipe_sha256': sha('tools/build_canaries.py'),
                      'source_sha256': sha('kernels/resources.hip'),
                      'tool_versions': {'lld': 'LLD 18.1.8'}, 'profiles': [profile]}
            path = root / 'build/canaries/resources-build.json'
            path.write_text(json.dumps(report))
            # A present canary receipt/object must never substitute for resources.
            (root / 'build/canaries/build.json').write_text('{}')
            (root / 'build/canaries/gfx1013/canary.hsaco').write_bytes(b'canary decoy')
            verify = lambda: native.verified_device_object(root, 'gfx1013', '18.1.8', fixture='resources')
            with patch.object(native.shutil, 'which', return_value='/pinned/lld'):
                self.assertEqual(verify(), (b'resource object', report))
                for name, data in files.items():
                    (root / name).write_bytes(data + b'changed')
                    with self.subTest(stale=name), self.assertRaises(ValueError):
                        verify()
                    (root / name).write_bytes(data)
                for profiles in ([], [dict(profile, target='gfx1030')], [profile, profile],
                                 [dict(profile, bytes=1)], [dict(profile, code_sha256='wrong')],
                                 [{k: v for k, v in profile.items() if k != 'driver_sha256'}]):
                    path.write_text(json.dumps(dict(report, profiles=profiles)))
                    with self.subTest(profiles=profiles), self.assertRaises(ValueError):
                        verify()
                for field in ('recipe_sha256', 'source_sha256'):
                    path.write_text(json.dumps({k: v for k, v in report.items() if k != field}))
                    with self.subTest(missing=field), self.assertRaises(ValueError):
                        verify()
                path.write_text(json.dumps(dict(report, tool_versions={'lld': 'LLD 19.1.8'})))
                with self.assertRaises(ValueError):
                    verify()
                path.write_text(json.dumps(report))
                with patch.object(native.shutil, 'which', return_value='/other/lld'), self.assertRaises(ValueError):
                    verify()
            for fixture, wave in (('resources', root / 'wave'), ('../resources', None), ('unknown', None)):
                with self.subTest(fixture=fixture), self.assertRaisesRegex(ValueError, 'fixture selection'):
                    native.verified_device_object(root, 'gfx1013', '18.1.8', wave, fixture)

    def test_native_wave_build_rejects_stale_or_ambiguous_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            files = {'tools/build_canaries.py': b'shared', 'tools/probe_wave_compiler.py': b'wave recipe',
                     'kernels/wave.hip': b'wave source', 'wave/gfx1013-O2/wave.hsaco': b'object',
                     'wave/gfx1013-O2/driver.txt': b'"/pinned/lld" "elf64_amdgpu" "--whole-archive"'}
            for name, data in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            sha = lambda name: hashlib.sha256(files[name]).hexdigest()
            case = {'target': 'gfx1013', 'optimization': '-O2', 'bytes': 6,
                    'code_sha256': sha('wave/gfx1013-O2/wave.hsaco'),
                    'driver_sha256': sha('wave/gfx1013-O2/driver.txt')}
            report = {'recipe_sha256': sha('tools/probe_wave_compiler.py'),
                      'shared_recipe_sha256': sha('tools/build_canaries.py'),
                      'source_sha256': sha('kernels/wave.hip'),
                      'tool_versions': {'lld': 'LLD 18.1.8'}, 'cases': [case]}
            path = root / 'wave/comparison.json'
            path.write_text(json.dumps(report))
            with patch.object(native.shutil, 'which', return_value='/pinned/lld'):
                self.assertEqual(native.verified_device_object(root, 'gfx1013', '18.1.8', root / 'wave')[0], b'object')
                for name, data in files.items():
                    (root / name).write_bytes(data + b'changed')
                    with self.assertRaises(ValueError):
                        native.verified_device_object(root, 'gfx1013', '18.1.8', root / 'wave')

                    (root / name).write_bytes(data)
                for cases in ([], [dict(case, optimization='-O1')], [case, case], [dict(case, bytes=7)],
                              [{key: value for key, value in case.items() if key != 'driver_sha256'}]):
                    report['cases'] = cases
                    path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError):
                        native.verified_device_object(root, 'gfx1013', '18.1.8', root / 'wave')

    def test_wave_probe_option_is_bound_to_wave_kernels(self):
        for kernel, extra in (('wave_full', []), ('wave_partial', []),
                              ('runtime', ['--wave-probes', '/unused']),
                              ('indexed_store', ['--wave-probes', '/unused']),
                              ('register_spill', ['--wave-probes', '/unused'])):
            argv = ['native', '--native-source', '/unused', '--target', 'gfx1013', '--kernel', kernel, *extra]
            with patch.object(sys, 'argv', argv), patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                native.main()
            self.assertEqual(error.exception.code, 2)

    def test_pinned_compiler_linker_and_inspectors(self):
        builder.require_tool_versions({"clang": "Ubuntu clang version 18.1.8 (20ubuntu8)",
                                       "lld": "Ubuntu LLD 18.1.8 (compatible with GNU linkers)",
                                       "readobj": "Ubuntu LLVM version 18.1.8\n  Optimized build."}, "18.1.8")

    def test_mixed_unrecognized_and_nearby_versions_fail(self):
        for text in ("", "18.1.8", "clang version 18.1.7", "LLD 19.1.8",
                     "LLVM version 18.1.80", "LLVM version 18.1.8custom", "LLVM version 18.1.8-rc1"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                builder.require_tool_versions({"clang": "clang version 18.1.8", "other": text}, "18.1.8")

    def test_lds_provenance_and_profile_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            files = {'tools/build_canaries.py': b'shared', 'tools/probe_lds_compiler.py': b'recipe',
                     'kernels/lds.hip': b'kernel', 'lds/gfx1013-O2/lds.hsaco': b'object',
                     'lds/gfx1013-O2/driver.txt': b'"/pinned/lld" "elf64_amdgpu" "--whole-archive"'}
            for name, data in files.items():
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
            sha = lambda name: hashlib.sha256(files[name]).hexdigest()
            profiles = {name: {'group_segment_fixed_size': 0 if name == 'lds_dynamic' else 256,
                              'private_segment_fixed_size': 0, 'kernarg_segment_size': 16 if name == 'lds_static' else 20,
                              'wavefront_size': 32} for name in ('lds_static', 'lds_dynamic', 'lds_mixed')}
            case = {'target': 'gfx1013', 'optimization': '-O2', 'bytes': 6, 'profiles': profiles,
                    'code_sha256': sha('lds/gfx1013-O2/lds.hsaco'), 'driver_sha256': sha('lds/gfx1013-O2/driver.txt')}
            report = {'recipe_sha256': sha('tools/probe_lds_compiler.py'), 'source_sha256': sha('kernels/lds.hip'),
                      'shared_recipe_sha256': sha('tools/build_canaries.py'), 'tool_versions': {'lld': 'LLD 18.1.8'}, 'cases': [case]}
            path = root / 'lds/comparison.json'; path.write_text(json.dumps(report))
            verify = lambda: native.verified_device_object(root, 'gfx1013', '18.1.8', lds_directory=root / 'lds')
            with patch.object(native.shutil, 'which', return_value='/pinned/lld'):
                self.assertEqual(verify()[0], b'object')
                for name, data in files.items():
                    (root / name).write_bytes(data + b'changed')
                    with self.assertRaises(ValueError): verify()
                    (root / name).write_bytes(data)
                for cases in ([], [case, case], [dict(case, optimization='-O1')], [dict(case, target='gfx1030')],
                              [dict(case, bytes=7)], [dict(case, profiles={})]):
                    report['cases'] = cases; path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError): verify()
                report['cases'] = [case]
                for value in (False, 4):
                    profiles['lds_dynamic']['private_segment_fixed_size'] = value; path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError): verify()
                profiles['lds_dynamic']['private_segment_fixed_size'] = 0; path.write_text(json.dumps(report))
                with patch.object(native.shutil, 'which', return_value='/wrong/lld'), self.assertRaises(ValueError): verify()
                with self.assertRaises(ValueError):
                    native.verified_device_object(root, 'gfx1013', '18.1.8', root / 'wave', lds_directory=root / 'lds')
        for kernel, extra in (('lds_static', []), ('lds_dynamic', []), ('lds_mixed', []),
                              ('runtime', ['--lds-probes', '/unused']), ('register_spill', ['--lds-probes', '/unused']),
                              ('wave_full', ['--wave-probes', '/unused', '--lds-probes', '/unused'])):
            with patch.object(sys, 'argv', ['native', '--native-source', '/unused', '--target', 'gfx1013', '--kernel', kernel, *extra]), \
                    patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                native.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
