"""Reject stale/unsupported compiler fixtures before native linking."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_public_hip as build


class EmbeddedBuildTest(unittest.TestCase):
    def test_verified_inputs_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'compiler'
            fixture.mkdir()
            sources = ['tests/public_hip_embedded.hip', 'tools/build_public_hip_embedded_fixture.py']
            for name in sources:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'source')
            obj = fixture / 'ps5/public-embedded.o'
            obj.parent.mkdir()
            obj.write_bytes(b'compiler object')
            tool = root / 'compiler-tool'
            tool.write_bytes(b'tool')
            report = {'classification': 'public-hip-genuine-embedded-compiler-fixture',
                      'runtime_executed': False, 'native_executed': False, 'gpu_executed': False,
                      'inputs': {str(root / n): build.sha(root / n) for n in sources},
                      'artifacts': {'ps5/public-embedded.o': build.sha(obj)},
                      'tools': {'compiler': {'path': str(tool), 'sha256': build.sha(tool)}},
                      'profiles': {'ps5': {'a': {'object': 'ps5/public-embedded.o'}}}}
            receipt = fixture / 'receipt.json'
            def save():
                receipt.write_text(json.dumps(report))
            with patch.object(build, 'ROOT', root):
                save()
                result = build.embedded_fixture_inputs(fixture)
                self.assertEqual(result['object_sha256'], build.sha(obj))
                for path in (obj, tool, root / sources[0]):
                    original = path.read_bytes()
                    path.write_bytes(b'drift')
                    with self.assertRaises(ValueError):
                        build.embedded_fixture_inputs(fixture)
                    path.write_bytes(original)
                report['native_executed'] = True
                save()
                with self.assertRaises(ValueError):
                    build.embedded_fixture_inputs(fixture)
                report['native_executed'] = False
                report['artifacts']['../outside'] = '0' * 64
                save()
                with self.assertRaises(ValueError):
                    build.embedded_fixture_inputs(fixture)

    def test_aggregate_uses_strict_verifier_and_exact_native_object(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            receipt = folder / 'receipt.json'
            obj = folder / 'ps5-fixture.o'
            verified = {str(receipt): 'receipt-digest', str(obj): 'object-digest'}
            with patch.object(build, 'aggregate_compiler_inputs', return_value=verified) as verifier:
                result = build.compiler_fixture_inputs(folder, 'aggregate')
                verifier.assert_called_once_with(folder)
                self.assertEqual(result, dict(receipt=str(receipt), receipt_sha256='receipt-digest',
                                              object=str(obj), object_sha256='object-digest'))
            with patch.object(build, 'aggregate_compiler_inputs', side_effect=ValueError('stale source')):
                with self.assertRaisesRegex(ValueError, 'stale source'):
                    build.compiler_fixture_inputs(folder, 'aggregate')
            with self.assertRaisesRegex(ValueError, 'embedded or aggregate'):
                build.compiler_fixture_inputs(folder, 'launch')
            with patch.object(build, 'embedded_fixture_inputs', return_value={'embedded': True}) as verifier:
                self.assertEqual(build.compiler_fixture_inputs(folder, 'embedded'), {'embedded': True})
                verifier.assert_called_once_with(folder)

    def test_compiler_link_uses_exact_object_and_variant_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deps = root / 'deps'
            original = deps / 'sources/native-tools/tooling/native/ps5-pie.ld'
            original.parent.mkdir(parents=True); original.write_text('native linker')
            port = root / 'ports/clr/ps5-pie.ld'
            port.parent.mkdir(parents=True); port.write_text('/* Audited optional provider absence */')
            for fixture, launcher, objname in [('embedded', 'launch_public_embedded', 'public-embedded.o'),
                                               ('aggregate', 'launch_aggregate_fixture', 'ps5-fixture.o')]:
                output = root / fixture; output.mkdir()
                obj = root / objname
                names = ['hipInit','hipGetDeviceCount','ps5hip_clr_admit_device','hipMalloc','hipFree',
                         'hipMemcpy','hipDeviceSynchronize','hipLaunchKernel',launcher,
                         '__hipRegisterFatBinary','__hipRegisterFunction','__hipUnregisterFatBinary',
                         '__hipPushCallConfiguration','__hipPopCallConfiguration']
                symbols = '\n'.join('DEF '+name for name in names)
                with patch.object(build, 'ROOT', root), patch.object(build, 'DEPS', deps), \
                     patch.object(build, 'embedded_linker', return_value='linked bounds') as linker, \
                     patch.object(build, 'run') as run, \
                     patch.object(build.subprocess, 'check_output', return_value=symbols):
                    build.link_title(output, [], fixture, {'object': str(obj)})
                    linker.assert_called_once_with('native linker', [objname])
                    self.assertIn(obj, run.call_args.args[2])
                    self.assertEqual(run.call_args.args[2].count(obj), 1)
                    with self.assertRaisesRegex(ValueError, 'verified compiler fixture'):
                        build.link_title(output, [], fixture)


if __name__ == '__main__':
    unittest.main()
