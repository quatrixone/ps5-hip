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


if __name__ == '__main__':
    unittest.main()
