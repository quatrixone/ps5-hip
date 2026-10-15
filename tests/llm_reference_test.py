import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_llm_cpu_reference as builder


class ReferenceTest(unittest.TestCase):
    def test_driver_alias_and_failed_rerun_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('llm-inputs.json', 'tools/build_llm_cpu_reference.py', 'tools/fetch_locked_deps.py'):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('test input')
            client = root / 'client'
            prerequisite = client / 'ggml/src/ggml-cpu/CMakeLists.txt'
            prerequisite.parent.mkdir(parents=True)
            prerequisite.write_text('test')
            binaries = root / 'bin'
            binaries.mkdir()
            for name in ('cmake', 'ninja', 'clang-18', 'clang++-18'):
                (binaries / name).symlink_to(Path(sys.executable).resolve())
            output = root / 'build/reference'
            spec = {'client': {'revision': 'test-revision', 'source_url': 'test-source'}}
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'read', return_value=spec), \
                 patch.object(builder, 'source', return_value=client), \
                 patch.object(builder.shutil, 'which', side_effect=lambda name: str(binaries / name)), \
                 patch.object(builder.subprocess, 'check_output', return_value='clang version 18.1.8\n'), \
                 patch.object(sys, 'argv', ['builder', '--output', str(output)]):
                with patch.object(builder.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
                    with self.assertRaises(subprocess.CalledProcessError):
                        builder.main()
                config = json.loads((output / 'configuration.json').read_text())
                self.assertIn('-DCMAKE_CXX_COMPILER=' + str(binaries / 'clang++-18'), config['flags'])
                self.assertNotEqual(config['tools']['clang++-18']['path'], config['tools']['clang++-18']['resolved_path'])
                (output / 'receipt.json').write_text('{"old_success":true}')
                with patch.object(builder.subprocess, 'run', side_effect=subprocess.TimeoutExpired('configure', 300)):
                    with self.assertRaises(subprocess.TimeoutExpired):
                        builder.main()
                self.assertFalse((output / 'receipt.json').exists())
                self.assertTrue(json.loads((output / 'commands.json').read_text())[0]['timed_out'])
                config['source_revision'] = 'changed'
                (output / 'configuration.json').write_text(json.dumps(config))
                with patch.object(builder.subprocess, 'run') as run:
                    with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
                        builder.main()
                    run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
