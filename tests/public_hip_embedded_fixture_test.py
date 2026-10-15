"""Compiler probe failures must leave a diagnostic record and no success receipt."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from build_public_hip_embedded_fixture import build


class EmbeddedFixtureTest(unittest.TestCase):
    def test_timeout_retained_and_existing_output_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
            output = Path(directory) / 'probe'
            timeout = subprocess.TimeoutExpired(['clang++-18', '--version'], 180, output=b'partial', stderr=b'diagnostic')
            with patch('build_public_hip_embedded_fixture.subprocess.run', side_effect=timeout):
                with self.assertRaises(subprocess.TimeoutExpired):
                    build(output)
            commands = json.loads((output / 'commands.json').read_text())
            self.assertTrue(commands[0]['timed_out'])
            self.assertEqual((output / 'clang++-18-version.stdout').read_bytes(), b'partial')
            self.assertEqual((output / 'clang++-18-version.stderr').read_bytes(), b'diagnostic')
            self.assertFalse((output / 'receipt.json').exists())
            with self.assertRaises(FileExistsError):
                build(output)


if __name__ == '__main__':
    unittest.main()
