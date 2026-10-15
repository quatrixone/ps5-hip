"""Offline memory patch and fixture-isolation regressions; no console access."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_public_hip_memory import prepare, digest, AFTER, SVM
from source_public_hip_log import check_log, EXPECTED, MEMORY_EXPECTED, COPY_EXPECTED


class PublicMemoryTest(unittest.TestCase):
    def test_fixture_transcripts_are_not_interchangeable(self):
        self.assertEqual(check_log(MEMORY_EXPECTED, 'memory')['kernel'], 'public_hip_memory')
        self.assertEqual(check_log(COPY_EXPECTED, 'copy')['kernel'], 'public_hip_copy')
        for raw, fixture in ((EXPECTED, 'memory'), (MEMORY_EXPECTED, 'bootstrap'),
                             (COPY_EXPECTED, 'memory'), (MEMORY_EXPECTED, 'copy'),
                             (COPY_EXPECTED.replace(b'compared_bytes=1734', b'compared_bytes=1733'), 'copy'),
                             (COPY_EXPECTED.replace(b'queues=1 retired=1', b'queues=1 retired=0'), 'copy'),
                             (MEMORY_EXPECTED.replace(b'frees=2', b'frees=1'), 'memory'),
                             (MEMORY_EXPECTED.replace(b'interior_rejected=true', b'interior_rejected=false'), 'memory'),
                             (MEMORY_EXPECTED[:-1], 'memory'), (MEMORY_EXPECTED, 'unknown')):
            with self.assertRaises(ValueError):
                check_log(raw, fixture)

    def test_exact_patch_and_corrupt_source_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in ('rocclr/platform/memory.cpp', 'hipamd/src/hip_memory.cpp'):
                target = source / name
                target.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / '.deps/sources/clr' / name, target)
            patch = (ROOT / 'ports/clr/ps5-public-memory.patch').read_text().split('--- a/hipamd/')[0]
            subprocess.run(['patch', '--batch', '--fuzz=0', '-p1'], cwd=source, input=patch,
                           text=True, check=True, capture_output=True)
            self.assertEqual(digest(source / 'rocclr/platform/memory.cpp'), SVM)
            prepare(source)
            self.assertEqual(digest(source / 'hipamd/src/hip_memory.cpp'), AFTER)
            prepare(source)  # Cached validation must be idempotent, never apply twice.
            with (source / 'hipamd/src/hip_memory.cpp').open('a') as stream:
                stream.write('\n// corruption\n')
            with self.assertRaises(ValueError):
                prepare(source)

    def test_fixture_mismatch_rejected_before_source_mutation(self):
        (ROOT / 'build').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as temporary:
            output = Path(temporary)
            receipt = output / 'link-receipt.json'
            receipt.write_text(json.dumps({'fixture': 'bootstrap'}))
            before = receipt.read_bytes()
            result = subprocess.run([sys.executable, str(ROOT / 'tools/build_public_hip.py'),
                                     '--fixture', 'memory', '--output', str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'cached fixture differs', result.stderr)
            self.assertEqual(receipt.read_bytes(), before)
            self.assertEqual(list(output.iterdir()), [receipt])


if __name__ == '__main__':
    unittest.main()
