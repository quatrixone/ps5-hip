"""Fail-closed lifecycle and transcript contracts; no sockets or console access."""
from pathlib import Path
import sys
import unittest
import tempfile
import subprocess
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from source_console import Control, lifecycle, snapshot, native_fault, sha
from source_public_hip_log import EXPECTED, MODULE_EXPECTED, LAUNCH_EXPECTED, check_log


class TransferAndRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.c = Control.__new__(Control)
        self.c.record = {}
        self.c.out = Path(self.temp.name)
        self.c.baseline_log = '/results/memory.log'
        self.c.log = '/results/copy.log'
        self.raw = b'previous verified transcript\n'
        self.digest = sha(self.raw)
        self.backup = self.c.baseline_log + '.' + self.digest + '.preserved'
        self.c.manifest = {'baseline_log_sha256': self.digest}
        self.c.baseline_log_preserved = True
        self.files = {self.backup: self.raw}
        self.renames = []
        owner = self
        class FTP:
            def retrbinary(self, command, append, blocksize):
                append(owner.files[command.removeprefix('RETR ')])
            def rename(self, src, dst):
                owner.renames.append((src, dst))
                owner.files[dst] = owner.files.pop(src)
        self.c.ftp = FTP()
        self.c.listing = lambda path: ['file ' + Path(p).name for p in self.files if str(Path(p).parent) == path]

    def test_existing_preserved_log_verified_without_rename(self):
        previous = self.c.baseline_log_read()
        self.c.preserve_baseline_log(previous)
        self.assertEqual(self.renames, [])
        self.assertEqual(self.c.record['log_backup'], self.backup)
        self.assertTrue(self.c.record['log_already_preserved'])
        self.assertEqual((self.c.out / 'previous-application.log').read_bytes(), self.raw)

    def test_preserved_log_missing_corrupt_or_conflicting_fails(self):
        for files in ({}, {self.backup: b'wrong'},
                      {self.backup: self.raw, self.c.baseline_log: self.raw},
                      {self.backup: self.raw, self.c.log: b'stale'}):
            self.files = files
            with self.assertRaises((RuntimeError, KeyError)):
                self.c.baseline_log_read()
            self.assertEqual(self.renames, [])

    def test_preserved_log_changed_after_initial_verification_fails(self):
        previous = self.c.baseline_log_read()
        self.files[self.backup] = b'changed after upload'
        with self.assertRaisesRegex(RuntimeError, 'preserved log mismatch'):
            self.c.preserve_baseline_log(previous)
        self.assertNotIn('log_backup', self.c.record)
        self.assertEqual(self.renames, [])

    def test_original_log_reappearing_aborts_preserved_mode(self):
        previous = self.c.baseline_log_read()
        self.files[self.c.baseline_log] = self.raw
        with self.assertRaisesRegex(RuntimeError, 'path already exists'):
            self.c.preserve_baseline_log(previous)
        self.assertNotIn('log_backup', self.c.record)
        self.assertEqual(self.renames, [])

    def test_freezer_requires_preserved_log_digest_before_output(self):
        root = Path(__file__).resolve().parents[1]
        output = self.c.out / 'must-not-exist'
        result = subprocess.run([sys.executable, str(root / 'tools/freeze_source_candidate.py'),
                                 '--build', str(self.c.out / 'missing-build'), '--output', str(output),
                                 '--baseline-image-sha256', '0' * 64, '--baseline-log-preserved'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('--baseline-log-preserved requires --baseline-log-sha256', result.stderr)
        self.assertFalse(output.exists())

    def test_original_preservation_and_collision(self):
        self.c.baseline_log_preserved = False
        self.files = {self.c.baseline_log: self.raw}
        previous = self.c.baseline_log_read()
        self.c.preserve_baseline_log(previous)
        self.assertEqual(self.renames, [(self.c.baseline_log, self.backup)])
        self.assertFalse(self.c.record['log_already_preserved'])
        self.files[self.c.baseline_log] = self.raw
        with self.assertRaisesRegex(RuntimeError, 'path already exists'):
            self.c.baseline_log_read()

    def test_fetch_success_and_size_failure_recorded(self):
        self.assertEqual(self.c.fetch(self.backup, len(self.raw)), self.raw)
        entry = self.c.record['ftp_reads'][-1]
        self.assertTrue(entry['complete'])
        self.assertEqual(entry['received_sha256'], self.digest)
        with self.assertRaisesRegex(RuntimeError, 'size bound'):
            self.c.fetch(self.backup, len(self.raw) - 1)
        entry = self.c.record['ftp_reads'][-1]
        self.assertFalse(entry['complete'])
        self.assertEqual(entry['received_bytes'], len(self.raw))

    def test_deadline_and_socket_failure_never_complete(self):
        with patch('source_console.time.monotonic', side_effect=[0, 31, 31]):
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                self.c.fetch(self.backup)
        self.assertFalse(self.c.record['ftp_reads'][-1]['complete'])
        def partial(command, append, blocksize):
            append(b'prefix')
            raise TimeoutError('stalled')
        self.c.ftp.retrbinary = partial
        with self.assertRaises(TimeoutError):
            self.c.fetch(self.backup)
        entry = self.c.record['ftp_reads'][-1]
        self.assertFalse(entry['complete'])
        self.assertEqual(entry['received_bytes'], 6)
        self.assertEqual(entry['received_sha256'], sha(b'prefix'))

    def test_late_empty_retrieval_does_not_pass(self):
        self.c.ftp.retrbinary = lambda *args, **kwargs: None
        with patch('source_console.time.monotonic', side_effect=[0, 31, 31]):
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                self.c.fetch(self.backup)
        entry = self.c.record['ftp_reads'][-1]
        self.assertFalse(entry['complete'])
        self.assertEqual(entry['received_bytes'], 0)
        self.assertEqual(entry['received_sha256'], sha(b''))


class ConsoleContractTest(unittest.TestCase):
    def test_native_faults_rejected(self):
        for raw in (b'# exception: 6', b'# A user thread receives a fatal signal', b'Kernel panic', b'panic: fatal'):
            self.assertTrue(native_fault(raw))
        self.assertFalse(native_fault(b'[KSTUFF] game stopped: PPSA99014 pid=629'))

    start = b'[GAME] started: PPSA99014 pid=501\n'
    stop = b'[KSTUFF] game stopped: PPSA99014 pid=501\n'

    def test_ordered_lifecycle(self):
        self.assertTrue(lifecycle(self.start + self.stop)[0])

    def test_reject_contaminated_lifecycles(self):
        for raw in (self.stop + self.start, self.start, self.start * 2 + self.stop,
                    self.start + self.stop * 2, self.start + self.stop.replace(b'501', b'502'),
                    self.start + b'[GAME] started: PPSA99015 pid=502\n' + self.stop):
            with self.subTest(raw=raw):
                self.assertFalse(lifecycle(raw)[0])

    def test_exact_application_log(self):
        self.assertEqual(check_log(EXPECTED)['application_result'], 'application-pass')
        for raw in (EXPECTED[:-1], EXPECTED + EXPECTED, EXPECTED + b'FAIL\n'):
            with self.assertRaises(ValueError):
                check_log(raw)

    def test_launch_transcript_requires_exact_results_and_cleanup(self):
        result = check_log(LAUNCH_EXPECTED, 'launch')
        self.assertEqual(result['kernel'], 'public_hip_launch')
        self.assertFalse(result['hardware_gate_passed'])
        lines = LAUNCH_EXPECTED.splitlines(keepends=True)
        invalid = [b''.join(lines[:i] + lines[i+1:]) for i in range(len(lines))]
        invalid += [LAUNCH_EXPECTED * 2, MODULE_EXPECTED, LAUNCH_EXPECTED + b'FAIL\n']
        for before, after in ((b'launches=2', b'launches=1'), (b'values=146', b'values=145'),
                              (b'guards=16', b'guards=15'), (b'rejected=2', b'rejected=0'),
                              (b'frees=1', b'frees=0'), (b'retired=1', b'retired=0'),
                              (b'gate_destroyed=true', b'gate_destroyed=false')):
            invalid.append(LAUNCH_EXPECTED.replace(before, after))
        for raw in invalid:
            with self.assertRaises(ValueError):
                check_log(raw, 'launch')

    def test_module_transcript_rejects_incomplete_or_wrong_claims(self):
        result = check_log(MODULE_EXPECTED, 'module')
        self.assertEqual(result['kernel'], 'public_hip_module')
        self.assertFalse(result['hardware_gate_passed'])
        lines = MODULE_EXPECTED.splitlines(keepends=True)
        invalid = [b''.join(lines[:i] + lines[i+1:]) for i in range(len(lines))]
        invalid += [MODULE_EXPECTED + MODULE_EXPECTED, EXPECTED]
        for before, after in ((b'loads=4', b'loads=3'), (b'lookups=8', b'lookups=7'),
                              (b'unloads=4', b'unloads=3'), (b'isolated=true', b'isolated=false'),
                              (b'gpu_execution=false', b'gpu_execution=true'),
                              (b'queues=0 retired=0', b'queues=1 retired=1'),
                              (b'allocations=0 frees=0', b'allocations=1 frees=1')):
            invalid.append(MODULE_EXPECTED.replace(before, after))
        for raw in invalid:
            with self.assertRaises(ValueError):
                check_log(raw, 'module')
        for fixture in ('bootstrap', 'memory', 'copy', 'unknown'):
            with self.assertRaises(ValueError):
                check_log(MODULE_EXPECTED, fixture)

    def test_snapshot_requires_positive_mount_evidence(self):
        for raw in (b'', b'PS5HIP TEARDOWN BEGIN\nPS5HIP TEARDOWN END snapshots_stable=1 status_stable=1\n'):
            with self.assertRaises(RuntimeError):
                snapshot(raw)

    def test_snapshot_mounts_and_idle_queries(self):
        rows = [('registered_image', 'ufs', '/mnt/shadowmnt/PPSA99014_b6d9b2e4', '/dev/lvd1'),
                ('runtime_layer', 'nullfs', '/system_ex/app/PPSA99014', '/mnt/shadowmnt/PPSA99014_b6d9b2e4')]
        lines = ['PS5HIP TEARDOWN BEGIN']
        for phase in ('before', 'after'):
            lines.append(f'PS5HIP TEARDOWN STATUS phase={phase} pid=134 rc=-1 errno=3 running=0xffffffff title_app=0xffffffff')
        for index in (1, 2):
            for kind, fs, target, source in rows:
                lines.append(f'PS5HIP TEARDOWN MOUNT snapshot={index} kind={kind} type=0 flags=0x0 fsid_hex=00 '
                             f'fstype_hex={fs.encode().hex()} to_hex={target.encode().hex()} from_hex={source.encode().hex()}')
        lines.append('PS5HIP TEARDOWN END snapshots_stable=1 status_stable=1')
        raw = ('\n'.join(lines) + '\n').encode()
        self.assertEqual(len(snapshot(raw)), 2)
        for changed in (raw.replace(b'running=0xffffffff', b'running=0x00000018', 1),
                        raw.replace(b'snapshot=2 kind=runtime_layer', b'snapshot=2 kind=unrelated')):
            with self.assertRaises(RuntimeError):
                snapshot(changed)


if __name__ == '__main__':
    unittest.main()
