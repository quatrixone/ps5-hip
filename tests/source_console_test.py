"""Fail-closed lifecycle and transcript contracts; no sockets or console access."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from source_console import lifecycle, snapshot, native_fault
from source_public_hip_log import EXPECTED, check_log


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
