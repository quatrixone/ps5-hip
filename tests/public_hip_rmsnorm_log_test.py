import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from source_public_hip_log import LINES, check_log


def transcript():
    lines = [LINES[0].replace('bootstrap', 'rmsnorm'), LINES[1]]
    for family in ('stress', 'embedding'):
        for rows in (1, 2, 17):
            lines += [f'HIP_RMSNORM DUMP family={family} rows={rows} output_bytes=139296 input_readback_bytes=139296 exclusive=true synced=true',
                      f'HIP_RMSNORM CASE family={family} rows={rows} values={rows*2048} max_abs=1e-06 max_budget_fraction=0.01 input_unchanged=true guards=true inactive=true']
    lines += ['HIP_RMSNORM PASS cases=6 launches=6 values=81920 guard_bytes=192 inactive_bytes=507904 input_bytes=835776 max_abs=1e-06 max_budget_fraction=0.01 allocations=2 frees=2 module_unloaded=true watchdog_joined=true finalization=pending',
              *LINES[3:6], 'HIP_RMSNORM FINAL allocations=2 frees=2 queues=1 retired=1 gate_destroyed=true application_fini=pending', *LINES[7:]]
    return ('\n'.join(lines)+'\n').encode('ascii')


class RmsnormLog(unittest.TestCase):
    def test_complete_transcript_is_not_hardware_proof(self):
        result = check_log(transcript(), 'rmsnorm')
        self.assertEqual(result['reported_metrics']['max_abs'], 1e-6)
        self.assertFalse(result['hardware_gate_passed'])

    def test_rejects_incomplete_or_changed_contract(self):
        raw = transcript()
        for broken in (raw[:-1], raw+b'extra\n', raw.replace(b'\n', b'\r\n'), raw.replace(b'guards=true', b'guards=false', 1),
                       raw.replace(b'values=81920', b'values=1'), raw.replace(b'output_bytes=139296', b'output_bytes=1', 1),
                       raw.replace(b'application-fini=completed', b'application-fini=pending'), raw.replace(b'family=stress', b'family=embedding', 1)):
            with self.subTest(broken=broken[-80:]), self.assertRaises(ValueError): check_log(broken, 'rmsnorm')

    def test_rejects_nonfinite_overbudget_or_inconsistent_metrics(self):
        raw = transcript()
        for before, after in ((b'max_abs=1e-06', b'max_abs=nan'), (b'max_abs=1e-06', b'max_abs=1e999'),
                              (b'max_abs=1e-06', b'max_abs=-1'), (b'max_budget_fraction=0.01', b'max_budget_fraction=1.1'),
                              (b'max_abs=1e-06', b'max_abs=2e-06')):
            with self.subTest(after=after), self.assertRaises(ValueError): check_log(raw.replace(before, after, 1), 'rmsnorm')

if __name__ == '__main__': unittest.main()
