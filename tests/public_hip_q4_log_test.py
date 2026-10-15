import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from source_public_hip_log import LINES,check_log
from verify_public_hip_q4 import STAGES


def transcript():
    lines=[LINES[0].replace('bootstrap','q4'),LINES[1]]
    for name,stage,width,rows,padded in STAGES:
        error,fraction=('0','0') if stage=='q8' else ('1e-06','0.01')
        lines.extend([f'HIP_Q4 DUMP case={name} stage={stage} files=4 bytes=324928 exclusive=true synced=true',
            f'HIP_Q4 CASE case={name} stage={stage} rows={rows} values={0 if stage=="q8" else rows} q8_bytes={padded//32*36} max_abs={error} max_budget_fraction={fraction} inputs_unchanged=true guards=true inactive=true'])
    lines.extend(['HIP_Q4 PASS cases=12 stages=28 launches=28 values=592 q8_bytes=80640 raw_files=112 raw_bytes=9097984 max_abs=1e-06 max_budget_fraction=0.01 allocations=4 frees=4 modules_unloaded=2 watchdog_joined=true finalization=pending',
        *LINES[3:6],'HIP_Q4 FINAL allocations=4 frees=4 queues=1 retired=1 gate_destroyed=true application_fini=pending',*LINES[7:]])
    return ('\n'.join(lines)+'\n').encode('ascii')

class Q4Log(unittest.TestCase):
    def test_complete_transcript_is_not_hardware_proof(self):
        result=check_log(transcript(),'q4')
        self.assertFalse(result['hardware_gate_passed'])
        self.assertEqual(len(result['reported_metrics']['stages']),28)
        self.assertEqual(result['reported_metrics']['max_abs'],1e-6)
    def test_framing_order_lifecycle_and_counts(self):
        raw=transcript();lines=raw.splitlines(keepends=True);lines[2],lines[4]=lines[4],lines[2]
        for broken in [raw[:-1],raw+b'extra\n',raw.replace(b'\n',b'\r\n'),b''.join(lines),
                       raw.replace(b'files=4',b'files=3',1),raw.replace(b'values=592',b'values=591'),
                       raw.replace(b'stage=chained',b'stage=frozen',1),raw.replace(b'inputs_unchanged=true',b'inputs_unchanged=false',1),
                       raw.replace(b'application-fini=completed',b'application-fini=pending'),
                       raw.replace(b'allocations=4 frees=4',b'allocations=4 frees=3',1)]:
            with self.subTest(broken=broken[-80:]),self.assertRaises(ValueError):check_log(broken,'q4')
    def test_numerical_metrics(self):
        raw=transcript()
        for before,after in [(b'max_abs=1e-06',b'max_abs=nan'),(b'max_abs=1e-06',b'max_abs=1e999'),
                             (b'max_abs=1e-06',b'max_abs=-1'),(b'max_abs=1e-06',b'max_abs=2e-06'),
                             (b'max_budget_fraction=0.01',b'max_budget_fraction=1.01'),
                             (b'max_abs=0 ',b'max_abs=1 '),(b'max_budget_fraction=0 ',b'max_budget_fraction=0.1 ')]:
            with self.subTest(after=after),self.assertRaises(ValueError):check_log(raw.replace(before,after,1),'q4')

if __name__=='__main__':unittest.main()
