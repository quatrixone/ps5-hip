"""Offline Q4 control transport/acceptance regressions; never contacts a console."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_console as c
import verify_public_hip_q4 as v

class Q4Control(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.control=c.Control.__new__(c.Control)
        x=self.control;x.fixture='q4';x.out=self.root/'execute';x.out.mkdir();x.record={}
        x.manifest={'results_dir':'/results','q4':{'fixture':'fixture','reference':'reference'}}
        self.sizes={f'q4-{n}.{s}.{b}.raw':cap+32 for n,s,*_ in v.STAGES for b,cap in v.CAPACITIES.items()}
        x.listing=lambda path:['file '+n for n in self.sizes]
        x.fetch=lambda path,bound,deadline_seconds:bytes(self.sizes[Path(path).name])
    def test_freshness_and_exact_capture(self):
        x=self.control
        with self.assertRaisesRegex(RuntimeError,'already exists'):x.fresh_q4_dumps()
        x.listing=lambda path:[];x.fresh_q4_dumps()
        self.assertEqual(set(x.record['q4_raw_absent_before_launch']),v.DUMP_NAMES)
        x.listing=lambda path:['file '+n for n in self.sizes];x.collect_q4_dumps()
        self.assertEqual(len(x.record['q4_raw_files']),112)
        self.assertEqual(sum(r['bytes'] for r in x.record['q4_raw_files'].values()),9097984)
        first=next(iter(self.sizes));(x.out/'raw'/first).write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'dump changed'):x.collect_q4_dumps()
    def test_inventory_wrong_size_and_partial_failure(self):
        x=self.control;x.listing=lambda path:['file q4-extra.raw']
        with self.assertRaisesRegex(RuntimeError,'inventory'):x.collect_q4_dumps()
        x.listing=lambda path:['file '+n for n in self.sizes]
        x.fetch=lambda *a,**k:b'short'
        with self.assertRaisesRegex(RuntimeError,'length'):x.collect_q4_dumps()
        self.assertFalse(x.record['q4_raw_files'])
        first=next(iter(self.sizes));self.assertEqual((x.out/'raw'/first).read_bytes(),b'short')
        x.collect_q4_dumps(partial=True)
        self.assertEqual(len(x.record['q4_raw_capture_errors']),112)
        self.assertEqual((x.out/'raw'/first).read_bytes(),b'short')
    def test_total_deadline_shared_with_partial_capture(self):
        x=self.control;x.q4_capture_deadline=10
        with patch.object(c.time,'monotonic',return_value=11):
            with self.assertRaisesRegex(RuntimeError,'total raw capture deadline'):x.collect_q4_dumps()
            x.collect_q4_dumps(partial=True)
        self.assertTrue(x.record['q4_raw_capture_deadline_exceeded'])
        self.assertFalse(x.record['q4_raw_files'])
    def test_metrics_must_match_every_stage(self):
        x=self.control;x.verify=lambda:None
        records=[dict(name=n,stage=s,maximum_absolute_error=0,maximum_budget_fraction=0) for n,s,*_ in v.STAGES]
        metrics=dict(max_abs=0,max_budget_fraction=0,stages=[dict(name=r['name'],stage=r['stage'],max_abs=0,max_budget_fraction=0) for r in records])
        x.record['application']={'reported_metrics':metrics}
        def verify(fixture,reference,dumps,output):
            output.parent.mkdir(exist_ok=True);output.write_text('{}')
            return dict(status='PASS',records=records)
        with patch('verify_public_hip_q4.verify',side_effect=verify):
            x.verify_q4_dumps();self.assertEqual(x.record['q4_numerical']['status'],'PASS')
            del x.record['q4_numerical'];metrics['stages'][0]['max_abs']=1
            with self.assertRaisesRegex(RuntimeError,'metrics differ'):x.verify_q4_dumps()
            self.assertNotIn('q4_numerical',x.record)
    def test_authentication_binds_reference_modules_and_package(self):
        x=self.control;x.candidate=self.root/'candidate';app=x.candidate/'package/app';app.mkdir(parents=True)
        fixture=self.root/'fixture';fixture.mkdir();ref=x.candidate/'reference';ref.mkdir()
        (ref/'receipt.json').write_bytes(b'receipt')
        assets={}
        for n in ('q4','q8'):
            f=fixture/(n+'.hsaco');f.write_bytes(n.encode());dest=app/'q4'/f.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(f.read_bytes());assets['q4/'+f.name]=str(f)
        x.manifest.update(files={'candidate/reference/receipt.json':c.sha(b'receipt')},package_files={n:c.sha(Path(f).read_bytes()) for n,f in assets.items()},
            q4={'fixture':'fixture','reference':'candidate/reference','reference_receipt_sha256':c.sha(b'receipt'),'module_sha256':{n:c.sha(n.encode()) for n in ('q8','q4')}})
        bound=dict(reference={'receipt_sha256':c.sha(b'receipt')},assets=assets)
        with patch.object(c,'ROOT',self.root),patch('public_hip_q4_assets.assets',return_value=bound):
            x.authenticate_q4()
            (app/'q4/q8.hsaco').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'packaged asset differs'):x.authenticate_q4()
            (app/'q4/q8.hsaco').write_bytes(b'q8');x.manifest['q4']['reference']='fixture'
            with self.assertRaisesRegex(RuntimeError,'outside frozen scope'):x.authenticate_q4()

if __name__=='__main__':unittest.main()
