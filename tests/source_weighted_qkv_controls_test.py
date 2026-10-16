"""Offline weighted QKV transcript, freeze and bounded capture gates."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_public_hip_log as log
import source_console as console
import freeze_source_candidate as freeze
import verify_weighted_qkv_readbacks as verifier

def transcript():
 lines=[log.LINES[0].replace('bootstrap','weighted_qkv'),log.LINES[1]]
 for s in verifier.STAGES:
  rows=2048 if s=='q' else 256 if s in ('k','v') else 0
  lines += [f'HIP_WEIGHTED_QKV DUMP stage={s} files=6 bytes=2394560 exclusive=true synced=true',f'HIP_WEIGHTED_QKV STAGE stage={s} weighted_values={2048 if s=="weighted" else 0} q8_bytes={2304 if s=="q8" else 0} projection_values={rows} unchanged=true guards=true inactive=true shared_intermediates=true numerical_verification=pending']
 lines += ['HIP_WEIGHTED_QKV TRANSPORT_COMPLETE stages=5 launches=5 weighted_values=2048 projection_values=2560 q8_bytes=2304 raw_files=30 raw_bytes=11972800 weighted_max_abs=0 weighted_max_budget_fraction=0 allocations=6 frees=6 modules_unloaded=3 watchdog_joined=true numerical_verification=pending finalization=pending',*log.LINES[3:6],'HIP_WEIGHTED_QKV FINAL allocations=6 frees=6 queues=1 retired=1 gate_destroyed=true application_fini=pending',*log.LINES[7:]]
 return ('\n'.join(lines)+'\n').encode()

class Controls(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.c=console.Control.__new__(console.Control);self.c.fixture='weighted_qkv';self.c.out=self.root;self.c.record={};self.c.manifest={'results_dir':'/data/ps5-hip/test','weighted_qkv':{'reference':'unused'}}
  self.c.listing=lambda _:['-rw------- 1 x x 0 Oct 1 00:00 '+n for n in sorted(verifier.DUMP_NAMES)]
  self.c.fetch=lambda path,bound,deadline_seconds:bytes(bound)
  self.c.verify=lambda:None
 def test_transport_only_and_strict_transcript(self):
  raw=transcript();result=log.check_log(raw,'weighted_qkv');self.assertEqual(result['application_result'],'transport-complete');self.assertFalse(result['hardware_gate_passed'])
  for changed in [raw[:-1],raw+b'extra\n',raw.replace(b'pending',b'pass',1),raw.replace(b'launches=5',b'launches=4'),raw.replace(b'weighted_max_budget_fraction=0',b'weighted_max_budget_fraction=1.1'),raw.replace(b'stage=q8',b'stage=q',1),raw.replace(b'raw_files=30',b'raw_files=29')]:
   with self.assertRaises(ValueError):log.check_log(changed,'weighted_qkv')
 def test_freshness_and_complete_capture(self):
  with self.assertRaises(RuntimeError):self.c.fresh_weighted_qkv_dumps()
  self.c.collect_weighted_qkv_dumps();self.assertEqual(len(self.c.record['weighted_qkv_raw_files']),30);self.assertEqual(sum(x['bytes'] for x in self.c.record['weighted_qkv_raw_files'].values()),11972800)
 def test_partial_retention_and_global_deadline(self):
  self.c.listing=lambda _:['-rw x weighted-qkv-weighted.input.raw'];self.c.collect_weighted_qkv_dumps(partial=True)
  self.assertEqual(len(self.c.record['weighted_qkv_raw_files']),1)
  self.c.weighted_qkv_capture_deadline=0;self.c.listing=lambda _:['-rw x '+n for n in verifier.DUMP_NAMES]
  with self.assertRaises(RuntimeError):self.c.collect_weighted_qkv_dumps()
  self.assertTrue((self.root/'raw/weighted-qkv-weighted.input.raw').exists())
 def test_missing_inventory_and_short_transfer(self):
  self.c.listing=lambda _:[]
  with self.assertRaises(RuntimeError):self.c.collect_weighted_qkv_dumps()
  self.c.listing=lambda _:['-rw x '+n for n in verifier.DUMP_NAMES];self.c.fetch=lambda *a,**k:b'partial'
  with self.assertRaises(RuntimeError):self.c.collect_weighted_qkv_dumps()
  self.assertTrue(any((self.root/'raw').iterdir()))
 def test_numerical_gate_and_metric_binding(self):
  self.c.record['application']={'reported_metrics':{'max_abs':0.,'max_budget_fraction':0.}}
  result={'status':'PASS','raw_files':30,'projection_values':2560,'weighted_metrics':{'maximum_absolute_error':0.,'maximum_budget_fraction':0.}}
  def verify(*args):
   p=args[-1];p.parent.mkdir(exist_ok=True);p.write_text('{}');return result
  with patch.object(verifier,'verify',side_effect=verify):self.c.verify_weighted_qkv_dumps()
  self.assertEqual(self.c.record['weighted_qkv_numerical']['status'],'PASS')
  result['weighted_metrics']['maximum_absolute_error']=1.
  with patch.object(verifier,'verify',side_effect=verify),self.assertRaises(RuntimeError):self.c.verify_weighted_qkv_dumps()

class Freeze(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.build=self.root/'build';self.output=self.root/'candidate';self.output.mkdir();self.ref=self.root/'reference';self.ref.mkdir();self.weighted=self.root/'weighted';self.q4=self.root/'q4';self.weighted.mkdir();self.q4.mkdir()
  names=['weighted.hsaco','q8.hsaco','q4.hsaco','input.f32le','norm.f32le','weighted.fp64le','weighted.budget.f64le','q.weights.q4_0','k.weights.q4_0','v.weights.q4_0'];assets={};artifacts={}
  for name in names:
   p=(self.weighted if name=='weighted.hsaco' else self.q4 if name.endswith('.hsaco') else self.ref)/name;p.write_bytes(name.encode());assets['weighted-qkv/'+name]=str(p)
   dest=self.build/'package/app/weighted-qkv'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
   if not name.endswith('.hsaco'):artifacts[name]={'bytes':p.stat().st_size,'sha256':freeze.digest(p)}
  r=self.ref/'receipt.json';r.write_text(json.dumps({'artifacts':artifacts}));self.bound={'assets':assets,'reference':{'receipt':str(r),'receipt_sha256':freeze.digest(r)}};self.receipt={'weighted_qkv':self.bound,'package_files':{n:freeze.digest(Path(p)) for n,p in assets.items()}}
  self.patch=patch.object(freeze,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
 def run_freeze(self):
  with patch('public_hip_weighted_qkv_assets.assets',return_value=self.bound):return freeze.freeze_weighted_qkv_reference(self.build,self.receipt,self.output)
 def test_exact_copy_and_three_modules(self):
  r=self.run_freeze();self.assertEqual(set(r['module_sha256']),{'weighted','q8','q4'});self.assertEqual((self.output/'weighted-qkv-reference/receipt.json').read_bytes(),(self.ref/'receipt.json').read_bytes())
 def test_package_inventory_and_copy_bound(self):
  self.receipt['package_files']['weighted-qkv/extra']='0'*64
  with self.assertRaises(ValueError):self.run_freeze()
  del self.receipt['package_files']['weighted-qkv/extra']
  with patch.object(freeze,'WEIGHTED_QKV_REFERENCE_LIMIT',1),self.assertRaises(ValueError):self.run_freeze()
  self.assertFalse((self.output/'weighted-qkv-reference').exists())
 def test_unsafe_copy_path(self):
  p=self.ref/'receipt.json';r=json.loads(p.read_text());r['artifacts']['../escape']={'bytes':1,'sha256':'0'*64};p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):self.run_freeze()
 def test_audit_receipt_identity_required(self):
  with self.assertRaises(ValueError):freeze.validate_weighted_qkv_audit(self.build,{})

if __name__=='__main__':unittest.main()
