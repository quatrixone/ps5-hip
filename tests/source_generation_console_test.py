import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_console as c
from llm_generation_readbacks_test import write_generation_raw

class GenerationControl(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.candidate=self.root/'candidate';self.reference=self.candidate/'full-reference';self.files=write_generation_raw(self.reference/'raw')
  self.obj=c.Control.__new__(c.Control);self.obj.fixture='llm_client';self.obj.candidate=self.candidate;self.obj.out=self.root/'execute';self.obj.out.mkdir();self.obj.record={}
  self.obj.manifest={'llm_client':{'phase':'model-full','full':{'profile':'generation'}},'model_full':{'profile':'generation','reference':'candidate/full-reference','reference_receipt_sha256':'a'*64,'raw_files':self.files},'results_dir':'/user/hip','files':{str(p.relative_to(self.root)):c.sha(p.read_bytes()) for p in (self.reference/'raw').iterdir()}}
  self.obj.listing=lambda _:list(self.files);self.obj.fetch=lambda path,size,deadline:(self.reference/'raw'/path.rsplit('/',1)[-1]).read_bytes();self.patch=patch.object(c,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
 def test_collect_all_boundaries_and_metadata(self):
  self.obj.collect_prefix_dumps();self.assertEqual(set(self.obj.record['prefix_raw_files']),set(self.files));self.assertEqual(len(self.obj.record['prefix_raw_files']),386)
 def test_stale_metadata_alone_blocks_launch(self):
  for name in ('generation.json','steps.jsonl'):
   self.obj.listing=lambda _,name=name:[name]
   with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'already exist'):self.obj.fresh_prefix_dumps()
 def test_missing_metadata_and_extra_float_rejected(self):
  for names in ([n for n in self.files if n!='generation.json'],[*self.files,'run-2-pos-0-logits.f32le']):
   self.obj.listing=lambda _:names
   with self.assertRaisesRegex(RuntimeError,'inventory'):self.obj.collect_prefix_dumps()
 def test_frozen_policy_identity(self):
  policy,sizes=self.obj.generation_policy();self.assertEqual(sizes,{n:self.files[n] for n in ('generation.json','steps.jsonl')});self.assertEqual(len(policy['runs']),2)
  p=self.reference/'raw/generation.json';p.write_bytes(p.read_bytes().replace(b'Once',b'once'))
  with self.assertRaisesRegex(RuntimeError,'identity'):self.obj.generation_policy()
 def test_policy_outside_candidate_rejected(self):
  self.obj.manifest['model_full']['reference']='elsewhere'
  with self.assertRaisesRegex(RuntimeError,'outside candidate'):self.obj.generation_policy()
 def test_generation_checker_routing(self):
  module=types.ModuleType('source_llm_generation_log');module.check_full_log=lambda raw,policy,sizes:{'raw':raw,'policy':policy,'sizes':sizes}
  with patch.dict(sys.modules,{'source_llm_generation_log':module}):r=self.obj.check_application(b'raw')
  self.assertEqual(r['raw'],b'raw');self.assertEqual(r['sizes']['generation.json'],self.files['generation.json'])
 def test_numerical_routes_profile_and_same_captured_bytes(self):
  self.obj.collect_prefix_dumps();module=types.ModuleType('verify_llm_full_readbacks')
  def verify(reference,raw,output,profile):
   self.assertEqual(profile,'generation');self.assertEqual(reference,self.reference);output.write_text('{}');return {'status':'PASS','raw_files':386,'raw_bytes':sum(self.files.values()),'reference_receipt_sha256':'a'*64,'raw_sha256':{n:r['sha256'] for n,r in self.obj.record['prefix_raw_files'].items()}}
  module.verify=verify
  with patch.dict(sys.modules,{'verify_llm_full_readbacks':module}):self.obj.verify_prefix_dumps()
  self.assertEqual(self.obj.record['prefix_numerical']['status'],'PASS')

if __name__=='__main__':unittest.main()
