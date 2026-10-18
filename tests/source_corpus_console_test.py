import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools')]
import source_console as c
from build_llm_generation_reference import corpus_cases

class CorpusControl(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.obj=c.Control.__new__(c.Control);self.obj.fixture='llm_client';self.obj.candidate=self.root/'candidate';self.obj.out=self.root/'execute';self.obj.out.mkdir();self.obj.record={}
  self.raw=self.obj.candidate/'reference/raw';self.raw.mkdir(parents=True)
  contents={'corpus.json':b'{}'}
  for case,_ in corpus_cases():contents.update({case+'/generation.json':b'{"runs":[]}',case+'/steps.jsonl':b'{}\n',case+'/run-0-pos-0-logits.f32le':b'1234'})
  for name,data in contents.items():
   p=self.raw/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data)
  self.files={n:len(v) for n,v in contents.items()}
  self.obj.manifest={'llm_client':{'phase':'model-full'},'model_full':{'profile':'corpus','reference':'candidate/reference','raw_files':self.files},'results_dir':'/user/hip','files':{str(p.relative_to(self.root)):c.sha(p.read_bytes()) for p in self.raw.rglob('*') if p.is_file()}}
  self.obj.listing=lambda remote:[p.name for p in (self.raw/remote.removeprefix('/user/hip').lstrip('/')).iterdir()]
  self.obj.fetch=lambda remote,size,deadline:(self.raw/remote.removeprefix('/user/hip/')).read_bytes()
  self.patch=patch.object(c,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
 def test_nested_capture_and_limits(self):
  self.obj.collect_prefix_dumps()
  self.assertEqual(set(self.obj.record['prefix_raw_files']),set(self.files));self.assertEqual(self.obj.record['prefix_raw_capture_limit_seconds'],600)
  self.assertEqual(self.obj.observation_seconds(),1830);self.assertEqual(self.obj.application_log_limit(),2*1024*1024)
  del self.obj.phase_deadline
  with patch.object(c.time,'monotonic',return_value=10):self.assertEqual(self.obj.phase_remaining(5000),4200)
  self.assertEqual(self.obj.phase_deadline,4210)
 def test_stale_case_blocks_launch(self):
  self.obj.listing=lambda _:['story']
  with self.assertRaisesRegex(RuntimeError,'already exist'):self.obj.fresh_prefix_dumps()
 def test_nested_extra_missing_and_duplicate_rejected(self):
  extra=self.raw/'story/extra';extra.write_bytes(b'x')
  with self.assertRaisesRegex(RuntimeError,'inventory'):self.obj.collect_prefix_dumps()
  extra.unlink();self.obj.collect_prefix_dumps();self.obj.record={}
  with self.assertRaises(FileExistsError):self.obj.collect_prefix_dumps()
 def test_partial_capture_missing_case(self):
  listing=self.obj.listing;self.obj.listing=lambda p:[n for n in listing(p) if n!='story'] if p=='/user/hip' else listing(p)
  self.obj.collect_prefix_dumps(partial=True)
  self.assertFalse(any(n.startswith('story/') for n in self.obj.record['prefix_raw_files']))
 def test_policy_hash_and_application_route(self):
  policies,sizes=self.obj.generation_policy();self.assertEqual(set(policies),{k for k,_ in corpus_cases()});self.assertEqual(len(sizes),11)
  module=types.ModuleType('source_llm_generation_log');module.check_corpus_full_log=lambda raw,p,s:dict(raw=raw,policies=p)
  with patch.dict(sys.modules,{'source_llm_generation_log':module}):self.assertEqual(self.obj.check_application(b'x')['policies'],policies)
  (self.raw/'corpus.json').write_bytes(b'[]')
  with self.assertRaisesRegex(RuntimeError,'identity'):self.obj.generation_policy()
 def test_local_case_symlink_rejected(self):
  folder=self.obj.out/'raw';folder.mkdir();(folder/'story').symlink_to(self.raw/'story',target_is_directory=True)
  with self.assertRaisesRegex(RuntimeError,'local output'):self.obj.collect_prefix_dumps()

if __name__=='__main__':unittest.main()
