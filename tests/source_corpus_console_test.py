import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tools')]
import source_console as c
from build_llm_generation_reference import corpus_cases

def list_row(name,directory=False):return ('drwxrwxrwx' if directory else '-rw-r--r--')+' 2 0 0 65536 Oct 18 02:31 '+name

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
  self.obj.listing=lambda remote:[list_row('.',True),list_row('..',True)]+[list_row(p.name,p.is_dir()) for p in (self.raw/remote.removeprefix('/user/hip').lstrip('/')).iterdir()]
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
  self.obj.listing=lambda _:[list_row('story',True)]
  with self.assertRaisesRegex(RuntimeError,'already exist'):self.obj.fresh_prefix_dumps()
 def test_empty_real_dot_directory_listing_is_fresh(self):
  self.obj.listing=lambda _:[list_row('.',True),list_row('..',True)]
  self.obj.fresh_prefix_dumps()
  self.assertEqual(self.obj.record['prefix_raw_absent_before_launch'],sorted(self.files))
 def test_real_dot_entries_do_not_hide_unexpected_root(self):
  self.obj.listing=lambda _:[list_row('.',True),list_row('..',True),list_row('unexpected')]
  with self.assertRaisesRegex(RuntimeError,'unexpected'):self.obj.fresh_prefix_dumps()
  with self.assertRaisesRegex(RuntimeError,'root inventory'):self.obj.collect_prefix_dumps()
 def test_malformed_symlink_and_unsafe_entries_rejected(self):
  rows=['story','',list_row('.'),list_row('..'),list_row('../escape'),list_row('a/b'),list_row('a\\b'),list_row('bad\tname'),
        'lrwxrwxrwx 1 0 0 8 Oct 18 02:31 . -> story',list_row('story').replace('-rw-r--r--','not-a-mode')]
  for row in rows:
   self.obj.listing=lambda _,row=row:[row]
   with self.subTest(row=row),self.assertRaisesRegex(RuntimeError,'LIST|dot entry'):self.obj.fresh_prefix_dumps()
 def test_child_unsafe_entry_rejected_before_fetch(self):
  listing=self.obj.listing
  for name in ('../escape','a/b','a\\b'):
   self.obj.listing=lambda path,name=name:listing(path)+([list_row(name)] if path.endswith('/story') else [])
   with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'unsafe LIST'):self.obj.collect_prefix_dumps()
 def test_full_filename_not_aliased_to_valid_suffix(self):
  self.obj.listing=lambda _:[list_row('unexpected '+c.LOG_NAMES['llm_client'])]
  with self.assertRaisesRegex(RuntimeError,'unexpected'):self.obj.fresh_prefix_dumps()
  self.assertEqual(self.obj.prefix_listing_names('/user/hip'),{'unexpected '+c.LOG_NAMES['llm_client']})
 def test_duplicate_names_rejected(self):
  self.obj.listing=lambda _:[list_row('story',True),list_row('story',True)]
  with self.assertRaisesRegex(RuntimeError,'duplicate LIST'):self.obj.collect_prefix_dumps()
 def test_nested_extra_missing_and_duplicate_rejected(self):
  extra=self.raw/'story/extra';extra.write_bytes(b'x')
  with self.assertRaisesRegex(RuntimeError,'inventory'):self.obj.collect_prefix_dumps()
  extra.unlink();self.obj.collect_prefix_dumps();self.obj.record={}
  with self.assertRaises(FileExistsError):self.obj.collect_prefix_dumps()
 def test_partial_capture_missing_case(self):
  listing=self.obj.listing;self.obj.listing=lambda p:[n for n in listing(p) if n.split(None,8)[8]!='story'] if p=='/user/hip' else listing(p)
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
