import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import source_console as c
from source_llm_layer_log import FILES
def list_rows(names):return ['-rw-r--r-- 1 root wheel 1 Oct 18 02:31 '+name for name in names]

class LayerControl(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.obj=c.Control.__new__(c.Control);self.obj.fixture='llm_client';self.obj.manifest={'llm_client':{'phase':'model-layer'},'results_dir':'/user/hip','model_layer':{'reference':'reference','reference_receipt_sha256':'a'*64}}
  self.obj.record={};self.obj.out=Path(self.tmp.name);self.obj.listing=lambda _:list_rows(FILES);self.obj.fetch=lambda p,b,d:bytes(b)
  fake=types.ModuleType('source_llm_layer_log');fake.FILES=FILES;fake.check_layer_log=lambda raw:{'prefix':raw}
  self.mods=patch.dict(sys.modules,{'source_llm_layer_log':fake});self.mods.start();self.addCleanup(self.mods.stop)
 def test_phase_routes(self):
  self.assertTrue(self.obj.model_phase());self.assertTrue(self.obj.prefix_phase());self.assertEqual(self.obj.check_application(b'test'),{'prefix':b'test'})
  self.obj.manifest['llm_client']['phase']='model-gate';self.assertTrue(self.obj.model_phase());self.assertFalse(self.obj.prefix_phase())
  self.obj.manifest['llm_client']['phase']='vocab';self.assertFalse(self.obj.model_phase())
 def test_fresh_requires_absence(self):
  with self.assertRaisesRegex(RuntimeError,'already exist'):self.obj.fresh_prefix_dumps()
  self.obj.listing=lambda _:[];self.obj.fresh_prefix_dumps();self.assertEqual(len(self.obj.record['prefix_raw_absent_before_launch']),151)
 def test_collect_exact_bytes(self):
  self.obj.collect_prefix_dumps();self.assertEqual(len(self.obj.record['prefix_raw_files']),151);self.assertEqual(sum(x['bytes'] for x in self.obj.record['prefix_raw_files'].values()),2878464)
 def test_extra_inventory(self):
  self.obj.listing=lambda _:list_rows(list(FILES)+['run-2-pos-0-out-00.f32le'])
  with self.assertRaisesRegex(RuntimeError,'inventory'):self.obj.collect_prefix_dumps()
 def test_short_rejected(self):
  self.obj.fetch=lambda p,b,d:bytes(b-1)
  with self.assertRaisesRegex(RuntimeError,'byte count'):self.obj.collect_prefix_dumps()
 def test_partial_preserves_available(self):
  self.obj.listing=lambda _:list_rows(list(FILES)[:2]);self.obj.collect_prefix_dumps(partial=True);self.assertEqual(len(self.obj.record['prefix_raw_files']),2)
 def test_capture_deadline(self):
  self.obj.prefix_capture_deadline=0
  with self.assertRaisesRegex(RuntimeError,'deadline'):self.obj.collect_prefix_dumps()
 def test_exclusive_outputs(self):
  folder=self.obj.out/'raw';folder.mkdir();(folder/next(iter(FILES))).write_bytes(b'preserve')
  with self.assertRaises(FileExistsError):self.obj.collect_prefix_dumps()
  self.assertEqual((folder/next(iter(FILES))).read_bytes(),b'preserve')
 def test_numerical_requires_same_captured_bytes(self):
  self.obj.collect_prefix_dumps();v=types.ModuleType('verify_llm_layer_readbacks')
  def verify(reference,raw,output):
   output.write_text('{}');return {'status':'PASS','raw_files':151,'raw_bytes':2878464,'reference_receipt_sha256':'a'*64,'raw_sha256':{n:r['sha256'] for n,r in self.obj.record['prefix_raw_files'].items()}}
  v.verify=verify
  with patch.dict(sys.modules,{'verify_llm_layer_readbacks':v}):self.obj.verify_prefix_dumps()
  self.assertEqual(self.obj.record['prefix_numerical']['status'],'PASS')
  def bad(*args):return {'status':'PASS','raw_files':151,'raw_bytes':2878464,'reference_receipt_sha256':'a'*64,'raw_sha256':{}}
  v.verify=bad
  with patch.dict(sys.modules,{'verify_llm_layer_readbacks':v}),self.assertRaisesRegex(RuntimeError,'captured bytes'):self.obj.verify_prefix_dumps()
 def test_real_ftp_list_rows(self):
  self.obj.listing=lambda _:['-rw-r--r-- 1 root wheel %d Oct 16 12:00 %s' % (size,name) for name,size in FILES.items()]
  with self.assertRaisesRegex(RuntimeError,'already exist'):self.obj.fresh_prefix_dumps()
  self.obj.collect_prefix_dumps();self.assertEqual(set(self.obj.record['prefix_raw_files']),set(FILES))
 def test_real_ftp_partial_and_unexpected_rows(self):
  names=list(FILES);self.obj.listing=lambda _:['-rw-r--r-- 1 root wheel 8192 Oct 16 12:00 '+names[0]]
  self.obj.collect_prefix_dumps(partial=True);self.assertEqual(set(self.obj.record['prefix_raw_files']),{names[0]})
  self.obj.listing=lambda _:['-rw-r--r-- 1 root wheel 8192 Oct 16 12:00 run-2-pos-0-out-00.f32le']
  with self.assertRaisesRegex(RuntimeError,'inventory'):self.obj.collect_prefix_dumps()
if __name__=='__main__':unittest.main()
