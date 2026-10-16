from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import unittest,tempfile,json,threading
from pathlib import Path
from unittest.mock import patch
import source_console as c
from source_llm_model_log import EXPECTED,check_model_log
class Worker:
 def __init__(self,*a,**k):self.live=False
 def start(self):self.live=True
 def is_alive(self):return self.live
 def join(self,**k):self.live=False
class Sock:
 def close(self):pass
class Tests(unittest.TestCase):
 def test_exact_and_mutations(self):
  self.assertFalse(check_model_log(EXPECTED)['inference'])
  for old,new in [(b'graph_reused=true',b'graph_reused=false'),(b'expected_errors=2',b'expected_errors=1'),(b'allocations=0',b'allocations=1'),(b'load_complete',b'load_failed'),(b'forwarded=0',b'forwarded=1')]:
   with self.assertRaises(ValueError):check_model_log(EXPECTED.replace(old,new,1))
  for raw in [EXPECTED+b'M0_APP LLAMA_ERROR extra\n',EXPECTED[:-1],EXPECTED.replace(b'LLM_GATE PHASE cleanup_complete\n',b'')]:
   with self.assertRaises(ValueError):check_model_log(raw)
 def execute(self,raw):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'deploy').mkdir();(p/'deploy/receipt.json').write_text(json.dumps(dict(accepted=True,healthy_after=True,lock_released=True,phase='deploy',manifest_sha256='m',package_sha256='p')))
   o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'}};o.candidate=o.out=p;o.manifest_sha='m';o.record={'package_sha256':'p','launched':False};o.stop=threading.Event();o.klog=bytearray();o.capture_errors=[];o.log='/log'
   for name in ('idle','installed','fresh_log','fresh_rmsnorm_dumps','fresh_q4_dumps','fresh_weighted_qkv_dumps','collect_rmsnorm_dumps','collect_q4_dumps','collect_weighted_qkv_dumps','verify_rmsnorm_dumps','verify_q4_dumps','verify_weighted_qkv_dumps'):setattr(o,name,lambda:None)
   o.history_read=lambda name:b'';o.send=lambda name:b'';o.fetch=lambda *a:raw
   with patch.object(c.socket,'create_connection',return_value=Sock()),patch.object(c.threading,'Thread',Worker),patch.object(c,'snapshot',return_value=[]),patch.object(c,'lifecycle',return_value=(True,[(c.TITLE,'1')])):o.execute()
   return o.record
 def test_actual_execute(self):
  r=self.execute(EXPECTED);self.assertEqual(r['observation_seconds'],150);self.assertTrue(r['accepted']);self.assertTrue(r['natural_stop']);self.assertEqual(r['application']['actual_calls'],2)
 def test_actual_execute_rejects_transcript(self):
  with self.assertRaisesRegex(ValueError,'transcript'):self.execute(EXPECTED.replace(b'graph_reused=true',b'graph_reused=false'))
 def test_vocab_deadline_unchanged(self):
  o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'vocab'}};self.assertEqual(o.observation_seconds(),40)
if __name__=='__main__':unittest.main()
