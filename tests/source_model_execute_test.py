from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import unittest,tempfile,json,threading,subprocess
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
 def test_memory_snapshots_preserve_measurements(self):
  raw=EXPECTED.replace(b'largest_free_bytes=10737418240',b'largest_free_bytes=0',1)
  memory=check_model_log(raw)['memory']
  self.assertEqual([m['phase'] for m in memory],['before_load','after_load','after_context','after_cleanup'])
  self.assertEqual(memory[0]['largest_free_bytes'],0)
  self.assertIsNone(memory[0]['total_free_bytes'])
  for old,new in [(b'largest_free_bytes=10737418240',b'largest_free_bytes=12884901889'),(b'pool_bytes=12884901888',b'pool_bytes=0'),(b'pool_bytes=12884901888',b'pool_bytes=9223372036854775808'),(b'pool_bytes=12884901888',b'pool_bytes=12884901889'),(b'total_free_bytes=unknown',b'total_free_bytes=10737418240'),(b'alignment=16384',b'alignment=65536'),(b'phase=before_load',b'phase=after_load')]:
   with self.assertRaises(ValueError):check_model_log(EXPECTED.replace(old,new,1))
  line=next(line for line in EXPECTED.splitlines(keepends=True) if line.startswith(b'M0_APP MEMORY'))
  for raw in (EXPECTED.replace(line,b''),EXPECTED.replace(line,line+line),EXPECTED.replace(line,b'')+line):
   with self.assertRaises(ValueError):check_model_log(raw)
 def test_native_memory_snapshot_checks_kernel_results(self):
  source=(Path(__file__).resolve().parents[1]/'tools/public_hip_llm_client_console_title.cpp').read_text()
  function=source[source.index('__attribute__((noinline)) static void model_memory_snapshot'):source.index('enum class ClientPhase')]
  harness='''#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
int64_t pool=12884901888LL,start=0;size_t largest=10737418240ULL;int rc=0;
int64_t sceKernelGetDirectMemorySize(){return pool;}
int sceKernelAvailableDirectMemorySize(int64_t lo,int64_t hi,size_t align,int64_t* out,size_t* bytes){assert(lo==0&&hi==pool&&align==16384);*out=start;*bytes=largest;return rc;}
[[noreturn]] void ps5hip_clr_gpu_fail(const char* reason){assert(!std::strcmp(reason,"memory-query"));throw 1;}
'''+function+'''
int main(){
 model_memory_snapshot("before_load");
 largest=0;model_memory_snapshot("after_cleanup");
 for(int i=0;i<6;++i){pool=12884901888LL;start=0;largest=1;rc=0;
  if(i==0)pool=0;if(i==1)pool=-1;if(i==2)rc=-1;if(i==3)start=-1;if(i==4)start=pool+1;if(i==5){start=pool;largest=1;}
  bool failed=false;try{model_memory_snapshot("before_load");}catch(int){failed=true;}assert(failed);
 }
}
'''
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'test.cpp').write_text(harness)
   subprocess.run(['c++','-std=c++17',str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
   result=subprocess.run([str(p/'test')],check=True,capture_output=True,text=True)
   self.assertIn('pool_bytes=12884901888 largest_free_bytes=10737418240 total_free_bytes=unknown',result.stdout)
   self.assertEqual(result.stdout.count('MEMORY_QUERY_FAILED'),6)
 def test_host_storage_requires_balanced_owned_memory(self):
  raw=EXPECTED.replace(b'HOST_STORAGE allocations=1 releases=1',b'HOST_STORAGE allocations=7 releases=7').replace(b'peak_requested_bytes=619094016',b'peak_requested_bytes=700000000')
  self.assertEqual(check_model_log(raw)['host_storage']['allocations'],7)
  with self.assertRaises(ValueError):
   check_model_log(EXPECTED.replace(b'scope=ggml_aligned',b'external_ggml_only=true'))
  for old,new in [(b'HOST_STORAGE allocations=1',b'HOST_STORAGE allocations=0'),(b'releases=1',b'releases=2'),(b'live_allocations=0',b'live_allocations=1'),(b'live_requested_bytes=0',b'live_requested_bytes=1'),(b'peak_requested_bytes=619094016',b'peak_requested_bytes=619094015'),(b'peak_requested_bytes=619094016',b'peak_requested_bytes=18446744073709551616'),(b'failures=0',b'failures=1'),(b'poisoned=false',b'poisoned=true')]:
   with self.assertRaises(ValueError):check_model_log(EXPECTED.replace(old,new,1))
  line=next(line for line in EXPECTED.splitlines(keepends=True) if line.startswith(b'M0_APP HOST_STORAGE'))
  for raw in (EXPECTED.replace(line,b''),EXPECTED.replace(line,line+line)):
   with self.assertRaises(ValueError):check_model_log(raw)
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
