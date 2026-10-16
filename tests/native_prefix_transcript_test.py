import sys,unittest,tempfile
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dependencies.json').is_file())
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'tools')]
import source_llm_prefix_log as log

def fixture():
 lines=list(log.LLM_CLIENT_LINES[:2])
 def memory(phase,largest):lines.append(f'M0_APP MEMORY phase={phase} pool_bytes=12884901888 largest_free_bytes={largest} total_free_bytes=unknown alignment=16384')
 lines.append('M2_PREFIX PHASE load_begin');memory('before_load',10737418240)
 lines.append('M2_PREFIX PHASE load_complete');memory('after_load',10066329600)
 lines.extend(['M2_PREFIX PHASE context_begin','M2_PREFIX PHASE context_complete']);memory('after_context',9965666304)
 for attempt,token in enumerate((0,1,2,31999)):
  for name,size in [('embedding',8192),('normalized',8192),('weighted',8192),('q',8192),('k',1024),('v',1024)]:
   lines.append(f'M2_PREFIX RAW token={token} tensor={name} file=token-{token}-{name}.f32le bytes={size} exclusive=true synced=true')
  lines.append('LLM_GATE REJECT node=0 op=GET_ROWS selected=CPU')
  reuse='false' if attempt==0 else 'true'
  lines.append(f'M2_PREFIX ACTUAL_CLIENT attempt={attempt} token={token} position=0 result=-3 calls=1 rejected=1 forwarded=0 post_compute=0 expected_errors=2 graph_reused={reuse} prefix_nodes=6 launches=9 uploaded_leaf_bytes=39821316 inference=false')
 lines.append('M2_PREFIX PHASE cleanup_begin')
 for site,count,size in [('module_load',5,0),('module_function',5,0),('scratch_allocate',1,2304),('token_read',8,32),('launch',36,0),('completion_sync',36,0),('release_sync',1,0),('scratch_free',1,2304),('module_unload',5,0)]:
  lines.append(f'M2_PREFIX HIP_CALL site={site} attempts={count} completions={count} successes={count} failures=0 bytes={size} last_result=0 first_error=0')
 lines.append('M2_PREFIX PHASE cleanup_complete');memory('after_cleanup',10737418240)
 lines.append('M2_PREFIX TRANSPORT_COMPLETE tokens=4 prefix_nodes=6 launches=36 raw_files=24 raw_bytes=139264 uploaded_leaf_bytes=159285264 modules_unloaded=5 scratch_allocations=1 scratch_frees=1 numerical_verification=pending inference=false finalization=pending')
 lines.append('M0_APP HOST_STORAGE allocations=101 releases=101 live_allocations=0 live_requested_bytes=0 peak_requested_bytes=700000000 failures=0 poisoned=false scope=ggml_aligned')
 lines.extend([log.APP_COMPLETE,*log.LINES[3:6],log.FINAL,*log.LINES[7:]])
 return ('\n'.join(lines)+'\n').encode('ascii')

class PrefixTranscript(unittest.TestCase):
 def test_raw_inventory_size_finite_and_symlink(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory)
   for name,size in log.FILES.items():(folder/name).write_bytes(bytes(size))
   self.assertEqual(len(log.read_raw(folder)),24)
   first=folder/next(iter(log.FILES));data=first.read_bytes()
   for bad in (data[:-1],b'\x00\x00\xc0\x7f'+data[4:]):
    first.write_bytes(bad)
    with self.assertRaises(ValueError):log.read_raw(folder)
   first.write_bytes(data);extra=folder/'extra';extra.write_bytes(b'')
   with self.assertRaises(ValueError):log.read_raw(folder)
   extra.unlink();first.unlink()
   with self.assertRaises(ValueError):log.read_raw(folder)
   target=folder.parent/(folder.name+'-target');target.write_bytes(data)
   try:
    first.symlink_to(target)
    with self.assertRaises(ValueError):log.read_raw(folder)
   finally:target.unlink()
 def test_complete(self):
  data=log.check_prefix_log(fixture());self.assertEqual(data['final_counts'],[5,5,1,1])
  self.assertEqual(data['tokens'],[0,1,2,31999]);self.assertEqual(data['raw_files'],24)
  self.assertEqual(data['host_storage']['allocations'],101)
  self.assertTrue(data['finalization_transcript_complete'])
  for name in ('numerical_verified','gpu_origin_established','inference'):self.assertFalse(data[name])
 def test_every_line_required_and_ordered(self):
  lines=fixture().splitlines(keepends=True)
  for index in range(len(lines)):
   cases=[b''.join(lines[:index]+lines[index+1:]),b''.join(lines[:index]+[lines[index],lines[index]]+lines[index+1:])]
   if index+1<len(lines):cases.append(b''.join(lines[:index]+[lines[index+1],lines[index]]+lines[index+2:]))
   for raw in cases:
    with self.subTest(index=index),self.assertRaises(ValueError):log.check_prefix_log(raw)
 def test_extra_and_encoding(self):
  valid=fixture()
  for raw in [None,'text',b'',valid[:-1],valid+b'warning\n',b'warning\n'+valid,valid.replace(b'\n',b'\r\n'),valid.replace(b'\n',b'\v'),valid+b'\x00\n',valid+b'\xff\n',b'x'*65537]:
   with self.subTest(raw=repr(raw)[:40]),self.assertRaises(ValueError):log.check_prefix_log(raw)
 def test_memory_and_ownership(self):
  cases=[(b'releases=101',b'releases=100'),(b'live_allocations=0',b'live_allocations=1'),(b'peak_requested_bytes=700000000',b'peak_requested_bytes=1'),(b'allocations=101',b'allocations=18446744073709551616'),(b'pool_bytes=12884901888',b'pool_bytes=9223372036854775808'),(b'phase=after_cleanup pool_bytes=12884901888 largest_free_bytes=10737418240',b'phase=after_cleanup pool_bytes=12884901888 largest_free_bytes=10737401856'),(b'largest_free_bytes=10737418240',b'largest_free_bytes=10737418241'),(b'allocations=5 frees=5',b'allocations=6 frees=6'),(b'queues=1 retired=1',b'queues=2 retired=2'),(b'gate_destroyed=true',b'gate_destroyed=false')]
  for before,after in cases:
   with self.subTest(before=before),self.assertRaises(ValueError):log.check_prefix_log(fixture().replace(before,after,1))
 def test_transport_claim_mutations(self):
  for before,after in [(b'token=31999',b'token=32000'),(b'graph_reused=true',b'graph_reused=false'),(b'forwarded=0',b'forwarded=1'),(b'post_compute=0',b'post_compute=1'),(b'launches=9',b'launches=8'),(b'failures=0',b'failures=1'),(b'exclusive=true',b'exclusive=false'),(b'numerical=pending',b'numerical=pass'),(b'inference=false',b'inference=true'),(b'watchdog_joined=true',b'watchdog_joined=false'),(b'application-fini=completed',b'application-fini=pending')]:
   with self.subTest(before=before),self.assertRaises(ValueError):log.check_prefix_log(fixture().replace(before,after,1))
if __name__=='__main__':unittest.main()
