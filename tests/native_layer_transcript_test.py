import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_llm_layer_log as log

def fixture():
 lines=list(log.LLM_CLIENT_LINES[:2])
 def memory(phase,largest):lines.append(f'M0_APP MEMORY phase={phase} pool_bytes=12884901888 largest_free_bytes={largest} total_free_bytes=unknown alignment=16384')
 lines.append('M2_LAYER PHASE load_begin');memory('before_load',10737418240)
 lines.append('M2_LAYER PHASE load_complete');memory('after_load',10066329600)
 lines.extend(['M2_LAYER PHASE context_begin','M2_LAYER PHASE context_complete']);memory('after_context',9965666304)
 for run,positions in ((0,(0,1,255,256,511)),(1,(0,1))):
  for pos in positions:
   extent=256 if pos<256 else 512
   sizes=[8192]*5+[1024]*3+[128*extent]*2+[8192]*6+[22528]*3+[8192]*2
   for i,size in enumerate(sizes):lines.append(f'M2_LAYER RAW file=run-{run}-pos-{pos}-out-{i:02d}.f32le bytes={size} exclusive=true synced=true')
   if pos==511 or (run==1 and pos==1):
    for key in ('k','v'):lines.append(f'M2_LAYER RAW file=run-{run}-cache-{key}.f16le bytes=262144 exclusive=true synced=true')
   lines.append(f'M2_LAYER CHECKPOINT run={run} position={pos} token={(0,1,2,31999)[pos%4]} extent={extent} nodes=37')
  count=512 if run==0 else 2
  lines.extend(['LLM_GATE REJECT node=0 op=GET_ROWS selected=CPU',f'M2_LAYER SEQUENCE run={run} requests={count} original_result=-3 rejected=1 forwarded=0 post_compute=0 expected_errors=2 launches={33*count} control_source=declared layer_only=true inference=false'])
 lines.append('M2_LAYER PHASE cleanup_begin')
 for site,count,size in [('module_load',6,0),('module_function',17,0),('scratch_allocate',2,39104),('token_read',3084,1851432),('launch',16962,0),('completion_sync',16962,0),('release_sync',1,0),('scratch_free',2,39104),('module_unload',6,0)]:lines.append(f'M2_LAYER HIP_CALL site={site} attempts={count} completions={count} successes={count} failures=0 bytes={size} last_result=0 first_error=0')
 lines.append('M2_LAYER PHASE cleanup_complete');memory('after_cleanup',10737418240)
 lines.append('M2_LAYER TRANSPORT_COMPLETE requests=514 nodes=37 launches=16962 raw_files=151 raw_bytes=2878464 uploaded_weight_bytes=61652992 uploaded_request_bytes=1849376 cache_resets=1 numerical=pending inference=false finalization=pending')
 lines.append('M0_APP HOST_STORAGE allocations=101 releases=101 live_allocations=0 live_requested_bytes=0 peak_requested_bytes=700000000 failures=0 poisoned=false scope=ggml_aligned')
 lines.extend([log.APP_COMPLETE,*log.LINES[3:6],log.FINAL,*log.LINES[7:]])
 return ('\n'.join(lines)+'\n').encode()

class LayerTranscript(unittest.TestCase):
 def test_complete(self):
  result=log.check_layer_log(fixture());self.assertEqual(result['raw_files'],151);self.assertEqual(result['raw_bytes'],2878464);self.assertEqual(result['final_counts'],[528,528,1,1])
  for field in ('numerical_verified','gpu_origin_established','inference'):self.assertFalse(result[field])
 def test_every_line_required_unique_ordered(self):
  lines=fixture().splitlines(keepends=True)
  for i in range(len(lines)):
   cases=[lines[:i]+lines[i+1:],lines[:i]+[lines[i]]+lines[i:]]
   if i+1<len(lines):cases.append(lines[:i]+[lines[i+1],lines[i]]+lines[i+2:])
   for case in cases:
    with self.subTest(line=i),self.assertRaises(ValueError):log.check_layer_log(b''.join(case))
 def test_counters_reset_and_claims(self):
  for before,after in [(b'attempts=16962',b'attempts=16961'),(b'cache_resets=1',b'cache_resets=0'),(b'run=1 requests=2',b'run=1 requests=1'),(b'position=256 token=0 extent=512',b'position=256 token=0 extent=256'),(b'bytes=39104',b'bytes=2304'),(b'bytes=1851432',b'bytes=1849376'),(b'allocations=528 frees=528',b'allocations=516 frees=516'),(b'forwarded=0',b'forwarded=1'),(b'numerical=pending',b'numerical=pass'),(b'inference=false',b'inference=true'),(b'releases=101',b'releases=100')]:
   self.assertIn(before,fixture())
   with self.subTest(before=before),self.assertRaises(ValueError):log.check_layer_log(fixture().replace(before,after,1))
 def test_bad_framing(self):
  good=fixture()
  for bad in (good[:-1],good+b'warning\n',good.replace(b'\n',b'\r\n'),b'\0'+good,b'x'*65537):
   with self.assertRaises(ValueError):log.check_layer_log(bad)
 def test_raw_cache_nonfinite_and_missing(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory)
   for name,size in log.FILES.items():(folder/name).write_bytes(bytes(size))
   self.assertEqual(len(log.read_raw(folder)),151)
   p=folder/'run-0-cache-k.f16le';p.write_bytes(b'\x00\x7c'+bytes(262142))
   with self.assertRaises(ValueError):log.read_raw(folder)
   p.unlink()
   with self.assertRaises(ValueError):log.read_raw(folder)

if __name__=='__main__':unittest.main()
