"""Strict one-layer transcript; numerical comparison and GPU origin are separate."""
import re
from source_llm_layer_raw import CHECKPOINTS,FILES,RAW_BYTES,output_sizes,read_raw
from source_public_hip_log import LLM_CLIENT_LINES,LINES
COUNTS=(6,17,2,3084,16962,16962,1,2,6)
SITES=('module_load','module_function','scratch_allocate','token_read','launch','completion_sync','release_sync','scratch_free','module_unload')
FINAL_COUNTS=(528,528,1,1)
APP_COMPLETE='M2_LAYER APP_COMPLETE watchdog_joined=true numerical=pending finalization=pending inference=false'
FINAL='M0_APP FINAL allocations=528 frees=528 queues=1 retired=1 gate_destroyed=true application_fini=pending'
def check_fragment(raw):
 if not isinstance(raw,bytes) or not raw.endswith(b'\n') or b'\r' in raw or b'\0' in raw:raise ValueError('layer fragment encoding')
 lines=iter(raw.decode('ascii').splitlines());memory=[]
 def exact(expected):
  if next(lines,None)!=expected:raise ValueError('layer transcript expected '+expected)
 def snapshot(phase):
  m=re.fullmatch(r'M0_APP MEMORY phase='+phase+r' pool_bytes=([1-9][0-9]{0,18}) largest_free_bytes=([1-9][0-9]{0,18}) total_free_bytes=unknown alignment=16384',next(lines,''))
  if not m:raise ValueError('layer memory snapshot')
  pool,largest=map(int,m.groups())
  if not 0<largest<=pool<2**63 or pool<619094016 or largest%16384:raise ValueError('layer memory bounds')
  memory.append(dict(phase=phase,pool_bytes=pool,largest_free_bytes=largest,total_free_bytes=None,alignment=16384))
 exact('M2_LAYER PHASE load_begin');snapshot('before_load');exact('M2_LAYER PHASE load_complete');snapshot('after_load')
 exact('M2_LAYER PHASE context_begin');exact('M2_LAYER PHASE context_complete');snapshot('after_context')
 for run,positions in CHECKPOINTS:
  for pos in positions:
   for i,size in enumerate(output_sizes(pos)):
    name=f'run-{run}-pos-{pos}-out-{i:02d}.f32le';exact(f'M2_LAYER RAW file={name} bytes={size} exclusive=true synced=true')
   if pos==511 or (run==1 and pos==1):
    for key in ('k','v'):exact(f'M2_LAYER RAW file=run-{run}-cache-{key}.f16le bytes=262144 exclusive=true synced=true')
   exact(f'M2_LAYER CHECKPOINT run={run} position={pos} token={(0,1,2,31999)[pos%4]} extent={256 if pos<256 else 512} nodes=37')
  exact('LLM_GATE REJECT node=0 op=GET_ROWS selected=CPU');count=512 if run==0 else 2
  exact(f'M2_LAYER SEQUENCE run={run} requests={count} original_result=-3 rejected=1 forwarded=0 post_compute=0 expected_errors=2 launches={count*33} control_source=declared layer_only=true inference=false')
 exact('M2_LAYER PHASE cleanup_begin')
 for i,(site,count) in enumerate(zip(SITES,COUNTS)):
  size=39104 if i in (2,7) else 1851432 if i==3 else 0
  exact(f'M2_LAYER HIP_CALL site={site} attempts={count} completions={count} successes={count} failures=0 bytes={size} last_result=0 first_error=0')
 exact('M2_LAYER PHASE cleanup_complete');snapshot('after_cleanup')
 exact(f'M2_LAYER TRANSPORT_COMPLETE requests=514 nodes=37 launches=16962 raw_files=151 raw_bytes={RAW_BYTES} uploaded_weight_bytes=61652992 uploaded_request_bytes=1849376 cache_resets=1 numerical=pending inference=false finalization=pending')
 if next(lines,None) is not None:raise ValueError('layer trailing transcript')
 if len({m['pool_bytes'] for m in memory})!=1 or memory[0]['largest_free_bytes']!=memory[3]['largest_free_bytes'] or memory[1]['largest_free_bytes']>=memory[0]['largest_free_bytes'] or memory[2]['largest_free_bytes']>memory[1]['largest_free_bytes']:raise ValueError('layer memory lifecycle')
 return dict(status='PASS',classification='layer-fragment-transport-only',requests=514,nodes=37,gpu_submits=16962,raw_files=151,raw_bytes=RAW_BYTES,memory=memory,uploaded_weight_bytes=61652992,uploaded_request_bytes=1849376,numerical_verified=False,gpu_origin_established=False,inference=False)

def check_layer_log(raw):
 if not isinstance(raw,bytes) or not raw or len(raw)>65536:raise ValueError('layer transcript bounds')
 lines=raw.decode('ascii').splitlines()
 if raw!=('\n'.join(lines)+'\n').encode() or b'\0' in raw or tuple(lines[:2])!=LLM_CLIENT_LINES[:2]:raise ValueError('layer transcript framing/startup')
 footer=lines[-9:]
 m=re.fullmatch(r'M0_APP HOST_STORAGE allocations=([1-9][0-9]{0,19}) releases=([1-9][0-9]{0,19}) live_allocations=0 live_requested_bytes=0 peak_requested_bytes=([1-9][0-9]{0,19}) failures=0 poisoned=false scope=ggml_aligned',footer[0] if footer else '')
 if not m:raise ValueError('layer host ownership')
 allocations,releases,peak=map(int,m.groups())
 if allocations!=releases or max(allocations,releases,peak)>2**64-1 or peak<619094016:raise ValueError('layer host ownership balance')
 if footer[1:]!=[APP_COMPLETE,*LINES[3:6],FINAL,*LINES[7:]]:raise ValueError('layer finalization')
 result=check_fragment(('\n'.join(lines[2:-9])+'\n').encode())
 result.update(phase='model-layer',classification='native-layer-transcript-only',final_counts=list(FINAL_COUNTS),host_storage=dict(allocations=allocations,releases=releases,peak_requested_bytes=peak),finalization_transcript_complete=True)
 return result

