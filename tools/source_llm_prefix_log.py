"""Strict complete native-prefix transcript; numerical and origin checks are separate."""
import re, math, struct
from pathlib import Path
from source_public_hip_log import LLM_CLIENT_LINES, LINES
TOKENS=(0,1,2,31999)
OUTPUTS={'embedding':8192,'normalized':8192,'weighted':8192,'q':8192,'k':1024,'v':1024}
FILES={f'token-{token}-{name}.f32le':size for token in TOKENS for name,size in OUTPUTS.items()}
SITES=('module_load','module_function','scratch_allocate','token_read','launch','completion_sync','release_sync','scratch_free','module_unload')
COUNTS=(5,5,1,8,36,36,1,1,5)
def check_fragment(raw):
 if not isinstance(raw,bytes) or not raw.endswith(b'\n') or b'\r' in raw or b'\0' in raw:raise ValueError('invalid raw fragment encoding')
 lines=iter(raw.decode('ascii').splitlines());memory=[]
 def exact(expected):
  if next(lines,None)!=expected:raise ValueError('missing/unexpected transcript line; expected '+expected)
 def snapshot(phase):
  line=next(lines,'')
  match=re.fullmatch(r'M0_APP MEMORY phase='+phase+r' pool_bytes=([1-9][0-9]*) largest_free_bytes=([0-9]+) total_free_bytes=unknown alignment=16384',line)
  if not match:raise ValueError('memory snapshot')
  pool,largest=map(int,match.groups())
  if not 0<largest<=pool or pool<619094016 or largest%16384:raise ValueError('memory extent')
  memory.append((pool,largest))
 exact('M2_PREFIX PHASE load_begin');snapshot('before_load')
 exact('M2_PREFIX PHASE load_complete');snapshot('after_load')
 exact('M2_PREFIX PHASE context_begin');exact('M2_PREFIX PHASE context_complete');snapshot('after_context')
 for attempt,token in enumerate(TOKENS):
  for name,size in OUTPUTS.items():
   filename=f'token-{token}-{name}.f32le'
   exact(f'M2_PREFIX RAW token={token} tensor={name} file={filename} bytes={size} exclusive=true synced=true')
  exact('LLM_GATE REJECT node=0 op=GET_ROWS selected=CPU')
  reuse='false' if attempt==0 else 'true'
  exact(f'M2_PREFIX ACTUAL_CLIENT attempt={attempt} token={token} position=0 result=-3 calls=1 rejected=1 forwarded=0 post_compute=0 expected_errors=2 graph_reused={reuse} prefix_nodes=6 launches=9 uploaded_leaf_bytes=39821316 inference=false')
 exact('M2_PREFIX PHASE cleanup_begin')
 for index,(site,count) in enumerate(zip(SITES,COUNTS)):
  size=2304 if index in (2,7) else 32 if index==3 else 0
  exact(f'M2_PREFIX HIP_CALL site={site} attempts={count} completions={count} successes={count} failures=0 bytes={size} last_result=0 first_error=0')
 exact('M2_PREFIX PHASE cleanup_complete');snapshot('after_cleanup')
 exact('M2_PREFIX TRANSPORT_COMPLETE tokens=4 prefix_nodes=6 launches=36 raw_files=24 raw_bytes=139264 uploaded_leaf_bytes=159285264 modules_unloaded=5 scratch_allocations=1 scratch_frees=1 numerical_verification=pending inference=false finalization=pending')
 if next(lines,None) is not None:raise ValueError('trailing transcript line')
 if len({pool for pool,_ in memory})!=1 or memory[0][1]!=memory[-1][1] or memory[1][1]>=memory[0][1] or memory[2][1]>memory[1][1]:raise ValueError('memory lifecycle did not balance')
 return {'classification':'prefix-fragment-transport-only','tokens':list(TOKENS),'files':24,'bytes':139264,'gpu_origin_established':False,'numerical_verified':False,'finalization_verified':False}

# One scratch allocation plus one combined ggml buffer for each of four tokens.
# Default-stream transfers and launches create one CLR Virtual queue; the app
# finalizer observes it retired before the common shell-exit probe executes.
FINAL_COUNTS=(5,5,1,1)
APP_COMPLETE='M2_PREFIX APP_COMPLETE watchdog_joined=true numerical=pending finalization=pending inference=false'
FINAL='M0_APP FINAL allocations=5 frees=5 queues=1 retired=1 gate_destroyed=true application_fini=pending'

def check_prefix_log(raw):
 if not isinstance(raw,bytes) or not raw or len(raw)>65536:raise ValueError('prefix transcript type/size')
 try:lines=raw.decode('ascii').splitlines()
 except UnicodeDecodeError as error:raise ValueError('prefix transcript encoding') from error
 if raw!=('\n'.join(lines)+'\n').encode('ascii') or b'\0' in raw:raise ValueError('prefix transcript framing')
 if len(lines)<12 or tuple(lines[:2])!=LLM_CLIENT_LINES[:2]:raise ValueError('prefix transcript startup')
 footer=lines[-9:]
 pattern=r'M0_APP HOST_STORAGE allocations=([1-9][0-9]{0,19}) releases=([1-9][0-9]{0,19}) live_allocations=0 live_requested_bytes=0 peak_requested_bytes=([1-9][0-9]{0,19}) failures=0 poisoned=false scope=ggml_aligned'
 match=re.fullmatch(pattern,footer[0])
 if not match:raise ValueError('prefix host storage transcript')
 allocations,releases,peak=map(int,match.groups())
 if allocations!=releases or max(allocations,releases,peak)>2**64-1 or peak<619094016:raise ValueError('prefix host ownership balance')
 if footer[1:]!=[APP_COMPLETE,*LINES[3:6],FINAL,*LINES[7:]]:raise ValueError('prefix application shutdown/finalizer transcript')
 fragment=check_fragment(('\n'.join(lines[2:-9])+'\n').encode('ascii'))
 memory=[]
 for line in lines[2:-9]:
  if not line.startswith('M0_APP MEMORY '):continue
  match=re.fullmatch(r'M0_APP MEMORY phase=(before_load|after_load|after_context|after_cleanup) pool_bytes=([1-9][0-9]{0,18}) largest_free_bytes=([1-9][0-9]{0,18}) total_free_bytes=unknown alignment=16384',line)
  if not match:raise ValueError('prefix memory encoding/bounds')
  phase,pool,largest=match.groups();pool,largest=int(pool),int(largest)
  if pool>2**63-1 or largest>pool:raise ValueError('prefix memory bounds')
  memory.append(dict(phase=phase,pool_bytes=pool,largest_free_bytes=largest,total_free_bytes=None,alignment=16384))
 return dict(status='PASS',phase='model-prefix',classification='native-prefix-transcript-only',actual_calls=4,
  tokens=list(TOKENS),prefix_nodes=6,gpu_submits=36,forwarded=0,post_compute=0,expected_errors=8,
  graph_reused_after_first=True,raw_files=24,raw_bytes=139264,uploaded_leaf_bytes=159285264,
  final_counts=list(FINAL_COUNTS),memory=memory,
  host_storage=dict(allocations=allocations,releases=releases,peak_requested_bytes=peak,live_requested_bytes=0,failures=0,poisoned=False),
  numerical_verified=False,gpu_origin_established=False,inference=False,finalization_transcript_complete=True)

def read_raw(folder):
 folder=Path(folder)
 if {p.name for p in folder.iterdir()}!=set(FILES):raise ValueError('raw file inventory')
 result={}
 for name,size in FILES.items():
  path=folder/name
  if path.is_symlink() or not path.is_file() or path.stat().st_size!=size:raise ValueError('raw file kind/size')
  data=path.read_bytes()
  if len(data)!=size or not all(math.isfinite(v) for v in struct.unpack('<'+'f'*(size//4),data)):raise ValueError('raw file nonfinite/changed')
  result[name]=data
 return result
