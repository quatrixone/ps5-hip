"""Exact staged model-rejection transcript; no inference success claim."""
from source_public_hip_log import LLM_CLIENT_LINES
import re
MARKERS = ('LLM_GATE PHASE load_begin', 'LLM_GATE PHASE load_complete', 'LLM_GATE PHASE context_begin', 'LLM_GATE PHASE context_complete', 'LLM_GATE PHASE gate_begin', 'LLM_GATE REJECT node=0 op=GET_ROWS selected=CPU', 'LLM_GATE ACTUAL_CLIENT attempt=0 position=0 result=-3 calls=1 rejected=1 forwarded=0 post_compute=0 expected_errors=2 graph_reused=false inference=false', 'LLM_GATE REJECT node=0 op=GET_ROWS selected=CPU', 'LLM_GATE ACTUAL_CLIENT attempt=1 position=0 result=-3 calls=1 rejected=1 forwarded=0 post_compute=0 expected_errors=2 graph_reused=true inference=false', 'LLM_GATE PHASE gate_complete', 'LLM_GATE PHASE cleanup_begin', 'LLM_GATE PHASE cleanup_complete')
HOST_STORAGE = 'M0_APP HOST_STORAGE allocations=1 releases=1 live_allocations=0 live_requested_bytes=0 peak_requested_bytes=619094016 failures=0 poisoned=false scope=ggml_aligned'
EXPECTED = ('\n'.join((*LLM_CLIENT_LINES[:2],*MARKERS,HOST_STORAGE,*LLM_CLIENT_LINES[4:8],
 'M0_APP FINAL allocations=0 frees=0 queues=0 retired=0 gate_destroyed=true application_fini=pending',*LLM_CLIENT_LINES[9:]))+'\n').encode('ascii')
def check_model_log(raw):
 if not isinstance(raw,bytes) or len(raw)>65536:raise ValueError('model-gate transcript differs')
 pattern=rb'^M0_APP HOST_STORAGE allocations=([1-9][0-9]{0,19}) releases=([1-9][0-9]{0,19}) live_allocations=0 live_requested_bytes=0 peak_requested_bytes=([1-9][0-9]{0,19}) failures=0 poisoned=false scope=ggml_aligned$'
 matches=list(re.finditer(pattern,raw,re.M))
 if len(matches)!=1:raise ValueError('model-gate transcript host storage differs')
 match=matches[0];allocations,releases,peak=map(int,match.groups())
 if allocations!=releases or max(allocations,releases,peak)>2**64-1 or peak<619094016:raise ValueError('model-gate transcript host ownership differs')
 if raw[:match.start()]+HOST_STORAGE.encode('ascii')+raw[match.end():]!=EXPECTED:raise ValueError('model-gate transcript differs')
 return dict(status='PASS',phase='model-gate',actual_calls=2,second_graph_reused=True,expected_errors=4,forwarded=0,post_compute=0,gpu_submits=0,inference=False,final_counts=[0,0,0,0],host_storage=dict(allocations=allocations,releases=releases,peak_requested_bytes=peak,live_requested_bytes=0,failures=0,poisoned=False))
