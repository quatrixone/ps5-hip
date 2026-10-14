"""Bind the complete offline native GPU candidate without claiming execution."""
if not __debug__: raise RuntimeError('Validation requires ordinary Python')
from pathlib import Path
import hashlib,json,subprocess,functools
C=Path(__file__).resolve().parents[1];O=C/'results/offline-candidate-v2';O.mkdir(exist_ok=False)
@functools.lru_cache(None)
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
proofs={}
for name in ['source-v1','adapter-address-v1','adapter-thread-v1','client-address-v2','client-thread-v2','native-title-v1','native-conversion-v1','native-admission-v1']:
    p=C/'results'/name/'receipt.json';q=json.loads(p.read_text());proofs[str(p)]=h(p)
    for n,v in q['inputs'].items():assert h(n)==v,n
    for n,v in q.get('artifacts',{}).items():assert h(p.parent/n)==v,n
    if name.startswith('adapter-'):assert q['source_units']==39 and len(q['modes'])==14
    if name.startswith('client-'):assert len(q['modes'])==11 and 'join-fail' in q['modes']
    if name=='native-title-v1':assert q['new_units']==46 and q['reused_cpu_units']==[] and q['reused_aot_objects']==[]
paths=['docs/CLR_NATIVE_GPU.md','tools/prepare_clr_gpu_source.py','tools/qualify_clr_gpu_adapter.py','tools/probe_clr_gpu_client.py','tools/build_clr_gpu_title.py','tools/convert_clr_gpu_title.py','tools/audit_clr_gpu_title.py','tools/package_clr_gpu_title.py','tools/record_clr_gpu_candidate.py','tools/clr_gpu_title.cpp','tests/clr_gpu/client_model.cpp']
paths += ['tools/clr_gpu/'+n for n in ['client.cpp','client.h','watchdog.cpp','watchdog.h']]
sources={n:h(C/n) for n in paths}
report=dict(classification='offline-native-clr-gpu-startup-candidate-ready-for-review',
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip(),sources=sources,proofs=proofs,
    native={'fself_sha256':h(C/'results/native-conversion-v1/eboot.bin'),'raw_elf_sha256':h(C/'results/native-title-v1/title.elf')},
    expected={'dispatches':1,'exact_values':73,'readback_words':81,'guards':8,'allocations':2,'frees':2,'queues_joined':2},
    limits=['No console execution or public HIP/G5 acceptance.','Model transport arithmetic is synthetic; native image uses real AgcBackend.','Active arithmetic uses captured scalar/index; guard words observe copied input.','Normal teardown requires positive joins; uncertainty quarantines without normal owner/global teardown.'])
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(h(O/'receipt.json'))
