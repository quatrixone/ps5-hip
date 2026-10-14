"""Verify the frozen allocation prerequisite without claiming public HIP execution."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];B=a.reference_root.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();bindings={};proofs={}
def bind(path,expected=None):
 path=Path(path).resolve();digest=H(path)
 assert expected is None or digest==expected,str(path)
 assert str(path) not in bindings or bindings[str(path)]==digest,str(path)
 bindings[str(path)]=digest;return digest
for name in ['svm-asan-v2','svm-tsan-v1','svm-native-v1','hip-memory-native-v2']:
 folder=R/'results'/name;path=folder/'receipt.json';proofs[name]=bind(path);proof=json.loads(path.read_text())
 for q,z in proof['inputs'].items():bind(q,z)
 for q,z in proof['artifacts'].items():bind(folder/q,z)
 if name.startswith('svm-a') or name.startswith('svm-t'):
  assert proof['source_units']==39 and len(proof['modes'])==18 and not proof['gpu_executed']
  assert 'svm-launch' in proof['modes']
 if name=='svm-native-v1':
  outcomes={q['label']:q['exit_code'] for q in proof['commands']}
  assert outcomes['device']==outcomes['memory']==0 and outcomes['hip-memory']!=0
 if name=='hip-memory-native-v2':assert proof['compile_passed'] and not proof['native_linked']
# The fresh snapshot adds only genuine HIP profiling headers; all qualified CLR
# implementation and model sources remain the same bytes as both sanitizer runs.
old=R/'results/clr-source-v7/source';fresh=R/'results/clr-source-v8/source'
for q in old.rglob('*'):
 if q.is_file():assert H(q)==H(fresh/q.relative_to(old)),str(q);bind(fresh/q.relative_to(old))
for q in [R/'results/clr-source-v8/preparation.json',fresh/'hipamd/src/hip_prof_api.h',fresh/'hipamd/src/hip_internal.hpp']:bind(q)
for rel in ['tools/ps5-clangxx','.deps/sdk/bin/prospero-clang++','.deps/sdk/bin/clang++','.deps/sdk/bin/prospero-llvm-config']:bind(B/rel)
for name in ['llvm-config-18','clang++-18','ld.lld-18']:bind(shutil.which(name))
manifest=json.loads((R/'ports/clr/adapter-manifest.json').read_text())
for name,z in manifest['patches'].items():bind(R/'ports/clr'/name,z)
changed=subprocess.check_output(['git','diff','--name-only','bd5138b','HEAD'],cwd=R,text=True).splitlines()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True).strip(),'freeze requires clean source'
frozen={}
for name in changed:
 blob=subprocess.check_output(['git','show','HEAD:'+name],cwd=R)
 assert hashlib.sha256(blob).hexdigest()==H(R/name),name
 frozen[name]=bind(R/name)
command=['python3',str(R/'tests/clr_startup_port_test.py'),'--upstream',str(B/'results/g4-runtime-architecture-v1/sources/clr-65d174c3e35423bf25c9c369513780c9d9ca760c')]
q=subprocess.run(command,capture_output=True,text=True);(O/'roundtrip.stdout').write_text(q.stdout);(O/'roundtrip.stderr').write_text(q.stderr);assert q.returncode==0,q.stderr
report=dict(classification='offline-real-clr-svm-allocation-prerequisite',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),proofs=proofs,exact_bindings=bindings,frozen_sources=frozen,roundtrip=dict(argv=command,exit_code=q.returncode,stdout_sha256=H(O/'roundtrip.stdout'),stderr_sha256=H(O/'roundtrip.stderr')),hardware_executed=False,public_hip_qualified=False,limits=['Synthetic native providers and transport in sanitizer tests; no GPU execution.','Native Device and genuine memory.cpp/hip_memory.cpp compile; complete public HIP link not performed.','Public initialization, device registration, stream synchronization and hip::tls emulated TLS remain unresolved integration prerequisites.','64 MiB single-device zero-flag allocation profile; no fine-grained, atomic, imported, host or OpenCL SVM mapping support.'])
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(H(O/'receipt.json'))
