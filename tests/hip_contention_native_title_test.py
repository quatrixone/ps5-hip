#!/usr/bin/env python3
"""Link-only Linux proof of actual production/runtime/compiler/interposer paths."""
import argparse,hashlib,json,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from native_vector_title import validate_workers_compiler,audit_contention,profile_for
from probe_hip_generic import CORE
from probe_hip_routes import prepared_inputs
# These helpers are also imported lazily by the validators below. Load their
# transitive local imports before snapshotting the executed helper closure.
import probe_hipify_client,probe_host_registration,fat_binary_fixture_test
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(b,msg):
 if not b:raise RuntimeError(msg)
def local_helpers():
 paths=set()
 for module in tuple(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if not name:continue
  path=pathlib.Path(name).resolve()
  if path.suffix=='.py' and path.is_relative_to(ROOT):paths.add(str(path.relative_to(ROOT)))
 return paths
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--compiler',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 ref=a.reference_root.resolve();proof=a.compiler.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 helper_sources={n:sha(ROOT/n) for n in sorted(local_helpers()|{'dependencies.json'})}
 raw=(proof/'receipt.json').read_bytes();obj=(proof/'ps5/threads.o').read_bytes();authority=validate_workers_compiler(raw,obj,ROOT)
 inputs={str(proof/'receipt.json'):sha(proof/'receipt.json')}
 for n,h in authority['artifacts'].items():require(sha(proof/n)==h,'compiler drift '+n);inputs[str(proof/n)]=h
 for n,h in authority['inputs'].items():require(sha(n)==h,'compiler input drift '+n);inputs[n]=h
 _,headers,_,prepared=prepared_inputs(ref);inputs.update(prepared)
 sdk=ref/'.deps/sdk-source-95c08f27386fc698f6bbe21dde3030140a41d10b/platform/include'
 for folder in (headers/'include',sdk):inputs.update({str(f):sha(f) for f in folder.rglob('*') if f.is_file()})
 sources=[*['src/'+n for n in CORE],'src/agc_backend.cpp','src/hip_launch_profile.cpp','tools/hip_contention_client.cpp','tools/hip_contention_trace.cpp','tools/hip_contention.h','tools/native_vector_title.py','tools/build_native_vector_title.py','tests/hip_contention_native_title_test.py',*[str(f.relative_to(ROOT)) for f in (ROOT/'include/ps5hip').glob('*.h')]]
 sources={n:sha(ROOT/n) for n in sources};sources.update(helper_sources);commands=[]
 for name in ('clang++-18','ld.lld-18','llvm-objdump-18'):
  tool=shutil.which(name);require(tool,'tool '+name);inputs[tool]=sha(tool)
 def run(label,argv):
  q=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,timeout=180);(out/(label+'.stdout')).write_bytes(q.stdout);(out/(label+'.stderr')).write_bytes(q.stderr)
  commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':q.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');require(q.returncode==0,label+': '+q.stderr.decode()[-3000:]);return q.stdout.decode()
 # Harmless providers exist only to resolve imports. This linked image is NEVER run.
 (out/'imports.cpp').write_text('extern "C" {int sceAgcInit(unsigned){return 0;} int sceAgcDriverSubmitDcb(void*){return 0;} int sceAgcSuspendPoint(){return 0;} long sceKernelGetDirectMemorySize(){return 0;} int sceKernelAllocateDirectMemory(long,long,unsigned long,unsigned long,int,long*){return -1;} int sceKernelMapDirectMemory(void**,unsigned long,int,int,long,unsigned long){return -1;} int sceKernelMunmap(void*,unsigned long){return -1;} int sceKernelReleaseDirectMemory(long,unsigned long){return -1;}}\n')
 (out/'main.cpp').write_text('int main(){return 0;}\n')
 (out/'sections.ld').write_text('SECTIONS { .ps5hip_fatbin : { __ps5hip_bundle_begin = .; KEEP(*(.hip_fatbin)) __ps5hip_bundle_end = .; } .ps5hip_wrappers : { __ps5hip_wrappers_begin = .; KEEP(*(.hipFatBinSegment)) __ps5hip_wrappers_end = .; } } INSERT AFTER .rodata; HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));\n')
 run('providers',['clang++-18','-shared','-fPIC',out/'imports.cpp','-o',out/'libimports.so'])
 common=['clang++-18','-std=c++17','-O2','-g','-pthread','-Wall','-Wextra','-Werror','-fPIE','-I',ROOT/'include','-I',headers/'include','-I',sdk]
 run('link',[*common,'-pie','-fuse-ld=lld-18',*[ROOT/'src'/n for n in CORE],ROOT/'src/agc_backend.cpp',ROOT/'src/hip_launch_profile.cpp',ROOT/'tools/hip_contention_client.cpp',ROOT/'tools/hip_contention_trace.cpp',proof/'ps5/threads.o',out/'main.cpp',out/'libimports.so','-Wl,-T,'+str(out/'sections.ld'),*['-Wl,--wrap='+n for n in ('hipLaunchKernel','__hipPushCallConfiguration','pthread_mutex_lock','sceAgcInit','sceAgcDriverSubmitDcb','sceAgcSuspendPoint')],'-o',out/'linked'])
 dis=run('disassembly',['llvm-objdump-18','-d','--no-show-raw-insn',out/'linked']);audit=audit_contention(dis,out/'linked')
 negatives=0
 for name in ('hipLaunchKernel','__hipPushCallConfiguration','pthread_mutex_lock','sceAgcInit','sceAgcDriverSubmitDcb','sceAgcSuspendPoint'):
  try:audit_contention(dis.replace('<__wrap_'+name+'>','<removed_redirect>'),out/'linked')
  except ValueError:negatives+=1
  else:raise RuntimeError('missing wrapper accepted '+name)
 profile=profile_for('hip_contend_main');require((profile['dispatches'],profile['exact_values'],profile['word_comparisons'])==(3,331,813),'profile counts')
 require(local_helpers()<=set(helper_sources),'unbound late local helper import')
 require(all(sha(ROOT/n)==h for n,h in sources.items()),'source drift');require(all(sha(n)==h for n,h in inputs.items()),'input drift')
 receipt={'classification':'contention-actual-runtime-compiler-Linux-link-only-pass','native_built':False,'executed':False,'sources':sources,'helper_sources':helper_sources,'inputs':inputs,'audit':audit,'negative_cases':negatives,'commands':commands,'artifacts':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS',sha(out/'receipt.json'))
if __name__=='__main__':main()
