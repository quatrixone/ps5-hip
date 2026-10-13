#!/usr/bin/env python3
"""Compile genuine HIP stubs and exercise the checked runtime with deterministic threads."""
import argparse,hashlib,json,os,pathlib,re,shutil,subprocess
from probe_hip_routes import prepared_inputs,audit_object,audit_registration
from probe_hip_generic import CORE
from probe_hipify_client import device_object
from fat_binary_fixture_test import section
ROOT=pathlib.Path(__file__).resolve().parents[1]
MODES=('context','context-recover','marker-exit','stacks','vector','ownership','blocked-free','blocked-sync','blocked-copy','blocked-launch','lock','unlock','key','registration-oom','setspecific','uncertain-install','clear','destructor-lock','destructor-unlock','duplicate','abandoned','live-finish','finish-wait','no-finish','reject-unregister','submit','kick','timeout','throw','unknown','postflush','cleanup')
CPU_MODES=('marker-exit','stacks','vector','lock','key','registration-oom','setspecific','uncertain-install','clear','destructor-lock','destructor-unlock','duplicate','abandoned','live-finish')
MODES+=('finish-clear','state-oom')
CPU_MODES+=('finish-clear','state-oom')
MODES+=('memset','blocked-memset')
CPU_MODES+=('memset',)
MODES+=('discovery','discovery-unlock')
CPU_MODES+=('discovery','discovery-unlock')
MODES+=('device-copy','blocked-device-copy')
CPU_MODES+=('device-copy',)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise RuntimeError(message)
def validate(text,mode):
 require(not any(s in text for s in ('AddressSanitizer','UndefinedBehaviorSanitizer','ThreadSanitizer','runtime error:','assertion failed')),'runtime diagnostic')
 terminal=mode in ('submit','kick','timeout','throw','unknown','postflush','cleanup')
 require(text.splitlines().count('THREAD_MODEL_PASS '+mode+(' retained=true' if terminal else ''))==1,'missing exact mode result')
 if mode not in ('submit','kick','timeout','throw','unknown','postflush','cleanup'):
  expected=mode not in ('lock','unlock','key','registration-oom','setspecific','uncertain-install','clear','destructor-lock','destructor-unlock','duplicate','abandoned','live-finish','finish-wait','no-finish','reject-unregister','finish-clear','state-oom','discovery-unlock')
  require(text.count('THREAD_MODEL_FINAL finalized='+str(int(expected))+' ')==1,'finalizer result')
  if not expected:require('VECTOR_FINAL ' not in text,'unsafe automatic State destruction')
  if expected:
   count=2 if mode=='blocked-launch' else int(mode in ('ownership','blocked-free','blocked-sync','blocked-copy','blocked-memset','blocked-device-copy'))
   require(text.splitlines().count(f'VECTOR_FINAL registered=1 removed=1 completed={count} active=0 terminal=0 main_finished=1 hardware_validated=false')==1,'exact complete State destructor')
   require(text.splitlines().count('THREAD_MODEL_FINAL finalized=1 live=0')==1,'complete backing closure')
  elif mode in ('unlock','finish-wait'):
   require(text.splitlines().count('THREAD_MODEL_FINAL finalized=0 live=1')==1,'retained data owner')
 else:require('VECTOR_FINAL ' not in text and 'THREAD_MODEL_FINAL ' not in text,'terminal process falsely finalized')
 if mode=='discovery':require(text.splitlines().count('DISCOVERY_EARLY state_attempts=0 backend_attempts=0 worker_joined=true')==1,'pre-registration discovery missing')
def check(out):
 r=json.loads((out/'receipt.json').read_text());require(r['classification']=='checked-threaded-runtime-host-model-pass','scope')
 for n,h in r['sources'].items():require(sha(ROOT/n)==h,'source '+n)
 for n,h in r['inputs'].items():require(sha(n)==h,'input '+n)
 for n,h in r['artifacts'].items():require(sha(out/n)==h,'artifact '+n)
 require(tuple(r['modes'])==(CPU_MODES if r['sanitizer']=='thread' else MODES),'mode inventory')
 for mode in r['modes']:validate((out/('run-'+mode+'.stdout')).read_text()+(out/('run-'+mode+'.stderr')).read_text(),mode)
 print('Checked',len(r['modes']),'deterministic threaded modes')
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);p.add_argument('--check',type=pathlib.Path);p.add_argument('--sanitizer',choices=('address,undefined','thread'),default='address,undefined');a=p.parse_args()
 if a.check:check(a.check.resolve());return
 require(a.reference_root and a.output,'reference/output required');ref=a.reference_root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 files=[ROOT/'src'/n for n in CORE]+[ROOT/'src/hip_launch_profile.cpp',ROOT/'tests/hip_threads.hip',ROOT/'tests/hip_threads_model.cpp',pathlib.Path(__file__),ROOT/'tests/hip_threads_recipe_test.py',ROOT/'docs/HIP_HOST_THREADS.md',*list((ROOT/'include/ps5hip').glob('*.h'))]
 files += [ROOT/n for n in ('tools/probe_hip_routes.py','tools/probe_hip_generic.py','tools/probe_hipify_client.py','tools/probe_host_registration.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py','tests/native_import_libraries_test.py','tests/native_agc_imports_test.py')]
 before={str(f.relative_to(ROOT)):sha(f) for f in files};commands=[]
 def run(label,argv,extra=None):
  env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18'),ASAN_OPTIONS='protect_shadow_gap=0:detect_leaks=0');env.update(extra or {})
  q=subprocess.run(list(map(str,argv)),cwd=ROOT,capture_output=True,timeout=60,env=env)
  (out/(label+'.stdout')).write_bytes(q.stdout);(out/(label+'.stderr')).write_bytes(q.stderr);commands.append({'label':label,'argv':list(map(str,argv)),'exit_code':q.returncode,'environment':{'ASAN_OPTIONS':env['ASAN_OPTIONS'],**(extra or {})}});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  require(q.returncode==0,label+' failed: '+q.stderr.decode(errors='replace')[-3000:]);return q.stdout.decode(),q.stderr.decode()
 inputs={}
 try:
  _,stage,libs,inputs=prepared_inputs(ref);lock=json.loads((ROOT/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
  inputs.update({str(f):sha(f) for f in platform.rglob('*.h') if f.is_file()})
  tools={n:shutil.which(n) for n in ('clang++-18','ld.lld-18','llvm-readobj-18','llvm-objdump-18')};require(all(tools.values()),'tools missing');inputs.update({v:sha(v) for v in tools.values()})
  obj=out/'threads.o';cmd=['clang++-18','-x','hip','-B'+str(pathlib.Path(tools['ld.lld-18']).resolve().parent),'--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-O2','-c',ROOT/'tests/hip_threads.hip','--hip-path='+str(stage),'-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-Xarch_host','-fsanitize=address,undefined','-o',obj]
  cmd=[x.replace('-fsanitize=address,undefined','-fsanitize='+a.sanitizer) if isinstance(x,str) else x for x in cmd]
  run('hip-compile',cmd);bundle=section(obj.read_bytes(),b'.hip_fatbin');(out/'threads.bundle').write_bytes(bundle);code=out/'threads.hsaco';code.write_bytes(device_object(bundle))
  metadata,_=run('metadata',['llvm-readobj-18','--notes',code]);isa,_=run('isa',['llvm-objdump-18','--disassemble','--mcpu=gfx1013',code]);audit=audit_object(code.read_bytes(),metadata,isa,('thread_words',));assembly,_=run('host-isa',['llvm-objdump-18','-dr',obj]);registration=audit_registration(assembly,('thread_words',))
  common=['clang++-18','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-pthread','-fsanitize=address,undefined','-fno-omit-frame-pointer','-fPIE','-DPS5HIP_RUNTIME_MODEL=1','-I',ROOT/'include','-I',stage/'include','-I',platform]
  common=[x.replace('-fsanitize=address,undefined','-fsanitize='+a.sanitizer) if isinstance(x,str) else x for x in common]
  runtime=out/'runtime.o';run('runtime-compile',[*common,'-DPS5HIP_THREAD_MODEL=1','-Dpthread_mutex_lock=thread_mutex_lock','-Dpthread_mutex_unlock=thread_mutex_unlock','-Dpthread_key_create=thread_key_create','-Dpthread_setspecific=thread_setspecific','-c',ROOT/'src/hip_runtime.cpp','-o',runtime])
  script=out/'linux.ld';script.write_text('''SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); } } INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array)); HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
  binary=out/'threads';run('model-link',[*common,'-pie','-fuse-ld=lld-18',*[ROOT/'src'/n for n in CORE if n!='hip_runtime.cpp'],ROOT/'src/hip_launch_profile.cpp',ROOT/'tests/hip_threads_model.cpp',runtime,obj,'-Wl,-T,'+str(script),'-Wl,--wrap=__hipUnregisterFatBinary','-o',binary])
  modes=CPU_MODES if a.sanitizer=='thread' else MODES
  for mode in modes:
   extra={'PS5HIP_KEY_FAULT':'1'} if mode=='key' else {'PS5HIP_CONTEXT_FAULT':'1'} if mode=='registration-oom' else {'PS5HIP_STATE_FAULT':'1'} if mode=='state-oom' else None
   if mode=='discovery':extra={'PS5HIP_DISCOVERY_EARLY':'1'}
   stdout,stderr=run('run-'+mode,[binary,mode],extra);validate(stdout+stderr,mode);print(mode+' PASS',flush=True)
  require(all(sha(ROOT/n)==h for n,h in before.items()),'source drift');require(all(sha(n)==h for n,h in inputs.items()),'input drift')
  r={'schema':1,'classification':'checked-threaded-runtime-host-model-pass','sources':before,'inputs':inputs,'commands':commands,'audit':audit,'registration':registration,'modes':list(MODES),'hardware_executed':False,'limits':['Synthetic transport checks packets and ownership, not AMD ISA or native scheduler/AGC affinity.','ASan fixed shader window requires protect_shadow_gap=0. Leak checking disabled for deliberately retained fault-state processes.','Fault unlock really unlocks then reports failure; no stuck-lock recovery claim.','TSAN qualification is separate.'],'artifacts':{str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}}
  r['modes']=list(modes);r['sanitizer']=a.sanitizer
  (out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS',sha(out/'receipt.json'))
 except BaseException as error:
  (out/'failure.json').write_text(json.dumps({'error':repr(error),'sources_before':before,'commands':commands},indent=2)+'\n');raise
if __name__=='__main__':main()
