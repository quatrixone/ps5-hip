"""Genuine two-TU compiler objects and bounded actual-runtime host models."""
import pathlib,sys,os,json,subprocess,hashlib,argparse,struct
import probe_hip_routes as shared
from probe_hip_generic import CORE
from build_canaries import require_device_linker,require_tool_versions
from fat_binary_fixture_test import section
R=pathlib.Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
ref=args.reference_root.resolve();out=args.output.resolve();out.mkdir(exist_ok=False,parents=True)
sources=[R/'dependencies.json',*[R/'src'/n for n in CORE],R/'src/hip_launch_profile.cpp',*list((R/'include/ps5hip').glob('*.h')),*[R/'tests'/n for n in ('hip_multi_a.hip','hip_multi_b.hip','hip_multi_model.cpp','host_multi_registration_test.cpp')],pathlib.Path(__file__)]
sources+= [pathlib.Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and pathlib.Path(m.__file__).resolve().is_relative_to(R)]
before={str(x.relative_to(R)):H(x) for x in sources};commands=[];inputs={}
env=dict(os.environ,LLVM_CONFIG='/usr/bin/llvm-config-18',ASAN_OPTIONS='protect_shadow_gap=0:detect_leaks=0')
def run(label,cmd,mode=None):
 e=dict(env)
 if mode:e['MULTI_MODE']=mode
 q=subprocess.run(list(map(str,cmd)),capture_output=True,text=True,env=e,timeout=180)
 (out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr)
 commands.append(dict(label=label,argv=list(map(str,cmd)),returncode=q.returncode,environment={k:e[k] for k in ('LLVM_CONFIG','ASAN_OPTIONS')}|({'MULTI_MODE':mode} if mode else {})))
 (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 assert q.returncode==0,(label,q.stdout[-2500:],q.stderr[-2500:]);assert not any(s in q.stderr for s in ('AddressSanitizer','runtime error:'))
 return q.stdout,q.stderr
try:
 _,stage,libs,inputs=shared.prepared_inputs(ref)
 lock=json.loads((R/'dependencies.json').read_text());platform=ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
 versions={}
 for tool in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-config'):
  t=pathlib.Path('/usr/bin/'+tool+'-18');inputs[str(t)]=H(t);versions[tool]=run(tool+'-version',[t,'--version'])[0]
 require_tool_versions({k:v for k,v in versions.items() if k!='llvm-config'},lock['llvm']['version'])
 for n in ('prospero-clang++','clang++','prospero-llvm-config','prospero-clang','clang'):
  t=ref/'.deps/sdk/bin'/n
  if t.exists():inputs[str(t)]=H(t)
 for t in platform.rglob('*'):
  if t.is_file():inputs[str(t)]=H(t)
 for host in ('linux','ps5'):
  d=out/host;d.mkdir()
  for actor in ('a','b'):
   obj=d/(actor+'.o');cmd=[ref/'.deps/sdk/bin/prospero-clang++' if host=='ps5' else '/usr/bin/clang++-18','-x','hip','-B/usr/lib/llvm-18/bin','--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-O2','-c',R/'tests'/('hip_multi_'+actor+'.hip'),'--hip-path='+str(stage),'-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',obj]
   if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
   _,trace=run(host+actor+'-driver',cmd+['-###']);require_device_linker(trace,'/usr/bin/ld.lld-18');run(host+actor+'-compile',cmd)
   ro,_=run(host+actor+'-readobj',['/usr/bin/llvm-readobj-18','--sections','--symbols','--relocations',obj]);dis,_=run(host+actor+'-disasm',['/usr/bin/llvm-objdump-18','-dr',obj])
   shared.audit_wrapper(section(obj.read_bytes(),b'.hipFatBinSegment'),ro,host);shared.audit_registration(dis,('multi_'+actor,))
   (d/(actor+'.bundle')).write_bytes(section(obj.read_bytes(),b'.hip_fatbin'))
 script=out/'multi.ld'
 script.write_text('''SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .);
 HIDDEN(bundle_a_begin = .); AOBJ(.hip_fatbin) HIDDEN(bundle_a_end = .);
 . = ALIGN(4096); HIDDEN(bundle_b_begin = .); BOBJ(.hip_fatbin) HIDDEN(bundle_b_end = .);
 HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(8); HIDDEN(__ps5hip_wrappers_begin = .);
 HIDDEN(wa = .); AOBJ(.hipFatBinSegment) HIDDEN(wa_end = .); . = ALIGN(8);
 HIDDEN(wb = .); BOBJ(.hipFatBinSegment) HIDDEN(wb_end = .); HIDDEN(__ps5hip_wrappers_end = .);
 HIDDEN(__ps5hip_module_spans_begin = .);
 QUAD(wa-__ps5hip_wrappers_begin); QUAD(wa_end-__ps5hip_wrappers_begin); QUAD(bundle_a_begin-__ps5hip_bundle_begin); QUAD(bundle_a_end-__ps5hip_bundle_begin);
 QUAD(wb-__ps5hip_wrappers_begin); QUAD(wb_end-__ps5hip_wrappers_begin); QUAD(bundle_b_begin-__ps5hip_bundle_begin); QUAD(bundle_b_end-__ps5hip_bundle_begin);
 HIDDEN(__ps5hip_module_spans_end = .); } } INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
'''.replace('AOBJ',str(out/'linux/a.o')).replace('BOBJ',str(out/'linux/b.o')))
 common=['-pthread','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-I'+str(R/'include'),'-I'+str(stage/'include'),'-I'+str(platform),'-fsanitize=address,undefined','-fno-omit-frame-pointer','-DPS5HIP_RUNTIME_MODEL=1','-fPIE','-pie','-fuse-ld=lld-18']
 models=[]
 for order in ('ab','ba'):
  binary=out/('model-'+order)
  run(order+'-build',['/usr/bin/clang++-18',*common,*[R/'src'/n for n in CORE],R/'src/hip_launch_profile.cpp',R/'tests/hip_multi_model.cpp',*[out/'linux'/(a+'.o') for a in order],'-Wl,-T,'+str(script),'-Wl,--wrap=__hipRegisterFatBinary,--wrap=__hipUnregisterFatBinary,--wrap=__hipRegisterFunction','-o',binary])
  for mode in ('normal','wrong-unregister','duplicate-unregister','skip-unregister','no-finish','live-finish','bad-wrapper','duplicate-token','missing-function','late-after-error','submit','cleanup'):
   text,_=run(order+'-'+mode,[binary],mode);assert ('MULTI_FINAL modules=2 callbacks='+('1' if mode=='bad-wrapper' else '2')) in text
   if mode not in ('bad-wrapper','duplicate-token','missing-function','late-after-error','submit','cleanup'):assert 'MULTI_PASS launches=3 sequence=A/B/A guarded=true source_preserved=true owning_code=true' in text
   models.append(dict(order=order,mode=mode))
 registry=out/'registry'
 run('registry-build',['/usr/bin/clang++-18','-std=c++17','-Wall','-Wextra','-Werror','-g','-fsanitize=address,undefined','-I'+str(R/'include'),R/'tests/host_multi_registration_test.cpp',*[R/'src'/n for n in ('host_registration.cpp','fat_binary.cpp','code_object.cpp')],'-o',registry])
 run('registry-test',[registry,out/'linux/a.bundle'])
 assert before=={str(x.relative_to(R)):H(x) for x in sources};assert all(H(pathlib.Path(k))==v for k,v in inputs.items())
 receipt=dict(classification='offline-two-module-runtime-model-pass',hardware_validated=False,sources=before,inputs=inputs,models=models,artifacts={str(x.relative_to(out)):H(x) for x in out.rglob('*') if x.is_file()},limits=['Synthetic transport validates exact materialized code selection and actual buffers, not GPU execution.','Compiler host objects are genuine, uninstrumented; runtime/model are ASan/UBSan.','No device linking, dynamic modules, framework or native execution claim.'])
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS two modules, both callback orders, A/B/A owning-code and lifetime models')
except BaseException as e:
 (out/'failure.json').write_text(json.dumps(dict(error=repr(e),sources=before,inputs=inputs),indent=2)+'\n');raise
