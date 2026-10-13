#!/usr/bin/env python3
"""Genuine GEMM compiler and actual-runtime qualification, synthetic transport only."""
import argparse,hashlib,json,os,pathlib,shutil,struct,subprocess,sys,shlex,re
if not __debug__:raise RuntimeError('assertions required')
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import probe_hip_routes as shared
import probe_host_registration
from probe_hip_generic import CORE
from build_canaries import require_device_linker,require_tool_versions
from fat_binary_fixture_test import section

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
 ref=args.reference_root.resolve();out=args.output.resolve();assert not out.exists();out.mkdir(parents=True)
 names=['docs/GEMM_SUBSET.md','src/gemm.hip','include/ps5hip/gemm.h','tests/gemm_client_model.cpp','tests/gemm_transport_model.cpp','tools/probe_gemm.py','dependencies.json']
 names+=['src/'+n for n in CORE]+['src/hip_launch_profile.cpp']
 names += [str(x.relative_to(ROOT)) for x in (ROOT/'include/ps5hip').glob('*.h')]
 names += ['tools/'+n for n in ('probe_hip_routes.py','probe_hip_generic.py','probe_host_registration.py','build_canaries.py')]+['tests/fat_binary_fixture_test.py']
 names += [str(pathlib.Path(m.__file__).resolve().relative_to(ROOT)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and pathlib.Path(m.__file__).resolve().is_relative_to(ROOT)]
 sources={n:sha(ROOT/n) for n in sorted(set(names))};commands=[]
 env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18'),ASAN_OPTIONS='protect_shadow_gap=0:detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
 def run(label,cmd,expected=0):
  cmd=list(map(str,cmd));z=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
  (out/(label+'.stdout')).write_text(z.stdout);(out/(label+'.stderr')).write_text(z.stderr)
  commands.append(dict(label=label,argv=cmd,exit_code=z.returncode,expected=expected));(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  assert z.returncode==expected,(label,z.returncode,z.stderr[-2000:]);return z.stdout,z.stderr
 try:
  _,stage,libs,inputs=shared.prepared_inputs(ref)
  tools={n:shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-ar','llvm-nm')};assert all(tools.values())
  for path in tools.values():inputs[path]=sha(path)
  versions={n:run(n+'-version',[path,'--version'])[0] for n,path in tools.items()}
  require_tool_versions(versions,json.loads((ROOT/'dependencies.json').read_text())['llvm']['version'])
  wrapper=ref/'.deps/sdk/bin/prospero-clang++';inputs[str(wrapper)]=sha(wrapper)
  def bind(path):
   path=pathlib.Path(path).resolve();digest=sha(path)
   assert str(path) not in inputs or inputs[str(path)]==digest
   inputs[str(path)]=digest
  for name in ('prospero-clang++','clang++','prospero-llvm-config','prospero-llvm','prospero-lld'):
   bind(ref/'.deps/sdk/bin'/name)
  bind(env['LLVM_CONFIG'])
  for name in ('bash','env','readlink','basename','dirname','realpath','cut'):
   bind(shutil.which(name))
  llvm_bindir=run('llvm-bindir',[env['LLVM_CONFIG'],'--bindir'])[0].strip()
  bind(pathlib.Path(llvm_bindir)/'clang++');bind(pathlib.Path(llvm_bindir)/'ld.lld')
  def scan(label,argv):
   dep=out/(label+'.d');run(label,list(argv)+['-M','-MF',dep])
   paths=shlex.split(dep.read_text().replace(chr(92)+chr(10),' ').split(':',1)[1])
   assert paths
   for path in paths:bind(path)
  audits={}
  sdk_revision=ref/'.deps/sdk/.ps5-sdk-revision';inputs[str(sdk_revision)]=sha(sdk_revision)
  assert sdk_revision.read_text().strip()==json.loads((ROOT/'dependencies.json').read_text())['sdk']['revision']
  for host in ('linux','ps5'):
   obj=out/(host+'.o')
   cmd=[wrapper if host=='ps5' else tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),'--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-c',ROOT/'src/gemm.hip','--hip-path='+str(stage),'-I'+str(stage/'include'),'-I'+str(ROOT/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',obj]
   if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
   else:cmd+=['-Xarch_host','-fsanitize=address,undefined','-Xarch_host','-fno-omit-frame-pointer']
   _,driver=run(host+'-driver',cmd+['-###']);require_device_linker(driver,tools['ld.lld'])
   scan_cmd=cmd.copy();at=scan_cmd.index('-o');del scan_cmd[at:at+2]
   scan(host+'-host-headers',scan_cmd+['--cuda-host-only'])
   scan(host+'-device-headers',scan_cmd+['--cuda-device-only','--no-gpu-bundle-output'])
   run(host+'-compile',cmd)
   run(host+'-archive',[tools['llvm-ar'],'rcs',out/(host+'-libps5hip-gemm.a'),obj])
   symbols,_=run(host+'-symbols',[tools['llvm-nm'],'--defined-only',obj])
   assert any(line.endswith(' T ps5hipSgemmNN') for line in symbols.splitlines())
   bundle=section(obj.read_bytes(),b'.hip_fatbin');(out/(host+'.bundle')).write_bytes(bundle)
   magic=b'__CLANG_OFFLOAD_BUNDLE__';assert bundle.startswith(magic);count=struct.unpack_from('<Q',bundle,len(magic))[0];at=len(magic)+8;selected=[]
   for _ in range(count):
    offset,size,length=struct.unpack_from('<QQQ',bundle,at);at+=24;identity=bundle[at:at+length].decode();at+=length
    assert offset<=len(bundle) and size<=len(bundle)-offset
    if identity=='hipv4-amdgcn-amd-amdhsa--gfx1013':selected.append(bundle[offset:offset+size])
   assert len(selected)==1;code=out/(host+'.hsaco');code.write_bytes(selected[0])
   meta,_=run(host+'-metadata',[tools['llvm-readobj'],'--notes',code]);isa,_=run(host+'-isa',[tools['llvm-objdump'],'--disassemble','--mcpu=gfx1013',code])
   audit=shared.audit_object(code.read_bytes(),meta,isa,('ps5hip_sgemm_nn',));audits[host]=audit
   d=audit['descriptors']['ps5hip_sgemm_nn'];ops=audit['isa']['ps5hip_sgemm_nn']['opcodes']
   assert d['group_bytes']==512 and d['private_bytes']==0 and d['kernarg_bytes']==328 and d['properties']==0x409
   assert 's_barrier' in ops and any(x.startswith('ds_read') for x in ops) and any(x.startswith('ds_write') for x in ops)
   assert not any(x.startswith(('scratch_','flat_','s_swappc')) for x in ops)
   print(host,d,flush=True)
  assert section((out/'linux.hsaco').read_bytes(),b'.text')==section((out/'ps5.hsaco').read_bytes(),b'.text')
  script=out/'linux.ld';script.write_text('''SECTIONS { .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); } } INSERT AFTER .rodata;
SECTIONS { .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(32); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = __ps5hip_wrappers_begin + 24); } } INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata)); HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array)); HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
  platform=ref/'.deps'/('sdk-source-'+json.loads((ROOT/'dependencies.json').read_text())['sdk']['revision'])/'platform/include'
  binary=out/'model'
  model_cmd=[tools['clang++'],'-std=c++17','-O1','-g','-pthread','-Wall','-Wextra','-Werror','-ffp-contract=off','-I',ROOT/'include','-I',stage/'include','-I',platform,'-fsanitize=address,undefined','-fno-omit-frame-pointer','-DPS5HIP_RUNTIME_MODEL=1','-D__HIP_PLATFORM_AMD__','-fPIE','-pie','-fuse-ld=lld-18',*[ROOT/'src'/n for n in CORE],ROOT/'src/hip_launch_profile.cpp',ROOT/'tests/gemm_transport_model.cpp',ROOT/'tests/gemm_client_model.cpp',out/'linux.o','-Wl,-T,'+str(script),'-o',binary]
  model_sources=[ROOT/'src'/n for n in CORE]+[ROOT/'src/hip_launch_profile.cpp',ROOT/'tests/gemm_transport_model.cpp',ROOT/'tests/gemm_client_model.cpp']
  first=model_cmd.index(str(model_sources[0])) if str(model_sources[0]) in model_cmd else model_cmd.index(model_sources[0])
  flags=[v for v in model_cmd[:first] if v not in ('-pie','-fuse-ld=lld-18')]
  for index,source in enumerate(model_sources):scan('model-headers-'+str(index),flags+[source])
  _,link_driver=run('model-driver',model_cmd+['-###'])
  libraries=set()
  for line in link_driver.splitlines():
   if line.lstrip().startswith('"'):
    for token in shlex.split(line):
     path=pathlib.Path(token)
     if path.is_absolute() and path.is_file():bind(path)
     if token.startswith('-l') and len(token)>2:libraries.add(token[2:])
  for library in sorted(libraries):
   selected=None
   for ext in ('so','a'):
    name='lib'+library+'.'+ext
    value=run('resolve-'+name,[tools['clang++'],'-print-file-name='+name])[0].strip()
    path=pathlib.Path(value)
    if path.is_absolute() and path.is_file():selected=path;break
   assert selected is not None,library
   bind(selected)
   if not selected.read_bytes().startswith(b'\x7fELF') and selected.suffix!='.a':
    for value in re.findall(r'/[^\s()]+',selected.read_text()):
     path=pathlib.Path(value)
     if path.is_file():bind(path)
  run('model-build',model_cmd)
  modes=('normal','init','preflush','missing','corrupt','input','guard','padding','alloc1','alloc2','alloc3','map1','submit','throw','kick','timeout','cleanup')
  terminal=('submit','throw','kick','timeout','cleanup')
  for mode in modes:
   expected=0 if mode=='normal' else 30 if mode in terminal else 20
   stdout,stderr=run('model-'+mode,[binary,mode],expected)
   assert not any(x in stdout+stderr for x in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer'))
   if mode=='normal':assert 'GEMM_PASS cases=14 values=11000 rejections=13 synthetic_transport=true' in stdout and 'live=0 backends=1 submits=14 releases=77 finalized=1' in stdout
   elif mode in terminal:assert f'GEMM_TERMINAL live={4 if mode=="cleanup" else 7} unresolved={0 if mode=="cleanup" else 1} submits=1' in stdout and 'GEMM_PASS' not in stdout
   else:
    assert 'GEMM_REJECT clean=1' in stdout and 'GEMM_PASS' not in stdout
    before=mode in ('alloc1','alloc2','alloc3','map1')
    submits=0 if before or mode in ('init','preflush') else 1
    releases={'alloc1':0,'alloc2':1,'alloc3':2,'map1':1,'init':5}.get(mode,7)
    assert f'GEMM_MODEL_FINAL live=0 backends={int(not before)} submits={submits} releases={releases} finalized={int(mode!="init")}' in stdout
    assert stdout.count('GEMM_ORACLE_BEGIN')==int(mode in ('missing','corrupt','input','guard','padding'))
  assert all(str(pathlib.Path(m.__file__).resolve().relative_to(ROOT)) in sources for m in list(sys.modules.values()) if getattr(m,'__file__',None) and pathlib.Path(m.__file__).resolve().is_relative_to(ROOT))
  assert all(sha(ROOT/n)==h for n,h in sources.items());assert all(sha(n)==h for n,h in inputs.items())
  report=dict(classification='offline-gemm-compiler-runtime-model',sources=sources,inputs=inputs,commands=commands,modes=modes,audits=audits,sanitizer_environment={k:env[k] for k in ('LLVM_CONFIG','ASAN_OPTIONS','UBSAN_OPTIONS')},gpu_executed=False,g6_accepted=False,limits=['Synthetic arithmetic tests host orchestration, not AMDGPU numerical results.','Compiler-discovered host/device/model header closure and selected tools are bound; this is not an OS/container hermetic build.'],artifacts={str(x.relative_to(out)):sha(x) for x in out.rglob('*') if x.is_file()})
  (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',sha(out/'receipt.json'))
 except BaseException as e:
  (out/'failure.json').write_text(json.dumps(dict(error=repr(e),sources=sources,commands=commands),indent=2)+'\n');raise
if __name__=='__main__':main()
