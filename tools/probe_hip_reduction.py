#!/usr/bin/env python3
"""Compile the unmodified upstream reduction; no runtime or hardware execution."""
if not __debug__: raise RuntimeError("optimized Python is forbidden")
import sys,pathlib,json,subprocess,shutil,struct,hashlib,os
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from probe_hip_routes import prepared_inputs,sha,audit_object
from build_canaries import require_device_linker
from fat_binary_fixture_test import section
import argparse,shlex
ap=argparse.ArgumentParser();ap.add_argument('--reference-root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args();ref=args.reference_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
src=ROOT/'tests/upstream-reduction/warp_size_reduction.hip';assert sha(src)=='42ffe47b3d8dfdd59d2c8ebaabcf7ae6451c97f1055fe9ca6fafebdcc274a57a'
(out/src.name).write_bytes(src.read_bytes())
(out/'popcount.hpp').write_bytes((src.parent/'popcount.hpp').read_bytes())
sources={str(p):sha(p) for p in [src,src.parent/'popcount.hpp',src.parent/'source.json',pathlib.Path(__file__),ROOT/'tools/probe_hip_routes.py',ROOT/'tools/build_canaries.py',ROOT/'tests/fat_binary_fixture_test.py']}
_,stage,libs,inputs=prepared_inputs(ref)
tools={n:shutil.which(n+'-18') for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump','llvm-nm','llvm-config')};assert all(tools.values())
sdk=(ref/'.deps/sdk').resolve();wrapper=sdk/'bin/prospero-clang++';inputs[str(wrapper)]=sha(wrapper)
lock=json.loads((ROOT/'dependencies.json').read_text());assert (sdk/'.ps5-sdk-revision').read_text().strip()==lock['sdk']['revision']
tool_hashes={n:{'path':p,'sha256':sha(pathlib.Path(p))} for n,p in tools.items()}
wrapper_inputs={str(p):sha(p) for p in (wrapper,wrapper.parent/'clang++',wrapper.parent/'prospero-llvm-config',sdk/'.ps5-sdk-revision')}
inputs.update(wrapper_inputs)
commands=[]
def run(label,argv):
 q=subprocess.run(list(map(str,argv)),cwd=ROOT,text=True,capture_output=True,timeout=180,env=dict(os.environ,LLVM_CONFIG=shutil.which('llvm-config-18')))
 (out/(label+'.stdout')).write_text(q.stdout);(out/(label+'.stderr')).write_text(q.stderr)
 commands.append({'environment':{'LLVM_CONFIG':shutil.which('llvm-config-18')},'label':label,'argv':list(map(str,argv)),'exit_code':q.returncode});(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 return q
for host in ('linux','ps5'):
 d=out/host;d.mkdir();obj=d/'reduction.o'
 cmd=[wrapper if host=='ps5' else tools['clang++'],'-x','hip','-B'+str(pathlib.Path(tools['ld.lld']).resolve().parent),'--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-D__HIP_PLATFORM_AMD__','-std=c++17','-O2','-fno-fast-math','-c',out/src.name,'--hip-path='+str(stage),'-I'+str(stage/'include'),'-include','__clang_hip_runtime_wrapper.h','--rocm-device-lib-path='+str(libs),'--hip-device-lib=ockl-workitem-only.bc','--hip-device-lib=oclc_abi_version_500.bc','--hip-device-lib=oclc_isa_version_1013.bc','-o',obj]
 if host=='ps5':cmd+=['-Xarch_device','-fno-short-wchar']
 driver=run(host+'-driver',cmd+['-###']);assert 'clang version 18.1.8' in driver.stderr
 require_device_linker(driver.stderr,tools['ld.lld'])
 # Bind the full actually included host and device header closure before compilation.
 for side in ('host','device'):
  dep=out/(host+'-'+side+'.d');dep_cmd=cmd[:];oi=dep_cmd.index('-o');del dep_cmd[oi:oi+2];dep_cmd+=['--offload-'+side+'-only','-M','-MF',dep]
  dq=run(host+'-'+side+'-dependencies',dep_cmd);assert dq.returncode==0,dq.stderr
  words=shlex.split(dep.read_text().replace('\\\n',' '));assert words[0].endswith(':')
  for word in words[1:]:
   hp=pathlib.Path(word);hp=hp if hp.is_absolute() else ROOT/hp;inputs[str(hp.resolve())]=sha(hp)

 q=run(host+'-compile',cmd)
 assert q.returncode==0,q.stderr
 bundle=section(obj.read_bytes(),b'.hip_fatbin');(d/'reduction.bundle').write_bytes(bundle);magic=b'__CLANG_OFFLOAD_BUNDLE__';assert bundle.startswith(magic)
 at=len(magic)+8;found=[]
 for _ in range(struct.unpack_from('<Q',bundle,len(magic))[0]):
  offset,size,n=struct.unpack_from('<QQQ',bundle,at);at+=24;ident=bundle[at:at+n].decode();at+=n
  if ident=='hipv4-amdgcn-amd-amdhsa--gfx1013':found.append(bundle[offset:offset+size])
 assert len(found)==1;code=d/'reduction.hsaco';code.write_bytes(found[0])
 for label,args in [('metadata',[tools['llvm-readobj'],'--notes',code]),('isa',[tools['llvm-objdump'],'-d','--mcpu=gfx1013',code]),('symbols',[tools['llvm-nm'],'--undefined-only',obj]),('host',[tools['llvm-objdump'],'-dr',obj]),('device-symbols',[tools['llvm-nm'],'--undefined-only',code])]:
  sq=run(host+'-'+label,args);assert sq.returncode==0,sq.stderr
 observed=audit_object(code.read_bytes(),(out/(host+'-metadata.stdout')).read_text(),(out/(host+'-isa.stdout')).read_text(),('_Z12block_reducePiPmS_m',))
 (d/'observed.json').write_text(json.dumps(observed,indent=2)+'\n')
 assert not (out/(host+'-device-symbols.stdout')).read_text().strip()
assert all(sha(pathlib.Path(p))==h for p,h in inputs.items())
assert all(sha(pathlib.Path(p))==h for p,h in sources.items())
assert all(sha(pathlib.Path(v['path']))==v['sha256'] for v in tool_hashes.values())
r={'classification':'bound-upstream-reduction-compiler-proof','upstream_source':json.loads((src.parent/'source.json').read_text()),'companion_scope':'Local width-correct host-only popcount overloads; not upstream header. Exact sample unchanged.','sdk_root':str(sdk),'sdk_revision':lock['sdk']['revision'],'wrapper_inputs':wrapper_inputs,'commands':commands,'sources':sources,'inputs':inputs,'tools':tool_hashes,'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'runtime_or_hardware_executed':False}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(sha(out/'receipt.json'));print([(c['label'],c['exit_code']) for c in commands])


