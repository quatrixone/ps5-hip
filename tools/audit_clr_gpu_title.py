if not __debug__: raise RuntimeError("optimized Python disables required native admission assertions")
from pathlib import Path
import hashlib,json,struct,importlib.util,shlex
C=Path(__file__).resolve().parents[1];R=C.parents[1];B=C/'results/native-title-v1';V=C/'results/native-conversion-v1';O=C/'results/native-admission-v1';O.mkdir(exist_ok=False);h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());inputs={str(Path(__file__)):h(Path(__file__))}
for proof in [B/'receipt.json',V/'receipt.json']:
 inputs[str(proof)]=h(proof)
 for n,z in read(proof)['inputs'].items():assert h(n)==z,n;inputs[n]=z
 for n,z in read(proof)['artifacts'].items():assert h(proof.parent/n)==z,n;inputs[str(proof.parent/n)]=z
class Elf:
 def __init__(self,p):
  self.data=p.read_bytes();d=self.data;off=struct.unpack_from('<Q',d,40)[0];self.sections=[struct.unpack_from('<IIQQQQIIQQ',d,off+i*64) for i in range(struct.unpack_from('<H',d,60)[0])];po=struct.unpack_from('<Q',d,32)[0];self.ph=[struct.unpack_from('<IIQQQQQQ',d,po+i*56) for i in range(struct.unpack_from('<H',d,56)[0])]
 def offset(self,a):
  s=next(p for p in self.ph if p[0]==1 and p[3]<=a<p[3]+p[5]);return s[2]+a-s[3]
 def symbols(self,typ):
  s=next(s for s in self.sections if s[1]==typ);st=self.sections[s[6]];strings=self.data[st[4]:st[4]+st[5]];out={}
  for off in range(s[4],s[4]+s[5],24):
   n,i,o,idx,v,size=struct.unpack_from('<IBBHQQ',self.data,off);out[strings[n:].split(b'\0',1)[0].decode()]=(i,o,idx,v,size)
  return out
raw=Elf(B/'title.elf');symbols=raw.symbols(2);dyn=raw.symbols(11);undefined={n for n,v in dyn.items() if n and v[2]==0};assert not any(n.startswith(('hip','__hip')) for n in undefined)
assert {n for n in undefined if n.startswith('sceAgc')}=={'sceAgcInit','sceAgcDriverSubmitDcb','sceAgcSuspendPoint'}
helper=C/'tests/native_import_libraries_test.py';inputs[str(helper)]=h(helper);spec=importlib.util.spec_from_file_location('read_native_imports',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);modules,libraries,nids=m.read_imports(V/'eboot.elf');assert len(nids)==len(undefined)
sdk=R/'.deps/sdk/target/lib';providers={}
for name in ['libkernel','libSceLibcInternal','libSceNet','libSceSystemService','libSceAgc','libSceAgcDriver']:
 p=(R/'.local/g4-multi-reduction-integration/build/native/gfx1013-hip_multi_query-integration-v2' if name.startswith('libSceAgc') else sdk)/(name+'.so');inputs[str(p)]=h(p);providers[name]={n for n,v in Elf(p).symbols(11).items() if n and v[2]!=0}
mapped={}
for name in undefined:
 provider=nids[m.nid(name)];assert name in providers[provider[0]],(name,provider)
 expectedlib='libSceShellCoreUtil' if name=='sceShellCoreUtilExitApp' else provider[0];assert provider[1]==expectedlib,(name,provider);mapped[name]=provider
# Real init/fini arrays contain the authentic upstream OS constructor/destructor.
arrays={}
raw_relocations={struct.unpack_from("<Q",raw.data,o)[0]:struct.unpack_from("<QQq",raw.data,o)[1:] for sec in raw.sections if sec[1]==4 for o in range(sec[4],sec[4]+sec[5],24)}
for name,entry in [('init','_ZN3amdL4initEv'),('fini','_ZN3amdL6__exitEv')]:
 lo=symbols['__'+name+'_array_start'][3];hi=symbols['__'+name+'_array_end'][3];values=[]
 for address in range(lo,hi,8):
  info,addend=raw_relocations[address];assert info==8 and struct.unpack_from('<Q',raw.data,raw.offset(address))[0] in (0,addend);values.append(addend)
 assert values.count(symbols[entry][3])==1;arrays[name]=values
native=Elf(V/'eboot.elf');dh=next(p for p in native.ph if p[0]==2);tags=dict(struct.unpack_from('<QQ',native.data,i) for i in range(dh[2],dh[2]+dh[5],16));relocations=set()
for ptr,size in [(7,8),(23,2)]:
 if ptr in tags:
  off=native.offset(tags[ptr]);relocations.update(struct.unpack_from('<Q',native.data,p)[0] for p in range(off,off+tags[size],24))
slots=read(V/'receipt.json')['raw_optional_hooks_zero']
for name,addr in slots.items():assert addr not in relocations and struct.unpack_from('<Q',native.data,native.offset(addr))[0]==0,name
# All original and replaced source/header dependencies must match the combined port.
base_manifest=C/'ports/clr/manifest.json';adapter_manifest=C/'ports/clr/adapter-manifest.json'
inputs[str(base_manifest)]=h(base_manifest);inputs[str(adapter_manifest)]=h(adapter_manifest)
adapted=read(base_manifest)['files'];adapted.update(read(adapter_manifest)['files'])
proof=read(B/'receipt.json');P=R/'.local/g4-clr-cpu-lifetime/results/native-title-lifetime-v1';Q=R/'results/clr-adapter-native-compile-v1'
assert proof['reused_cpu_units']==[] and proof['reused_aot_objects']==[]
paths=list(B.glob('*.d'));assert len(paths)==proof['new_units']==46
scans=[]
for p in paths:
 assert str(p) in inputs and h(p)==inputs[str(p)],p
 deps=shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]);scans.append(str(p))
 for name in deps:
  for suffix,entry in adapted.items():
   if name.endswith('/'+suffix):assert h(name)==entry['adapted_sha256'],(p,name)
# The bound genuine HSACO is embedded as exact read-only bytes in the actual image.
code=symbols['ps5hip_clr_code'];size=symbols['ps5hip_clr_code_size'];assert code[2]!=0 and size[2]!=0
code_path=R/'.local/g4-native-workers/results/workers-compiler-v2/ps5/threads.hsaco';expected=code_path.read_bytes()
assert h(code_path)=='0484405bc2a5247de40c41105a170026be1ea1802d60930fb0df56099dfc3dc0'
assert raw.data[raw.offset(code[3]):raw.offset(code[3])+len(expected)]==expected
assert struct.unpack_from('<Q',raw.data,raw.offset(size[3]))[0]==len(expected)
assert native.data[native.offset(code[3]):native.offset(code[3])+len(expected)]==expected
assert struct.unpack_from('<Q',native.data,native.offset(size[3]))[0]==len(expected)
assert not any(p[0]==1 and p[1]&2 and max(p[3]//0x4000,code[3]//0x4000)<=min((p[3]+p[6]-1)//0x4000,(code[3]+len(expected)-1)//0x4000) for p in native.ph)
assert not any(p[0]==1 and p[1]&2 and max(p[3]//0x4000,code[3]//0x4000)<=min((p[3]+p[6]-1)//0x4000,(code[3]+len(expected)-1)//0x4000) for p in raw.ph)
for n,z in inputs.items():assert h(n)==z,n
r=dict(classification='pass-offline-native-gpu-adapter-admission',inputs=inputs,imports=mapped,import_count=len(mapped),actual_upstream_init_fini_arrays=arrays,converted_optional_slots_without_relocation=slots,dependency_scans=scans,genuine_embedded_code_sha256=h(code_path),real_agc_imports=True,limits=['Author admission only; independent native startup/provider and source review still required.','No GPU title execution/package or public HIP initialization.','Optional loaders unavailable; executable CPU mappings explicitly rejected.'])
(O/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(h(O/'receipt.json'))
