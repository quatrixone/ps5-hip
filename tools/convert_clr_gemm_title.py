if not __debug__: raise RuntimeError("Validation requires ordinary Python")
from pathlib import Path
import hashlib,json,subprocess,re,struct,ast
C=Path(__file__).resolve().parents[1];R=C.parent/'ps5-hip';B=C/'results/native-title-v2';O=C/'results/native-conversion-v2';O.mkdir(exist_ok=False);N=R/'.local/g4-multi-reduction-integration/build/native/gfx1013-hip_multi_query-integration-v2';h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());inputs={};commands=[]
for proof in [B/'receipt.json',N/'receipt.json']:
 inputs[str(proof)]=h(proof)
b=read(B/'receipt.json');n=read(N/'receipt.json')
for name,z in b['inputs'].items():assert h(name)==z,name;inputs[name]=z
for name,z in b['artifacts'].items():assert h(B/name)==z,name;inputs[str(B/name)]=z
for name in ['ps5-native-tool','libc-builder']:
 assert h(N/name)==n['raw_artifacts'][name];inputs[str(N/name)]=h(N/name)
for name,z in n['raw_artifacts'].items():
 if name.startswith('native-tools/'):
  assert h(N/name)==z;inputs[str(N/name)]=z
inputs[str(Path(__file__))]=h(Path(__file__))
def run(label,cmd):
 z=subprocess.run(cmd,capture_output=True,text=True,timeout=60);(O/(label+'.stdout')).write_text(z.stdout);(O/(label+'.stderr')).write_text(z.stderr);commands.append(dict(label=label,argv=cmd,exit_code=z.returncode));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert z.returncode==0,(label,z.stderr);return z.stdout
raw=(B/'title.elf').read_bytes();shoff=struct.unpack_from('<Q',raw,40)[0];num=struct.unpack_from('<H',raw,60)[0];secs=[struct.unpack_from('<IIQQQQIIQQ',raw,shoff+i*64) for i in range(num)]
def chunk(s):return raw[s[4]:s[4]+s[5]]
def symtab(s):
 strings=chunk(secs[s[6]]);out={}
 for off in range(s[4],s[4]+s[5],24):
  name,info,other,idx,value,size=struct.unpack_from('<IBBHQQ',raw,off);name=strings[name:].split(b'\0',1)[0].decode();out[name]=(info,other,idx,value,size)
 return out
symbols=symtab(next(s for s in secs if s[1]==2));dyn=symtab(next(s for s in secs if s[1]==11));relocs={struct.unpack_from('<Q',raw,o)[0] for s in secs if s[1]==4 for o in range(s[4],s[4]+s[5],24)}
def word(addr):
 s=next(s for s in secs if s[3]<=addr and addr+8<=s[3]+s[5] and s[1]!=8);return struct.unpack_from('<Q',raw,s[4]+addr-s[3])[0]
text=(B/'startup.stdout').read_text()
def body(name):
 m=re.search(r'^[0-9a-f]+ <'+re.escape(name)+r'>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',text,re.M|re.S);assert m,name;return m[1]
hooks=['__dlopen','__dlsym','__dladdr','__dlclose','__dlerror','kernel_mprotect'];slots={}
for name in hooks:
 assert symbols[name][1]&3==2 and symbols[name][2:4]==(0xfff1,0),name;assert name not in dyn
for name in ['dlopen','dlsym','dladdr','dlclose','dlerror']:
 code=body(name);m=re.search(r'cmpq\s+\$0x0,.*?# 0x([0-9a-f]+)',code);assert m,name;slot=int(m[1],16);assert word(slot)==0 and slot not in relocs;assert re.search(r'\bje\b',code) and ('# 0x'+format(slot,'x')) in code[code.index('jmpq'):];slots['__'+name]=slot
for name in ['mmap','mprotect']:
 code=body(name);s={int(v,16) for v in re.findall(r'callq\s+\*[^\n]+# 0x([0-9a-f]+)',code)};zero=[v for v in s if v not in relocs and word(v)==0];assert len(zero)==1,(name,zero);slots.setdefault('kernel_mprotect',zero[0]);assert slots['kernel_mprotect']==zero[0]
start=body('_start');order=['ps5_fp_ieee','ps5hip_install_shell_exit_probe','ps5hip_install_vector_title_finalizer','_init','main','catchReturnFromMain'];pos=[start.index('<'+x+'>') for x in order];assert pos==sorted(pos)
agc={'sceAgcInit','sceAgcDriverSubmitDcb','sceAgcSuspendPoint'}
assert {name for name in dyn if name.startswith('sceAgc')}==agc
assert all(dyn[name][2]==0 for name in agc)
assert not any(name.startswith(('hip','__hip')) for name in dyn)
sdk=R/'.deps/sdk/target/lib';converter=N/'ps5-native-tool'
run('convert',[str(converter),'link','--in',str(B/'title.elf'),'--out',str(O/'eboot.elf'),'--stub-dir',str(sdk),'--stub',str(N/'libSceAgc.so'),'--stub',str(N/'libSceAgcDriver.so'),'--file-name','eboot.elf','--import-library','libSceSystemService:sceShellCoreUtilExitApp:libSceShellCoreUtil'])
run('fself',[str(converter),'self','--sign','--in',str(O/'eboot.elf'),'--out',str(O/'eboot.bin')])
run('extract',[str(converter),'self','--extract','--file',str(O/'eboot.bin'),'--out',str(O/'extracted.elf')])
helper=C/'tools/native_vector_title.py';inputs[str(helper)]=h(helper)
node=next(n for n in ast.parse(helper.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='audit_roundtrip')
namespace={'struct':struct,'hashlib':hashlib};exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),namespace)
roundtrip=namespace['audit_roundtrip']((O/'eboot.elf').read_bytes(),(O/'extracted.elf').read_bytes())
# Converted load ranges retain each admitted raw optional GOT word as zero.
converted=(O/'eboot.elf').read_bytes();phoff=struct.unpack_from('<Q',converted,32)[0];phnum=struct.unpack_from('<H',converted,56)[0];loads=[struct.unpack_from('<IIQQQQQQ',converted,phoff+i*56) for i in range(phnum)]
for name,addr in slots.items():
 segment=next(v for v in loads if v[0]==1 and v[3]<=addr and addr+8<=v[3]+v[5]);assert struct.unpack_from('<Q',converted,segment[2]+addr-segment[3])[0]==0,name
for name,z in inputs.items():assert h(name)==z,name
out=dict(classification='converted-offline-native-clr-gemm-title',inputs=inputs,commands=commands,raw_optional_hooks_zero=slots,converted_got_words_zero=True,fself_roundtrip=roundtrip,raw_startup_order=order,artifacts={p.name:h(p) for p in O.iterdir() if p.is_file()},limits=['Converted native import identities/provider mapping and independent startup audit remain pending; zero GOT bytes and exact FSELF roundtrip checked.','No package/native execution; absent loaders explicitly unsupported, CLR RWX rejected.'])
(O/'receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(h(O/'receipt.json'))
