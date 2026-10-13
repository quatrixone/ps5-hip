#!/usr/bin/env python3
"""Bind declaration-only companion to the unchanged upstream compiler object."""
if not __debug__:raise RuntimeError('optimized Python forbidden')
import pathlib,sys,json,hashlib,subprocess,shutil,os,shlex,argparse
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from probe_hip_routes import sha
from fat_binary_fixture_test import section
p=argparse.ArgumentParser();p.add_argument('--compiler-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();C=a.compiler_root.resolve();P=C/'results/reduction-compiler-v3';O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
assert sha(P/'receipt.json')=='8cfc7008a5d93abbd09f0d88bfe754dabf06d3e79415860199dee9416696267b';r=json.loads((P/'receipt.json').read_text())
sources={str(p):sha(p) for p in [pathlib.Path(__file__),ROOT/'tests/reduction_companion.hip',ROOT/'tools/probe_hip_routes.py',ROOT/'tools/build_canaries.py',ROOT/'tests/fat_binary_fixture_test.py',ROOT/'dependencies.json',*list((ROOT/'include/ps5hip').glob('*.h'))]}
inputs={str(P/'receipt.json'):sha(P/'receipt.json'),str(C/'tests/reduction_compiler_test.py'):sha(C/'tests/reduction_compiler_test.py')};inputs.update(r['inputs']);inputs.update(r['sources']);tools=r['tools'];assert all(sha(pathlib.Path(x['path']))==x['sha256'] for x in tools.values());commands=[]
def run(label,args):
 q=subprocess.run(list(map(str,args)),cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,LLVM_CONFIG=tools['llvm-config']['path']));(O/(label+'.stdout')).write_text(q.stdout);(O/(label+'.stderr')).write_text(q.stderr);commands.append({'label':label,'argv':list(map(str,args)),'exit':q.returncode});(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert q.returncode==0,(label,q.stderr[-3000:]);return q.stdout
run('authority',['python3',C/'tests/reduction_compiler_test.py','--compiler',P])
for host in ('linux','ps5'):
 d=O/host;d.mkdir();old=P/host/'reduction.o';inputs[str(old)]=sha(old);cmd=next(x['argv'][:] for x in r['commands'] if x['label']==host+'-compile');cmd[cmd.index('-c')+1]=str(ROOT/'tests/reduction_companion.hip');cmd[cmd.index('-o')+1]=str(d/'companion.o');cmd+=['--offload-host-only','-I'+str(ROOT/'include')]
 dep=d/'headers.d';dc=cmd[:];i=dc.index('-o');del dc[i:i+2];dc+=['-M','-MF',str(dep)];run(host+'-dependencies',dc)
 words=shlex.split(dep.read_text().replace('\\\n',' '));assert words[0].endswith(':')
 for word in words[1:]:
  hp=pathlib.Path(word);hp=hp if hp.is_absolute() else ROOT/hp;inputs[str(hp.resolve())]=sha(hp)
 run(host+'-compile',cmd);names=run(host+'-symbols',[tools['llvm-nm']['path'],d/'companion.o']);sections=run(host+'-sections',[tools['llvm-readobj']['path'],'--sections',d/'companion.o'])
 assert '.hip_fatbin' not in sections and '.hipFatBinSegment' not in sections and '__hip_module_' not in names and '__hipRegister' not in names
 assert ' U _Z27__device_stub__block_reducePiPmS_m' in names and ' T ps5hip_reduction_companion_main' in names
 run(host+'-link',[tools['ld.lld']['path'],'-r',old,d/'companion.o','-o',d/'combined.o']);linked=run(host+'-combined-symbols',[tools['llvm-nm']['path'],d/'combined.o']);run(host+'-combined-disassembly',[tools['llvm-objdump']['path'],'-dr',d/'combined.o'])
 assert linked.count(' t __hip_module_ctor\n')==1 and linked.count(' t __hip_module_dtor\n')==1 and ' U _Z27__device_stub__block_reducePiPmS_m' not in linked
 assert linked.count(' T _Z27__device_stub__block_reducePiPmS_m\n')==1
 for name in (b'.hip_fatbin',b'.hipFatBinSegment'):assert section(old.read_bytes(),name)==section((d/'combined.o').read_bytes(),name)
assert all(sha(pathlib.Path(p))==h for p,h in sources.items());assert all(sha(pathlib.Path(p))==h for p,h in inputs.items());assert all(sha(pathlib.Path(x['path']))==x['sha256'] for x in tools.values())
v={'classification':'bound-declaration-only-reduction-companion-compiler','original_authority':{'path':str(P/'receipt.json'),'sha256':sha(P/'receipt.json')},'sources':sources,'inputs':inputs,'tools':tools,'commands':commands,'artifacts':{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},'native_linked':False,'hardware_executed':False,'limits':['Relocatable host link only; original one module/stub/device bytes retained.','Companion has no kernel definition or independent registration.']};(O/'receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(sha(O/'receipt.json'))
