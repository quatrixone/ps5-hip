"""Compile the complete real upstream HIP memory unit; do not supply fake symbols."""
from pathlib import Path
import argparse, hashlib, json, shlex, shutil, subprocess
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];O=a.output.resolve();O.mkdir(parents=True,exist_ok=False);H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
baseline=R/'results/svm-native-v1/receipt.json';old=json.loads(baseline.read_text());argv=next(x['argv'] for x in old['commands'] if x['label']=='hip-memory')
flags=[v.replace('results/clr-source-v7/','results/clr-source-v8/') for v in argv[:argv.index('-c')]]
source=R/'results/clr-source-v8/source/hipamd/src/hip_memory.cpp';prep=R/'results/clr-source-v8/preparation.json'
inputs={str(baseline):H(baseline),str(Path(__file__).resolve()):H(__file__),str(prep):H(prep),str(source):H(source)};commands=[]
def run(label,args):
 q=subprocess.run(args,capture_output=True,text=True,timeout=120);(O/(label+'.stdout')).write_text(q.stdout);(O/(label+'.stderr')).write_text(q.stderr)
 commands.append(dict(label=label,argv=args,exit_code=q.returncode));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');return q.returncode
dep=O/'hip-memory.d';assert run('headers',flags+['-M',str(source),'-MF',str(dep)])==0
for f in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):inputs[str(Path(f).resolve())]=H(f)
inputs[str(Path(flags[0]).resolve())]=H(flags[0]);compiler=shutil.which('clang++-18');inputs[str(Path(compiler).resolve())]=H(compiler)
rc=run('compile',flags+['-c',str(source),'-o',str(O/'hip-memory.o')])
if rc==0:
 nm=shutil.which('llvm-nm-18');inputs[str(Path(nm).resolve())]=H(nm);assert run('undefined-symbols',[nm,'--undefined-only','--demangle',str(O/'hip-memory.o')])==0
assert all(H(k)==v for k,v in inputs.items()),'input drift'
report=dict(classification='real-upstream-hip-memory-native-compile-only',inputs=inputs,commands=commands,artifacts={f.name:H(f) for f in O.iterdir() if f.is_file()},compile_passed=rc==0,native_linked=False,native_executed=False,public_hip_qualified=False)
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(H(O/'receipt.json'));raise SystemExit(rc)
