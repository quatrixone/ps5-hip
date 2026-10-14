"""Compile genuine PS5 memory prerequisites; record unsupported public HIP closure."""
from pathlib import Path
import argparse, hashlib, json, shlex, subprocess

p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];B=a.reference_root.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=B/'results/clr-adapter-native-compile-v1/receipt.json';old=json.loads(receipt.read_text())
flags=old['commands'][0]['argv'];flags=flags[:flags.index('-M')]
flags=[v.replace(str(B/'.local/g5-clr-ps5-adapter'),str(R)) for v in flags]
inputs={str(receipt):H(receipt),str(Path(__file__).resolve()):H(__file__),str(Path(flags[0])):H(flags[0])};commands=[]
def run(label,argv):
 q=subprocess.run(argv,capture_output=True,text=True,timeout=120)
 (O/(label+'.stdout')).write_text(q.stdout);(O/(label+'.stderr')).write_text(q.stderr)
 commands.append(dict(label=label,argv=argv,exit_code=q.returncode));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 return q.returncode
for label,source in [('device',R/'ports/clr/ps5-device.cpp'),('memory',R/'results/clr-source-v7/source/rocclr/platform/memory.cpp'),('hip-memory',R/'results/clr-source-v7/source/hipamd/src/hip_memory.cpp')]:
 inputs[str(source)]=H(source);dep=O/(label+'.d')
 if run(label+'-headers',flags+['-M',str(source),'-MF',str(dep)]):continue
 for f in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):inputs[str(Path(f).resolve())]=H(f)
 run(label,flags+['-c',str(source),'-o',str(O/(label+'.o'))])
assert all(H(k)==v for k,v in inputs.items()),'input drift'
report=dict(classification='native-public-memory-compile-probe',inputs=inputs,commands=commands,artifacts={f.name:H(f) for f in O.iterdir() if f.is_file()},native_linked=False,native_executed=False,public_hip_qualified=False)
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(H(O/'receipt.json'))
