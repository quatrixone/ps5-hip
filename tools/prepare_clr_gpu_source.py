"""Compose the exact portable native CLR patch stack into a fresh source tree."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess
if not __debug__:
    raise RuntimeError('Validation requires ordinary Python')
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
C=Path(__file__).resolve().parents[1];R=C.parents[1];O=a.output.resolve()
assert not O.exists()
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=json.loads((C/'ports/clr/manifest.json').read_text())
adapter=json.loads((C/'ports/clr/adapter-manifest.json').read_text())
S=R/'results/g4-runtime-architecture-v1/sources'/('clr-'+base['upstream_revision'])
assert adapter['upstream_revision']==base['upstream_revision']
inputs={str(p):h(p) for p in [Path(__file__),C/'ports/clr/manifest.json',C/'ports/clr/adapter-manifest.json']}
for name,entry in base['files'].items():
    assert h(S/name)==entry['upstream_sha256'];inputs[str(S/name)]=h(S/name)
O.mkdir(parents=True);shutil.copytree(S,O/'source',ignore=shutil.ignore_patterns('.git'))
patches=['ps5-platform.patch']+adapter['patch_order'];tool=Path(shutil.which('patch')).resolve();inputs[str(tool)]=h(tool)
commands=[]
for i,name in enumerate(patches):
    patch=C/'ports/clr'/name;expected=base['patch_sha256'] if i==0 else adapter['patches'][name]
    assert h(patch)==expected;inputs[str(patch)]=expected
    argv=[str(tool),'--batch','--forward','-p1','-i',str(patch)]
    z=subprocess.run(argv,cwd=O/'source',capture_output=True,text=True,timeout=30)
    (O/(name+'.stdout')).write_text(z.stdout);(O/(name+'.stderr')).write_text(z.stderr)
    commands.append(dict(argv=argv,exit_code=z.returncode));assert z.returncode==0,z.stderr
expected=dict(base['files']);expected.update(adapter['files'])
for name,entry in expected.items():assert h(O/'source'/name)==entry['adapted_sha256'],name
assert all(h(n)==v for n,v in inputs.items())
report=dict(classification='exact-combined-native-clr-gpu-source',inputs=inputs,commands=commands,
            generated={n:h(O/'source'/n) for n in expected},patch_order=patches)
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(h(O/'receipt.json'))
