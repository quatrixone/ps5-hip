"""Compose the reviewed SVM patch over the frozen native public HIP source."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


checkout = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--bootstrap-source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source = args.bootstrap_source.resolve()
out = args.output.resolve()
patch = checkout / 'ports/clr/ps5-public-memory.patch'
before = {
    'rocclr/platform/memory.cpp': 'fe9384d6a8a7e763238164ecbf9e0c7cf0b42e0d09f511ec5a975395723e40b8',
    'hipamd/src/hip_memory.cpp': '833ff2bf952d77cdea083751022d4b9989c0d5e17e566213c95e92369674cb4b',
}
after = {
    'rocclr/platform/memory.cpp': '7cb0364bb313e3fb60292511dbeca5dae6375a6aa5e4d3d0429c2b8633b4f215',
    'hipamd/src/hip_memory.cpp': '897238c92852724fa84f126090a10b4129c3a8848a85243383af3ee85927f75f',
}
assert sha(patch) == 'e12744ddeab27cc4ea390c7eca063ac02277724892ff5838a94bc10c3c41b23a'
for name, digest in before.items():
    assert sha(source / name) == digest, name
assert sha(source / 'hipamd/src/hip_prof_api.h') == '57a603b91235d4cf2e9d1683190316952cbe3b3b9bc57339e9dfdaa5f70094ed'
assert not out.exists(), 'Use a fresh output directory'
shutil.copytree(source, out)
subprocess.run(['git', 'apply', '--check', str(patch)], cwd=out, check=True)
subprocess.run(['git', 'apply', str(patch)], cwd=out, check=True)
for name, digest in after.items():
    assert sha(out / name) == digest, name
unchanged = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file() and str(p.relative_to(source)) not in before}
for name, digest in unchanged.items():
    assert sha(out / name) == digest, name
receipt = {'classification': 'exact-native-public-hip-svm-composition',
           'bootstrap_preparation_sha256': sha(source / 'public-hip-preparation.json'),
           'patch_sha256': sha(patch), 'changed': {n: {'before': before[n], 'after': after[n]} for n in before},
           'unchanged_file_count': len(unchanged), 'unchanged_sha256': hashlib.sha256(json.dumps(unchanged, sort_keys=True).encode()).hexdigest(),
           'native_executed': False}
(out / 'native-memory-composition.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(sha(out / 'native-memory-composition.json'))
