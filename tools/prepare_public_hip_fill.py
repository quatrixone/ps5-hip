"""Apply contiguous public fill admission and synchronous error propagation."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_public_hip_copy import prepare as prepare_copy, PATCH as COPY_PATCH
from prepare_public_hip_memory import digest, SVM
ROOT = Path(__file__).resolve().parents[1]
def prepare(source):
    spec = json.loads((ROOT / 'ports/clr/ps5-public-fill.json').read_text())
    patch = ROOT / 'ports/clr/ps5-public-fill.patch'
    target = source / 'hipamd/src/hip_memory.cpp'
    if digest(patch) != spec['patch_sha256']:
        raise ValueError('fill patch changed')
    if digest(source / 'rocclr/platform/memory.cpp') != SVM or digest(ROOT / 'ports/clr/ps5-public-copy.patch') != COPY_PATCH:
        raise ValueError('fill copy or SVM prerequisite changed')
    if digest(target) == spec['after']:
        return
    prepare_copy(source)
    if digest(target) != spec['before']:
        raise ValueError('fill prerequisite changed')
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)], cwd=source, check=True, capture_output=True)
    if digest(target) != spec['after']:
        raise ValueError('fill output differs')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
