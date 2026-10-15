"""Apply the reviewed public HIP free guards over the already-patched SVM tree."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BEFORE = '833ff2bf952d77cdea083751022d4b9989c0d5e17e566213c95e92369674cb4b'
AFTER = '897238c92852724fa84f126090a10b4129c3a8848a85243383af3ee85927f75f'
SVM = '7cb0364bb313e3fb60292511dbeca5dae6375a6aa5e4d3d0429c2b8633b4f215'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source):
    patch = ROOT / 'ports/clr/ps5-public-memory.patch'
    if digest(patch) != 'e12744ddeab27cc4ea390c7eca063ac02277724892ff5838a94bc10c3c41b23a':
        raise ValueError('reviewed memory patch changed')
    if digest(source / 'rocclr/platform/memory.cpp') != SVM:
        raise ValueError('SVM ownership source differs')
    target = source / 'hipamd/src/hip_memory.cpp'
    current = digest(target)
    if current == BEFORE:
        marker = '--- a/hipamd/src/hip_memory.cpp\n'
        hip_patch = marker + patch.read_text().split(marker)[1]
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1'], cwd=source,
                       input=hip_patch, text=True, check=True, capture_output=True)
    elif current != AFTER:
        raise ValueError('public HIP memory source differs')
    if digest(target) != AFTER or digest(source / 'rocclr/platform/memory.cpp') != SVM:
        raise ValueError('public HIP memory patch result differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
