"""Apply exact PS5 public module extraction error handling to prepared HIP source."""
import argparse
from pathlib import Path
import subprocess
from prepare_public_hip_memory import digest

ROOT = Path(__file__).resolve().parents[1]
BEFORE = 'b0b77db5df71f03220a2a546ad9d9f43a8f7aafa72826c9a0fdf8636dc82e87c'
AFTER = 'd685dae2e23bfe08c96262bfac722e2941b3b92d9c67972cebb3530058d44a3a'
PATCH = 'e2613f9971df6028d4284a0ae7af1ffb47329cf0f80fd4f73e8d74fc8c351b5b'


def prepare(source):
    patch = ROOT / 'ports/clr/ps5-public-module-errors.patch'
    target = source / 'hipamd/src/hip_fatbin.cpp'
    if digest(patch) != PATCH or digest(target) not in (BEFORE, AFTER):
        raise ValueError('module patch or prepared HIP source changed')
    if digest(target) == BEFORE:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                       cwd=source, check=True, capture_output=True)
    if digest(target) != AFTER:
        raise ValueError('public HIP module patch result differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
