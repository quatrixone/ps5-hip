"""Enumerate public HIP functions from the validated PS5 AOT kernel table."""
import argparse
from pathlib import Path
import subprocess
from prepare_public_hip_memory import digest

ROOT = Path(__file__).resolve().parents[1]
BEFORE = '0ed766e1836cb69ffc0ee0c43b94aa4b383f7d2ad374c868c2e1c28ccd5e38e4'
AFTER = '0ef5d74277109005eb60e0e80890d99bd6ccd710fc4afae717c775c985fcca93'
PATCH = 'db19181078377c19e79b96cda26adfe7351310d081c6dc958f54bfe6a9c83f78'


def prepare(source):
    patch = ROOT / 'ports/clr/ps5-public-module-symbols.patch'
    target = source / 'rocclr/device/devprogram.cpp'
    if digest(patch) != PATCH or digest(target) not in (BEFORE, AFTER):
        raise ValueError('module symbols patch or prepared AOT source changed')
    if digest(target) == BEFORE:
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                       cwd=source, check=True, capture_output=True)
    if digest(target) != AFTER:
        raise ValueError('public HIP module symbols patch result differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
