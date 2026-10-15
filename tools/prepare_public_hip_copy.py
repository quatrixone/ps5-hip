"""Apply the PS5 synchronous public-copy completion contract to pinned HIP source."""
import argparse
from pathlib import Path
import subprocess
from prepare_public_hip_memory import prepare as prepare_memory, digest, SVM

ROOT = Path(__file__).resolve().parents[1]
AFTER = 'c64c1cbb2686fa70e322ae6e3d2a1cb84126957516f2a349a7f7fe01ab4e8aa7'
PATCH = 'd43dd1ff0816a6c779a5b69adf646fa2e765e1799d462982dd114de257f316dd'


def prepare(source):
    patch = ROOT / 'ports/clr/ps5-public-copy.patch'
    if digest(patch) != PATCH or digest(source / 'rocclr/platform/memory.cpp') != SVM:
        raise ValueError('copy patch or SVM ownership source changed')
    target = source / 'hipamd/src/hip_memory.cpp'
    if digest(target) != AFTER:
        prepare_memory(source)
        subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                       cwd=source, check=True, capture_output=True)
    if digest(target) != AFTER:
        raise ValueError('public HIP copy patch result differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
