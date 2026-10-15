"""Compose bounded file-backed public HIP AOT admission on prepared source."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_public_hip_memory import digest
from prepare_public_hip_module import prepare as prepare_errors

ROOT = Path(__file__).resolve().parents[1]


def prepare(source):
    spec = json.loads((ROOT / 'ports/clr/ps5-public-module-aot.json').read_text())
    patch = ROOT / 'ports/clr/ps5-public-module-aot.patch'
    if digest(patch) != spec['patch_sha256']:
        raise ValueError('AOT patch changed')
    if all(digest(source / name) == row['after'] for name, row in spec['files'].items()):
        return
    prepare_errors(source)
    if any(digest(source / name) != row['before'] for name, row in spec['files'].items()):
        raise ValueError('AOT source prerequisites changed')
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                   cwd=source, check=True, capture_output=True)
    if any(digest(source / name) != row['after'] for name, row in spec['files'].items()):
        raise ValueError('AOT source output differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
