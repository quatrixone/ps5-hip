"""Apply bounded per-kernel launch resources after embedded-module preparation."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_public_hip_memory import digest

ROOT = Path(__file__).resolve().parents[1]


def specification():
    spec = json.loads((ROOT/'ports/clr/ps5-public-resources.json').read_text())
    if digest(ROOT/'ports/clr/ps5-public-resources.patch') != spec['patch_sha256']:
        raise ValueError('public resource patch changed')
    return spec


def prepared(source):
    return all(digest(source/name) == row['after'] for name,row in specification()['files'].items())


def prepare(source, reverse=False):
    spec = specification()
    before,after = ('after','before') if reverse else ('before','after')
    if all(digest(source/name) == row[after] for name,row in spec['files'].items()):
        return
    if any(digest(source/name) != row[before] for name,row in spec['files'].items()):
        raise ValueError('public resource source prerequisites changed')
    command = ['patch','--batch','--fuzz=0','-p1','-i',str(ROOT/'ports/clr/ps5-public-resources.patch')]
    if reverse:
        command.append('-R')
    subprocess.run(command,cwd=source,check=True,capture_output=True)
    if any(digest(source/name) != row[after] for name,row in spec['files'].items()):
        raise ValueError('public resource source output differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--reverse',action='store_true')
    args = parser.parse_args()
    prepare(args.source.resolve(),args.reverse)
