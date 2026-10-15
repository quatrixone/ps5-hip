"""Apply bounded compiler embedded-image admission to prepared public HIP source."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_public_hip_memory import digest

ROOT = Path(__file__).resolve().parents[1]


def prepared(source):
    spec = json.loads((ROOT / "ports/clr/ps5-public-embedded.json").read_text())
    if digest(ROOT / "ports/clr/ps5-public-embedded.patch") != spec["patch_sha256"]:
        raise ValueError("embedded module patch changed")
    previous = json.loads((ROOT / "ports/clr/ps5-public-static-module.json").read_text())
    expected = {name: row for name, row in previous["files"].items() if name not in spec["files"]}
    expected.update(spec["files"])
    return all(digest(source / name) == row["after"] for name, row in expected.items())


def prepare(source):
    spec = json.loads((ROOT / 'ports/clr/ps5-public-embedded.json').read_text())
    patch = ROOT / 'ports/clr/ps5-public-embedded.patch'
    if digest(patch) != spec['patch_sha256']:
        raise ValueError('embedded module patch changed')
    if all(digest(source / name) == row['after'] for name, row in spec['files'].items()):
        return
    if any(digest(source / name) != row['before'] for name, row in spec['files'].items()):
        raise ValueError('embedded module source prerequisites changed')
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)],
                   cwd=source, check=True, capture_output=True)
    if any(digest(source / name) != row['after'] for name, row in spec['files'].items()):
        raise ValueError('embedded module source output differs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
