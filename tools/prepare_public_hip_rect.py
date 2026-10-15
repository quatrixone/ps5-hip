"""Apply checked rectangular D2D admission and synchronous error propagation."""
import argparse
import json
from pathlib import Path
import subprocess
from prepare_public_hip_fill import prepare as prepare_fill
from prepare_public_hip_memory import digest, SVM
from prepare_public_hip_copy import PATCH as COPY_PATCH
ROOT = Path(__file__).resolve().parents[1]
def prepare(source):
    spec = json.loads((ROOT / 'ports/clr/ps5-public-rect.json').read_text())
    fill = json.loads((ROOT / 'ports/clr/ps5-public-fill.json').read_text())
    patch = ROOT / 'ports/clr/ps5-public-rect.patch'
    target = source / 'hipamd/src/hip_memory.cpp'
    if digest(patch) != spec['patch_sha256'] or digest(ROOT / 'ports/clr/ps5-public-fill.patch') != fill['patch_sha256'] or spec['before'] != fill['after']:
        raise ValueError('rectangular patch or prerequisite changed')
    graph = source / 'hipamd/src/hip_graph.cpp'
    if digest(source / 'rocclr/platform/memory.cpp') != SVM or digest(ROOT / 'ports/clr/ps5-public-copy.patch') != COPY_PATCH:
        raise ValueError('rectangular SVM or copy prerequisite changed')
    if digest(target) == spec['after']:
        if digest(graph) != spec['graph_after']:
            # A later, independently hash-bound stage repairs capture cleanup.
            # Accept only its complete output, never a lone matching graph file.
            capture = json.loads((ROOT / 'ports/clr/ps5-public-capture.json').read_text())
            capture_patch = ROOT / 'ports/clr/ps5-public-capture.patch'
            if (digest(capture_patch) != capture['patch_sha256'] or
                capture['files']['hipamd/src/hip_graph.cpp']['before'] != spec['graph_after'] or
                any(digest(source / name) != row['after']
                    for name, row in capture['files'].items())):
                raise ValueError('rectangular graph output differs')
        return
    if digest(graph) != spec['graph_before']:
        raise ValueError('rectangular graph prerequisite changed')
    prepare_fill(source)
    if digest(target) != spec['before']:
        raise ValueError('rectangular prerequisite changed')
    subprocess.run(['patch', '--batch', '--fuzz=0', '-p1', '-i', str(patch)], cwd=source, check=True, capture_output=True)
    if digest(target) != spec['after'] or digest(graph) != spec['graph_after']:
        raise ValueError('rectangular output differs')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    prepare(parser.parse_args().source.resolve())
