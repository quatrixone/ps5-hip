"""Reject incomplete saved proof inventories without modifying the original proof."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--proof', type=Path, required=True)
proof = parser.parse_args().proof.resolve()
original = json.loads((proof / 'receipt.json').read_text())
mutations = []
for group in ('sources', 'inputs', 'artifacts', 'tools'):
    changed = copy.deepcopy(original)
    changed[group] = {}
    mutations.append((group + '-empty', changed))
for group, key in [('sources', 'tools/hip_vector_title.cpp'),
                   ('sources', 'tests/reduction_title_test.py'),
                   ('artifacts', 'production.log'), ('artifacts', 'commands.json'),
                   ('artifacts', 'run-normal.stdout'), ('artifacts', 'title'),
                   ('tools', 'clang++'), ('tools', 'ld.lld'),
                   ('inputs', next(iter(original['inputs'])))]:
    changed = copy.deepcopy(original)
    del changed[group][key]
    mutations.append((group + '-missing-' + key, changed))
changed = copy.deepcopy(original)
changed['commands'][0]['exit'] = 1
mutations.append(('failed-command', changed))
changed = copy.deepcopy(original)
changed['commands'].append(copy.deepcopy(changed['commands'][-1]))
mutations.append(('duplicate-command', changed))
with tempfile.TemporaryDirectory(prefix='reduction-title-inventory-') as directory:
    target = Path(directory)
    for item in proof.iterdir():
        if item.name != 'receipt.json':
            (target / item.name).symlink_to(item, target_is_directory=item.is_dir())
    for label, receipt in mutations:
        (target / 'receipt.json').write_text(json.dumps(receipt))
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('reduction_title_saved_test.py')),
                                 '--check', str(target)], capture_output=True, text=True, timeout=60)
        assert result.returncode != 0 and 'AssertionError' in result.stderr, (label, result.stdout, result.stderr)
assert json.loads((proof / 'receipt.json').read_text()) == original
print(f'PASS {len(mutations)} incomplete inventory and command rejections')
