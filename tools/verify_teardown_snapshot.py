"""Validate a preserved offline snapshot build; never rebuild or contact the console."""
import argparse
import json
import shutil
from pathlib import Path
from build_teardown_snapshot import ROOT, sha, audit


def verify(output, sdk):
    receipt = json.loads((output / 'receipt.json').read_text())
    if receipt.get('classification') != 'offline-built-imports-checked-not-executed':
        raise ValueError('snapshot build did not succeed; preserve it and use a new SNAPSHOT_BUILD_DIR')
    lock = json.loads((ROOT / 'dependencies.json').read_text())
    if (receipt['sdk_revision'] != lock['sdk']['revision'] or
            receipt['llvm_version'] != lock['llvm']['version'] or
            (sdk / '.ps5-sdk-revision').read_text().strip() != lock['sdk']['revision']):
        raise ValueError('snapshot toolchain identity changed')
    required = {'tools/teardown_snapshot.cpp', 'tools/teardown_snapshot.h',
                'tools/build_teardown_snapshot.py', 'dependencies.json'}
    if set(receipt['source_sha256']) != required:
        raise ValueError('snapshot source closure incomplete')
    for field, base in (('source_sha256', ROOT), ('sdk_inputs_sha256', sdk),
                        ('included_file_sha256', Path('/'))):
        if not receipt[field]:
            raise ValueError('empty snapshot input identity: ' + field)
        for name, expected in receipt[field].items():
            if sha(base / name) != expected:
                raise ValueError('snapshot input changed: ' + name)
    tool_names = {'llvm-config', 'llvm-readelf', 'llvm-objdump'}
    if set(receipt['tools_sha256']) != tool_names:
        raise ValueError('snapshot inspection tool closure incomplete')
    for name, expected in receipt['tools_sha256'].items():
        path = shutil.which(name + '-18')
        if path is None or sha(Path(path)) != expected:
            raise ValueError('snapshot inspection tool changed: ' + name)
    artifact = receipt['artifact']
    if artifact['file'] != 'teardown-snapshot.elf':
        raise ValueError('unexpected snapshot artifact')
    elf = output / artifact['file']
    if elf.stat().st_size != artifact['bytes'] or sha(elf) != artifact['sha256']:
        raise ValueError('snapshot artifact changed')
    if not receipt['commands'] or any(command['exit_code'] != 0 for command in receipt['commands']):
        raise ValueError('snapshot command failed')
    for command in receipt['commands']:
        if sha(output / command['log']) != command['sha256']:
            raise ValueError('snapshot command log changed')
    providers = {name: (output / (name + '.exports.log')).read_text()
                 for name in receipt['import_audit']['needed_libraries']}
    if audit((output / 'imports.log').read_text(), providers) != receipt['import_audit']:
        raise ValueError('snapshot import audit differs')
    return artifact['sha256']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    args = parser.parse_args()
    print('Verified offline snapshot:', verify(args.output.resolve(), args.sdk.resolve()))
