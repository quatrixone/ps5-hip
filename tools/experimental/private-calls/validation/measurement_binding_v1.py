"""Bind fresh independent measurements to the immutable historical semantics."""
import hashlib
import json
from pathlib import Path

REFERENCE_SHA256 = '33179c0a773c1072602b5c6b6842a430a9a81a4e9a1ad4eb65751f7680cf5699'
PROVENANCE_FIELDS = {'input_sha256', 'raw_files', 'comparison_old_ipra_on'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def compare_semantics(measured, reference):
    assert measured['configuration'] == reference['configuration'] == ['-enable-ipra=false', '-verify-machineinstrs']
    def profiles(document):
        rows = document['profiles']
        assert len(rows) == 8 and len({row['id'] for row in rows}) == 8
        return {row['id']: {key: value for key, value in row.items() if key not in PROVENANCE_FIELDS}
                for row in rows}
    assert profiles(measured) == profiles(reference), 'measured semantic contract differs from pinned reference'
    assert measured['contract_reason'] == reference['contract_reason']
    assert measured['wait_reason'] == reference['wait_reason']


def verify_measurement(contract):
    directory = contract.parent
    reference_path = directory / 'reference-contracts.json'
    assert sha(reference_path) == REFERENCE_SHA256, 'historical reference changed'
    measured, reference = read(contract), read(reference_path)
    binding = read(directory / 'binding.json')
    assert binding['contract_sha256'] == sha(contract)
    assert binding['reference_sha256'] == REFERENCE_SHA256
    assert binding['semantic_contracts_equal'] is True
    assert binding['compile_receipt_sha256'] == measured['compile_receipt_sha256'] == sha(directory / 'compile-receipt.json')
    receipt = read(directory / 'compile-receipt.json')
    assert binding['request_sha256'] == receipt['request_sha256'] == sha(directory / 'request.json')
    request = read(directory / 'request.json')
    assert {key: request['compiler'][key] for key in ('path', 'sha256')} == receipt['compiler']
    assert receipt['compiler'] == measured['compiler']
    assert binding['compiler'] == receipt['compiler']
    compare_semantics(measured, reference)
    assert len(receipt['cases']) == 8
    recorded = {item['case']['id']: item for item in receipt['cases']}
    assert set(recorded) == {profile['id'] for profile in measured['profiles']}
    for profile in measured['profiles']:
        item = recorded[profile['id']]
        assert profile['input_sha256'] == item['case']['sha256'] == sha(Path(item['source_path']))
        assert profile['raw_files'] == item['files']
        for name, digest in profile['raw_files'].items():
            assert Path(name).name == name and sha(directory / profile['id'] / name) == digest
    commands = read(directory / 'commands.json')
    assert len(commands) == 40 and all(command['exit_code'] == 0 for command in commands)
    llc_commands = [command['argv'] for command in commands if command['argv'][0] == receipt['compiler']['path']]
    assert len(llc_commands) == 16
    assert all('-enable-ipra=false' in command and '-verify-machineinstrs' in command
               and '-amdgpu-experimental-prove-buffer-calls' not in command for command in llc_commands)
    assert binding['commands_sha256'] == sha(directory / 'commands.json')
    return measured
