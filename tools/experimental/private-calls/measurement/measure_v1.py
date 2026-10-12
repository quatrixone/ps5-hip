"""Measure eight profiles with a clean source-built llc; never invoke the proof."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

out = Path(__file__).resolve().parent
sys.path.insert(0, str(out.parent / 'g3-private-backend-validation'))
from measurement_binding_v1 import compare_semantics, REFERENCE_SHA256, verify_measurement

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text())
parser = argparse.ArgumentParser()
parser.add_argument('--compiler', type=Path, required=True)
args = parser.parse_args()
compiler = args.compiler.resolve(strict=True)
request = read(out / 'request.json')
assert request['compiler']['path'] == str(compiler) and request['compiler']['sha256'] == sha(compiler)
source_root = out.parent / 'g3-private-backend-validation'
matrix = read(source_root / 'matrix.json')
cases = [case for case in matrix['cases'] if case['kind'] in ('ordinary', 'volatile') and case['order'] == 'original']
assert len(cases) == 8
assert sha(out / 'reference-contracts.json') == REFERENCE_SHA256
results, commands = [], []
for case in cases:
    directory = out / case['id']
    directory.mkdir()
    source = source_root / case['input']
    assert sha(source) == case['sha256']
    base = [str(compiler), '-enable-ipra=false', '-mtriple=amdgcn-amd-amdhsa',
            '-mcpu=' + case['target'], '-' + case['optimization'], '-verify-machineinstrs']
    for name, command in [
        ('asm', base + ['-filetype=asm', str(source), '-o', str(directory / 'output.s')]),
        ('object', base + ['-filetype=obj', str(source), '-o', str(directory / 'output.o')]),
        ('link', ['ld.lld-18', '-shared', '--no-undefined', str(directory / 'output.o'), '-o', str(directory / 'output.hsaco')]),
        ('isa', ['llvm-objdump-18', '-d', '--mcpu=' + case['target'], str(directory / 'output.hsaco')]),
        ('metadata', ['llvm-readobj-18', '--notes', '--symbols', '--relocations', str(directory / 'output.hsaco')])]:
        result = subprocess.run(command, capture_output=True, text=True)
        (directory / (name + ('.txt' if name in ('isa', 'metadata') else '.log'))).write_text(result.stdout + result.stderr)
        commands.append({'case': case['id'], 'argv': command, 'exit_code': result.returncode})
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        result.check_returncode()
    results.append({'case': case, 'source_path': str(source),
                    'files': {path.name: sha(path) for path in directory.iterdir()}})
assert sha(compiler) == request['compiler']['sha256']
receipt = {'classification': 'fresh independent clean-baseline IPRA-off measurement',
           'compiler': {'path': str(compiler), 'sha256': sha(compiler)}, 'cases': results,
           'request_sha256': sha(out / 'request.json'), 'matrix_sha256': sha(source_root / 'matrix.json'),
           'hardware_validated': False, 'patched_acceptance': False, 'script_sha256': sha(Path(__file__))}
(out / 'compile-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
subprocess.run([sys.executable, str(out / 'analyze_portable_v1.py')], check=True)
contract = out / 'measured-contracts.json'
compare_semantics(read(contract), read(out / 'reference-contracts.json'))
binding = {'schema': 1, 'contract_sha256': sha(contract), 'reference_sha256': REFERENCE_SHA256,
           'semantic_contracts_equal': True, 'compiler': receipt['compiler'],
           'compile_receipt_sha256': sha(out / 'compile-receipt.json'),
           'commands_sha256': sha(out / 'commands.json'), 'request_sha256': sha(out / 'request.json'),
           'hardware_validated': False, 'accepted_compiler': False}
(out / 'binding.json').write_text(json.dumps(binding, indent=2) + '\n')
verify_measurement(contract)
print('Eight fresh clean-baseline semantic contracts exactly match the historical reference')
