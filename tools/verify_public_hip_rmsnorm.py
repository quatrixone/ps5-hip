"""Verify collected RMSNorm bytes against authenticated CPU references, offline.

A passing comparison does not establish the origin of the dumps. Bind them to a
separate frozen package/deployment/run receipt before claiming GPU execution.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from build_llm_rmsnorm_reference import TOLERANCE, WIDTH

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 16
TOTAL = PREFIX + WIDTH * 17 * 4 + 16
CASES = tuple((f'{family}-{rows}', rows) for family in ('stress', 'embedding') for rows in (1, 2, 17))
ASSET_NAMES = {f'rmsnorm/{case}.{suffix}.f32le' for case, _ in CASES for suffix in ('input', 'cpu')}
DUMP_NAMES = {f'rmsnorm-{case}.{suffix}.raw' for case, _ in CASES for suffix in ('output', 'input-readback')}


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def exact_bytes(path, size):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size != size:
        raise ValueError('missing, symlinked or wrong size file: ' + str(path))
    with path.open('rb') as source:
        data = source.read(size + 1)
    if len(data) != size:
        raise ValueError('file changed size: ' + str(path))
    return data


def verify_buffers(dumps, assets):
    """Compare bytes; caller must authenticate the twelve reference assets first."""
    dumps = Path(dumps)
    if set(assets) != ASSET_NAMES:
        raise ValueError('exact twelve reference assets required')
    if {p.name for p in dumps.glob('rmsnorm-*.raw')} != DUMP_NAMES:
        raise ValueError('exact twelve case dumps required')
    records = []
    hashes = {}
    for case, rows in CASES:
        count = rows * WIDTH
        size = count * 4
        input_path = Path(assets[f'rmsnorm/{case}.input.f32le'])
        cpu_path = Path(assets[f'rmsnorm/{case}.cpu.f32le'])
        original = exact_bytes(input_path, size)
        reference_bytes = exact_bytes(cpu_path, size)
        reference = struct.unpack('<' + 'f' * count, reference_bytes)
        if not all(math.isfinite(v) for v in reference) or not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', original)):
            raise ValueError('non-finite reference asset: ' + case)
        output_path = dumps / f'rmsnorm-{case}.output.raw'
        input_dump_path = dumps / f'rmsnorm-{case}.input-readback.raw'
        output = exact_bytes(output_path, TOTAL)
        input_dump = exact_bytes(input_dump_path, TOTAL)
        expected_input = b'\xa5' * PREFIX + original + b'\xa5' * (TOTAL - PREFIX - size)
        if input_dump != expected_input:
            raise ValueError('input or input guard changed: ' + case)
        if output[:PREFIX] != b'\x5a' * PREFIX or output[PREFIX+size:] != b'\x5a' * (TOTAL-PREFIX-size):
            raise ValueError('output guard or inactive bytes changed: ' + case)
        values = struct.unpack('<' + 'f' * count, output[PREFIX:PREFIX+size])
        maximum_error = 0.0
        maximum_fraction = 0.0
        maximum_relative = 0.0
        squared = []
        for index, (value, expected) in enumerate(zip(values, reference)):
            if not math.isfinite(value):
                raise ValueError(f'non-finite output: {case} index={index}')
            error = abs(float(value) - float(expected))
            budget = TOLERANCE['absolute'] + TOLERANCE['relative'] * abs(float(expected))
            if error > budget:
                raise ValueError(f'numerical mismatch: {case} index={index} error={error} budget={budget}')
            maximum_error = max(maximum_error, error)
            maximum_fraction = max(maximum_fraction, error / budget)
            if expected != 0:
                maximum_relative = max(maximum_relative, error / abs(float(expected)))
            squared.append(error * error)
        for path, data in ((input_path, original), (cpu_path, reference_bytes), (output_path, output), (input_dump_path, input_dump)):
            name = str(path.resolve())
            digest = hashlib.sha256(data).hexdigest()
            if name in hashes and hashes[name] != digest:
                raise ValueError('aliased file changed during comparison')
            hashes[name] = digest
        records.append(dict(name=case, rows=rows, values=count, finite_values=count,
                            maximum_absolute_error=maximum_error, maximum_budget_fraction=maximum_fraction,
                            maximum_relative_error_nonzero_reference=maximum_relative,
                            rmse=math.sqrt(math.fsum(squared) / count), guard_bytes=32,
                            inactive_bytes=TOTAL-size-32, unchanged_input_bytes=TOTAL))
    for name, digest in hashes.items():
        if sha(name) != digest:
            raise ValueError('file changed during verification: ' + name)
    totals = {key: sum(c[key] for c in records) for key in ('values', 'finite_values', 'guard_bytes', 'inactive_bytes', 'unchanged_input_bytes')}
    if totals != dict(values=81920, finite_values=81920, guard_bytes=192, inactive_bytes=507904, unchanged_input_bytes=835776):
        raise ValueError('case totals changed')
    return dict(cases=records, totals=totals, verified_files=hashes)


def verify(reference, dumps, output):
    # Reuse the package builder's exact CPU source/model/reference receipt gate.
    from build_public_hip import rmsnorm_reference_inputs
    reference, dumps, output = (Path(p).resolve() for p in (reference, dumps, output))
    if not output.is_relative_to(ROOT / 'results') or output == ROOT / 'results':
        raise ValueError('fresh output must be under results/')
    if output.exists():
        raise ValueError('fresh output directory required')
    tools = {str(ROOT/name): sha(ROOT/name) for name in (
        'tools/verify_public_hip_rmsnorm.py', 'tests/public_hip_rmsnorm_readback_test.py',
        'tools/build_public_hip.py', 'tools/build_llm_rmsnorm_reference.py',
        'tools/qualify_public_hip_rmsnorm.py')}
    tools[str(Path(sys.executable).resolve())] = sha(sys.executable)
    authenticated = rmsnorm_reference_inputs(reference)
    result = verify_buffers(dumps, authenticated['assets'])
    if rmsnorm_reference_inputs(reference) != authenticated:
        raise ValueError('authenticated reference changed during verification')
    for name, digest in tools.items():
        if sha(name) != digest:
            raise ValueError('verifier implementation changed: ' + name)
    for name, digest in result['verified_files'].items():
        if sha(name) != digest:
            raise ValueError('readback changed before receipt: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=1, status='PASS', classification='offline-rmsnorm-readback-comparison',
                   console_contacted=False, gpu_origin_proven=False, reference=authenticated,
                   verifier_identities=tools, tolerance=TOLERANCE, **result,
                   limits=['Byte comparison only: separately verified deployment/run provenance must establish GPU origin.',
                           'Frozen contiguous unfused F32 RMSNorm cases only; no full-model inference or performance qualification.'])
    (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('PASS ' + str(output/'receipt.json'))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--dumps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.reference, args.dumps, args.output)
