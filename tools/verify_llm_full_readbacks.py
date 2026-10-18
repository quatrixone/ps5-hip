"""Compare all native smoke decoder boundaries against the authenticated CPU oracle."""
import hashlib
import json
from pathlib import Path
from build_llm_full_numeric_test import FILES, ABS_TOL, REL_TOL, L2_TOL, compare, status
from build_llm_full_model_reference import authenticate_full_model_reference


def generation_files(reference):
    from build_llm_generation_reference import validate
    folder = Path(reference) / 'raw'
    summary = validate(folder)
    if summary['requests'] != 16 or summary['raw_files'] != 384:
        raise ValueError('generation reference profile')
    return {p.name: p.stat().st_size for p in folder.iterdir()}


def verify_generation(reference, raw_folder, output):
    from build_llm_generation_reference import authenticate_generation_reference, validate
    from build_llm_generation_numeric_test import compare_generation
    data, inputs = authenticate_generation_reference(reference)
    files = generation_files(reference)
    raw_folder = Path(raw_folder)
    if raw_folder.is_symlink() or {p.name for p in raw_folder.iterdir()} != set(files):
        raise ValueError('generation readback inventory')
    for name, size in files.items():
        path = raw_folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError('generation readback kind/size')
    before = {name: hashlib.sha256((raw_folder / name).read_bytes()).hexdigest() for name in files}
    validate(raw_folder)
    for name in ('generation.json', 'steps.jsonl'):
        if (raw_folder / name).read_bytes() != (Path(reference) / 'raw' / name).read_bytes():
            raise ValueError('generation metadata differs')
    rows, floats, requests = compare_generation(raw_folder, Path(reference) / 'raw', host_backend=False)
    if len(rows) != 384 or not all(row['pass_budget'] for row in rows):
        raise ValueError('generation decoder numerical budget')
    if before != {name: hashlib.sha256((raw_folder / name).read_bytes()).hexdigest() for name in files}:
        raise ValueError('generation readbacks changed during comparison')
    for name, expected in inputs.items():
        with Path(name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('generation reference changed')
    result = dict(status='PASS', classification='actual-model-generation-numerical-comparison', profile='generation', reference_receipt_sha256=data['receipt_sha256'], model=data['model'], requests=requests, layers=22, logits=32000, float_files=len(floats), raw_files=len(files), raw_bytes=sum(files.values()), raw_sha256=before, metrics=rows, budgets=dict(absolute=ABS_TOL, relative=REL_TOL, relative_l2=L2_TOL), gpu_origin_established=False, inference=False, limits=['Native origin, actual feedback and cleanup require the separate controller and image audit', 'One short prompt and reset repeat; wider generation and long-context gates remain open'])
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify(reference, raw_folder, output, profile='smoke'):
    if profile == 'generation':
        return verify_generation(reference, raw_folder, output)
    if profile != 'smoke':
        raise ValueError('unsupported full readback profile')
    data, inputs = authenticate_full_model_reference(reference)
    raw_folder = Path(raw_folder)
    if raw_folder.is_symlink() or {p.name for p in raw_folder.iterdir()} != set(FILES):
        raise ValueError('full readback inventory')
    for name, size in FILES.items():
        path = raw_folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError('full readback kind/size')
    before = {name: hashlib.sha256((raw_folder / name).read_bytes()).hexdigest() for name in FILES}
    rows = compare(raw_folder, Path(reference) / 'raw')
    if status(rows)[0] != 'PASS':
        raise ValueError('full decoder numerical budget')
    if before != {name: hashlib.sha256((raw_folder / name).read_bytes()).hexdigest() for name in FILES}:
        raise ValueError('full readbacks changed during comparison')
    for name, expected in inputs.items():
        with Path(name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('full reference changed')
    result = dict(status='PASS', classification='actual-model-full-numerical-comparison', reference_receipt_sha256=data['receipt_sha256'], model=data['model'], requests=3, layers=22, logits=32000, raw_files=len(FILES), raw_bytes=sum(FILES.values()), raw_sha256=before, metrics=rows, budgets=dict(absolute=ABS_TOL, relative=REL_TOL, relative_l2=L2_TOL), gpu_origin_established=False, inference=False, limits=['Native origin and cleanup require the separate controller receipt', 'Three forced-token requests; generation and long-context qualification remain separate'])
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result
