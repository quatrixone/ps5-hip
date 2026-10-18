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


def corpus_files(reference):
    from build_llm_generation_reference import validate
    folder = Path(reference) / 'raw'
    summary = validate(folder, 'corpus')
    if summary['requests'] != 272 or summary['raw_files'] != 6528:
        raise ValueError('corpus reference profile')
    return {p.relative_to(folder).as_posix(): p.stat().st_size for p in folder.rglob('*') if p.is_file()}


def corpus_inventory(folder, files):
    directories = {str(Path(name).parent).replace('\\', '/') for name in files if '/' in name}
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('corpus readback inventory')
    paths = list(folder.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('corpus readback kind/size')
    if {p.relative_to(folder).as_posix() for p in paths if p.is_file()} != set(files) or {p.relative_to(folder).as_posix() for p in paths if p.is_dir()} != directories or any(not p.is_dir() and not p.is_file() for p in paths):
        raise ValueError('corpus readback inventory')


def verify_generation(reference, raw_folder, output, profile='generation'):
    from build_llm_generation_reference import authenticate_generation_reference, validate
    from build_llm_generation_numeric_test import compare_generation, compare_corpus
    data, inputs = authenticate_generation_reference(reference)
    corpus = profile == 'corpus'
    if data.get('profile', 'short') != ('corpus' if corpus else 'short'):
        raise ValueError('generation reference profile')
    files = (corpus_files if corpus else generation_files)(reference)
    raw_folder = Path(raw_folder)
    if corpus:
        corpus_inventory(raw_folder, files)
    elif raw_folder.is_symlink() or {p.name for p in raw_folder.iterdir()} != set(files):
        raise ValueError('generation readback inventory')
    for name, size in files.items():
        path = raw_folder / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise ValueError('generation readback kind/size')
    before = {name: hashlib.sha256((raw_folder / name).read_bytes()).hexdigest() for name in files}
    validate(raw_folder, 'corpus' if corpus else 'short')
    for name in (name for name in files if name.endswith(('.json', '.jsonl'))):
        if (raw_folder / name).read_bytes() != (Path(reference) / 'raw' / name).read_bytes():
            raise ValueError('generation metadata differs')
    rows, floats, requests = (compare_corpus if corpus else compare_generation)(raw_folder, Path(reference) / 'raw', host_backend=False)
    if len(rows) != (6528 if corpus else 384) or not all(row['pass_budget'] for row in rows):
        raise ValueError('generation decoder numerical budget')
    if before != {name: hashlib.sha256((raw_folder / name).read_bytes()).hexdigest() for name in files}:
        raise ValueError('generation readbacks changed during comparison')
    if corpus:
        corpus_inventory(raw_folder, files)
    for name, expected in inputs.items():
        with Path(name).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError('generation reference changed')
    result = dict(status='PASS', classification='actual-model-generation-numerical-comparison', profile=profile, reference_receipt_sha256=data['receipt_sha256'], model=data['model'], requests=requests, layers=22, logits=32000, float_files=len(floats), raw_files=len(files), raw_bytes=sum(files.values()), raw_sha256=before, metrics=rows, budgets=dict(absolute=ABS_TOL, relative=REL_TOL, relative_l2=L2_TOL), gpu_origin_established=False, inference=False, limits=['Native origin, actual feedback and cleanup require the separate controller and image audit', 'Five fixed prompts and sixteen generated tokens with reset repeat; long-context and release gates remain open' if corpus else 'One short prompt and reset repeat; wider generation and long-context gates remain open'])
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify(reference, raw_folder, output, profile='smoke'):
    if profile in ('generation', 'corpus'):
        return verify_generation(reference, raw_folder, output, profile)
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
