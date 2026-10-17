"""Compare all native smoke decoder boundaries against the authenticated CPU oracle."""
import hashlib
import json
from pathlib import Path
from build_llm_full_numeric_test import FILES, ABS_TOL, REL_TOL, L2_TOL, compare, status
from build_llm_full_model_reference import authenticate_full_model_reference


def verify(reference, raw_folder, output):
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
