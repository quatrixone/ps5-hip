"""Local, network-free identity gate shared by inert GEMM D/E controls."""
if not __debug__:
    raise RuntimeError('optimized Python is forbidden')

import hashlib
import json
import subprocess
from pathlib import Path

TITLE = 'PPSA99014'
SOURCE = 'b00dac10adcc1ba45d9e2407b9a41f3c6c513c69'
IMAGE = '9c229e4221570716ec1cc2ee9bea08d41299bd6b433aad46a30ebf121f5777e4'
MANIFEST = 'f15f40572e3fe23b17ec5ac1a1208ac8e92c7d70ccccbf466e905bcfc26f700e'
ADMISSION_REVIEW = '8423e584b5414980dc1384dff2d97c8e2dbb9413f84f0a2f20089333ebe0a5cc'
STAGE = 'dist/PPSA99014/gfx1013-clr-gemm-832962dca512'
FILES = {'eboot.bin', 'lapy.elf', 'sce_module/libc.prx', 'sce_sys/param.json',
         'sce_sys/icon0.png', 'results-dir.txt'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def verify_package(candidate):
    candidate = Path(candidate)
    stage = candidate / STAGE
    frozen = json.loads((stage / 'manifest.json').read_text())
    receipt = json.loads((stage / 'package-receipt.json').read_text())
    evidence = json.loads((candidate / 'evidence/offline-clr-gemm-package.json').read_text())
    require(evidence['admission_review_sha256'] == ADMISSION_REVIEW and
            sha(candidate / 'results/gemm-independent-review-v1/review.json') == ADMISSION_REVIEW,
            'admission review changed')
    require(evidence['source_commit'] == SOURCE and frozen['source_commit'] == SOURCE,
            'source identity changed')
    require(evidence['stage'] == STAGE and evidence['manifest_sha256'] == MANIFEST and
            sha(stage / 'manifest.json') == MANIFEST, 'package manifest changed')
    require(sha(stage / 'package-receipt.json') == evidence['package_receipt_sha256'],
            'package receipt changed')
    require(evidence['image_sha256'] == IMAGE and
            receipt['image'] == {'file': 'PPSA99014.ffpkg', 'bytes': 5636096, 'sha256': IMAGE} and
            sha(stage / 'PPSA99014.ffpkg') == IMAGE, 'package image changed')
    require(receipt['manifest_sha256'] == MANIFEST and
            receipt['filesystem_check'] == 'passed' and receipt['extracted_files_match'] is True,
            'package UFS2 verification missing')
    require(frozen['title_id'] == TITLE and frozen['target'] == 'gfx1013' and
            frozen['kernel'] == 'clr_gemm' and
            frozen['expected_log'] == '/data/ps5-hip/results/ps5-hip-clr-gemm.log',
            'package identity changed')
    require(set(frozen['files']) == FILES and set(evidence['files']) == FILES,
            'six-file inventory changed')
    actual = {p.relative_to(stage / 'app').as_posix():
              {'sha256': sha(p), 'bytes': p.stat().st_size}
              for p in (stage / 'app').rglob('*') if p.is_file()}
    require(actual == frozen['files'] and all(actual[name]['sha256'] == evidence['files'][name]
            for name in FILES), 'staged files changed')
    require(not any(p.is_symlink() for p in (stage / 'app').rglob('*')),
            'staged symlink forbidden')
    return frozen, receipt


def verify_review(candidate, control_manifest):
    entry = control_manifest['independent_package_review']
    require(isinstance(entry, dict) and entry.get('path') and entry.get('sha256'),
            'independent package review is not bound')
    relative = Path(entry['path'])
    require(not relative.is_absolute() and '..' not in relative.parts,
            'independent package review escapes candidate')
    path = Path(candidate) / relative
    require(sha(path) == entry['sha256'], 'independent package review changed')
    review = json.loads(path.read_text())
    require(review.get('decision') == 'CLEAR_FOR_INERT_CONTROLS_ONLY',
            'package review has not cleared inert controls')
    require(review.get('image_sha256') == IMAGE and review.get('manifest_sha256') == MANIFEST,
            'package review identity mismatch')
    return review


def verify_control(candidate, control_dir, *, require_review=True):
    candidate = Path(candidate)
    control_dir = Path(control_dir)
    control = json.loads((control_dir / 'manifest.json').read_text())
    require(control['activation_forbidden'] is True and
            control['candidate_package'] == STAGE and control['source_commit'] == SOURCE and
            control['image_sha256'] == IMAGE and control['stage_manifest_sha256'] == MANIFEST and
            control['admission_review_sha256'] == ADMISSION_REVIEW,
            'control manifest identity mismatch or activation enabled')
    baseline = control['baseline']
    require(baseline == {
        'prior_image_sha256': '84e52f366b9c7e598d81c3d37197ff8e71d548ebe6cfe933176639066abcc8da',
        'prior_image_bytes': 5636096,
        'prior_log': '/data/ps5-hip/results/ps5-hip-clr-gpu.log',
        'prior_log_sha256': '5c91d896f50ec94b7092b90ed30faa84e9b69f3c84e9a7819c5850f3772edc0e',
        'prior_log_bytes': 870,
        'new_log': '/data/ps5-hip/results/ps5-hip-clr-gemm.log',
        'new_log_must_be_absent': True,
        'host': '192.168.4.40', 'title': TITLE,
    }, 'fresh .40 baseline contract changed')
    frozen, receipt = verify_package(candidate)
    for path, digest in control['direct_sha256'].items():
        relative = Path(path)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'direct input escapes candidate')
        target = candidate / relative
        require(sha(target) == digest, 'control direct input changed: ' + path)
    for path, digest in control['external_sha256'].items():
        relative = Path(path)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'external input escapes root')
        target = candidate.parent / 'ps5-hip' / relative
        require(sha(target) == digest, 'external control input changed: ' + path)
    if require_review:
        verify_review(candidate, control)
    return control, frozen, receipt
