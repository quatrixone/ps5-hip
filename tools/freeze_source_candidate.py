"""Copy a completed source build into a new, reviewable console candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-image-sha256', required=True)
    parser.add_argument('--baseline-log-sha256')
    parser.add_argument('--baseline-log-fixture', choices=('bootstrap', 'memory', 'copy'), default='bootstrap')
    args = parser.parse_args()
    import re
    for value in (args.baseline_image_sha256, args.baseline_log_sha256):
        if value is not None and not re.fullmatch('[0-9a-f]{64}', value):
            parser.error('baseline digests must be lowercase SHA-256')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'results'):
        parser.error('candidate must be under results/')
    receipt = json.loads((args.build / 'link-receipt.json').read_text())
    fixture = receipt.get('fixture', 'bootstrap')
    if fixture not in ('bootstrap', 'memory', 'copy'):
        parser.error('unknown fixture')
    if receipt['results_dir'] != '/data/ps5-hip/results' or receipt['object_count'] != 74:
        parser.error('not the complete .40 candidate')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT):
        parser.error('commit source before freezing')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.build / 'package', output / 'package')
    shutil.copyfile(args.build / 'link-receipt.json', output / 'build-receipt.json')
    shutil.copytree(ROOT / 'build/controllers/PPSA99014', output / 'controllers')
    shutil.copytree(ROOT / 'build/source-make-snapshot', output / 'snapshot')
    package = output / 'package/PPSA99014.ffpkg'
    if digest(package) != receipt['package_sha256']:
        parser.error('package hash mismatch')
    for name, expected in receipt['package_files'].items():
        if digest(output / 'package/app' / name) != expected:
            parser.error('package stage hash mismatch: ' + name)
    files = {str(path.relative_to(ROOT)): digest(path) for path in output.rglob('*') if path.is_file()}
    for name in ('tools/source_console.py', 'tools/source_public_hip_log.py', 'tools/freeze_source_candidate.py',
                 'dependencies.json', 'elevation-inputs.json', 'controller-inputs.json', 'title.json'):
        files[name] = digest(ROOT / name)
    relative = str(output.relative_to(ROOT))
    manifest = {'host': '192.168.4.40', 'title_id': 'PPSA99014', 'results_dir': receipt['results_dir'],
                'baseline_image_sha256': args.baseline_image_sha256,
                'baseline_log_sha256': args.baseline_log_sha256,
                'baseline_log_fixture': args.baseline_log_fixture, 'fixture': fixture,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'package': relative + '/package/PPSA99014.ffpkg', 'app': relative + '/package/app',
                'launch': relative + '/controllers/launch.elf', 'close': relative + '/controllers/close.elf',
                'snapshot': relative + '/snapshot/teardown-snapshot.elf',
                'post_snapshot': relative + '/snapshot/teardown-snapshot.elf',
                'package_files': receipt['package_files'], 'files': files,
                'hypothesis': ('New source-built upstream hipInit and device discovery complete with count=1 and clean natural teardown.' if fixture == 'bootstrap' else
                               'Public hipMalloc allocates two distinct buffers, hipFree rejects interior/double frees and retires both, followed by clean natural teardown.' if fixture == 'memory' else
                               'Synchronous public HIP H2D/D2D/D2H copies preserve exact payload, guards and sources across two changed rounds, reject bounds violations without mutation, then cleanly retire.'),
                'expected': 'Exact ' + fixture + ' transcript, one ordered title start/stop, equal two-row mount snapshots, healthy services.',
                'reject': 'Any identity, log, mount, lifecycle, timeout, exception, or health failure; never retry automatically.',
                'deadline_seconds': 40, 'public_hip_gpu_execution_claimed': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output / 'manifest.json')


if __name__ == '__main__':
    main()
