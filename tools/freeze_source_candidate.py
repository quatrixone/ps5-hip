"""Copy a completed source build into a new, reviewable console candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from source_public_hip_log import LOG_NAMES

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_inputs(receipt):
    inputs = receipt.get('input_sha256')
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('candidate lacks source input identities')
    for name, expected in inputs.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('candidate source input changed: ' + name)


def validate_embedded_audit(build, receipt):
    path = build / 'embedded-native-audit.json'
    if (receipt.get('compiler_object_count') != 1 or
            receipt.get('total_linked_object_count') != 77 or
            digest(path) != receipt.get('embedded_audit_sha256')):
        raise ValueError('embedded candidate lacks the exact compiler/layout audit')
    audit = json.loads(path.read_text())
    if (audit.get('classification') != 'offline-public-hip-embedded-native-layout-audit' or
            audit.get('fixture', 'embedded') != receipt.get('fixture', 'embedded') or
            audit.get('native_executed') is not False or not audit.get('inputs') or
            any(digest(Path(name)) != expected for name, expected in audit['inputs'].items())):
        raise ValueError('embedded audit inputs changed or classification differs')
    roundtrip = receipt.get('fself_roundtrip', {})
    required = {str((build / 'title.elf').resolve()): receipt.get('title_elf_sha256'),
                str((build / 'eboot.elf').resolve()): roundtrip.get('elf_sha256'),
                str((build / 'extracted.elf').resolve()): roundtrip.get('extracted_sha256')}
    compiler = receipt.get('embedded_compiler', {})
    required[compiler.get('object', '')] = compiler.get('object_sha256')
    if any(expected is None or audit['inputs'].get(name) != expected
           for name, expected in required.items()):
        raise ValueError('embedded audit does not bind this build and compiler object')
    return path


def validate_rmsnorm_audit(build, receipt):
    path = build/'rmsnorm-native-audit.json'
    audit = json.loads(path.read_text())
    if (audit.get('status') != 'PASS' or
            audit.get('classification') != 'offline-public-hip-rmsnorm-native-image-audit' or
            audit.get('native_executed') is not False or not isinstance(audit.get('inputs'), dict) or
            not audit['inputs']):
        raise ValueError('RMSNorm native image audit missing or unqualified')
    for name, expected in audit['inputs'].items():
        if digest(Path(name)) != expected:
            raise ValueError('RMSNorm image audit input changed: '+name)
    roundtrip = receipt.get('fself_roundtrip', {})
    required = {str((build/'title.elf').resolve()): receipt.get('title_elf_sha256'),
                str((build/'eboot.elf').resolve()): roundtrip.get('elf_sha256'),
                str((build/'extracted.elf').resolve()): roundtrip.get('extracted_sha256')}
    for name in ('tools/audit_public_hip_rmsnorm_native.py', 'tools/audit_public_hip_embedded_native.py',
                 'tools/probe_host_registration.py', 'tools/native_multi_query.py',
                 'tests/fat_binary_fixture_test.py', 'tests/native_import_libraries_test.py',
                 'tools/build_native_vector_title.py', 'tools/public_hip_rmsnorm_console_title.cpp',
                 'tools/public_hip_app_crt.cpp', 'tests/public_hip_rmsnorm_native_audit_test.py'):
        required[str((ROOT/name).resolve())] = digest(ROOT/name)
    if any(expected is None or audit['inputs'].get(name) != expected for name, expected in required.items()):
        raise ValueError('RMSNorm image audit does not bind candidate images and implementation')
    return path


def freeze_rmsnorm_reference(build, receipt, output):
    from build_public_hip import rmsnorm_assets, rmsnorm_reference_inputs
    bound = receipt.get('rmsnorm', {})
    reference = bound.get('reference', {})
    original = Path(reference.get('receipt', '')).resolve()
    module = Path(bound.get('assets', {}).get('rmsnorm/norm.hsaco', '')).resolve()
    if (not original.is_file() or not module.is_file() or
            rmsnorm_assets(module.parent, original.parent) != bound):
        raise ValueError('RMSNorm build inputs no longer match')
    for name, source in bound['assets'].items():
        expected = digest(Path(source))
        if receipt['package_files'].get(name) != expected or digest(build/'package/app'/name) != expected:
            raise ValueError('RMSNorm packaged oracle or module differs')
    report = json.loads(original.read_text())
    destination = output/'rmsnorm-reference'
    destination.mkdir()
    shutil.copyfile(original, destination/'receipt.json')
    total = 0
    for name, expected in report['artifacts'].items():
        source = (original.parent/name).resolve()
        if Path(name).is_absolute() or not source.is_relative_to(original.parent):
            raise ValueError('invalid reference artifact path')
        total += source.stat().st_size
        if total > 16*1024*1024 or digest(source) != expected:
            raise ValueError('reference artifact bound or identity changed')
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    copied = rmsnorm_reference_inputs(destination)
    if copied['receipt_sha256'] != reference['receipt_sha256']:
        raise ValueError('frozen reference receipt differs')
    return {'reference': str(destination.relative_to(ROOT)),
            'reference_receipt_sha256': copied['receipt_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot-build', type=Path, default=ROOT / 'build/source-make-snapshot',
                        help='Offline teardown helper from make console-controls')
    parser.add_argument('--baseline-image-sha256', required=True)
    parser.add_argument('--baseline-log-sha256')
    parser.add_argument('--baseline-log-preserved', action='store_true',
                        help='Verify an existing SHA-suffixed preserved baseline log; never rename it again')
    parser.add_argument('--baseline-log-fixture', choices=tuple(LOG_NAMES), default='bootstrap')
    args = parser.parse_args()
    if args.baseline_log_preserved and not args.baseline_log_sha256:
        parser.error('--baseline-log-preserved requires --baseline-log-sha256')
    import re
    for value in (args.baseline_image_sha256, args.baseline_log_sha256):
        if value is not None and not re.fullmatch('[0-9a-f]{64}', value):
            parser.error('baseline digests must be lowercase SHA-256')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'results'):
        parser.error('candidate must be under results/')
    receipt = json.loads((args.build / 'link-receipt.json').read_text())
    fixture = receipt.get('fixture', 'bootstrap')
    if fixture not in LOG_NAMES:
        parser.error('unknown fixture')
    if receipt['results_dir'] != '/data/ps5-hip/results' or receipt['object_count'] != 76:
        parser.error('not the complete .40 candidate')
    validate_source_inputs(receipt)
    audit = (validate_rmsnorm_audit(args.build, receipt) if fixture == 'rmsnorm' else
             validate_embedded_audit(args.build, receipt) if fixture in ('embedded', 'aggregate') else None)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT):
        parser.error('commit source before freezing')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.build / 'package', output / 'package')
    shutil.copyfile(args.build / 'link-receipt.json', output / 'build-receipt.json')
    if audit is not None:
        shutil.copyfile(audit, output / audit.name)
    shutil.copytree(ROOT / 'build/controllers/PPSA99014', output / 'controllers')
    shutil.copytree(args.snapshot_build, output / 'snapshot')
    package = output / 'package/PPSA99014.ffpkg'
    if digest(package) != receipt['package_sha256']:
        parser.error('package hash mismatch')
    for name, expected in receipt['package_files'].items():
        if digest(output / 'package/app' / name) != expected:
            parser.error('package stage hash mismatch: ' + name)
    rmsnorm = freeze_rmsnorm_reference(args.build, receipt, output) if fixture == 'rmsnorm' else None
    files = {str(path.relative_to(ROOT)): digest(path) for path in output.rglob('*') if path.is_file()}
    for name in ('tools/source_console.py', 'tools/source_public_hip_log.py', 'tools/freeze_source_candidate.py',
                 'dependencies.json', 'elevation-inputs.json', 'controller-inputs.json', 'title.json'):
        files[name] = digest(ROOT / name)
    if rmsnorm is not None:
        for name, expected in json.loads(audit.read_text())['inputs'].items():
            path = Path(name).resolve()
            if path.is_relative_to(ROOT):
                files[str(path.relative_to(ROOT))] = expected
        for name in ('tools/verify_public_hip_rmsnorm.py', 'tools/build_public_hip.py',
                     'tools/qualify_public_hip_rmsnorm.py', 'tools/build_llm_rmsnorm_reference.py',
                     'tests/public_hip_rmsnorm_readback_test.py'):
            files[name] = digest(ROOT/name)
        for name, expected in receipt['rmsnorm']['reference']['inputs'].items():
            path = Path(name).resolve()
            if path.is_relative_to(ROOT):
                files[str(path.relative_to(ROOT))] = expected
    relative = str(output.relative_to(ROOT))
    manifest = {'host': '192.168.4.40', 'title_id': 'PPSA99014', 'results_dir': receipt['results_dir'],
                'baseline_image_sha256': args.baseline_image_sha256,
                'baseline_log_sha256': args.baseline_log_sha256,
                'baseline_log_preserved': args.baseline_log_preserved,
                'baseline_log_fixture': args.baseline_log_fixture, 'fixture': fixture,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'package': relative + '/package/PPSA99014.ffpkg', 'app': relative + '/package/app',
                'launch': relative + '/controllers/launch.elf', 'close': relative + '/controllers/close.elf',
                'snapshot': relative + '/snapshot/teardown-snapshot.elf',
                'post_snapshot': relative + '/snapshot/teardown-snapshot.elf',
                'package_files': receipt['package_files'], 'files': files,
                'hypothesis': ('Unchanged upstream RMSNorm computes six CPU-oracle cases with81920 finite values,192 outputguardbytes,507904 inactivebytes and835776 unchangedinputbytes. Require independently verified full raw readbacks, exact two allocation/frees and one retiredqueue; transcript alone is insufficient.' if fixture == 'rmsnorm' else
                               'Genuine compiler stubs pass 12/16/128-byte aggregate arguments to two changed-input GPU rounds, preserving320memberbytes64guardbytes, one allocation/free and one retired queue. Internal struct padding is unspecified; no nested-pointer lifetime claim. Audited compiler unregister precedes CLR teardown; final device backing checks do not prove full static-registry retirement.' if fixture == 'aggregate' else
                               'Genuine compiler registration and launch stubs execute two changed-input GPU rounds, preserving146values16guards, one allocation/free and one retired queue. Audited compiler unregister precedes CLR teardown; final device backing checks do not prove full static-registry retirement.' if fixture == 'embedded' else
                               'After the existing joined-copy recovery oracle, a two-kernel joined graph alternates eager/captured order over four warmup and sixteen further changed-input rounds. Validate 5840 added values and640guards,84totaldispatches,28callbacks,fourallocations/frees and seven retired queues. Require32post-sync backing samples at32768bytes/twoallocations,128balanced allocations/releases per path,98304accountedpeak and zero final backing. This measures NativeMemory backing only, not process heap, command lifetime or timing.' if fixture == 'copy_gpu' else
                               'Public HIP module kernel capture records a kernel and dependent host callback without dispatch; disabled and changed-argument replay, executable update and source graph destruction preserve146values16guards and clean3queue retirement.' if fixture == 'capture_gpu' else
                               'Explicit public HIP GPU graph owns arguments, updates the executable, survives source destruction, preserves146values16guards across2launches and2disabledreplays with4callbacks, then frees1buffer and retires3queues.' if fixture == 'graph_gpu' else
                               'Public HIP producer GPU launch records an event; a nonblocking consumer waits before D2D copy, preserving146outputs16guards and source over2changedrounds with event reuse and3retiredqueues.' if fixture == 'stream' else
                               'Public HIP explicit and captured host graphs replay32changed inputs, execute64callbacks, update callback/user data, survive source-graph destruction and recover6invalidations while retaining3caller-owned graphs, then retire5queues without GPU work.' if fixture == 'graph' else
                               'New source-built upstream hipInit and device discovery complete with count=1 and clean natural teardown.' if fixture == 'bootstrap' else
                               'Public hipMalloc allocates two distinct buffers, hipFree rejects interior/double frees and retires both, followed by clean natural teardown.' if fixture == 'memory' else
                               'Public HIP contiguous and pitched 2D/3D fills validate4096bytes across two changed-pattern rounds, reject16invalid requests, preserve guards and row/slice padding and finalize without GPU submission.' if fixture == 'fill' else
                               'Public HIP loads the owned AOT kernel and executes two changed-input dispatches, validates146values and16guards, rejects unsupported geometry, and finalizes with exact resource cleanup.' if fixture == 'launch' else
                               'Two independent public HIP modules load their owned AOT file, expose distinct functions, retain the surviving module after the other unloads, and finalize without queues or GPU work.' if fixture == 'module' else
                               'Public HIP contiguous and rectangular copies validate 7168 bytes and reject 16 invalid requests, then shared queued-free and parameter-migration cases validate 384 lifetime bytes with executable/source graphs still alive, eight allocations/frees and four retired queues without GPU submission.'),
                'expected': 'Exact ' + fixture + ' transcript, one ordered title start/stop, equal two-row mount snapshots, healthy services.',
                'reject': 'Any identity, log, mount, lifecycle, timeout, exception, or health failure; never retry automatically.',
                'deadline_seconds': 40, 'public_hip_gpu_execution_claimed': False}
    if rmsnorm is not None:
        manifest['rmsnorm'] = rmsnorm
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output / 'manifest.json')


if __name__ == '__main__':
    main()
