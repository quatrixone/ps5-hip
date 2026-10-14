"""Offline negative tests for the frozen, inert GEMM D/E package gates."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('gemm_guard', ROOT / 'tools/clr_gemm_control_guard.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def denied(fn, label):
    try:
        fn()
    except (RuntimeError, ValueError, FileNotFoundError):
        return
    raise AssertionError(label + ' was admitted')


def verify_variant(original, change, label):
    with tempfile.TemporaryDirectory(prefix='gemm-control-negative-') as temp:
        path = Path(temp)
        variant = copy.deepcopy(original)
        change(variant)
        (path / 'manifest.json').write_text(json.dumps(variant))
        denied(lambda: guard.verify_control(ROOT, path, require_review=False), label)


def main():
    frozen, package = guard.verify_package(ROOT)
    assert frozen['kernel'] == 'clr_gemm' and package['filesystem_check'] == 'passed'
    deployment_path = ROOT / 'results/clr-gemm-controls-v1/deployment/run.py'
    spec = importlib.util.spec_from_file_location('gemm_deployment', deployment_path)
    deployment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deployment)
    backup = 'PPSA99014.ffpkg.clr-gpu-success.sha256-' + deployment.OLD_IMAGE + '.preserved'
    temporary = '.PPSA99014.clr-gemm-' + deployment.IMAGE + '.upload'
    log_backup = 'ps5-hip-clr-gpu.log.clr-gpu-success-PID407-appa018.sha256-' + deployment.LOG_SHA
    entries = {'PPSA99014.ffpkg': {'type': '-', 'bytes': 5636096}}
    logs = {'ps5-hip-clr-gpu.log': {'type': '-', 'bytes': 870}}
    args = [frozen['expected_log'], backup, temporary, log_backup, 5636096]
    deployment.verify_fresh_listing(entries, logs, *args)
    denied(lambda: deployment.verify_fresh_listing(entries, {**logs,
        'ps5-hip-clr-gemm.log': {'type': '-', 'bytes': 1}}, *args), 'stale GEMM result')
    denied(lambda: deployment.verify_fresh_listing(entries, {**logs,
        'ps5-hip-clr-gpu.log': {'type': '-', 'bytes': 869}}, *args), 'wrong prior log size')
    denied(lambda: deployment.verify_fresh_listing({**entries, backup:
        {'type': '-', 'bytes': 5636096}}, logs, *args), 'preservation conflict')
    denied(lambda: deployment.verify_fresh_listing({**entries, 'PPSA99014.ffpkg':
        {'type': '-', 'bytes': 1}}, logs, *args), 'wrong prior image size')
    for phase in ('deployment', 'execution'):
        directory = ROOT / 'results/clr-gemm-controls-v1' / phase
        manifest = json.loads((directory / 'manifest.json').read_text())
        guard.verify_control(ROOT, directory, require_review=False)
        denied(lambda: guard.verify_control(ROOT, directory), phase + ' unbound package review')
        verify_variant(manifest, lambda m: m.update(image_sha256='0' * 64), phase + ' wrong image')
        verify_variant(manifest, lambda m: m.update(source_commit='0' * 40), phase + ' wrong source')
        verify_variant(manifest, lambda m: m.update(activation_forbidden=False), phase + ' activation')
        verify_variant(manifest, lambda m: m['baseline'].update(prior_log_sha256='0' * 64),
                       phase + ' wrong prior log')
        verify_variant(manifest, lambda m: m['baseline'].update(new_log_must_be_absent=False),
                       phase + ' stale new log')
        verify_variant(manifest, lambda m: m['direct_sha256'].update(
            {'tools/clr_gemm_control_guard.py': '0' * 64}), phase + ' changed guard')
        script = directory / 'run.py'
        result = subprocess.run([sys.executable, str(script)], capture_output=True,
                                timeout=10, text=True)
        assert result.returncode != 0 and 'INERT:' in result.stderr
    checker_path = ROOT / 'results/clr-gemm-controls-v1/execution/check_log.py'
    spec = importlib.util.spec_from_file_location('gemm_log', checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert checker.check_log(checker.EXPECTED)['exact_values'] == 99
    denied(lambda: checker.check_log(checker.EXPECTED.replace(b'exact_values=99',
        b'exact_values=98')), 'wrong numerical count')
    denied(lambda: checker.check_log(checker.EXPECTED[:-1]), 'truncated log')
    print('GEMM inert D/E controls: package identity, fresh-state, review and negative gates pass offline')


if __name__ == '__main__':
    main()
