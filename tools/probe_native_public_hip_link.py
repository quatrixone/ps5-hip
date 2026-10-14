"""Compile genuine upstream HIP units against the PS5 ROCclr GPU port."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', action='append')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    checkout = Path(__file__).resolve().parents[1]
    root = checkout.parents[1]
    source = checkout / 'results/native-source-v1'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline = root / '.local/g5-clr-gpu-startup/results/native-title-v1/commands.json'
    commands = json.loads(baseline.read_text())
    previous = next(item['argv'] for item in commands if item['label'] == 'compile-0')
    flags = previous[:previous.index('-c')]
    old_source = str(root / '.local/g5-clr-gpu-startup/results/source-v1/source')
    flags = [value.replace(old_source, str(source)) for value in flags]
    flags[1:1] = ['-I' + str(source / 'hipamd/src'), '-I' + str(source / 'hipamd/include'),
                  '-I' + str(checkout / 'ports/clr')]
    flags += ['-DPS5HIP_PUBLIC_HIP', '-DDISABLE_DIRECT_DISPATCH', '-ffunction-sections',
              '-fdata-sections']
    names = args.unit or []
    if args.all:
        names += [p.stem for p in sorted((source / 'hipamd/src').glob('*.cpp'))
                  if p.stem not in ('fixme', 'hip_runtime') and p.stem not in names]
    assert names

    def compile_one(name):
        path = source / 'hipamd/src' / (name + '.cpp')
        assert path.is_file(), path
        extra = (['-include', 'hip/hip_runtime.h', '-include', 'hip/hip_deprecated.h',
                  '-include', 'hip/amd_detail/amd_hip_gl_interop.h']
                 if name in ('hip_api_trace', 'hip_table_interface', 'hip_table_interface_c') else [])
        argv = flags + extra + ['-c', str(path), '-o', str(output / (name + '.o'))]
        run = subprocess.run(argv, capture_output=True, text=True, timeout=240)
        (output / (name + '.stdout')).write_text(run.stdout)
        (output / (name + '.stderr')).write_text(run.stderr)
        print(name, run.returncode, flush=True)
        return {'unit': name, 'argv': argv, 'exit_code': run.returncode,
                'source_sha256': sha(path)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        recorded = list(pool.map(compile_one, names))
    report = {'classification': 'native-upstream-public-hip-compile-probe',
              'baseline_commands_sha256': sha(baseline),
              'source_preparation_sha256': sha(source / 'public-hip-preparation.json'),
              'correlation_patch_sha256': sha(root / 'ports/clr/ps5-hip-correlation.patch'),
              'profile_header_sha256': sha(source / 'hipamd/src/hip_prof_api.h'),
              'native_graphics_guard_sha256': sha(checkout / 'tools/guard_native_hip_gl.py'),
              'graphics_source_sha256': sha(source / 'hipamd/src/hip_gl.cpp'),
              'commands': recorded, 'native_linked': False, 'native_executed': False,
              'artifacts': {path.name: sha(path) for path in output.iterdir() if path.is_file()}}
    (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if all(item['exit_code'] == 0 for item in recorded) else 1


if __name__ == '__main__':
    raise SystemExit(main())
