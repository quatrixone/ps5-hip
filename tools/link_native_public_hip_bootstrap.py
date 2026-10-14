"""Strict offline PS5 link of the genuine upstream hipInit/device-count closure."""
from pathlib import Path
import argparse
import concurrent.futures
import hashlib
import json
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


checkout = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--console-title', action='store_true')
args = parser.parse_args()
root = checkout.parents[1]
source = checkout / 'results/native-source-v1'
baseline = root / '.local/g5-clr-gpu-startup/results/native-title-v1'
hip = checkout / 'results/native-hip-compile-all-v2'
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
commands = json.loads((baseline / 'commands.json').read_text())
compiled = {int(item['label'][8:]): item['argv'] for item in commands
            if item['label'].startswith('compile-')}
hip_commands = json.loads((hip / 'receipt.json').read_text())['commands']
old_source = str(root / '.local/g5-clr-gpu-startup/results/source-v1/source')
bindings = {str(baseline / 'commands.json'): sha(baseline / 'commands.json'),
            str(baseline / 'receipt.json'): sha(baseline / 'receipt.json'),
            str(source / 'public-hip-preparation.json'): sha(source / 'public-hip-preparation.json'),
            str(hip / 'receipt.json'): sha(hip / 'receipt.json'),
            str(Path(__file__).resolve()): sha(__file__)}
recorded = []
units = []


def run(label, argv):
    process = subprocess.run(argv, capture_output=True, text=True, timeout=240)
    (out / (label + '.stdout')).write_text(process.stdout)
    (out / (label + '.stderr')).write_text(process.stderr)
    recorded.append({'label': label, 'argv': argv, 'exit_code': process.returncode})
    print(label, process.returncode, flush=True)
    return process.returncode


def native_flags(index):
    command = compiled[index]
    flags = command[:command.index('-c')]
    old_checkout = str(root / '.local/g5-clr-gpu-startup')
    flags = [v.replace(old_source, str(source)).replace(old_checkout, str(checkout))
             for v in flags]
    flags[1:1] = ['-I' + str(checkout / 'ports/clr'),
                  '-I' + str(checkout / 'ports/elevation'),
                  '-I' + str(source / 'hipamd/src'), '-I' + str(source / 'hipamd/include')]
    if not args.console_title:
        flags = [v for v in flags if v not in ('-DPS5HIP_CLR_GPU_TITLE=1',
                                              '-DPS5HIP_SHELL_EXIT_PROBE=1')]
    return flags + [
        '-DPS5HIP_PUBLIC_HIP', '-DDISABLE_DIRECT_DISPATCH', '-ffunction-sections',
        '-fdata-sections']


rebuild = {4: source / 'rocclr/device/device.cpp',
           14: source / 'rocclr/platform/commandqueue.cpp',
           21: source / 'rocclr/platform/runtime.cpp',
           24: source / 'rocclr/thread/thread.cpp',
           27: checkout / 'ports/clr/ps5-clr-tls.cpp',
           28: checkout / 'ports/clr/ps5-device.cpp',
           30: checkout / 'tools/clr_gpu/watchdog.cpp',
           32: checkout / 'tools/public_hip_app_crt.cpp',
           45: checkout / 'ports/elevation/elevation.cpp',
           46: checkout / 'tools/public_hip_bootstrap_console_title.cpp' if args.console_title
               else checkout / 'tools/public_hip_bootstrap_title.cpp'}

selected = list(range(29)) + list(range(34, 44)) + [32, 46]
if args.console_title:
    selected += [30, 44, 45]


def source_for(index):
    if index in rebuild:
        return rebuild[index]
    original = Path(compiled[index][compiled[index].index('-c') + 1])
    old_checkout = root / '.local/g5-clr-gpu-startup'
    if str(original).startswith(old_source + '/'):
        return source / original.relative_to(old_source)
    if str(original).startswith(str(old_checkout) + '/'):
        return checkout / original.relative_to(old_checkout)
    raise ValueError(f'Unmapped source {index}: {original}')


def compile_one(pair):
    index, path = pair
    obj = out / (str(index) + '.o')
    status = run('compile-' + str(index), native_flags(index if index < 46 else 28) +
                 ['-c', str(path), '-o', str(obj)])
    return {'unit': str(index), 'source': str(path), 'source_sha256': sha(path),
            'object': str(obj), 'object_sha256': sha(obj) if status == 0 else None,
            'exit_code': status}


def compile_hip(command):
    argv = list(command['argv'])
    old_checkout = str(root / '.local/g5-clr-gpu-startup')
    argv = [v.replace(old_checkout, str(checkout)) for v in argv]
    path = Path(argv[argv.index('-c') + 1])
    obj = out / ('hip-' + command['unit'] + '.o')
    argv[argv.index('-o') + 1] = str(obj)
    status = run('compile-hip-' + command['unit'], argv)
    return {'unit': 'hip-' + command['unit'], 'source': str(path),
            'source_sha256': sha(path), 'object': str(obj),
            'object_sha256': sha(obj) if status == 0 else None,
            'exit_code': status}


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    units = list(pool.map(compile_one, ((i, source_for(i)) for i in selected)))
    units += list(pool.map(compile_hip, hip_commands))
if any(item['exit_code'] for item in units):
    raise SystemExit(1)

assert len(units) == (74 if args.console_title else 71)
objects = [Path(item['object']) for item in units]
for item in units:
    bindings[item['source']] = item['source_sha256']
    bindings[item['object']] = item['object_sha256']
assert len(set(objects)) == len(objects)

link = next(item['argv'] for item in commands if item['label'] == 'native-title-link')
first = link.index(str(baseline / '0.o'))
last = link.index('--start-group')
link[first:last] = [str(p) for p in objects]
link[link.index('-o') + 1] = str(out / 'title.elf')
link[1:1] = ['--gc-sections']
for item in link:
    path = Path(item)
    if path.is_file():
        bindings[str(path)] = sha(path)
status = run('link', link)
report = {'classification': 'strict-native-public-hip-bootstrap-link-probe',
          'bindings': bindings, 'commands': recorded, 'object_count': len(objects),
          'units': units, 'clean_closure': True,
          'console_title': args.console_title,
          'native_linked': status == 0, 'native_executed': False,
          'artifacts': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
(out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(status)
