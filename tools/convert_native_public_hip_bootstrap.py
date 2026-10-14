"""Convert and sign the strict offline public HIP bootstrap PIE."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--link', type=Path)
parser.add_argument('--console-title', action='store_true')
args = parser.parse_args()
checkout = Path(__file__).resolve().parents[1]
root = checkout.parents[1]
source = args.link.resolve() if args.link else checkout / 'results/native-hip-link-v4'
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
supplier = root / '.local/g4-multi-reduction-integration/build/native/gfx1013-hip_multi_query-integration-v2'
converter = supplier / 'ps5-native-tool'
sdk = root / '.deps/sdk/target/lib'
commands = [
    ['link', '--in', str(source / 'title.elf'), '--out', str(out / 'eboot.elf'),
     '--stub-dir', str(sdk), '--stub', str(supplier / 'libSceAgc.so'),
     '--stub', str(supplier / 'libSceAgcDriver.so'), '--file-name', 'eboot.elf'],
    ['self', '--sign', '--in', str(out / 'eboot.elf'), '--out', str(out / 'eboot.bin')],
    ['self', '--extract', '--file', str(out / 'eboot.bin'), '--out', str(out / 'extracted.elf')],
]
if args.console_title:
    commands[0] += ['--import-library',
                    'libSceSystemService:sceShellCoreUtilExitApp:libSceShellCoreUtil']
inputs = {str(path): sha(path) for path in
          (Path(__file__).resolve(), source / 'receipt.json', source / 'title.elf',
           converter, supplier / 'libSceAgc.so', supplier / 'libSceAgcDriver.so')}
records = []
for index, command in enumerate(commands):
    argv = [str(converter), *command]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    (out / (str(index) + '.stdout')).write_text(result.stdout)
    (out / (str(index) + '.stderr')).write_text(result.stderr)
    records.append({'argv': argv, 'exit_code': result.returncode})
    assert result.returncode == 0, result.stderr
assert all(sha(path) == digest for path, digest in inputs.items())
report = {'classification': 'offline-native-public-hip-conversion',
          'inputs': inputs, 'commands': records,
          'console_title': args.console_title,
          'artifacts': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
          'native_executed': False}
(out / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
print(sha(out / 'conversion.json'))
