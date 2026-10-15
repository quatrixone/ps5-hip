"""Install the locked Linux x86-64 CPU converter in an isolated directory."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import tarfile

from fetch_llm_model import fetch_file

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'llm-conversion-inputs.json'
REQUIREMENTS = ROOT / 'ports/llm-conversion-requirements.txt'


def prepare(output):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('locked converter environment requires Linux x86-64 (WSL supported)')
    lock_raw = LOCK.read_bytes()
    lock = json.loads(lock_raw)
    requirements_raw = REQUIREMENTS.read_bytes()
    if hashlib.sha256(requirements_raw).hexdigest() != lock['requirements_sha256']:
        raise ValueError('conversion requirements differ from lock')
    identity = dict(lock_sha256=hashlib.sha256(lock_raw).hexdigest())
    tools = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in (Path(__file__), ROOT / 'tools/fetch_llm_model.py')}
    marker = output / 'configuration.json'
    if output.exists():
        previous = json.loads(marker.read_text()) if marker.is_file() else {}
        legacy = (set(previous) == {'lock_sha256', 'tool_sha256'}
                  and re.fullmatch(r'[0-9a-f]{64}', previous.get('tool_sha256', ''))
                  and previous['lock_sha256'] == identity['lock_sha256'])
        if previous != identity and not legacy:
            raise ValueError('use a fresh environment directory; existing identity is unrecognized')
    else:
        output.mkdir(parents=True)
        marker.write_text(json.dumps(identity, indent=2) + '\n')
    receipt = output / 'receipt.json'
    if receipt.exists():
        raw = receipt.read_bytes()
        old = json.loads(raw)
        if old.get('classification') != 'isolated-converter-installed' or old.get('lock_sha256') != identity['lock_sha256']:
            raise ValueError('existing receipt is unrecognized; preserve and inspect it')
        history = output / 'receipts'
        history.mkdir(exist_ok=True)
        archived = history / (hashlib.sha256(raw).hexdigest() + '.json')
        if archived.exists() and archived.read_bytes() != raw:
            raise ValueError('prior receipt archive differs')
        if not archived.exists():
            archived.write_bytes(raw)
        receipt.unlink()
    archive = output / 'python.tar.gz'
    fetch_file(archive, lock['python'])
    python = output / 'runtime/python/bin/python3'
    if not (output / 'runtime-ready').is_file():
        runtime = output / 'runtime'
        if runtime.exists():
            raise ValueError('partial Python runtime preserved; use a fresh environment directory')
        runtime.mkdir()
        with tarfile.open(archive) as bundle:
            # Optional terminal database contains link chains unsupported on /mnt/c.
            bundle.extractall(runtime, members=(m for m in bundle if not m.name.startswith('python/share/terminfo/')),
                              filter='data')
        (output / 'runtime-ready').write_text(lock['python']['sha256'] + '\n')
    python_version = subprocess.check_output([str(python), '--version'], text=True).strip()
    if python_version != lock['python_version']:
        raise ValueError('unexpected standalone Python version')
    venv = output / 'venv'
    if not venv.exists():
        subprocess.run([str(python), '-m', 'venv', str(venv)], check=True)
    interpreter = venv / 'bin/python'
    normalize = lambda name: name.lower().replace('_', '-').replace('.', '-')

    def installed_packages():
        return json.loads(subprocess.check_output([str(interpreter), '-m', 'pip', 'list', '--format=json'], text=True))

    def matches(packages):
        actual = {normalize(row['name']): row['version'] for row in packages}
        return all(actual.get(normalize(name)) == version for name, version in lock['packages'].items())

    reused = matches(installed_packages())
    with (output / 'install.stdout').open('w') as stdout, (output / 'install.stderr').open('w') as stderr:
        if not reused:
            subprocess.run([str(interpreter), '-m', 'pip', 'install', '--no-cache-dir', '--no-index', '--no-compile',
                            '--require-hashes', '--no-deps', '-r', str(REQUIREMENTS)],
                           stdout=stdout, stderr=stderr, check=True, timeout=1800)
        subprocess.run([str(interpreter), '-m', 'pip', 'check'], stdout=stdout, stderr=stderr,
                       check=True, timeout=120)
    installed = installed_packages()
    if not matches(installed):
        raise ValueError('installed packages differ from lock')
    if LOCK.read_bytes() != lock_raw or REQUIREMENTS.read_bytes() != requirements_raw:
        raise ValueError('conversion inputs changed during preparation')
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in tools.items()):
        raise ValueError('conversion setup tool changed during preparation')
    marker.write_text(json.dumps(identity, indent=2) + '\n')
    receipt.write_text(json.dumps(dict(schema=1, classification='isolated-converter-installed',
        **identity, tools=tools, python_version=python_version, packages=installed,
        reused_installed_packages=reused, model_executed=False), indent=2) + '\n')
    print('isolated conversion environment ready', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.deps/llm-conversion-env')
    prepare(parser.parse_args().output.resolve())
