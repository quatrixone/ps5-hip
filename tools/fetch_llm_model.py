"""Fetch exact locked model sources; never convert or execute a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'models/tinyllama-1.1b-chat-v1.0.json'
CHUNK = 1024 * 1024


def verify(path, size, digest):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f'incorrect file type or size: {path}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(CHUNK), b''):
            h.update(block)
    if h.hexdigest() != digest:
        raise ValueError(f'incorrect SHA256: {path}; preserve and inspect it manually')


def fetch_file(path, spec):
    size, digest = spec['bytes'], spec['sha256']
    if path.exists() or path.is_symlink():
        verify(path, size, digest)
        return
    request = urllib.request.Request(spec['url'], headers={
        'Accept-Encoding': 'identity', 'User-Agent': 'ps5-hip-model-fetch/1'})
    temporary = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get('Content-Length')
            if (response.status != 200 or response.headers.get('Content-Encoding', 'identity') != 'identity'
                    or (length is not None and (not length.isdigit() or int(length) != size))):
                raise ValueError('unexpected response headers; body not read')
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.model-fetch-', delete=False) as stream:
                temporary = Path(stream.name)
                h, count = hashlib.sha256(), 0
                while True:
                    block = response.read(min(CHUNK, size - count + 1))
                    if not block:
                        break
                    count += len(block)
                    if count > size:
                        raise ValueError('response exceeds locked size')
                    stream.write(block)
                    h.update(block)
                if count != size or h.hexdigest() != digest:
                    raise ValueError('download size or SHA256 differs from lock')
                stream.flush()
                os.fsync(stream.fileno())
        # Hard-link publication is atomic and refuses to replace an existing file.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink()


def download(output, receipt, lock_path=LOCK):
    tool_path = Path(__file__)
    tool_sha256 = hashlib.sha256(tool_path.read_bytes()).hexdigest()
    if receipt.exists() or receipt.is_symlink():
        raise ValueError('receipt must be a fresh path; preserve previous run evidence')
    raw = lock_path.read_bytes()
    lock = json.loads(raw)
    specs = dict(lock['metadata'])
    specs['model.safetensors'] = dict(url=lock['weights']['url'],
                                    bytes=lock['weights']['file_bytes'],
                                    sha256=lock['weights']['expected_file_sha256'])
    for name, spec in specs.items():
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..')
                or type(spec['bytes']) is not int or spec['bytes'] <= 0
                or not re.fullmatch(r'[0-9a-f]{64}', spec['sha256'])
                or not spec['url'].startswith('https://huggingface.co/')):
            raise ValueError('invalid model lock entry')
    output.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        fetch_file(output / name, spec)
        print(f'verified {name}: {spec["bytes"]} bytes', flush=True)
    # Recheck the full input set before publishing evidence, including reused files.
    for name, spec in specs.items():
        verify(output / name, spec['bytes'], spec['sha256'])
    if lock_path.read_bytes() != raw:
        raise ValueError('model lock changed during download')
    if hashlib.sha256(tool_path.read_bytes()).hexdigest() != tool_sha256:
        raise ValueError('download tool changed during download')
    result = dict(schema=1, classification='locked-model-source-verified',
                  model=lock['model'], revision=lock['revision'],
                  lock_sha256=hashlib.sha256(raw).hexdigest(),
                  tool_sha256=tool_sha256,
                  files=specs, output=str(output.resolve()),
                  model_executed=False, converted=False, native_executed=False)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=receipt.parent, prefix='.model-receipt-', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(result, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, receipt)
        finally:
            temporary.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'models/tinyllama-source')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    download(args.output, args.receipt)
