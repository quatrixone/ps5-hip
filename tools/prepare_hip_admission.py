"""Apply the bounded public-HIP patch to a separately qualified CLR adapter tree."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess

if not __debug__:
    raise RuntimeError("Validation requires ordinary Python")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-source', type=Path, required=True)
    p.add_argument('--hip-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    patch = root / 'ports/clr/ps5-public-hip.patch'
    manifest_path = root / 'ports/clr/ps5-public-hip.json'
    manifest = json.loads(manifest_path.read_text())
    assert sha(patch) == manifest['patch_sha256']
    assert manifest['files']
    inputs = {str(patch): sha(patch), str(manifest_path): sha(manifest_path),
              str(Path(__file__).resolve()): sha(__file__)}
    for name, record in manifest['files'].items():
        source = (a.hip_source if name.startswith('hipamd/') else a.base_source) / name
        assert sha(source) == record['base_sha256'], name
        inputs[str(source.resolve())] = sha(source)
    out = a.output.resolve()
    assert not out.exists(), 'Use a fresh output directory'
    shutil.copytree(a.base_source, out)
    shutil.copytree(a.hip_source / 'hipamd', out / 'hipamd', dirs_exist_ok=True)
    tool = Path(shutil.which('patch')).resolve()
    inputs[str(tool)] = sha(tool)
    command = [str(tool), '--batch', '--forward', '-p1', '-i', str(patch)]
    result = subprocess.run(command, cwd=out, capture_output=True, text=True, timeout=30)
    (out / 'public-hip-patch.stdout').write_text(result.stdout)
    (out / 'public-hip-patch.stderr').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    for name, record in manifest['files'].items():
        assert sha(out / name) == record['result_sha256'], name
    assert all(sha(path) == digest for path, digest in inputs.items())
    receipt = dict(classification='exact-public-hip-patch-roundtrip', inputs=inputs,
                   command=command, exit_code=result.returncode,
                   files={n: sha(out / n) for n in manifest['files']})
    (out / 'public-hip-preparation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(sha(out / 'public-hip-preparation.json'))

if __name__ == '__main__':
    main()
