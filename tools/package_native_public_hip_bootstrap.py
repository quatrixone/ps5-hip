"""Stage and fsck a frozen PPSA99014 public HIP startup image offline."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(folder):
    assert not any(p.is_symlink() for p in folder.rglob('*'))
    return {p.relative_to(folder).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in folder.rglob('*') if p.is_file()}


parser = argparse.ArgumentParser()
parser.add_argument('--results-dir-source', type=Path, required=True)
args = parser.parse_args()
checkout = Path(__file__).resolve().parents[1]
root = checkout.parents[1]
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=checkout).strip()
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
converted = checkout / 'results/native-hip-console-convert-v1'
receipt = converted / 'conversion.json'
link = checkout / 'results/native-hip-console-link-v1/receipt.json'
assert json.loads(receipt.read_text())['console_title'] is True
assert json.loads(link.read_text())['console_title'] is True
image = converted / 'eboot.bin'
assert sha(image) == json.loads(receipt.read_text())['artifacts']['eboot.bin']
prior = root / '.local/g5-clr-gpu-startup/dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42'
prior_manifest = json.loads((prior / 'manifest.json').read_text())
assert prior_manifest['title_id'] == 'PPSA99014'
results_source = args.results_dir_source.resolve()
directory = results_source.read_text()
assert 2 <= len(directory) <= 240 and directory.startswith('/') and directory[-1] != '/'
assert all(32 <= ord(c) <= 126 and c != '\\' for c in directory)
assert all(part not in ('', '.', '..') for part in directory[1:].split('/'))
files = {'eboot.bin': image.read_bytes(), 'results-dir.txt': directory.encode('ascii')}
for name in ('lapy.elf', 'sce_module/libc.prx', 'sce_sys/param.json', 'sce_sys/icon0.png'):
    path = prior / 'app' / name
    assert sha(path) == prior_manifest['files'][name]['sha256']
    files[name] = path.read_bytes()
assert json.loads(files['sce_sys/param.json'])['titleId'] == 'PPSA99014'
stage = checkout / 'dist/PPSA99014' / ('gfx1013-public-bootstrap-' + sha(image)[:12])
stage.mkdir(parents=True, exist_ok=False)
for name, data in files.items():
    path = stage / 'app' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
(stage / 'build-receipt.json').write_bytes(receipt.read_bytes())
bindings = {str(p): sha(p) for p in (Path(__file__).resolve(), receipt, link, image,
                                     results_source, prior / 'manifest.json',
                                     prior / 'package-receipt.json', checkout / 'title.json')}
manifest = {'schema': 1, 'title_id': 'PPSA99014', 'target': 'gfx1013',
            'kernel': 'public_hip_bootstrap', 'source_commit': commit,
            'build_receipt_sha256': sha(receipt), 'hardware_validated': False,
            'expected_log': directory + '/ps5-hip-public-bootstrap.log',
            'inputs': bindings, 'files': inventory(stage / 'app')}
(stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
deps = json.loads((root / 'dependencies.json').read_text())['ufs2tool']
tool = root / '.deps' / ('ufs2tool-' + deps['revision'])
tool_receipt = json.loads((tool / 'receipt.json').read_text())
assert tool_receipt['revision'] == deps['revision']
for name, digest in tool_receipt['artifacts'].items():
    assert sha(tool / 'bin' / name) == digest
command = ['dotnet', str(tool / 'bin/UFS2Tool.dll')]
env = dict(os.environ, DOTNET_ROLL_FORWARD='Major', DOTNET_CLI_TELEMETRY_OPTOUT='1')
with tempfile.TemporaryDirectory(prefix='.pack-', dir=stage) as temporary:
    temporary = Path(temporary)
    packed = temporary / 'title.ffpkg'
    extracted = temporary / 'extracted'
    with (stage / 'package.log').open('w') as log:
        for arguments in (
            ['makefs', '-S', '4096', '-b', '20%', '-t', 'ffs', '-o',
             'version=2,bsize=32768,fsize=4096,minfree=0,softupdates=0,optimization=space',
             str(packed), str(stage / 'app')],
            ['fsck_ufs', '-f', '-n', str(packed)],
            ['extract', str(packed), str(extracted)]):
            subprocess.run(command + arguments, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=180)
    assert inventory(extracted) == manifest['files']
    data = packed.read_bytes()
    assert data[0x1055c:0x10560] == bytes.fromhex('19015419')
    final = stage / 'PPSA99014.ffpkg'
    with final.open('xb') as output:
        output.write(data)
assert all(sha(path) == digest for path, digest in bindings.items())
package_receipt = {'classification': 'offline-public-hip-bootstrap-package',
                   'title_id': 'PPSA99014', 'source_commit': commit,
                   'manifest_sha256': sha(stage / 'manifest.json'),
                   'image': {'file': final.name, 'sha256': sha(final),
                             'bytes': final.stat().st_size},
                   'filesystem_check': 'passed', 'extracted_files_match': True,
                   'hardware_validated': False}
(stage / 'package-receipt.json').write_text(json.dumps(package_receipt, indent=2) + '\n')
print(stage)
