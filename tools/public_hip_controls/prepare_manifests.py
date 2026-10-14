"""Freeze exact offline inputs for the inert D/E public HIP controls."""
from pathlib import Path
import hashlib
import json

checkout = Path(__file__).resolve().parents[2]
root = checkout.parents[1]
controls = Path(__file__).resolve().parent
stage = checkout / 'dist/PPSA99014/gfx1013-public-bootstrap-19107f507822'
prior = root / 'results/clr-gpu-startup-controls-v1/deployment/deployment-20261014T011917.375604Z/deployment.json'
review = checkout / 'results/public-hip-clean-bootstrap-independent-review-v1/review.json'
image = stage / 'PPSA99014.ffpkg'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(root).as_posix()


assert sha(review) == 'b76e9437deb47bd054410f93e579acc1c1b1d51263f99803b35e4f51a96dc348'
assert sha(image) == '836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8'
package = json.loads((stage / 'package-receipt.json').read_text())
assert package['source_commit'] == 'ea37055286903f680e43b049d70730743ffa2358'
assert package['image']['bytes'] == 9437184
old = json.loads(prior.read_text())
assert old['accepted'] is True and old['image_sha256'] == '84e52f366b9c7e598d81c3d37197ff8e71d548ebe6cfe933176639066abcc8da'
history = old['prior_backups_unchanged']['after']
assert len(history) == 34
files = [review, prior, stage / 'manifest.json', stage / 'package-receipt.json', image,
         root / 'results/deploy-private-buffer-milestone-v2.py',
         root / 'results/teardown-console-control-v3-initial-focus/run.py',
         root / 'results/arithmetic-prospective-cycle2/acceptance.py',
         root / 'results/visibility-bounded-lifecycle-v1/lifecycle.py',
         root / 'evidence/offline-controllers.json',
         root / 'build/controllers/PPSA99014/receipt.json',
         root / 'build/controllers/PPSA99014/launch.elf',
         root / 'build/controllers/PPSA99014/close.elf',
         root / '.local/teardown-snapshot-v3/build/teardown-qualified-v3/teardown-snapshot.elf',
         root / '.local/ENVIRONMENT.md', Path(__file__).resolve()]
files += sorted(p for p in (stage / 'app').rglob('*') if p.is_file())
files += sorted(p for p in controls.rglob('*.py') if p.name != 'prepare_manifests.py')
all_hashes = {relative(p): sha(p) for p in files}
direct_names = [relative(p) for p in files if p == image or p == review or
                p.parent == controls / 'deployment' or p.parent == controls / 'execution' or
                p.parent == root / 'build/controllers/PPSA99014']
direct = {name: all_hashes[name] for name in direct_names}
for kind in ('deployment', 'execution'):
    manifest = {
        'classification': 'inert-public-hip-bootstrap-controls',
        'activation_forbidden': True,
        'candidate_package': relative(stage),
        'source_commit': package['source_commit'],
        'image_sha256': package['image']['sha256'],
        'stage_manifest_sha256': sha(stage / 'manifest.json'),
        'independent_package_review': {'path': relative(review), 'sha256': sha(review)},
        'deployment_binding': None,
        'prior_backups': history,
        'files_sha256': all_hashes,
        'direct_sha256': direct,
        'hardware_validated': False,
    }
    (controls / kind / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(f'{len(all_hashes)} pinned files; {len(history)} prior backups')
