#!/usr/bin/env python3
"""Reproduce device objects in a fresh package root using networkless bubblewrap.

The root must be independently extracted from authenticated distribution packages.
Only selected source files and a new output directory enter the sandbox.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rootfs', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    rootfs, output = args.rootfs.resolve(), args.output.resolve()
    if rootfs == pathlib.Path('/') or not (rootfs / 'usr/bin/clang++-18').exists():
        parser.error('provide a separate prepared compiler root')
    if not (rootfs / 'work').is_dir():
        parser.error('create an empty /work mount point in the package root first')
    if output.exists() or output.is_relative_to(rootfs):
        parser.error('output must be a new directory outside the package root')
    bwrap = shutil.which('bwrap')
    if not bwrap:
        parser.error('bubblewrap is required')
    sources = ('tools/build_canaries.py', 'dependencies.json', 'kernels/canary.hip', 'kernels/resources.hip')
    output.mkdir(parents=True)
    for name in sources:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    command = [bwrap, '--unshare-all', '--die-with-parent', '--new-session',
               '--ro-bind', str(rootfs), '/', '--proc', '/proc', '--dev', '/dev',
               '--tmpfs', '/tmp', '--bind', str(output), '/work', '--chdir', '/work',
               '--clearenv', '--setenv', 'PATH', '/usr/bin:/bin',
               '--setenv', 'LC_ALL', 'C', '--', '/usr/bin/python3', '/work/tools/build_canaries.py']
    for fixture in ('canary', 'resources'):
        with (output / (fixture + '.log')).open('w') as log:
            subprocess.run(command + ['--fixture', fixture], check=True, stdout=log, stderr=subprocess.STDOUT)
    objects = {}
    for target in ('gfx1013', 'gfx1030'):
        for name in ('canary.hsaco', 'resources/resources.hsaco'):
            relative = pathlib.Path('build/canaries') / target / name
            actual, expected = digest(output / relative), digest(ROOT / relative)
            if actual != expected:
                raise RuntimeError(f'isolated object differs: {relative}: {actual} != {expected}')
            objects[f'{target}/{name}'] = actual
    report = {'schema': 1, 'classification': 'isolated-userspace-reproduction-pass',
              'hardware_validated': False, 'network_disabled_during_build': True,
              'sources': {name: digest(ROOT / name) for name in sources}, 'objects': objects,
              'package_root_os_release': (rootfs / 'etc/os-release').read_text(),
              'limits': ['Shared host kernel and CPU; not a second physical machine',
                         'Root provenance must be recorded separately',
                         'No native title or GPU execution qualification'],
              'recipe_sha256': digest(pathlib.Path(__file__))}
    (output / 'reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
