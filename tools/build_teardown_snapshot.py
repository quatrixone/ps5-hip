#!/usr/bin/env python3
"""Offline-only build and import audit of the read-only teardown diagnostic."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_IMPORTS = {
    'getfsstat', 'getuid', 'geteuid', 'kill', '__error', 'sceSystemServiceGetAppId',
    'sceLncUtilGetAppIdOfRunningBigApp', 'printf', 'free', 'calloc', 'memchr',
    'qsort', 'putchar', 'strlen', 'strncmp', 'strcmp', 'memcmp', 'memcpy', 'memset',
}
NEEDED = {'libSceLibcInternal.sprx', 'libSceNet.sprx',
          'libSceSystemService.sprx', 'libkernel_sys.sprx'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def symbols(text, undefined):
    result = set()
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 8 and re.fullmatch(r'\d+:', fields[0]):
            if (fields[6] == 'UND') == undefined:
                result.add(fields[7])
    return result


def audit(import_text, providers):
    needed = set(re.findall(r'^  (\S+\.sprx)$', import_text, re.M))
    imported = symbols(import_text, True)
    required = {'getfsstat', 'kill', 'sceSystemServiceGetAppId',
                'sceLncUtilGetAppIdOfRunningBigApp'}
    if needed != NEEDED or not required <= imported or not imported <= ALLOWED_IMPORTS:
        raise ValueError('Unexpected library or missing/forbidden diagnostic import')
    owners = sorted(name for name, text in providers.items() if 'getfsstat' in symbols(text, False))
    if set(providers) != needed or owners != ['libkernel_sys.sprx']:
        raise ValueError('getfsstat must have exactly the pinned libkernel_sys provider')
    return {'needed_libraries': sorted(needed), 'imports': sorted(imported),
            'getfsstat_provider': owners[0], 'release_certification': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    sdk = args.sdk.resolve()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'build').resolve()):
        parser.error('Use a new output directory inside this isolated checkout/build')
    lock = json.loads((ROOT / 'dependencies.json').read_text())
    if (sdk / '.ps5-sdk-revision').read_text().strip() != lock['sdk']['revision']:
        parser.error('SDK revision mismatch')
    config = shutil.which('llvm-config-18')
    readelf = shutil.which('llvm-readelf-18')
    objdump = shutil.which('llvm-objdump-18')
    if not all((config, readelf, objdump)):
        parser.error('Pinned LLVM18 tools unavailable')
    if subprocess.check_output([config, '--version'], text=True).strip() != lock['llvm']['version']:
        parser.error('LLVM version mismatch')
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite failed or frozen output.
    receipt = {'schema': 1, 'classification': 'building', 'console_access': False,
               'hardware_validated': False, 'release_certification': False,
               'sdk_revision': lock['sdk']['revision'], 'llvm_version': lock['llvm']['version'],
               'commands': [], 'source_sha256': {name: sha(ROOT / name) for name in
                   ('tools/teardown_snapshot.cpp', 'tools/teardown_snapshot.h',
                    'tools/build_teardown_snapshot.py', 'dependencies.json')},
               'sdk_inputs_sha256': {name: sha(sdk / name) for name in
                   ('bin/prospero-clang', 'bin/clang', 'target/include/sys/mount.h',
                    'target/lib/crt1.o', 'target/lib/libc.a')},
               'tools_sha256': {name: sha(Path(path)) for name, path in
                   (('llvm-config', config), ('llvm-readelf', readelf), ('llvm-objdump', objdump))}}

    def run(command, name):
        result = subprocess.run(list(map(str, command)), cwd=ROOT,
                                env=dict(os.environ, LLVM_CONFIG=config),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = output / name
        with path.open('xb') as stream:
            stream.write(result.stdout)
        receipt['commands'].append({'argv': list(map(str, command)), 'exit_code': result.returncode,
                                    'log': name, 'sha256': sha(path)})
        if result.returncode:
            raise RuntimeError('Offline command failed: ' + name)
        return result.stdout.decode('utf-8')

    try:
        elf = output / 'teardown-snapshot.elf'
        run([sdk / 'bin/prospero-clang', '-std=c++17', '-O2', '-Wall', '-Wextra',
             '-Wpedantic', '-Werror', '-fno-exceptions', '-fno-rtti',
             '-MD', '-MF', output / 'headers.d',
             '-lkernel_sys', '-lSceSystemService', ROOT / 'tools/teardown_snapshot.cpp',
             '-o', elf], 'compile.log')
        imported = run([readelf, '--dyn-symbols', '--needed-libs', elf], 'imports.log')
        providers = {}
        for library in sorted(NEEDED):
            stub = sdk / 'target/lib' / library.replace('.sprx', '.so')
            receipt['sdk_inputs_sha256']['target/lib/' + stub.name] = sha(stub)
            providers[library] = run([readelf, '--dyn-symbols', stub], library + '.exports.log')
        receipt['import_audit'] = audit(imported, providers)
        run([objdump, '-d', '--disassemble-symbols=main', elf], 'main-disassembly.log')
        dependencies = shlex.split((output / 'headers.d').read_text().split(':', 1)[1].replace('\\\n', ' '))
        receipt['included_file_sha256'] = {name: sha(Path(name)) for name in sorted(set(dependencies))}
        for name, expected in receipt['source_sha256'].items():
            if sha(ROOT / name) != expected:
                raise RuntimeError('Diagnostic source changed during build: ' + name)
        for name, expected in receipt['sdk_inputs_sha256'].items():
            if sha(sdk / name) != expected:
                raise RuntimeError('SDK input changed during build: ' + name)
        receipt['artifact'] = {'file': elf.name, 'bytes': elf.stat().st_size, 'sha256': sha(elf)}
        receipt['classification'] = 'offline-built-imports-checked-not-executed'
        receipt['limits'] = ['Firmware statfs ABI and global namespace visibility unqualified.',
                             'Known PID queries do not enumerate detached title descendants.',
                             'The diagnostic always returns2 and release=unknown; no gate acceptance.']
    except Exception as error:
        receipt['classification'] = 'failed'
        receipt['error'] = repr(error)
        raise
    finally:
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Offline diagnostic built; no console execution:', elf)


if __name__ == '__main__':
    main()
