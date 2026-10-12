#!/usr/bin/env python3
"""Offline pinned-header memory ABI control: Linux syscall model and PS5 link only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from native_import_libraries_test import read_imports, nid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ref, out = args.reference_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, *command):
        argv = [str(part) for part in command]
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=180)
        (out / (label + '.stdout')).write_text(result.stdout)
        (out / (label + '.stderr')).write_text(result.stderr)
        commands.append({'label': label, 'argv': argv, 'exit_code': result.returncode})
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        require(result.returncode == 0, label + ' failed: ' + result.stderr[-3000:])
        return result.stdout

    evidence_path = ROOT / 'evidence/offline-compatible-hip-headers.json'
    evidence = json.loads(evidence_path.read_text())
    fixtures = ref / 'results/hip-host-abi-spike/rocm-6.2.4'
    include_dirs, headers = [], {}
    for key, archive, subdir in [('HIP', 'HIP.tar.gz', 'include'), ('clr', 'clr.tar.gz', 'hipamd/include')]:
        package = fixtures / archive
        require(sha(package) == evidence['input_sources'][key]['archive_sha256'], 'pinned header archive changed')
        with tarfile.open(package) as tar:
            prefix = tar.getmembers()[0].name.split('/')[0]
            header_root = fixtures / prefix
            for member in tar.getmembers():
                relative = member.name[len(prefix)+1:]
                if member.isfile() and relative.startswith(subdir + '/'):
                    path = header_root / relative
                    require(path.read_bytes() == tar.extractfile(member).read(), 'header differs from pinned archive')
                    headers[key + '/' + relative] = sha(path)
        include_dirs.append(header_root / subdir)
    generated = fixtures / 'package-build/include/hip/hip_version.h'
    require(sha(generated) == evidence['raw_file_hashes']['package-stage/include/hip/hip_version.h']['sha256'],
            'generated HIP version changed')
    include_dirs.append(generated.parents[1])
    lock = json.loads((ROOT / 'dependencies.json').read_text())
    platform = ref / '.deps' / ('sdk-source-' + lock['sdk']['revision']) / 'platform/include'
    includes = [part for directory in [ROOT / 'include', platform, *include_dirs] for part in ('-I', directory)]
    common = ['-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-Werror', *includes]
    sources = [ROOT / 'src/native_memory.cpp', ROOT / 'src/native_data.cpp', ROOT / 'tools/hip_memory_control.cpp',
               ROOT / 'tests/hip_memory_client.cpp']
    require('18.1.8' in run('compiler-version', 'clang++-18', '--version'), 'unexpected compiler version')
    run('linux-build', 'clang++-18', *common, '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        '-DPS5HIP_MEMORY_MOCK_CLIENT=1', *sources, ROOT / 'tests/hip_memory_syscalls.cpp', '-o', out / 'linux-control')
    log = run('linux-run', out / 'linux-control')
    require(log.count('HIP_MEMORY_CLIENT_PASS actual_native_memory=true gpu_calls=0 launch=unsupported hardware_validated=false') == 1,
            'client did not complete memory-only control')
    run('model-build', 'clang++-18', *common, '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
        *sources[:2], ROOT / 'tests/native_data_test.cpp', '-o', out / 'model')
    run('model-run', out / 'model')

    native_source = ref / '.deps' / ('native-tools-' + lock['native_tools']['revision'])
    authority_path = ROOT / 'evidence/offline-lds-native-readiness.json'
    expected = json.loads(authority_path.read_text())['profiles'][0]['native_tool_source_sha256']
    for name, digest in expected.items():
        require(sha(native_source / name) == digest, 'native export identity changed: ' + name)
    before = {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob('*') if p.is_file()}
    native = out / 'native-tools'
    shutil.copytree(native_source, native)
    sdk = ref / '.deps/sdk'
    require((sdk / '.ps5-sdk-revision').read_text().strip() == lock['sdk']['revision'], 'SDK revision mismatch')
    objects = []
    for source in [*sources, native / 'app_crt.cpp']:
        obj = out / ('ps5-' + source.stem + '.o')
        run('native-compile-' + source.stem, ROOT / 'tools/ps5-clangxx', *common, '-fPIC', '-fvisibility=hidden', '-c', source, '-o', obj)
        objects.append(obj)
    (out / 'symbols.map').write_text('{ local: *; };\n')
    libs = sdk / 'target/lib'
    sdk_names = ('libc++.a', 'libc++abi.a', 'libunwind.a', 'libc.a', 'libps5platform.a',
                 'libkernel.so', 'libSceLibcInternal.so', 'libSceNet.so')
    sdk_inputs = {name: sha(libs / name) for name in sdk_names}
    unwind = [part for name in ('eh_frame', 'eh_frame_hdr') for part in
              ('--defsym', f'__{name}_start=ADDR(.{name})', '--defsym', f'__{name}_end=ADDR(.{name})+SIZEOF(.{name})')]
    pie, elf = out / 'llvm-pie.elf', out / 'eboot.elf'
    run('native-link', 'ld.lld-18', '-pie', '-T', native / 'ps5-pie.ld', '--eh-frame-hdr', *unwind,
        '-z', 'max-page-size=0x4000', '--hash-style=gnu', '--version-script', out / 'symbols.map', '--exclude-libs=ALL',
        '-L', libs, '-e', '_start', '-o', pie, *objects, '--start-group', *[libs / n for n in sdk_names[:5]],
        '--end-group', '--as-needed', *[libs / n for n in sdk_names[5:]])
    zlib = ref / '.deps' / ('host-zlib-' + lock['host_zlib']['version']) / 'root'
    zlib_receipt = json.loads((zlib.parent / 'receipt.json').read_text())
    require(zlib_receipt['source_sha256'] == lock['host_zlib']['sha256'], 'zlib source identity changed')
    for name, digest in zlib_receipt['artifacts'].items():
        require(sha(zlib / name) == digest, 'zlib artifact changed')
    converter = out / 'ps5-native-tool'
    run('converter-build', 'clang++-18', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
        *[native / n for n in ('native_app_builder.cpp', 'elf_object.cpp', 'sce_module_writer.cpp', 'self_container.cpp')],
        '-I', zlib / 'include', zlib / 'lib/libz.a', '-o', converter)
    run('native-convert', converter, 'link', '--in', pie, '--out', elf, '--stub-dir', libs, '--file-name', 'eboot.elf')
    symbols = run('native-symbols', 'llvm-nm-18', '-n', pie)
    modules, libraries, imports = read_imports(elf)
    named = {parts[1]: imports[nid(parts[1])] for line in symbols.splitlines()
             if len(parts := line.split()) == 2 and parts[0] == 'U' and nid(parts[1]) in imports}
    require(len(named) == len(imports), 'native import name mapping incomplete')
    memory_calls = ('sceKernelGetDirectMemorySize', 'sceKernelAllocateDirectMemory', 'sceKernelMapDirectMemory',
                    'sceKernelMunmap', 'sceKernelReleaseDirectMemory')
    require(all(name in named for name in memory_calls), 'actual native memory imports absent')
    require(not any('Agc' in name or name.startswith('hip') for name in named), 'unexpected GPU/HIP import')
    exports = ('hipMalloc', 'hipFree', 'hipMemcpy', 'hipGetLastError', 'hipPeekAtLastError', 'hipGetErrorString', 'hipLaunchKernel')
    require(all(any(line.split()[-1:] == [name] and len(line.split()) == 3 for line in symbols.splitlines())
                for name in exports), 'control C ABI definitions missing')
    (out / 'native-import-names.json').write_text(json.dumps({'modules': modules, 'libraries': libraries,
        'imports': named, 'native_executed': False}, indent=2) + '\n')
    require(before == {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob('*') if p.is_file()},
            'shared native export changed')
    require(sdk_inputs == {name: sha(libs / name) for name in sdk_names}, 'shared SDK libraries changed')
    source_files = [*sources, ROOT / 'tests/native_data_test.cpp', ROOT / 'tests/hip_memory_syscalls.cpp',
                    ROOT / 'include/ps5hip/native_data.h', ROOT / 'include/ps5hip/native_memory.h', Path(__file__)]
    receipt = {'schema': 1, 'classification': 'Pinned full-header memory ABI, host syscall model and offline PS5 link only',
               'gpu_calls': 0, 'native_executed': False, 'hardware_validated': False,
               'header_authority_sha256': sha(evidence_path), 'header_hashes': headers,
               'generated_version_sha256': sha(generated), 'native_source_authority_sha256': sha(authority_path),
               'native_source_revision': lock['native_tools']['revision'], 'native_inputs_before_after': before,
               'sdk_inputs_before_after': sdk_inputs, 'sdk_wrapper_sha256': sha(sdk / 'bin/prospero-clang++'),
               'tools': {name: {'path': shutil.which(name), 'sha256': sha(Path(shutil.which(name)))}
                         for name in ('clang++-18', 'ld.lld-18', 'llvm-nm-18')},
               'native_memory_imports': list(memory_calls), 'abi_definitions': list(exports),
               'limits': ['Serialized single host caller; no per-thread error qualification', 'Trusted valid host/output storage',
                          'No cache/GPU visibility qualification', 'No GPU/native execution',
                          'Installed compiler resource headers/static archives not hermetically inventoried'],
               'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_files},
               'artifacts': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'native_executed': False, 'gpu_calls': 0}))


if __name__ == '__main__':
    main()
