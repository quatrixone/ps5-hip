#!/usr/bin/env python3
"""Offline single-TU registration control; never runs a native executable or submits GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from fat_binary_fixture_test import section
from native_import_libraries_test import read_imports, nid


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_regions(path):
    data = path.read_bytes()
    require(data[:6] == b'\x7fELF\x02\x01', 'expected ELF64LE')
    phoff = struct.unpack_from('<Q', data, 32)[0]
    width, count = struct.unpack_from('<HH', data, 54)
    require(width == 56 and phoff + count * width <= len(data), 'invalid program table')
    headers = [struct.unpack_from('<IIQQQQQQ', data, phoff+i*width) for i in range(count)]

    def read(address, size):
        matches = [h for h in headers if h[0] == 1 and h[3] <= address and
                   address - h[3] <= h[5] and size <= h[5] - (address - h[3])]
        require(len(matches) == 1, 'extent not in one stored PT_LOAD')
        h = matches[0]
        at = h[2] + address - h[3]
        require(at + size <= len(data), 'PT_LOAD outside file')
        return data[at:at+size], h

    dynamic = next(h for h in headers if h[0] == 2)
    tags = dict(struct.unpack_from('<QQ', data, at)
                for at in range(dynamic[2], dynamic[2]+dynamic[5], 16))
    relas = []
    if 7 in tags:
        raw, _ = read(tags[7], tags[8])
        require(tags.get(9) == 24 and len(raw) % 24 == 0, 'bad RELA table')
        relas = [struct.unpack_from('<QQq', raw, at) for at in range(0, len(raw), 24)]
    return read, headers, relas


def audit_image(path, symbols, expected_bundle, native=False):
    read, headers, relas = load_regions(path)
    b, e = symbols['__ps5hip_bundle_begin'], symbols['__ps5hip_bundle_end']
    w, we = symbols['__ps5hip_wrappers_begin'], symbols['__ps5hip_wrappers_end']
    require(e-b == len(expected_bundle) and we-w == 24 and b % 4096 == 0, 'linked extents changed')
    bundle, bp = read(b, e-b)
    wrapper, wp = read(w, 24)
    require(bundle == expected_bundle and bp[1] == 4, 'bundle bytes/protection changed')
    require(wrapper[:8] == bytes.fromhex('4650494801000000') and wrapper[16:] == bytes(8), 'wrapper changed')
    ptr = [r for r in relas if r[0] == w+8]
    require(len(ptr) == 1 and ptr[0][1] == 8 and ptr[0][2] == b, 'wrapper pointer lacks exact RELATIVE relocation')
    # The loader applies B + addend; no native execution is inferred from this model.
    for base in (0, 0x400000, 0x123400000):
        require(base + ptr[0][2] == base + b, 'relocation model mismatch')
    require(wp[1] == 6, 'wrapper not in writable relocation load')
    if native:
        require(any(h[0] == 0x6474e552 and h[3] <= w and we <= h[3]+h[6] for h in headers),
                'converted wrapper not inside PT_GNU_RELRO')
    else:
        require(symbols['__ps5hip_relro_begin'] <= w < we <= symbols['__ps5hip_relro_end'],
                'raw wrapper not in converter-designated .data.rel.ro')
    ctor = symbols['__hip_module_ctor']
    ib, ie = symbols['__init_array_start'], symbols['__init_array_end']
    read(ib, ie-ib)
    require(any(ib <= r[0] < ie and r[1] == 8 and r[2] == ctor for r in relas), 'compiler constructor absent from init array')
    if native:
        _, _, imports = read_imports(path)
        for name in ('__hipRegisterFatBinary', '__hipRegisterFunction', '__hipUnregisterFatBinary',
                     '__hipPushCallConfiguration', '__hipPopCallConfiguration', 'hipLaunchKernel'):
            require(nid(name) not in imports, 'control ABI unexpectedly imported')
    return {'bundle': [b, e], 'wrapper': [w, we], 'init_array': [ib, ie], 'ctor': ctor,
            'wrapper_relative_addend': ptr[0][2], 'bundle_sha256': hashlib.sha256(bundle).hexdigest(),
            'relro_contract': 'PT_GNU_RELRO' if native else 'linker-designated .data.rel.ro RW load',
            'native_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-launch', action='store_true', help='Synthetic-address argument capture and packet preparation control only')
    args = parser.parse_args()
    ref, out = args.reference_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, *command):
        argv = [str(c) for c in command]
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=180)
        (out / (label+'.stdout')).write_text(r.stdout)
        (out / (label+'.stderr')).write_text(r.stderr)
        commands.append({'label': label, 'argv': argv, 'exit_code': r.returncode})
        (out / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        require(r.returncode == 0, label + ' failed: ' + r.stderr[-3000:])
        return r.stdout

    fixtures = ref / 'results/hip-host-abi-spike/rocm-6.2.4'
    evidence = json.loads((ROOT / 'evidence/offline-compatible-hip-headers.json').read_text())
    manifest = fixtures / 'compatible-emitted-abi.json'
    require(sha(manifest) == evidence['raw_file_hashes'][manifest.name]['sha256'], 'fixture manifest changed')
    cases = json.loads(manifest.read_text())
    selected = {}
    for host, suffix in [('linux', 'with-routed-workitem-slice'), ('ps5', 'device-wchar4')]:
        name = f'{host}-vector-add-gfx1013-{suffix}'
        source = fixtures / (name+'.o')
        require(sha(source) == cases[name]['sha256'], 'combined object changed')
        selected[host] = out / (host+'-combined.o')
        shutil.copyfile(source, selected[host])
        (out / (host+'.bundle')).write_bytes(section(source.read_bytes(), b'.hip_fatbin'))
    include_dirs = []
    header_hashes = {}
    for key, archive, subdir in [('HIP', 'HIP.tar.gz', 'include'), ('clr', 'clr.tar.gz', 'hipamd/include')]:
        record = evidence['input_sources'][key]
        package = fixtures / archive
        require(sha(package) == record['archive_sha256'], 'header archive changed')
        with tarfile.open(package) as tar:
            prefix = tar.getmembers()[0].name.split('/')[0]
            root = fixtures / prefix
            for member in tar.getmembers():
                relative = member.name[len(prefix)+1:]
                if member.isfile() and relative.startswith(subdir+'/'):
                    path = root / relative
                    require(path.read_bytes() == tar.extractfile(member).read(), 'header source differs from pinned archive')
                    header_hashes[key+'/'+relative] = sha(path)
        include_dirs.append(root/subdir)
    generated = fixtures / 'package-build/include/hip/hip_version.h'
    require(sha(generated) == evidence['raw_file_hashes']['package-stage/include/hip/hip_version.h']['sha256'], 'generated HIP version changed')
    include_dirs.append(generated.parents[1])
    compiler_version = run('compiler-version', 'clang++-18', '--version')
    require('18.1.8' in compiler_version, 'unexpected host compiler version')
    includes = [item for directory in [ROOT/'include', *include_dirs] for item in ('-I', directory)]
    core = [ROOT/'src'/name for name in ('code_object.cpp', 'fat_binary.cpp', 'host_registration.cpp')]
    if args.capture_launch:
        core += [ROOT/'src/host_launch.cpp', ROOT/'src/launch.cpp']
    common = ['-std=c++17', '-O1', '-g', '-Wall', '-Wextra', '-Werror', *includes]
    if args.capture_launch:
        common.append('-DPS5HIP_ARGUMENT_CAPTURE_CONTROL=1')
    linux_script = out/'linux.ld'
    linux_script.write_text('''SECTIONS {
 .hip_fatbin ALIGN(4096) : { HIDDEN(__ps5hip_bundle_begin = .); KEEP(*(.hip_fatbin)) HIDDEN(__ps5hip_bundle_end = .); }
} INSERT AFTER .rodata;
SECTIONS {
 .data.rel.ro : { *(.data.rel.ro .data.rel.ro.*) . = ALIGN(8); HIDDEN(__ps5hip_wrappers_begin = .); KEEP(*(.hipFatBinSegment)) HIDDEN(__ps5hip_wrappers_end = .); }
} INSERT BEFORE .dynamic;
HIDDEN(__ps5hip_names_begin = ADDR(.rodata));
HIDDEN(__ps5hip_names_end = ADDR(.rodata)+SIZEOF(.rodata));
HIDDEN(__init_array_start = ADDR(.init_array));
HIDDEN(__init_array_end = ADDR(.init_array)+SIZEOF(.init_array));
''')
    sanitizers = ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
    run('model-build', 'clang++-18', *common, *sanitizers, *core, ROOT/'tests/host_registration_test.cpp', '-o', out/'model')
    run('model-run', out/'model', out/'linux.bundle', 'gfx1013')
    if args.capture_launch:
        run('capture-model-build', 'clang++-18', *common, *sanitizers, *core, ROOT/'tests/host_launch_test.cpp', '-o', out/'capture-model')
        run('capture-model-run', out/'capture-model', out/'linux.bundle', 'gfx1013')
    for failure in (False, True):
        label = 'linux-partial' if failure else 'linux-normal'
        run(label+'-build', 'clang++-18', *common, *sanitizers, '-fPIE', '-pie', '-fuse-ld=lld-18',
            *(['-DPS5HIP_REGISTRATION_FAIL_SYMBOL=1'] if failure else []), *core,
            ROOT/'tools/hip_registration_control.cpp', selected['linux'], '-Wl,-T,'+str(linux_script), '-o', out/label)
        log = run(label+'-run', out/label)
        require('MAIN_PASS partial_failure='+str(int(failure)) in log and
                log.index('MAIN_PASS') < log.index('COMPILER_UNREGISTER') < log.index('REGISTRY_FINAL') and
                'valid=1 hardware_validated=false' in log, 'genuine destructor lifecycle not observed')
        if args.capture_launch and not failure:
            require('COMPILER_CAPTURE arguments=4 kernarg_bytes=288' in log and
                    'CAPTURE_OWNED after_stub_return=true caller_values_changed=true synthetic_model=true' in log,
                    'genuine compiler argument capture not observed')
    # Native source copies are private to this fresh output; no cached export is edited.
    lock = json.loads((ROOT/'dependencies.json').read_text())
    native_source = ref/'.deps'/('native-tools-'+lock['native_tools']['revision'])
    native = out/'native-tools'
    native_authority = ROOT/'evidence/offline-lds-native-readiness.json'
    native_expected = json.loads(native_authority.read_text())['profiles'][0]['native_tool_source_sha256']
    for name, digest in native_expected.items():
        require(sha(native_source/name) == digest, 'pinned native export changed: '+name)
    shutil.copytree(native_source, native)
    before = {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob('*') if p.is_file()}
    original_script = (native/'ps5-pie.ld').read_text()
    needle = '    .rodata : { *(.rodata .rodata.*) } : ro'
    require(original_script.count(needle) == 1, 'native linker source changed')
    modified = original_script.replace(needle, '''    .rodata : {
        HIDDEN(__ps5hip_names_begin = .);
        *(.rodata .rodata.*)
        HIDDEN(__ps5hip_names_end = .);
        . = ALIGN(4096);
        HIDDEN(__ps5hip_bundle_begin = .);
        KEEP(*(.hip_fatbin))
        HIDDEN(__ps5hip_bundle_end = .);
    } : ro''').replace('        *(.data.rel.ro .data.rel.ro.*)', '''        HIDDEN(__ps5hip_relro_begin = .);
        *(.data.rel.ro .data.rel.ro.*)
        . = ALIGN(8);
        HIDDEN(__ps5hip_wrappers_begin = .);
        KEEP(*(.hipFatBinSegment))
        HIDDEN(__ps5hip_wrappers_end = .);
        HIDDEN(__ps5hip_relro_end = .);''')
    (native/'ps5-pie.ld').write_text(modified)
    sdk = ROOT/'.deps/sdk'
    require((sdk/'.ps5-sdk-revision').read_text().strip() == lock['sdk']['revision'], 'SDK revision changed')
    objects = [selected['ps5']]
    for source in [*core, ROOT/'tools/hip_registration_control.cpp', native/'app_crt.cpp']:
        obj = out/('ps5-'+source.stem+'.o')
        run('native-compile-'+source.stem, ROOT/'tools/ps5-clangxx', *common, '-fPIC', '-fvisibility=hidden', '-c', source, '-o', obj)
        objects.append(obj)
    (out/'symbols.map').write_text('{ local: *; };\n')
    unwind = [part for name in ('eh_frame', 'eh_frame_hdr') for part in
              ('--defsym', f'__{name}_start=ADDR(.{name})', '--defsym', f'__{name}_end=ADDR(.{name})+SIZEOF(.{name})')]
    libs = sdk/'target/lib'
    pie, elf = out/'llvm-pie.elf', out/'eboot.elf'
    run('native-pie-link', 'ld.lld-18', '-pie', '-T', native/'ps5-pie.ld', '--eh-frame-hdr', *unwind,
        '-z', 'max-page-size=0x4000', '--hash-style=gnu', '--version-script', out/'symbols.map', '--exclude-libs=ALL',
        '-L', libs, '-e', '_start', '-o', pie, *objects, '--start-group',
        *[libs/n for n in ('libc++.a', 'libc++abi.a', 'libunwind.a', 'libc.a', 'libps5platform.a')],
        '--end-group', '--as-needed', *[libs/n for n in ('libkernel.so', 'libSceLibcInternal.so', 'libSceNet.so')])
    zlib = ref/'.deps'/('host-zlib-'+lock['host_zlib']['version'])/'root'
    zlib_receipt = json.loads((zlib.parent/'receipt.json').read_text())
    require(zlib_receipt['source_sha256'] == lock['host_zlib']['sha256'], 'zlib source identity changed')
    for name, digest in zlib_receipt['artifacts'].items():
        require(sha(zlib/name) == digest, 'zlib artifact changed')
    converter = out/'ps5-native-tool'
    run('converter-build', 'clang++-18', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
        *[native/n for n in ('native_app_builder.cpp', 'elf_object.cpp', 'sce_module_writer.cpp', 'self_container.cpp')],
        '-I', zlib/'include', zlib/'lib/libz.a', '-o', converter)
    run('native-convert', converter, 'link', '--in', pie, '--out', elf, '--stub-dir', libs, '--file-name', 'eboot.elf')
    symbols = {}
    symbol_lines = run('native-symbols', 'llvm-nm-18', '-n', pie).splitlines()
    for line in symbol_lines:
        parts = line.split()
        if len(parts) == 3:
            symbols[parts[2]] = int(parts[0], 16)
    expected_bundle = (out/'ps5.bundle').read_bytes()
    audits = {'raw_pie': audit_image(pie, symbols, expected_bundle),
              'converted_eboot': audit_image(elf, symbols, expected_bundle, native=True)}
    modules, libraries, imports = read_imports(elf)
    (out/'native-imports.json').write_text(json.dumps({'modules': modules, 'libraries': libraries, 'symbols': imports}, indent=2)+'\n')
    named_imports = {parts[1]: imports[nid(parts[1])] for line in symbol_lines
                     if len(parts := line.split()) == 2 and parts[0] == 'U' and nid(parts[1]) in imports}
    require(len(named_imports) == len(imports), 'unresolved native import identity')
    require(not any('Agc' in name for name in named_imports), 'registration control unexpectedly imports GPU operations')
    (out/'native-import-names.json').write_text(json.dumps({'eboot_sha256': sha(elf),
        'native_executed': False, 'imports': named_imports}, indent=2)+'\n')
    after = {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob('*') if p.is_file()}
    require(before == after, 'shared native inputs changed')
    receipt = {'schema': 1, 'classification': 'Linux compiler control and offline PS5 link/conversion only',
               'capture_launch': args.capture_launch, 'synthetic_gpu_addresses': args.capture_launch,
               'fixture_manifest_sha256': sha(manifest), 'selected_host_objects': {k: sha(v) for k,v in selected.items()},
               'header_hashes': header_hashes, 'generated_version_sha256': sha(generated),
               'native_source_revision': lock['native_tools']['revision'], 'native_inputs_before_after': before,
               'native_source_authority_sha256': sha(native_authority),
               'sdk_inputs': {n: sha(libs/n) for n in ('libc++.a', 'libc++abi.a', 'libunwind.a', 'libc.a',
                   'libps5platform.a', 'libkernel.so', 'libSceLibcInternal.so', 'libSceNet.so')},
               'tools': {name: {'path': shutil.which(name), 'sha256': sha(Path(shutil.which(name)))}
                         for name in ('clang++-18', 'ld.lld-18', 'llvm-nm-18')},
               'sdk_wrapper_sha256': sha(sdk/'bin/prospero-clang++'),
               'audits': audits, 'gpu_calls': 0, 'native_executed': False, 'hardware_validated': False,
               'artifacts': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()},
               'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                   [*core, ROOT/'tools/hip_registration_control.cpp', Path(__file__), ROOT/'tests/host_registration_test.cpp', ROOT/'include/ps5hip/host_registration.h'] +
                   ([ROOT/'tests/host_launch_test.cpp', ROOT/'include/ps5hip/host_launch.h'] if args.capture_launch else [])}}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'native_executed': False, 'gpu_calls': 0}))


if __name__ == '__main__':
    main()
