#!/usr/bin/env python3
"""Offline genuine vector runtime control: host transport models and PS5 link only."""
import argparse
import hashlib
import json
import os
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


from probe_host_registration import audit_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--host-only', action='store_true', help='Stop after host models for review before any native linking')
    args = parser.parse_args()
    ref, out = args.reference_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, *command, expected=0):
        argv = [str(c) for c in command]
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=180,
                           env=dict(os.environ,ASAN_OPTIONS='protect_shadow_gap=0'))
        (out / (label+'.stdout')).write_text(r.stdout)
        (out / (label+'.stderr')).write_text(r.stderr)
        commands.append({'label': label, 'argv': argv, 'exit_code': r.returncode,
                         'environment': {'ASAN_OPTIONS':'protect_shadow_gap=0'}})
        (out / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        require(r.returncode == expected, label + ' failed: ' + r.stderr[-3000:])
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
    lock = json.loads((ROOT/'dependencies.json').read_text())
    platform = ref/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
    includes += ['-I', platform]
    core = [ROOT/'src'/name for name in ('code_object.cpp','fat_binary.cpp','host_registration.cpp',
        'host_launch.cpp','launch.cpp','native_memory.cpp','native_data.cpp','native_launch.cpp','submission.cpp')]
    clients = [ROOT/'tools/hip_vector_control.cpp', ROOT/'tests/hip_vector_client.cpp']
    common = ['-std=c++17','-O1','-g','-Wall','-Wextra','-Werror',*includes]
    source_files = [*core,*clients,ROOT/'src/agc_backend.cpp',ROOT/'tests/hip_vector_model.cpp',Path(__file__),
        ROOT/'tools/probe_host_registration.py',ROOT/'tests/native_vector_recipe_test.py',
        *[ROOT/'include/ps5hip'/n for n in ('native_launch.h','native_data.h','native_memory.h',
        'host_launch.h','host_registration.h','submission.h','code_object.h','launch.h','fat_binary.h','agc_backend.h')]]
    source_before = {str(p.relative_to(ROOT)):sha(p) for p in source_files}
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
    from native_vector_recipe_test import validate_model
    for partial in (False,True):
        binary=out/('linux-partial' if partial else 'linux-vector')
        run(binary.name+'-build','clang++-18',*common,*sanitizers,'-DPS5HIP_VECTOR_MODEL=1',
            *(['-DPS5HIP_VECTOR_PARTIAL=1'] if partial else []),'-fPIE','-pie','-fuse-ld=lld-18',
            *core,*clients,ROOT/'tests/hip_vector_model.cpp',selected['linux'],'-Wl,-T,'+str(linux_script),'-o',binary)
        modes=['partial'] if partial else ['normal','preflush','init','submit','kick','timeout',
            'throw-submit','unknown-submit','throw-kick','postflush','missing','corrupt','input','guard',
            'code-allocate','kernarg-map','command-map','marker-map','init-unmap','init-release',
            'preflush-unmap','preflush-release','marker-map-unmap','completed-unmap','completed-release']
        for mode in modes:
            code=20 if mode in ('missing','corrupt','input','guard') else 0
            log=run('model-'+mode,binary,mode,expected=code)
            validate_model(log,mode,code)
    require(source_before=={str(p.relative_to(ROOT)):sha(p) for p in source_files},'source changed during host checks')
    if args.host_only:
        receipt={'schema':1,'classification':'Genuine compiler launch through host syscall/backend models only',
            'native_executed':False,'hardware_validated':False,'model_is_isa_emulation':False,
            'sources':source_before,'selected_host_objects':{k:sha(v) for k,v in selected.items()},
            'artifacts':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print('Host vector models complete; native phase not run:',out)
        return
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
    for source in [*core,*clients,ROOT/'src/agc_backend.cpp',native/'app_crt.cpp']:
        obj = out/('ps5-'+source.stem+'.o')
        run('native-compile-'+source.stem, ROOT/'tools/ps5-clangxx', *common, '-fPIC', '-fvisibility=hidden', '-c', source, '-o', obj)
        objects.append(obj)
    stubs=[]
    for library,declarations in {
        'libSceAgc':'int sceAgcInit(unsigned){return -1;} int sceAgcSuspendPoint(){return -1;}',
        'libSceAgcDriver':'int sceAgcDriverSubmitDcb(void*){return -1;}'}.items():
        source=out/(library+'.cpp');source.write_text('extern "C" {'+declarations+'}\n')
        obj=out/(library+'.o');stub=out/(library+'.so')
        run('stub-compile-'+library,ROOT/'tools/ps5-clangxx','-fPIC','-c',source,'-o',obj)
        run('stub-link-'+library,'ld.lld-18','-shared','-soname',library+'.prx',obj,'-o',stub)
        stubs.append(stub)
    (out/'symbols.map').write_text('{ local: *; };\n')
    unwind = [part for name in ('eh_frame', 'eh_frame_hdr') for part in
              ('--defsym', f'__{name}_start=ADDR(.{name})', '--defsym', f'__{name}_end=ADDR(.{name})+SIZEOF(.{name})')]
    libs = sdk/'target/lib'
    pie, elf = out/'llvm-pie.elf', out/'eboot.elf'
    run('native-pie-link', 'ld.lld-18', '-pie', '-T', native/'ps5-pie.ld', '--eh-frame-hdr', *unwind,
        '-z', 'max-page-size=0x4000', '--hash-style=gnu', '--version-script', out/'symbols.map', '--exclude-libs=ALL',
        '-L', libs, '-e', '_start', '-o', pie, *objects, *stubs, '--start-group',
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
    run('native-convert', converter, 'link', '--in', pie, '--out', elf, '--stub-dir', libs, '--stub',stubs[0],'--stub',stubs[1],'--file-name','eboot.elf')
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
    run('native-agc-providers','python3',ROOT/'tests/native_agc_imports_test.py',elf)
    require(not any(name.startswith('hip') or name.startswith('__hip') for name in named_imports),'HIP ABI unexpectedly imported')
    (out/'native-import-names.json').write_text(json.dumps({'eboot_sha256': sha(elf),
        'native_executed': False, 'imports': named_imports}, indent=2)+'\n')
    after = {str(p.relative_to(native_source)): sha(p) for p in native_source.rglob('*') if p.is_file()}
    require(before == after, 'shared native inputs changed')
    receipt = {'schema': 1, 'classification': 'Genuine HIP vector host transport model and offline PS5 link/conversion only',
               'serialized_blocking_vector': True, 'model_is_isa_emulation': False,
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
               'source_hashes': source_before}
    require(source_before=={str(p.relative_to(ROOT)):sha(p) for p in source_files},'source changed during recipe')
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'native_executed': False, 'gpu_calls': 0}))


if __name__ == '__main__':
    main()
