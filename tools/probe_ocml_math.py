#!/usr/bin/env python3
"""Pinned OCML-only math compile/link probe; no runtime or GPU use."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from build_canaries import isa_inventory, require_device_linker, require_tool_versions

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = {
    'oclc_abi_version_500': ('__oclc_ABI_version', 'i32', 500),
    'oclc_isa_version_1013': ('__oclc_ISA_version', 'i32', 10103),
    'oclc_isa_version_1030': ('__oclc_ISA_version', 'i32', 10300),
    'oclc_finite_only_off': ('__oclc_finite_only_opt', 'i8', 0),
    'oclc_unsafe_math_off': ('__oclc_unsafe_math_opt', 'i8', 0),
    'oclc_daz_opt_off': ('__oclc_daz_opt', 'i8', 0),
    'oclc_correctly_rounded_sqrt_on': ('__oclc_correctly_rounded_sqrt32', 'i8', 1),
    'oclc_wavefrontsize64_off': ('__oclc_wavefrontsize64', 'i8', 0),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Input tree must not contain symlinks')
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_inputs(reference, pins):
    for name, expected in pins['reference_files'].items():
        path = reference / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Pinned reference file changed: ' + name)
    manifest = json.loads((reference / 'device-libs-sources.json').read_text())
    if (manifest['commit'], manifest['subtree']) != (pins['device_libraries_commit'], pins['device_libraries_subtree']):
        raise ValueError('Device-library revision mismatch')
    wanted = {entry['path']: entry['sha256'] for entry in manifest['files']}
    if len(wanted) != len(manifest['files']):
        raise ValueError('Duplicate source manifest path')
    sources = inventory(reference / 'device-libs-source')
    if sources != wanted:
        raise ValueError('Device-library source tree differs from pinned manifest')
    stage = inventory(reference / 'package-stage')
    if len(stage) != pins['package_stage']['files'] or digest(stage) != pins['package_stage']['inventory_sha256']:
        raise ValueError('HIP header/configuration stage differs from pinned inventory')
    return {'device-libs-source': sources, 'package-stage': stage,
            'reference_files': {name: sha(reference / name) for name in pins['reference_files']}}


def check_output(output, reference):
    if output.exists() or output.is_symlink():
        raise ValueError('Output must not already exist; preserve failed and frozen evidence')
    if not any(output.is_relative_to(ROOT / name) for name in ('build', 'results')):
        raise ValueError('Output must be under this checkout/build or results')
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError('Output must not overlap reference inputs')


def audit_driver(trace, linker, target, libraries):
    require_device_linker(trace, linker)
    commands = [shlex.split(line) for line in trace.splitlines() if line.lstrip().startswith('"')]
    cc = [c for c in commands if '-cc1' in c and '-fcuda-is-device' in c]
    if len(cc) != 1 or cc[0][cc[0].index('-target-cpu') + 1] != target:
        raise ValueError('Wrong device compiler target')
    linked = [Path(cc[0][i + 1]).resolve() for i, value in enumerate(cc[0]) if value == '-mlink-builtin-bitcode']
    if linked != [p.resolve() for p in libraries]:
        raise ValueError('Device bitcode library inventory/order changed')
    if any(flag in cc[0] for flag in ('-ffast-math', '-ffinite-math-only', '-funsafe-math-optimizations')):
        raise ValueError('Unsafe compiler math mode')
    link = [c for c in commands if 'elf64_amdgpu' in c]
    if len(link) != 1 or '--no-undefined' not in link[0]:
        raise ValueError('Device link must reject unresolved symbols')


def audit_constant(text, expected):
    name, kind, value = expected
    matches = re.findall(r'^@' + re.escape(name) + r' = .*\bconstant ' + kind + r' (\d+),', text, re.M)
    if matches != [str(value)]:
        raise ValueError('Wrong device-library constant ' + name)


def audit_caller_ir(text):
    for name in ('sin', 'log', 'tanh'):
        symbol = '__ocml_' + name + '_f32'
        if not re.search(r'\bcall\b[^\n]*@' + symbol + r'\(', text):
            raise ValueError('Caller IR lost direct OCML call: ' + symbol)
        if not re.search(r'^declare [^\n]*@' + symbol + r'\(', text, re.M):
            raise ValueError('Missing prelink OCML declaration: ' + symbol)
        if re.search(r'^define [^\n]*@' + symbol + r'\(', text, re.M):
            raise ValueError('Caller IR unexpectedly contains linked OCML body')


def audit_object(code, target, notes, isa, inspection, undefined, relocations):
    if code.read_bytes()[:4] != b'\x7fELF':
        raise ValueError('Expected direct ELF, not an offload bundle')
    if set(re.findall(r'^\s+\.name:\s+(\w+)\s*$', notes, re.M)) != {'fp32_math'}:
        raise ValueError('Unexpected metadata kernel inventory')
    if 'amdgcn-amd-amdhsa--' + target not in notes:
        raise ValueError('Wrong metadata target')
    for field, value in (('group_segment_fixed_size', 0), ('private_segment_fixed_size', 0),
                         ('sgpr_spill_count', 0), ('vgpr_spill_count', 0),
                         ('kernarg_segment_size', 280), ('kernarg_segment_align', 8),
                         ('wavefront_size', 32)):
        if re.findall(r'^\s+\.' + field + r':\s+(\d+)\s*$', notes, re.M) != [str(value)]:
            raise ValueError('Unexpected kernel metadata: ' + field)
    if (not inspection.startswith(target + ' ') or
        'fp32_math entry=6144 wave=32 kernarg=280 lds=0 private=0 vgpr_spills=0 sgpr_spills=0 initial_profile=eligible' not in inspection or
        'rsrc1=0x60af0041 rsrc2=0x8c rsrc3=0x0 properties=0x409 preload=0x0 kernarg_alignment=8 max_workgroup=1024' not in inspection or
        'explicit_arguments=3 hidden_arguments=13' not in inspection):
        raise ValueError('Loader did not admit the measured structural profile')
    if undefined.strip() or relocations.strip() != 'There are no relocations in this file.':
        raise ValueError('Unresolved symbol or relocation remains')
    found = isa_inventory(isa, {'fp32_math'})['fp32_math']
    ops = found['opcodes']
    if found['flat_scratch_setup'] or any(op.startswith(('flat_', 'scratch_', 's_swappc', 's_getreg')) for op in ops):
        raise ValueError('Unexpected flat/private/register/call instruction')
    if not all(any(op.startswith(prefix) for op in ops) for prefix in
               ('global_load', 'global_store', 'v_exp_f32', 'v_log_f32', 'v_sqrt_f32')):
        raise ValueError('Probe lost actual memory/math instructions')
    stores = re.findall(r'^\s+global_store_dword(x[234])?\s', isa, re.M)
    if sum(int(x[1:]) if x else 1 for x in stores) != 6:
        raise ValueError('Probe no longer stores six result words')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--inspector', required=True, type=Path)
    args = parser.parse_args()
    reference, output, inspector = args.reference_root.resolve(), args.output.resolve(), args.inspector.resolve()
    check_output(output, reference)
    pins = json.loads((ROOT / 'tools/ocml_math_inputs.json').read_text())
    inputs = verify_inputs(reference, pins)
    source = ROOT / 'kernels/ocml_math.hip'
    if sha(source) != pins['fixture_sha256']:
        raise ValueError('Pinned math fixture changed')
    if not inspector.is_file():
        raise ValueError('Build the existing ps5hip-inspect executable first')
    tools = {name: Path(shutil.which(name + '-18') or '') for name in
             ('clang', 'clang++', 'ld.lld', 'llvm-config', 'llvm-readelf', 'llvm-objdump', 'llvm-nm', 'llvm-dis')}
    if not all(path.is_file() for path in tools.values()):
        raise ValueError('Required LLVM18 executables unavailable')
    versions = {name: subprocess.check_output([str(path), '--version'], text=True)
                for name, path in tools.items() if name != 'llvm-config'}
    require_tool_versions(versions, pins['llvm_version'])
    if subprocess.check_output([str(tools['llvm-config']), '--version'], text=True).strip() != pins['llvm_version']:
        raise ValueError('Wrong llvm-config version')
    # Clang selects C++ driver/link behavior from argv[0]; do not resolve clang++ to clang.
    tools = {name: path.absolute() for name, path in tools.items()}
    output.mkdir(parents=True)
    report = {'schema': 1, 'status': 'running', 'hardware_validated': False, 'numeric_accuracy_qualified': False,
              'complete_ockl': False, 'full_device_library_port': False, 'native_host_compiled': False,
              'source_pins': pins, 'inputs': inputs, 'commands': [], 'profiles': [],
              'tool_versions': versions,
              'sources': {name: sha(ROOT / name) for name in ('tools/probe_ocml_math.py', 'tools/ocml_math_inputs.json',
                          'tools/build_canaries.py', 'kernels/ocml_math.hip')},
              'tools': {name: {'path': str(path), 'sha256': sha(path)} for name, path in tools.items()},
              'inspector': {'path': str(inspector), 'sha256': sha(inspector)}}

    def run(label, command):
        command = list(map(str, command))
        try:
            result = subprocess.run(command, cwd=output, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired as error:
            for suffix, data in (('stdout', error.stdout), ('stderr', error.stderr)):
                (output / (label + '.' + suffix)).write_bytes(data.encode() if isinstance(data, str) else data or b'')
            report['commands'].append({'label': label, 'argv': command, 'timeout_seconds': 300})
            raise
        for suffix, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            (output / (label + '.' + suffix)).write_text(text)
        report['commands'].append({'label': label, 'argv': command, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError('Command failed: ' + label)
        return result.stdout, result.stderr

    try:
        for name in ('device-libs-source', 'package-stage'):
            shutil.copytree(reference / name, output / name)
            if inventory(output / name) != inputs[name]:
                raise ValueError('Copied input changed: ' + name)
        shutil.copyfile(source, output / 'ocml_math.hip')
        derived = output / 'device-libs-source/cmake/OCL.cmake'
        text = derived.read_text()
        old = 'cmake_policy(SET CMP0053 OLD)'
        if text.count(old) != 1:
            raise ValueError('CMake policy patch precondition changed')
        derived.write_text(text.replace(old, 'cmake_policy(SET CMP0053 NEW)'))
        report['derived_policy_sha256'] = sha(derived)
        wrapper = output / 'wrapper'; wrapper.mkdir()
        llvm_dir = subprocess.check_output([str(tools['llvm-config']), '--cmakedir'], text=True).strip()
        bin_dir = subprocess.check_output([str(tools['llvm-config']), '--bindir'], text=True).strip()
        wrapper_text = '''cmake_minimum_required(VERSION 3.20)
project(ps5hip_ocml_probe LANGUAGES C CXX)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
find_package(LLVM REQUIRED)
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
foreach(tool clang llvm-link opt llvm-objdump)
  if(NOT TARGET ${tool})
    add_executable(${tool} IMPORTED GLOBAL)
    set_target_properties(${tool} PROPERTIES IMPORTED_LOCATION "@BIN@/${tool}")
  endif()
endforeach()
set(ROCM_DEVICELIB_STANDALONE_BUILD ON)
add_subdirectory("../device-libs-source" upstream)
'''
        (wrapper / 'CMakeLists.txt').write_text(wrapper_text.replace('@BIN@', bin_dir))
        report['wrapper_sha256'] = sha(wrapper / 'CMakeLists.txt')
        for name in ('clang', 'llvm-link', 'opt', 'llvm-objdump', 'lld'):
            path = Path(bin_dir) / name
            report['tools']['cmake-' + name] = {'path': str(path), 'sha256': sha(path)}
        run('configure', ['cmake', '-S', wrapper, '-B', output / 'build', '-G', 'Ninja',
                         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=' + str(tools['clang']),
                         '-DCMAKE_CXX_COMPILER=' + str(tools['clang++']), '-DLLVM_DIR=' + llvm_dir])
        run('build', ['cmake', '--build', output / 'build', '--target', 'ocml', *CONSTANTS, '--parallel', '4'])
        libs = output / 'libs'; libs.mkdir()
        for name in ('ocml', *CONSTANTS):
            shutil.copyfile(output / 'build/upstream/amdgcn/bitcode' / (name + '.bc'), libs / (name + '.bc'))
        shutil.copyfile(reference / 'ockl-workitem-only.bc', libs / 'ockl-workitem-only.bc')
        if sha(libs / 'ockl-workitem-only.bc') != pins['reference_files']['ockl-workitem-only.bc']:
            raise ValueError('Workitem slice changed during copying')
        for name, expected in CONSTANTS.items():
            ir, _ = run(name, [tools['llvm-dis'], libs / (name + '.bc'), '-o', '-'])
            audit_constant(ir, expected)
            if name == 'oclc_wavefrontsize64_off':
                audit_constant(ir, ('__oclc_wavefrontsize_log2', 'i32', 5))
        report['libraries'] = inventory(libs)
        for target in ('gfx1013', 'gfx1030'):
            library_names = ['ockl-workitem-only', 'ocml', 'oclc_abi_version_500', 'oclc_isa_version_' + target[3:],
                             *[x for x in CONSTANTS if not x.startswith(('oclc_abi_', 'oclc_isa_'))]]
            libraries = [libs / (name + '.bc') for name in library_names]
            for optimization in ('O1', 'O2'):
                label = target + '-' + optimization
                code = output / (label + '.hsaco')
                common = [tools['clang++'], '-x', 'hip', '--cuda-device-only', '--no-gpu-bundle-output',
                          '-B' + bin_dir, '--offload-arch=' + target, '-mcode-object-version=5', '-mno-wavefrontsize64',
                          '-D__HIP_PLATFORM_AMD__', '-std=c++17', '-' + optimization, '-fno-fast-math',
                          '--hip-path=' + str(output / 'package-stage'), '-I' + str(output / 'package-stage/include'),
                          '-include', '__clang_hip_runtime_wrapper.h']
                command = common + ['--rocm-device-lib-path=' + str(libs),
                                    *['--hip-device-lib=' + p.name for p in libraries],
                                    output / 'ocml_math.hip', '-o', code]
                _, trace = run(label + '-driver', command + ['-###'])
                audit_driver(trace, tools['ld.lld'], target, libraries)
                run(label + '-compile', command)
                caller = output / (label + '-caller.ll')
                run(label + '-caller', common + ['-nogpulib', '-S', '-emit-llvm', '-Xclang', '-disable-llvm-passes',
                                                output / 'ocml_math.hip', '-o', caller])
                audit_caller_ir(caller.read_text())
                notes, _ = run(label + '-notes', [tools['llvm-readelf'], '--notes', code])
                isa, _ = run(label + '-isa', [tools['llvm-objdump'], '-d', '--mcpu=' + target, code])
                inspection, _ = run(label + '-inspect', [inspector, code])
                undefined, _ = run(label + '-undefined', [tools['llvm-nm'], '--undefined-only', code])
                relocs, _ = run(label + '-relocations', [tools['llvm-readelf'], '--relocations', code])
                actual = audit_object(code, target, notes, isa, inspection, undefined, relocs)
                report['profiles'].append({'target': target, 'optimization': optimization, 'code_sha256': sha(code),
                                           'bytes': code.stat().st_size, 'caller_sha256': sha(caller), 'isa_inventory': actual})
        if verify_inputs(reference, pins) != inputs:
            raise ValueError('Reference inputs changed during build')
        for value in [*report['tools'].values(), report['inspector']]:
            if sha(Path(value['path'])) != value['sha256']:
                raise ValueError('Tool changed during build')
        report['reference_inputs_unchanged'] = True
        report['status'] = 'passed_offline_compile_link_only'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        report['artifacts'] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        (output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print('OCML partial probe passed; no native/GPU/numerical qualification:', output)


if __name__ == '__main__':
    main()
