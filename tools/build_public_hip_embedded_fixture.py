"""Rebuild genuine HIP constructor/stub objects for public CLR, without executing them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from build_canaries import require_device_linker, require_tool_versions
from probe_hip_routes import audit_object, audit_registration, audit_wrapper
from fat_binary_fixture_test import section

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'results') and not output.is_relative_to(ROOT / 'build'):
        raise ValueError('output must be under results/ or build/')
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, argv):
        argv = list(map(str, argv))
        try:
            result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=180,
                                    env=dict(os.environ, PATH=str(Path(shutil.which("ld.lld-18")).resolve().parent) + os.pathsep + os.environ["PATH"]))
        except subprocess.TimeoutExpired as error:
            for suffix, value in (('stdout', error.stdout), ('stderr', error.stderr)):
                (output / (label + '.' + suffix)).write_bytes(
                    value.encode() if isinstance(value, str) else value or b'')
            commands.append({'label': label, 'argv': argv, 'timed_out': True, 'deadline_seconds': 180})
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
            raise
        (output / (label + '.stdout')).write_text(result.stdout)
        (output / (label + '.stderr')).write_text(result.stderr)
        commands.append({'label': label, 'argv': argv, 'exit_code': result.returncode})
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(label + ': ' + result.stderr[-3000:])
        return result.stdout, result.stderr

    tools = {}
    versions = {}
    for name in ('clang++-18', 'ld.lld-18', 'llvm-readobj-18', 'llvm-objdump-18',
                 'llvm-nm-18', 'clang-offload-bundler-18', 'llvm-config-18'):
        found = shutil.which(name)
        if not found:
            raise RuntimeError('missing ' + name)
        path = Path(found).absolute()
        versions[name] = run(name + '-version', [path, '--version'])[0]
        tools[name] = {'path': str(path), 'resolved_path': str(path.resolve()), 'sha256': sha(path)}
    require_tool_versions({k: v for k, v in versions.items() if k != 'llvm-config-18'},
                          json.loads((ROOT / 'dependencies.json').read_text())['llvm']['version'])
    inputs = {str(ROOT / name): sha(ROOT / name) for name in (
        'dependencies.json', 'tools/build_public_hip_embedded_fixture.py', 'tools/build_canaries.py',
        'tools/probe_hip_routes.py', 'tests/fat_binary_fixture_test.py',
        'tests/public_hip_embedded.hip', 'tools/ps5-clangxx',
        '.deps/sdk/bin/prospero-clang++', '.deps/sdk/bin/clang++')}
    profiles = {}
    for host in ('linux', 'ps5'):
        directory = output / host
        directory.mkdir()
        profiles[host] = {}
        for module in ('a', 'b'):
            stem = 'public-embedded' + ('-b' if module == 'b' else '')
            kernel_name = 'thread_words' + ('_b' if module == 'b' else '')
            label = host + '-' + module
            obj = directory / (stem + '.o')
            compiler = ROOT / 'tools/ps5-clangxx' if host == 'ps5' else tools['clang++-18']['path']
            argv = [compiler, '-x', 'hip', '-nogpuinc', '-nogpulib',
                    '-B' + str(Path(tools['ld.lld-18']['resolved_path']).parent),
                    '--offload-arch=gfx1013', '-mcode-object-version=5', '-mno-wavefrontsize64',
                    '-D__HIP_PLATFORM_AMD__', '-std=c++17', '-O2', '-fPIC',
                    '-I' + str(ROOT / '.deps/sources/hip-headers/include'),
                    '-I' + str(ROOT / '.deps/sources/clr/hipamd/include'),
                    '-I' + str(ROOT / 'ports/hip'), '-include', '__clang_hip_runtime_wrapper.h',
                    '-c', ROOT / 'tests/public_hip_embedded.hip', '-o', obj]
            if module == 'b':
                argv += ['-Dthread_words=thread_words_b', '-Dlaunch_public_embedded=launch_public_embedded_b']
            if host == 'ps5':
                argv += ['-nostdlib', '-Xarch_device', '-fno-short-wchar']
            _, trace = run(label + '-driver', argv + ['-###'])
            require_device_linker(trace, tools['ld.lld-18']['path'])
            for line in trace.splitlines():
                if line.lstrip().startswith('"'):
                    executable = Path(shlex.split(line)[0]).resolve()
                    inputs[str(executable)] = sha(executable)
            for side in ('host', 'device'):
                dep = directory / (module + '-' + side + '.d')
                run(label + '-' + side + '-headers', argv + ['--cuda-' + side + '-only', '-M', '-MF', dep])
                for name in shlex.split(dep.read_text().replace('\\\n', ' ').split(':', 1)[1]):
                    path = Path(name).resolve()
                    inputs[str(path)] = sha(path)
            run(label + '-compile', argv)
            readobj, _ = run(label + '-readobj', [tools['llvm-readobj-18']['path'], '--sections', '--symbols', '--relocations', obj])
            disasm, _ = run(label + '-host-isa', [tools['llvm-objdump-18']['path'], '-dr', obj])
            imports, _ = run(label + '-imports', [tools['llvm-nm-18']['path'], '--undefined-only', obj])
            expected = {'__hipPopCallConfiguration', '__hipPushCallConfiguration', '__hipRegisterFatBinary',
                        '__hipRegisterFunction', '__hipUnregisterFatBinary', 'atexit', 'hipLaunchKernel'}
            if {line.split()[-1] for line in imports.splitlines() if line.strip()} != expected:
                raise RuntimeError('compiler registration/launch imports changed')
            wrapper = audit_wrapper(section(obj.read_bytes(), b'.hipFatBinSegment'), readobj, host)
            registration = audit_registration(disasm, (kernel_name,))
            bundle = directory / (stem + '.bundle')
            bundle.write_bytes(section(obj.read_bytes(), b'.hip_fatbin'))
            code = directory / (stem + '.hsaco')
            run(label + '-unbundle', [tools['clang-offload-bundler-18']['path'], '-unbundle', '-type=bc',
                                    '-targets=hipv4-amdgcn-amd-amdhsa--gfx1013', '-input=' + str(bundle),
                                    '-output=' + str(code)])
            metadata, _ = run(label + '-metadata', [tools['llvm-readobj-18']['path'], '--notes', code])
            isa, _ = run(label + '-device-isa', [tools['llvm-objdump-18']['path'], '--disassemble', '--mcpu=gfx1013', code])
            audit = audit_object(code.read_bytes(), metadata, isa, (kernel_name,))
            descriptor = audit['descriptors'][kernel_name]
            if (descriptor['group_bytes'], descriptor['private_bytes'], descriptor['kernarg_bytes']) != (0, 0, 16):
                raise RuntimeError('initial kernel argument/resource profile changed')
            if any(op.startswith(('flat_', 'scratch_', 's_swappc')) for op in audit['isa'][kernel_name]['opcodes']):
                raise RuntimeError('unsupported initial-profile ISA')
            profiles[host][module] = {'object': str(obj.relative_to(output)), 'wrapper': wrapper,
                              'registration': registration, 'imports': sorted(expected), 'device': audit}
    for name in ('public-embedded.hsaco', 'public-embedded-b.hsaco'):
        if section((output / 'linux' / name).read_bytes(), b'.text') != section((output / 'ps5' / name).read_bytes(), b'.text'):
            raise RuntimeError('Linux/native device instructions differ')
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise RuntimeError('input changed: ' + name)
    for tool in tools.values():
        if sha(tool['path']) != tool['sha256']:
            raise RuntimeError('tool changed: ' + tool['path'])
    receipt = {'classification': 'public-hip-genuine-embedded-compiler-fixture', 'inputs': inputs,
               'tools': tools, 'profiles': profiles,
               'artifacts': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()},
               'runtime_executed': False, 'native_executed': False, 'gpu_executed': False,
               'scope': 'Fresh compiler constructors, launch stubs and matching gfx1013 instructions; no runtime substitution or linkage proof.'}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
