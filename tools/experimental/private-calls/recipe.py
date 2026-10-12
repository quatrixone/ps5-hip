#!/usr/bin/env python3
"""Prepare and validate the isolated LLVM18 private-call experiment; never build llc."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
TOOLS = ('clang++-18', 'opt-18', 'llvm-config-18', 'ld.lld-18',
         'llvm-readobj-18', 'llvm-objdump-18')
LAYOUT = {'specialization': 'g3-private-specialization-v3',
          'validation': 'g3-private-backend-validation',
          'mir': 'g3-private-backend-mir-controls',
          'backend': 'g3-private-backend-design',
          'register': 'g3-private-backend-register-controls',
          'measurement': 'g3-private-backend-measurement',
          'guard': 'g3-private-backend-guard-v4'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def package_identity():
    pins = read(PACKAGE / 'pins.json')
    for name, entry in pins['source_files'].items():
        if sha(PACKAGE / name) != entry['sha256']:
            raise ValueError('Pinned package source changed: ' + name)
    files = {name: sha(PACKAGE / name) for name in pins['source_files']}
    files.update({name: sha(PACKAGE / name) for name in ('pins.json', 'recipe.py')})
    return pins, files


def binary_identity(path):
    path = path.resolve(strict=True)
    libraries = {}
    result = subprocess.run(['ldd', str(path)], capture_output=True, text=True)
    # Static llc is valid; retain every resolved linked library for dynamic tools.
    if result.returncode and not any(s in result.stdout + result.stderr
                                    for s in ('not a dynamic executable', 'statically linked')):
        raise RuntimeError('Cannot inspect linked libraries: ' + str(path))
    for name in re.findall(r'=> (/\S+) ', result.stdout):
        resolved = Path(name).resolve(strict=True)
        libraries[str(resolved)] = sha(resolved)
    if 'not found' in result.stdout:
        raise RuntimeError('Missing tool library: ' + str(path))
    return {'path': str(path), 'sha256': sha(path), 'libraries': libraries}


def tool_identity():
    identities = {}
    for name in TOOLS:
        found = shutil.which(name)
        if not found:
            raise RuntimeError('Missing tool: ' + name)
        version = subprocess.check_output([found, '--version'], text=True)
        if not re.search(r'(?<![\d.])18\.1\.8(?![\d.])', version):
            raise RuntimeError('Expected LLVM 18.1.8: ' + name)
        identities[name] = {**binary_identity(Path(found)), 'version': version}
    return identities


def verify_source(path, pins):
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pins['llvm']['commit']:
        raise ValueError('LLVM source commit mismatch')
    if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Use the clean pinned source, not the patched checkout')
    if not (path / 'llvm/lib/Target/AMDGPU/SIFixVGPRCopies.cpp').is_file():
        raise ValueError('LLVM AMDGPU source is absent')


def run(output, label, command):
    logs = output / 'recipe-logs'
    logs.mkdir(exist_ok=True)
    log = logs / (label + '.log')
    if log.exists():
        raise ValueError('Refusing to overwrite operation log: ' + str(log))
    write_json(log.with_suffix('.command.json'), [str(x) for x in command])
    with log.open('w') as stream:
        result = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.result.json'), {'exit_code': result.returncode})
    result.check_returncode()


def prepare(args):
    pins, sources = package_identity()
    if args.output.exists():
        raise ValueError('Output must be a new directory; failed runs are preserved')
    verify_source(args.llvm_source, pins)
    tools = tool_identity()
    args.output.mkdir(parents=True)
    output = args.output
    write_json(output / 'prepare-inputs.json', {'package': sources, 'llvm_source': str(args.llvm_source),
                                               'llvm_commit': pins['llvm']['commit'], 'tools': tools})
    for name in pins['source_files']:
        source = PACKAGE / name
        category, relative = name.split('/', 1)
        destination = output / ('fixtures' if category == 'fixtures' else 'results/' + LAYOUT[category]) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    spec = output / 'results' / LAYOUT['specialization']
    validation = output / 'results' / LAYOUT['validation']
    mir = output / 'results' / LAYOUT['mir']
    flags = shlex.split(subprocess.check_output(['llvm-config-18', '--cxxflags', '--ldflags', '--libs',
                                                'core', 'passes', 'transformutils', 'analysis', '--system-libs'], text=True))
    run(output, 'plugin-build', ['clang++-18', '-fPIC', '-shared', spec / 'PrivateSpecialize.cpp',
                                 *flags, '-O1', '-o', spec / 'PrivateSpecialize.so'])
    inputs = read(output / 'prepare-inputs.json')
    inputs['plugin_sha256'] = sha(spec / 'PrivateSpecialize.so')
    write_json(output / 'prepare-inputs.json', inputs)
    linker = Path(tools['ld.lld-18']['path'])
    for fixture, folder, stem in [('probes-observable.hip', 'g3-flat-call-observable', 'probes'),
                                  ('volatile_calls.hip', 'g3-volatile-call-observable', 'volatile_calls')]:
        source = output / 'fixtures' / fixture
        for target in ('gfx1013', 'gfx1030'):
            for optimization in ('O1', 'O2'):
                directory = output / 'results' / folder / (target + '-' + optimization)
                directory.mkdir(parents=True)
                base = ['clang++-18', '-x', 'hip', '--cuda-device-only', '--no-gpu-bundle-output',
                        '-B' + str(linker.parent), '-nogpuinc', '-nogpulib', '--offload-arch=' + target,
                        '-mcode-object-version=5', '-mno-wavefrontsize64', '-' + optimization,
                        '-std=c++17', '-fno-fast-math']
                label = folder + '-' + target + '-' + optimization
                linked = base + ['-c', str(source), '-o', str(directory / (stem + '.hsaco'))]
                trace = subprocess.run(linked + ['-###'], capture_output=True, text=True, check=True).stderr
                (directory / 'driver.txt').write_text(trace)
                commands = [shlex.split(line) for line in trace.splitlines() if line.lstrip().startswith('"')]
                selected = [c[0] for c in commands if 'elf64_amdgpu' in c and '--whole-archive' in c]
                if len(selected) != 1 or Path(selected[0]).resolve() != linker:
                    raise ValueError('Unexpected actual HIP device linker: ' + repr(selected))
                run(output, label + '-link', linked)
                run(output, label + '-ir', base + ['-S', '-emit-llvm', '-c', source, '-o', directory / (stem + '.ll')])
    for name in ('check_attributes.py', 'check_negatives.py', 'check_adversarial.py',
                 'check_unchanged_attrs.py', 'check_budgets.py'):
        run(output, name[:-3], [sys.executable, spec / name])
    for name, extra in [('prepare.py', []), ('prepare_negatives.py', []), ('prepare_stress.py', []),
                        ('prepare_stress.py', ['--count', '4', '--label', 'late-callee-save']),
                        ('prepare_unbounded.py', []), ('test_diagnostics.py', [])]:
        label = name[:-3] + ('-small' if extra else '')
        run(output, label, [sys.executable, validation / name, *extra])
    # The stress generator's historical manifest spelling differs from its runner.
    shutil.copyfile(validation / 'late-spill-input.json', validation / 'stress-input.json')
    run(output, 'mir-runner-synthetic-tests', [sys.executable, mir / 'test_run_v2.py'])
    if tool_identity() != tools or package_identity()[1] != sources:
        raise RuntimeError('Preparation tool or source identity changed')
    artifacts = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output / 'prepared.json', {'schema': 1, 'package': sources, 'tools': tools,
                                        'artifacts': artifacts, 'hardware_validated': False,
                                        'accepted_compiler': False})
    print('Prepared fresh plugin, 8 real fixture profiles and strict validation inputs:', output)


def require_prepared(output):
    report = read(output / 'prepared.json')
    if report['package'] != package_identity()[1]:
        raise ValueError('Prepared package identity differs')
    for name, digest in report['artifacts'].items():
        if sha(output / name) != digest:
            raise ValueError('Prepared input changed: ' + name)
    if report['tools'] != tool_identity():
        raise ValueError('LLVM tools or linked libraries changed since preparation')
    return report


def require_measurement(output):
    sys.path.insert(0, str(PACKAGE / 'validation'))
    from measurement_binding_v1 import verify_measurement
    folder = output / 'results' / LAYOUT['measurement']
    complete = read(output / 'measurement-complete.json')
    if complete['binding_sha256'] != sha(folder / 'binding.json'):
        raise ValueError('Fresh measurement binding changed')
    measured = verify_measurement(folder / 'measured-contracts.json')
    return measured, complete


def measure(args):
    require_prepared(args.output)
    if (args.output / 'measurement-complete.json').exists():
        raise ValueError('Refusing to repeat independent measurement')
    pins, _ = package_identity()
    llvm_source = Path(read(args.output / 'prepare-inputs.json')['llvm_source'])
    verify_source(llvm_source, pins)
    compiler = binary_identity(args.compiler)
    version = subprocess.check_output([str(args.compiler), '--version'], text=True)
    if 'LLVM version 18.1.8' not in version:
        raise ValueError('Expected clean source-built LLVM 18.1.8 llc')
    cache_path = args.compiler.parent.parent / 'CMakeCache.txt'
    cache = cache_path.read_text()
    match = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$', cache, re.M)
    if not match or Path(match[1]).resolve() != (llvm_source / 'llvm').resolve():
        raise ValueError('Measurement compiler must come from the clean pinned source build')
    for key, value in [('LLVM_ENABLE_ASSERTIONS', 'ON'), ('LLVM_TARGETS_TO_BUILD', 'AMDGPU'),
                       ('CMAKE_BUILD_TYPE', 'Release')]:
        if not re.search(r'^' + key + r':[^=]+=' + value + r'$', cache, re.M):
            raise ValueError('Unexpected clean-build option: ' + key)
    folder = args.output / 'results' / LAYOUT['measurement']
    request = folder / 'request.json'
    if request.exists():
        raise ValueError('Failed measurements are preserved; use a fresh preparation')
    write_json(request, {'compiler': compiler, 'llvm_source': str(llvm_source),
                        'llvm_commit': pins['llvm']['commit'], 'cache_path': str(cache_path),
                        'cache_sha256': sha(cache_path), 'version': version})
    run(args.output, 'independent-measurement', [sys.executable, folder / 'measure_v1.py',
                                               '--compiler', args.compiler])
    validation = args.output / 'results' / LAYOUT['validation']
    run(args.output, 'measured-frame-mutations', [sys.executable, folder / 'test_frame_audit_portable_v1.py'])
    run(args.output, 'measured-linked-mutations', [sys.executable, validation / 'test_post_audit_portable_v2.py'])
    verify_source(llvm_source, pins)
    if binary_identity(args.compiler) != compiler or sha(cache_path) != read(request)['cache_sha256']:
        raise ValueError('Clean measurement compiler, libraries or build cache changed')
    require_prepared(args.output)
    write_json(args.output / 'measurement-complete.json',
               {'binding_sha256': sha(folder / 'binding.json'), 'compiler': compiler,
                'accepted_compiler': False, 'hardware_validated': False})
    require_measurement(args.output)
    print('Fresh clean-baseline measurement and synthetic frame/linked checks passed')


def validate(args):
    require_prepared(args.output)
    measured, complete = require_measurement(args.output)
    compiler = binary_identity(args.compiler)
    if args.mode == 'baseline' and compiler != complete['compiler']:
        raise ValueError('Baseline compiler differs from independent measurement')
    version = subprocess.check_output([str(args.compiler), '--version'], text=True)
    if 'LLVM version 18.1.8' not in version:
        raise ValueError('Expected llc from pinned LLVM 18.1.8 build')
    validation = args.output / 'results' / LAYOUT['validation']
    mir = args.output / 'results' / LAYOUT['mir']
    run(args.output, 'validate-' + args.mode, [sys.executable, validation / 'run_all_portable_v2.py',
                                            '--compiler', args.compiler, '--mode', args.mode])
    if args.mode == 'baseline':
        pins, _ = package_identity()
        verify_source(Path(read(args.output / 'prepare-inputs.json')['llvm_source']), pins)
        run(args.output, 'prepare-mir', [sys.executable, mir / 'prepare.py'])
    run(args.output, 'validate-mir-' + args.mode, [sys.executable, mir / 'run_ipra_off_v1.py',
                                                '--compiler', args.compiler, '--mode', args.mode])
    if args.mode == 'patched-on':
        run(args.output, 'test-diagnostics-v2-real-record',
            [sys.executable, validation / 'test_diagnostics_v2.py', '--records', validation,
             '--output', validation / 'diagnostic-parser-v2-tests.json'])
    if binary_identity(args.compiler) != compiler:
        raise RuntimeError('llc or its libraries changed during validation')
    require_prepared(args.output)
    write_json(args.output / ('validation-' + args.mode + '.json'),
               {'compiler': compiler, 'version': version, 'mode': args.mode,
                'selected_checks_completed': True, 'accepted_compiler': False,
                'hardware_validated': False,
                'limit': 'Separate same-source/configuration build provenance and additional post-audits remain required.'})


def main():
    if not __debug__:
        raise RuntimeError('Assertions are required; do not use Python -O or PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare', help='Build only plugin; compile fixtures and generate fresh test inputs')
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--llvm-source', type=Path, required=True)
    measurement = sub.add_parser('measure', help='Independently measure fresh inputs with clean pinned source-built llc')
    measurement.add_argument('--output', type=Path, required=True)
    measurement.add_argument('--compiler', type=Path, required=True)
    val = sub.add_parser('validate', help='Run one explicit backend mode; never builds LLVM')
    val.add_argument('--output', type=Path, required=True)
    val.add_argument('--compiler', type=Path, required=True)
    val.add_argument('--mode', choices=('baseline', 'patched-off', 'patched-on'), required=True)
    guard = sub.add_parser('guard-controls', help='Explicit14 effective-IPRA/default-off controls after paired baseline/off')
    guard.add_argument('--output', type=Path, required=True)
    guard.add_argument('--baseline-compiler', type=Path, required=True)
    guard.add_argument('--patched-compiler', type=Path, required=True)
    reg = sub.add_parser('register-controls', help='Explicit four-operand MIR validation after baseline and patched-on modes')
    reg.add_argument('--output', type=Path, required=True)
    reg.add_argument('--baseline-compiler', type=Path, required=True)
    reg.add_argument('--patched-compiler', type=Path, required=True)
    audit = sub.add_parser('audit', help='Audit exact ordinary linked graph delta after all three modes')
    audit.add_argument('--output', type=Path, required=True)
    audit.add_argument('--allow-frame-move', action='store_true',
                       help='Explicitly enable only the reviewed bounded entry MOV commute')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.operation == 'prepare':
        args.llvm_source = args.llvm_source.resolve(strict=True)
        prepare(args)
    elif args.operation == 'measure':
        args.compiler = args.compiler.resolve(strict=True)
        measure(args)
    elif args.operation == 'validate':
        args.compiler = args.compiler.resolve(strict=True)
        validate(args)
    elif args.operation in ('register-controls', 'guard-controls'):
        require_prepared(args.output)
        compilers = {mode: binary_identity(path) for mode, path in
                     [('baseline', args.baseline_compiler),
                      ('patched-off' if args.operation == 'guard-controls' else 'patched-on', args.patched_compiler)]}
        for mode, identity in compilers.items():
            if read(args.output / ('validation-' + mode + '.json'))['compiler'] != identity:
                raise ValueError('Compiler identity differs from completed mode: ' + mode)
        kind, script = ('guard', 'run_portable_v1.py') if args.operation == 'guard-controls' else ('register', 'run_ipra_off_v1.py')
        path = args.output / 'results' / LAYOUT[kind] / script
        run(args.output, args.operation, [sys.executable, path,
             '--baseline-compiler', args.baseline_compiler.resolve(strict=True),
             '--patched-compiler', args.patched_compiler.resolve(strict=True)])
        for mode, identity in compilers.items():
            if binary_identity(Path(identity['path'])) != identity:
                raise ValueError('Compiler identity changed during register controls')
        require_prepared(args.output)
    else:
        prepared = require_prepared(args.output)
        require_measurement(args.output)
        path = args.output / 'results' / LAYOUT['validation'] / 'post_audit_portable_v2.py'
        command = [sys.executable, path, '--baseline', path.parent / 'runs/baseline',
                   '--enabled', path.parent / 'runs/patched-on', '--default-off', path.parent / 'runs/patched-off',
                   '--matrix', path.parent / 'matrix.json', '--frame-contract',
                   args.output / 'results' / LAYOUT['measurement'] / 'measured-contracts.json',
                   '--output', args.output / 'ordinary-post-audit.json']
        if args.allow_frame_move:
            command.append('--allow-frame-move')
        run(args.output, 'ordinary-post-audit', command)
        run(args.output, 'extended-post-audit', [sys.executable, path.with_name('post_audit_extensions_v2.py'),
             '--validation', path.parent, '--mir', args.output / 'results' / LAYOUT['mir'],
             '--guard-receipt', args.output / 'results' / LAYOUT['guard'] / 'runs-v1/result.json',
             '--frame-contract', args.output / 'results' / LAYOUT['measurement'] / 'measured-contracts.json',
             '--output', args.output / 'extended-post-audit.json'])
        run(args.output, 'extended-post-mutations',
            [sys.executable, path.with_name('test_post_extensions_portable_v2.py')])
        run(args.output, 'linked-alias-tool-tests',
            [sys.executable, path.with_name('test_link_alias_tools_v2.py')])
        run(args.output, 'linked-alias-audit', [sys.executable, path.with_name('link_alias_portable_v2.py'),
            '--validation', path.parent, '--output', args.output / 'linked-alias-audit',
            '--prepared', args.output / 'prepared.json',
            '--linker', prepared['tools']['ld.lld-18']['path'],
            '--objdump', prepared['tools']['llvm-objdump-18']['path']])
        run(args.output, 'linked-alias-mutations',
            [sys.executable, path.with_name('test_link_alias_portable_v1.py')])
        require_prepared(args.output)
        require_measurement(args.output)


if __name__ == '__main__':
    main()
