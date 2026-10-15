"""Link genuine compiler HIP stubs into the qualified CLR host model."""
import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
from build_public_hip_embedded_fixture import ROOT, sha, section
from build_canaries import require_tool_versions
from embedded_module_linker import linux_linker_fragment


def qualify(base, compiler, output):
    base, compiler, output = (Path(p).resolve() for p in (base, compiler, output))
    if not output.is_relative_to(ROOT / 'results'):
        raise ValueError('output must be under results/')
    base_receipt = json.loads((base / 'receipt.json').read_text())
    compiler_receipt = json.loads((compiler / 'receipt.json').read_text())
    if base_receipt['classification'] != 'source-only-genuine-public-hip-model' or compiler_receipt['classification'] != 'public-hip-genuine-embedded-compiler-fixture':
        raise ValueError('unexpected input qualification')
    inputs = {}
    for group in ('inputs', 'prepared_sources', 'included_files'):
        for name, expected in base_receipt[group].items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha(path) != expected:
                raise ValueError('base input changed: ' + name)
            inputs[str(path)] = expected
    for name, expected in compiler_receipt['inputs'].items():
        if sha(name) != expected:
            raise ValueError('compiler input changed: ' + name)
        inputs[name] = expected
    for name, expected in compiler_receipt['artifacts'].items():
        path = compiler / name
        if sha(path) != expected:
            raise ValueError('compiler artifact changed: ' + name)
        inputs[str(path)] = expected
    for receipt in (base_receipt, compiler_receipt):
        for tool in receipt.get('toolchain', receipt.get('tools', {})).values():
            if sha(tool['path']) != tool['sha256']:
                raise ValueError('tool changed')
            inputs[tool['path']] = tool['sha256']
    for path in (base / 'receipt.json', compiler / 'receipt.json', Path(__file__),
                 ROOT / 'tools/embedded_module_linker.py', ROOT / 'tests/public_hip_embedded_model.cpp'):
        inputs[str(path.resolve())] = sha(path)
    output.mkdir(parents=True, exist_ok=False)
    commands = []

    def run(label, argv, negative=False, deferred=True):
        argv = list(map(str, argv))
        environment = dict(os.environ, ASAN_OPTIONS='protect_shadow_gap=0:halt_on_error=1:detect_leaks=1',
                           UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1', HIP_USE_RUNTIME_UNBUNDLER='0',
                           HIP_ENABLE_DEFERRED_LOADING='1' if deferred else '0')
        try:
            result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=180, env=environment)
        except subprocess.TimeoutExpired as error:
            for suffix, value in (('stdout', error.stdout), ('stderr', error.stderr)):
                (output / (label + '.' + suffix)).write_bytes(
                    value.encode() if isinstance(value, str) else value or b'')
            commands.append({'label': label, 'argv': argv, 'timed_out': True, 'deadline_seconds': 180})
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
            raise
        for suffix, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            (output / (label + '.' + suffix)).write_text(text)
        commands.append({'label': label, 'argv': argv, 'exit_code': result.returncode,
                         'environment': {key: environment[key] for key in
                             ('ASAN_OPTIONS', 'UBSAN_OPTIONS', 'HIP_USE_RUNTIME_UNBUNDLER', 'HIP_ENABLE_DEFERRED_LOADING')}})
        (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        if negative:
            if result.returncode != -6 or 'PUBLIC_EMBEDDED_OUTPUT_MISMATCH index=0' not in result.stderr:
                raise RuntimeError('negative control did not trip output oracle: ' + label)
        elif result.returncode:
            raise RuntimeError(label + ': ' + result.stderr[-3000:])
        return result.stdout

    compile_records = []
    for path in base.glob('compile-*.json'):
        record = json.loads(path.read_text())
        if str(ROOT / 'tests/public_hip_launch_model.cpp') in record['argv']:
            compile_records.append((path, record))
    if len(compile_records) != 1:
        raise ValueError('expected one genuine public launch model compilation')
    path, record = compile_records[0]
    if record['exit_code'] != 0:
        raise ValueError('base model compilation failed')
    inputs[str(path)] = sha(path)
    original_compile = record['argv'][:]
    original_object = original_compile[original_compile.index('-o') + 1]
    link_record = base / 'launch-link.json'
    inputs[str(link_record)] = sha(link_record)
    original_link = json.loads(link_record.read_text())['argv'][:]
    if original_link.count(original_object) != 1:
        raise ValueError('base link must contain exactly one launch fixture')
    for name in original_link:
        if name.endswith('.o'):
            inputs[name] = sha(name)
    for order in ('a', 'ab', 'ba'):
        directory = output / order
        directory.mkdir()
        argv = original_compile[:]
        argv[argv.index('-c') + 1] = str(ROOT / 'tests/public_hip_embedded_model.cpp')
        model_object = directory / 'embedded-model.o'
        argv[argv.index('-o') + 1] = str(model_object)
        argv += ['-DPS5HIP_EMBEDDED_MODULES=' + str(len(order))]
        run(order + '-compile', argv)
        argv = original_link[:]
        argv[argv.index(original_object)] = str(model_object)
        binary = directory / 'model'
        argv[argv.index('-o') + 1] = str(binary)
        objects = [compiler / 'linux' / ('public-embedded' + ('-b' if module == 'b' else '') + '.o') for module in order]
        linker = directory / 'embedded.ld'
        linker.write_text(linux_linker_fragment([p.name for p in objects]))
        argv += ['-pie', *map(str, objects), '-Wl,-T,' + str(linker)]
        run(order + '-link', argv)
        for deferred in (True, False):
            normal = run(order + ('-deferred' if deferred else '-eager'), [binary, 'normal'], deferred=deferred)
            if normal.splitlines() != [f'PUBLIC_EMBEDDED_PASS compiler_stubs=true modules={len(order)} values={73*len(order)} guards={8*len(order)} gpu_execution=false',
                                       'PUBLIC_EMBEDDED_FINAL ownership=clean gpu_execution=false']:
                raise RuntimeError('compiler/CLR output or teardown differs')
        for mode in ('noop', 'corrupt'):
            run(order + '-' + mode, [binary, mode], negative=True)
    # Mutate only the compiler's linked name, leaving its device symbol and stubs intact.
    directory = output / 'missing-name'
    directory.mkdir()
    original = compiler / 'linux/public-embedded.o'
    names = section(original.read_bytes(), b'.rodata.str1.1')
    if names != b'thread_words\0':
        raise ValueError('unexpected registration string section')
    replacement = directory / 'name.bin'
    replacement.write_bytes(b'ghost__words\0')
    mutated = directory / 'public-embedded.o'
    objcopy = shutil.which('llvm-objcopy-18')
    if not objcopy:
        raise RuntimeError('missing llvm-objcopy-18')
    inputs[objcopy] = sha(objcopy)
    require_tool_versions({'llvm-objcopy-18': run('objcopy-version', [objcopy, '--version'])},
                          json.loads((ROOT / 'dependencies.json').read_text())['llvm']['version'])
    run('missing-name-mutate', [objcopy, '--update-section', '.rodata.str1.1=' + str(replacement), original, mutated])
    for name in (b'.text', b'.rela.text', b'.data.rel.ro', b'.rela.data.rel.ro', b'.hip_fatbin',
                 b'.hipFatBinSegment', b'.rela.hipFatBinSegment', b'.init_array', b'.rela.init_array'):
        if section(original.read_bytes(), name) != section(mutated.read_bytes(), name):
            raise RuntimeError('mutation changed non-name section')
    if section(mutated.read_bytes(), b'.rodata.str1.1') != b'ghost__words\0':
        raise RuntimeError('registration name mutation missing')
    argv = original_compile[:]
    argv[argv.index('-c') + 1] = str(ROOT / 'tests/public_hip_embedded_model.cpp')
    model_object = directory / 'embedded-model.o'
    argv[argv.index('-o') + 1] = str(model_object)
    argv += ['-DPS5HIP_EMBEDDED_MISSING_NAME=1']
    run('missing-name-compile', argv)
    argv = original_link[:]
    argv[argv.index(original_object)] = str(model_object)
    binary = directory / 'model'
    argv[argv.index('-o') + 1] = str(binary)
    linker = directory / 'embedded.ld'
    linker.write_text(linux_linker_fragment([mutated.name]))
    argv += ['-pie', str(mutated), '-Wl,-T,' + str(linker)]
    run('missing-name-link', argv)
    for deferred in (True, False):
        text = run('missing-name-' + ('deferred' if deferred else 'eager'), [binary, 'normal'], deferred=deferred)
        if text.splitlines() != ['PUBLIC_EMBEDDED_MISSING_NAME_PASS attribute_and_launch_rejected=true retries=2 submits=0 gpu_execution=false',
                                'PUBLIC_EMBEDDED_FINAL ownership=clean gpu_execution=false']:
            raise RuntimeError('missing symbol rejection or cleanup differs')
    for name, expected in inputs.items():
        if sha(name) != expected:
            raise RuntimeError('input changed during qualification: ' + name)
    result = {'classification': 'genuine-compiler-public-clr-embedded-host-model', 'inputs': inputs,
              'artifacts': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()},
              'runtime_executed': True, 'native_executed': False, 'gpu_executed': False,
              'cases': ['compiler-ctor', 'compiler-launch-stub', 'single-module-73-values-8-guards', 'two-modules-146-values-16-guards', 'reversed-object-and-span-order', 'deferred-and-eager-loading', 'compiler-dtor',
                        'clean-modeled-device-backing-and-queues', 'noop-rejected', 'corruption-rejected',
                        'linked-missing-name-attribute-and-launch-rejected-twice-deferred-and-eager'],
              'limits': ['One/two compiler modules; transport arithmetic is modeled, not GPU execution.',
                         'Final ownership checks cover modeled device backing and queues; static registry retirement counts are not measured.',
                         'Reuses currently hashed core objects from source-qualified build; base receipt does not independently bind every object byte. Fresh builds must reproduce the base first.']}
    (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    qualify(args.base, args.compiler, args.output)
