#!/usr/bin/env python3
"""Compare a built compiler against current pinned device fixtures; never deploy."""
import argparse
import hashlib
import json
import pathlib
import subprocess
from build_canaries import require_device_linker

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command, output):
    result = subprocess.run([str(x) for x in command], text=True, capture_output=True)
    output.write_text(result.stdout + result.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-bin', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    tools = {name: args.compiler_bin.resolve() / name for name in ('clang++', 'lld', 'llvm-readobj', 'llvm-objdump')}
    if not all(p.is_file() for p in tools.values()):
        parser.error('compiler directory must contain clang++, lld, llvm-readobj and llvm-objdump')
    if args.output.exists():
        parser.error('use a new output directory; existing comparison evidence is preserved')
    inspector = ROOT / 'build/host/ps5hip-inspect'
    if not inspector.is_file():
        parser.error('build the host object inspector first')
    reports = {}
    for fixture in ('canary', 'resources'):
        report = json.loads((ROOT / 'build/canaries' / ('build.json' if fixture == 'canary' else 'resources-build.json')).read_text())
        if sorted(p['target'] for p in report['profiles']) != ['gfx1013', 'gfx1030']:
            parser.error('baseline must contain exactly the two declared targets')
        if (report['source_sha256'] != sha(ROOT / 'kernels' / (fixture + '.hip')) or
                report['recipe_sha256'] != sha(ROOT / 'tools/build_canaries.py')):
            parser.error('baseline source or recipe changed; rebuild baseline')
        for profile in report['profiles']:
            relative = pathlib.Path(profile['target']) / (fixture + '.hsaco' if fixture == 'canary' else 'resources/resources.hsaco')
            if sha(ROOT / 'build/canaries' / relative) != profile['code_sha256']:
                parser.error('baseline object changed')
        reports[fixture] = report
    args.output.mkdir(parents=True)
    result = {'schema': 1, 'hardware_validated': False, 'tools': {}, 'cases': []}
    for name, path in tools.items():
        version = capture([path, *(['-flavor', 'gnu'] if name == 'lld' else []), '--version'],
                          args.output / (name.replace('+', 'p') + '-version.txt'))
        version.check_returncode()
        result['tools'][name] = {'sha256': sha(path), 'version': version.stdout.splitlines()[0]}
    for fixture, report in reports.items():
        for profile in report['profiles']:
            target = profile['target']
            directory = args.output / (target + '-' + fixture)
            directory.mkdir()
            code = directory.resolve() / (fixture + '.hsaco')
            command = list(profile['compile_command'])
            command[0] = str(tools['clang++'])
            command = ['-B' + str(args.compiler_bin.resolve()) if x.startswith('-B') else x for x in command]
            command[command.index('-c') + 1] = str(ROOT / 'kernels' / (fixture + '.hip'))
            command[command.index('-o') + 1] = str(code)
            case = {'target': target, 'fixture': fixture, 'source_sha256': report['source_sha256'],
                    'baseline_sha256': profile['code_sha256'], 'compiled': False}
            result['cases'].append(case)
            trace = capture(command + ['-###'], directory / 'driver.txt')
            if trace.returncode:
                case['failure'] = 'driver rejected profile'
                continue
            require_device_linker(trace.stderr, tools['lld'])
            compiled = capture(command, directory / 'compile.txt')
            if compiled.returncode:
                case['failure'] = 'compilation failed'
                continue
            case.update(compiled=True, code_sha256=sha(code), bytes=code.stat().st_size)
            case['byte_identical'] = case['code_sha256'] == case['baseline_sha256']
            for label, command in (
                ('metadata', [tools['llvm-readobj'], '--file-headers', '--program-headers', '--sections', '--symbols', '--relocations', '--notes', code]),
                ('isa', [tools['llvm-objdump'], '--disassemble', '--mcpu=' + target, code]),
                ('loader', [inspector, code])):
                check = capture(command, directory / (label + '.txt'))
                case[label + '_accepted'] = check.returncode == 0
                case[label + '_report_sha256'] = sha(directory / (label + '.txt'))
            case['loader_report'] = (directory / 'loader.txt').read_text()
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if all(c['compiled'] and c.get('metadata_accepted') and c.get('isa_accepted') and c.get('loader_accepted') for c in result['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
