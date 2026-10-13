#!/usr/bin/env python3
"""Compile and inspect isolated dependent visibility probes; no native launch or baseline edits."""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import struct
import subprocess
from build_canaries import isa_inventory, require_device_linker, require_tool_versions

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a new output directory to preserve evidence')
    tools = {name: shutil.which(name + '-18') for name in ('clang++', 'ld.lld', 'llvm-readobj', 'llvm-objdump')}
    if not all(tools.values()):
        parser.error('LLVM 18 compiler, linker and inspection tools are required')
    versions = {name: subprocess.check_output([path, '--version'], text=True) for name, path in tools.items()}
    require_tool_versions(versions, json.loads((ROOT / 'dependencies.json').read_text())['llvm']['version'])
    args.output.mkdir(parents=True)
    source = ROOT / 'kernels/visibility.hip'
    result = {'schema': 1, 'hardware_validated': False, 'source_sha256': sha(source),
              'recipe_sha256': sha(pathlib.Path(__file__)), 'shared_recipe_sha256': sha(ROOT / 'tools/build_canaries.py'),
              'tool_versions': versions, 'cases': [], 'oracle': {'elements':257,'rounds':3,'groups':5,'threads':64,'dispatches':6}}
    for target in ('gfx1013', 'gfx1030'):
        for opt in ('-O1', '-O2'):
            directory = args.output / (target + opt)
            directory.mkdir()
            code = directory.resolve() / 'visibility.hsaco'
            command = [tools['clang++'], '-x', 'hip', '--cuda-device-only', '--no-gpu-bundle-output',
                       '-B' + str(pathlib.Path(tools['ld.lld']).resolve().parent), '-nogpuinc', '-nogpulib',
                       '--offload-arch=' + target, '-mcode-object-version=5', '-mno-wavefrontsize64',
                       opt, '-std=c++17', '-fno-fast-math', '-c', str(source), '-o', str(code)]
            trace = subprocess.run(command + ['-###'], capture_output=True, text=True, check=True).stderr
            (directory / 'driver.txt').write_text(trace)
            require_device_linker(trace, tools['ld.lld'])
            compile_result = subprocess.run(command, capture_output=True, text=True)
            (directory / 'compile.txt').write_text(compile_result.stdout + compile_result.stderr)
            compile_result.check_returncode()
            metadata = subprocess.check_output([tools['llvm-readobj'], '--notes', str(code)], text=True)
            disassembly = subprocess.check_output([tools['llvm-objdump'], '--disassemble', '--mcpu=' + target, str(code)], text=True)
            (directory / 'metadata.txt').write_text(metadata)
            (directory / 'isa.txt').write_text(disassembly)
            expected = {'visibility_producer', 'visibility_consumer'}
            assert set(re.findall(r'^\s+\.name:\s+(\w+)\s*$', metadata, re.MULTILINE)) == expected
            for field, value in (('wavefront_size', '32'), ('private_segment_fixed_size', '0'), ('group_segment_fixed_size', '0')):
                assert re.findall(r'\.' + field + r':\s*(\d+)', metadata) == [value, value]
            inventory = isa_inventory(disassembly, expected)
            for kernel in inventory.values():
                ops = kernel['opcodes']
                assert any(op.startswith('global_load_') for op in ops)
                assert any(op.startswith('global_store_') for op in ops)
                assert not kernel['flat_scratch_setup']
                assert not any(op.startswith(('scratch_', 'flat_', 'buffer_', 'ds_', 's_swappc')) for op in ops)
            assert re.findall(r'\.kernarg_segment_size:\s*(\d+)', metadata) == ['16', '16']
            result['cases'].append({'target': target, 'optimization': opt, 'command': command,
                                    'code_sha256': sha(code), 'bytes': code.stat().st_size,
                                    'driver_sha256': sha(directory / 'driver.txt'), 'isa': inventory,
                                    'metadata_sha256': sha(directory / 'metadata.txt'),
                                    'disassembly_sha256': sha(directory / 'isa.txt')})
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Four dependent visibility compiler profiles passed; no hardware qualification.')


if __name__ == '__main__':
    main()
