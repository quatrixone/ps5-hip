#!/usr/bin/env python3
"""Compile isolated static/dynamic LDS fixtures; no native execution or baseline writes."""
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
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('use a fresh output directory to preserve evidence')
    tools = {name: shutil.which(name + '-18') for name in ('clang++', 'ld.lld', 'llvm-readobj', 'llvm-objdump')}
    if not all(tools.values()):
        parser.error('LLVM 18 compiler, linker and inspectors required')
    versions = {name: subprocess.check_output([path, '--version'], text=True) for name, path in tools.items()}
    require_tool_versions(versions, json.loads((ROOT / 'dependencies.json').read_text())['llvm']['version'])
    args.output.mkdir(parents=True)
    source = ROOT / 'kernels/lds.hip'
    result = {'schema': 1, 'hardware_validated': False, 'source_sha256': sha(source),
              'recipe_sha256': sha(pathlib.Path(__file__)), 'shared_recipe_sha256': sha(ROOT / 'tools/build_canaries.py'),
              'tool_versions': versions, 'cases': [], 'oracle': {}}
    expected = {name: [] for name in ('lds_static', 'lds_dynamic', 'lds_mixed')}
    for group in range(257):
        values = [group * 4099 + lane * 17 + (65536 if lane >= 32 else 0) for lane in range(64)]
        total = sum(values)
        assert total == group * 262336 + 2131424 and total < 2**32
        # Independently model each barrier-separated reduction phase and padded layout.
        for stride in (1, 5, 16):
            shared = [0xdeadbeef] * (64 * stride)
            for lane, value in enumerate(values):
                shared[lane * stride] = value
            for step in (32, 16, 8, 4, 2, 1):
                before = shared.copy()
                for lane in range(step):
                    shared[lane * stride] = before[lane * stride] + before[(lane + step) * stride]
            assert shared[0] == total
            assert all(value == 0xdeadbeef for index, value in enumerate(shared) if index % stride)
        expected['lds_static'].append(total)
        expected['lds_dynamic'].append(total)
        expected['lds_mixed'].append(total ^ ((group * 64 + 33) ^ 0x53a971bd))
    assert len(set(expected['lds_static'])) == 257
    for name, values in expected.items():
        data = struct.pack('<257I', *values)
        (args.output / (name + '-expected.bin')).write_bytes(data)
        result['oracle'][name] = {'groups': 257, 'threads_per_group': 64, 'bytes': len(data),
                                  'sha256': hashlib.sha256(data).hexdigest()}
    for target in ('gfx1013', 'gfx1030'):
        for opt in ('-O1', '-O2'):
            directory = args.output / (target + opt)
            directory.mkdir()
            code = directory.resolve() / 'lds.hsaco'
            command = [tools['clang++'], '-x', 'hip', '--cuda-device-only', '--no-gpu-bundle-output',
                       '-B' + str(pathlib.Path(tools['ld.lld']).resolve().parent), '-nogpuinc', '-nogpulib',
                       '--offload-arch=' + target, '-mcode-object-version=5', '-mno-wavefrontsize64',
                       opt, '-std=c++17', '-fno-fast-math', '-c', str(source), '-o', str(code)]
            trace = subprocess.run(command + ['-###'], capture_output=True, text=True, check=True).stderr
            (directory / 'driver.txt').write_text(trace)
            require_device_linker(trace, tools['ld.lld'])
            compilation = subprocess.run(command, capture_output=True, text=True)
            (directory / 'compile.txt').write_text(compilation.stdout + compilation.stderr)
            compilation.check_returncode()
            metadata = subprocess.check_output([tools['llvm-readobj'], '--notes', '--symbols', '--relocations', str(code)], text=True)
            isa = subprocess.check_output([tools['llvm-objdump'], '--disassemble', '--mcpu=' + target, str(code)], text=True)
            (directory / 'metadata.txt').write_text(metadata)
            (directory / 'isa.txt').write_text(isa)
            assert 'Relocations [\n]' in metadata and 'hidden_' not in metadata
            profiles = {}
            for block in re.split(r'(?m)^  - \.args:', metadata)[1:]:
                name = re.search(r'\.name:\s+(\w+)', block)[1]
                profiles[name] = {field: int(re.search(r'\.' + field + r':\s*(\d+)', block)[1])
                                  for field in ('group_segment_fixed_size', 'private_segment_fixed_size', 'kernarg_segment_size', 'wavefront_size')}
            assert set(profiles) == set(expected)
            for name, profile in profiles.items():
                assert profile == {'group_segment_fixed_size': 0 if name == 'lds_dynamic' else 256,
                                   'private_segment_fixed_size': 0, 'kernarg_segment_size': 16 if name == 'lds_static' else 20,
                                   'wavefront_size': 32}
            inventory = isa_inventory(isa, set(expected))
            for kernel in inventory.values():
                ops = kernel['opcodes']
                assert 's_barrier' in ops and any(op.startswith('ds_read') for op in ops) and any(op.startswith('ds_write') for op in ops)
                assert not kernel['flat_scratch_setup'] and 's_swappc_b64' not in ops
            result['cases'].append({'target': target, 'optimization': opt, 'command': command,
                                    'code_sha256': sha(code), 'bytes': code.stat().st_size,
                                    'driver_sha256': sha(directory / 'driver.txt'), 'profiles': profiles, 'isa': inventory,
                                    'metadata_sha256': sha(directory / 'metadata.txt'), 'disassembly_sha256': sha(directory / 'isa.txt')})
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Four LDS compiler profiles and 257-group scalar/padded-layout oracles passed; no GPU evidence.')


if __name__ == '__main__':
    main()
