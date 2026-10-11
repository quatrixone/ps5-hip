#!/usr/bin/env python3
"""Compile and inspect isolated wave32 probes; no native launch or baseline edits."""
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
    source = ROOT / 'kernels/wave.hip'
    result = {'schema': 1, 'hardware_validated': False, 'source_sha256': sha(source),
              'recipe_sha256': sha(pathlib.Path(__file__)), 'shared_recipe_sha256': sha(ROOT / 'tools/build_canaries.py'),
              'tool_versions': versions, 'cases': [], 'oracle': {}}
    # Independent scalar oracle: collect active lanes and sum their original values.
    for name, first, count in (('wave_full', 0, 32), ('wave_partial', 4, 16)):
        values = [0xdeadbeef] * 256
        for base in (0, 32):
            active = list(range(first, first + count))
            ballot = sum(1 << lane for lane in active if (base + lane + 1) & 1)
            total = sum(base + lane + 1 for lane in active)
            for lane in active:
                i = base + lane
                values[i * 4:i * 4 + 4] = [base + first + 1, base + first + ((lane - first) ^ 1) + 1, ballot, total]
        if name == 'wave_full':
            assert values[:4] == [1, 2, 0x55555555, 528]
            assert values[128:132] == [33, 34, 0x55555555, 1552]
        else:
            assert values[16:20] == [5, 6, 0x55550, 200]
            assert values[144:148] == [37, 38, 0x55550, 712]
            assert all(values[i * 4:i * 4 + 4] == [0xdeadbeef] * 4 for i in range(64) if not 4 <= (i & 31) < 20)
        raw = struct.pack('<256I', *values)
        (args.output / (name + '-expected.bin')).write_bytes(raw)
        result['oracle'][name] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    for target in ('gfx1013', 'gfx1030'):
        for opt in ('-O1', '-O2'):
            directory = args.output / (target + opt)
            directory.mkdir()
            code = directory.resolve() / 'wave.hsaco'
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
            expected = {'wave_full', 'wave_partial'}
            assert set(re.findall(r'^\s+\.name:\s+(\w+)\s*$', metadata, re.MULTILINE)) == expected
            for field, value in (('wavefront_size', '32'), ('private_segment_fixed_size', '0'), ('group_segment_fixed_size', '0')):
                assert re.findall(r'\.' + field + r':\s*(\d+)', metadata) == [value, value]
            inventory = isa_inventory(disassembly, expected)
            for kernel in inventory.values():
                assert {'ds_bpermute_b32', 'v_readfirstlane_b32'} <= set(kernel['opcodes'])
                assert not kernel['flat_scratch_setup'] and 's_swappc_b64' not in kernel['opcodes']
            partial_ops = set(inventory['wave_partial']['opcodes'])
            assert 's_cbranch_execz' in partial_ops
            assert partial_ops & {'s_and_saveexec_b32', 'v_cmpx_gt_u32_e32'}
            result['cases'].append({'target': target, 'optimization': opt, 'command': command,
                                    'code_sha256': sha(code), 'isa': inventory,
                                    'metadata_sha256': sha(directory / 'metadata.txt'),
                                    'disassembly_sha256': sha(directory / 'isa.txt')})
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Four wave32 compiler profiles and scalar oracle self-checks passed; no hardware qualification.')


if __name__ == '__main__':
    main()
