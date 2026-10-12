#!/usr/bin/env python3
"""Compile and audit the retained-call fixture offline; never qualifies hardware."""
import argparse
import hashlib
import json
import pathlib
import re
import shutil
import struct
import subprocess
from build_canaries import require_device_linker, require_tool_versions

ROOT = pathlib.Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def inspect_calls(isa, assembly, ir, metadata):
    require('Relocations [\n]' in metadata, 'unresolved relocations')
    symbols = {m[2]: (int(m[1], 16), m[3]) for m in re.finditer(
        r'(?ms)^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)', isa)}
    names = ('nested_private', '_Z11call_middlePjj', '_Z9call_leafPjj')
    require(all(name in symbols for name in names), 'missing retained function')
    graph = []
    for caller, callee in zip(names, names[1:]):
        body = symbols[caller][1]
        require(len(re.findall(r'\bs_swappc_b64\b', body)) == 1, 'expected one retained call')
        require(re.search(r's_swappc_b64 s\[30:31\], s\[6:7\]\s', body), 'call register pair mismatch')
        pc = re.search(r's_getpc_b64 s\[6:7\].*?// ([0-9A-F]+):', body)
        require(pc is not None, 'missing call PC base')
        add = re.search(r's_add_u32 s6, s6, (0x[0-9a-f]+)', body[pc.end():])
        require(add is not None, 'missing direct call displacement')
        delta = int(add[1], 16)
        if delta & (1 << 31):
            delta -= 1 << 32
        high = re.search(r's_addc_u32 s7, s7, (-?\d+)\s', body[pc.end():])
        require(high is not None and int(high[1]) == (-1 if delta < 0 else 0), 'call displacement high word mismatch')
        target = int(pc[1], 16) + 4 + delta
        require(target == symbols[callee][0], 'direct call target mismatch')
        graph.append({'caller': caller, 'callee': callee, 'address': target})
    entry, middle, leaf = (symbols[name][1] for name in names)
    require('s_swappc' not in leaf and 's_setpc_b64 s[30:31]' in leaf, 'unexpected leaf calls/return')
    require(len(re.findall(r'^\s*flat_store_dword\b', leaf, re.M)) == 1 and
            len(re.findall(r'^\s*flat_load_dword\b', leaf, re.M)) == 2, 'missing leaf private access')
    post = middle.split('s_swappc_b64', 1)[1]
    operations = [line.split('//')[0].strip() for line in post.splitlines() if '//' in line]
    reload = next((i for i, op in enumerate(operations) if re.fullmatch(
        r'buffer_load_dword v\d+, v\d+, s\[0:3\], 0 offen', op)), None)
    require(reload is not None and reload + 3 < len(operations), 'missing post-call memory observation')
    private_register = operations[reload].split()[1].rstrip(',')
    global_load = re.fullmatch(r'flat_load_dword (v\d+), v\[\d+:\d+\]', operations[reload + 1])
    require(global_load and operations[reload + 2] == 's_waitcnt vmcnt(0) lgkmcnt(0)' and
            operations[reload + 3] == f'v_add3_u32 v0, {private_register}, v0, {global_load[1]}',
            'post-call reload does not feed returned sum')
    require('s_movk_i32 s32, 0x1200' in entry and 's_addk_i32 s32, 0x1200' in middle and
            's_addk_i32 s32, 0xee00' in middle, 'unexpected fixed frames')
    require('HW_REG_FLAT_SCR_LO' in entry and 'HW_REG_FLAT_SCR_HI' in entry, 'flat setup changed')
    descriptor = re.search(r'\.amdhsa_kernel nested_private\n(.*?)\.end_amdhsa_kernel', assembly, re.S)
    require(descriptor is not None, 'missing nested descriptor')
    fields = {k: int(v, 0) for k, v in re.findall(r'^\s*\.amdhsa_(\w+)\s+(\w+)', descriptor[1], re.M)}
    require(fields.get('private_segment_fixed_size') == 288 and fields.get('user_sgpr_count') == 8 and
            fields.get('uses_dynamic_stack') == 0, 'unexpected stack/entry contract')
    attrs = dict(re.findall(r'^attributes #(\d+) = \{(.*?)\}', ir, re.M))
    for name in names[1:]:
        decl = re.search(r'^define[^\n]* @' + name + r'\([^\n]*#(\d+)[^\n]*', ir, re.M)
        require(decl and 'noinline' in attrs.get(decl[1], ''), 'retained noinline contract missing')
    return {'call_graph': graph, 'descriptor': fields, 'post_call_reload_present': True}


def oracle(x, drop_store=False):
    mask = 0xffffffff
    outer = [(x + j) & mask for j in range(32)]
    inner = [(outer[j] + j) & mask for j in range(32)]
    k, q = x & 31, (x + 7) & 31
    updated = (inner[k] + 3) & mask
    if not drop_store:
        inner[k] = updated
    return (updated + inner[q] + inner[k] + outer[k]) & mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'use a fresh output directory')
    tools = {name: shutil.which(name + '-18') for name in ('clang++', 'ld.lld', 'llvm-readobj', 'llvm-objdump')}
    require(all(tools.values()), 'LLVM18 tools required')
    versions = {n: subprocess.check_output([p, '--version'], text=True) for n, p in tools.items()}
    require_tool_versions(versions, json.loads((ROOT / 'dependencies.json').read_text())['llvm']['version'])
    paths = {n: pathlib.Path(p).resolve() for n, p in tools.items()}
    for name in ('clang++', 'ld.lld'):
        for path in re.findall(r'=> (/\S+) ', subprocess.check_output(['ldd', str(paths[name])], text=True)):
            if 'libLLVM' in path or 'libclang-cpp' in path:
                paths[pathlib.Path(path).name] = pathlib.Path(path).resolve()
    before = {n: identity(p) for n, p in paths.items()}
    source = ROOT / 'kernels/private_calls.hip'
    args.output.mkdir(parents=True)
    result = {'hardware_validated': False, 'source': identity(source), 'recipe': identity(pathlib.Path(__file__)),
              'shared_recipe': identity(ROOT / 'tools/build_canaries.py'), 'tools': before, 'versions': versions, 'profiles': []}
    inputs = [i * 4099 + 17 for i in range(257 * 64)]
    edges = list(range(32)) + list(range(0xffffffc0, 0x100000000))
    for x in inputs + edges:
        expected = (4*x + 5*(x & 31) + 2*((x + 7) & 31) + 6) & 0xffffffff
        require(oracle(x) == expected and (expected - oracle(x, True)) & 0xffffffff == 3, 'store-loss oracle failed')
    for name, values in [('input', inputs), ('expected', list(map(oracle, inputs)))]:
        (args.output / (name + '.bin')).write_bytes(struct.pack('<' + str(len(values)) + 'I', *values))
    for target in ('gfx1013', 'gfx1030'):
        for opt in ('O1', 'O2'):
            folder = args.output / (target + '-' + opt)
            folder.mkdir()
            base = [tools['clang++'], '-x', 'hip', '--cuda-device-only', '--no-gpu-bundle-output',
                    '-B' + str(paths['ld.lld'].parent), '-nogpuinc', '-nogpulib', '--offload-arch=' + target,
                    '-mcode-object-version=5', '-mno-wavefrontsize64', '-' + opt, '-std=c++17', '-fno-fast-math']
            commands = []
            for mode, extra, name in [('linked', [], 'private_calls.hsaco'), ('assembly', ['-S'], 'private_calls.s'),
                                      ('ir', ['-S', '-emit-llvm'], 'private_calls.ll')]:
                cmd = base + extra + ['-c', str(source), '-o', str(folder / name)]
                if mode == 'linked':
                    trace = subprocess.run(cmd + ['-###'], capture_output=True, text=True, check=True).stderr
                    (folder / 'driver.txt').write_text(trace)
                    require_device_linker(trace, tools['ld.lld'])
                run = subprocess.run(cmd, capture_output=True, text=True)
                (folder / (mode + '.log')).write_text(run.stdout + run.stderr)
                commands.append({'command': cmd, 'exit_code': run.returncode})
                (folder / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
                run.check_returncode()
            metadata = subprocess.check_output([tools['llvm-readobj'], '--notes', '--symbols', '--relocations', str(folder / 'private_calls.hsaco')], text=True)
            isa = subprocess.check_output([tools['llvm-objdump'], '-d', '--mcpu=' + target, str(folder / 'private_calls.hsaco')], text=True)
            (folder / 'metadata.txt').write_text(metadata)
            (folder / 'isa.txt').write_text(isa)
            audit = inspect_calls(isa, (folder / 'private_calls.s').read_text(), (folder / 'private_calls.ll').read_text(), metadata)
            result['profiles'].append({'target': target, 'optimization': opt, 'audit': audit,
                                       'files': {p.name: identity(p) for p in sorted(folder.iterdir())}})
    require(before == {n: identity(p) for n, p in paths.items()}, 'tool identity changed')
    result['oracle'] = {'values': len(inputs), 'boundary_inputs': len(edges), 'store_loss_difference': 3,
                        'files': {n: identity(args.output / n) for n in ('input.bin', 'expected.bin')}}
    (args.output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print('Four retained-call profiles and observable-store oracles passed; hardware remains unqualified.')


if __name__ == '__main__':
    main()
