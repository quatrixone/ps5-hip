#!/usr/bin/env python3
"""Capture the executed batch-one CPU graph; this does not establish HIP coverage."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
import re
import shlex
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / 'tests/llm_executed_graph.cpp'
TEST = ROOT / 'tests/llm_executed_graph_test.py'
POSITIONS = (0, 1, 2, 3, 31, 32, 255, 256, 510, 511)
PROMPT = (1, 9038, 2501, 263)
OPS = {'GET_ROWS', 'DUP', 'SET_ROWS', 'RMS_NORM', 'MUL', 'MUL_MAT', 'RESHAPE',
       'ROPE', 'VIEW', 'PERMUTE', 'SOFT_MAX', 'CONT', 'ADD', 'SWIGLU'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def merge(inputs, key, digest):
    key = str(key)
    if key in inputs and inputs[key] != digest:
        raise ValueError('conflicting input identity')
    inputs[key] = digest


def dependencies(path):
    text = path.read_text().replace('\\\n', ' ')
    if ': ' not in text:
        raise ValueError('invalid dependency record')
    result = {str(Path(p).resolve()) for p in shlex.split(text.split(': ', 1)[1])}
    if not result:
        raise ValueError('empty dependency record')
    return result


def runtime_inputs(text, lib, authenticated):
    paths = {}
    names = set()
    for line in text.splitlines():
        if 'not found' in line:
            raise ValueError('runtime library unresolved')
        match = re.search(r'=> (/.+) \(', line) or re.match(r'\s*(/.+) \(', line)
        if not match:
            if 'linux-vdso.so' in line or not line.strip():
                continue
            raise ValueError('unrecognized runtime dependency')
        path = Path(match.group(1)).resolve()
        digest = sha(path)
        if path.name.startswith(('libllama.', 'libggml')):
            names.add(path.name.split('.so')[0])
            if not path.is_relative_to(lib.resolve()) or not any(Path(k).resolve() == path and v == digest for k, v in authenticated.items()):
                raise ValueError('unauthenticated CPU runtime library')
        merge(paths, path, digest)
    if not {'libllama', 'libggml', 'libggml-base', 'libggml-cpu'}.issubset(names):
        raise ValueError('missing CPU runtime libraries')
    return paths


def recheck(inputs):
    if any(sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('input changed during graph qualification')


def inventory(path):
    counts, ops, signatures, attention = Counter(), Counter(), {}, {}
    total = 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            position = row['position']
            if position not in POSITIONS or row['node'] != total:
                raise ValueError('unexpected position/node sequence')
            if row['phase'] != ('prefill' if position < 4 else 'decode'):
                raise ValueError('phase mismatch')
            t = row['tensor']
            indices = [s['index'] for s in row['sources']]
            if indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < 10 for i in indices):
                raise ValueError('source indices invalid')
            for x in [t] + [s['tensor'] for s in row['sources']]:
                if (len(x['ne']) != 4 or len(x['nb']) != 4 or
                        any(type(v) is not int or v <= 0 for v in x['ne'] + x['nb']) or
                        type(x['view_offset']) is not int or x['view_offset'] < 0 or
                        len(x['op_params_hex']) != 128 or len(bytes.fromhex(x['op_params_hex'])) != 64):
                    raise ValueError('tensor shape/stride/parameter metadata invalid')
            signature = json.dumps(dict(tensor=t, sources=row['sources']), sort_keys=True)
            if signature not in signatures:
                signatures[signature] = dict(first_position=position, count=0, tensor=t, sources=row['sources'])
            signatures[signature]['count'] += 1
            counts[position] += 1
            ops[t['description']] += 1
            total += 1
            if t['name'] in ('kq-0', 'kqv-0'):
                attention.setdefault(position, {})[t['name']] = row
    if set(counts) != set(POSITIONS) or set(ops) != OPS:
        raise ValueError('missing positions or changed operator contract')
    for position in POSITIONS:
        expected = 256 if position < 256 else 512
        if attention.get(position, {}).get('kq-0', {}).get('tensor', {}).get('ne') != [expected, 1, 32, 1]:
            raise ValueError('attention cache growth mismatch')
    return dict(node_count=total, positions=dict(counts), ops=dict(ops),
                signatures=list(signatures.values()), attention=attention)


def validate_steps(logits_path, steps_path, trace):
    steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
    if len(steps) != 512 or logits_path.stat().st_size != 512 * 32000 * 4:
        raise ValueError('incomplete context/logits')
    previous = None
    with logits_path.open('rb') as stream:
        for position, step in enumerate(steps):
            row = struct.unpack('<32000f', stream.read(128000))
            if not all(math.isfinite(x) for x in row):
                raise ValueError('nonfinite logits')
            best = max(range(32000), key=row.__getitem__)
            if (step['position'] != position or step['phase'] != ('prefill' if position < 4 else 'decode') or
                    step['input_token'] != (PROMPT[position] if position < 4 else previous) or
                    step['next_token'] != best or type(step['completed_nodes']) is not int or step['completed_nodes'] <= 0):
                raise ValueError('sequential token/step contract mismatch')
            if position in POSITIONS and step['completed_nodes'] != trace['positions'][position]:
                raise ValueError('post-compute node count mismatch')
            previous = best
    return dict(positions=512, completed_nodes=sum(s['completed_nodes'] for s in steps),
                finite_logits=512 * 32000, greedy_tie='lowest token ID')


def cpu_inputs(cpu, model):
    client = ROOT / '.deps/sources/llama-client'
    lock = json.loads((ROOT / 'llm-inputs.json').read_text())
    receipt = json.loads((cpu / 'receipt.json').read_text())
    expected = json.loads((ROOT / 'evidence/offline-llm-model-conversion.json').read_text())['outputs']['tinyllama-q4_0.gguf']
    if receipt['classification'] != 'host-cpu-reference-tools-built' or receipt['source_revision'] != lock['client']['revision']:
        raise ValueError('CPU source identity mismatch')
    if subprocess.check_output(['git', '-C', str(client), 'rev-parse', 'HEAD'], text=True).strip() != lock['client']['revision'] or subprocess.check_output(['git', '-C', str(client), 'status', '--porcelain'], text=True).strip():
        raise ValueError('client must be clean and locked')
    if model.stat().st_size != expected['bytes'] or sha(model) != expected['sha256']:
        raise ValueError('model differs from qualified conversion')
    identities = {str(cpu / 'receipt.json'): sha(cpu / 'receipt.json'), str(model): expected['sha256']}
    for key, digest in receipt['identity']['inputs'].items():
        merge(identities, ROOT / key, digest)
    for tool in receipt['identity']['tools'].values():
        merge(identities, tool['path'], tool['sha256'])
    artifacts = receipt['artifacts']
    if not artifacts or not all(any(Path(p).name.startswith(n) for p in artifacts) for n in ('libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so')):
        raise ValueError('CPU libraries missing from receipt')
    for key, digest in artifacts.items():
        path = Path(key)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe CPU artifact path')
        merge(identities, cpu / path, digest)
    recheck(identities)
    return identities


def build(output, cpu, model):
    if output.exists() or not output.is_relative_to(ROOT / 'results'):
        raise ValueError('fresh output directory below results/ required')
    compiler = Path('/usr/bin/clang++-18')
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    if '18.1.8' not in version:
        raise ValueError('LLVM18.1.8 required')
    inputs = {str(p): sha(p) for p in (Path(__file__), HARNESS, TEST, compiler, Path('/usr/bin/ldd'),
              ROOT / 'llm-inputs.json', ROOT / 'evidence/offline-llm-model-conversion.json')}
    for key, digest in cpu_inputs(cpu, model).items():
        merge(inputs, key, digest)
    output.mkdir(parents=True)
    commands = []
    env = {k: v for k, v in os.environ.items() if k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'OBJC_INCLUDE_PATH', 'LIBRARY_PATH', 'COMPILER_PATH', 'GCC_EXEC_PREFIX')}
    def run(label, argv):
        record = dict(label=label, argv=list(map(str, argv)))
        commands.append(record)
        try:
            with (output / (label + '.stdout')).open('w') as out, (output / (label + '.stderr')).open('w') as err:
                result = subprocess.run(record['argv'], stdout=out, stderr=err, env=env, timeout=900)
            record['exit_code'] = result.returncode
            result.check_returncode()
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            raise
        finally:
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    client = ROOT / '.deps/sources/llama-client'
    lib = cpu / 'cmake/bin'
    binary = output / 'graph'
    flags = ['-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
             '-I' + str(client / 'include'), '-I' + str(client / 'ggml/include')]
    run('dependencies', [compiler, *flags, '-M', '-MT', 'graph', '-MF', output / 'headers.d', HARNESS])
    headers = dependencies(output / 'headers.d')
    for path in headers:
        merge(inputs, path, sha(path))
    recheck(inputs)
    run('compile', [compiler, *flags, '-MD', '-MT', 'graph', '-MF', output / 'compiled.d', HARNESS,
        '-L' + str(lib), '-Wl,-rpath,' + str(lib), '-lllama', '-lggml', '-lggml-base', '-o', binary])
    if dependencies(output / 'compiled.d') != headers:
        raise ValueError('compile header closure changed')
    run('runtime-libraries', ['/usr/bin/ldd', binary])
    for path, digest in runtime_inputs((output / 'runtime-libraries.stdout').read_text(), lib, inputs).items():
        merge(inputs, path, digest)
    binary_sha = sha(binary)
    run('execute', [binary, model, output / 'run'])
    trace = inventory(output / 'run.trace.jsonl')
    steps = validate_steps(output / 'run.f32le', output / 'run.steps.jsonl', trace)
    (output / 'inventory.json').write_text(json.dumps(trace, indent=2) + '\n')
    if sha(binary) != binary_sha:
        raise ValueError('executed binary changed')
    cpu_inputs(cpu, model)
    recheck(inputs)
    receipt = dict(schema=1, status='PASS', classification='executed-cpu-graph-inventory', inputs=inputs,
        artifacts={p.name: sha(p) for p in output.iterdir() if p.is_file()}, commands=commands,
        compiler_version=version, configuration=dict(context=512, batch=1, ubatch=1, threads=4,
        prefill=4, decode=508, prompt_tokens=list(PROMPT), kv='F16', flash=False, cpu_only=True,
        use_extra_weight_buffers=False), steps=steps, traced_positions=list(POSITIONS), ops=trace['ops'],
        native_executed=False, gpu_executed=False, hip_calls_observed=False,
        limits=['CPU scheduler callback synchronizes individual graph nodes; fusion and timing are diagnostic.',
                'All512 positions execute; tensor metadata retained at ten stated checkpoints only.',
                'No tensor-value oracle, native inference, HIP kernel/library/call coverage, or context-reset proof.'])
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['steps']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-build', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    build(args.output.resolve(), args.cpu_build.resolve(), args.model.resolve())
