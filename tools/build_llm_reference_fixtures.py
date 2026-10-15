"""Capture raw CPU logits and greedy token fixtures from the locked Q4_0 model."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

from convert_llm_model import sha

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / '.deps/sources/llama-client'
PROMPTS = ROOT / 'tests/fixtures/llm-reference-prompts.json'


def validate_case(case):
    tokens, steps = case['tokens'], case['decode_steps']
    if (not re.fullmatch(r'[a-z_]+', case['name']) or not tokens
            or any(type(x) is not int or not 0 <= x < 32000 for x in tokens)
            or type(steps) is not int or not 16 <= steps <= 64 or len(tokens) + steps > 512):
        raise ValueError('invalid fixed-token case')


def validate_logits(path, token_path, steps, vocabulary=32000):
    if path.stat().st_size != (steps + 1) * vocabulary * 4:
        raise ValueError('wrong full-logit byte count')
    generated = [int(token) for token in token_path.read_text().split()]
    if len(generated) != steps or any(not 0 <= token < vocabulary for token in generated):
        raise ValueError('wrong greedy token output')
    minimum, maximum = math.inf, -math.inf
    with path.open('rb') as stream:
        for index in range(steps + 1):
            row = struct.unpack('<' + 'f' * vocabulary, stream.read(vocabulary * 4))
            if any(not math.isfinite(value) for value in row):
                raise ValueError('non-finite reference logits')
            best = max(range(vocabulary), key=row.__getitem__)
            if index < steps and generated[index] != best:
                raise ValueError('generated token differs from raw-logit argmax')
            minimum, maximum = min(minimum, min(row)), max(maximum, max(row))
    return dict(rows=steps + 1, vocabulary=vocabulary, bytes=path.stat().st_size,
                sha256=sha(path), generated_tokens=generated, token_sha256=sha(token_path),
                minimum_logit=minimum, maximum_logit=maximum)


def build(output, cpu_build, model):
    if output.exists():
        raise ValueError('reference output must be a fresh directory')
    source_lock = json.loads((ROOT / 'llm-inputs.json').read_text())
    conversion = json.loads((ROOT / 'evidence/offline-llm-model-conversion.json').read_text())
    profile = json.loads(PROMPTS.read_text())
    if profile['tokenizer'] != dict(add_special=True, parse_special=True):
        raise ValueError('fixture harness requires both tokenizer flags enabled')
    expected = conversion['outputs']['tinyllama-q4_0.gguf']
    if model.stat().st_size != expected['bytes'] or sha(model) != expected['sha256']:
        raise ValueError('model differs from qualified conversion')
    if profile['model_revision'] != conversion['model_revision']:
        raise ValueError('prompt tokenizer/model revision differs')
    names = [case['name'] for case in profile['cases']]
    if len(names) != 7 or len(set(names)) != 7:
        raise ValueError('expected seven unique reference cases')
    for case in profile['cases']:
        validate_case(case)
    if (sum('prompt' in case and case['decode_steps'] >= 16 for case in profile['cases']) < 5
            or not any(case['decode_steps'] == 64 for case in profile['cases'])
            or not any(len(case['tokens']) + case['decode_steps'] == 512 for case in profile['cases'])):
        raise ValueError('reference case coverage is incomplete')
    compiler = Path('/usr/bin/clang++-18')
    compiler_version = subprocess.check_output([str(compiler), '--version'], text=True).splitlines()[0]
    if 'version 18.1.8' not in compiler_version:
        raise ValueError('reference compiler must be LLVM18.1.8')
    inputs = [Path(__file__), ROOT / 'tools/convert_llm_model.py', ROOT / 'tools/fetch_llm_model.py',
              ROOT / 'tests/llm_cpu_reference.cpp', PROMPTS, ROOT / 'llm-inputs.json',
              ROOT / 'evidence/offline-llm-model-conversion.json', cpu_build / 'receipt.json', compiler]
    identities = {str(path): sha(path) for path in inputs}
    cpu = json.loads((cpu_build / 'receipt.json').read_text())

    def check_tools():
        revision = subprocess.check_output(['git', '-C', str(CLIENT), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(CLIENT), 'status', '--porcelain'], text=True).strip()
        if dirty or revision != source_lock['client']['revision'] or cpu['source_revision'] != revision:
            raise ValueError('reference requires the clean locked client and CPU build')
        if any(sha(cpu_build / path) != digest for path, digest in cpu['artifacts'].items()):
            raise ValueError('CPU build artifact changed')
    check_tools()
    output.mkdir(parents=True)
    commands = []
    artifacts = {}
    env = dict(os.environ)
    env.pop('LD_LIBRARY_PATH', None)
    env.pop('LD_PRELOAD', None)

    def run(label, argv):
        row = dict(label=label, argv=[str(arg) for arg in argv])
        commands.append(row)
        try:
            with (output / (label + '.stdout')).open('w') as out, (output / (label + '.stderr')).open('w') as err:
                result = subprocess.run(row['argv'], stdout=out, stderr=err, env=env, timeout=600)
            row['exit_code'] = result.returncode
            result.check_returncode()
        except subprocess.TimeoutExpired:
            row['timed_out'] = True
            raise
        finally:
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        print(label + ' PASS', flush=True)

    binary = output / 'fixture'
    libraries = cpu_build / 'cmake/bin'
    run('compile', [compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    ROOT / 'tests/llm_cpu_reference.cpp', '-I' + str(CLIENT / 'include'),
                    '-I' + str(CLIENT / 'ggml/include'), '-L' + str(libraries),
                    '-Wl,-rpath,' + str(libraries), '-lllama', '-o', binary])
    binary_sha = sha(binary)
    cases = []
    for case in profile['cases']:
        name = case['name']
        token_file = output / (name + '.input.tokens')
        token_file.write_text(''.join(str(token) + '\n' for token in case['tokens']))
        artifacts[token_file.name] = sha(token_file)
        if 'prompt' in case:
            prompt_file = output / (name + '.prompt.txt')
            prompt_file.write_text(case['prompt'])
            artifacts[prompt_file.name] = sha(prompt_file)
            retokenized = output / (name + '.retokenized.tokens')
            run(name + '-tokenize', [binary, 'tokenize', model, prompt_file, retokenized, 0])
            if retokenized.read_bytes() != token_file.read_bytes():
                raise ValueError('fixed prompt tokenization changed')
            artifacts[retokenized.name] = sha(retokenized)
        runs = []
        for repeat in (1, 2):
            prefix = output / (name + f'-run{repeat}')
            run(prefix.name, [binary, 'decode', model, token_file, prefix, case['decode_steps']])
            runs.append(validate_logits(prefix.with_suffix('.f32le'), prefix.with_suffix('.tokens'), case['decode_steps']))
            artifacts[prefix.with_suffix('.f32le').name] = runs[-1]['sha256']
            artifacts[prefix.with_suffix('.tokens').name] = runs[-1]['token_sha256']
        if runs[0] != runs[1]:
            raise ValueError('repeated CPU reference differs; inspect preserved raw logits')
        cases.append(dict(name=name, input_tokens=case['tokens'], input_sha256=sha(token_file),
                          decode_steps=case['decode_steps'], repeated_bitwise_equal=True, **runs[0]))
    check_tools()
    if (sha(binary) != binary_sha or sha(model) != expected['sha256']
            or any(sha(Path(path)) != digest for path, digest in identities.items())
            or any(sha(output / path) != digest for path, digest in artifacts.items())):
        raise ValueError('reference model/tool identities changed during run')
    receipt = dict(schema=1, classification='host-cpu-logit-reference', source_revision=cpu['source_revision'],
        model_sha256=expected['sha256'], identities=identities, compiler_version=compiler_version,
        binary_sha256=binary_sha, configuration=dict(context=512, batch=512, ubatch=512, threads=4,
        gpu_layers=0, offload_kqv=False, op_offload=False, flash_attention=False, kv_type='F16',
        use_extra_weight_buffers=False, greedy_tie='lowest token ID', stop_at_eog=False),
        logits_format='little-endian IEEE float32, rows: prefill then each incremental decode',
        cases=cases, commands=commands, artifacts=artifacts, native_executed=False,
        limits=['CPU numerical reference only; no native or GPU inference.',
                'Fixed-step fixtures continue through EOG; not chat termination behavior.',
                'Each repeat uses a fresh process/context; same-context reset/unload qualification remains open.',
                'No cross-machine bitwise guarantee or GPU tolerance is established.'])
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('CPU reference fixtures verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-build', type=Path, default=ROOT / 'build/llm-cpu-reference')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/tinyllama-gguf/tinyllama-q4_0.gguf')
    args = parser.parse_args()
    build(args.output.resolve(), args.cpu_build.resolve(), args.model.resolve())
