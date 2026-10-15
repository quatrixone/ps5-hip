"""Convert verified TinyLlama sources to F16 then pure Q4_0 using locked upstream tools."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from fetch_llm_model import verify

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / '.deps/sources/llama-client'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(reader, expected, allowed, config):
    metadata = {'general.architecture': 'llama', 'llama.context_length': config['max_position_embeddings'],
                'llama.embedding_length': config['hidden_size'], 'llama.block_count': config['num_hidden_layers'],
                'llama.feed_forward_length': config['intermediate_size'],
                'llama.attention.head_count': config['num_attention_heads'],
                'llama.attention.head_count_kv': config['num_key_value_heads']}
    for name, value in metadata.items():
        if name not in reader.fields or reader.fields[name].contents() != value:
            raise ValueError(f'converted GGUF metadata differs from profile: {name}')
    tensors = [dict(name=t.name, dtype=t.tensor_type.name, shape=[int(x) for x in t.shape],
                    elements=int(t.n_elements), bytes=int(t.n_bytes)) for t in reader.tensors]
    if (len(tensors) != expected['tensor_count'] or len({t['name'] for t in tensors}) != len(tensors)
            or sum(t['elements'] for t in tensors) != expected['parameters']
            or set(t['dtype'] for t in tensors) - set(allowed)
            or allowed[0] not in {t['dtype'] for t in tensors}):
        raise ValueError('converted tensor inventory differs from bounded model profile')
    return dict(tensors=tensors, dtype_counts=dict(Counter(t['dtype'] for t in tensors)),
                tensor_bytes=sum(t['bytes'] for t in tensors), metadata=metadata)


def convert(source, output, cpu_build, environment):
    if output.exists():
        raise ValueError('conversion output must be a fresh directory; preserve prior outputs')
    if Path(sys.prefix).resolve() != (environment / 'venv').resolve():
        raise ValueError('run with the prepared conversion environment Python')
    locks = [ROOT / 'llm-inputs.json', ROOT / 'llm-conversion-inputs.json',
             ROOT / 'models/tinyllama-1.1b-chat-v1.0.json', Path(__file__),
             ROOT / 'tools/fetch_llm_model.py',
             environment / 'receipt.json', cpu_build / 'receipt.json']
    identities = {str(p): sha(p) for p in locks}
    client_lock, conversion_lock, model = [json.loads(p.read_text()) for p in locks[:3]]
    revision = client_lock['client']['revision']

    def check_client():
        head = subprocess.check_output(['git', '-C', str(CLIENT), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(CLIENT), 'status', '--porcelain'], text=True).strip()
        if head != revision or dirty or conversion_lock['client_revision'] != revision:
            raise ValueError('converter source must match the clean locked client')

    check_client()
    env_receipt = json.loads((environment / 'receipt.json').read_text())
    if env_receipt['lock_sha256'] != sha(ROOT / 'llm-conversion-inputs.json'):
        raise ValueError('converter environment does not match lock')
    import importlib.metadata
    for name, version in conversion_lock['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'converter package differs from lock: {name}')
    build_receipt = json.loads((cpu_build / 'receipt.json').read_text())
    if build_receipt['source_revision'] != revision:
        raise ValueError('quantizer build differs from locked client')
    def check_cpu_artifacts():
        for path, digest in build_receipt['artifacts'].items():
            if sha(cpu_build / path) != digest:
                raise ValueError(f'CPU artifact changed: {path}')
    check_cpu_artifacts()
    for name, spec in model['metadata'].items():
        verify(source / name, spec['bytes'], spec['sha256'])
    verify(source / 'model.safetensors', model['weights']['file_bytes'], model['weights']['expected_file_sha256'])
    output.mkdir(parents=True)
    commands = []
    process_env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4')
    process_env.pop('NO_LOCAL_GGUF', None)
    process_env.pop('PYTHONPATH', None)
    process_env.pop('LD_LIBRARY_PATH', None)
    process_env.pop('LD_PRELOAD', None)

    def run(label, argv):
        record = dict(label=label, argv=[str(a) for a in argv])
        commands.append(record)
        try:
            with (output / (label + '.stdout')).open('w') as stdout, (output / (label + '.stderr')).open('w') as stderr:
                result = subprocess.run(record['argv'], stdout=stdout, stderr=stderr, env=process_env, timeout=1800)
            record['exit_code'] = result.returncode
            result.check_returncode()
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            raise
        finally:
            (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')

    f16, q4 = output / 'tinyllama-f16.gguf', output / 'tinyllama-q4_0.gguf'
    run('convert-f16', [sys.executable, CLIENT / 'convert_hf_to_gguf.py', source,
                        '--outtype', 'f16', '--outfile', f16])
    run('quantize-q4_0', [cpu_build / 'cmake/bin/llama-quantize', '--pure', f16, q4, 'Q4_0', '4'])
    sys.path.insert(0, str(CLIENT / 'gguf-py'))
    from gguf import GGUFReader
    outputs = {}
    for path, allowed in ((f16, ('F16', 'F32')), (q4, ('Q4_0', 'F32'))):
        outputs[path.name] = dict(bytes=path.stat().st_size, sha256=sha(path),
                                 **inventory(GGUFReader(path), model['estimates'], allowed, model['config']))
    check_client()
    check_cpu_artifacts()
    if any(sha(Path(path)) != digest for path, digest in identities.items()):
        raise ValueError('conversion tool or input identity changed')
    for name, spec in model['metadata'].items():
        verify(source / name, spec['bytes'], spec['sha256'])
    verify(source / 'model.safetensors', model['weights']['file_bytes'], model['weights']['expected_file_sha256'])
    (output / 'receipt.json').write_text(json.dumps(dict(schema=1, classification='host-model-conversion-verified',
        identities=identities, source_revision=revision, model_revision=model['revision'],
        commands=commands, outputs=outputs, model_inference_executed=False, native_executed=False), indent=2) + '\n')
    print('F16 and pure Q4_0 conversion verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'models/tinyllama-source')
    parser.add_argument('--output', type=Path, default=ROOT / 'models/tinyllama-gguf')
    parser.add_argument('--cpu-build', type=Path, default=ROOT / 'build/llm-cpu-reference')
    parser.add_argument('--environment', type=Path, default=ROOT / '.deps/llm-conversion-env')
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve(), args.cpu_build.resolve(), args.environment.resolve())
