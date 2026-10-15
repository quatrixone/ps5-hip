"""Profile pinned TinyLlama metadata and safetensors headers; never fetch tensor data."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import urllib.request

MODEL = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
REVISION = 'fe8a4ea1ffedaf415f4da2f062534de366a451e6'
LIMIT = 8 * 1024 * 1024
FILES = ('config.json', 'generation_config.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model', 'README.md')


def bounded_read(response, limit, byte_range=None):
    """Validate headers before reading, especially when a weight server ignores Range."""
    total = None
    if response.headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('encoded response rejected')
    if byte_range is not None:
        start, end = byte_range
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
        if response.status != 206 or not match:
            raise ValueError('weight server ignored or malformed Range; body not read')
        first, last, total = map(int, match.groups())
        if (first, last) != (start, end) or total <= end or end - start + 1 > limit:
            raise ValueError('unexpected weight range; body not read')
        expected = end - start + 1
    else:
        if response.status != 200:
            raise ValueError('unexpected metadata status')
        expected = None
    length = response.headers.get('Content-Length')
    if length is not None and (not length.isdigit() or int(length) > limit or
                              (expected is not None and int(length) != expected)):
        raise ValueError('oversized or inconsistent response; body not read')
    data = response.read(limit + 1)
    if len(data) > limit or (expected is not None and len(data) != expected) or (length is not None and len(data) != int(length)):
        raise ValueError('oversized or truncated response')
    return data, total


def fetch(url, limit=LIMIT, byte_range=None):
    headers = {'Accept-Encoding': 'identity', 'User-Agent': 'ps5-hip-model-profile/1'}
    if byte_range is not None:
        headers['Range'] = 'bytes=%d-%d' % byte_range
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
        return bounded_read(response, limit, byte_range)


def parse_lfs_pointer(data):
    match = re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([1-9][0-9]*)\n\n?', data)
    if not match:
        raise ValueError('invalid bounded Git LFS pointer')
    return dict(sha256=match[1].decode('ascii'), size=int(match[2]))


def inventory(header, payload_bytes):
    widths = {'BF16': 2, 'F16': 2, 'F32': 4, 'F64': 8, 'I64': 8, 'I32': 4}
    result = []
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        shape = tensor['shape']
        if not shape or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError('invalid tensor shape')
        elements = math.prod(shape)
        start, end = tensor['data_offsets']
        if type(start) is not int or type(end) is not int or start < 0 or end > payload_bytes:
            raise ValueError('invalid tensor offsets')
        size = elements * widths[tensor['dtype']]
        if end - start != size:
            raise ValueError('tensor size differs from dtype and shape')
        result.append(dict(name=name, dtype=tensor['dtype'], shape=shape, elements=elements,
                           stored_bytes=size, data_offsets=[start, end]))
    cursor = 0
    for tensor in sorted(result, key=lambda row: row['data_offsets'][0]):
        if tensor['data_offsets'][0] != cursor:
            raise ValueError('tensor ranges overlap or leave gaps')
        cursor = tensor['data_offsets'][1]
    if not result or cursor != payload_bytes:
        raise ValueError('tensor ranges do not cover payload')
    return sorted(result, key=lambda row: row['name'])


def derive(config, tensors, context=512):
    heads, kv_heads = config['num_attention_heads'], config['num_key_value_heads']
    hidden, layers = config['hidden_size'], config['num_hidden_layers']
    if any(type(n) is not int or n <= 0 for n in (heads, kv_heads, hidden, layers, context)) or hidden % heads or heads % kv_heads:
        raise ValueError('invalid attention geometry')
    elements = sum(row['elements'] for row in tensors)
    # An explicit storage scenario, not a claim about any GGUF converter's choices.
    q4 = sum(row['elements'] // 32 * 18 if len(row['shape']) >= 2 and row['shape'][-1] % 32 == 0
             else row['elements'] * 4 for row in tensors)
    return dict(parameters=elements, tensor_count=len(tensors),
                stored_tensor_bytes=sum(row['stored_bytes'] for row in tensors),
                f16_weight_bytes=elements * 2,
                largest_tensors=[row['name'] for row in tensors if row['elements'] == max(t['elements'] for t in tensors)],
                largest_tensor_f16_bytes=max(row['elements'] for row in tensors) * 2,
                q4_0_estimate_bytes=q4,
                q4_0_assumption='Rank>=2 tensors with last dimension divisible by32 use18bytes/32elements; all other tensors FP32. Excludes GGUF metadata/alignment. Converter choices and actual GGUF unmeasured.',
                kv_cache=dict(context=context, batch=1, layers=layers, kv_heads=kv_heads,
                              head_dimension=hidden // heads, dtype='F16',
                              bytes=2 * layers * context * kv_heads * (hidden // heads) * 2),
                excludes='Activations, attention workspace, runtime/heap allocations, tokenizer, file mappings and allocator padding; no native model fit or execution claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-lock', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/'
    records, contents = {}, {}
    for name in FILES:
        data, _ = fetch(base + name)
        (args.output / name).write_bytes(data)
        contents[name] = data
        records[name] = dict(url=base + name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    prefix, total = fetch(base + 'model.safetensors', 8, (0, 7))
    pointer_url = f'https://huggingface.co/{MODEL}/raw/{REVISION}/model.safetensors'
    pointer, _ = fetch(pointer_url, 1024)
    lfs = parse_lfs_pointer(pointer)
    if lfs['size'] != total:
        raise ValueError('Git LFS size differs from range response')
    (args.output / 'model.safetensors.lfs').write_bytes(pointer)
    records['model.safetensors.lfs'] = dict(url=pointer_url, bytes=len(pointer), sha256=hashlib.sha256(pointer).hexdigest())
    length = struct.unpack('<Q', prefix)[0]
    if not 2 <= length <= LIMIT or total < 8 + length:
        raise ValueError('invalid bounded safetensors header length')
    header, second_total = fetch(base + 'model.safetensors', length, (8, 7 + length))
    if total != second_total:
        raise ValueError('weight file size changed between ranges')
    (args.output / 'safetensors-header.json').write_bytes(header)
    tensors = inventory(json.loads(header), total - 8 - length)
    config = json.loads(contents['config.json'])
    tokenizer = json.loads(contents['tokenizer.json'])
    profile = dict(schema=1, model=MODEL, revision=REVISION, metadata=records,
                   weights=dict(url=base + 'model.safetensors', file_bytes=total,
                                expected_file_sha256=lfs['sha256'], file_hash_verified=False,
                                header_bytes=length, header_sha256=hashlib.sha256(header).hexdigest(),
                                payload_downloaded=False),
                   config=config, tokenizer=dict(type=tokenizer['model']['type'],
                                                  vocabulary_entries=len(tokenizer['model']['vocab']),
                                                  added_tokens=tokenizer.get('added_tokens', [])),
                   estimates=derive(config, tensors), tensors=tensors)
    if args.check_lock and profile != json.loads(args.check_lock.read_text()):
        raise ValueError('profile differs from locked metadata')
    (args.output / 'profile.json').write_text(json.dumps(profile, indent=2) + '\n')
    print(json.dumps(profile['estimates'], indent=2))


if __name__ == '__main__':
    main()
