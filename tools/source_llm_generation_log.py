"""Strict helper transcript checks against an authenticated CPU generation policy.

This checks controls and timing only. It cannot establish GPU origin or numerics.
"""
import re


def check_fragment(raw, metadata, metadata_sizes):
    if not isinstance(raw, bytes) or not raw or len(raw) > 65536 or b'\0' in raw or b'\r' in raw:
        raise ValueError('generation transcript encoding/bound')
    text = raw.decode('ascii')
    if not text.endswith('\n'):
        raise ValueError('generation transcript framing')
    lines = iter(text.splitlines())
    previous = 0
    timings = []

    def exact(value):
        if next(lines, None) != value:
            raise ValueError('generation transcript expected ' + value)

    def match(pattern):
        result = re.fullmatch(pattern, next(lines, ''))
        if result is None:
            raise ValueError('generation transcript fields')
        return result

    def interval(begin, end, elapsed):
        nonlocal previous
        if not 0 < begin < end < 2**64 or end - begin != elapsed or begin < previous:
            raise ValueError('generation clock/interval')
        previous = end

    def phase(scope, run=0):
        row = match(r'M4_GEN TIMING scope=' + scope + r' run=' + str(run) + r' position=0 begin_ns=(\d+) end_ns=(\d+) elapsed_ns=(\d+)')
        begin, end, elapsed = map(int, row.groups())
        interval(begin, end, elapsed)
        timings.append(dict(scope=scope, run=run, elapsed_ns=elapsed))

    if metadata.get('max_new_tokens') != 4 or len(metadata.get('runs', [])) != 2:
        raise ValueError('generation reference policy')
    if set(metadata_sizes) != {'generation.json', 'steps.jsonl'} or any(type(n) is not int or not 0 < n <= 65536 for n in metadata_sizes.values()):
        raise ValueError('generation metadata sizes')
    exact('M4_GEN PROFILE name=greedy-short runs=2 max_new_tokens=4 context=512')
    exact('M3_FULL PHASE load_begin')
    phase('load')
    exact('M3_FULL PHASE load_complete')
    phase('context')
    exact('M3_FULL PHASE context_complete')
    requests = 0
    for run, policy in enumerate(metadata['runs']):
        phase('reset', run)
        reset_end = previous
        decode_sum = 0
        first_decode = None
        for position, token in enumerate(policy['processed']):
            for stage in [*(f'layer-{i}' for i in range(22)), 'norm', 'logits']:
                name = f'run-{run}-pos-{position}-{stage}.f32le'
                size = 128000 if stage == 'logits' else 8192
                exact(f'M3_FULL RAW file={name} bytes={size} synced=true')
            row = match(r'M4_GEN DECODE run=' + str(run) + r' position=' + str(position) + r' token=' + str(token) + r' committed_position=' + str(position) + r' layers=22 logits=32000 begin_ns=(\d+) end_ns=(\d+) elapsed_ns=(\d+) readback_ns=(\d+) inclusive_readback=true')
            begin, end, elapsed, readback = map(int, row.groups())
            interval(begin, end, elapsed)
            if not 0 < readback <= elapsed:
                raise ValueError('generation readback interval')
            if first_decode is None:
                first_decode = begin
            decode_sum += elapsed
            timings.append(dict(scope='decode', run=run, position=position, elapsed_ns=elapsed, readback_ns=readback))
            requests += 1
        row = match(r'M4_GEN RUN_TIME run=' + str(run) + r' begin_ns=(\d+) end_ns=(\d+) elapsed_ns=(\d+) decode_ns=(\d+) scheduling_sampling_log_ns=(\d+)')
        begin, end, elapsed, decoded, overhead = map(int, row.groups())
        if first_decode is None or not reset_end <= begin <= first_decode or end < previous or end >= 2**64 or end - begin != elapsed or decoded != decode_sum or decoded + overhead != elapsed:
            raise ValueError('generation run timing')
        previous = end
        exact(f"M4_GEN RESULT run={run} processed={len(policy['processed'])} generated={len(policy['generated'])} stop={policy['stop']} text_hex={policy['text_hex']}")
    generation_bytes, steps_bytes = (metadata_sizes[k] for k in ('generation.json', 'steps.jsonl'))
    exact(f'M4_GEN METADATA files=2 generation_bytes={generation_bytes} steps_bytes={steps_bytes} synced=true')
    exact('M3_FULL PHASE cleanup_begin')
    phase('cleanup')
    row = match(r'M4_GEN TRANSPORT_COMPLETE requests=' + str(requests) + r' layers=22 logits=32000 nodes=' + str(requests * 798) + r' launches=' + str(requests * 733) + r' raw_files=' + str(requests * 24) + r' raw_bytes=' + str(requests * 316416) + r' metadata_files=2 metadata_bytes=' + str(generation_bytes + steps_bytes) + r' instances=([1-9]\d{0,9})/([1-9]\d{0,9}) numerical=pending finalization=pending')
    if row[1] != row[2] or next(lines, None) is not None:
        raise ValueError('generation completion/ownership')
    return dict(status='PASS', classification='generation-helper-transcript-only', requests=requests, raw_files=requests * 24, metadata_files=2, timings=timings, numerical_verified=False, gpu_origin_established=False, finalization_verified=False)
