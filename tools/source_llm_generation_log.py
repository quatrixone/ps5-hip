"""Strict helper transcript checks against an authenticated CPU generation policy.

This checks controls and timing only. It cannot establish GPU origin or numerics.
"""
import re
from source_llm_full_log import application_frame, memory_snapshot, check_memory_lifecycle

APP_COMPLETE = 'M4_GEN APP_COMPLETE profile=generation watchdog_joined=true numerical=pending finalization=pending inference=false'


def _check_fragment(raw, policies, metadata_sizes, native=False, corpus=False):
    if not isinstance(raw, bytes) or not raw or len(raw) > (2 * 1024 * 1024 if corpus else 65536) or b'\0' in raw or b'\r' in raw:
        raise ValueError('generation transcript encoding/bound')
    text = raw.decode('ascii')
    if not text.endswith('\n'):
        raise ValueError('generation transcript framing')
    lines = iter(text.splitlines())
    previous = 0
    timings = []
    memory = []

    def snapshot(phase):
        if native:
            memory.append(memory_snapshot(next(lines, ''), phase))

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

    tag = ''

    def phase(scope, run=0):
        row = match(r'M4_GEN TIMING' + tag + r' scope=' + scope + r' run=' + str(run) + r' position=0 begin_ns=(\d+) end_ns=(\d+) elapsed_ns=(\d+)')
        begin, end, elapsed = map(int, row.groups())
        interval(begin, end, elapsed)
        timings.append(dict(scope=scope, run=run, elapsed_ns=elapsed, **({'case': case_id} if tag else {})))

    exact('M4_GEN PROFILE name=greedy-corpus cases=5 runs=2 max_new_tokens=16 context=512' if corpus else 'M4_GEN PROFILE name=greedy-short runs=2 max_new_tokens=4 context=512')
    exact('M3_FULL PHASE load_begin')
    snapshot('before_load')
    phase('load')
    exact('M3_FULL PHASE load_complete')
    snapshot('after_load')
    phase('context')
    exact('M3_FULL PHASE context_complete')
    snapshot('after_context')
    requests = 0
    for case_id, metadata in policies:
        tag = ' case=' + case_id if corpus else ''
        prefix = case_id + '/' if corpus else ''
        for run, policy in enumerate(metadata['runs']):
            phase('reset', run)
            reset_end = previous
            decode_sum = 0
            first_decode = None
            for position, token in enumerate(policy['processed']):
                for stage in [*(f'layer-{i}' for i in range(22)), 'norm', 'logits']:
                    name = prefix + f'run-{run}-pos-{position}-{stage}.f32le'
                    size = 128000 if stage == 'logits' else 8192
                    exact(f'M3_FULL RAW file={name} bytes={size} synced=true')
                row = match(r'M4_GEN DECODE' + tag + r' run=' + str(run) + r' position=' + str(position) + r' token=' + str(token) + r' committed_position=' + str(position) + r' layers=22 logits=32000 begin_ns=(\d+) end_ns=(\d+) elapsed_ns=(\d+) readback_ns=(\d+) inclusive_readback=true')
                begin, end, elapsed, readback = map(int, row.groups())
                interval(begin, end, elapsed)
                if not 0 < readback <= elapsed:
                    raise ValueError('generation readback interval')
                if first_decode is None:
                    first_decode = begin
                decode_sum += elapsed
                timings.append(dict(scope='decode', run=run, position=position, elapsed_ns=elapsed, readback_ns=readback, **({'case': case_id} if corpus else {})))
                requests += 1
            row = match(r'M4_GEN RUN_TIME' + tag + r' run=' + str(run) + r' begin_ns=(\d+) end_ns=(\d+) elapsed_ns=(\d+) decode_ns=(\d+) scheduling_sampling_log_ns=(\d+)')
            begin, end, elapsed, decoded, overhead = map(int, row.groups())
            if first_decode is None or not reset_end <= begin <= first_decode or end < previous or end >= 2**64 or end - begin != elapsed or decoded != decode_sum or decoded + overhead != elapsed:
                raise ValueError('generation run timing')
            previous = end
            exact(f"M4_GEN RESULT{tag} run={run} processed={len(policy['processed'])} generated={len(policy['generated'])} stop={policy['stop']} text_hex={policy['text_hex']}")
        generation_bytes, steps_bytes = (metadata_sizes[prefix + k] for k in ('generation.json', 'steps.jsonl'))
        exact(f'M4_GEN METADATA{tag} files=2 generation_bytes={generation_bytes} steps_bytes={steps_bytes} synced=true')
    tag = ''
    if corpus:
        exact(f'M4_GEN CORPUS cases=5 runs=2 max_new_tokens=16 index_bytes={metadata_sizes["corpus.json"]} synced=true')
    exact('M3_FULL PHASE cleanup_begin')
    snapshot('after_cleanup')
    phase('cleanup')
    row = match(r'M4_GEN TRANSPORT_COMPLETE requests=' + str(requests) + r' layers=22 logits=32000 nodes=' + str(requests * 798) + r' launches=' + str(requests * 733) + r' raw_files=' + str(requests * 24) + r' raw_bytes=' + str(requests * 316416) + r' metadata_files=' + str(11 if corpus else 2) + r' metadata_bytes=' + str(sum(metadata_sizes.values())) + r' instances=([1-9]\d{0,9})/([1-9]\d{0,9}) numerical=pending finalization=pending')
    if row[1] != row[2] or (corpus and row[1] != '1') or next(lines, None) is not None:
        raise ValueError('generation completion/ownership')
    if native:
        check_memory_lifecycle(memory)
    return dict(status='PASS', classification='generation-corpus-helper-transcript-only' if corpus else 'generation-helper-transcript-only', requests=requests, raw_files=requests * 24, metadata_files=11 if corpus else 2, metadata_bytes=sum(metadata_sizes.values()), raw_bytes=requests*316416, nodes=requests*798, gpu_submits=requests*733, instances=int(row[1]), memory=memory, timings=timings, numerical_verified=False, gpu_origin_established=False, finalization_verified=False)


def check_fragment(raw, metadata, metadata_sizes, native=False):
    if metadata.get('max_new_tokens') != 4 or len(metadata.get('runs', [])) != 2:
        raise ValueError('generation reference policy')
    if set(metadata_sizes) != {'generation.json', 'steps.jsonl'} or any(type(n) is not int or not 0 < n <= 65536 for n in metadata_sizes.values()):
        raise ValueError('generation metadata sizes')
    return _check_fragment(raw, [(None, metadata)], metadata_sizes, native)


def check_corpus_fragment(raw, policies_by_id, metadata_sizes_flat, native=False):
    from build_llm_generation_reference import corpus_cases
    cases = corpus_cases()
    if not isinstance(policies_by_id, dict) or set(policies_by_id) != {key for key, _ in cases}:
        raise ValueError('corpus policy cases')
    expected_sizes = {'corpus.json'} | {key + '/' + name for key, _ in cases for name in ('generation.json', 'steps.jsonl')}
    if set(metadata_sizes_flat) != expected_sizes or any(type(n) is not int or not 0 < n <= 65536 for n in metadata_sizes_flat.values()):
        raise ValueError('corpus metadata sizes')
    for key, prompt in cases:
        policy = policies_by_id[key]
        if not isinstance(policy, dict) or policy.get('prompt') != prompt or type(policy.get('max_new_tokens')) is not int or policy['max_new_tokens'] != 16:
            raise ValueError('corpus reference policy')
        tokens, eog, runs = (policy.get(name) for name in ('prompt_tokens', 'eog_tokens', 'runs'))
        valid = lambda token: type(token) is int and 0 <= token < 32000
        if not isinstance(tokens, list) or not 1 <= len(tokens) <= 32 or not all(map(valid, tokens)) or not isinstance(eog, list) or not all(map(valid, eog)) or not isinstance(runs, list) or len(runs) != 2:
            raise ValueError('corpus reference token policy')
        for run, row in enumerate(runs):
            if not isinstance(row, dict) or type(row.get('run')) is not int or row['run'] != run:
                raise ValueError('corpus reference run')
            generated = row.get('generated')
            if not isinstance(generated, list) or len(generated) != 16 or not all(map(valid, generated)) or any(token in eog for token in generated) or row.get('stop') != 'step_limit' or row.get('processed') != tokens + generated[:-1]:
                raise ValueError('corpus reference feedback/EOS')
            if not isinstance(row.get('text_hex'), str) or re.fullmatch(r'(?:[0-9a-f]{2})*', row['text_hex']) is None:
                raise ValueError('corpus reference text')
        if any(runs[0][name] != runs[1][name] for name in ('processed', 'generated', 'stop', 'text_hex')):
            raise ValueError('corpus reset repeat')
    return _check_fragment(raw, [(key, policies_by_id[key]) for key, _ in cases], metadata_sizes_flat, native, corpus=True)


def check_full_log(raw, metadata, metadata_sizes):
    if sum(len(row['processed']) for row in metadata['runs']) != 16:
        raise ValueError('native generation request profile')
    fragment, lifecycle = application_frame(raw, APP_COMPLETE)
    result = check_fragment(fragment, metadata, metadata_sizes, native=True)
    result.update(phase='model-full', profile='generation', classification='native-generation-transcript-only', **lifecycle)
    return result


def check_corpus_full_log(raw, policies_by_id, metadata_sizes_flat):
    if sum(len(row['processed']) for policy in policies_by_id.values() for row in policy['runs']) != 272:
        raise ValueError('native corpus request profile')
    complete = 'M4_GEN APP_COMPLETE profile=corpus watchdog_joined=true numerical=pending finalization=pending inference=false'
    fragment, lifecycle = application_frame(raw, complete)
    result = check_corpus_fragment(fragment, policies_by_id, metadata_sizes_flat, native=True)
    result.update(phase='model-full', profile='corpus', classification='native-corpus-transcript-only', **lifecycle)
    return result
