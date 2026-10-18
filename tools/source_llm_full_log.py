"""Exact smoke-decoder transport contract; numerical and native origin are separate."""
import re
from build_llm_full_numeric_test import FILES
from source_public_hip_log import LLM_CLIENT_LINES, LINES

RAW_BYTES = sum(FILES.values())
APP_COMPLETE = 'M3_FULL APP_COMPLETE profile=smoke watchdog_joined=true numerical=pending finalization=pending inference=false'


def memory_snapshot(line, phase):
    match = re.fullmatch(r'M0_APP MEMORY phase=' + phase + r' pool_bytes=([1-9][0-9]{0,18}) largest_free_bytes=([1-9][0-9]{0,18}) total_free_bytes=unknown alignment=16384', line)
    if not match:
        raise ValueError('full memory snapshot')
    pool, largest = map(int, match.groups())
    if not 0 < largest <= pool < 2**63 or pool < 619094016 or largest % 16384:
        raise ValueError('full memory bounds')
    return dict(phase=phase, pool_bytes=pool, largest_free_bytes=largest)


def check_memory_lifecycle(memory):
    if len(memory) != 4 or len({m['pool_bytes'] for m in memory}) != 1 or memory[0]['largest_free_bytes'] != memory[3]['largest_free_bytes'] or memory[1]['largest_free_bytes'] >= memory[0]['largest_free_bytes'] or memory[2]['largest_free_bytes'] > memory[1]['largest_free_bytes']:
        raise ValueError('full memory lifecycle')


def check_fragment(raw):
    if not isinstance(raw, bytes) or not raw.endswith(b'\n') or b'\r' in raw or b'\0' in raw:
        raise ValueError('full fragment encoding')
    lines = iter(raw.decode('ascii').splitlines()); memory = []

    def exact(expected):
        if next(lines, None) != expected:
            raise ValueError('full transcript expected ' + expected)

    def snapshot(phase):
        memory.append(memory_snapshot(next(lines, ''), phase))

    exact('M3_FULL PROFILE name=smoke requests=3 raw_files=72 raw_bytes=949248')
    exact('M3_FULL PHASE load_begin'); snapshot('before_load')
    exact('M3_FULL PHASE load_complete'); snapshot('after_load')
    exact('M3_FULL PHASE context_complete'); snapshot('after_context')
    for run, position in ((0, 0), (0, 1), (1, 0)):
        for stage in [*(f'layer-{i}' for i in range(22)), 'norm', 'logits']:
            name = f'run-{run}-pos-{position}-{stage}.f32le'
            exact(f'M3_FULL RAW file={name} bytes={FILES[name]} synced=true')
        exact(f'M3_FULL CHECKPOINT run={run} position={position} token={position} committed_position={position} layers=22 logits=32000')
    exact('M3_FULL PHASE cleanup_begin'); snapshot('after_cleanup')
    match = re.fullmatch(r'M3_FULL TRANSPORT_COMPLETE requests=3 layers=22 logits=32000 nodes=2394 launches=2199 raw_files=72 raw_bytes=949248 instances=([1-9][0-9]{0,9})/([1-9][0-9]{0,9}) numerical=pending inference=false finalization=pending', next(lines, ''))
    if not match or match[1] != match[2] or next(lines, None) is not None:
        raise ValueError('full transport completion')
    check_memory_lifecycle(memory)
    return dict(status='PASS', classification='full-fragment-transport-only', requests=3, layers=22, logits=32000, nodes=2394, gpu_submits=2199, raw_files=72, raw_bytes=RAW_BYTES, instances=int(match[1]), memory=memory, numerical_verified=False, gpu_origin_established=False, inference=False)


def application_frame(raw, app_complete):
    if not isinstance(raw, bytes) or not raw or len(raw) > (2 * 1024 * 1024 if app_complete == 'M4_GEN APP_COMPLETE profile=corpus watchdog_joined=true numerical=pending finalization=pending inference=false' else 65536):
        raise ValueError('full transcript bounds')
    lines = raw.decode('ascii').splitlines()
    if raw != ('\n'.join(lines) + '\n').encode() or b'\0' in raw or tuple(lines[:2]) != LLM_CLIENT_LINES[:2]:
        raise ValueError('full transcript framing/startup')
    footer = lines[-9:]
    host = re.fullmatch(r'M0_APP HOST_STORAGE allocations=([1-9][0-9]{0,19}) releases=([1-9][0-9]{0,19}) live_allocations=0 live_requested_bytes=0 peak_requested_bytes=([1-9][0-9]{0,19}) failures=0 poisoned=false scope=ggml_aligned', footer[0] if footer else '')
    final = re.fullmatch(r'M0_APP FINAL allocations=([1-9][0-9]{0,19}) frees=([1-9][0-9]{0,19}) queues=([1-9][0-9]{0,19}) retired=([1-9][0-9]{0,19}) gate_destroyed=true application_fini=pending', footer[5] if len(footer) == 9 else '')
    if not host or not final:
        raise ValueError('full ownership/finalization')
    counts = list(map(int, final.groups())); allocations, releases, peak = map(int, host.groups())
    if allocations != releases or counts[0] != counts[1] or counts[2] != counts[3] or max(*counts, allocations, releases, peak) > 2**64-1:
        raise ValueError('full ownership balance')
    if footer[1:5] != [app_complete, *LINES[3:6]] or footer[6:] != list(LINES[7:]):
        raise ValueError('full finalization')
    return ('\n'.join(lines[2:-9]) + '\n').encode(), dict(final_counts=counts, host_storage=dict(allocations=allocations, releases=releases, peak_requested_bytes=peak), finalization_transcript_complete=True)


def check_full_log(raw):
    fragment, lifecycle = application_frame(raw, APP_COMPLETE)
    result = check_fragment(fragment)
    result.update(phase='model-full', classification='native-full-transcript-only', **lifecycle)
    return result
