"""Strict public HIP bootstrap transcript; freshness and lifecycle are separate."""
import hashlib
import math
import re

LINES = (
    'PS5HIP START target=gfx1013 kernel=public_hip_bootstrap',
    'PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass',
    'HIP_BOOTSTRAP PASS init=0 count_error=0 count=1 watchdog_joined=true finalization=pending',
    'PS5HIP SHUTDOWN main-return=0',
    'PS5HIP SHUTDOWN atexit=observed',
    'PS5HIP SHUTDOWN static-dtor=observed',
    'HIP_BOOTSTRAP FINAL allocations=0 frees=0 queues=0 retired=0 gate_destroyed=true application_fini=pending',
    'PS5HIP SHELL_EXIT application-fini=completed',
    'PS5HIP SHELL_EXIT request=begin',
    'PS5HIP SHELL_EXIT request-return=0',
)
EXPECTED = ('\n'.join(LINES) + '\n').encode('ascii')
MEMORY_LINES = list(LINES)
MEMORY_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_memory'
MEMORY_LINES[2] = 'HIP_MEMORY PASS init=0 count=1 allocations=2 frees=2 interior_rejected=true double_free_rejected=true watchdog_joined=true finalization=pending'
MEMORY_LINES[6] = 'HIP_MEMORY FINAL allocations=2 frees=2 queues=0 retired=0 gate_destroyed=true application_fini=pending'
MEMORY_EXPECTED = ('\n'.join(MEMORY_LINES) + '\n').encode('ascii')
COPY_LINES = list(LINES)
COPY_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_copy'
COPY_LINES[2] = 'HIP_COPY PASS init=0 count=1 rounds=2 compared_bytes=7168 bounds_rejected=16 rectangular=true padding=true gpu_execution=false lifetime_cases=2 lifetime_bytes=384 gated_free_observed=true migration=true allocations=8 frees=8 watchdog_joined=true finalization=pending'
COPY_LINES[6] = 'HIP_COPY FINAL allocations=8 frees=8 queues=4 retired=4 gate_destroyed=true application_fini=pending'
COPY_EXPECTED = ('\n'.join(COPY_LINES) + '\n').encode('ascii')
MODULE_LINES = list(LINES)
MODULE_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_module'
MODULE_LINES[2] = 'HIP_MODULE PASS init=0 count=1 loads=4 lookups=8 unloads=4 isolated=true gpu_execution=false watchdog_joined=true finalization=pending'
MODULE_LINES[6] = 'HIP_MODULE FINAL allocations=0 frees=0 queues=0 retired=0 gate_destroyed=true application_fini=pending'
MODULE_EXPECTED = ('\n'.join(MODULE_LINES) + '\n').encode('ascii')
LAUNCH_LINES = list(LINES)
LAUNCH_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_launch'
LAUNCH_LINES[2] = 'HIP_LAUNCH PASS init=0 count=1 rounds=2 launches=2 values=146 guards=16 rejected=2 allocations=1 frees=1 watchdog_joined=true finalization=pending'
LAUNCH_LINES[6] = 'HIP_LAUNCH FINAL allocations=1 frees=1 queues=1 retired=1 gate_destroyed=true application_fini=pending'
LAUNCH_EXPECTED = ('\n'.join(LAUNCH_LINES) + '\n').encode('ascii')
EMBEDDED_EXPECTED = LAUNCH_EXPECTED.replace(b'public_hip_launch', b'public_hip_embedded').replace(b'HIP_LAUNCH', b'HIP_EMBEDDED').replace(b'rejected=2', b'compiler_stubs=true')
AGGREGATE_EXPECTED = EMBEDDED_EXPECTED.replace(b'public_hip_embedded', b'public_hip_aggregate').replace(b'HIP_EMBEDDED', b'HIP_AGGREGATE').replace(b'values=146 guards=16', b'member_bytes=320 guard_bytes=64 padding=unspecified')
FILL_LINES = list(LINES)
FILL_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_fill'
FILL_LINES[2] = 'HIP_FILL PASS init=0 count=1 rounds=2 compared_bytes=4096 rejected=16 pitched=true padding=true allocations=1 frees=1 gpu_execution=false watchdog_joined=true finalization=pending'
FILL_LINES[6] = 'HIP_FILL FINAL allocations=1 frees=1 queues=1 retired=1 gate_destroyed=true application_fini=pending'
FILL_EXPECTED = ('\n'.join(FILL_LINES) + '\n').encode('ascii')
GRAPH_LINES = list(LINES)
GRAPH_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_graph'
GRAPH_LINES[2] = 'HIP_GRAPH PASS init=0 count=1 graphs=2 replays=32 callbacks=64 invalidations=6 external_graphs=3 updates=true source_destroyed=true gpu_execution=false watchdog_joined=true finalization=pending'
GRAPH_LINES[6] = 'HIP_GRAPH FINAL allocations=0 frees=0 queues=5 retired=5 gate_destroyed=true application_fini=pending'
GRAPH_EXPECTED = ('\n'.join(GRAPH_LINES) + '\n').encode('ascii')
STREAM_LINES = list(LINES)
STREAM_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_stream'
STREAM_LINES[2] = 'HIP_STREAM PASS init=0 count=1 rounds=2 launches=2 values=146 guards=16 source_preserved=true event_reused=true allocations=2 frees=2 watchdog_joined=true finalization=pending'
STREAM_LINES[6] = 'HIP_STREAM FINAL allocations=2 frees=2 queues=3 retired=3 gate_destroyed=true application_fini=pending'
STREAM_EXPECTED = ('\n'.join(STREAM_LINES) + '\n').encode('ascii')
GRAPH_GPU_LINES = list(LINES)
GRAPH_GPU_LINES[0] = 'PS5HIP START target=gfx1013 kernel=public_hip_graph_gpu'
GRAPH_GPU_LINES[2] = 'HIP_GRAPH_GPU PASS init=0 count=1 rounds=2 launches=2 values=146 guards=16 owned_args=true updated=true source_destroyed=true disabled_replays=2 callbacks=4 allocations=1 frees=1 watchdog_joined=true finalization=pending'
GRAPH_GPU_LINES[6] = 'HIP_GRAPH_GPU FINAL allocations=1 frees=1 queues=3 retired=3 gate_destroyed=true application_fini=pending'
GRAPH_GPU_EXPECTED = ('\n'.join(GRAPH_GPU_LINES) + '\n').encode('ascii')
CAPTURE_GPU_EXPECTED = GRAPH_GPU_EXPECTED.replace(b'public_hip_graph_gpu', b'public_hip_capture_gpu').replace(b'HIP_GRAPH_GPU', b'HIP_CAPTURE_GPU').replace(b'owned_args=true', b'captured=true owned_args=true')
COPY_GPU_EXPECTED = CAPTURE_GPU_EXPECTED.replace(b'public_hip_capture_gpu', b'public_hip_copy_gpu').replace(b'HIP_CAPTURE_GPU', b'HIP_COPY_GPU').replace(b'launches=2 values=146 guards=16', b'launches=84 values=292 guards=32 recovery_cases=27 setter_rejections=4 update_rejections=2 joined_streams=2 rejoined=true events_reused=true eager_equal=true').replace(b'disabled_replays=2 callbacks=4', b'disabled_replays=6 callbacks=28').replace(b'allocations=1 frees=1', b'allocations=4 frees=4').replace(b'queues=3 retired=3', b'queues=7 retired=7').replace(b'watchdog_joined=true', b'replay_rounds=20 warmup_rounds=4 verification_rounds=16 kernels_per_replay=2 replay_values=5840 replay_guards=640 alternating_order=true watchdog_joined=true')
COPY_GPU_BACKING = b'HIP_COPY_GPU BACKING samples=32 live_bytes=32768 live_allocations=2 eager_allocations=128 eager_releases=128 graph_allocations=128 graph_releases=128 peak_bytes=98304 final_live_bytes=0\n'
COPY_GPU_EXPECTED = COPY_GPU_EXPECTED.replace(b'HIP_COPY_GPU PASS ', COPY_GPU_BACKING + b'HIP_COPY_GPU PASS ')
LOG_NAMES = {fixture: 'ps5-hip-public-' + fixture + '.log' for fixture in ('bootstrap', 'memory', 'copy', 'module', 'launch', 'embedded', 'aggregate', 'fill', 'graph', 'stream', 'graph_gpu', 'capture_gpu', 'copy_gpu')}
LOG_NAMES['rmsnorm'] = 'ps5-hip-public-rmsnorm.log'
LOG_NAMES['q4'] = 'ps5-hip-public-q4.log'


def check_rmsnorm_lines(raw):
    lines = raw.decode('ascii').splitlines()
    if raw != ('\n'.join(lines) + '\n').encode('ascii') or len(lines) != 22:
        raise ValueError('RMSNorm transcript framing')
    expected = [LINES[0].replace('bootstrap', 'rmsnorm'), LINES[1]]
    errors, fractions = [], []
    number = r'([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)'
    for family in ('stress', 'embedding'):
        for rows in (1, 2, 17):
            expected.append(f'HIP_RMSNORM DUMP family={family} rows={rows} output_bytes=139296 input_readback_bytes=139296 exclusive=true synced=true')
            pattern = (f'HIP_RMSNORM CASE family={family} rows={rows} values={rows*2048} '
                       + 'max_abs=' + number + ' max_budget_fraction=' + number
                       + ' input_unchanged=true guards=true inactive=true')
            line = lines[len(expected)]
            match = re.fullmatch(pattern, line)
            if not match:
                raise ValueError('RMSNorm case transcript')
            error, fraction = map(float, match.groups())
            if not math.isfinite(error) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError('RMSNorm reported numerical error')
            errors.append(error); fractions.append(fraction); expected.append(line)
    pattern = ('HIP_RMSNORM PASS cases=6 launches=6 values=81920 guard_bytes=192 inactive_bytes=507904 input_bytes=835776 max_abs='
               + number + ' max_budget_fraction=' + number
               + ' allocations=2 frees=2 module_unloaded=true watchdog_joined=true finalization=pending')
    match = re.fullmatch(pattern, lines[14])
    if not match or tuple(map(float, match.groups())) != (max(errors), max(fractions)):
        raise ValueError('RMSNorm aggregate metrics differ')
    expected.append(lines[14])
    expected.extend(LINES[3:6])
    expected.append('HIP_RMSNORM FINAL allocations=2 frees=2 queues=1 retired=1 gate_destroyed=true application_fini=pending')
    expected.extend(LINES[7:])
    if lines != expected:
        raise ValueError('RMSNorm lifecycle or dump transcript differs')
    return {'max_abs': max(errors), 'max_budget_fraction': max(fractions)}



def check_q4_lines(raw):
    from verify_public_hip_q4 import STAGES
    lines=raw.decode('ascii').splitlines()
    if raw!=('\n'.join(lines)+'\n').encode('ascii') or len(lines)!=66:
        raise ValueError('Q4 transcript framing')
    expected=[LINES[0].replace('bootstrap','q4'),LINES[1]]
    number=r'([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)'
    metrics=[]
    for name,stage,width,rows,padded in STAGES:
        expected.append(f'HIP_Q4 DUMP case={name} stage={stage} files=4 bytes=324928 exclusive=true synced=true')
        line=lines[len(expected)]
        pattern=(f'HIP_Q4 CASE case={name} stage={stage} rows={rows} values={0 if stage=="q8" else rows} q8_bytes={padded//32*36} '
                 +'max_abs='+number+' max_budget_fraction='+number+' inputs_unchanged=true guards=true inactive=true')
        match=re.fullmatch(pattern,line)
        if not match:raise ValueError('Q4 case transcript')
        error,fraction=map(float,match.groups())
        if not math.isfinite(error) or not math.isfinite(fraction) or not 0<=fraction<=1:
            raise ValueError('Q4 reported numerical error')
        if stage=='q8' and (error or fraction):raise ValueError('Q8 reports Q4 numerical metrics')
        metrics.append(dict(name=name,stage=stage,max_abs=error,max_budget_fraction=fraction))
        expected.append(line)
    maximum=max(m['max_abs'] for m in metrics);fraction=max(m['max_budget_fraction'] for m in metrics)
    line=lines[len(expected)]
    pattern=('HIP_Q4 PASS cases=12 stages=28 launches=28 values=592 q8_bytes=80640 raw_files=112 raw_bytes=9097984 max_abs='
             +number+' max_budget_fraction='+number+' allocations=4 frees=4 modules_unloaded=2 watchdog_joined=true finalization=pending')
    match=re.fullmatch(pattern,line)
    if not match or tuple(map(float,match.groups()))!=(maximum,fraction):raise ValueError('Q4 aggregate metrics differ')
    expected.extend([line,*LINES[3:6],'HIP_Q4 FINAL allocations=4 frees=4 queues=1 retired=1 gate_destroyed=true application_fini=pending',*LINES[7:]])
    if expected!=lines:raise ValueError('Q4 lifecycle or dump transcript differs')
    return dict(max_abs=maximum,max_budget_fraction=fraction,stages=metrics)


def check_log(raw, fixture='bootstrap'):
    if not isinstance(raw, bytes) or len(raw) > 65536:
        raise ValueError('public HIP transcript type or size')
    if fixture in ('rmsnorm', 'q4'):
        try:
            metrics = check_rmsnorm_lines(raw) if fixture == 'rmsnorm' else check_q4_lines(raw)
        except (UnicodeError, IndexError) as error:
            raise ValueError(fixture+' transcript invalid') from error
        return {'application_result': 'application-pass', 'kernel': 'public_hip_'+fixture,
                'target': 'gfx1013', 'log_sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'hardware_gate_passed': False, 'reported_metrics': metrics,
                'unproven': ['independent raw-buffer numerical validation', 'freshness',
                             'installed identity', 'native process lifecycle', 'visible mount state',
                             'service health', 'owned lock cleanup']}
    expected = {'bootstrap': EXPECTED, 'memory': MEMORY_EXPECTED, 'copy': COPY_EXPECTED, 'module': MODULE_EXPECTED, 'launch': LAUNCH_EXPECTED, 'embedded': EMBEDDED_EXPECTED, 'aggregate': AGGREGATE_EXPECTED, 'fill': FILL_EXPECTED, 'graph': GRAPH_EXPECTED, 'stream': STREAM_EXPECTED, 'graph_gpu': GRAPH_GPU_EXPECTED, 'capture_gpu': CAPTURE_GPU_EXPECTED, 'copy_gpu': COPY_GPU_EXPECTED}.get(fixture)
    if expected is None or not isinstance(raw, bytes) or len(raw) > 65536 or raw != expected:
        raise ValueError('public HIP transcript incomplete, changed, wrong fixture or contaminated')
    return {'application_result': 'application-pass', 'kernel': 'public_hip_' + fixture,
            'target': 'gfx1013', 'log_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'hip_init': 0, 'device_count': 1,
            'hardware_gate_passed': False,
            'unproven': ['freshness', 'installed identity', 'native process lifecycle',
                         'visible mount state', 'service health', 'owned lock cleanup']}
