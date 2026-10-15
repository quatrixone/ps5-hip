"""Strict public HIP bootstrap transcript; freshness and lifecycle are separate."""
import hashlib

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
COPY_GPU_EXPECTED = CAPTURE_GPU_EXPECTED.replace(b'public_hip_capture_gpu', b'public_hip_copy_gpu').replace(b'HIP_CAPTURE_GPU', b'HIP_COPY_GPU').replace(b'launches=2 values=146 guards=16', b'launches=4 values=292 guards=32 recovery_cases=27 setter_rejections=4 update_rejections=2 joined_streams=2 rejoined=true events_reused=true eager_equal=true').replace(b'disabled_replays=2 callbacks=4', b'disabled_replays=6 callbacks=8').replace(b'allocations=1 frees=1', b'allocations=2 frees=2').replace(b'queues=3 retired=3', b'queues=4 retired=4')
LOG_NAMES = {fixture: 'ps5-hip-public-' + fixture + '.log' for fixture in ('bootstrap', 'memory', 'copy', 'module', 'launch', 'fill', 'graph', 'stream', 'graph_gpu', 'capture_gpu', 'copy_gpu')}


def check_log(raw, fixture='bootstrap'):
    expected = {'bootstrap': EXPECTED, 'memory': MEMORY_EXPECTED, 'copy': COPY_EXPECTED, 'module': MODULE_EXPECTED, 'launch': LAUNCH_EXPECTED, 'fill': FILL_EXPECTED, 'graph': GRAPH_EXPECTED, 'stream': STREAM_EXPECTED, 'graph_gpu': GRAPH_GPU_EXPECTED, 'capture_gpu': CAPTURE_GPU_EXPECTED, 'copy_gpu': COPY_GPU_EXPECTED}.get(fixture)
    if expected is None or not isinstance(raw, bytes) or len(raw) > 65536 or raw != expected:
        raise ValueError('public HIP transcript incomplete, changed, wrong fixture or contaminated')
    return {'application_result': 'application-pass', 'kernel': 'public_hip_' + fixture,
            'target': 'gfx1013', 'log_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'hip_init': 0, 'device_count': 1,
            'hardware_gate_passed': False,
            'unproven': ['freshness', 'installed identity', 'native process lifecycle',
                         'visible mount state', 'service health', 'owned lock cleanup']}
