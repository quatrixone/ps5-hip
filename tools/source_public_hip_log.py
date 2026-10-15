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
COPY_LINES[2] = 'HIP_COPY PASS init=0 count=1 rounds=2 compared_bytes=1734 bounds_rejected=6 allocations=2 frees=2 watchdog_joined=true finalization=pending'
COPY_LINES[6] = 'HIP_COPY FINAL allocations=2 frees=2 queues=1 retired=1 gate_destroyed=true application_fini=pending'
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
FILL_LINES[2] = 'HIP_FILL PASS init=0 count=1 rounds=2 compared_bytes=512 rejected=8 allocations=1 frees=1 gpu_execution=false watchdog_joined=true finalization=pending'
FILL_LINES[6] = 'HIP_FILL FINAL allocations=1 frees=1 queues=1 retired=1 gate_destroyed=true application_fini=pending'
FILL_EXPECTED = ('\n'.join(FILL_LINES) + '\n').encode('ascii')
LOG_NAMES = {fixture: 'ps5-hip-public-' + fixture + '.log' for fixture in ('bootstrap', 'memory', 'copy', 'module', 'launch', 'fill')}


def check_log(raw, fixture='bootstrap'):
    expected = {'bootstrap': EXPECTED, 'memory': MEMORY_EXPECTED, 'copy': COPY_EXPECTED, 'module': MODULE_EXPECTED, 'launch': LAUNCH_EXPECTED, 'fill': FILL_EXPECTED}.get(fixture)
    if expected is None or not isinstance(raw, bytes) or len(raw) > 65536 or raw != expected:
        raise ValueError('public HIP transcript incomplete, changed, wrong fixture or contaminated')
    return {'application_result': 'application-pass', 'kernel': 'public_hip_' + fixture,
            'target': 'gfx1013', 'log_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'hip_init': 0, 'device_count': 1,
            'hardware_gate_passed': False,
            'unproven': ['freshness', 'installed identity', 'native process lifecycle',
                         'visible mount state', 'service health', 'owned lock cleanup']}
