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
LOG_NAMES = {'bootstrap': 'ps5-hip-public-bootstrap.log', 'memory': 'ps5-hip-public-memory.log'}


def check_log(raw, fixture='bootstrap'):
    expected = {'bootstrap': EXPECTED, 'memory': MEMORY_EXPECTED}.get(fixture)
    if expected is None or not isinstance(raw, bytes) or len(raw) > 65536 or raw != expected:
        raise ValueError('public HIP transcript incomplete, changed, wrong fixture or contaminated')
    return {'application_result': 'application-pass', 'kernel': 'public_hip_' + fixture,
            'target': 'gfx1013', 'log_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'hip_init': 0, 'device_count': 1,
            'hardware_gate_passed': False,
            'unproven': ['freshness', 'installed identity', 'native process lifecycle',
                         'visible mount state', 'service health', 'owned lock cleanup']}
