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


def check_log(raw):
    if not isinstance(raw, bytes) or len(raw) > 65536 or raw != EXPECTED:
        raise ValueError('public HIP bootstrap transcript incomplete, changed or contaminated')
    return {'application_result': 'application-pass', 'kernel': 'public_hip_bootstrap',
            'target': 'gfx1013', 'log_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'hip_init': 0, 'device_count': 1,
            'hardware_gate_passed': False,
            'unproven': ['freshness', 'installed identity', 'native process lifecycle',
                         'visible mount state', 'service health', 'owned lock cleanup']}
