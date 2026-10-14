"""Exact native GEMM transcript; freshness and lifecycle are separate gates."""
import hashlib

if not __debug__:
    raise RuntimeError('optimized Python is forbidden')

LINES = (
    'PS5HIP START target=gfx1013 kernel=clr_gemm',
    'PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass',
    'CLR_GEMM PASS launches=1 exact_values=99 padding_guards=45 allocations=3 frees=3 queues_joined=1 gate_destroyed=true',
    'CLR_GEMM_TITLE PASS launches=1 exact_values=99 padding_guards=45 allocations=3 frees=3 queues_joined=1 watchdog_joined=true finalization=pending',
    'PS5HIP SHUTDOWN main-return=0',
    'PS5HIP SHUTDOWN atexit=observed',
    'PS5HIP SHUTDOWN static-dtor=observed',
    'CLR_GEMM_TITLE FINAL launches=1 exact_values=99 padding_guards=45 allocations=3 frees=3 queues_joined=1 watchdog_joined=true application_fini=pending',
    'PS5HIP SHELL_EXIT application-fini=completed',
    'PS5HIP SHELL_EXIT request=begin',
    'PS5HIP SHELL_EXIT request-return=0',
)
EXPECTED = ('\n'.join(LINES) + '\n').encode('ascii')


def check_log(raw):
    if not isinstance(raw, bytes) or len(raw) > 65536 or raw != EXPECTED:
        raise ValueError('GEMM ROCclr transcript incomplete, changed or contaminated')
    return {'application_result': 'application-pass', 'kernel': 'clr_gemm',
            'target': 'gfx1013', 'log_sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw), 'launches': 1, 'exact_values': 99,
            'padding_guards': 45, 'queues_joined': 1, 'hardware_gate_passed': False,
            'public_hip': False,
            'unproven': ['freshness', 'installed identity', 'native process lifecycle',
                         'visible mount state', 'service health', 'owned lock cleanup']}
