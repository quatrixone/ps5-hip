"""Pure GEMM ROCclr adapter over the reviewed prospective lifecycle contract."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2].parent / 'ps5-hip'
COMMON_PATH = ROOT / 'results/arithmetic-prospective-cycle2/acceptance.py'
COMMON_SHA = '4c5d72a664bcae17b70e229f61087e8097e9260e2cf9f2ac2806bc209168b71e'
CHECKER_PATH = HERE / 'check_log.py'
CHECKER_SHA = '064b87d00777b1f0990040e8d37919ec364f527821d393404e3678edeea044e1'
SOURCE = 'b00dac10adcc1ba45d9e2407b9a41f3c6c513c69'
IMAGE_SHA = '9c229e4221570716ec1cc2ee9bea08d41299bd6b433aad46a30ebf121f5777e4'
MOUNT = '/mnt/shadowmnt/PPSA99014_b6d9b2e4'


def load_pinned(path, expected, name):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('frozen acceptance dependency changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_pinned(COMMON_PATH, COMMON_SHA, 'spill_common_acceptance')
lifecycle = load_pinned(ROOT / 'results/visibility-bounded-lifecycle-v1/lifecycle.py',
    'a58c7441346678761081ac97e719f6d8eab91b566090381782babcfdd68f8fae', 'wave_bounded_lifecycle')
PREREQUISITES = common.PREREQUISITES + ('deployment_receipt_verified', 'mounted_candidate_files_verified')


def validate_deployment(receipt):
    expected = {'host': '192.168.4.40', 'title_id': 'PPSA99014',
                'kernel': 'clr_gemm', 'source_commit': SOURCE,
                'image_sha256': IMAGE_SHA, 'remote': '/data/homebrew/PPSA99014.ffpkg',
                'stage': 'registered'}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('wrong GEMM ROCclr registered deployment identity')
    if receipt.get('registered') is not True or receipt.get('launched') is not False or receipt.get('hardware_gate_passed') is not False:
        raise ValueError('deployment must report registration without launch/hardware acceptance')
    if receipt.get('accepted') is not True or receipt.get('owned_locks_released') is not True or any(
            receipt.get(key) for key in ('error', 'lock_errors', 'idle_error', 'health_error', 'ftp_close_error', 'local_final_error')):
        raise ValueError('deployment did not finish cleanly')
    if receipt.get('health_after') != {'2121': 'reachable', '3232': 'reachable', '9021': 'reachable'}:
        raise ValueError('deployment services were not healthy')
    image = receipt.get('image', {})
    if image != {'file': 'PPSA99014.ffpkg', 'bytes': 5636096, 'sha256': IMAGE_SHA}:
        raise ValueError('deployment image inventory mismatch')
    return {'verified_identity': True, **expected}


def validate_pre(snapshot_bytes):
    # New deployment may remap the image to another LVD device. Fresh mounted
    # eboot/hsaco bytes, metadata and image identity are controller prerequisites.
    rows = common._snapshot(snapshot_bytes)
    ufs = [row for row in rows if row['kind'] == 'registered_image' and
           row['fstype'] == 'ufs' and row['to'] == MOUNT and
           re.fullmatch(r'/dev/lvd[0-9]+', row['from'])]
    nullfs = [row for row in rows if row['kind'] == 'runtime_layer' and
              row['fstype'] == 'nullfs' and row['to'] == '/system_ex/app/PPSA99014' and row['from'] == MOUNT]
    if len(ufs) != 1 or len(nullfs) != 1 or len(rows) != 2:
        raise ValueError('GEMM ROCclr baseline lacks unique positive UFS/title-nullfs anchors')
    return {'ready': True, 'matching_rows': rows, 'matching_count': 2,
            'namespace': 'unqualified', 'release': 'unknown', 'accepted': False}


def check_lifecycle(shadow_delta_bytes, klog_bytes):
    return lifecycle.check_lifecycle(shadow_delta_bytes, klog_bytes)


def assess_cycle(pre_bytes, post_bytes, shadow_delta_bytes, klog_bytes,
                 application_log, stopped_before_close, *, prerequisites, deployment_receipt):
    result = {'classification': 'inconclusive', 'clr_gemm_cycle_accepted': False,
              'accepted': False, 'native_gemm_verified': False, 'hardware_gate_passed': False, 'full_g4_complete': False,
              'namespace': 'unqualified', 'release': 'unknown', 'reasons': [],
              'limits': [
                  'Only one genuine ROCclr GEMM AOT dispatch after three buffer writes: 99 exact FP32 outputs, 45 C padding guards, one checked queue join, watchdog and finalization.',
                  'No public HIP initialization, general GEMM shapes, asynchronous DMA, simultaneous GPU queues, performance or G5/full ROCm acceptance.',
                  'Visible matching-row equivalence is not global namespace completeness or reference-release proof.',
                  'Deployment/freshness/mounted bytes/capture boundaries/health/locks require independently reviewed controller evidence.',
              ]}
    absent = [key for key in PREREQUISITES if prerequisites.get(key) is not True]
    if absent:
        result['reasons'].append('missing controller prerequisites: ' + ', '.join(absent))
    if stopped_before_close is not True:
        result['reasons'].append('natural self-stop before controller cleanup not evidenced')
    try:
        result['deployment'] = validate_deployment(deployment_receipt)
        sys.path.insert(0, str(CHECKER_PATH.parent))
        checker = load_pinned(CHECKER_PATH, CHECKER_SHA, 'visibility_frozen_checker')
        result['application'] = checker.check_log(application_log)
        if result['application']['application_result'] != 'application-pass':
            result['reasons'].append('exact GEMM ROCclr dispatch, retirement, watchdog and finalization transcript did not pass')
            result['classification'] = 'failed'
        pre = validate_pre(pre_bytes)
        post_rows = common._snapshot(post_bytes)
        result['pre'] = pre
        result['post_matching_rows'] = post_rows
        result['matching_mount_multiset_equal'] = common._multiset(pre['matching_rows']) == common._multiset(post_rows)
        if not result['matching_mount_multiset_equal']:
            result['reasons'].append('post matching mount multiset differs from fresh deployed-image baseline')
            result['classification'] = 'failed'
        result['lifecycle'] = check_lifecycle(shadow_delta_bytes, klog_bytes)
    except (ValueError, RuntimeError, UnicodeError) as error:
        result['reasons'].append(str(error))
        if isinstance(error, (common.LifecycleFault, lifecycle.LifecycleFault)):
            result['classification'] = 'failed'
    if not result['reasons']:
        result.update(classification='pass', clr_gemm_cycle_accepted=True, accepted=True, native_gemm_verified=True)
    return result
