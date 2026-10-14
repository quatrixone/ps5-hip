"""Pure public HIP bootstrap assessment over the reviewed lifecycle contract."""
import hashlib
import importlib.util
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[5]
COMMON_PATH = ROOT / 'results/arithmetic-prospective-cycle2/acceptance.py'
COMMON_SHA = '4c5d72a664bcae17b70e229f61087e8097e9260e2cf9f2ac2806bc209168b71e'
CHECKER_PATH = ROOT / '.local/g5-hip-native-link/tools/public_hip_controls/execution/check_log.py'
CHECKER_SHA = '73a43f6ad0e5d53d5fd8d923fd4bdcb4433c7130349bfb82ec97f3dfe3ab7d6d'
SOURCE = 'ea37055286903f680e43b049d70730743ffa2358'
IMAGE_SHA = '836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8'
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
                'kernel': 'public_hip_bootstrap', 'source_commit': SOURCE,
                'image_sha256': IMAGE_SHA, 'remote': '/data/homebrew/PPSA99014.ffpkg',
                'stage': 'registered'}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError('wrong public HIP registered deployment identity')
    if receipt.get('registered') is not True or receipt.get('launched') is not False or receipt.get('hardware_gate_passed') is not False:
        raise ValueError('deployment must report registration without launch/hardware acceptance')
    if receipt.get('accepted') is not True or receipt.get('owned_locks_released') is not True or any(
            receipt.get(key) for key in ('error', 'lock_errors', 'idle_error', 'health_error', 'ftp_close_error', 'local_final_error')):
        raise ValueError('deployment did not finish cleanly')
    if receipt.get('health_after') != {'2121': 'reachable', '3232': 'reachable', '9021': 'reachable'}:
        raise ValueError('deployment services were not healthy')
    image = receipt.get('image', {})
    if image != {'file': 'PPSA99014.ffpkg', 'bytes': 9437184, 'sha256': IMAGE_SHA}:
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
        raise ValueError('public HIP baseline lacks unique positive UFS/title-nullfs anchors')
    return {'ready': True, 'matching_rows': rows, 'matching_count': 2,
            'namespace': 'unqualified', 'release': 'unknown', 'accepted': False}


def check_lifecycle(shadow_delta_bytes, klog_bytes):
    return lifecycle.check_lifecycle(shadow_delta_bytes, klog_bytes)


def assess_cycle(pre_bytes, post_bytes, shadow_delta_bytes, klog_bytes,
                 application_log, stopped_before_close, *, prerequisites, deployment_receipt):
    result = {'classification': 'inconclusive', 'public_hip_bootstrap_cycle_accepted': False,
              'accepted': False, 'public_hip_bootstrap_verified': False, 'hardware_gate_passed': False,
              'namespace': 'unqualified', 'release': 'unknown', 'reasons': [],
              'limits': [
                  'Only public hipInit(0) and hipGetDeviceCount(&count)==1, watchdog join, finalization and natural title stop are assessed.',
                  'Public HIP allocation, streams, GPU kernels, performance and full ROCm acceptance remain unproven.',
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
            result['reasons'].append('exact public HIP bootstrap and finalization transcript did not pass')
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
        result.update(classification='pass', public_hip_bootstrap_cycle_accepted=True, accepted=True, public_hip_bootstrap_verified=True)
    return result
