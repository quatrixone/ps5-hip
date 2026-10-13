"""Exact completed-queue visibility fixture binding; no device execution."""
import hashlib
import json
from pathlib import Path
from build_canaries import require_tool_versions

PROFILE = {'elements': 257, 'groups': 5, 'threads': 64, 'rounds': 3,
           'dispatches': 6, 'persistent_buffers': 3, 'timeout_ms': 2000,
           'visibility': 'completed-queue-cpu-cache-maintenance'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inputs(root):
    return json.loads((root / 'tools/visibility_inputs.json').read_text())


def verified_device_object(root, directory, target, version):
    lock = inputs(root)
    for name, expected in lock['sources'].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError('visibility compiler source changed: ' + name)
    for name, expected in lock['artifacts'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest((directory / path).read_bytes()) != expected:
            raise ValueError('visibility compiler artifact changed: ' + name)
    raw = (directory / 'comparison.json').read_bytes()
    if digest(raw) != lock['report_sha256']:
        raise ValueError('visibility compiler receipt changed')
    report = json.loads(raw)
    require_tool_versions(report['tool_versions'], version)
    data = (directory / (target + '-O2') / 'visibility.hsaco').read_bytes()
    if lock['objects'][target] != {'sha256': digest(data), 'bytes': len(data)}:
        raise ValueError('visibility selected object changed')
    return data, report


def validate_build(receipt, data, root):
    lock = inputs(root)
    expected_sources = [*lock['sources'], 'tools/native_visibility.h',
                        'tools/visibility_fixture.py', 'tools/visibility_inputs.json']
    if receipt.get('kernel') != 'dependent_visibility' or receipt.get('visibility_profile') != PROFILE:
        raise ValueError('visibility profile mismatch')
    if receipt.get('target') not in lock['objects'] or lock['objects'][receipt['target']] != {'sha256': digest(data), 'bytes': len(data)}:
        raise ValueError('visibility object mismatch')
    for name in expected_sources:
        actual = digest((root / name).read_bytes())
        if receipt.get('source_sha256', {}).get(name) != actual or (name in lock['sources'] and lock['sources'][name] != actual):
            raise ValueError('visibility source provenance mismatch: ' + name)
    # Builder records the exact qualified report, not merely adjacent mutable metadata.
    raw = (json.dumps(receipt.get('device_build'), indent=2) + '\n').encode()
    if digest(raw) != lock['report_sha256']:
        raise ValueError('visibility report provenance mismatch')
