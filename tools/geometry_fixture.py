"""Exact measured geometry fixture binding; no device execution."""
import hashlib
import json
from pathlib import Path
from build_canaries import require_tool_versions

PROFILES = {
    'geometry_2d': {'dimensions':2,'groups':[3,2,1],'threads':[8,8,1],'invocations':384,'record_words':6144,'logical_points':273,'inactive_points':111,'kernarg_bytes':296,'timeout_ms':2000},
    'geometry_3d': {'dimensions':3,'groups':[2,3,4],'threads':[8,4,2],'invocations':1536,'record_words':24576,'logical_points':819,'inactive_points':717,'kernarg_bytes':296,'timeout_ms':2000},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inputs(root):
    return json.loads((root / 'tools/geometry_inputs.json').read_text())


def verified_device_object(root, directory, target, version):
    lock = inputs(root)
    for name, expected in lock['sources'].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError('geometry compiler source changed: ' + name)
    for name, expected in lock['artifacts'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or digest((directory / path).read_bytes()) != expected:
            raise ValueError('geometry compiler artifact changed: ' + name)
    raw = (directory / 'comparison.json').read_bytes()
    if digest(raw) != lock['report_sha256']:
        raise ValueError('geometry compiler receipt changed')
    report = json.loads(raw)
    require_tool_versions({k:v['version'] for k,v in report['tools'].items()}, version)
    for name, expected in report['inputs'].items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError('geometry compiler input changed: ' + name)
    for tool in report['tools'].values():
        if digest(Path(tool['path']).read_bytes()) != tool['sha256']:
            raise ValueError('geometry compiler tool changed')
    data = (directory / (target + '-O2') / 'geometry.hsaco').read_bytes()
    if lock['objects'][target] != {'sha256': digest(data), 'bytes': len(data)}:
        raise ValueError('geometry selected object changed')
    return data, report


def validate_build(receipt, data, root):
    lock = inputs(root)
    expected_sources = [*lock['sources'], 'tools/native_geometry.h',
                        'tools/geometry_fixture.py', 'tools/geometry_inputs.json']
    if receipt.get('kernel') not in PROFILES or receipt.get('geometry_profile') != PROFILES[receipt['kernel']]:
        raise ValueError('geometry profile mismatch')
    if receipt.get('target') not in lock['objects'] or lock['objects'][receipt['target']] != {'sha256': digest(data), 'bytes': len(data)}:
        raise ValueError('geometry object mismatch')
    for name in expected_sources:
        actual = digest((root / name).read_bytes())
        if receipt.get('source_sha256', {}).get(name) != actual or (name in lock['sources'] and lock['sources'][name] != actual):
            raise ValueError('geometry source provenance mismatch: ' + name)
    # Builder records the exact qualified report, not merely adjacent mutable metadata.
    raw = (json.dumps(receipt.get('device_build'), indent=2) + '\n').encode()
    if digest(raw) != lock['report_sha256']:
        raise ValueError('geometry report provenance mismatch')
