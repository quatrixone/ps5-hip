"""Fixed offline OCML fixture and numerical-contract admission for the native probe."""
import hashlib
import json
import struct
from pathlib import Path
import probe_ocml_math as probe

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def lock(root=ROOT):
    value = json.loads((root / 'tools/math_native_inputs.json').read_text())
    if not value['oracle_sources']:
        raise ValueError('math numerical oracle is not frozen')
    for name, expected in value['oracle_sources'].items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError('math oracle source changed: ' + name)
    return value


def profile(target, root=ROOT):
    value = lock(root)
    selected = [p for p in value['profiles'] if p['target'] == target and p['optimization'] == 'O2']
    if len(selected) != 1:
        raise ValueError('unsupported math target')
    return dict(selected[0], inputs=257, output_words=1542, groups=5, kernarg_bytes=280,
                explicit_arguments=3, hidden_arguments=13, user_sgprs=6,
                contract_sha256=value['oracle_sources']['tools/math_oracle_contract.json'],
                oracle_sha256=value['oracle_sources']['tools/math_oracle.h'],
                device_receipt_sha256=value['device_receipt_sha256'])


def validate_build(build, data, root=ROOT):
    value = lock(root)
    expected = profile(build['target'], root)
    if (build.get('kernel') != 'fp32_math' or build.get('math_profile') != expected or
            canonical(build.get('device_build')) != value['device_report_sha256'] or
            sha(data) != expected['code_sha256'] or len(data) != expected['bytes']):
        raise ValueError('math profile, compiler receipt or code object changed')
    required = set(value['oracle_sources']) | set(build['device_build']['sources']) | {
        'tools/math_native_fixture.py', 'tools/math_native_inputs.json'}
    for name in required:
        if build['source_sha256'].get(name) != sha((root / name).read_bytes()):
            raise ValueError('missing or changed math source binding: ' + name)


def verified_original_device_object(root, directory, target, version):
    directory = directory.resolve()
    value = lock(root)['original']
    raw = (directory / 'receipt.json').read_bytes()
    report = json.loads(raw)
    if sha(raw) != value['device_receipt_sha256'] or canonical(report) != value['device_report_sha256']:
        raise ValueError('unrecognized OCML compiler receipt')
    if version != report['source_pins']['llvm_version']:
        raise ValueError('math compiler version changed')
    for name, expected in report['sources'].items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError('math compiler source changed: ' + name)
    for name, expected in report['artifacts'].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or path.is_symlink() or sha(path.read_bytes()) != expected:
            raise ValueError('math compiler artifact changed: ' + name)
    for name in ('package-stage', 'device-libs-source'):
        wanted = dict(report['inputs'][name])
        if name == 'device-libs-source':
            wanted['cmake/OCL.cmake'] = report['derived_policy_sha256']
        if probe.inventory(directory / name) != wanted:
            raise ValueError('math compiler source/header inventory changed')
    if (probe.inventory(directory / 'libs') != report['libraries'] or
            probe.sha(directory / 'wrapper/CMakeLists.txt') != report['wrapper_sha256']):
        raise ValueError('math device libraries or build wrapper changed')
    for tool in [*report['tools'].values(), report['inspector']]:
        if probe.sha(Path(tool['path'])) != tool['sha256']:
            raise ValueError('math compiler or inspector changed')
    for item in report['profiles']:
        label = item['target'] + '-' + item['optimization']
        texts = [(directory / (label + '-' + suffix + '.stdout')).read_text()
                 for suffix in ('notes', 'isa', 'inspect', 'undefined', 'relocations')]
        probe.audit_object(directory / (label + '.hsaco'), item['target'], *texts)
        probe.audit_caller_ir((directory / (label + '-caller.ll')).read_text())
        command = next(c['argv'] for c in report['commands'] if c['label'] == label + '-compile')
        libraries = [directory / 'libs' / a.split('=', 1)[1] for a in command if a.startswith('--hip-device-lib=')]
        probe.audit_driver((directory / (label + '-driver.stderr')).read_text(),
                           Path(report['tools']['ld.lld']['path']), item['target'], libraries)
    selected = next(p for p in value['profiles'] if p['target'] == target and p['optimization'] == 'O2')
    data = (directory / (target + '-O2.hsaco')).read_bytes()
    if sha(data) != selected['code_sha256'] or len(data) != selected['bytes']:
        raise ValueError('math O2 selection changed')
    return data, report


def corrected_bytes(original, offsets):
    if (len(offsets) != 2 or any(type(x) is not int for x in offsets) or
            abs(offsets[0] - offsets[1]) < 4):
        raise ValueError('expected two distinct natural-exp sites')
    result = bytearray(original)
    for offset in offsets:
        if type(offset) is not int or offset < 0 or offset + 4 > len(result):
            raise ValueError('natural-exp site outside object')
        if struct.unpack_from('<I', original, offset)[0] != 0xc2ce8ed0:
            raise ValueError('original natural-exp immediate changed')
        struct.pack_into('<I', result, offset, 0xc2cff1b4)
    return bytes(result)


def verified_device_object(root, directory, target, version):
    directory = directory.resolve()
    value = lock(root)
    raw = (directory / 'receipt.json').read_bytes()
    report = json.loads(raw)
    if sha(raw) != value['device_receipt_sha256'] or canonical(report) != value['device_report_sha256']:
        raise ValueError('unrecognized corrected math fixture receipt')
    if report.get('classification') != 'derived-ocml-natural-exp-correction' or report.get('hardware_validated') is not False:
        raise ValueError('wrong corrected math fixture classification')
    original_directory = Path(report['original_directory']).resolve()
    _, original_report = verified_original_device_object(root, original_directory, target, version)
    if canonical(original_report) != report['original_report_sha256']:
        raise ValueError('original OCML authority changed')
    sources = dict(original_report['sources'])
    sources['tools/export_math_correction.py'] = sha((root / 'tools/export_math_correction.py').read_bytes())
    if report['sources'] != sources or report['profiles'] != value['profiles']:
        raise ValueError('correction source or profile changed')
    for name, expected in report['correction_files'].items():
        if sha(Path(name).read_bytes()) != expected:
            raise ValueError('corrected compiler authority changed: ' + name)
    for item in report['profiles']:
        label = item['target'] + '-' + item['optimization']
        path = directory / (label + '.hsaco')
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError('corrected object path escaped fixture')
        data = path.read_bytes()
        original = (original_directory / (label + '.hsaco')).read_bytes()
        if (sha(data) != item['code_sha256'] or len(data) != item['bytes'] or
                data != corrected_bytes(original, report['cutoff_offsets'][label])):
            raise ValueError('object changed beyond the two reviewed natural-exp immediates')
    selected = profile(target, root)
    return (directory / (target + '-O2.hsaco')).read_bytes(), report
