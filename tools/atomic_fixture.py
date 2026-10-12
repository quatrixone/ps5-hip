"""Fixed native atomic diagnostic contract and byte-pinned compiler fixtures."""
import hashlib

SOURCE_SHA256 = '42bc699fb32a445ffbb8ad31cc87c21d9f124ec257ad811bcff8402c4fda403b'
PROBE_EVIDENCE_SHA256 = '05d2d666eac562ab956d6635db8f041f484d1f8f5170290f6d5ad5fe4534cdfa'
KERNELS = ['atomic_ticket32', 'atomic_ticket64', 'atomic_cas32', 'atomic_cas64',
           'atomic_indirect32', 'atomic_indirect64']
CONTRACT = {'groups': 257, 'threads': 64, 'invocations': 16448, 'timeout_ms': 2000,
            'initial_wrap_distance': 4096, 'cas64_token_base': '0x13579bdf00000000',
            'exact_values_per_dispatch': 16449, 'exact_values': 98694}
PROFILES = {
    ('gfx1013', 'O1'): ('39e8c08b947673d1f6620e6d7871a06bef1c61183d2c0a2e0a37a791fcedb0d9',
                       'fead49a163ae536d587e5daaba30271686c85a16a803257cc203de8b0437ac52'),
    ('gfx1013', 'O2'): ('fb8c32a84c62755f37157bf60435f2d86ade4f8d093bc3d5b2a9f6c9c7186bc1',
                       '3088135cbc499ce90fa2e781906491f6c2330fca8b1ee70a51742461543652f8'),
    ('gfx1030', 'O1'): ('66fb1eed5f70bb8b3837ab5dc42e7d8d35685612f9638432cf44ca347e1fd0af',
                       '9f3596098d84f840ecc21bb62db87aaccd4cf8cd4593901cf4b3e134f1dd9eda'),
    ('gfx1030', 'O2'): ('fc65d98ae2cdb34e6d5e37b940d3170abc416b5b5d3f1fd98dfc4fa2e51d16d2',
                       '1d047c9b858ad626ad6bcd05a892ddcaa264fd1cb2ee9cc3761862dc30a89072'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def binding(target, optimization='O2'):
    if (target, optimization) not in PROFILES: raise ValueError('unsupported atomic fixture profile')
    code, driver = PROFILES[target, optimization]
    return {'kind': 'fixed-atomic-contention-probe', 'target': target, 'optimization': optimization,
            'source_sha256': SOURCE_SHA256, 'probe_evidence_sha256': PROBE_EVIDENCE_SHA256,
            'code_sha256': code, 'bytes': 8272, 'driver_sha256': driver, 'llvm_version': '18.1.8'}


def verified_object(root, target, optimization='O2'):
    expected = binding(target, optimization)
    if digest((root / 'kernels/atomic_probe.hip').read_bytes()) != SOURCE_SHA256:
        raise ValueError('atomic fixture source changed')
    directory = root / 'build/atomic-probes' / (target + '-' + optimization)
    code = (directory / 'atomics.hsaco').read_bytes()
    trace = (directory / 'driver.txt').read_bytes()
    if (digest(code) != expected['code_sha256'] or len(code) != expected['bytes'] or
            digest(trace) != expected['driver_sha256']):
        raise ValueError('atomic fixture object or driver trace changed')
    return code, trace, expected


def validate_contract(item, target):
    if item.get('kernel') != 'atomic_suite' or item.get('target') != target:
        raise ValueError('atomic suite profile mismatch')
    if type(item.get('dispatches')) is not int or item['dispatches'] != 6 or item.get('dispatch_kernels') != KERNELS:
        raise ValueError('atomic suite requires exact six-dispatch order')
    contract = item.get('atomic')
    if (contract != CONTRACT or not isinstance(contract, dict) or
            any(type(contract[key]) is not type(value) for key, value in CONTRACT.items())):
        raise ValueError('atomic suite resource/oracle contract mismatch')


def validate_build(receipt):
    target = receipt.get('target')
    validate_contract(receipt, target)
    expected = binding(target)
    if (receipt.get('device_build') != expected or type(receipt['device_build'].get('bytes')) is not int or
            receipt.get('artifacts', {}).get('canary.hsaco') != expected['code_sha256']):
        raise ValueError('atomic suite requires pinned O2 device object')
    if receipt.get('source_sha256', {}).get('kernels/atomic_probe.hip') != SOURCE_SHA256:
        raise ValueError('atomic suite fixture source provenance missing')
    if not all(name in receipt.get('source_sha256', {}) for name in ('tools/native_atomics.h', 'tools/atomic_fixture.py')):
        raise ValueError('atomic suite native contract provenance missing')
    return expected
