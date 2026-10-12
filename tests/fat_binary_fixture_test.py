"""Feed the bounded C++ reader unchanged bundles from12 pinned real HIP objects."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def section(data, wanted):
    check(data[:6] == b'\x7fELF\x02\x01' and len(data) >= 64, 'expected ELF64 host object')
    start = struct.unpack_from('<Q', data, 40)[0]
    width, count, names_index = struct.unpack_from('<HHH', data, 58)
    check(width == 64 and 0 < count <= 1024 and names_index < count, 'invalid section table')
    check(start <= len(data) and count * width <= len(data) - start, 'section table outside object')
    entries = [struct.unpack_from('<IIQQQQIIQQ', data, start + i * width) for i in range(count)]

    def contents(entry):
        offset, size = entry[4:6]
        check(offset <= len(data) and size <= len(data) - offset, 'section outside object')
        return data[offset:offset + size]

    strings = contents(entries[names_index])
    found = []
    for entry in entries:
        check(entry[0] < len(strings), 'section name outside string table')
        end = strings.find(b'\0', entry[0])
        check(end >= 0, 'unterminated section name')
        if strings[entry[0]:end] == wanted:
            found.append(contents(entry))
    check(len(found) == 1, 'expected one exact named section')
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--reader', type=Path, required=True)
    parser.add_argument('--inspector', type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads((ROOT / 'evidence/offline-compatible-hip-headers.json').read_text())
    manifest_bytes = (args.fixtures / 'compatible-emitted-abi.json').read_bytes()
    check(sha(manifest_bytes) == evidence['raw_file_hashes']['compatible-emitted-abi.json']['sha256'],
          'real compiler fixture manifest changed')
    cases = json.loads(manifest_bytes)
    expected = {f'{host}-{kernel}-{target}-{suffix}'
                for host, suffix in [('linux', 'with-routed-workitem-slice'), ('ps5', 'device-wchar4')]
                for kernel in ('launch', 'vector-add', 'dimensions') for target in ('gfx1013', 'gfx1030')}
    check(set(cases) == expected, 'real compiler fixture matrix changed')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output = args.reader.resolve().parent / ('fat-binary-real-fixtures-' + stamp)
    output.mkdir()
    results = []
    for name, case in sorted(cases.items()):
        data = (args.fixtures / (name + '.o')).read_bytes()
        device_path = args.fixtures / (name + '.hsaco')
        check(len(data) == case['bytes'] and sha(data) == case['sha256'], 'host object identity changed')
        check(sha(device_path.read_bytes()) == case['device_sha256'], 'device object identity changed')
        wrapper = section(data, b'.hipFatBinSegment')
        check(wrapper.hex() == case['fatbin_wrapper_hex'], 'compiler wrapper changed')
        bundle = section(data, b'.hip_fatbin')
        path = output / (name + '.bundle')
        path.write_bytes(bundle)
        target = 'gfx1013' if '-gfx1013-' in name else 'gfx1030'
        command = [str(args.reader.resolve()), '--bundle', str(path), target, str(device_path.resolve())]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        (output / (name + '.stdout')).write_text(result.stdout)
        (output / (name + '.stderr')).write_text(result.stderr)
        check(result.returncode == 0, name + ': ' + result.stderr)
        inspection = subprocess.run([str(args.inspector.resolve()), '--bundle', target, str(path)],
                                    capture_output=True, text=True, timeout=15)
        (output / (name + '.inspect.stdout')).write_text(inspection.stdout)
        (output / (name + '.inspect.stderr')).write_text(inspection.stderr)
        check(inspection.returncode == 0 and inspection.stdout.startswith(target + ' image_bytes=') and
              'hardware_validated=false' in inspection.stdout, 'bundle inspector failed')
        for descriptor in case['descriptors']:
            check(descriptor['symbol'].endswith('.kd') and
                  descriptor['symbol'][:-3] + ' entry=' in inspection.stdout, 'kernel not inspected')
        results.append({'case': name, 'host_sha256': sha(data), 'bundle_sha256': sha(bundle),
                        'device_sha256': case['device_sha256'], 'exit_code': result.returncode,
                        'inspector_exit_code': inspection.returncode})
    for rejected_target in ('gfx1010', 'gfx1030' if target == 'gfx1013' else 'gfx1013'):
        rejection = subprocess.run([str(args.inspector.resolve()), '--bundle', rejected_target, str(path)],
                                   capture_output=True, text=True, timeout=15)
        check(rejection.returncode == 1, 'inspector accepted missing/unsupported target')
    receipt = {'schema': 1, 'cases_passed': len(results), 'source_manifest_sha256': sha(manifest_bytes),
               'reader_sha256': sha(args.reader.read_bytes()), 'inspector_sha256': sha(args.inspector.read_bytes()),
               'cases': results,
               'console_access': False, 'registration_executed': False, 'hardware_validated': False}
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'cases_passed': len(results), 'output': str(output)}))


if __name__ == '__main__':
    main()
