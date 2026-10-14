"""Bind the offline public HIP SVM title and FSELF without claiming execution."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import struct
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


checkout = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--console-title', action='store_true')
args = parser.parse_args()
root = checkout.parents[1]
link = checkout / ('results/native-memory-console-link-v1' if args.console_title
                   else 'results/native-memory-link-v1')
converted = checkout / ('results/native-memory-console-convert-v1' if args.console_title
                        else 'results/native-memory-convert-v1')
raw = converted / 'eboot.elf'
extracted = converted / 'extracted.elf'
fself = converted / 'eboot.bin'
assert all(p.is_file() for p in (raw, extracted, fself))
conversion = converted / 'conversion.json'
converted_receipt = json.loads(conversion.read_text())
assert all(command['exit_code'] == 0 for command in converted_receipt['commands'])
assert all(sha(path) == digest for path, digest in converted_receipt['inputs'].items())
assert all(sha(converted / name) == digest for name, digest in converted_receipt['artifacts'].items())
receipt = json.loads((link / 'receipt.json').read_text())
assert receipt['native_linked'] and not receipt['native_executed']
assert receipt['classification'] == 'strict-native-public-hip-memory-link-probe'
assert all(command['exit_code'] == 0 for command in receipt['commands'])
assert receipt['object_count'] == (74 if args.console_title else 71), receipt['object_count']
assert receipt['clean_closure'] and len(receipt['units']) == receipt['object_count']
assert all(u['exit_code'] == 0 and sha(u['source']) == u['source_sha256']
           and sha(u['object']) == u['object_sha256'] for u in receipt['units'])
assert all(Path(u['object']).parent == link for u in receipt['units'])
assert all(str(checkout / 'results/native-memory-source-v2') in u['source']
           for u in receipt['units'] if '/rocclr/' in u['source'] or '/hipamd/' in u['source'])
assert any(u['source'].endswith('/hipamd/src/hip_memory.cpp') for u in receipt['units'])
assert any(u['source'].endswith('/rocclr/platform/memory.cpp') for u in receipt['units'])
assert any(u['source'].endswith('/tools/public_hip_memory_console_title.cpp') for u in receipt['units'])
assert receipt['console_title'] is args.console_title
assert converted_receipt['console_title'] is args.console_title
assert all(sha(path) == digest for path, digest in receipt['bindings'].items())
compiled = checkout / 'results/native-hip-compile-all-v2/receipt.json'
units = json.loads(compiled.read_text())['commands']
assert len(units) == 30 and all(u['exit_code'] == 0 for u in units)
assert sha(checkout / 'results/native-source-v1/hipamd/src/hip_gl.cpp') == 'ead07db2ba6fd19cdabbfdaf8b7458cfdfc90062a4d5c8c31d1bc85e81bc51cc'
symbol = subprocess.run(['llvm-readelf-18', '-Ws', str(link / 'title.elf')],
                        capture_output=True, text=True, check=True).stdout
assert '__cxa_thread_atexit_impl' in symbol
assert not any('UND __cxa_thread_atexit_impl' in line for line in symbol.splitlines())
assert all(name in symbol for name in ('hipInit', 'hipGetDeviceCount', 'hipMalloc', 'hipFree',
                                       'ps5hip_clr_admit_device', 'ps5___cxa_thread_atexit_impl'))
helper = root / 'tools/native_vector_title.py'
node = next(n for n in ast.parse(helper.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == 'audit_roundtrip')
scope = {'struct': struct, 'hashlib': hashlib}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), 'exec'), scope)
roundtrip = scope['audit_roundtrip'](raw.read_bytes(), extracted.read_bytes())
assert roundtrip['matches_except_unmapped_sie_note']
report = {'classification': 'offline-native-public-hip-memory-link-and-conversion',
          'console_title': args.console_title,
          'link_receipt_sha256': sha(link / 'receipt.json'),
          'bindings': {str(helper): sha(helper), str(Path(__file__).resolve()): sha(__file__),
                       str(checkout / 'tools/guard_native_hip_gl.py'): sha(checkout / 'tools/guard_native_hip_gl.py'),
                       str(conversion): sha(conversion),
                       str(compiled): sha(compiled)},
          'raw_pie_sha256': sha(link / 'title.elf'),
          'converted_elf_sha256': sha(raw), 'fself_sha256': sha(fself),
          'extracted_elf_sha256': sha(extracted), 'fself_roundtrip': roundtrip,
          'hip_init_count_malloc_free_symbols_present': True,
          'standard_tls_destructor_hook_resolved': True,
          'native_executed': False,
          'public_hip_memory_qualified': False,
          'limits': ['No console execution; this is a strict link and conversion proof.',
                     'The target uses the reviewed HIP admission and SVM patches plus an isolated no-GL guard.',
                     'All 74 selected native objects were rebuilt from bound source; console execution remains required.',
                     'GPU-backed public HIP allocation, teardown and launch remain unverified.']}
(converted / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
print(sha(converted / 'receipt.json'))
