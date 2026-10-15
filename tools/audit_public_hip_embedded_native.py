"""Audit bounded compiler registration in raw and converted native public-HIP images."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

from probe_host_registration import audit_image, load_regions, require
from native_multi_query import symbol_values, blocks
from fat_binary_fixture_test import section
from build_native_vector_title import audit_startup


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


FIXTURES = {"embedded": ("launch_public_embedded", 1), "aggregate": ("launch_aggregate_fixture", 2)}


def fixture_contract(fixture):
    require(fixture in FIXTURES, "unknown compiler fixture")
    return FIXTURES[fixture]


def audit_compiler_calls(ctor_text, entry_text, main_text, fixture):
    launcher, registrations = fixture_contract(fixture)
    require(ctor_text.count("<__hipRegisterFatBinary>") == 1 and
            ctor_text.count("<__hipRegisterFunction>") == registrations,
            "unexpected compiler module/function registration count")
    require("<" + launcher + ">" in main_text, "title does not call plain compiler launch entry")
    if fixture == "aggregate":
        for symbol in ("__hipPushCallConfiguration", "__hipPopCallConfiguration", "hipLaunchKernel"):
            require(entry_text.count("<" + symbol + ">") == 1, "aggregate compiler launch chain missing: " + symbol)
        require("<launch_pointer_aggregate_fixture>" not in main_text, "title launches pointer aggregate")


def unique_symbols(image, fixture="embedded"):
    values = symbol_values(image.read_bytes())
    def one(name):
        require(name in values and len(values[name]) == 1, 'missing or ambiguous symbol: ' + name)
        return values[name][0]
    names = ('__ps5hip_bundle_begin', '__ps5hip_bundle_end', '__ps5hip_wrappers_begin',
             '__ps5hip_wrappers_end', '__ps5hip_relro_begin', '__ps5hip_relro_end',
             '__ps5hip_module_spans_begin', '__ps5hip_module_spans_end',
             '__ps5hip_names_begin', '__ps5hip_names_end', '__init_array_start', '__init_array_end',
             '__hip_module_ctor', '__hip_module_dtor', '__hip_gpubin_handle',
             '_GLOBAL__sub_I_runtime.cpp', '_ZN3amd15RuntimeTearDownD1Ev',
             fixture_contract(fixture)[0], 'main', '_start', 'ps5hip_install_vector_title_finalizer')
    return {name: one(name) for name in names}


def audit_layout(image, symbols, bundle, native=False):
    result = audit_image(image, symbols, bundle, native)
    read, headers, relas = load_regions(image)
    lo, hi = symbols['__ps5hip_module_spans_begin'], symbols['__ps5hip_module_spans_end']
    require(hi - lo == 32, 'expected exact one-module span table')
    table, region = read(lo, 32)
    require(struct.unpack('<4Q', table) == (0, 24, 0, len(bundle)), 'module table does not own exact wrapper/bundle')
    require(not any(lo <= row[0] < hi for row in relas), 'relative-offset module table contains relocation')
    protected = [('table', lo, hi), ('bundle', symbols['__ps5hip_bundle_begin'], symbols['__ps5hip_bundle_end']),
                 ('names', symbols['__ps5hip_names_begin'], symbols['__ps5hip_names_end'])]
    for label, begin, end in protected:
        require(end > begin, 'empty protected ' + label)
        _, segment = read(begin, end - begin)
        require(segment[1] == 4, label + ' is not read-only')
        page_begin, page_end = begin & ~0x3fff, (end + 0x3fff) & ~0x3fff
        for h in headers:
            if h[0] == 1 and h[1] & 2:
                require(((h[3] + h[6] + 0x3fff) & ~0x3fff) <= page_begin or
                        (h[3] & ~0x3fff) >= page_end, 'writable load aliases protected ' + label + ' page')
    result.update(table=[lo, hi], table_sha256=hashlib.sha256(table).hexdigest(), read_only_page_bytes=16384)
    return result


def audit_initializer_order(image, symbols):
    read, _, relas = load_regions(image)
    begin, end = symbols['__init_array_start'], symbols['__init_array_end']
    require(end > begin and (end - begin) % 8 == 0, 'invalid initializer array')
    read(begin, end - begin)
    slots = {}
    for address, info, addend in relas:
        if begin <= address < end:
            require(info == 8 and (address - begin) % 8 == 0 and address not in slots,
                    'invalid or duplicate initializer relocation')
            slots[address] = addend
    require(len(slots) == (end - begin) // 8, 'initializer lacks exact relative relocation')
    entries = [slots[address] for address in sorted(slots)]
    runtime = symbols['_GLOBAL__sub_I_runtime.cpp']
    compiler = symbols['__hip_module_ctor']
    require(entries.count(runtime) == entries.count(compiler) == 1, 'runtime/compiler initializer missing or duplicated')
    require(entries.index(runtime) < entries.index(compiler), 'compiler unregister would follow runtime teardown')
    return {'runtime_initializer_index': entries.index(runtime), 'compiler_initializer_index': entries.index(compiler),
            'initializer_count': len(entries), 'exit_order': 'compiler unregister before CLR RuntimeTearDown'}


def audit_text(readers, functions, sizes):
    require(len(readers) == 3, 'expected raw, converted and extracted text readers')
    for address in functions:
        require(address in sizes, 'audited text has no bounded symbol size')
        expected, region = readers[0](address, sizes[address])
        require(region[1] == 5, 'raw audited code must be RX')
        for reader in readers[1:]:
            actual, region = reader(address, sizes[address])
            require(actual == expected, 'audited text changed during conversion')
            # sce_module_writer.cpp emits PT_LOAD/kFlagExecute for native text.
            require(region[1] == 1, 'native audited code must be execute-only')


def audit(raw, converted, extracted, compiler_object, fixture="embedded"):
    launcher, registrations = fixture_contract(fixture)
    paths = list(map(Path, (raw, converted, extracted, compiler_object)))
    raw, converted, extracted, compiler_object = paths
    root = Path(__file__).resolve().parents[1]
    helpers = [Path(__file__), root / 'tools/probe_host_registration.py', root / 'tools/native_multi_query.py',
               root / 'tools/build_native_vector_title.py', root / 'tests/fat_binary_fixture_test.py',
               root / 'tests/native_import_libraries_test.py']
    objdump = Path(shutil.which('llvm-objdump-18') or '')
    require(objdump.is_file(), 'LLVM18 objdump unavailable')
    identities = {str(p.resolve()): sha(p) for p in [*paths, *helpers, objdump]}
    symbols = unique_symbols(raw, fixture)
    bundle = section(compiler_object.read_bytes(), b'.hip_fatbin')
    require(bundle.startswith(b'__CLANG_OFFLOAD_BUNDLE__'), 'compiler object has no Clang bundle')
    disassembly = subprocess.check_output([str(objdump), '-d', '--no-show-raw-insn', '--print-imm-hex', raw], text=True)
    startup = audit_startup(disassembly, raw)
    bodies = {address: (name, body) for address, name, body in blocks(disassembly)}
    def text(address):
        require(address in bodies, 'audited function missing from disassembly')
        return bodies[address][1]
    refs = lambda value: {int(x, 16) for x in re.findall(r'0x([0-9a-f]+)', value)}
    ctor, dtor, handle = (symbols[n] for n in ('__hip_module_ctor', '__hip_module_dtor', '__hip_gpubin_handle'))
    audit_compiler_calls(text(ctor), text(symbols[launcher]), text(symbols['main']), fixture)
    require({symbols['__ps5hip_wrappers_begin'], handle, dtor} <= refs(text(ctor)), 'compiler ctor ownership changed')
    require(text(dtor).count('<__hipUnregisterFatBinary>') == 1 and handle in refs(text(dtor)), 'compiler dtor does not retire own handle')
    runtime = symbols['_GLOBAL__sub_I_runtime.cpp']
    visited, pending = set(), [runtime]
    while pending:
        address = pending.pop()
        if address in visited:
            continue
        visited.add(address)
        require(len(visited) <= 32, 'runtime initializer callgraph exceeds bounded audit')
        for target in re.findall(r'\bcallq?\s+0x([0-9a-f]+)', text(address)):
            target = int(target, 16)
            if target in bodies and bodies[target][0].startswith('__cxx_global_var_init'):
                pending.append(target)
    destructors = [address for address in visited
                   if symbols['_ZN3amd15RuntimeTearDownD1Ev'] in refs(text(address))]
    require(len(destructors) == 1, 'runtime initializer does not uniquely name RuntimeTearDown destructor')
    raw_bytes = raw.read_bytes()
    dynsym, dynstr = section(raw_bytes, b'.dynsym'), section(raw_bytes, b'.dynstr')
    indices = []
    for at in range(0, len(dynsym), 24):
        name = struct.unpack_from('<I', dynsym, at)[0]
        if dynstr[name:].split(b'\0', 1)[0] == b'__cxa_atexit':
            indices.append(at // 24)
    require(len(indices) == 1, 'runtime destructor registration import missing/ambiguous')
    slots = set()
    for name in (b'.rela.dyn', b'.rela.plt'):
        data = section(raw_bytes, name)
        for at in range(0, len(data), 24):
            address, info, _ = struct.unpack_from('<QQq', data, at)
            if info >> 32 == indices[0] and info & 0xffffffff in (6, 7):
                slots.add(address)
    require(slots, 'runtime destructor registration relocation missing')
    reader = load_regions(raw)[0]
    matches = []
    helper = text(destructors[0])
    for target in re.findall(r'\b(?:jmp|callq?)\s+0x([0-9a-f]+)', helper):
        address = int(target, 16)
        code, _ = reader(address, 6)
        if code[:2] == b'\xff\x25' and address + 6 + struct.unpack_from('<i', code, 2)[0] in slots:
            matches.append(address)
    indirect = [int(x, 16) for x in re.findall(r'callq?\s+\*[^\n]+#\s+0x([0-9a-f]+)', helper)]
    require(len(matches) + len([x for x in indirect if x in slots]) == 1,
            'RuntimeTearDown destructor is not registered through actual __cxa_atexit')
    startup['runtime_cxa_atexit_slots'] = sorted(slots)
    startup['runtime_destructor_registration_helper'] = destructors[0]
    stages = {name: {'layout': audit_layout(path, symbols, bundle, name != 'raw'),
                     'order': audit_initializer_order(path, symbols)}
              for name, path in (('raw', raw), ('converted', converted), ('extracted', extracted))}
    # The converter preserves virtual addresses and these ordinary text bytes.
    raw_bytes = raw.read_bytes()
    symtab = section(raw_bytes, b'.symtab')
    sizes = {}
    for offset in range(0, len(symtab), 24):
        _, info, _, index, value, size = struct.unpack_from('<IBBHQQ', symtab, offset)
        if index and info & 15 == 2 and size:
            sizes[value] = max(sizes.get(value, 0), size)
    functions = visited | {ctor, dtor, symbols['main'], symbols['_start'], symbols[launcher],
                           symbols['ps5hip_install_vector_title_finalizer']}
    readers = [load_regions(path)[0] for path in (raw, converted, extracted)]
    audit_text(readers, functions, sizes)
    require(all(sha(Path(name)) == digest for name, digest in identities.items()), 'audited artifact changed during audit')
    return {'schema': 1, 'classification': 'offline-public-hip-embedded-native-layout-audit',
            'fixture': fixture, 'registered_function_count': registrations,
            'inputs': identities, 'stages': stages, 'startup': startup,
            'audited_function_count': len(functions), 'native_executed': False,
            'limits': ['Static compiler call chain/layout audit, not on-console execution.',
                       'Final device backing counters do not prove complete static-module ownership retirement.',
                       'One compiler module; only the selected launcher is qualified; no eager native loading qualification.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'converted', 'extracted', 'compiler-object', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--fixture', choices=tuple(FIXTURES), default='embedded')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be fresh')
    report = audit(args.raw, args.converted, args.extracted, args.compiler_object, args.fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
