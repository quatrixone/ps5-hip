"""Static preservation and lifecycle audit for the dynamic-module RMSNorm title."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess

from audit_public_hip_embedded_native import audit_text
from probe_host_registration import load_regions, require
from native_multi_query import symbol_values, blocks
from fat_binary_fixture_test import section

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_slots(data, name):
    symbols, strings = section(data, b'.dynsym'), section(data, b'.dynstr')
    indices = [at//24 for at in range(0, len(symbols), 24)
               if strings[struct.unpack_from('<I', symbols, at)[0]:].split(b'\0', 1)[0] == name.encode()]
    require(len(indices) == 1, 'missing or ambiguous import: ' + name)
    slots = []
    for label in (b'.rela.dyn', b'.rela.plt'):
        relocations = section(data, label)
        for at in range(0, len(relocations), 24):
            address, info, _ = struct.unpack_from('<QQq', relocations, at)
            if info >> 32 == indices[0] and info & 0xffffffff in (6, 7):
                slots.append(address)
    require(slots and len(slots) == len(set(slots)), 'missing or duplicate import relocation: ' + name)
    return set(slots)


def bound_calls(body, slots, reader):
    """Return instruction positions bound to a real import GOT or PLT slot."""
    found = []
    for match in re.finditer(r'\b(?:jmp|callq?)\s+([^\n]+)', body):
        operand = match.group(1)
        indirect = re.search(r'^\*.*#\s+0x([0-9a-f]+)', operand)
        if indirect and int(indirect.group(1), 16) in slots:
            found.append(match.start())
        direct = re.match(r'0x([0-9a-f]+)', operand)
        if direct:
            address = int(direct.group(1), 16)
            code, _ = reader(address, 6)
            if code[:2] == b'\xff\x25' and address + 6 + struct.unpack_from('<i', code, 2)[0] in slots:
                found.append(match.start())
    return found


def symbol_calls(body, name):
    direct = [m.start() for m in re.finditer(r'\bcallq?\s+0x[0-9a-f]+\s+<' + re.escape(name) + r'>', body)]
    # The O0 native startup emits LEA(symbol), then register-indirect CALL.
    for match in re.finditer(r'leaq\s+[^\n]*, (%[a-z0-9]+)\s+# [^\n]*<' + re.escape(name) + r'>[^\n]*\n', body):
        register = match.group(1)
        families = (('rax','eax','ax','al','ah'), ('rbx','ebx','bx','bl','bh'),
                    ('rcx','ecx','cx','cl','ch'), ('rdx','edx','dx','dl','dh'),
                    ('rsi','esi','si','sil'), ('rdi','edi','di','dil'),
                    ('rbp','ebp','bp','bpl'), ('rsp','esp','sp','spl'))
        family = next((row for row in families if register[1:] in row), None)
        if family is None and re.fullmatch(r'%r(?:[89]|1[0-5])', register):
            family = tuple(register[1:] + suffix for suffix in ('','d','w','b'))
        require(family is not None, 'unsupported indirect-call register')
        mentions = r'%(?:' + '|'.join(family) + r')(?![a-z0-9])'
        cursor = match.end()
        for line in body[cursor:].splitlines(keepends=True)[:8]:
            if re.search(r'callq?\s+\*' + re.escape(register) + r'\s*$', line):
                direct.append(cursor); break
            if re.search(r'\b(?:callq?|j[a-z]+|loop[a-z]*|retq?)\b', line) or re.search(mentions, line):
                break
            cursor += len(line)
    return sorted(direct)


def ordered_calls(body, names):
    positions = []
    for name in names:
        matches = symbol_calls(body, name)
        require(len(matches) == 1, 'missing or repeated call: ' + name)
        positions.append(matches[0])
    require(positions == sorted(positions), 'startup call order changed')
    return positions


def audit(raw, converted, extracted):
    paths = [Path(p).resolve() for p in (raw, converted, extracted)]
    raw, converted, extracted = paths
    objdump = Path(shutil.which('llvm-objdump-18') or '')
    require(objdump.is_file(), 'LLVM18 objdump unavailable')
    sources = ['tools/audit_public_hip_rmsnorm_native.py', 'tests/public_hip_rmsnorm_native_audit_test.py',
               'tools/audit_public_hip_embedded_native.py', 'tools/probe_host_registration.py',
               'tools/native_multi_query.py', 'tests/fat_binary_fixture_test.py',
               'tests/native_import_libraries_test.py', 'tools/build_native_vector_title.py',
               'tools/public_hip_rmsnorm_console_title.cpp', 'tools/public_hip_app_crt.cpp']
    identities = {str(p): sha(p) for p in [*paths, objdump.resolve(), *(ROOT/s for s in sources)]}
    data = raw.read_bytes()
    values = symbol_values(data)
    def one(name):
        require(name in values and len(values[name]) == 1, 'missing or ambiguous symbol: ' + name)
        return values[name][0]
    disassembly = subprocess.check_output([str(objdump), '-d', '--no-show-raw-insn', '--print-imm-hex', str(raw)], text=True)
    bodies = {address: (name, text) for address, name, text in blocks(disassembly)}
    def body(name):
        address = one(name)
        require(address in bodies, 'missing disassembly: ' + name)
        return bodies[address][1]
    names = ['_start', '_init', 'main', 'ps5_fp_ieee', 'ps5hip_install_shell_exit_probe',
             'ps5hip_install_vector_title_finalizer']
    startup_positions = ordered_calls(body('_start'), names[3:] + ['_init', 'main'])
    dump_names = [name for name in values if 'dump_exclusive' in name]
    finalized_names = [name for name in values if name.endswith('finalizedEv')]
    require(len(dump_names) == len(finalized_names) == 1, 'dump/finalizer helper missing or ambiguous')
    dump, finalized = dump_names[0], finalized_names[0]
    names += [dump, finalized]
    main = body('main')
    public = ['hipModuleLoad', 'hipModuleGetFunction', 'hipModuleLaunchKernel',
              'hipDeviceSynchronize', 'hipMemcpy', 'hipFree', 'hipModuleUnload']
    for name in public:
        require(symbol_calls(main, name), 'main missing public call: ' + name)
        names.append(name)
    require(len(symbol_calls(main, dump)) == 2, 'main must call raw dump helper twice')
    reader = load_regions(raw)[0]
    slots = {name: import_slots(data, name) for name in ('atexit', '_init_env', 'open', 'write', 'fsync', 'close')}
    environment = bound_calls(body('_start'), slots['_init_env'], reader)
    require(len(environment) == 1 and startup_positions[0] < environment[0] < startup_positions[1], 'runtime environment startup order changed')
    finalizer = body('ps5hip_install_vector_title_finalizer')
    require(len(bound_calls(finalizer, slots['atexit'], reader)) == 1, 'finalizer not registered with actual atexit')
    refs = {int(x, 16) for x in re.findall(r'0x([0-9a-f]+)', finalizer)}
    require(one(finalized) in refs, 'wrong callback passed to atexit')
    dump_body = body(dump)
    for name in ('open', 'write', 'fsync', 'close'):
        require(len(bound_calls(dump_body, slots[name], reader)) == 1, 'dump missing actual import call: ' + name)
    require(bound_calls(dump_body, slots['fsync'], reader)[0] < bound_calls(dump_body, slots['close'], reader)[0], 'dump close precedes sync')
    sizes = {}
    symtab = section(data, b'.symtab')
    for at in range(0, len(symtab), 24):
        _, info, _, index, value, size = struct.unpack_from('<IBBHQQ', symtab, at)
        if index and info & 15 == 2 and size:
            sizes[value] = max(sizes.get(value, 0), size)
    functions = {one(name) for name in names}
    audit_text([load_regions(path)[0] for path in paths], functions, sizes)
    require(all(sha(path) == digest for path, digest in identities.items()), 'audit input changed')
    return dict(schema=1, status='PASS', classification='offline-public-hip-rmsnorm-native-image-audit',
                inputs=identities, audited_function_count=len(functions), public_calls=public,
                startup_order=['IEEE', 'runtime environment', 'shell exit', 'title finalizer', 'constructors', 'main'],
                imports={name: sorted(addresses) for name, addresses in slots.items()},
                dump_helper=dump, finalizer_callback=finalized, native_executed=False,
                limits=['Static linked image identity/call and text-preservation audit only; no numerical execution proof.',
                        'Dynamic module title: no embedded compiler constructor is required.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'converted', 'extracted', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('fresh output required')
    report = audit(args.raw, args.converted, args.extracted)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
