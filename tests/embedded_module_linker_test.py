"""Link actual ELF objects and check exact immutable module-span records."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from embedded_module_linker import patch_linker, linux_linker_fragment

LAYOUT = '''OUTPUT_FORMAT("elf64-x86-64")
PHDRS { text PT_LOAD FLAGS(5); ro PT_LOAD FLAGS(4); relro PT_LOAD FLAGS(6); }
SECTIONS {
 .text 0x10000 : { *(.text) } : text
 . = ALIGN(4096);
    .rodata : { *(.rodata .rodata.*) } : ro
 . = ALIGN(4096);
 .data.rel.ro : {
        *(.data.rel.ro .data.rel.ro.*)
 } : relro
    .got : { *(.got) } : relro
 /DISCARD/ : { *(.comment .note.GNU-stack) }
}
'''


class LinkerTest(unittest.TestCase):
    def test_input_rejection(self):
        for names in ([], ['a.o'] * 2, ['../a.o'], ['a*.o'], ['a.o;'], ['a.o'] * 33):
            with self.assertRaises(ValueError): patch_linker(LAYOUT, names)
        with self.assertRaises(ValueError): patch_linker(LAYOUT + '__ps5hip_', ['a.o'])

    def test_linked_spans_and_missing_or_unlisted_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, value in (('a', 17), ('b', 29), ('extra', 41)):
                assembly = root / (name + '.s')
                assembly.write_text(f'''.section .hip_fatbin,"a",@progbits
bundle: .fill 32,1,{value}
.section .hipFatBinSegment,"aw",@progbits
.long 0x48495046,1
.quad bundle,0
''')
                subprocess.run(['clang-18', '-c', assembly, '-o', root / (name + '.o')], check=True)
            for names in (['a.o'], ['a.o', 'b.o'], ['b.o', 'a.o']):
                script = root / 'layout.ld'
                script.write_text(patch_linker(LAYOUT, names))
                image = root / 'image.elf'
                command = ['ld.lld-18', '-T', script, '-o', image]
                linked = subprocess.run([*command, *[root / n for n in names]], capture_output=True)
                self.assertEqual(linked.returncode, 0, linked.stderr.decode())
                symbols = {}
                for line in subprocess.check_output(['llvm-nm-18', image], text=True).splitlines():
                    fields = line.split()
                    if len(fields) == 3: symbols[fields[2]] = int(fields[0], 16)
                raw = image.read_bytes()
                phoff = struct.unpack_from('<Q', raw, 32)[0]
                width, count = struct.unpack_from('<HH', raw, 54)
                self.assertEqual(width, 56)
                headers = [struct.unpack_from('<II6Q', raw, phoff + width * i) for i in range(count)]
                def read(address, length):
                    h = next(h for h in headers if h[0] == 1 and h[3] <= address
                             and address + length <= h[3] + h[5])
                    offset = h[2] + address - h[3]
                    return raw[offset:offset + length], h
                table, region = read(symbols['__ps5hip_module_spans_begin'], len(names) * 32)
                self.assertEqual(region[1], 4)
                for i, name in enumerate(names):
                    wl, wh, bl, bh = struct.unpack_from('<4Q', table, i * 32)
                    self.assertEqual(wh - wl, 24)
                    self.assertEqual(bh - bl, 32)
                    bundle = symbols['__ps5hip_bundle_begin'] + bl
                    data, region = read(bundle, 32)
                    self.assertEqual(region[1], 4)
                    self.assertEqual(data, bytes([17 if name == 'a.o' else 29]) * 32)
                    wrapper, _ = read(symbols['__ps5hip_wrappers_begin'] + wl, 24)
                    self.assertEqual(struct.unpack('<IIQQ', wrapper), (0x48495046, 1, bundle, 0))
                # Missing listed inputs and unexpected compiler payloads fail at link time.
                for objects in (names[:-1], names + ['extra.o']):
                    result = subprocess.run([*command, *[root / n for n in objects]], capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
            script.write_text(patch_linker(LAYOUT, ['a.o']))
            other = root / 'other'
            other.mkdir()
            (other / 'a.o').write_bytes((root / 'a.o').read_bytes())
            (root / 'not-a.o').write_bytes((root / 'a.o').read_bytes())
            for objects in ([root / 'a.o', other / 'a.o'], [root / 'not-a.o']):
                self.assertNotEqual(subprocess.run([*command, *objects], capture_output=True).returncode, 0)
            main = root / 'main.cpp'
            main.write_text('''#include <cassert>
#include <cstdint>
#include <cstring>
extern "C" { extern const unsigned char __ps5hip_module_spans_begin[], __ps5hip_module_spans_end[];
extern const unsigned char __ps5hip_bundle_begin[], __ps5hip_wrappers_begin[]; }
int main() {
 assert(__ps5hip_module_spans_end - __ps5hip_module_spans_begin == 64);
 for(int i=0;i<2;++i) {
  uint64_t row[4], pointer;
  std::memcpy(row,__ps5hip_module_spans_begin+32*i,32);
  assert(row[1]-row[0]==24 && row[3]-row[2]==32);
  std::memcpy(&pointer,__ps5hip_wrappers_begin+row[0]+8,8);
  assert(pointer==reinterpret_cast<uintptr_t>(__ps5hip_bundle_begin+row[2]));
  assert(__ps5hip_bundle_begin[row[2]]==(i==0?29:17));
 }
}
''')
            script.write_text(linux_linker_fragment(['b.o', 'a.o']))
            linked = subprocess.run(['clang++-18', '-fPIE', '-pie', '-fuse-ld=lld-18',
                                     '-Wl,-T,' + str(script), main, root / 'a.o', root / 'b.o',
                                     '-o', image], capture_output=True)
            self.assertEqual(linked.returncode, 0, linked.stderr.decode())
            subprocess.run([image], check=True)


if __name__ == '__main__': unittest.main()
