"""Exercise native span/order audits against real ELF and corrupted copies."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_public_hip_embedded_native import audit_layout,audit_initializer_order,audit_text
from embedded_module_linker import patch_linker
from native_multi_query import symbol_values
from probe_host_registration import load_regions

class NativeAuditTest(unittest.TestCase):
    def test_real_layout_and_negative_controls(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);assembly=root/'fixture.s';obj=root/'fixture.o';image=root/'image.elf'
            assembly.write_text('''.text
.global _start, __hip_module_ctor, _GLOBAL__sub_I_runtime.cpp
_start: lea __init_array_start(%rip),%rax
lea __init_array_end(%rip),%rax
ret
__hip_module_ctor: ret
_GLOBAL__sub_I_runtime.cpp: ret
.section .rodata,"a",@progbits
.asciz "thread_words"
.section .hip_fatbin,"a",@progbits
bundle: .fill 32,1,17
.section .hipFatBinSegment,"aw",@progbits
.long 0x48495046,1
.quad bundle,0
.section .init_array,"aw",@init_array
.quad _GLOBAL__sub_I_runtime.cpp,__hip_module_ctor
''')
            subprocess.run(['clang-18','-c',assembly,'-o',obj],check=True,capture_output=True)
            script=root/'layout.ld';script.write_text(patch_linker((ROOT/'.deps/sources/native-tools/tooling/native/ps5-pie.ld').read_text(),['fixture.o']))
            subprocess.run(['ld.lld-18','-pie','-z','max-page-size=0x4000','-T',script,obj,'-o',image],check=True,capture_output=True)
            symbols={k:v[0] for k,v in symbol_values(image.read_bytes()).items() if len(v)==1}
            audit_layout(image,symbols,bytes([17])*32)
            self.assertEqual(audit_initializer_order(image,symbols)['compiler_initializer_index'],1)
            original=image.read_bytes()
            _,headers,relas=load_regions(image)
            def offset(address):
                h=next(h for h in headers if h[0]==1 and h[3]<=address<h[3]+h[5]);return h[2]+address-h[3]
            # Table bounds, wrapper magic, wrapper addend and writable-page alias all reject.
            bad=root/'bad.elf'
            for address in [symbols['__ps5hip_module_spans_begin'],symbols['__ps5hip_wrappers_begin']]:
                data=bytearray(original);data[offset(address)]^=1;bad.write_bytes(data)
                with self.assertRaises(RuntimeError):audit_layout(bad,symbols,bytes([17])*32)
            phoff=struct.unpack_from('<Q',original,32)[0]
            # The converter preserves exact instructions but emits X-only native text.
            text_address = symbols['_start']
            text_index = next(i for i,h in enumerate(headers) if h[0]==1 and h[3]<=text_address<h[3]+h[5])
            native = root/'native.elf'
            native_bytes = bytearray(original)
            struct.pack_into('<I',native_bytes,phoff+56*text_index+4,1)
            native.write_bytes(native_bytes)
            sizes = {text_address: 7}
            readers = lambda: [load_regions(p)[0] for p in (image,native,native)]
            audit_text(readers(), {text_address}, sizes)
            for flags in (0,2,3,4,5,6,7):
                data=bytearray(native_bytes);struct.pack_into('<I',data,phoff+56*text_index+4,flags)
                native.write_bytes(data)
                with self.assertRaisesRegex(RuntimeError,'execute-only'):
                    audit_text(readers(), {text_address}, sizes)
            data=bytearray(native_bytes);data[offset(text_address)]^=1;native.write_bytes(data)
            with self.assertRaisesRegex(RuntimeError,'text changed'):
                audit_text(readers(), {text_address}, sizes)
            native.write_bytes(native_bytes)
            with self.assertRaisesRegex(RuntimeError,'raw audited code must be RX'):
                audit_text([load_regions(native)[0]]*3, {text_address}, sizes)
            hindex=next(i for i,h in enumerate(headers) if h[0]==1 and h[3]<=symbols['__ps5hip_bundle_begin']<h[3]+h[5])
            data=bytearray(original);struct.pack_into('<I',data,phoff+56*hindex+4,6);bad.write_bytes(data)
            with self.assertRaises(RuntimeError):audit_layout(bad,symbols,bytes([17])*32)
            dynamic=next(h for h in headers if h[0]==2)
            tags=dict(struct.unpack_from('<QQ',original,at) for at in range(dynamic[2],dynamic[2]+dynamic[5],16))
            rela_offset=offset(tags[7]);wrapper=symbols['__ps5hip_wrappers_begin']+8
            index=next(i for i,row in enumerate(relas) if row[0]==wrapper)
            data=bytearray(original);struct.pack_into('<q',data,rela_offset+index*24+16,symbols['__ps5hip_bundle_begin']+1);bad.write_bytes(data)
            with self.assertRaises(RuntimeError):audit_layout(bad,symbols,bytes([17])*32)
            # Reverse actual initializer addends; static ordering oracle must fail.
            begin=symbols['__init_array_start'];ix=[next(i for i,row in enumerate(relas) if row[0]==begin+j*8) for j in range(2)]
            data=bytearray(original)
            for a,b in zip(ix,ix[::-1]):struct.pack_into('<q',data,rela_offset+a*24+16,relas[b][2])
            bad.write_bytes(data)
            with self.assertRaisesRegex(RuntimeError,'unregister would follow'):audit_initializer_order(bad,symbols)
            symbols['__ps5hip_module_spans_end']+=32
            with self.assertRaisesRegex(RuntimeError,'one-module'):audit_layout(image,symbols,bytes([17])*32)

if __name__=='__main__':unittest.main()
