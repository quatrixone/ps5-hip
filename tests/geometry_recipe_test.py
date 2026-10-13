"""Actual full-HIP objects and adversarial recipe descriptor/metadata/ISA checks."""
import sys,unittest,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_geometry_compiler as probe
class Tests(unittest.TestCase):
    def test_actual_matrix_and_mutations(self):
        directory=ROOT/'build/geometry-probes-final'
        if not directory.exists():self.skipTest('fresh geometry compiler matrix unavailable')
        for p in directory.glob('*/geometry.hsaco'):
            data=p.read_bytes();metadata=(p.parent/'metadata.txt').read_text();isa=(p.parent/'isa.txt').read_text()
            result=probe.audit_object(data,metadata,isa)
            self.assertEqual(len(result['descriptors']),2)
            for old,new in [('.wavefront_size: 32','.wavefront_size: 64'),('.kernarg_segment_size: 296','.kernarg_segment_size: 40'),('.group_segment_fixed_size: 0','.group_segment_fixed_size: 256'),('.private_segment_fixed_size: 0','.private_segment_fixed_size: 4')]:
                import re
                field,value=old.split(': ');bad=re.sub(re.escape(field)+r':\s*'+value, new,metadata)
                self.assertNotEqual(metadata,bad)
                with self.assertRaises(RuntimeError):probe.audit_object(data,bad,isa)
            for prefix in ['flat_load','scratch_load','buffer_load','ds_read','s_swappc']:
                bad=isa.replace('global_load_dword',prefix+'_dword',1)
                self.assertNotEqual(bad,isa)
                with self.assertRaises(RuntimeError):probe.audit_object(data,metadata,bad)
            # Exact rsrc2 byte pattern appears only in the two descriptor records.
            needle=struct.pack('<IHH',0x138c,0x409,0)
            found=[i for i in range(len(data)) if data.startswith(needle,i)]
            self.assertEqual(len(found),2)
            for at in found:
                for bit in range(32):
                    bad=bytearray(data);bad[at+bit//8]^=1<<(bit%8)
                    with self.assertRaises(RuntimeError):probe.audit_object(bad,metadata,isa)
if __name__=='__main__':unittest.main(verbosity=2)
