"""Fixed upstream embedding ABI; reject widened or altered device contracts."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_q4_fixture as b

class GetRows(unittest.TestCase):
 def notes(self):
  p=b.GET_ROWS
  rows=[(o,s,'global_buffer' if i in p['pointers'] else 'by_value') for i,(o,s) in enumerate(zip(p['offsets'],p['sizes']))]
  rows += [(128+o,s,'hidden_'+k) for o,s,k in [(0,4,'block_count_x'),(4,4,'block_count_y'),(8,4,'block_count_z'),(12,2,'group_size_x'),(14,2,'group_size_y'),(16,2,'group_size_z'),(18,2,'remainder_x'),(20,2,'remainder_y'),(22,2,'remainder_z'),(40,8,'global_offset_x'),(48,8,'global_offset_y'),(56,8,'global_offset_z'),(64,2,'grid_dims')]]
  return 'amdhsa.target:   amdgcn-amd-amdhsa--gfx1013\n  - .args:\n'+''.join('      - .offset: %d\n        .size: %d\n        .value_kind: %s\n'%r for r in rows)+'    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .kernarg_segment_size: 384\n    .kernarg_segment_align: 8\n    .max_flat_workgroup_size: 1024\n    .wavefront_size: 32\n    .name: '+p['symbol']+'\n'
 def test_exact_upstream_abi(self):
  self.assertEqual(len(b.metadata_contract(self.notes(),b.GET_ROWS)['arguments']),28)
  self.assertEqual(b.GET_ROWS['explicit'],128)
  self.assertEqual(b.GET_ROWS['pointers'],[0,1,2])
 def test_reject_metadata_mutations(self):
  notes=self.notes()
  for a,z in [('size: 12','size: 16'),('.offset: 56','.offset: 52'),('hidden_grid_dims','by_value'),('size: 384','size: 328'),('size: 1024','size: 256'),('wavefront_size: 32','wavefront_size: 64'),('gfx1013','gfx1030')]:
   with self.subTest(a=a),self.assertRaises(ValueError):b.metadata_contract(notes.replace(a,z),b.GET_ROWS)
 def test_exact_descriptor_and_reject_every_field(self):
  d=dict(group_bytes=0,private_bytes=0,kernarg_bytes=384,rsrc1=0x60af01c1,rsrc2=0x38c,rsrc3=0,properties=0x409,preload=0)
  b.descriptor_contract(d,b.GET_ROWS)
  for key in d:
   changed=dict(d);changed[key]^=1
   with self.subTest(key=key),self.assertRaises(ValueError):b.descriptor_contract(changed,b.GET_ROWS)
  with self.assertRaises(ValueError):b.descriptor_contract(dict(d,rsrc1=0x60af00c1),b.GET_ROWS)
 def test_existing_q8_q4_profiles_unchanged(self):
  self.assertEqual(set(b.PROFILES),{'q8','q4'})
  self.assertEqual((b.PROFILES['q8']['explicit'],b.PROFILES['q8']['threads'],b.PROFILES['q8']['rsrc1']),(72,256,0x60af00c1))
  self.assertEqual((b.PROFILES['q4']['explicit'],b.PROFILES['q4']['static'],b.PROFILES['q4']['rsrc1']),(192,384,0x60af00c3))
if __name__=='__main__':unittest.main()
