"""Reject altered extraction bodies, metadata, descriptors and input identities."""
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_q4_fixture as b

class Fixture(unittest.TestCase):
 def test_preserved_body_allows_only_identifier_renumbering(self):
  original='define amdgpu_kernel void @kernel(i32 %0) #7 {\n  %2 = add i32 %0, 1, !range !8\n  ret void\n}\nattributes #7 = { nounwind }\n!8 = !{i32 0, i32 12}\n'
  selected=original.replace('#7','#0').replace('!8','!0')
  self.assertTrue(b.preserved_function(original,selected,'kernel'))
  for change in (selected.replace('add i32','sub i32'),selected.replace('nounwind','alwaysinline'),selected.replace('i32 12','i32 13'),selected+'define amdgpu_kernel void @other() {\n ret void\n}\n'):
   with self.assertRaises(ValueError):b.preserved_function(original,change,'kernel')
 def test_retained_helper_and_target_preserved(self):
  ir='target triple = "amdgcn-amd-amdhsa"\ntarget datalayout = "layout"\ndefine amdgpu_kernel void @k() {\n call void @helper()\n ret void\n}\ndefine void @helper() {\n ret void\n}\n'
  self.assertEqual(len(b.preserved_definitions(ir,ir,'k')[1]),2)
  for changed in (ir.replace('define void @helper() {\n ret void','define void @helper() {\n unreachable'),ir.replace('layout"','different"')):
   with self.assertRaises(ValueError):b.preserved_definitions(ir,changed,'k')
 def test_named_type_and_lds_transform(self):
  name='@_ZZ'+b.Q4[2:]+'E10tmp_shared'
  original='%struct.block = type { i32 }\n'+name+' = internal unnamed_addr addrspace(3) global [3 x [1 x [1 x [32 x float]]]] undef, align 16\n'
  selected=original.replace('internal unnamed_addr','external hidden unnamed_addr').replace(' undef,',',')
  self.assertEqual(len(b.preserved_data(original,selected,b.Q4)['allowed_lds_transformations']),1)
  for bad in (selected.replace('type { i32 }','type { i64 }'),selected.replace('align 16','align 8'),selected.replace('[32 x float]','[31 x float]'),selected.replace('addrspace(3)','addrspace(1)')):
   with self.assertRaises(ValueError):b.preserved_data(original,bad,b.Q4)
 def test_missing_function_and_metadata_rejected(self):
  with self.assertRaises(ValueError):b.canonical_function('','missing')
  with self.assertRaises(ValueError):b.canonical_function('define amdgpu_kernel void @k() {\n ret void, !x !3\n}\n','k')
 def test_stale_identity_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'input';p.write_bytes(b'first');inputs={};b.bind(inputs,p);p.write_bytes(b'changed')
   with self.assertRaises(ValueError):b.bind(inputs,p)
 def test_exact_descriptors(self):
  for profile in b.PROFILES.values():
   desc=dict(group_bytes=profile['static'],private_bytes=0,kernarg_bytes=profile['explicit']+profile['implicit'],rsrc1=profile['rsrc1'],rsrc2=profile['rsrc2'],rsrc3=0,properties=0x409,preload=0)
   b.descriptor_contract(desc,profile)
   for key in desc:
    wrong=dict(desc);wrong[key]^=1
    with self.subTest(key=key),self.assertRaises(ValueError):b.descriptor_contract(wrong,profile)
 def test_full_metadata_rejects_missing_hidden_argument(self):
  p=b.PROFILES['q8'];rows=[]
  for i,(o,s) in enumerate(zip(p['offsets'],p['sizes'])):rows.append((o,s,'global_buffer' if i in p['pointers'] else 'by_value'))
  for o,s,k in [(0,4,'block_count_x'),(4,4,'block_count_y'),(8,4,'block_count_z'),(12,2,'group_size_x'),(14,2,'group_size_y'),(16,2,'group_size_z'),(18,2,'remainder_x'),(20,2,'remainder_y'),(22,2,'remainder_z'),(40,8,'global_offset_x'),(48,8,'global_offset_y'),(56,8,'global_offset_z'),(64,2,'grid_dims')]:rows.append((72+o,s,'hidden_'+k))
  notes='amdhsa.target:   amdgcn-amd-amdhsa--gfx1013\n  - .args:\n'+''.join('      - .offset: %d\n        .size: %d\n        .value_kind: %s\n'%r for r in rows)
  notes+='    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n    .kernarg_segment_size: 328\n    .kernarg_segment_align: 8\n    .max_flat_workgroup_size: 256\n    .wavefront_size: 32\n    .name: '+p['symbol']+'\n'
  self.assertEqual(len(b.metadata_contract(notes,p)['arguments']),22)
  for changed in (notes.replace('hidden_grid_dims','by_value'),notes.replace('.offset: 60','.offset: 64'),notes.replace('size: 12','size: 16'),notes.replace('wavefront_size: 32','wavefront_size: 64'),notes+'    .name: extra\n'):
   with self.assertRaises(ValueError):b.metadata_contract(changed,p)

if __name__=='__main__':unittest.main()
