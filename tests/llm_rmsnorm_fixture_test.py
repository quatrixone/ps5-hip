"""Authentication and ABI rejection tests for the source-only RMSNorm fixture."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_rmsnorm_fixture as fixture

class FixtureTest(unittest.TestCase):
    def test_tree_authentication(self):
        data=b'source\n'; blob=fixture.git_hash('blob',data)
        raw=b'100644 sample.cl\0'+bytes.fromhex(blob)
        digest=fixture.git_hash('tree',raw)
        tree=dict(sha=digest,truncated=False,tree=[dict(path='sample.cl',mode='100644',type='blob',sha=blob,size=len(data))])
        self.assertEqual(len(fixture.verify_tree(tree,digest)),1)
        for mutation in ('hash','truncated','path','mode','duplicate'):
            bad=copy.deepcopy(tree)
            if mutation=='hash':bad['tree'][0]['sha']='0'*40
            if mutation=='truncated':bad['truncated']=True
            if mutation=='path':bad['tree'][0]['path']='../sample.cl'
            if mutation=='mode':bad['tree'][0]['mode']='120000'
            if mutation=='duplicate':bad['tree'].append(bad['tree'][0].copy())
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):fixture.verify_tree(bad,digest)

    def notes(self):
        text='amdhsa.kernels:\n  - .args:\n'
        for i,(offset,size) in enumerate(zip(fixture.OFFSETS,fixture.SIZES)):
            kind='global_buffer' if i in (0,1,9,17) else 'by_value'
            text+=f'      - .offset: {offset}\n        .size: {size}\n        .value_kind: {kind}\n'
        for offset,size,kind in fixture.HIDDEN:
            text+=f'      - .offset: {offset}\n        .size: {size}\n        .value_kind: {kind}\n'
        text+='    .language_version:\n      - 2\n      - 0\n'
        for name,value in dict(name=fixture.SYMBOL,kernarg_segment_size=488,kernarg_segment_align=8,group_segment_fixed_size=0,private_segment_fixed_size=0,max_flat_workgroup_size=1024,wavefront_size=32).items():
            text+=f'    .{name}: {value}\n'
        return text

    def test_metadata_contract(self):
        text=self.notes();self.assertEqual(len(fixture.selected_metadata(text)['arguments']),40)
        for before,after in [('offset: 352','offset: 152'),('size: 12','size: 16'),('private_segment_fixed_size: 0','private_segment_fixed_size: 4'),('wavefront_size: 32','wavefront_size: 64'),('global_buffer','by_value')]:
            with self.subTest(before=before),self.assertRaises(ValueError):fixture.selected_metadata(text.replace(before,after,1))
        with self.assertRaises(ValueError):fixture.selected_metadata(text+text)
        hidden='      - .offset: 232\n        .size: 4\n        .value_kind: hidden_block_count_x\n'
        for bad in (text.replace(hidden,''),text.replace(hidden,hidden+hidden),text.replace('offset: 232','offset: 233')):
            with self.assertRaises(ValueError):fixture.selected_metadata(bad)

if __name__=='__main__':unittest.main()
