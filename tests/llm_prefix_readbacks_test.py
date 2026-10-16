import struct
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_llm_prefix_readbacks as v

def pack(values):return struct.pack('<'+'f'*len(values),*values)
class Compare(unittest.TestCase):
    def test_embedding_nibble_order_and_scale(self):
        block=struct.pack('<e',0.5)+bytes(range(16))
        expected=[0.5*(i-8) for i in range(16)]+[-4.0]*16
        self.assertEqual(v.embedding_reference(block*64),pack(expected*64))
        for row in (block*63,(struct.pack('<e',float('nan'))+bytes(16))*64):
            with self.assertRaises(ValueError):v.embedding_reference(row)
    def setUp(self):
        self.embedding=pack([1.0]*2048)
        self.weights=pack([1.0]*2048)
        normalized=v.norm.fp64_reference([1.0]*2048)
        self.raw={n:pack([0.0]*w) for n,w in v.WIDTHS.items()}
        self.raw.update(embedding=self.embedding,normalized=pack(normalized),weighted=pack(normalized))
        self.chain={'projections':{n:{'fp64':[0.0]*w,'budget':[1e-8]*w} for n,w in v.weighted.PROFILE['projections'].items()}}
    def check(self,raw):
        with patch.object(v.weighted,'chained_reference',return_value=self.chain) as chain:
            result=v.compare_token(raw,self.embedding,self.weights,{})
            chain.assert_called_once_with(raw['weighted'],{})
            return result
    def test_all_values_and_chained_activation(self):
        self.assertEqual(self.check(self.raw)['q']['elements'],2048)
    def test_each_output_corruption(self):
        for name,width in v.WIDTHS.items():
            for value in (1000.0,float('nan'),float('inf')):
                raw=dict(self.raw);raw[name]=raw[name][:-4]+pack([value])
                with self.subTest(name=name,value=value),self.assertRaises(ValueError):self.check(raw)
    def test_missing_extra_short_outputs(self):
        raw=dict(self.raw);raw.pop('q')
        with self.assertRaises(ValueError):self.check(raw)
        raw=dict(self.raw);raw['extra']=b''
        with self.assertRaises(ValueError):self.check(raw)
        raw=dict(self.raw);raw['k']=raw['k'][:-4]
        with self.assertRaises(ValueError):self.check(raw)
    def test_inventory(self):
        dumps={f'token-{t}-{n}.f32le':data for t in v.TOKENS for n,data in self.raw.items()}
        embeddings={t:self.embedding for t in v.TOKENS}
        with patch.object(v.weighted,'chained_reference',return_value=self.chain):
            self.assertEqual(len(v.compare_buffers(dumps,embeddings,self.weights,{})),4)
        for name in ('missing','extra'):
            changed=dict(dumps)
            if name=='missing':changed.pop(next(iter(changed)))
            else:changed['extra']=b''
            with self.assertRaises(ValueError):v.compare_buffers(changed,embeddings,self.weights,{})
    def test_invalid_budget(self):
        for budget in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):v.compare_projection(pack([0.0]),{'fp64':[0.0],'budget':[budget]},1)

class VerifyBoundary(unittest.TestCase):
    def test_authenticated_reference_and_readback_identity(self):
        import tempfile,sys,types,hashlib,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);source=folder/'cpu';source.mkdir()
            row=(struct.pack('<e',0.5)+bytes(range(16)))*64
            embedding=v.embedding_reference(row)
            for token in v.TOKENS:
                (source/f'token-{token}-embedding.q4_0').write_bytes(row)
                (source/f'token-{token}-embedding.f32le').write_bytes(embedding)
            (source/'norm.f32le').write_bytes(bytes(8192))
            for name in ('q','k','v'):(source/f'{name}.weights.q4_0').write_bytes(b'weights')
            bound=folder/'bound';bound.write_bytes(b'bound');digest=hashlib.sha256(b'bound').hexdigest()
            identities={str(bound):digest,**{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}}
            auth=types.ModuleType('build_llm_prefix_reference');auth.authenticate_prefix_reference=lambda p:({'reference':str(source),'receipt_sha256':'a'*64,'model':{'sha256':'b'*64}},identities)
            logs=types.ModuleType('source_llm_prefix_log');raw={name:bytes(size) for name,size in v.FILES.items()};logs.read_raw=lambda p:dict(raw)
            output=folder/'result.json'
            with patch.dict(sys.modules,{'build_llm_prefix_reference':auth,'source_llm_prefix_log':logs}),patch.object(v,'compare_buffers',return_value={'checked':True}):
                result=v.verify(folder,folder,output)
                self.assertFalse(result['gpu_origin_established']);self.assertEqual(result['raw_files'],24);self.assertEqual(result['raw_bytes'],139264)
                self.assertEqual(result['raw_sha256'],{n:hashlib.sha256(b).hexdigest() for n,b in raw.items()})
                bound.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'reference changed'):v.verify(folder,folder,output)
                bound.write_bytes(b'bound')
                changed=dict(raw);changed[next(iter(changed))]=b'changed'
                with patch.object(logs,'read_raw',side_effect=[raw,changed]),self.assertRaisesRegex(ValueError,'readbacks changed'):v.verify(folder,folder,output)
                original_read=Path.read_bytes
                def substituted(path):
                    value=original_read(path)
                    return b'changed weights' if path.name=='q.weights.q4_0' else value
                # Disk is unchanged: reject the actual substituted bytes consumed by the comparison.
                with patch.object(Path,'read_bytes',substituted),self.assertRaisesRegex(ValueError,'consumed reference bytes'):v.verify(folder,folder,output)
                (source/'token-0-embedding.f32le').write_bytes(bytes(8192))
                with self.assertRaisesRegex(ValueError,'consumed reference bytes|independent Q4 decode'):v.verify(folder,folder,output)

if __name__=='__main__':unittest.main()
