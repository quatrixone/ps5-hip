from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
# Staged runs use repository authentication helpers without copying them.
for parent in Path(__file__).resolve().parents:
 if (parent/'tools/build_llm_executed_graph.py').is_file():sys.path.append(str(parent/'tools'));break
import build_llm_vocab_fixture as v
class Field:
 def __init__(self,value,types=(8,)):self.value=value;self.types=types
 def contents(self):return self.value
class Reader:
 def __init__(self):self.fields={'general.architecture':Field('llama'),'tokenizer.ggml.tokens':Field(['a','b'],(9,8))};self.tensors=[]
 def get_field(self,n):return self.fields[n]
class Tests(unittest.TestCase):
 def test_exact_metadata(self):self.assertEqual(v.validate_metadata(Reader(),Reader()),2)
 def test_reject_tensor_missing_type_value(self):
  for mode in ('tensor','missing','type','value'):
   b=Reader()
   if mode=='tensor':b.tensors=[1]
   elif mode=='missing':del b.fields['general.architecture']
   elif mode=='type':b.fields['general.architecture'].types=(4,)
   else:b.fields['general.architecture'].value='wrong'
   with self.subTest(mode=mode),self.assertRaises(ValueError):v.validate_metadata(Reader(),b)
 def rows(self):return [dict(case=c,add_special=a,parse_special=s,tokens=[1]) for c in range(8) for a in (False,True) for s in (False,True)]
 def test_token_flags_exact(self):
  raw='\n'.join(map(json.dumps,self.rows()));v.validate_tokens(raw,raw)
  with self.assertRaises(ValueError):v.validate_tokens(raw,raw+' ')
 def test_incomplete_duplicate_bad_tokens(self):
  for mode in ('short','duplicate','token'):
   rows=self.rows()
   if mode=='short':rows.pop()
   elif mode=='duplicate':rows[-1]=rows[0]
   else:rows[0]['tokens']=[32000]
   raw='\n'.join(map(json.dumps,rows))
   with self.subTest(mode=mode),self.assertRaises(ValueError):v.validate_tokens(raw,raw)
 def test_asset_mutation(self):
  with tempfile.TemporaryDirectory() as temp:
   p=Path(temp)/'vocab';p.write_bytes(b'original');h=v.g.sha(p);v.verify_asset(p,h);p.write_bytes(b'mutated')
   with self.assertRaises(ValueError):v.verify_asset(p,h)
 def test_fresh_output(self):
  with tempfile.TemporaryDirectory() as temp:
   with self.assertRaisesRegex(ValueError,'fresh'):v.build(Path(temp),Path(temp),Path(temp),Path(temp))
if __name__=='__main__':unittest.main()