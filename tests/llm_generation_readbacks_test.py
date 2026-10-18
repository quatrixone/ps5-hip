import json,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_llm_full_readbacks as verifier
import build_llm_generation_reference as generation
import build_llm_generation_numeric_test as numeric


def write_generation_raw(folder):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 meta=dict(schema=1,prompt='Once upon a time',add_special=True,parse_special=False,max_new_tokens=4,vocab_size=32000,prompt_tokens=[1,2,3,4,5],eog_tokens=[10],runs=[]);steps=[]
 for run in range(2):
  meta['runs'].append(dict(run=run,processed=list(range(1,9)),generated=[6,7,8,9],stop='step_limit',pieces_hex=['61','62','63','64'],text_hex='61626364'))
  for pos in range(8):
   for stage in generation.STAGES:
    count=32000 if stage=='logits' else 2048;raw=bytearray(count*4)
    if stage=='logits':struct.pack_into('<f',raw,(pos+2 if pos>=4 else 0)*4,1.)
    (folder/f'run-{run}-pos-{pos}-{stage}.f32le').write_bytes(raw)
   steps.append(dict(run=run,position=pos,token=pos+1,status=0,committed_position=pos))
 (folder/'generation.json').write_text(json.dumps(meta)+'\n');(folder/'steps.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in steps))
 return {p.name:p.stat().st_size for p in folder.iterdir()}

class GenerationReadbacks(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.reference=self.root/'reference';self.raw=self.root/'raw'
  self.files=write_generation_raw(self.reference/'raw');write_generation_raw(self.raw)
  self.data={'receipt_sha256':'a'*64,'model':{'sha256':'b'*64},'requests':16,'raw_files':384}
  self.auth=patch.object(generation,'authenticate_generation_reference',return_value=(self.data,{}));self.auth.start();self.addCleanup(self.auth.stop)
 def verify(self):return verifier.verify(self.reference,self.raw,self.root/'comparison.json',profile='generation')
 def test_full_inventory_and_no_origin_claim(self):
  r=self.verify();self.assertEqual((r['requests'],r['float_files'],r['raw_files']),(16,384,386));self.assertEqual(r['raw_bytes'],sum(self.files.values()));self.assertEqual(len(r['metrics']),384);self.assertFalse(r['gpu_origin_established']);self.assertFalse(r['inference']);self.assertEqual(json.loads((self.root/'comparison.json').read_text()),r)
 def test_intermediate_error_fails_with_correct_tokens(self):
  for run in range(2):(self.raw/f'run-{run}-pos-0-layer-21.f32le').write_bytes(struct.pack('<f',.1)+bytes(8188))
  with self.assertRaisesRegex(ValueError,'budget'):self.verify()
 def test_reset_must_remain_bitwise(self):
  (self.raw/'run-1-pos-0-norm.f32le').write_bytes(struct.pack('<f',1e-8)+bytes(8188))
  with self.assertRaisesRegex(ValueError,'reset'):self.verify()
 def test_host_backend_file_rejected(self):
  (self.raw/'backend.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'inventory'):self.verify()
 def test_boolean_metadata_rejected(self):
  p=self.raw/'generation.json';p.write_text(p.read_text().replace('"run": 0','"run": false'))
  with self.assertRaises(ValueError):self.verify()
 def test_metadata_bytes_must_match(self):
  p=self.raw/'generation.json';p.write_text(p.read_text().replace('"text_hex": ','"text_hex" :'))
  with self.assertRaisesRegex(ValueError,'metadata'):self.verify()
 def test_symlink_and_short_before_hash(self):
  p=self.raw/'run-0-pos-0-norm.f32le';p.unlink();p.symlink_to(self.reference/'raw'/p.name)
  with self.assertRaisesRegex(ValueError,'kind/size'):self.verify()
  p.unlink();p.write_bytes(b'')
  with self.assertRaisesRegex(ValueError,'kind/size'):self.verify()
 def test_changed_reference_closure(self):
  p=self.root/'identity';p.write_bytes(b'x')
  with patch.object(generation,'authenticate_generation_reference',return_value=(self.data,{str(p):'a'*64})),self.assertRaisesRegex(ValueError,'reference changed'):self.verify()
 def test_mutation_during_comparison(self):
  original=numeric.compare_generation
  def mutate(*args,**kwargs):
   result=original(*args,**kwargs);p=self.raw/'run-0-pos-0-layer-1.f32le';p.write_bytes(b'X'+p.read_bytes()[1:]);return result
  with patch.object(numeric,'compare_generation',side_effect=mutate),self.assertRaisesRegex(ValueError,'changed during'):self.verify()
 def test_unknown_profile_rejected(self):
  with self.assertRaisesRegex(ValueError,'unsupported'):verifier.verify(self.reference,self.raw,self.root/'out',profile='long')

if __name__=='__main__':unittest.main()
