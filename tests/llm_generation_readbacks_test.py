import hashlib,json,os,shutil,struct,sys,tempfile,unittest
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

class CorpusReadbacks(unittest.TestCase):
 def test_actual_corpus_inventory_and_numerical_gate(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);reference=root/'reference';folder=reference/'raw';folder.mkdir(parents=True);raw=root/'raw';zero=root/'zero';zero.write_bytes(bytes(8192))
   cases=generation.corpus_cases();(folder/'corpus.json').write_text(json.dumps(dict(schema=1,profile='corpus',max_new_tokens=16,cases=[key for key,_ in cases])))
   for (key,prompt),count in zip(cases,[5,6,18,19,13]):
    sub=folder/key;sub.mkdir();tokens=[1]*count;generated=list(range(3,19));processed=tokens+generated[:-1];runs=[];steps=[]
    for run in range(2):
     runs.append(dict(run=run,processed=processed,generated=generated,stop='step_limit',pieces_hex=['61']*16,text_hex='61'*16))
     for pos,token in enumerate(processed):
      values=bytearray(128000);winner=3+pos-count+1 if pos>=count-1 else 0;struct.pack_into('<f',values,winner*4,1)
      (sub/f'run-{run}-pos-{pos}-logits.f32le').write_bytes(values)
      for stage in generation.STAGES[:-1]:os.link(zero,sub/f'run-{run}-pos-{pos}-{stage}.f32le')
      steps.append(dict(run=run,position=pos,token=token,status=0,committed_position=pos))
    (sub/'generation.json').write_text(json.dumps(dict(schema=1,prompt=prompt,add_special=True,parse_special=False,max_new_tokens=16,vocab_size=32000,prompt_tokens=tokens,eog_tokens=[2],runs=runs)))
    (sub/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in steps))
   shutil.copytree(folder,raw,copy_function=os.link)
   data=dict(profile='corpus',receipt_sha256='a'*64,model={'sha256':'b'*64})
   with patch.object(generation,'authenticate_generation_reference',return_value=(data,{})):
    result=verifier.verify(reference,raw,root/'verified.json',profile='corpus')
    self.assertEqual((result['requests'],result['float_files'],result['raw_files']),(272,6528,6539));self.assertEqual(result['profile'],'corpus');self.assertFalse(result['gpu_origin_established']);self.assertFalse(result['inference']);self.assertEqual(len(result['raw_sha256']),6539)
    for run in range(2):
     path=raw/'story'/f'run-{run}-pos-0-layer-21.f32le';path.unlink();path.write_bytes(struct.pack('<f',.1)+bytes(8188))
    with self.assertRaisesRegex(ValueError,'budget'):verifier.verify(reference,raw,root/'bad.json',profile='corpus')
    self.assertFalse((root/'bad.json').exists())
 def test_nested_capture_guards_and_final_mutation(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);reference=root/'reference';raw=root/'raw';(reference/'raw/story').mkdir(parents=True);(raw/'story').mkdir(parents=True)
   files={'corpus.json':2,'story/generation.json':2,'story/steps.jsonl':2,'story/x.f32le':4}
   for name,size in files.items():
    for base in (raw,reference/'raw'):(base/name).write_bytes(b'x'*size)
   data=dict(profile='corpus',receipt_sha256='a'*64,model={})
   rows=[{'pass_budget':True}]*6528
   with patch.object(generation,'authenticate_generation_reference',return_value=(data,{})),patch.object(verifier,'corpus_files',return_value=files),patch.object(generation,'validate'),patch.object(numeric,'compare_corpus',return_value=(rows,{'story/x.f32le':4},272)) as compare:
    (raw/'extra').mkdir()
    with self.assertRaisesRegex(ValueError,'inventory'):verifier.verify(reference,raw,root/'out',profile='corpus')
    (raw/'extra').rmdir();path=raw/'story/x.f32le';path.unlink();path.symlink_to(reference/'raw/story/x.f32le')
    with self.assertRaisesRegex(ValueError,'kind'):verifier.verify(reference,raw,root/'out',profile='corpus')
    path.unlink();path.write_bytes(b'x')
    with self.assertRaisesRegex(ValueError,'kind/size'):verifier.verify(reference,raw,root/'out',profile='corpus')
    path.write_bytes(b'xxxx');meta=raw/'story/generation.json';meta.write_bytes(b'yy')
    with self.assertRaisesRegex(ValueError,'metadata'):verifier.verify(reference,raw,root/'out',profile='corpus')
    meta.write_bytes(b'xx')
    def mutate(*args,**kwargs):path.write_bytes(b'yyyy');return rows,{'story/x.f32le':4},272
    compare.side_effect=mutate
    with self.assertRaisesRegex(ValueError,'changed during'):verifier.verify(reference,raw,root/'out',profile='corpus')
    path.write_bytes(b'xxxx')
    def extra(*args,**kwargs):(raw/'extra').mkdir();return rows,{'story/x.f32le':4},272
    compare.side_effect=extra
    with self.assertRaisesRegex(ValueError,'inventory'):verifier.verify(reference,raw,root/'out',profile='corpus')
    (raw/'extra').rmdir();compare.side_effect=None
    identity=root/'identity';identity.write_bytes(b'x')
    with patch.object(generation,'authenticate_generation_reference',return_value=(data,{str(identity):'a'*64})),self.assertRaisesRegex(ValueError,'reference changed'):verifier.verify(reference,raw,root/'out',profile='corpus')
    with patch.object(generation,'authenticate_generation_reference',return_value=({**data,'profile':'short'},{})),self.assertRaisesRegex(ValueError,'reference profile'):verifier.verify(reference,raw,root/'out',profile='corpus')
    self.assertFalse((root/'out').exists())

if __name__=='__main__':unittest.main()
