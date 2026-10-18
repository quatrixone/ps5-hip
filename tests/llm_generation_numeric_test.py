import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_generation_numeric_test as test

class NumericGeneration(unittest.TestCase):
 def test_corpus_not_silently_run_through_short_harness(self):
  from unittest.mock import patch
  with patch.object(test,'ROOT',self.root),patch.object(test,'authenticate_generation_reference',return_value=({'profile':'corpus'},{})),patch.object(test.provenance,'cpu_inputs') as cpu:
   with self.assertRaisesRegex(ValueError,'corpus numerical execution'):
    test.build(self.root/'cpu',self.root/'model',self.ref,self.root/'results/output')
   cpu.assert_not_called()
  self.assertFalse((self.root/'results/output').exists())
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.raw=self.root/'raw';self.ref=self.root/'reference';self.raw.mkdir();self.ref.mkdir()
  metadata={'prompt_tokens':[1],'runs':[{'run':r,'processed':[1],'generated':[2]} for r in range(2)]}
  for folder in (self.raw,self.ref):
   (folder/'generation.json').write_text(json.dumps(metadata));(folder/'steps.jsonl').write_text('steps\n')
  for r in range(2):
   for stage in [*(f'layer-{i}' for i in range(22)),'norm','logits']:
    name=f'run-{r}-pos-0-{stage}.f32le';size=128000 if stage=='logits' else 8192
    content=bytes(8)+struct.pack('<f',1.)+bytes(size-12) if stage=='logits' else bytes(size)
    (self.raw/name).write_bytes(content)
    (self.ref/name).write_bytes(content)
  (self.raw/'backend.json').write_text(json.dumps(dict(requests=2,nodes=1596,launches=1466,CPU_arithmetic=0,scheduler_forwarded=2,allocations=16,frees=16,instances_created=1,instances_freed=1)))
 def compare(self):return test.compare_generation(self.raw,self.ref)
 def test_pass_inventory_and_budget(self):
  rows,files,requests=self.compare();self.assertEqual((len(rows),len(files),requests),(48,48,2));self.assertTrue(all(r['pass_budget'] for r in rows));self.assertEqual((test.ABS_TOL,test.REL_TOL,test.L2_TOL),(2e-5,2e-4,2e-4))
 def test_feedback_metadata(self):
  p=self.raw/'generation.json';p.write_text(p.read_text().replace('"generated": [2]','"generated": [3]'))
  with self.assertRaisesRegex(ValueError,'feedback'):self.compare()
 def test_extra_inventory(self):
  (self.raw/'unexpected').write_bytes(b'')
  with self.assertRaisesRegex(ValueError,'inventory'):self.compare()
 def test_boundary_nonfinite_and_short(self):
  p=self.raw/'run-0-pos-0-layer-0.f32le';p.write_bytes(struct.pack('<f',float('nan'))+bytes(8188))
  with self.assertRaisesRegex(ValueError,'nonfinite'):self.compare()
  p.write_bytes(b'')
  with self.assertRaisesRegex(ValueError,'size'):self.compare()
 def test_reset_all_boundaries(self):
  p=self.raw/'run-1-pos-0-norm.f32le';p.write_bytes(struct.pack('<f',1e-8)+bytes(8188))
  with self.assertRaisesRegex(ValueError,'reset'):self.compare()
 def test_logit_error_fails_unchanged_gate(self):
  for r in range(2):(self.raw/f'run-{r}-pos-0-logits.f32le').write_bytes(bytes(8)+struct.pack('<f',1.)+bytes(8)+struct.pack('<f',.1)+bytes(127976))
  rows,_,_=self.compare();self.assertFalse(all(r['pass_budget'] for r in rows))
 def test_intermediate_error_fails_even_with_correct_logits(self):
  for r in range(2):(self.raw/f'run-{r}-pos-0-layer-21.f32le').write_bytes(struct.pack('<f',.1)+bytes(8188))
  rows,_,_=self.compare();self.assertEqual(sum(not row['pass_budget'] for row in rows),2)
 def test_no_cpu_or_owner_leak(self):
  p=self.raw/'backend.json';original=json.loads(p.read_text())
  for key,value in [('CPU_arithmetic',1),('frees',15),('scheduler_forwarded',1),('CPU_arithmetic',False)]:
   p.write_text(json.dumps({**original,key:value}))
   with self.subTest(key=key),self.assertRaises(ValueError):self.compare()
 def test_within_budget_near_tie_wrong_greedy(self):
  for r in range(2):
   name=f'run-{r}-pos-0-logits.f32le'
   (self.ref/name).write_bytes(bytes(8)+struct.pack('<ff',1.,1.-1e-6)+bytes(127984))
   (self.raw/name).write_bytes(bytes(8)+struct.pack('<ff',1.-1e-6,1.)+bytes(127984))
  with self.assertRaisesRegex(ValueError,'argmax'):self.compare()
 def test_symlink_boundary(self):
  p=self.raw/'run-0-pos-0-logits.f32le';p.unlink();p.symlink_to(self.ref/p.name)
  with self.assertRaisesRegex(ValueError,'kind'):self.compare()

class CorpusComparison(unittest.TestCase):
 def test_corpus_real_validation_comparison_and_inventory(self):
  import os,shutil
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);reference=root/'reference';reference.mkdir();raw=root/'raw'
   cases=test.corpus_cases();index=dict(schema=1,profile='corpus',max_new_tokens=16,cases=[key for key,_ in cases])
   (reference/'corpus.json').write_text(json.dumps(index));zero=root/'zero';zero.write_bytes(bytes(8192))
   for key,prompt in cases:
    folder=reference/key;folder.mkdir();steps=[];runs=[]
    for run in range(2):
     processed=[1]+list(range(3,18));generated=list(range(3,19))
     runs.append(dict(run=run,processed=processed,generated=generated,stop='step_limit',pieces_hex=['61']*16,text_hex='61'*16))
     for pos,token in enumerate(processed):
      values=bytearray(128000);struct.pack_into('<f',values,(pos+3)*4,1)
      (folder/f'run-{run}-pos-{pos}-logits.f32le').write_bytes(values)
      for stage in [*(f'layer-{i}' for i in range(22)),'norm']:os.link(zero,folder/f'run-{run}-pos-{pos}-{stage}.f32le')
      steps.append(dict(run=run,position=pos,token=token,status=0,committed_position=pos))
    (folder/'generation.json').write_text(json.dumps(dict(schema=1,prompt=prompt,add_special=True,parse_special=False,max_new_tokens=16,vocab_size=32000,prompt_tokens=[1],eog_tokens=[2],runs=runs)))
    (folder/'steps.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in steps))
   shutil.copytree(reference,raw,copy_function=os.link)
   counters=dict(requests=160,nodes=160*798,launches=160*733,CPU_arithmetic=0,scheduler_forwarded=160,allocations=16,frees=16,instances_created=1,instances_freed=1)
   (raw/'backend.json').write_text(json.dumps(counters))
   rows,files,requests=test.compare_corpus(raw,reference)
   self.assertEqual((len(rows),len(files),requests),(3840,3840,160));self.assertTrue(all(row['pass_budget'] for row in rows));self.assertTrue(all('/run-' in name for name in files))
   with self.assertRaisesRegex(ValueError,'inventory'):test.compare_corpus(raw,reference,host_backend=False)
   (raw/'backend.json').unlink();rows,_,_=test.compare_corpus(raw,reference,host_backend=False);self.assertEqual(len(rows),3840)
   (raw/'extra').write_text('x')
   with self.assertRaisesRegex(ValueError,'inventory'):test.compare_corpus(raw,reference,False)
   (raw/'extra').unlink();indexfile=raw/'corpus.json';indexfile.unlink();indexfile.write_text('{}')
   with self.assertRaisesRegex(ValueError,'index'):test.compare_corpus(raw,reference,False)
   indexfile.unlink();os.link(reference/'corpus.json',indexfile)
   meta=raw/cases[0][0]/'generation.json';original=meta.read_bytes();meta.unlink();meta.write_bytes(original.replace(b'\"schema\": 1',b'\"schema\": true'))
   with self.assertRaisesRegex(ValueError,'metadata'):test.compare_corpus(raw,reference,False)
   meta.write_bytes(original)
   nested=raw/cases[0][0]/'backend.json';nested.write_text('{}')
   with self.assertRaisesRegex(ValueError,'inventory'):test.compare_corpus(raw,reference,False)
   nested.unlink()
   target=raw/cases[0][0]/'run-0-pos-0-logits.f32le';target.unlink();target.symlink_to(reference/cases[0][0]/target.name)
   with self.assertRaisesRegex(ValueError,'kind'):test.compare_corpus(raw,reference,False)
 def test_aggregate_backend_checked_once(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);raw=root/'raw';reference=root/'ref';raw.mkdir();reference.mkdir()
   for base in (raw,reference):(base/'corpus.json').write_text('{}')
   (raw/'backend.json').write_text('{}')
   for key,_ in test.corpus_cases():
    for base in (raw,reference):
     (base/key).mkdir();(base/key/'generation.json').write_text('{}')
   with patch.object(test,'validate',return_value={'requests':10,'raw_files':5}) as validate,patch.object(test,'compare_generation',return_value=([{'file':'x','pass_budget':True}],{'x':4},2)) as compare,patch.object(test,'check_backend',return_value={'instances_created':1,'instances_freed':1}) as backend:
    rows,files,requests=test.compare_corpus(raw,reference)
    validate.assert_called_once_with(reference,'corpus');self.assertEqual(compare.call_count,5)
    self.assertTrue(all(call.kwargs=={'host_backend':False} for call in compare.call_args_list));backend.assert_called_once_with(raw,10)
    self.assertEqual((len(rows),len(files),requests),(5,5,10))
    backend.return_value={'instances_created':2,'instances_freed':2}
    with self.assertRaisesRegex(ValueError,'one resident'):test.compare_corpus(raw,reference)

if __name__=='__main__':unittest.main()
