import copy,json,struct,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_generation_reference as ref
class Reference(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)/'raw';self.root.mkdir()
  self.meta=dict(schema=1,prompt='Once upon a time',add_special=True,parse_special=False,max_new_tokens=4,vocab_size=32000,prompt_tokens=[1,4],eog_tokens=[2],runs=[])
  self.steps=[]
  for run in range(2):
   self.meta['runs'].append(dict(run=run,processed=[1,4,5,6,7],generated=[5,6,7,8],stop='step_limit',pieces_hex=['61','62','63','64'],text_hex='61626364'))
   for pos,t in enumerate([1,4,5,6,7]):
    values=[0.0]*32000;values[[3,5,6,7,8][pos]]=1.0
    (self.root/f'run-{run}-pos-{pos}-logits.f32le').write_bytes(struct.pack('<32000f',*values))
    for stage in ref.STAGES[:-1]:(self.root/f'run-{run}-pos-{pos}-{stage}.f32le').write_bytes(bytes(8192))
    self.steps.append(dict(run=run,position=pos,token=t,status=0,committed_position=pos))
  self.save()
 def save(self):
  (self.root/'generation.json').write_text(json.dumps(self.meta));(self.root/'steps.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in self.steps))
 def test_valid_feedback_and_reset(self):self.assertEqual(ref.validate(self.root)['generated'],[5,6,7,8])
 def test_wrong_argmax(self):
  p=self.root/'run-0-pos-1-logits.f32le';v=bytearray(p.read_bytes());struct.pack_into('<f',v,24,2);p.write_bytes(v)
  with self.assertRaisesRegex(ValueError,'argmax'):ref.validate(self.root)
 def test_tie_requires_lowest_index(self):
  for run in range(2):
   p=self.root/f'run-{run}-pos-1-logits.f32le';v=bytearray(p.read_bytes());struct.pack_into('<f',v,16,1);p.write_bytes(v)
  with self.assertRaisesRegex(ValueError,'argmax'):ref.validate(self.root)
 def test_teacher_forced_feedback_rejected(self):
  self.meta['runs'][0]['processed'][2]=9;self.save()
  with self.assertRaisesRegex(ValueError,'feedback'):ref.validate(self.root)
 def test_nonfinite_logits(self):
  p=self.root/'run-0-pos-0-logits.f32le';p.write_bytes(struct.pack('<f',float('nan'))+p.read_bytes()[4:])
  with self.assertRaisesRegex(ValueError,'nonfinite'):ref.validate(self.root)
 def test_eos_stops_without_feedback(self):
  self.meta['eog_tokens']=[5]
  self.steps=[s for s in self.steps if s['position']<2]
  for run,row in enumerate(self.meta['runs']):
   row.update(processed=[1,4],generated=[5],stop='eos',pieces_hex=['61'],text_hex='61')
   for pos in range(2,5):
    for stage in ref.STAGES:(self.root/f'run-{run}-pos-{pos}-{stage}.f32le').unlink()
  self.save();self.assertEqual(ref.validate(self.root)['stop'],'eos')
  self.meta['runs'][0]['stop']='step_limit';self.save()
  with self.assertRaisesRegex(ValueError,'EOS'):ref.validate(self.root)
 def test_reset_drift_not_hidden_by_same_argmax(self):
  p=self.root/'run-1-pos-0-logits.f32le';p.write_bytes(struct.pack('<f',0.1)+p.read_bytes()[4:])
  with self.assertRaisesRegex(ValueError,'reset'):ref.validate(self.root)
 def test_commit_and_numeric_boolean_rejected(self):
  for key in ('run','position','token','status','committed_position'):
   original=copy.deepcopy(self.steps);self.steps[0][key]=False;self.save()
   with self.subTest(key=key),self.assertRaises(ValueError):ref.validate(self.root)
   self.steps=original
  for key in ('schema','max_new_tokens','vocab_size'):
   original=self.meta[key];self.meta[key]=True;self.save()
   with self.subTest(key=key),self.assertRaises(ValueError):ref.validate(self.root)
   self.meta[key]=original
 def test_duplicate_keys(self):
  p=self.root/'generation.json';p.write_text(p.read_text().replace('"schema": 1','"schema": 1, "schema": 1'))
  with self.assertRaisesRegex(ValueError,'duplicate'):ref.validate(self.root)
 def test_extra_missing_and_symlink_inventory(self):
  p=self.root/'extra';p.write_text('x')
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate(self.root)
  p.unlink();p=self.root/'run-0-pos-0-logits.f32le';p.unlink();p.symlink_to(self.root/'run-1-pos-0-logits.f32le')
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate(self.root)
 def test_raw_directory_symlink(self):
  p=self.root.parent/'alias';p.symlink_to(self.root,target_is_directory=True)
  with self.assertRaisesRegex(ValueError,'directory'):ref.validate(p)
 def test_text_and_malformed_schema(self):
  self.meta['runs'][0]['text_hex']='00';self.save()
  with self.assertRaisesRegex(ValueError,'decoded text'):ref.validate(self.root)
  for bad in (None,[],3):
   (self.root/'generation.json').write_text(json.dumps(bad))
   with self.assertRaises(ValueError):ref.validate(self.root)
 def test_all_boundaries_count_size_finite_and_reset(self):
  summary=ref.validate(self.root);self.assertEqual((summary['raw_files'],summary['raw_bytes']),(240,10*(23*8192+128000)))
  p=self.root/'run-1-pos-0-layer-21.f32le';original=p.read_bytes()
  for content,error in ((b'', 'size'),(struct.pack('<f',float('nan'))+original[4:], 'nonfinite'),(struct.pack('<f',1.)+original[4:], 'reset')):
   p.write_bytes(content)
   with self.subTest(error=error),self.assertRaisesRegex(ValueError,error):ref.validate(self.root)
  p.write_bytes(original);p.unlink()
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate(self.root)
 def test_exact_compiler_commands(self):
  cpu=Path('/cpu');model=Path('/model');producer=ref.ROOT/'results/test-producer';plan=ref.command_plan(producer,cpu,model)
  rows=[dict(label=k,argv=v,exit_code=0) for k,v in plan.items()];p=self.root/'commands.json';p.write_text(json.dumps(rows));ref.commands_valid({'commands':rows,'producer_output':str(producer)},self.root,cpu,model)
  for label in ('headers','compile','ldd','capture'):
   bad=copy.deepcopy(rows);next(x for x in bad if x['label']==label)['argv'].append('-DUNREVIEWED=1');p.write_text(json.dumps(bad))
   with self.subTest(label=label),self.assertRaisesRegex(ValueError,'exact command'):ref.commands_valid({'commands':bad,'producer_output':str(producer)},self.root,cpu,model)
  bad=copy.deepcopy(rows);bad[0]['exit_code']=False;p.write_text(json.dumps(bad))
  with self.assertRaisesRegex(ValueError,'schema'):ref.commands_valid({'commands':bad,'producer_output':str(producer)},self.root,cpu,model)
 def test_artifacts_rechecked_after_authentication(self):
  receipt=self.root/'receipt.json';receipt.write_text('{}');before=ref.artifact_inventory(self.root,receipt);digest=ref.provenance.sha(receipt)
  ref.final_artifact_check(self.root,receipt,before,digest)
  raw=self.root/'run-0-pos-0-logits.f32le';raw.write_bytes(b'X'+raw.read_bytes()[1:])
  with self.assertRaisesRegex(ValueError,'changed during'):ref.final_artifact_check(self.root,receipt,before,digest)
class Corpus(unittest.TestCase):
 def test_fixed_cases_commands_and_profile_bounds(self):
  cases=ref.corpus_cases();self.assertEqual(len(cases),5);self.assertTrue(any(not text.isascii() for _,text in cases))
  short=ref.command_plan(Path('/out'),Path('/cpu'),Path('/model'))
  corpus=ref.command_plan(Path('/out'),Path('/cpu'),Path('/model'),'corpus')
  self.assertEqual(corpus['capture'],short['capture']+['corpus'])
  self.assertEqual(corpus['compile'],short['compile'])
  self.assertEqual(ref.profile_policy('corpus')[2],128*1024*1024)
  with self.assertRaisesRegex(ValueError,'profile'):ref.command_plan(Path('/out'),Path('/cpu'),Path('/model'),'unknown')
 def test_corpus_all_boundaries_and_rejections(self):
  import os
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'raw';root.mkdir();cases=ref.corpus_cases()
   index=dict(schema=1,profile='corpus',max_new_tokens=16,cases=[key for key,_ in cases])
   (root/'corpus.json').write_text(json.dumps(index))
   zeros=Path(tmp)/'zeros';zeros.write_bytes(bytes(8192))
   for key,prompt in cases:
    folder=root/key;folder.mkdir();steps=[];runs=[]
    for run in range(2):
     processed=[1]+list(range(3,18));generated=list(range(3,19))
     runs.append(dict(run=run,processed=processed,generated=generated,stop='step_limit',pieces_hex=['61']*16,text_hex='61'*16))
     for pos,token in enumerate(processed):
      logits=bytearray(128000);struct.pack_into('<f',logits,(pos+3)*4,1)
      (folder/f'run-{run}-pos-{pos}-logits.f32le').write_bytes(logits)
      for stage in ref.STAGES[:-1]:os.link(zeros,folder/f'run-{run}-pos-{pos}-{stage}.f32le')
      steps.append(dict(run=run,position=pos,token=token,status=0,committed_position=pos))
    meta=dict(schema=1,prompt=prompt,add_special=True,parse_special=False,max_new_tokens=16,vocab_size=32000,prompt_tokens=[1],eog_tokens=[2],runs=runs)
    (folder/'generation.json').write_text(json.dumps(meta));(folder/'steps.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in steps))
   summary=ref.validate(root,'corpus');self.assertEqual((summary['requests'],summary['raw_files'],summary['raw_bytes']),(160,3840,160*316416))
   first=root/cases[0][0];p=first/'generation.json';original=p.read_text();meta=json.loads(original)
   meta['prompt']='different';p.write_text(json.dumps(meta))
   with self.assertRaisesRegex(ValueError,'metadata'):ref.validate(root,'corpus')
   meta=json.loads(original);meta['eog_tokens']=[18];meta['runs'][0]['stop']='eos';p.write_text(json.dumps(meta))
   with self.assertRaisesRegex(ValueError,'sixteen'):ref.validate(root,'corpus')
   p.write_text(original);index['cases'].reverse();(root/'corpus.json').write_text(json.dumps(index))
   with self.assertRaisesRegex(ValueError,'metadata'):ref.validate(root,'corpus')
   index['cases'].reverse();index['schema']=True;(root/'corpus.json').write_text(json.dumps(index))
   with self.assertRaisesRegex(ValueError,'metadata'):ref.validate(root,'corpus')
 def test_corpus_capture_cannot_be_authenticated_as_short(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);producer=ref.ROOT/'results/test-corpus';cpu=Path('/cpu');model=Path('/model')
   plan=ref.command_plan(producer,cpu,model,'corpus');rows=[dict(label=k,argv=v,exit_code=0) for k,v in plan.items()]
   (folder/'commands.json').write_text(json.dumps(rows));data=dict(commands=rows,producer_output=str(producer),profile='corpus')
   ref.commands_valid(data,folder,cpu,model);del data['profile']
   with self.assertRaisesRegex(ValueError,'exact command'):ref.commands_valid(data,folder,cpu,model)

if __name__=='__main__':unittest.main()
