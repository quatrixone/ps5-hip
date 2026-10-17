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
   for pos in range(2,5):(self.root/f'run-{run}-pos-{pos}-logits.f32le').unlink()
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
 def test_exact_compiler_commands(self):
  cpu=Path('/cpu');model=Path('/model');plan=ref.command_plan(self.root,cpu,model)
  rows=[dict(label=k,argv=v,exit_code=0) for k,v in plan.items()];p=self.root/'commands.json';p.write_text(json.dumps(rows));ref.commands_valid({'commands':rows},self.root,cpu,model)
  for label in ('headers','compile','ldd','capture'):
   bad=copy.deepcopy(rows);next(x for x in bad if x['label']==label)['argv'].append('-DUNREVIEWED=1');p.write_text(json.dumps(bad))
   with self.subTest(label=label),self.assertRaisesRegex(ValueError,'exact command'):ref.commands_valid({'commands':bad},self.root,cpu,model)
  bad=copy.deepcopy(rows);bad[0]['exit_code']=False;p.write_text(json.dumps(bad))
  with self.assertRaisesRegex(ValueError,'schema'):ref.commands_valid({'commands':bad},self.root,cpu,model)
 def test_artifacts_rechecked_after_authentication(self):
  receipt=self.root/'receipt.json';receipt.write_text('{}');before=ref.artifact_inventory(self.root,receipt);digest=ref.provenance.sha(receipt)
  ref.final_artifact_check(self.root,receipt,before,digest)
  raw=self.root/'run-0-pos-0-logits.f32le';raw.write_bytes(b'X'+raw.read_bytes()[1:])
  with self.assertRaisesRegex(ValueError,'changed during'):ref.final_artifact_check(self.root,receipt,before,digest)
if __name__=='__main__':unittest.main()
