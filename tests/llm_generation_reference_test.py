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
class PromptCapacity(unittest.TestCase):
 def test_combined_budget_before_raw_reads(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);(folder/'steps.jsonl').write_text('')
   for count,message in ((33,'reset runs'),(508,'reset runs'),(509,'context budget'),(512,'context budget')):
    metadata=dict(schema=1,prompt='Once upon a time',add_special=True,parse_special=False,max_new_tokens=4,vocab_size=32000,prompt_tokens=[1]*count,eog_tokens=[2],runs=[])
    (folder/'generation.json').write_text(json.dumps(metadata))
    # Deliberately invalid runs isolate prompt admission without manufacturing model outputs.
    with self.subTest(count=count),patch.object(Path,'read_bytes',side_effect=AssertionError('raw read before policy check')),self.assertRaisesRegex(ValueError,message):ref.validate_case(folder,'Once upon a time',4)
 def test_profile_policy_unchanged(self):
  self.assertEqual(ref.profile_policy('short')[0],4);self.assertEqual(ref.profile_policy('corpus')[0],16)
  with self.assertRaisesRegex(ValueError,'profile'):ref.profile_policy('long')

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

class ExtendedPolicy(unittest.TestCase):
 def test_exact_capture_inventory_and_bounds(self):
  cases=ref.extended_cases();totals=[0,0,0]
  for case in cases:
   commits=case['prompt_count']+case['generated']-1
   files=2*(commits+23*len(case['checkpoints']));size=2*(commits*128000+len(case['checkpoints'])*23*8192)
   totals=[a+b for a,b in zip(totals,(commits*2,files,size))]
   self.assertLess(max(case['checkpoints']),commits)
  self.assertEqual(totals,[1176,2096,158064640]);self.assertEqual(cases[-1]['checkpoints'][-1],510)
  self.assertEqual(ref.profile_policy('extended'),(64,ref.EXTENDED_LINE,256*1024*1024))
  command=ref.command_plan(Path('/out'),Path('/cpu'),Path('/model'),'extended')
  self.assertEqual(command['capture'],['/out/trace','/model','/out/raw','extended'])
 def test_cpp_and_python_policy_agree(self):
  import shutil,subprocess
  compiler=shutil.which('clang++-18') or shutil.which('clang++');self.assertIsNotNone(compiler)
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp)/'policy.cpp';binary=Path(tmp)/'policy'
   source.write_text('#include "extended_generation.h"\n#include <cstdio>\nint main(){for(const auto& c:ps5hip_generation::extended_cases){std::printf("%s|%zu|%zu|%zu|%s|",c.id,c.prompt_count,c.budget,c.generated,c.stop);for(auto p:c.checkpoints){if(!ps5hip_generation::extended_checkpoint(c,p))return 1;std::printf("%u,",p);}std::printf("|%s\\n",c.prompt);}return 0;}\n')
   subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ref.ROOT/'ports/ggml'),str(source),'-o',str(binary)],check=True,timeout=30)
   rows=subprocess.check_output([str(binary)],text=True,timeout=10).splitlines()
  expected=[f"{c['id']}|{c['prompt_count']}|{c['budget']}|{c['generated']}|{c['stop']}|"+''.join(str(p)+',' for p in c['checkpoints'])+'|'+c['prompt'] for c in ref.extended_cases()]
  self.assertEqual(rows,expected)
 def test_native_extended_still_requires_parents_and_watchdog(self):
  import public_hip_full_assets,build_llm_native_app
  with self.assertRaisesRegex(ValueError,'exact five module parents required'):public_hip_full_assets.assemble({},Path('/unused'),'extended')
  model=dict(sha256='unchanged',bytes=1)
  with self.assertRaisesRegex(ValueError,'watchdog not qualified'):build_llm_native_app.validate_full_profile(dict(profile='extended',model=model),model)
 def test_fixed_checkpoint_definition_rejects_changes(self):
  from unittest.mock import patch
  original=ref.EXTENDED_CASES.read_text()
  with tempfile.TemporaryDirectory() as tmp:
   policy=Path(tmp)/'cases.def';policy.write_text(original.replace('449, 510','449, 511'))
   with patch.object(ref,'EXTENDED_CASES',policy),self.assertRaisesRegex(ValueError,'fixed policy'):ref.extended_cases()
 def test_case_symlink_rejected_before_metadata_read(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp)/'raw';folder.mkdir();outside=Path(tmp)/'outside';outside.mkdir()
   (folder/'extended.json').write_text(json.dumps(ref.extended_index()))
   for case in ref.extended_cases():
    target=folder/case['id']
    if case['id']=='short64':target.symlink_to(outside,target_is_directory=True)
    else:target.mkdir()
   read=Path.read_text
   def checked(path,*args,**kwargs):
    if path.name=='generation.json':self.fail('metadata read through case symlink')
    return read(path,*args,**kwargs)
   with patch.object(Path,'read_text',checked),self.assertRaisesRegex(ValueError,'case directory'):ref.validate(folder,'extended')
   (folder/'short64').unlink();(folder/'short64').write_text('not a directory')
   with patch.object(Path,'read_text',checked),self.assertRaisesRegex(ValueError,'case directory'):ref.validate(folder,'extended')
 def test_invalid_metadata_containers_raise_value_error(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);(folder/'extended.json').write_text(json.dumps(ref.extended_index()))
   for case in ref.extended_cases():
    sub=folder/case['id'];sub.mkdir();(sub/'steps.jsonl').write_text('')
   base=dict(prompt_tokens=[1]*5,eog_tokens=[2],runs=[dict(generated=[3]*64,stop='step_limit') for _ in range(2)])
   values=[[],None,{**base,'prompt_tokens':None},{**base,'runs':None},{**base,'runs':[None,None]},{**base,'runs':[dict(generated=None,stop='step_limit'),base['runs'][1]]}]
   for metadata in values:
    (folder/'short64'/'generation.json').write_text(json.dumps(metadata))
    with self.subTest(metadata=metadata),self.assertRaises(ValueError):ref.validate(folder,'extended')
 def test_extended_metadata_dispatch_and_stop_rejection(self):
  from unittest.mock import patch
  cases=ref.extended_cases()
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);(folder/'extended.json').write_text(json.dumps(ref.extended_index()))
   for c in cases:
    sub=folder/c['id'];sub.mkdir();(sub/'steps.jsonl').write_text('')
    (sub/'generation.json').write_text(json.dumps(dict(prompt_tokens=[1]*c['prompt_count'],eog_tokens=[2],runs=[dict(generated=[2 if c['stop']=='eos' else 3]*c['generated'],stop=c['stop']) for _ in range(2)])))
   def summary(sub,prompt,budget,checkpoints):
    c=next(c for c in cases if c['id']==sub.name);self.assertEqual((prompt,budget,checkpoints),(c['prompt'],64,c['checkpoints']))
    requests=2*(c['prompt_count']+c['generated']-1)
    return dict(requests=requests,raw_files=requests+46*len(checkpoints),raw_bytes=requests*128000+46*len(checkpoints)*8192)
   with patch.object(ref,'validate_case',side_effect=summary):self.assertEqual(ref.validate(folder,'extended')['raw_files'],2096)
   for key,count in (('short64',63),('short64',65),('eos',2),('boundary',63)):
    path=folder/key/'generation.json';original=path.read_text();metadata=json.loads(original);metadata['runs'][0]['generated']=[3]*count;path.write_text(json.dumps(metadata))
    with self.subTest(case=key,count=count),patch.object(ref,'validate_case',side_effect=summary),self.assertRaisesRegex(ValueError,'stop/count'):ref.validate(folder,'extended')
    path.write_text(original)
   (folder/'extended.json').write_text(json.dumps({**ref.extended_index(),'schema':True}))
   with self.assertRaisesRegex(ValueError,'index'):ref.validate(folder,'extended')
   (folder/'extended.json').write_text(json.dumps(ref.extended_index()))
   sub=folder/'eos';metadata=json.loads((sub/'generation.json').read_text());metadata['runs'][0]['stop']='step_limit';(sub/'generation.json').write_text(json.dumps(metadata))
   with patch.object(ref,'validate_case',side_effect=summary),self.assertRaisesRegex(ValueError,'stop/count'):ref.validate(folder,'extended')

class CheckpointReference(unittest.TestCase):
 setUp=Reference.setUp
 save=Reference.save
 def selected(self):
  for run in range(2):
   for pos in (1,2,3):
    for stage in ref.STAGES[:-1]:(self.root/f'run-{run}-pos-{pos}-{stage}.f32le').unlink()
 def test_sparse_intermediates_keep_every_logit_and_reset(self):
  self.selected();result=ref.validate_case(self.root,'Once upon a time',4,(0,4));self.assertEqual(result['raw_files'],102)
  path=self.root/'run-1-pos-0-layer-0.f32le';path.write_bytes(struct.pack('<f',0.1)+path.read_bytes()[4:])
  with self.assertRaisesRegex(ValueError,'reset'):ref.validate_case(self.root,'Once upon a time',4,(0,4))
 def test_missing_logit_rejected_even_without_intermediates(self):
  self.selected();(self.root/'run-0-pos-2-logits.f32le').unlink()
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate_case(self.root,'Once upon a time',4,(0,4))
 def test_unexpected_intermediate_rejected(self):
  self.selected();(self.root/'run-0-pos-2-layer-0.f32le').write_bytes(bytes(8192))
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate_case(self.root,'Once upon a time',4,(0,4))
 def test_missing_checkpoint_rejected(self):
  self.selected();(self.root/'run-0-pos-4-layer-0.f32le').unlink()
  with self.assertRaisesRegex(ValueError,'inventory'):ref.validate_case(self.root,'Once upon a time',4,(0,4))
 def test_checkpoint_outside_history_rejected(self):
  with self.assertRaisesRegex(ValueError,'checkpoints'):ref.validate_case(self.root,'Once upon a time',4,(0,511))

if __name__=='__main__':unittest.main()
