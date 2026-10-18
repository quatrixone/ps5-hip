import copy,json,struct,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_generation_numeric_test as numeric
import build_llm_generation_reference as reference
import source_llm_generation_log as transcript
from llm_generation_numeric_test import NumericGeneration

class SparseComparison(NumericGeneration):
 def setUp(self):
  super().setUp()
  for folder in (self.raw,self.ref):
   metadata={'prompt_tokens':[1],'runs':[{'run':r,'processed':[1,2],'generated':[2,3]} for r in range(2)]}
   (folder/'generation.json').write_text(json.dumps(metadata))
   for run in range(2):(folder/f'run-{run}-pos-1-logits.f32le').write_bytes(bytes(12)+struct.pack('<f',1)+bytes(128000-16))
  counters=dict(requests=4,nodes=4*798,launches=4*733,CPU_arithmetic=0,scheduler_forwarded=4,allocations=16,frees=16,instances_created=1,instances_freed=1)
  (self.raw/'backend.json').write_text(json.dumps(counters))
 def compare(self):return numeric.compare_generation(self.raw,self.ref,checkpoints=(0,))
 def test_feedback_metadata(self):
  path=self.raw/'generation.json';data=json.loads(path.read_text());data['runs'][0]['generated'][1]=4;path.write_text(json.dumps(data))
  with self.assertRaisesRegex(ValueError,'feedback'):self.compare()
 def test_pass_inventory_and_budget(self):
  rows,files,count=self.compare();self.assertEqual((len(files),count),(50,4));self.assertTrue(all(row['pass_budget'] for row in rows))
  self.assertEqual((numeric.ABS_TOL,numeric.REL_TOL,numeric.L2_TOL),(2e-5,2e-4,2e-4))
 def test_noncheckpoint_logits_required(self):
  (self.raw/'run-0-pos-1-logits.f32le').unlink()
  with self.assertRaisesRegex(ValueError,'inventory'):self.compare()
 def test_noncheckpoint_intermediate_rejected(self):
  (self.raw/'run-0-pos-1-layer-0.f32le').write_bytes(bytes(8192))
  with self.assertRaisesRegex(ValueError,'inventory'):self.compare()
 def test_noncheckpoint_argmax_and_reset_checked(self):
  for run in range(2):(self.raw/f'run-{run}-pos-1-logits.f32le').write_bytes(bytes(16)+struct.pack('<f',1)+bytes(128000-20))
  with self.assertRaisesRegex(ValueError,'argmax'):self.compare()
 def test_noncheckpoint_reset_bytes_checked(self):
  path=self.raw/'run-1-pos-1-logits.f32le';content=bytearray(path.read_bytes());struct.pack_into('<f',content,0,1e-8);path.write_bytes(content)
  with self.assertRaisesRegex(ValueError,'reset'):self.compare()
 def test_noncheckpoint_numeric_tolerance_unchanged(self):
  for run in range(2):
   path=self.raw/f'run-{run}-pos-1-logits.f32le';content=bytearray(path.read_bytes());struct.pack_into('<f',content,0,.1);path.write_bytes(content)
  rows,_,_=self.compare();self.assertEqual(sum(not row['pass_budget'] for row in rows),2)
 def test_checkpoint_argument_rejected(self):
  for checkpoints in ([0],(False,),(0,0),(2,)):
   with self.subTest(checkpoints=checkpoints),self.assertRaisesRegex(ValueError,'checkpoint'):
    numeric.compare_generation(self.raw,self.ref,checkpoints=checkpoints)

class ExtendedRouting(unittest.TestCase):
 def test_native_helper_required_before_producer_work(self):
  with tempfile.TemporaryDirectory() as folder,patch.object(numeric,'ROOT',Path(folder)),patch.object(numeric,'authenticate_generation_reference',return_value=({'profile':'extended'},{})),patch.object(numeric.provenance,'cpu_inputs') as cpu:
   with self.assertRaisesRegex(ValueError,'extended numerical execution'):numeric.build(Path(folder)/'cpu',Path(folder)/'model',Path(folder)/'ref',Path(folder)/'results/new')
   cpu.assert_not_called()
 def test_extended_cases_keep_shared_checkpoints_and_totals(self):
  cases=reference.extended_cases()
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);raw=root/'raw';ref=root/'ref';raw.mkdir();ref.mkdir()
   for directory in (raw,ref):
    (directory/'extended.json').write_text('{}')
    for case in cases:(directory/case['id']).mkdir();(directory/case['id']/'generation.json').write_text('{}')
   (raw/'backend.json').write_text('{}')
   with patch.object(numeric,'validate',return_value={'requests':6,'raw_files':3,'raw_bytes':12}) as validated,patch.object(numeric,'compare_generation',return_value=([{'file':'x','pass_budget':True}],{'x':4},2)) as compared,patch.object(numeric,'check_backend',return_value={'instances_created':1,'instances_freed':1}):
    rows,files,count=numeric.compare_corpus(raw,ref,profile='extended');self.assertEqual((len(rows),len(files),count),(3,3,6))
    validated.assert_called_once_with(ref,'extended')
    self.assertEqual([call.kwargs for call in compared.call_args_list],[{'host_backend':False,'checkpoints':case['checkpoints']} for case in cases])
    (raw/'corpus.json').write_text('{}')
    with self.assertRaisesRegex(ValueError,'inventory'):numeric.compare_corpus(raw,ref,profile='extended')
 def test_actual_runner_routes_extended_without_running_model(self):
  with tempfile.TemporaryDirectory() as temporary:
   root=Path(temporary);model=root/'model';model.write_bytes(b'm');ref=root/'ref';ref.mkdir();output=root/'results/output'
   cases=reference.extended_cases();policies={case['id']:{'id':case['id']} for case in cases}
   for case in cases:
    directory=ref/'raw'/case['id'];directory.mkdir(parents=True);(directory/'generation.json').write_text(json.dumps(policies[case['id']]))
   bound={'profile':'extended','model':{'path':str(model),'bytes':1,'sha256':'a'},'receipt':'ref','receipt_sha256':'a'}
   def process(argv,**kwargs):
    label=str(argv[0]);stdout=b''
    if '-o' in argv:Path(argv[argv.index('-o')+1]).write_bytes(b'bin')
    if label.endswith('/full-numeric'):
     self.assertEqual(argv[-1],'extended');self.assertEqual(kwargs['timeout'],7200)
     raw=output/'raw';(raw/'extended.json').write_text('{}')
     for case in cases:
      directory=raw/case['id'];directory.mkdir()
      for name in ('generation.json','steps.jsonl'):(directory/name).write_text('{}')
     stdout=b'fragment\nGENERATION_NUMERIC_PASS profile=extended cases=3 runs=2 max_new_tokens=64 actual_scheduler=true CPU_arithmetic=0 live_logit_feedback=true GPU_execution=false\n'
    return types.SimpleNamespace(returncode=0,stdout=stdout,stderr=b'')
   with patch.object(numeric,'ROOT',root),patch.object(numeric,'authenticate_generation_reference',return_value=(bound,{str(model):'a'})),patch.object(numeric.provenance,'cpu_inputs',return_value={}),patch.object(numeric.provenance,'sha',return_value='a'),patch.object(numeric.provenance,'dependencies',return_value=set()),patch.object(numeric.provenance,'runtime_inputs',return_value={}),patch.object(numeric.provenance,'recheck'),patch.object(numeric.subprocess,'run',side_effect=process),patch.object(transcript,'check_extended_fragment',return_value={'status':'PASS'}) as checked,patch.object(numeric,'compare_corpus',return_value=([{'pass_budget':True}],{'x':4},1176)) as compared:
    self.assertEqual(numeric.build(root/'cpu',model,ref,output,True),0)
    checked.assert_called_once();self.assertEqual(checked.call_args.args[0],b'fragment\n');self.assertEqual(checked.call_args.args[1],policies)
    compared.assert_called_once_with(output/'raw',ref/'raw',profile='extended')
    receipt=json.loads((output/'receipt.json').read_text());self.assertEqual(receipt['profile'],'extended');self.assertIn(str(numeric.EXTENDED_CASES),receipt['inputs'])

def extended_transcript_fixture():
 cases=reference.extended_cases();policies={};sizes={'extended.json':133};lines=['M4_GEN PROFILE name=greedy-extended cases=3 runs=2 max_new_tokens=64 context=512','M3_FULL PHASE load_begin'];tick=100
 def phase(scope,run=0,key=None):
  nonlocal tick
  tag=' case='+key if key else '';lines.append(f'M4_GEN TIMING{tag} scope={scope} run={run} position=0 begin_ns={tick} end_ns={tick+2} elapsed_ns=2');tick+=3
 phase('load');lines.append('M3_FULL PHASE load_complete');phase('context');lines.append('M3_FULL PHASE context_complete')
 for case in cases:
  key=case['id'];tokens=[1]*case['prompt_count'];generated=[2] if case['stop']=='eos' else list(range(100,100+case['generated']))
  runs=[dict(run=run,processed=tokens+generated[:-1],generated=generated,stop=case['stop'],text_hex='78') for run in range(2)]
  policies[key]=dict(prompt=case['prompt'],max_new_tokens=64,prompt_tokens=tokens,eog_tokens=[2],runs=runs)
  sizes[key+'/generation.json']=500;sizes[key+'/steps.jsonl']=700
  for run,row in enumerate(runs):
   phase('reset',run,key);began=tick
   for position,token in enumerate(row['processed']):
    for stage in (reference.STAGES if position in case['checkpoints'] else ('logits',)):
     size=128000 if stage=='logits' else 8192;lines.append(f'M3_FULL RAW file={key}/run-{run}-pos-{position}-{stage}.f32le bytes={size} synced=true')
    lines.append(f'M4_GEN DECODE case={key} run={run} position={position} token={token} committed_position={position} layers=22 logits=32000 begin_ns={tick} end_ns={tick+10} elapsed_ns=10 readback_ns=3 inclusive_readback=true');tick+=12
   elapsed=tick-began;decoded=len(row['processed'])*10;lines.append(f'M4_GEN RUN_TIME case={key} run={run} begin_ns={began} end_ns={tick} elapsed_ns={elapsed} decode_ns={decoded} scheduling_sampling_log_ns={elapsed-decoded}')
   lines.append(f"M4_GEN RESULT case={key} run={run} processed={len(row['processed'])} generated={len(generated)} stop={case['stop']} text_hex=78")
  lines.append(f'M4_GEN METADATA case={key} files=2 generation_bytes=500 steps_bytes=700 synced=true')
 lines.extend(['M4_GEN EXTENDED cases=3 runs=2 max_new_tokens=64 synced=true','M3_FULL PHASE cleanup_begin']);phase('cleanup')
 lines.append(f'M4_GEN TRANSPORT_COMPLETE requests=1176 layers=22 logits=32000 nodes={1176*798} launches={1176*733} raw_files=2096 raw_bytes=158064640 metadata_files=7 metadata_bytes={sum(sizes.values())} instances=1/1 numerical=pending finalization=pending')
 return ('\n'.join(lines)+'\n').encode(),policies,sizes

class ExtendedTranscript(unittest.TestCase):
 def test_all_logits_shared_checkpoints(self):
  result=transcript.check_extended_fragment(*extended_transcript_fixture());self.assertEqual((result['requests'],result['raw_files'],result['metadata_files']),(1176,2096,7));self.assertFalse(result['numerical_verified'])
 def test_missing_unexpected_checkpoints_and_decode_rejected(self):
  raw,policies,sizes=extended_transcript_fixture()
  for old,new in [(b'run-0-pos-255-layer-0',b'run-0-pos-254-layer-0'),(b'run-0-pos-510-logits',b'run-0-pos-511-logits'),(b'raw_files=2096',b'raw_files=28224'),(b'instances=1/1',b'instances=2/2'),(b'case=eos run=0 processed=9 generated=1 stop=eos',b'case=eos run=0 processed=9 generated=1 stop=step_limit')]:
   self.assertIn(old,raw)
   with self.subTest(old=old),self.assertRaises(ValueError):transcript.check_extended_fragment(raw.replace(old,new,1),policies,sizes)
 def test_wrong_eos_feedback_and_reset_policy(self):
  raw,policies,sizes=extended_transcript_fixture()
  for field,value in [('generated',[3]),('processed',[2]*9),('stop','step_limit')]:
   bad=copy.deepcopy(policies);bad['eos']['runs'][0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):transcript.check_extended_fragment(raw,bad,sizes)

if __name__=='__main__':unittest.main()
