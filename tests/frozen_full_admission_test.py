import importlib.util,json,os,pathlib,shutil,struct,sys,tempfile,unittest
from unittest.mock import patch,Mock
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import frozen_full_admission as f
import frozen_full_readbacks as reads
import source_console as control
import source_model_freeze as producer
from external_model import MODEL_BYTES,MODEL_SHA,DEFAULT_MODEL_PATH
from build_llm_full_numeric_test import FILES
from build_llm_generation_reference import corpus_cases,STAGES

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
def raw_fixture(folder,profile):
 folder.mkdir(parents=True);zero=folder.parent/'zero';zero.write_bytes(bytes(8192))
 if profile=='smoke':
  for name,size in FILES.items():(folder/name).write_bytes(bytes(size))
 else:
  cases=corpus_cases() if profile=='corpus' else [('short','Once upon a time')]
  counts=[5,9,21,15,11] if profile=='corpus' else [5];budget=16 if profile=='corpus' else 4
  if profile=='corpus':dump(folder/'corpus.json',dict(schema=1,profile='corpus',max_new_tokens=16,cases=[n for n,_ in cases]))
  for (name,prompt),n in zip(cases,counts):
   target=folder/name if profile=='corpus' else folder;target.mkdir(exist_ok=True);runs=[];steps=[]
   for run in range(2):
    processed=[1]*n+list(range(3,3+budget-1));generated=list(range(3,3+budget))
    runs.append(dict(run=run,processed=processed,generated=generated,stop='step_limit',pieces_hex=['61']*budget,text_hex='61'*budget))
    for pos,token in enumerate(processed):
     for stage in STAGES:
      path=target/f'run-{run}-pos-{pos}-{stage}.f32le'
      if stage=='logits':
       value=bytearray(128000);struct.pack_into('<f',value,(pos-n+4 if pos>=n-1 else 0)*4,1);path.write_bytes(value)
      else:os.link(zero,path)
     steps.append(dict(run=run,position=pos,token=token,status=0,committed_position=pos))
   dump(target/'generation.json',dict(schema=1,prompt=prompt,add_special=True,parse_special=False,max_new_tokens=budget,vocab_size=32000,prompt_tokens=[1]*n,eog_tokens=[2],runs=runs))
   (target/'steps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in steps))
 zero.unlink();return {p.relative_to(folder).as_posix():p.stat().st_size for p in folder.rglob('*') if p.is_file()}

class Integration(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name)
  self.addCleanup(patch.stopall)
  patch.object(f,'runtime_identities',side_effect=lambda root:{'tools/run.py':f.digest(root/'tools/run.py')}).start()
  self.auth=patch.object(producer,'validate_model_candidate',return_value={}).start()
  self.root.joinpath('tools').mkdir();self.root.joinpath('tools/run.py').write_text('fixed runner')
 def fixture(self,profile='smoke',before_emit=None):
  root=self.root;c=root/'candidate';c.mkdir();app=c/'package/app';app.mkdir(parents=True)
  for name,content in {'client-phase.txt':'model-full\n','results-dir.txt':'/user/homebrew/hip','model-path.txt':DEFAULT_MODEL_PATH,'vocab.gguf':'vocab','eboot.bin':'native'}.items():(app/name).write_text(content)
  (app/'model-layer').mkdir()
  for name in producer.LAYER_MODULES:(app/'model-layer'/(name+'.hsaco')).write_text(name)
  package=c/'package/PPSA99014.ffpkg';package.write_bytes(b'package')
  for directory,name in [('controllers','launch.elf'),('controllers','close.elf'),('snapshot','snapshot.elf')]:
   p=c/directory/name;p.parent.mkdir(exist_ok=True);p.write_bytes(name.encode())
  original=root/'original';original.mkdir();model=dict(path='/gone/model',bytes=MODEL_BYTES,sha256=MODEL_SHA)
  bound={'phase':'model-full','model':dict(model,asset='/gone/model'),'full':{'profile':profile},'native_client':{},'vocabulary':{}}
  for name,key in [('aggregate','full'),('client','native_client'),('model','model'),('vocab','vocabulary')]:
   p=original/(name+'.json');dump(p,{'provenance':name});bound[key].update(receipt=str(p),receipt_sha256=f.digest(p))
  audit=dict(status='PASS',classification='offline-public-hip-llm-client-native-image-audit',phase='model-full',full_profile=profile,wrapper_retained=True,wrapper_no_bypass=True,actual_client_call_chain_audited=True,actual_full_call_chain_audited=True,actual_generation_call_chain_audited=True,actual_client_gate_executed=False,native_executed=False,gpu_executed=False)
  dump(c/'llm-client-native-audit.json',audit);bound['audit']={'sha256':f.digest(c/'llm-client-native-audit.json')}
  external=dict(path=DEFAULT_MODEL_PATH,bytes=MODEL_BYTES,sha256=MODEL_SHA)
  files=raw_fixture(c/'full-reference/raw',profile);oracle=dict(status='PASS',gpu_executed=False,model=model)
  oracle.update(classification='successful-full-model-cpu-boundary-reference' if profile=='smoke' else 'successful-greedy-generation-cpu-reference',requests=514 if profile=='smoke' else 272 if profile=='corpus' else 16,raw_files=168 if profile=='smoke' else 6528 if profile=='corpus' else 384)
  if profile!='smoke':oracle['profile']='corpus' if profile=='corpus' else 'short'
  dump(c/'full-reference/receipt.json',oracle)
  packagefiles={p.relative_to(app).as_posix():f.digest(p) for p in app.rglob('*') if p.is_file()}
  transfer=dict(bytes=7,sha256=f.digest(package),deadline_seconds=1800)
  m=dict(fixture='llm_client',host='192.168.4.40',title_id='PPSA99014',llm_client=bound,external_model=external,package='candidate/package/PPSA99014.ffpkg',app='candidate/package/app',launch='candidate/controllers/launch.elf',close='candidate/controllers/close.elf',snapshot='candidate/snapshot/snapshot.elf',post_snapshot='candidate/snapshot/snapshot.elf',results_dir='/user/homebrew/hip',baseline_results_dir='/user/homebrew/prior',baseline_log_fixture='llm_client',baseline_image_sha256=transfer['sha256'],baseline_log_sha256=None,baseline_log_preserved=False,model_transfer=transfer,baseline_transfer=transfer,package_files=packagefiles,files={},deadline_seconds=1830 if profile=='corpus' else 150,control_phase_seconds=4200 if profile=='corpus' else 1800)
  if profile=='corpus':m['raw_capture_seconds']=600
  m['model_full']=dict(profile=profile,reference='candidate/full-reference',reference_receipt_sha256=f.digest(c/'full-reference/receipt.json'),raw_files=files)
  m['model_full']['module_sha256']={name:f.digest(app/'model-layer'/(name+'.hsaco')) for name in producer.LAYER_MODULES}
  receipt=dict(status='PASS',classification='source-only-model-full-native-client-app',phase='model-full',fixture='llm_client',native_executed=False,gpu_executed=False,llm_client=bound,external_model=external,package_sha256=transfer['sha256'],package_bytes=7,package_files=packagefiles)
  dump(c/'build-receipt.json',receipt);m['files']={str(p.relative_to(root)):f.digest(p) for p in c.rglob('*') if p.is_file()};m['files'].update(f.runtime_identities(root));
  if before_emit:before_emit(m)
  m=f.emit(c,m,root);dump(c/'manifest.json',m)
  dump(c/'review.json',dict(approved=True,manifest_sha256=f.digest(c/'manifest.json')))
  return c,m,original
 def test_smoke_relocation_real_controller_and_frozen_result(self):
  c,m,original=self.fixture();self.auth.assert_called_once();shutil.rmtree(original)
  destination=self.root.parent/(self.root.name+'-moved');shutil.move(self.root,destination);self.addCleanup(shutil.rmtree,destination);self.root=destination;c=destination/'candidate'
  with patch.object(control,'ROOT',destination),patch.object(control,'validate_model_candidate',side_effect=AssertionError('runtime producer')):
   obj=control.Control(c,'deploy');obj.verify();obj.authenticate_llm_client()
   raw=obj.out/'raw';shutil.copytree(c/'full-reference/raw',raw);obj.record['prefix_raw_files']={n:dict(sha256=f.digest(raw/n)) for n in m['model_full']['raw_files']}
   obj.verify_prefix_dumps();self.assertEqual(obj.record['prefix_numerical']['status'],'PASS')
   result=json.loads((obj.out/'prefix-numerical.json').read_text());self.assertEqual((result['raw_files'],result['requests']),(72,3));self.assertFalse(result['gpu_origin_established'])
 def test_generation_frozen_result_no_producer(self):
  c,m,original=self.fixture('generation');shutil.rmtree(original);raw=self.root/'raw';shutil.copytree(c/'full-reference/raw',raw)
  out=self.root/'result.json';result=reads.verify_frozen(f.FrozenReference(c,m,self.root),raw,out)
  self.assertEqual((result['raw_files'],result['requests']),(386,16))
 def test_corpus_frozen_result_no_producer(self):
  c,m,original=self.fixture('corpus');shutil.rmtree(original);raw=self.root/'raw';shutil.copytree(c/'full-reference/raw',raw,copy_function=os.link)
  result=reads.verify_frozen(f.FrozenReference(c,m,self.root),raw,self.root/'result.json')
  self.assertEqual((result['raw_files'],result['requests']),(6539,272))
 def test_runtime_and_consumed_mutation(self):
  c,m,_=self.fixture();p=self.root/'tools/run.py';p.write_text('changed')
  with self.assertRaisesRegex(ValueError,'runtime'):f.validate(c,m,self.root)
  p.write_text('fixed runner');(c/'controllers/launch.elf').write_text('changed')
  with self.assertRaisesRegex(ValueError,'artifact'):f.validate(c,m,self.root)
 def test_unapproved_and_changed_manifest_control(self):
  c,m,_=self.fixture();dump(c/'review.json',dict(approved=False))
  with patch.object(control,'ROOT',self.root),self.assertRaisesRegex(RuntimeError,'independent review'):control.Control(c,'deploy')
 def test_no_emission_after_failed_producer(self):
  self.auth.side_effect=ValueError('producer fails')
  with self.assertRaisesRegex(ValueError,'producer fails'):self.fixture()
  self.assertFalse((self.root/'candidate/artifact-admission.json').exists())
 def test_no_runtime_original_model_upload_path(self):
  c,m,_=self.fixture();obj=control.Control.__new__(control.Control);obj.manifest=m;obj.phase='deploy';obj.record={};obj.token='ours';obj.model_file=None
  obj.ensure_directory=Mock();obj.external_model_entry=Mock(return_value=False);obj.model_upload=Mock(side_effect=AssertionError('upload reached'))
  with self.assertRaisesRegex(RuntimeError,'explicit local model'):obj.provision_external_model()
  obj.model_upload.assert_not_called()
 def test_real_run_final_authentication_after_cleanup(self):
  c,m,original=self.fixture();shutil.rmtree(original)
  class FTP:
   def connect(self,*a,**k):pass
   def login(self,*a):pass
   def sendcmd(self,*a):return '226 SELF transfer mode disabled'
   def close(self):pass
  lock=self.root/'lock'
  with patch.object(control,'ROOT',self.root),patch.object(control,'LOCK',lock),patch.object(control.ftplib,'FTP',FTP),patch.object(control,'validate_model_candidate',side_effect=AssertionError('producer read')):
   obj=control.Control(c,'deploy');obj.health=Mock(return_value=True);obj.deploy=lambda:obj.record.update(accepted=True,promoted=True)
   self.assertEqual(obj.run(defer_prefix_authentication=True),2);self.assertFalse(lock.exists());self.assertFalse(obj.record['accepted'])
   obj.record['secondary_lock_released']=True;self.assertEqual(obj.finish_prefix_authentication(),0);self.assertTrue(obj.record['accepted'])
 def test_descriptor_approval_and_semantic_audit_reject(self):
  c,m,_=self.fixture();dump(c/'review.json',dict(approved=False))
  with self.assertRaisesRegex(ValueError,'reviewed manifest'):f.FrozenReference(c,m,self.root)
  audit=c/'llm-client-native-audit.json';value=json.loads(audit.read_text());value['actual_full_call_chain_audited']=False;dump(audit,value)
  attpath=c/'artifact-admission.json';att=json.loads(attpath.read_text());att['files']['llm-client-native-audit.json']=dict(bytes=audit.stat().st_size,sha256=f.digest(audit));dump(attpath,att)
  m['artifact_admission']['sha256']=f.digest(attpath);m['files']['candidate/artifact-admission.json']=f.digest(attpath);m['files']['candidate/llm-client-native-audit.json']=f.digest(audit)
  with self.assertRaisesRegex(ValueError,'audit call chain'):f.validate(c,m,self.root)
 def test_final_consumed_mutation_rejects_after_locks(self):
  c,m,original=self.fixture();shutil.rmtree(original)
  class FTP:
   def connect(self,*a,**k):pass
   def login(self,*a):pass
   def sendcmd(self,*a):return '226 SELF transfer mode disabled'
   def close(self):pass
  lock=self.root/'lock'
  with patch.object(control,'ROOT',self.root),patch.object(control,'LOCK',lock),patch.object(control.ftplib,'FTP',FTP):
   obj=control.Control(c,'deploy');obj.health=Mock(return_value=True);obj.deploy=lambda:obj.record.update(accepted=True,promoted=True)
   self.assertEqual(obj.run(defer_prefix_authentication=True),2);self.assertFalse(lock.exists());obj.record['secondary_lock_released']=True
   (c/'package/PPSA99014.ffpkg').write_bytes(b'changed')
   self.assertEqual(obj.finish_prefix_authentication(),1);self.assertFalse(obj.record['accepted']);self.assertTrue(obj.record['promoted'])
 def test_weakened_attestation_budget_and_escape_reject(self):
  c,m,_=self.fixture();attpath=c/'artifact-admission.json';att=json.loads(attpath.read_text());att['budgets']['absolute']=1;dump(attpath,att);m['artifact_admission']['sha256']=f.digest(attpath)
  with self.assertRaisesRegex(ValueError,'attestation contract'):f.validate(c,m,self.root)
  m['launch']='../outside'
  with self.assertRaisesRegex(ValueError,'noncanonical path'):f.validate(c,m,self.root)
 def test_profile_relabel_rejected(self):
  c,m,_=self.fixture();m['model_full']['profile']='generation'
  with self.assertRaisesRegex(ValueError,'attestation contract'):f.validate(c,m,self.root)
 def test_new_cli_options_declared(self):
  import subprocess
  env=dict(os.environ,PYTHONPATH=str(ROOT/'tools'))
  for name,option in [('freeze_source_candidate.py','--frozen-artifact-admission'),('source_console.py','--model-file')]:
   run=subprocess.run([sys.executable,str(ROOT/'tools'/name),'--help'],capture_output=True,text=True,env=env,timeout=10)
   self.assertEqual(run.returncode,0,run.stderr);self.assertIn(option,run.stdout)
 def test_issuer_rejects_post_auth_controller_mutation(self):
  def mutate(candidate,manifest):(candidate/'controllers/launch.elf').write_bytes(b'changed after authentication')
  self.auth.side_effect=mutate
  with self.assertRaisesRegex(ValueError,'candidate changed after producer authentication'):self.fixture()
 def test_issuer_rejects_post_auth_oracle_mutation(self):
  def mutate(candidate,manifest):
   path=next((candidate/'full-reference/raw').glob('*.f32le'));raw=bytearray(path.read_bytes());raw[:4]=b'\x00\x00\x80\x3f';path.write_bytes(raw)
  self.auth.side_effect=mutate
  with self.assertRaisesRegex(ValueError,'candidate changed after producer authentication'):self.fixture()
 def test_issuer_rejects_post_auth_runtime_mutation(self):
  self.auth.side_effect=lambda candidate,manifest:(self.root/'tools/run.py').write_text('changed')
  with self.assertRaisesRegex(ValueError,'runtime not bound'):self.fixture()
 def test_issuer_rejects_unbound_consumed_file(self):
  with self.assertRaisesRegex(ValueError,'candidate changed after producer authentication'):
   self.fixture(before_emit=lambda m:m['files'].pop('candidate/controllers/launch.elf'))
 def test_issuer_rejects_unbound_runtime_file(self):
  with self.assertRaisesRegex(ValueError,'runtime not bound'):
   self.fixture(before_emit=lambda m:m['files'].pop('tools/run.py'))

class RuntimeInventory(unittest.TestCase):
 def test_actual_runtime_inventory_includes_corpus_definition(self):
  # Exercise real module discovery; observing digest calls avoids producer I/O.
  with patch.object(producer,'stream_digest',return_value='observed'),patch.object(f,'digest',return_value='observed'):
   identities=f.runtime_identities(producer.ROOT)
  for name in ('ports/ggml/generation_cases.def','tools/source_console.py','tools/frozen_full_admission.py','tools/frozen_full_readbacks.py','tools/admission.py'):
   self.assertIn(name,identities)

if __name__=='__main__':unittest.main()
