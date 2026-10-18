from pathlib import Path
import sys,json,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'tools'))
import source_model_freeze as f
import freeze_source_candidate as legacy
import public_hip_full_assets as assets
import build_llm_full_model_reference as reference
import source_model_auth_test as auth
from source_llm_full_log import FILES as RAW_FILES

class FullFreeze(unittest.TestCase):
 def setUp(self):
  self.base=auth.Auth();self.base.setUp();self.addCleanup(self.base.doCleanups)
  self.root=self.base.root;self.r=self.base.r;self.put=self.base.put;self.inputs=self.base.inputs
  self.r['phase']=self.r['llm_client']['phase']='model-full';self.r['classification']='source-only-model-full-native-client-app'
  self.base.audit.update(phase='model-full',actual_full_call_chain_audited=True)
  self.update_audit();self.put(self.base.app/'client-phase.txt',b'model-full\n')
  self.r['package_files']['client-phase.txt']=f.stream_digest(self.base.app/'client-phase.txt')
  aggregate=self.put(self.root/'aggregate/receipt.json',b'{}');self.refdir=self.root/'reference'
  self.put(self.refdir/'receipt.json',b'{}')
  for n,size in RAW_FILES.items():self.put(self.refdir/'raw'/n,b'0'*size)
  self.data=dict(receipt=str(aggregate),receipt_sha256=f.stream_digest(aggregate),reference=str(self.refdir),profile='smoke',raw_files=dict(RAW_FILES),model={'path':str(self.base.model),'bytes':1,'sha256':self.base.sha},inputs={},assets={})
  for name in f.LAYER_MODULES:
   p=self.put(self.root/'fixtures'/(name+'.hsaco'),name.encode());key='model-layer/'+name+'.hsaco'
   self.data['assets'][key]=str(p);self.put(self.base.app/key,p.read_bytes());self.r['package_files'][key]=f.stream_digest(p)
  self.data['inputs']={str(p):f.stream_digest(p) for p in [aggregate,*[Path(v) for v in self.data['assets'].values()],self.base.model,*self.refdir.rglob('*')] if p.is_file()}
  self.r['llm_client']['full']={k:self.data[k] for k in ('receipt','receipt_sha256','reference','profile')}
  self.addCleanup(patch.stopall);patch.object(f,'ROOT',self.root).start();patch.object(legacy,'ROOT',self.root).start()
  self.aggregate=patch.object(assets,'full_inputs',return_value=self.data).start()
  self.refmock=patch.object(reference,'authenticate_full_model_reference',side_effect=self.authenticate).start()
 def authenticate(self,folder):
  folder=Path(folder);closure={str(p):f.stream_digest(p) for p in folder.rglob('*') if p.is_file()}
  return {'receipt_sha256':f.stream_digest(folder/'receipt.json'),'model':self.data['model']},closure
 def update_audit(self):
  p=self.put(self.base.build/'llm-client-native-audit.json',json.dumps(self.base.audit).encode());self.r['llm_client']['audit']['sha256']=f.stream_digest(p)
 def validate(self):return self.base.validate()
 def freeze(self):
  out=self.root/'candidate';out.mkdir();return out,legacy.freeze_full_reference(self.base.build,self.r,out)
 def test_layer_receipt_and_frozen_manifest(self):
  self.assertEqual(self.validate()['phase'],'model-full');out,m=self.freeze()
  self.assertEqual(set(m),{'aggregate','aggregate_sha256','reference','reference_receipt_sha256','module_sha256','raw_files','profile'})
  self.assertEqual(m['raw_files'],RAW_FILES);self.assertEqual(len(m['module_sha256']),6)
  self.assertEqual(m,f.frozen_full_manifest(out,self.r['llm_client'],self.inputs))
  self.assertFalse((out/'model.gguf').exists());self.assertEqual(len(self.r['package_files']),15)
 def test_missing_layer_audit(self):
  self.base.audit.pop('actual_full_call_chain_audited');self.update_audit()
  with self.assertRaisesRegex(ValueError,'prefix call-chain'):self.validate()
 def test_wrong_profile(self):
  self.r['llm_client']['full']['profile']='long'
  with self.assertRaisesRegex(ValueError,'profile'):self.validate()
 def test_missing_profile(self):
  del self.r['llm_client']['full']['profile']
  with self.assertRaisesRegex(ValueError,'fields'):self.validate()
 def test_aggregate_profile_mismatch(self):
  self.data['profile']='long'
  with self.assertRaisesRegex(ValueError,'binding'):self.validate()
 def test_wrong_raw_contract(self):
  self.data['raw_files']={}
  with self.assertRaisesRegex(ValueError,'raw inventory'):self.validate()
 def test_layer_directory_required(self):
  self.data['assets']={name.replace('model-layer/','model-full/'):path for name,path in self.data['assets'].items()}
  with self.assertRaisesRegex(ValueError,'modules required'):self.validate()
 def test_wrong_classification(self):
  self.r['classification']='source-only-model-rejection-native-client-app'
  with self.assertRaisesRegex(ValueError,'receipt'):self.validate()
 def test_missing_module(self):
  del self.r['package_files']['model-layer/q8.hsaco']
  with self.assertRaisesRegex(ValueError,'assets'):self.validate()
 def test_wrong_packaged_module(self):
  p=self.put(self.base.app/'model-layer/q8.hsaco',b'wrong');self.r['package_files']['model-layer/q8.hsaco']=f.stream_digest(p)
  with self.assertRaisesRegex(ValueError,'modules differ'):self.validate()
 def test_wrong_layer_model(self):
  self.data['model']['sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'model identity'):self.validate()
 def test_missing_closure(self):
  del self.inputs[self.data['receipt']]
  with self.assertRaisesRegex(ValueError,'closure omitted'):self.validate()
 def test_changed_aggregate_identity(self):
  self.data['receipt_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'binding'):self.validate()
 def test_wrong_phase_file(self):
  p=self.put(self.base.app/'client-phase.txt',b'model-gate\n');self.r['package_files']['client-phase.txt']=f.stream_digest(p)
  with self.assertRaisesRegex(ValueError,'configuration'):self.validate()
 def test_extra_package_asset(self):
  p=self.put(self.base.app/'extra',b'x');self.r['package_files']['extra']=f.stream_digest(p)
  with self.assertRaisesRegex(ValueError,'assets'):self.validate()
 def test_frozen_reference_tamper(self):
  out,_=self.freeze();(out/'full-reference/receipt.json').write_bytes(b'bad')
  with self.assertRaisesRegex(ValueError,'reference differs'):f.frozen_full_manifest(out,self.r['llm_client'],self.inputs)
 def test_reference_symlink(self):
  (self.refdir/'unexpected').symlink_to(self.base.model)
  with self.assertRaisesRegex(ValueError,'symlink'):self.freeze()
 def test_reference_size_bound(self):
  with patch.object(f,'PREFIX_REFERENCE_LIMIT',1),self.assertRaisesRegex(ValueError,'16MiB'):self.freeze()
 def test_reference_copy_mutation(self):
  original=legacy.shutil.copyfile
  def bad(source,target):
   original(source,target);Path(target).write_bytes(b'changed')
  with patch.object(legacy.shutil,'copyfile',side_effect=bad),self.assertRaisesRegex(ValueError,'changed during copy'):self.freeze()
 def test_candidate_manifest_and_file_binding(self):
  out,m=self.freeze();(out/'build-receipt.json').write_text(json.dumps(self.r))
  manifest={'deadline_seconds':150,'control_phase_seconds':1800,'llm_client':self.r['llm_client'],'external_model':self.r['external_model'],'model_transfer':{'bytes':2,'sha256':self.r['package_sha256']},'package_files':self.r['package_files'],'model_full':m,'files':{str(p.relative_to(self.root)):f.stream_digest(p) for p in out.rglob('*') if p.is_file()}}
  with patch.object(f,'validator_identities',return_value={}),patch.object(f,'model_contract'),patch.object(f,'_validate_model_receipt',return_value=(self.r['llm_client'],self.data)):
   f.validate_model_candidate(out,manifest)
   manifest['model_full']['raw_files']={}
   with self.assertRaisesRegex(ValueError,'manifest differs'):f.validate_model_candidate(out,manifest)
   manifest['model_full']=m.copy();manifest['model_full']['raw_files']=dict(RAW_FILES);manifest['files']={}
   with self.assertRaisesRegex(ValueError,'file identity'):f.validate_model_candidate(out,manifest)

 def generation_setup(self):
  import build_llm_generation_reference as generation
  from llm_generation_readbacks_test import write_generation_raw
  files=write_generation_raw(self.refdir/'raw')
  self.data.update(profile='generation',raw_files=files)
  self.r['llm_client']['full']['profile']='generation'
  self.base.audit.update(full_profile='generation',actual_generation_call_chain_audited=True);self.update_audit()
  for path in (self.refdir/'raw').iterdir():
   self.data['inputs'][str(path)]=f.stream_digest(path);self.inputs[str(path)]=f.stream_digest(path)
  self.genmock=patch.object(generation,'authenticate_generation_reference',side_effect=self.authenticate).start()
 def test_generation_freeze_and_copied_manifest(self):
  self.generation_setup();self.assertEqual(self.validate()['full']['profile'],'generation')
  out,m=self.freeze();self.assertEqual(m['profile'],'generation');self.assertEqual(len(m['raw_files']),386)
  self.assertEqual(m,f.frozen_full_manifest(out,self.r['llm_client'],self.inputs));self.assertEqual(self.refmock.call_count,0)
  self.assertTrue((out/'full-reference/raw/generation.json').is_file())
 def test_generation_cannot_reuse_smoke_image_audit(self):
  self.generation_setup();self.base.audit.pop('full_profile');self.update_audit()
  with self.assertRaisesRegex(ValueError,'audit profile'):self.validate()
  self.base.audit['full_profile']='generation';self.base.audit['actual_generation_call_chain_audited']=False;self.update_audit()
  with self.assertRaisesRegex(ValueError,'generation call-chain'):self.validate()
 def test_generation_raw_contract_and_metadata_are_required(self):
  self.generation_setup();self.data['raw_files'].pop('generation.json')
  with self.assertRaisesRegex(ValueError,'raw inventory'):self.validate()
 def test_generation_copied_metadata_not_trusted_from_receipt(self):
  self.generation_setup();out,_=self.freeze();p=out/'full-reference/raw/generation.json';p.write_text(p.read_text().replace('"run": 0','"run": false'))
  with self.assertRaises(ValueError):f.frozen_full_manifest(out,self.r['llm_client'],self.inputs)

 def corpus_setup(self):
  import build_llm_generation_reference as generation
  import verify_llm_full_readbacks as readbacks
  for path in (self.refdir/'raw').iterdir():
   self.data['inputs'].pop(str(path),None);self.inputs.pop(str(path),None);path.unlink()
  files={'corpus.json':2,'story/generation.json':2,'story/steps.jsonl':2,'story/run-0-pos-0-logits.f32le':128000}
  for name,size in files.items():
   path=self.put(self.refdir/'raw'/name,b'0'*size);self.data['inputs'][str(path)]=f.stream_digest(path)
  self.data.update(profile='corpus',raw_files=files);self.r['llm_client']['full']['profile']='corpus'
  self.base.audit.update(full_profile='corpus',actual_generation_call_chain_audited=True);self.update_audit()
  self.genmock=patch.object(generation,'authenticate_generation_reference',side_effect=self.authenticate).start()
  def inventory(folder):
   folder=Path(folder)/'raw'
   actual={str(p.relative_to(folder)):p.stat().st_size for p in folder.rglob('*') if p.is_file()}
   if actual!=files:raise ValueError('corpus inventory')
   return dict(files)
  patch.object(readbacks,'corpus_files',side_effect=inventory).start()
 def test_corpus_copy_inventory_and_audit(self):
  self.corpus_setup();self.assertEqual(self.validate()['full']['profile'],'corpus')
  out,m=self.freeze();self.assertEqual(m['profile'],'corpus');self.assertTrue((out/'full-reference/raw/story/generation.json').is_file())
  self.assertEqual(m,f.frozen_full_manifest(out,self.r['llm_client'],self.inputs))
  (out/'full-reference/raw/story/extra').write_text('x')
  with self.assertRaisesRegex(ValueError,'inventory'):f.frozen_full_manifest(out,self.r['llm_client'],self.inputs)
 def test_corpus_exact_budgets(self):
  self.corpus_setup();out,m=self.freeze();(out/'build-receipt.json').write_text(json.dumps(self.r))
  manifest=dict(deadline_seconds=1830,control_phase_seconds=4200,raw_capture_seconds=600,llm_client=self.r['llm_client'],external_model=self.r['external_model'],model_transfer={'bytes':2,'sha256':self.r['package_sha256']},package_files=self.r['package_files'],model_full=m,files={str(p.relative_to(self.root)):f.stream_digest(p) for p in out.rglob('*') if p.is_file()})
  with patch.object(f,'validator_identities',return_value={}),patch.object(f,'model_contract'),patch.object(f,'_validate_model_receipt',return_value=(self.r['llm_client'],self.data)):
   f.validate_model_candidate(out,manifest)
   for field in ('deadline_seconds','control_phase_seconds','raw_capture_seconds'):
    previous=manifest[field];manifest[field]-=1
    with self.subTest(field=field),self.assertRaisesRegex(ValueError,'budget'):f.validate_model_candidate(out,manifest)
    manifest[field]=previous
 def test_corpus_limit_does_not_expand_other_profiles(self):
  self.assertEqual(f.reference_limit('full','corpus'),128*1024*1024)
  for kind,profile in [('full','generation'),('full','smoke'),('layer','corpus'),('prefix','corpus')]:self.assertEqual(f.reference_limit(kind,profile),16*1024*1024)
  self.corpus_setup()
  with patch.object(f,'CORPUS_REFERENCE_LIMIT',1),self.assertRaisesRegex(ValueError,'128MiB'):self.freeze()
 def test_corpus_rejects_short_image_audit(self):
  self.corpus_setup();self.base.audit['full_profile']='generation';self.update_audit()
  with self.assertRaisesRegex(ValueError,'audit profile'):self.validate()
  self.base.audit['full_profile']='corpus';self.base.audit['actual_generation_call_chain_audited']=False;self.update_audit()
  with self.assertRaisesRegex(ValueError,'generation call-chain'):self.validate()

 def test_actual_full_freeze_cli(self):self.run_full_freeze_cli()
 def test_actual_corpus_freeze_cli(self):self.run_full_freeze_cli(corpus=True)
 def run_full_freeze_cli(self,corpus=False):
  if corpus:self.corpus_setup()
  import external_model as em
  for name in ('source_console.py','source_public_hip_log.py','freeze_source_candidate.py','helper.py'):self.put(self.root/'tools'/name)
  for name in ('dependencies.json','elevation-inputs.json','controller-inputs.json','title.json'):self.put(self.root/name)
  for name in ('launch.elf','close.elf'):self.put(self.root/'build/controllers/PPSA99014'/name)
  self.put(self.root/'snapshot/teardown-snapshot.elf');(self.base.build/'receipt.json').write_text(json.dumps(self.r))
  output=self.root/'results/candidate';argv=['freeze','--build',str(self.base.build),'--output',str(output),'--snapshot-build',str(self.root/'snapshot'),'--baseline-image-sha256','f'*64,'--baseline-image-bytes','1','--client-phase','model-full']
  validators={'tools/helper.py':f.stream_digest(self.root/'tools/helper.py')}
  def git(*args,**kwargs):return 'commit\n' if kwargs.get('text') else b''
  with patch.object(em,'MODEL_BYTES',1),patch.object(em,'MODEL_SHA',self.base.sha),patch.object(f,'MODEL_BYTES',1),patch.object(f,'MODEL_SHA',self.base.sha),patch.object(f,'prerequisites',return_value=(self.base.v,self.base.c,self.base.m)),patch.object(f,'audit_sources',return_value=[]),patch.object(f,'validator_identities',return_value=validators),patch.object(legacy.subprocess,'check_output',side_effect=git),patch.object(sys,'argv',argv):legacy.main()
  self.assertEqual(self.aggregate.call_count,2)
  self.assertEqual(self.refmock.call_count,0 if corpus else 1)
  if corpus:self.assertEqual(self.genmock.call_count,1)
  manifest=json.loads((output/'manifest.json').read_text())
  self.assertEqual(manifest['llm_client']['phase'],'model-full');self.assertEqual(manifest['deadline_seconds'],1830 if corpus else 150)
  self.assertEqual(manifest['control_phase_seconds'],4200 if corpus else 1800);self.assertEqual(len(manifest['model_full']['raw_files']),4 if corpus else 72)
  if corpus:self.assertEqual(manifest['raw_capture_seconds'],600);self.assertIn('272 committed',manifest['hypothesis'])
  self.assertEqual(len(manifest['package_files']),15);self.assertFalse((output/'package/app/model.gguf').exists())

 def test_standalone_manifest_authenticates_original_and_copy_once(self):
  out,_=self.freeze();self.aggregate.reset_mock();self.refmock.reset_mock()
  f.frozen_full_manifest(out,self.r['llm_client'],self.inputs)
  self.assertEqual(self.aggregate.call_count,1);self.assertEqual(self.refmock.call_count,1)
 def test_copy_rejects_deleted_original_artifact(self):
  data=f.full_binding(self.r['llm_client'],self.inputs)
  next((self.refdir/'raw').iterdir()).unlink()
  out=self.root/'candidate';out.mkdir()
  with self.assertRaisesRegex(ValueError,'inventory changed'):legacy._freeze_full_reference(self.r,out,data)
 def test_final_admission_rejects_postcopy_mutations(self):
  # Reuse full CLI fixture in independent temp roots for every mutation.
  for mode in ('copied-bytes','copied-extra','model','module','original-reference'):
   with self.subTest(mode=mode):
    instance=FullFreeze();instance.setUp()
    try:
     original=legacy._freeze_full_reference
     def mutate(receipt,output,data):
      result=original(receipt,output,data)
      if mode=='copied-bytes':next((output/'full-reference/raw').iterdir()).write_bytes(b'changed')
      elif mode=='copied-extra':(output/'full-reference/extra').write_bytes(b'extra')
      elif mode=='model':instance.base.model.write_bytes(b'changed')
      elif mode=='module':Path(instance.data['assets']['model-layer/q8.hsaco']).write_bytes(b'changed')
      else:next((instance.refdir/'raw').iterdir()).write_bytes(b'changed')
      return result
     with patch.object(legacy,'_freeze_full_reference',side_effect=mutate),self.assertRaises(ValueError):instance.test_actual_full_freeze_cli()
    finally:instance.doCleanups()

if __name__=='__main__':unittest.main()
