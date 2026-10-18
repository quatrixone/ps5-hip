"""Original runtime admission/transfer checks, synthetic six-file bytes only."""
from pathlib import Path
import copy,hashlib,json,sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_console as c
import source_model_freeze as f
import external_original_model as original
from source_external_model_test import FTP

class Original(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  self.data={n:('tiny '+n).encode() for n in original.NAMES}
  self.files={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in self.data.items()}
  self.lock=dict(metadata={n:v for n,v in self.files.items() if n!='model.safetensors'},weights=dict(file_bytes=self.files['model.safetensors']['bytes'],expected_file_sha256=self.files['model.safetensors']['sha256']))
  p=self.root/'models/tinyllama-1.1b-chat-v1.0.json';p.parent.mkdir();p.write_text(json.dumps(self.lock))
  self.patch=patch.object(f,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
  self.model=dict(schema=1,format='original_safetensors',directory='/data/hip/models/tinyllama-original',files=self.files)
  self.o=c.Control.__new__(c.Control);self.o.manifest=dict(external_original_model=self.model,llm_client=dict(phase='model-full',full=dict(profile='generation')),model_full=dict(profile='generation'))
  self.o.phase='deploy';self.o.fixture='llm_client';self.o.ftp=FTP();self.o.record={};self.o.token='token';self.o.idle=lambda:None;self.o.model_file=None
  self.source=self.root/'source';self.source.mkdir()
  for n,v in self.data.items():(self.source/n).write_bytes(v)
  self.o.model_directory=self.source
 def test_upload_exact_six_and_reuse(self):
  self.o.provision_external_model();self.assertEqual(len(self.o.ftp.uploads),6);self.assertTrue(self.o.record['external_original_model']['complete'])
  self.o.model_directory=None;self.o.provision_external_model();self.assertEqual(len(self.o.ftp.uploads),6)
  self.o.phase='execute';self.o.verify_external_model();self.assertEqual(self.o.observation_seconds(),630)
  self.assertGreater(self.o.phase_remaining(3000),2300)
 def test_corrupt_existing_prevents_any_upload(self):
  self.o.ensure_directory(self.model['directory'],'model');self.o.ftp.files[self.model['directory']+'/tokenizer.json']=b'bad'
  with self.assertRaises(ValueError):self.o.provision_external_model()
  self.assertEqual(self.o.ftp.uploads,[])
 def test_changed_local_prevents_any_upload(self):
  (self.source/'tokenizer.json').write_bytes(b'bad')
  with self.assertRaises(ValueError):self.o.provision_external_model()
  self.assertEqual(self.o.ftp.uploads,[])
 def test_missing_explicit_source_and_wrong_legacy_arg(self):
  self.o.model_directory=None
  with self.assertRaisesRegex(RuntimeError,'explicit'):self.o.provision_external_model()
  self.o.model_file=self.source
  with self.assertRaisesRegex(RuntimeError,'model-directory'):self.o.verify_external_model()
 def test_partial_never_promoted(self):
  self.o.ftp.partial=True
  with self.assertRaises(Exception):self.o.provision_external_model()
  self.assertEqual(self.o.ftp.renames,[]);self.assertFalse(self.o.record['external_original_model']['complete'])
 def test_destination_race_never_promoted(self):
  name=next(iter(self.files));self.o.ftp.before_rename=lambda:self.o.ftp.files.update({self.model['directory']+'/'+name:b'another'})
  with self.assertRaisesRegex(RuntimeError,'appeared'):self.o.provision_external_model()
  self.assertEqual(self.o.ftp.renames,[])
 def test_six_file_prelaunch_auth_and_missing_directory_no_mutation(self):
  self.o.provision_external_model();self.o.phase='execute';self.o.record={}
  self.o.verify_external_model();self.assertEqual(len(self.o.record['external_original_model_reads']),6)
  self.o.ftp.files[self.model['directory']+'/model.safetensors']=b'wrong'
  with self.assertRaises(ValueError):self.o.verify_external_model()
  self.o.ftp.dirs.remove(self.model['directory'])
  with self.assertRaisesRegex(RuntimeError,'directory missing'):self.o.verify_external_model()
  self.assertNotIn(self.model['directory'],self.o.ftp.dirs)
 def test_mixed_wrong_phase_and_unlocked_identity_reject(self):
  for mutation in ('mixed','phase','profile','size','schema','missing','extra'):
   m=copy.deepcopy(self.o.manifest)
   if mutation=='mixed':m['external_model']={}
   if mutation=='phase':m['llm_client']['phase']='model-gate'
   if mutation=='profile':m['llm_client']['full']['profile']='corpus'
   if mutation=='size':m['external_original_model']['files']['model.safetensors']['bytes']+=1
   if mutation=='schema':m['external_original_model']['schema']=True
   if mutation=='extra':m['external_original_model']['files']['../extra']={'bytes':1,'sha256':'0'*64}
   if mutation=='missing':del m['external_original_model']['files']['config.json']
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):f.runtime_model(m)
 def test_model_contract_has_small_package_not_original_payload(self):
  m=copy.deepcopy(self.o.manifest);m.update(fixture='llm_client',model_transfer=dict(bytes=100,sha256='a'*64,deadline_seconds=1800),baseline_transfer=dict(bytes=100,sha256='b'*64,deadline_seconds=1800),baseline_image_sha256='b'*64)
  m['llm_client']['model']=dict(bytes=f.MODEL_BYTES,sha256=f.MODEL_SHA)
  f.model_contract(m);m['model_transfer']['bytes']=c.package_limit('llm_client')+1
  with self.assertRaises(ValueError):f.model_contract(m)

 def test_actual_full_receipt_canonical_asset_and_bound_sources(self):
  from source_model_full_freeze_test import FullFreeze
  fixture=FullFreeze();fixture.setUp();self.addCleanup(fixture.doCleanups)
  r=fixture.r;r.pop('external_model');r['external_original_model']=copy.deepcopy(self.model)
  r['llm_client']['full']['profile']='generation';fixture.data['profile']='generation'
  fixture.base.audit.update(full_profile='generation',actual_generation_call_chain_audited=True,original_model_call_chain_audited=True,model_format='original_safetensors');fixture.update_audit()
  (fixture.root/'models').mkdir();(fixture.root/'models/tinyllama-1.1b-chat-v1.0.json').write_text(json.dumps(self.lock))
  for n,v in self.data.items():fixture.put(fixture.root/n,v)
  (fixture.base.app/'model-path.txt').unlink();r['package_files'].pop('model-path.txt');fixture.inputs.pop(str(fixture.base.app/'model-path.txt'));fixture.put(fixture.root/'tools/external_original_model.py')
  asset=fixture.put(fixture.base.app/'original-model.json',original.encode(self.model));r['package_files']['original-model.json']=f.stream_digest(asset)
  with patch.object(f,'_binding',return_value=fixture.data):
   self.assertEqual(fixture.validate()['phase'],'model-full')
   asset.write_bytes(original.encode(self.model)+b' ');r['package_files']['original-model.json']=f.stream_digest(asset)
   fixture.inputs[str(asset)]=f.stream_digest(asset)
   with self.assertRaisesRegex(ValueError,'original manifest'):fixture.validate()
   asset.write_bytes(original.encode(self.model));r['package_files']['original-model.json']=f.stream_digest(asset);fixture.inputs[str(asset)]=f.stream_digest(asset)
   del fixture.inputs[str(fixture.root/'tokenizer.json')]
   with self.assertRaisesRegex(ValueError,'original source input'):fixture.validate()

 def test_real_frozen_artifact_admission_original_keeps_gguf_oracle(self):
  import frozen_full_admission as admission
  from frozen_full_admission_test import Integration,dump
  fixture=Integration();fixture.setUp();self.addCleanup(fixture.doCleanups)
  (fixture.root/'models').mkdir();(fixture.root/'models/tinyllama-1.1b-chat-v1.0.json').write_text(json.dumps(self.lock))
  def transform(m):
   candidate=fixture.root/'candidate';app=candidate/'package/app'
   audit=json.loads((candidate/'llm-client-native-audit.json').read_text());audit.update(original_model_call_chain_audited=True,model_format='original_safetensors');dump(candidate/'llm-client-native-audit.json',audit);m['llm_client']['audit']['sha256']=admission.digest(candidate/'llm-client-native-audit.json')
   m.pop('external_model');m['external_original_model']=copy.deepcopy(self.model);m.update(deadline_seconds=630,control_phase_seconds=2400)
   (app/'model-path.txt').unlink();(app/'original-model.json').write_bytes(original.encode(self.model))
   m['package_files']={p.relative_to(app).as_posix():admission.digest(p) for p in app.rglob('*') if p.is_file()}
   receipt=json.loads((candidate/'build-receipt.json').read_text());receipt.pop('external_model');receipt.update(external_original_model=self.model,package_files=m['package_files'],llm_client=m['llm_client']);dump(candidate/'build-receipt.json',receipt)
   m['files']={p.relative_to(fixture.root).as_posix():admission.digest(p) for p in candidate.rglob('*') if p.is_file()};m['files'].update(admission.runtime_identities(fixture.root))
  with patch.object(f,'ROOT',fixture.root),patch.object(c,'ROOT',fixture.root):
   candidate,m,_=fixture.fixture('generation',before_emit=transform)
   admission.validate(candidate,m,fixture.root)
   obj=c.Control(candidate,'deploy',model_directory=self.source)
   self.assertEqual(obj.observation_seconds(),630)
   self.assertNotEqual(m['llm_client']['model']['bytes'],self.files['model.safetensors']['bytes'])
   (candidate/'package/app/original-model.json').write_bytes(b'{}')
   with self.assertRaises(ValueError):admission.validate(candidate,m,fixture.root)

 def test_execute_rejects_changed_original_before_snapshot_or_launch(self):
  from unittest.mock import Mock
  self.o.provision_external_model();self.o.phase='execute';self.o.candidate=self.root/'candidate';(self.o.candidate/'deploy').mkdir(parents=True)
  self.o.manifest_sha='manifest';self.o.record['package_sha256']='package'
  (self.o.candidate/'deploy/receipt.json').write_text(json.dumps(dict(accepted=True,healthy_after=True,lock_released=True,phase='deploy',manifest_sha256='manifest',package_sha256='package')))
  for method in ('installed','fresh_log','fresh_rmsnorm_dumps','fresh_q4_dumps','fresh_weighted_qkv_dumps','fresh_prefix_dumps'):setattr(self.o,method,lambda:None)
  self.o.send=Mock(side_effect=AssertionError('no snapshot/launch allowed'))
  self.o.ftp.files[self.model['directory']+'/model.safetensors']=b'changed'
  with self.assertRaises(ValueError):self.o.execute()
  self.o.send.assert_not_called()

 def test_remote_symlink_directory_and_file_reject(self):
  self.o.provision_external_model();self.o.phase='execute';old=self.o.listing
  for path in ('/data/hip/models',self.model['directory']):
   def listing(directory):
    rows=old(directory)
    return [('l'+row[1:]+' -> elsewhere') if directory==path else row for row in rows]
   with patch.object(self.o,'listing',side_effect=listing),self.assertRaises(RuntimeError):self.o.verify_external_model()

if __name__=='__main__':unittest.main()
