from pathlib import Path
import sys,json,unittest
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'tools'))
import source_model_freeze as f
import freeze_source_candidate as legacy
import public_hip_prefix_assets as assets
import build_llm_prefix_reference as reference
import source_model_auth_test as auth

class PrefixFreeze(unittest.TestCase):
 def setUp(self):
  self.base=auth.Auth();self.base.setUp();self.addCleanup(self.base.doCleanups)
  self.root=self.base.root;self.r=self.base.r;self.put=self.base.put;self.inputs=self.base.inputs
  self.r['phase']=self.r['llm_client']['phase']='model-prefix';self.r['classification']='source-only-model-prefix-native-client-app'
  self.base.audit.update(phase='model-prefix',actual_prefix_call_chain_audited=True)
  self.update_audit();self.put(self.base.app/'client-phase.txt',b'model-prefix\n')
  self.r['package_files']['client-phase.txt']=f.stream_digest(self.base.app/'client-phase.txt')
  aggregate=self.put(self.root/'aggregate/receipt.json',b'{}');self.refdir=self.root/'reference'
  self.put(self.refdir/'receipt.json',b'{}')
  for n,size in f.PREFIX_RAW_FILES.items():self.put(self.refdir/'run-1'/n,b'0'*size)
  self.data=dict(receipt=str(aggregate),receipt_sha256=f.stream_digest(aggregate),reference=str(self.refdir),model={'path':str(self.base.model),'bytes':1,'sha256':self.base.sha},inputs={},assets={})
  for name in f.PREFIX_MODULES:
   p=self.put(self.root/'fixtures'/(name+'.hsaco'),name.encode());key='model-prefix/'+name+'.hsaco'
   self.data['assets'][key]=str(p);self.put(self.base.app/key,p.read_bytes());self.r['package_files'][key]=f.stream_digest(p)
  self.data['inputs']={str(p):f.stream_digest(p) for p in [aggregate,*[Path(v) for v in self.data['assets'].values()],self.base.model,*self.refdir.rglob('*')] if p.is_file()}
  self.r['llm_client']['prefix']={k:self.data[k] for k in ('receipt','receipt_sha256','reference')}
  self.addCleanup(patch.stopall);patch.object(f,'ROOT',self.root).start();patch.object(legacy,'ROOT',self.root).start()
  self.aggregate=patch.object(assets,'prefix_inputs',return_value=self.data).start()
  self.refmock=patch.object(reference,'authenticate_prefix_reference',side_effect=self.authenticate).start()
 def authenticate(self,folder):
  folder=Path(folder);closure={str(p):f.stream_digest(p) for p in folder.rglob('*') if p.is_file()}
  return {'receipt_sha256':f.stream_digest(folder/'receipt.json'),'model':self.data['model']},closure
 def update_audit(self):
  p=self.put(self.base.build/'llm-client-native-audit.json',json.dumps(self.base.audit).encode());self.r['llm_client']['audit']['sha256']=f.stream_digest(p)
 def validate(self):return self.base.validate()
 def freeze(self):
  out=self.root/'candidate';out.mkdir();return out,legacy.freeze_prefix_reference(self.base.build,self.r,out)
 def test_prefix_receipt_and_frozen_manifest(self):
  self.assertEqual(self.validate()['phase'],'model-prefix');out,m=self.freeze()
  self.assertEqual(set(m),{'aggregate','aggregate_sha256','reference','reference_receipt_sha256','module_sha256','raw_files'})
  self.assertEqual(m['raw_files'],f.PREFIX_RAW_FILES);self.assertEqual(len(m['module_sha256']),5)
  self.assertEqual(m,f.frozen_prefix_manifest(out,self.r['llm_client'],self.inputs))
  self.assertFalse((out/'model.gguf').exists());self.assertEqual(len(self.r['package_files']),14)
 def test_missing_prefix_audit(self):
  self.base.audit.pop('actual_prefix_call_chain_audited');self.update_audit()
  with self.assertRaisesRegex(ValueError,'prefix call-chain'):self.validate()
 def test_wrong_classification(self):
  self.r['classification']='source-only-model-rejection-native-client-app'
  with self.assertRaisesRegex(ValueError,'receipt'):self.validate()
 def test_missing_module(self):
  del self.r['package_files']['model-prefix/q8.hsaco']
  with self.assertRaisesRegex(ValueError,'assets'):self.validate()
 def test_wrong_packaged_module(self):
  p=self.put(self.base.app/'model-prefix/q8.hsaco',b'wrong');self.r['package_files']['model-prefix/q8.hsaco']=f.stream_digest(p)
  with self.assertRaisesRegex(ValueError,'modules differ'):self.validate()
 def test_wrong_prefix_model(self):
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
  out,_=self.freeze();(out/'prefix-reference/receipt.json').write_bytes(b'bad')
  with self.assertRaisesRegex(ValueError,'reference differs'):f.frozen_prefix_manifest(out,self.r['llm_client'],self.inputs)
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
  manifest={'deadline_seconds':150,'control_phase_seconds':1800,'llm_client':self.r['llm_client'],'external_model':self.r['external_model'],'model_transfer':{'bytes':2,'sha256':self.r['package_sha256']},'package_files':self.r['package_files'],'model_prefix':m,'files':{str(p.relative_to(self.root)):f.stream_digest(p) for p in out.rglob('*') if p.is_file()}}
  with patch.object(f,'validator_identities',return_value={}),patch.object(f,'model_contract'),patch.object(f,'validate_model_receipt',return_value=self.r['llm_client']):
   f.validate_model_candidate(out,manifest)
   manifest['model_prefix']['raw_files']={}
   with self.assertRaisesRegex(ValueError,'manifest differs'):f.validate_model_candidate(out,manifest)
   manifest['model_prefix']=m.copy();manifest['model_prefix']['raw_files']=dict(f.PREFIX_RAW_FILES);manifest['files']={}
   with self.assertRaisesRegex(ValueError,'file identity'):f.validate_model_candidate(out,manifest)

 def test_actual_prefix_freeze_cli(self):
  import external_model as em
  for name in ('source_console.py','source_public_hip_log.py','freeze_source_candidate.py','helper.py'):self.put(self.root/'tools'/name)
  for name in ('dependencies.json','elevation-inputs.json','controller-inputs.json','title.json'):self.put(self.root/name)
  for name in ('launch.elf','close.elf'):self.put(self.root/'build/controllers/PPSA99014'/name)
  self.put(self.root/'snapshot/teardown-snapshot.elf');(self.base.build/'receipt.json').write_text(json.dumps(self.r))
  output=self.root/'results/candidate';argv=['freeze','--build',str(self.base.build),'--output',str(output),'--snapshot-build',str(self.root/'snapshot'),'--baseline-image-sha256','f'*64,'--baseline-image-bytes','1','--client-phase','model-prefix']
  validators={'tools/helper.py':f.stream_digest(self.root/'tools/helper.py')}
  def git(*args,**kwargs):return 'commit\n' if kwargs.get('text') else b''
  with patch.object(em,'MODEL_BYTES',1),patch.object(em,'MODEL_SHA',self.base.sha),patch.object(f,'MODEL_BYTES',1),patch.object(f,'MODEL_SHA',self.base.sha),patch.object(f,'prerequisites',return_value=(self.base.v,self.base.c,self.base.m)),patch.object(f,'audit_sources',return_value=[]),patch.object(f,'validator_identities',return_value=validators),patch.object(legacy.subprocess,'check_output',side_effect=git),patch.object(sys,'argv',argv):legacy.main()
  manifest=json.loads((output/'manifest.json').read_text())
  self.assertEqual(manifest['llm_client']['phase'],'model-prefix');self.assertEqual(manifest['deadline_seconds'],150)
  self.assertEqual(manifest['control_phase_seconds'],1800);self.assertEqual(len(manifest['model_prefix']['raw_files']),24)
  self.assertEqual(len(manifest['package_files']),14);self.assertFalse((output/'package/app/model.gguf').exists())

if __name__=='__main__':unittest.main()
