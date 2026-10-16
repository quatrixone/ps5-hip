from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import tempfile,json,unittest,sys,copy
from pathlib import Path
from unittest.mock import patch
import source_model_freeze as f
sys.path.insert(0,str(f.ROOT/'tools'))
import freeze_source_candidate as legacy
class Auth(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.build=self.root/'build';self.app=self.build/'package/app';self.app.mkdir(parents=True);self.inputs={}
  def put(p,data=b'x'):
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);self.inputs[str(p)]=f.stream_digest(p);return p
  self.put=put;src=put(self.root/'source.cpp');units=[]
  for i in range(148):
   p=put(self.build/f'{i}.o');units.append(dict(object=str(p),object_sha256=self.inputs[str(p)],source=str(src),source_sha256=self.inputs[str(src)],headers={str(src):self.inputs[str(src)]}))
  opt=put(self.root/'.deps/sources/llama-client/ggml/src/ggml-opt.cpp')
  units[77].update(source=str(opt),source_sha256=self.inputs[str(opt)],headers={str(opt):self.inputs[str(opt)]})
  for n in ('build_llm_native_app.py','build_public_hip.py','build_llm_native_client.py','build_llm_vocab_fixture.py'):put(self.root/'tools'/n)
  names=legacy.LLM_CLIENT_PACKAGE_FILES|{'model.gguf'}
  for n in names:put(self.app/n,b'model-gate\n' if n=='client-phase.txt' else b'/data/test' if n=='results-dir.txt' else b'x')
  for n in ('title.elf','eboot.elf','extracted.elf','eboot.bin','libc.prx'):put(self.build/n)
  package=put(self.build/'package/PPSA99014.ffpkg',b'xx');self.sha=f.stream_digest(self.app/'model.gguf')
  receipt=put(self.root/'prereq.json',b'{}');base=dict(receipt=str(receipt),receipt_sha256=f.stream_digest(receipt),inputs={str(receipt):f.stream_digest(receipt)})
  self.v=dict(base,asset=str(self.app/'vocab.gguf'));self.c=dict(base,objects=[u['object'] for u in units[-70:]]);self.m=dict(base,asset=str(self.app/'model.gguf'),bytes=1,sha256=self.sha,source_dir=str(self.root),cpu_build=str(self.root))
  audit=dict(status='PASS',classification='offline-public-hip-llm-client-native-image-audit',phase='model-gate',wrapper_retained=True,wrapper_no_bypass=True,actual_client_call_chain_audited=True,actual_client_gate_executed=False,native_executed=False,gpu_executed=False,inputs={str(self.build/n):self.inputs[str(self.build/n)] for n in ('title.elf','eboot.elf','extracted.elf')});self.audit=audit
  ap=put(self.build/'llm-client-native-audit.json',json.dumps(audit).encode())
  bound=dict(phase='model-gate',vocabulary={k:self.v[k] for k in ('receipt','receipt_sha256','asset')},native_client={k:self.c[k] for k in ('receipt','receipt_sha256')},model={k:v for k,v in self.m.items() if k!='inputs'},audit={'path':str(ap),'sha256':f.stream_digest(ap)})
  self.r=dict(schema=1,status='PASS',classification='source-only-model-rejection-native-client-app',fixture='llm_client',phase='model-gate',native_executed=False,gpu_executed=False,object_count=148,units=units,inputs=self.inputs,llm_client=bound,package_bytes=2,package_sha256=f.stream_digest(package),package_files={n:f.stream_digest(self.app/n) for n in names},results_dir='/data/test')
 def validate(self):
  # Tiny complete file-backed receipt; only expensive prerequisite authentication and fixed model size substituted.
  with patch.object(f,'MODEL_BYTES',1),patch.object(f,'MODEL_SHA',self.sha),patch.object(f,'prerequisites',return_value=(self.v,self.c,self.m)),patch.object(f,'audit_sources',return_value=[]),patch.object(legacy,'ROOT',self.root):return f.validate_model_receipt(self.build,self.r)
 def test_actual_model_freeze_cli(self):
  self.run_model_freeze_cli(2)
 def test_actual_model_freeze_cli_ufs_overhead(self):
  self.run_model_freeze_cli(827719680)
 def run_model_freeze_cli(self, package_bytes):
  import sys
  for name in ('source_console.py','source_public_hip_log.py','freeze_source_candidate.py','helper.py'):self.put(self.root/'tools'/name)
  for name in ('dependencies.json','elevation-inputs.json','controller-inputs.json','title.json'):self.put(self.root/name)
  for name in ('launch.elf','close.elf'):self.put(self.root/'build/controllers/PPSA99014'/name)
  self.put(self.root/'snapshot/teardown-snapshot.elf');(self.build/'receipt.json').write_text(json.dumps(self.r))
  self.r['package_bytes']=package_bytes;(self.build/'receipt.json').write_text(json.dumps(self.r))
  output=self.root/'results/candidate';argv=['freeze','--build',str(self.build),'--output',str(output),'--snapshot-build',str(self.root/'snapshot'),'--baseline-image-sha256','f'*64,'--baseline-image-bytes','1','--client-phase','model-gate']
  validators={'tools/helper.py':f.stream_digest(self.root/'tools/helper.py')}
  def git(*args,**kwargs):return 'commit\n' if kwargs.get('text') else b''
  real_stat=Path.stat
  def sized_stat(path,*args,**kwargs):
   result=real_stat(path,*args,**kwargs)
   if path.name!='PPSA99014.ffpkg':return result
   class Sized:
    st_size=package_bytes
    def __getattr__(self,name):return getattr(result,name)
   return Sized()
  with patch.object(Path,'stat',sized_stat),patch.object(f,'ROOT',self.root),patch.object(f,'MODEL_BYTES',1),patch.object(f,'MODEL_SHA',self.sha),patch.object(f,'prerequisites',return_value=(self.v,self.c,self.m)),patch.object(f,'audit_sources',return_value=[]),patch.object(f,'validator_identities',return_value=validators),patch.object(legacy,'ROOT',self.root),patch.object(legacy.subprocess,'check_output',side_effect=git),patch.object(sys,'argv',argv):legacy.main()
  manifest=json.loads((output/'manifest.json').read_text());self.assertEqual(manifest['llm_client']['phase'],'model-gate');self.assertEqual(manifest['deadline_seconds'],150);self.assertEqual(manifest['control_phase_seconds'],1800);self.assertEqual(manifest['model_transfer']['bytes'],package_bytes);self.assertIn('tools/helper.py',manifest['files']);self.assertFalse((output/'review.json').exists())
 def test_default_vocab_cli_rejects_model_receipt(self):
  import sys
  (self.build/'receipt.json').write_text(json.dumps(self.r));output=self.root/'results/candidate'
  argv=['freeze','--build',str(self.build),'--output',str(output),'--baseline-image-sha256','f'*64]
  with patch.object(legacy,'ROOT',self.root),patch.object(sys,'argv',argv),self.assertRaises(SystemExit):legacy.main()
  self.assertFalse(output.exists())
 def test_complete(self):self.assertEqual(self.validate()['phase'],'model-gate')
 def test_model_package_overhead_and_ceiling(self):
  manifest=dict(fixture='llm_client',llm_client=dict(phase='model-gate',model=dict(bytes=f.MODEL_BYTES,sha256=f.MODEL_SHA)),model_transfer=dict(bytes=827719680,sha256='a'*64,deadline_seconds=1800),baseline_transfer=dict(bytes=19922944,sha256='b'*64,deadline_seconds=1800),baseline_image_sha256='b'*64,baseline_log_fixture='llm_client')
  f.model_contract(manifest)
  manifest['model_transfer']['bytes']=f.MODEL_PACKAGE_LIMIT;f.model_contract(manifest)
  for size in (0,True,f.MODEL_BYTES,f.MODEL_PACKAGE_LIMIT+1):
   manifest['model_transfer']['bytes']=size
   with self.subTest(size=size),self.assertRaisesRegex(ValueError,'model package bound'):f.model_contract(manifest)
 def test_exact_package_size_still_required(self):
  self.r['package_bytes']+=1
  with self.assertRaisesRegex(ValueError,'package identity'):self.validate()
 def test_wrong_phase(self):
  self.r['phase']='vocab'
  with self.assertRaises(ValueError):self.validate()
 def test_missing_callchain(self):
  self.audit.pop('actual_client_call_chain_audited');p=self.build/'llm-client-native-audit.json';p.write_text(json.dumps(self.audit));self.inputs[str(p)]=f.stream_digest(p);self.r['llm_client']['audit']['sha256']=f.stream_digest(p)
  with self.assertRaises(ValueError):self.validate()
 def test_missing_unit_identity(self):
  self.inputs.pop(self.r['units'][0]['object'])
  with self.assertRaises(ValueError):self.validate()
 def test_missing_model_cleanup_unit(self):
  self.r['units'].pop(77);self.r['object_count']=147
  with self.assertRaises(ValueError):self.validate()
 def test_substituted_model_cleanup_source(self):
  self.r['units'][77]['source']=self.r['units'][0]['source']
  with self.assertRaisesRegex(ValueError,'model cleanup'):self.validate()
 def test_model_cleanup_headers_required(self):
  self.r['units'][77]['headers']={}
  with self.assertRaisesRegex(ValueError,'source/header'):self.validate()
 def test_wrong_model_asset(self):
  (self.app/'model.gguf').write_bytes(b'y')
  with self.assertRaises(ValueError):self.validate()
if __name__=='__main__':unittest.main()
