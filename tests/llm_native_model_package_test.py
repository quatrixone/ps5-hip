import importlib.util,json,tempfile,unittest,ast,threading
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]/'tools'
import sys
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('model_app',HERE/'build_llm_native_app.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def actual_binder():
 tree=ast.parse((HERE/'build_llm_native_app.py').read_text());build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build');node=next(n for n in build.body if isinstance(n,ast.FunctionDef) and n.name=='bind')
 env={'Path':Path,'b':d.b,'lock':threading.RLock(),'inputs':{},'links':{}}
 exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-model-bind>','exec'),env);return env['bind']

class ModelPackage(unittest.TestCase):
 def test_phase_arguments_are_explicit(self):
  d.validate_phase_arguments('vocab',None,None,None);d.validate_phase_arguments('model-gate','model','source','cpu')
  for args in [('bad',None,None,None),('vocab','model',None,None),('model-gate','model',None,None)]:
   with self.assertRaises(ValueError):d.validate_phase_arguments(*args)
 def test_phase_asset_inventory(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);phase=folder/'phase';vocab=folder/'vocab';model=folder/'model';vocab.write_bytes(b'vocab');model.write_bytes(b'model');d.write_phase(phase)
   base={'vocab.gguf':vocab,'client-phase.txt':phase};d.b.validate_client_assets(base)
   with self.assertRaises(ValueError):d.b.validate_client_assets({**base,'model.gguf':model})
   d.write_phase(phase,'model-gate');self.assertEqual(phase.read_bytes(),b'model-gate\n')
   with self.assertRaises(ValueError):d.b.validate_client_assets(base)
   model.write_bytes(d.encode_model_path(d.DEFAULT_MODEL_PATH))
   d.b.validate_client_assets({**base,'model-path.txt':model})
   with self.assertRaises(ValueError):d.b.validate_client_assets({**base,'model.gguf':model})
   phase.write_bytes(b'model-gate')
   with self.assertRaises(ValueError):d.b.validate_client_assets({**base,'model.gguf':model})
 def test_short_and_symlink_models_reject_before_receipt_or_hash(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);model=folder/'model';model.write_bytes(b'short');link=folder/'alias';link.symlink_to(model)
   with patch.object(d.graph,'cpu_inputs') as cpu:
    with self.assertRaisesRegex(ValueError,'size differs'):d.model_inputs(model,folder,folder)
    with self.assertRaisesRegex(ValueError,'regular'):d.model_inputs(link,folder,folder)
   cpu.assert_not_called()
 def test_invalid_external_path_rejected(self):
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);model=root/'path';phase=root/'phase';d.write_phase(phase,'model-gate');vocab=root/'vocab';vocab.write_bytes(b'vocab');assets={'vocab.gguf':vocab,'client-phase.txt':phase,'model-path.txt':model}
   for value in (b'/data/../model.gguf',b'/data/model.gguf\n',b'/system_ex/app/PPSA99014/model.gguf',b'/data/model.gguf\x00'):
    model.write_bytes(value)
    with self.subTest(value=value),self.assertRaises(ValueError):d.b.validate_client_assets(assets)
 def test_missing_conversion_closure_rejected_without_model_hash(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);model=folder/'tinyllama-q4_0.gguf'
   with model.open('wb') as stream:stream.truncate(619831552)
   lock=d.read('llm-inputs.json')['client'];model_lock=json.loads((d.ROOT/'models/tinyllama-1.1b-chat-v1.0.json').read_text())
   receipt=dict(schema=1,classification='host-model-conversion-verified',model_inference_executed=False,native_executed=False,model_revision=model_lock['revision'],source_revision=lock['revision'],identities={})
   (folder/'receipt.json').write_text(json.dumps(receipt))
   with patch.object(d.graph,'cpu_inputs') as cpu:
    with self.assertRaisesRegex(ValueError,'missing conversion identities'):d.model_inputs(model,folder,folder)
   cpu.assert_not_called()
 def test_macro_is_scoped_to_model_title(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);source=folder/'source';hip=source/'hipamd/src';hip.mkdir(parents=True)
   for i in range(30):(hip/(str(i)+'.cpp')).write_text('')
   def compile_bound(out,path,obj,flags,label,source_root=None):
    calls.append((path,flags));return {'object':str(obj.relative_to(out))}
   for phase in ('vocab','model-gate','model-prefix','model-layer'):
    calls=[];out=folder/phase;out.mkdir()
    with patch.object(d.b,'compiler_flags',return_value=[]),patch.object(d.b,'compile_bound_unit',side_effect=compile_bound),patch.object(d.b,'_plan_bound_unit',side_effect=lambda *a,**kw: ({},lambda _: compile_bound(*a,**kw))):
     units=d.b.compile_units(out,source,'llm_client',client_phase=phase)
    self.assertEqual(len(units),76)
    enabled=[p.name for p,flags in calls if '-DPS5HIP_LLM_AUTHENTICATED_MODEL_GATE=1' in flags]
    self.assertEqual(enabled,['public_hip_llm_client_console_title.cpp'] if phase!='vocab' else [])
    for target in ('PREFIX','LAYER'):
     marked=[p.name for p,flags in calls if '-DPS5HIP_LLM_AUTHENTICATED_MODEL_'+target+'=1' in flags]
     self.assertEqual(marked,['public_hip_llm_client_console_title.cpp'] if phase=='model-'+target.lower() else [])
   with self.assertRaisesRegex(ValueError,'invalid client compile phase'):d.b.compile_units(folder,source,'bootstrap',client_phase='model-gate')
 def test_package_marker_and_streamed_copy(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);packed=folder/'source';final=folder/'final';data=bytearray(0x10600);data[0x1055c:0x10560]=bytes.fromhex('19015419');packed.write_bytes(data)
   d.b.copy_checked_package(packed,final);self.assertEqual(final.read_bytes(),data)
   with self.assertRaises(FileExistsError):d.b.copy_checked_package(packed,final)
   self.assertEqual(final.read_bytes(),data)
   final.unlink();data[0x1055c]=0;packed.write_bytes(data)
   with self.assertRaisesRegex(RuntimeError,'marker'):d.b.copy_checked_package(packed,final)
   self.assertFalse(final.exists());packed.write_bytes(b'short')
   with self.assertRaisesRegex(RuntimeError,'marker'):d.b.copy_checked_package(packed,final)
 def test_duplicate_headers_hashed_once_at_merge_boundary(self):
  with tempfile.TemporaryDirectory() as name:
   p=Path(name)/'header';p.write_bytes(b'one');digest=d.b.sha(p);units=[{'headers':{str(p):digest}} for _ in range(76)];bind=actual_binder()
   with patch.object(d.b,'sha',wraps=d.b.sha) as sha:
    for path,h in d.merged_unit_headers(units).items():bind(path,h)
   self.assertEqual(sha.call_count,1)
 def test_conflicting_compile_identity_rejected_before_hash(self):
  with patch.object(d.b,'sha') as sha:
   with self.assertRaisesRegex(ValueError,'conflicting compiled header'):d.merged_unit_headers([{'headers':{'header':'a'}},{'headers':{'header':'b'}}])
   sha.assert_not_called()
 def test_current_header_drift_rejected_by_original_bind(self):
  with tempfile.TemporaryDirectory() as name:
   p=Path(name)/'header';p.write_bytes(b'before');headers=d.merged_unit_headers([{'headers':{str(p):d.b.sha(p)}}]);p.write_bytes(b'after');bind=actual_binder()
   with self.assertRaisesRegex(ValueError,'identity differs'):
    for path,h in headers.items():bind(path,h)
class LayerPackage(unittest.TestCase):
 def test_exact_six_modules_and_phase_prerequisites(self):
  d.validate_phase_arguments('model-layer','model','source','cpu',layer_fixture='assets')
  for phase,prefix,layer in [('model-layer',None,None),('model-layer','prefix','layer'),('vocab',None,'layer'),('model-prefix','prefix','layer')]:
   with self.assertRaises(ValueError):d.validate_phase_arguments(phase,'model','source','cpu',prefix,layer)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);phase=root/'phase';phase.write_bytes(b'model-layer\n');model=root/'path';model.write_bytes(d.encode_model_path(d.DEFAULT_MODEL_PATH));data=root/'asset';data.write_bytes(b'hsaco')
   assets={'client-phase.txt':phase,'vocab.gguf':data,'model-path.txt':model,**{'model-layer/'+name+'.hsaco':data for name in ('getrows','normalized','weighted','q8','q4','layer')}}
   d.b.validate_client_assets(assets)
   for name in assets:
    bad=dict(assets);del bad[name]
    with self.subTest(missing=name),self.assertRaises(ValueError):d.b.validate_client_assets(bad)
   for name in ('model.gguf','model-prefix/q4.hsaco','model-layer/extra.hsaco'):
    with self.subTest(extra=name),self.assertRaises(ValueError):d.b.validate_client_assets({**assets,name:data})


class OriginalAdmissionReuse(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.source=self.root/'source';self.source.mkdir();self.cpu=self.root/'cpu';self.cpu.mkdir();self.model=self.root/'converted/model.gguf';self.model.parent.mkdir();self.model.write_bytes(b'g')
  self.expected=dict(bytes=619831552,sha256='c42ff3ebdfc0b8a0aa0c35a499ccf81303fb9fbdc1a0798c096a098df85003aa',metadata={},dtype_counts={},tensor_bytes=1,tensor_count=1,parameters=1)
  self.original={}
  for name in ('config.json','tokenizer.model','tokenizer_config.json','tokenizer.json','special_tokens_map.json','model.safetensors'):
   p=self.source/name;p.write_bytes(b'original');self.original[str(p)]=d.b.sha(p)
  lock=dict(revision='model',metadata={Path(p).name:dict(bytes=8,sha256=h) for p,h in self.original.items() if not p.endswith('.safetensors')},weights=dict(file_bytes=8,expected_file_sha256=self.original[str(self.source/'model.safetensors')]))
  def put(name,value):
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
  put('evidence/offline-llm-model-conversion.json',dict(outputs={'tinyllama-q4_0.gguf':self.expected}))
  put('models/tinyllama-1.1b-chat-v1.0.json',lock)
  for name in ('llm-inputs.json','llm-conversion-inputs.json','tools/convert_llm_model.py','tools/fetch_llm_model.py','cpu/receipt.json'):put(name,{})
  environment=put('environment/receipt.json',dict(lock_sha256=d.b.sha(self.root/'llm-conversion-inputs.json')))
  identities={str(p):d.b.sha(p) for p in [*(self.root/n for n in ('llm-inputs.json','llm-conversion-inputs.json','tools/convert_llm_model.py','tools/fetch_llm_model.py','models/tinyllama-1.1b-chat-v1.0.json','cpu/receipt.json')),environment]}
  output=dict(self.expected,tensors=[dict(name='one',elements=1)])
  commands=[dict(label='convert-f16',exit_code=0,argv=['python','convert',str(self.source)]),dict(label='quantize-q4_0',exit_code=0,argv=[str(self.cpu/'cmake/bin/llama-quantize'),'input','output',str(self.model)])]
  (self.model.parent/'receipt.json').write_text(json.dumps(dict(schema=1,classification='host-model-conversion-verified',model_inference_executed=False,native_executed=False,model_revision='model',source_revision='client',identities=identities,outputs={'tinyllama-q4_0.gguf':output},commands=commands)))
  self.addCleanup(patch.stopall);patch.object(d,'ROOT',self.root).start();patch.object(d,'read',return_value={'client':{'revision':'client'}}).start();patch.object(d.graph,'cpu_inputs',return_value={str(self.model):self.expected['sha256']}).start()
  real_stat=Path.stat;model=self.model
  def sized(path,*a,**kw):
   result=real_stat(path,*a,**kw)
   if path!=model:return result
   class Size:
    st_size=619831552
    def __getattr__(self,name):return getattr(result,name)
   return Size()
  patch.object(Path,'stat',sized).start()
 def call(self,**kwargs):return d.model_inputs(self.model,self.source,self.cpu,**kwargs)
 def test_standalone_hashes_and_rejects_content_mutation(self):
  import fetch_llm_model as fetch
  with patch.object(fetch,'verify',wraps=fetch.verify) as verify:self.call();self.assertEqual(verify.call_count,6)
  (self.source/'model.safetensors').write_bytes(b'mutation')
  with self.assertRaisesRegex(ValueError,'SHA256'):self.call()
 def test_authenticated_snapshot_skips_only_original_hashes(self):
  import fetch_llm_model as fetch
  with patch.object(fetch,'verify',wraps=fetch.verify) as verify:
   result=self.call(authenticated_original_inputs=dict(self.original));verify.assert_not_called()
  for p,h in self.original.items():self.assertEqual(result['inputs'][p],h)
  d.graph.cpu_inputs.assert_called_once() # CPU/GGUF authentication is not bypassed.
 def test_each_missing_or_wrong_original_identity_rejects(self):
  for name in self.original:
   for missing in (False,True):
    values=dict(self.original)
    if missing:values.pop(name)
    else:values[name]='0'*64
    with self.subTest(name=name,missing=missing),self.assertRaisesRegex(ValueError,'authenticated inputs'):self.call(authenticated_original_inputs=values)
 def test_authenticated_snapshot_keeps_type_and_size_checks(self):
  p=self.source/'model.safetensors';p.write_bytes(b'short')
  with self.assertRaisesRegex(ValueError,'type or size'):self.call(authenticated_original_inputs=self.original)
  p.unlink();p.symlink_to(self.source/'config.json')
  with self.assertRaisesRegex(ValueError,'type or size'):self.call(authenticated_original_inputs=self.original)

if __name__=='__main__':unittest.main()
