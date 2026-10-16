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
   with patch('fetch_llm_model.verify') as verify:
    d.b.validate_client_assets({**base,'model.gguf':model});self.assertEqual(verify.call_args.args[1],619831552)
   phase.write_bytes(b'model-gate')
   with self.assertRaises(ValueError):d.b.validate_client_assets({**base,'model.gguf':model})
 def test_short_and_symlink_models_reject_before_receipt_or_hash(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);model=folder/'model';model.write_bytes(b'short');link=folder/'alias';link.symlink_to(model)
   with patch.object(d.graph,'cpu_inputs') as cpu:
    with self.assertRaisesRegex(ValueError,'size differs'):d.model_inputs(model,folder,folder)
    with self.assertRaisesRegex(ValueError,'regular'):d.model_inputs(link,folder,folder)
   cpu.assert_not_called()
 def test_corrupt_packaged_model_rejected_with_small_fake_profile(self):
  with tempfile.TemporaryDirectory() as name:
   root=Path(name);(root/'evidence').mkdir();model=root/'model';model.write_bytes(b'right');expected={'bytes':5,'sha256':d.b.sha(model)}
   (root/'evidence/offline-llm-model-conversion.json').write_text(json.dumps({'outputs':{'tinyllama-q4_0.gguf':expected}}));phase=root/'phase';d.write_phase(phase,'model-gate');vocab=root/'vocab';vocab.write_bytes(b'vocab');assets={'vocab.gguf':vocab,'client-phase.txt':phase,'model.gguf':model}
   with patch.object(d.b,'ROOT',root):
    d.b.validate_client_assets(assets);model.write_bytes(b'wrong')
    with self.assertRaisesRegex(ValueError,'SHA256'):d.b.validate_client_assets(assets)
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
   def compile_bound(out,path,obj,flags,label):
    calls.append((path,flags));return {'object':str(obj.relative_to(out))}
   for phase in ('vocab','model-gate'):
    calls=[];out=folder/phase;out.mkdir()
    with patch.object(d.b,'compiler_flags',return_value=[]),patch.object(d.b,'compile_bound_unit',side_effect=compile_bound):
     units=d.b.compile_units(out,source,'llm_client',client_phase=phase)
    self.assertEqual(len(units),76)
    enabled=[p.name for p,flags in calls if '-DPS5HIP_LLM_AUTHENTICATED_MODEL_GATE=1' in flags]
    self.assertEqual(enabled,['public_hip_llm_client_console_title.cpp'] if phase=='model-gate' else [])
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
if __name__=='__main__':unittest.main()
