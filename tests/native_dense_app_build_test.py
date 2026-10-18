import importlib.util,tempfile,unittest,sys,subprocess,os
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('dense_app',ROOT/'tools/build_llm_native_app.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class DenseApp(unittest.TestCase):
 def test_phase_requires_only_dense_fixture(self):
  d.validate_phase_arguments('dense-layer',None,None,None,dense_fixture='assets')
  for args,kw in [((None,None,None),{}),(('model',None,None),{'dense_fixture':'assets'}),((None,None,None),{'dense_fixture':'assets','layer_fixture':'layer'})]:
   with self.assertRaises(ValueError):d.validate_phase_arguments('dense-layer',*args,**kw)
  with self.assertRaises(ValueError):d.validate_phase_arguments('vocab',None,None,None,dense_fixture='assets')
 def test_exact_package_no_model_or_vocab(self):
  with tempfile.TemporaryDirectory() as name:
   f=Path(name);phase=f/'phase';d.write_phase(phase,'dense-layer');module=f/'module';module.write_bytes(b'ELF')
   assets={'client-phase.txt':phase,**{'dense-layer/'+n+'.hsaco':module for n in ('getrows','normalized','weighted','q8','q4','layer')}}
   d.b.validate_client_assets(assets)
   for key in ('vocab.gguf','model-path.txt','original-model.json','model.gguf'):
    with self.assertRaises(ValueError):d.b.validate_client_assets({**assets,key:module})
   del assets['dense-layer/q8.hsaco']
   with self.assertRaises(ValueError):d.b.validate_client_assets(assets)
 def test_audit_explicit(self):
  d.validate_dense_audit({'phase':'dense-layer','actual_dense_call_chain_audited':True})
  for a in ({},{'phase':'dense-layer','actual_client_call_chain_audited':True},{'phase':'model-layer','actual_dense_call_chain_audited':True}):
   with self.assertRaises(ValueError):d.validate_dense_audit(a)
 def test_compile_dense_macro_and_default(self):
  b=d.b
  with tempfile.TemporaryDirectory() as name:
   f=Path(name);hip=f/'hipamd/src';hip.mkdir(parents=True)
   for i in range(30):(hip/(str(i)+'.cpp')).touch()
   for phase in ('dense-layer','vocab'):
    seen=[]
    def unit(out,path,obj,flags,*args,**kw):seen.append((path.name,flags));return {'object':str(obj)}
    with patch.object(b,'compiler_flags',return_value=[]),patch.object(b,'build_jobs',return_value=1),patch.object(b,'compile_bound_unit',side_effect=unit),patch.object(b,'_plan_bound_unit',side_effect=unit),patch.object(b,'_compile_bound_plans',side_effect=lambda plans,jobs:plans):
     self.assertEqual(len(b.compile_units(f,f,'llm_client',client_phase=phase)),76)
    flags=next(v for n,v in seen if n=='public_hip_llm_client_console_title.cpp')
    self.assertEqual('-DPS5HIP_LLM_AUTHENTICATED_DENSE_LAYER=1' in flags,phase=='dense-layer')
    self.assertNotIn('-DPS5HIP_LLM_AUTHENTICATED_MODEL_GATE=1',flags)
 def test_make_explicit_phase(self):
  env={k:v for k,v in os.environ.items() if not k.startswith('LLM_') and k not in ('MAKEFLAGS','MAKEOVERRIDES','MFLAGS')}
  p=subprocess.run(['make','-f',str(ROOT/'Makefile'),'-n','llm-native-dense-app','MAKE=echo','JOBS=1','UFS_REV=x','ZLIB_VERSION=x','TITLE_ID=x'],cwd=ROOT,env=env,text=True,capture_output=True)
  self.assertEqual(p.returncode,0,p.stderr);self.assertIn('LLM_NATIVE_APP_PHASE=dense-layer',p.stdout)
if __name__=='__main__':unittest.main()
