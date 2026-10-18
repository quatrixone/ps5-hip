import os
import subprocess
import sys
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_layer_fixture as m
class LayerFixtureTest(unittest.TestCase):
 def test_inventory(self):
  self.assertEqual(len(m.SYMBOLS),16);self.assertEqual(len(set(m.SYMBOLS)),16)
 def test_reject_outside_output(self):
  with self.assertRaisesRegex(ValueError,'outside'):m.build('/unused','/tmp/untrusted-layer-output')
 def test_reject_wrong_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'receipt.json').write_text('{"classification":"wrong"}')
   with self.assertRaisesRegex(ValueError,'invalid'):m.fixture_inputs(d)
 def test_undefined_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'undefined.stdout').write_text('U forbidden')
   with self.assertRaisesRegex(ValueError,'undefined'):m.contracts(d)
 def test_missing_symbol_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for n in ('undefined.stdout','notes.stdout','isa.stdout','admission.stdout'): (p/n).write_text('')
   (p/'layer.hsaco').write_bytes(b'fake')
   (p/'isa.stdout').write_text('<layer_q4_cpu>:\n global_load_ubyte '+ 'global_load_ubyte '*15)
   with patch.object(m,'audit_object',return_value={}):
    with self.assertRaisesRegex(ValueError,'admission'):m.contracts(d)
 def test_artifact_identity_and_missing_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);artifacts={}
   for n in m.REQUIRED:
    data=b'[]' if n=='commands.json' else b'fixture'
    (folder/n).write_bytes(data);artifacts[n]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   report=dict(classification=m.CLASSIFICATION,status='PASS',native_executed=False,original_fixture=str(folder/'parent'),artifacts=artifacts,commands=[])
   (folder/'receipt.json').write_text(json.dumps(report))
   with patch.object(m,'original_inputs',return_value={}),patch.object(m,'checked_inputs',return_value={}),patch.object(m,'checked_tools',return_value={}):
    (folder/'layer.hsaco').write_bytes(b'wrong')
    with self.assertRaisesRegex(ValueError,'artifact identity'):m.fixture_inputs(folder)
    del report['artifacts']['layer.hsaco'];(folder/'receipt.json').write_text(json.dumps(report))
    with self.assertRaisesRegex(ValueError,'required artifact'):m.fixture_inputs(folder)
 def test_dense_explicit_profile(self):
  self.assertEqual(len(m.selected_symbols('dense-f16')),17)
  self.assertEqual(m.selected_symbols('dense-f16')[:-1],m.SYMBOLS)
  with self.assertRaisesRegex(ValueError,'unknown'):m.selected_symbols('other')
 def test_dense_receipt_rejected_by_default_consumer(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'receipt.json').write_text(json.dumps(dict(classification=m.DENSE_CLASSIFICATION,status='PASS',native_executed=False,kernel_profile='dense-f16')))
   with self.assertRaisesRegex(ValueError,'invalid'):m.fixture_inputs(d)
 def test_dense_descriptor_and_isa_contract(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);symbols=m.selected_symbols('dense-f16')
   (p/'undefined.stdout').write_text('');(p/'notes.stdout').write_text('');(p/'layer.hsaco').write_bytes(b'fake')
   isa='<layer_q4_cpu>:\n'+'global_load_ubyte '*16+'\n0000 <layer_dense_f16>:\n'+'v_mad_u64_u32 v_lshlrev_b64 v_add_co_u32 v_add_co_ci_u32 '+'v_fma_mix_f32 '*32+'v_add_f32_e32 '*31
   (p/'isa.stdout').write_text(isa);(p/'admission.stdout').write_text('\n'.join(n+' rejection=' for n in symbols))
   descriptor=dict(kernarg_bytes=32,rsrc2=0x8c,properties=0x409,group_bytes=0,private_bytes=0,rsrc3=0,preload=0)
   audit={'descriptors':{'layer_dense_f16':descriptor}}
   with patch.object(m,'audit_object',return_value=audit):
    self.assertEqual(m.contracts(p,'dense-f16'),audit)
    descriptor['kernarg_bytes']=28
    with self.assertRaisesRegex(ValueError,'descriptor'):m.contracts(p,'dense-f16')
    descriptor['kernarg_bytes']=32;(p/'isa.stdout').write_text(isa.replace('v_mad_u64_u32','v_mul_lo_u32'))
    with self.assertRaisesRegex(ValueError,'address/reduction'):m.contracts(p,'dense-f16')
 def test_dense_missing_isa_symbol_is_value_error(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for n in ('undefined.stdout','notes.stdout','admission.stdout'): (p/n).write_text('')
   (p/'layer.hsaco').write_bytes(b'fake');(p/'isa.stdout').write_text('<layer_q4_cpu>:\n'+'global_load_ubyte '*16)
   (p/'admission.stdout').write_text('\n'.join(n+' rejection=' for n in m.selected_symbols('dense-f16')))
   with patch.object(m,'audit_object',return_value={'descriptors':{}}):
    with self.assertRaisesRegex(ValueError,'dense ISA symbol'):m.contracts(p,'dense-f16')
 def test_dense_make_reuses_existing_prerequisite(self):
  with tempfile.TemporaryDirectory() as d:
   rms=Path(d)/'rms';dense=Path(d)/'dense';rms.mkdir();dense.mkdir()
   for p in (rms,dense):(p/'receipt.json').write_text('{}')
   env={k:v for k,v in os.environ.items() if k not in ('MAKEFLAGS','MFLAGS','MAKEOVERRIDES','PYTHON','MAKE')}
   argv=['make','-f',str(m.ROOT/'Makefile'),'llm-dense-layer-fixture','PYTHON=echo','MAKE=false','JOBS=1','UFS_REV=x','ZLIB_VERSION=x','TITLE_ID=x','LLM_RMSNORM_FIXTURE_DIR='+str(rms),'LLM_DENSE_LAYER_FIXTURE_DIR='+str(dense)]
   result=subprocess.run(argv,cwd=m.ROOT,env=env,text=True,capture_output=True,timeout=20)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('fixture_inputs(Path(sys.argv[1]).resolve(),"dense-f16")',result.stdout)
   dry=subprocess.run(argv[:]+['-n'],cwd=m.ROOT,env=env,text=True,capture_output=True,timeout=20)
   self.assertEqual(dry.returncode,0,dry.stderr);self.assertIn('--profile dense-f16',dry.stdout)
if __name__=='__main__':unittest.main()

