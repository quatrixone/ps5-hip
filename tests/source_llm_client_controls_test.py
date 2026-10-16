from pathlib import Path
import sys,tempfile,unittest,json
from unittest.mock import patch
STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE
sys.path[:0]=[str(STAGE/'tools'),str(ROOT/'tools')]
import source_public_hip_log as log
import source_console as control
import freeze_source_candidate as freeze
class Controls(unittest.TestCase):
 def test_exact_log(self):
  result=log.check_log(log.LLM_CLIENT_EXPECTED,'llm_client');self.assertFalse(result['hardware_gate_passed']);self.assertEqual(result['kernel'],'m0_client_boundary')
 def test_every_log_line_required(self):
  rows=log.LLM_CLIENT_EXPECTED.splitlines(keepends=True)
  for i in range(len(rows)):
   with self.subTest(i=i),self.assertRaises(ValueError):log.check_log(b''.join(rows[:i]+rows[i+1:]),'llm_client')
 def test_mutated_values_fail(self):
  for a,b in [(b'cases=32',b'cases=31'),(b'tokens=236',b'tokens=235'),(b'cleared=16',b'cleared=15'),(b'cpu_compute_calls=0',b'cpu_compute_calls=1'),(b'hip_launches=0',b'hip_launches=1'),(b'allocations=1',b'allocations=2'),(b'retired=1',b'retired=0'),(b'gate_destroyed=true',b'gate_destroyed=false')]:
   with self.subTest(a=a),self.assertRaises(ValueError):log.check_log(log.LLM_CLIENT_EXPECTED.replace(a,b),'llm_client')
 def test_contamination_and_phase_fail(self):
  for suffix in [b'M0_APP LLAMA_ERROR error\n',b'LLM_GATE ACTUAL_CLIENT result=-3\n',b'\n',b'\xff']:
   with self.subTest(suffix=suffix),self.assertRaises(ValueError):log.check_log(log.LLM_CLIENT_EXPECTED+suffix,'llm_client')
  with self.assertRaises(ValueError):log.check_log(log.LLM_CLIENT_EXPECTED,'bootstrap')
 def test_fixture_limits(self):
  self.assertEqual(log.package_limit('llm_client'),33554432)
  for fixture in set(log.LOG_NAMES)-{'llm_client'}:self.assertEqual(log.package_limit(fixture),16777216)
  with self.assertRaises(ValueError):log.package_limit('model-gate')
 def test_baseline_bound_independent_of_new_fixture(self):
  for baseline,expected in [('bootstrap',16777216),('llm_client',33554432)]:
   c=control.Control.__new__(control.Control);c.fixture='llm_client';c.baseline_fixture=baseline;c.manifest={'baseline_image_sha256':'a'*64};c.idle=lambda:None;c.send=lambda n:b''
   with patch.object(control,'snapshot'),patch.object(c,'fetch_image',side_effect=RuntimeError('stop before mutation')) as fetch:
    with self.assertRaisesRegex(RuntimeError,'stop before mutation'):c.deploy()
    fetch.assert_called_once_with(control.REMOTE,expected)
 def test_exact_package_and_phase(self):
  with tempfile.TemporaryDirectory() as tmp:
   build=Path(tmp);app=build/'package/app';app.mkdir(parents=True)
   for name in freeze.LLM_CLIENT_PACKAGE_FILES:
    p=app/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'asset')
   (app/'client-phase.txt').write_bytes(b'vocab\n');(app/'results-dir.txt').write_text('/data/results')
   package=build/'package/PPSA99014.ffpkg';package.write_bytes(b'package')
   receipt={'package_sha256':freeze.digest(package),'package_files':{n:freeze.digest(app/n) for n in freeze.LLM_CLIENT_PACKAGE_FILES},'results_dir':'/data/results'}
   freeze.validate_client_package(build,receipt)
   (app/'model.gguf').write_bytes(b'model')
   with self.assertRaises(ValueError):freeze.validate_client_package(build,receipt)
   (app/'model.gguf').unlink();(app/'client-phase.txt').write_bytes(b'model-gate\n');receipt['package_files']['client-phase.txt']=freeze.digest(app/'client-phase.txt')
   with self.assertRaisesRegex(ValueError,'full model'):freeze.validate_client_package(build,receipt)
 def test_oversize_package(self):
  with tempfile.TemporaryDirectory() as tmp:
   build=Path(tmp);(build/'package').mkdir();package=build/'package/PPSA99014.ffpkg'
   with package.open('wb') as f:f.truncate(33554433)
   with self.assertRaises(ValueError):freeze.validate_client_package(build,{})
 def test_wrong_phase_receipt_rejected(self):
  with self.assertRaisesRegex(ValueError,'complete vocabulary'):freeze.validate_llm_client(Path('.'),{'phase':'model-gate'})
 def test_runtime_unit_closure(self):
  for mode in ('valid','missing-object','duplicate','missing-header','mutated-header','missing-producer','conflicting-header','current-drift','hash-counts'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);original=root/'build';original.mkdir();source=root/'source.cpp';source.write_bytes(b'source');header=root/'header.h';header.write_bytes(b'header')
    inputs={str(p):freeze.digest(p) for p in (source,header)};units=[]
    for i in range(147):
     obj=original/(str(i)+'.o');obj.write_bytes(str(i).encode());inputs[str(obj)]=freeze.digest(obj)
     units.append({'object':str(obj),'object_sha256':freeze.digest(obj),'source':str(source),'source_sha256':freeze.digest(source),'headers':{str(p):freeze.digest(p) for p in (source,header)}})
    for name in ('build_llm_native_app.py','build_public_hip.py','build_llm_native_client.py','build_llm_vocab_fixture.py'):
     path=root/'tools'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(b'producer');inputs[str(path)]=freeze.digest(path)
    receipt={'units':units,'inputs':inputs}
    if mode=='missing-object':del inputs[units[0]['object']]
    elif mode=='duplicate':units[1]=units[0]
    elif mode=='missing-header':del inputs[str(header)]
    elif mode=='mutated-header':header.write_bytes(b'changed')
    elif mode=='conflicting-header':units[76]['headers'][str(header)]='0'*64
    elif mode=='missing-producer':del inputs[str(root/'tools/build_llm_native_app.py')]
    with patch.object(freeze,'ROOT',root):
     if mode=='hash-counts':
      with patch.object(freeze,'digest',wraps=freeze.digest) as hashed:
       freeze.validate_client_units(original,receipt)
       for p in (source,header):self.assertEqual(sum(call.args==(p,) for call in hashed.call_args_list),1)
       self.assertEqual(hashed.call_count,147+2+4)
     elif mode=='current-drift':
      freeze.validate_client_units(original,receipt);header.write_bytes(b'changed after first validation')
      with self.assertRaisesRegex(ValueError,'runtime header'):freeze.validate_client_units(original,receipt)
     elif mode=='conflicting-header':
      with self.assertRaisesRegex(ValueError,'conflicting runtime header'):freeze.validate_client_units(original,receipt)
     elif mode=='valid':freeze.validate_client_units(original,receipt)
     else:
      with self.assertRaises(ValueError):freeze.validate_client_units(original,receipt)
 def test_package_native_image_binding(self):
  with tempfile.TemporaryDirectory() as tmp:
   original=Path(tmp);inputs={};files={}
   for name,source in [('eboot.bin','eboot.bin'),('sce_module/libc.prx','libc.prx')]:
    p=original/source;p.write_bytes(source.encode());inputs[str(p)]=freeze.digest(p);files[name]=freeze.digest(p)
   receipt={'inputs':inputs,'package_files':files};freeze.validate_client_image_assets(original,receipt)
   for name in files:
    before=files[name];files[name]='0'*64
    with self.assertRaises(ValueError):freeze.validate_client_image_assets(original,receipt)
    files[name]=before
if __name__=='__main__':unittest.main()
