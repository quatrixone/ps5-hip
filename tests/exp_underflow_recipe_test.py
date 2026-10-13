import importlib.util,pathlib,unittest
P=pathlib.Path(__file__).resolve().parents[1]/'tools/experimental/exp-underflow/probe.py'
spec=importlib.util.spec_from_file_location('exp_probe',P);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
spec=importlib.util.spec_from_file_location('regenerate',P.with_name('regenerate_math.py'));regenerate=importlib.util.module_from_spec(spec);spec.loader.exec_module(regenerate)
class Tests(unittest.TestCase):
 def test_unique_natural_constant_only(self):
  original='IsExp10 ? -0x1.66d3e8p+5f : '+probe.OLD
  self.assertEqual(probe.patch_source(original),'IsExp10 ? -0x1.66d3e8p+5f : '+probe.NEW)
  for bad in ('missing',original+original):
   with self.assertRaises(ValueError):probe.patch_source(bad)
 def fixture(self):
  result=''
  for i,n in enumerate(('exp_scalar','exp_vector','exp_fast','exp10_scalar','unrelated_control')):
   result+=n+':\n'+(' v_cmp 0xc2ce8ed0\n v_exp_f32\n v_ldexp_f32\n' if i<2 else ' unchanged\n')+'.Lfunc_end'+str(i)+':\n'
  return result+'.amdhsa_float_denorm_mode_32 3\n'
 def test_exact_paired_matrix(self):
  before=self.fixture();after=before.replace('0xc2ce8ed0','0xc2cff1b4')
  self.assertTrue(probe.audit_matrix(before,before,after)['no_op_exact'])
  for noop,patch in ((before+'changed',after),(before,after.replace(' unchanged',' changed',1)),(before,after.replace('v_ldexp_f32','missing',1)),(before,after.replace('0xc2cff1b4','0xc2cff1b5',1)),(before,after.replace('mode_32 3','mode_32 0'))):
   with self.assertRaises(ValueError):probe.audit_matrix(before,noop,patch)
  for patch in (after+'extra_function:\n s_endpgm\n',after+' .long 123\n',after.replace('mode_32 3','mode_32 3\n.changed')):
   with self.assertRaises(ValueError):probe.audit_matrix(before,before,patch)
 def test_missing_function(self):
  with self.assertRaises(ValueError):probe.function(self.fixture(),'foreign')
 def test_six_operation_two_measured_sites(self):
  a='v_cmp 0xc2ce8ed0\nv_cmp 0xc2ce8ed0\nunchanged\n';b=a.replace('0xc2ce8ed0','0xc2cff1b4')
  self.assertTrue(regenerate.audit_assembly(a,b)['remaining_assembly_exact'])
  for before,after in ((a+a,b+b),(a,b+'extra'),(a,b.replace('unchanged','changed')),(a.replace('v_cmp 0xc2ce8ed0\n','',1),b)):
   with self.assertRaises(ValueError):regenerate.audit_assembly(before,after)
if __name__=='__main__':unittest.main()
