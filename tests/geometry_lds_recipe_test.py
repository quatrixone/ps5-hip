import pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import probe_geometry_lds as recipe

class Tests(unittest.TestCase):
    def test_cfg_edges_barriers_and_malformed_targets(self):
        text='''define protected amdgpu_kernel void @tiled_matmul(i32 %k) {
entry:
  br label %loop
loop:
  call void @llvm.amdgcn.s.barrier()
  call void @llvm.amdgcn.s.barrier()
  br i1 true, label %loop, label %end
end:
  ret void
}
'''
        cfg=recipe.audit_cfg(text)
        self.assertEqual(cfg['blocks']['loop']['successors'],['loop','end'])
        self.assertEqual(cfg['barrier_sites'],2)
        self.assertFalse(cfg['barrier_uniformity_automatically_proven'])
        for bad in (text.replace('label %end','label %missing'),text.replace('end:','loop:'),text.replace('@tiled_matmul','@other')):
            with self.assertRaises(ValueError):recipe.audit_cfg(bad)

    def test_genuine_constructor_binding_and_launch(self):
        text='''0000 <__hip_module_ctor>:
 R_X86_64_PLT32 __hipRegisterFatBinary-0x4
 R_X86_64_PLT32 __hipRegisterFunction-0x4
 R_X86_64_REX_GOTPCRELX tiled_matmul-0x4
 R_X86_64_PLT32 atexit-0x4
0010 <__hip_module_dtor>:
 R_X86_64_PLT32 __hipUnregisterFatBinary-0x4
0020 <main>:
 R_X86_64_PLT32 __hipPushCallConfiguration-0x4
 R_X86_64_PLT32 __hipPopCallConfiguration-0x4
 R_X86_64_PLT32 hipLaunchKernel-0x4
'''
        self.assertTrue(recipe.audit_host(text)['ordinary_launch_stub'])
        for symbol in ('__hipRegisterFunction','tiled_matmul','__hipUnregisterFatBinary','__hipPopCallConfiguration'):
            with self.assertRaises(ValueError):recipe.audit_host(text.replace(symbol,'foreign'))
        with self.assertRaises(ValueError):recipe.audit_host(text.replace(' R_X86_64_PLT32 atexit-0x4',' R_X86_64_PLT32 atexit-0x4\n R_X86_64_PLT32 __hipRegisterFunction-0x4'))

    def test_oracle_edges_and_both_inputs_change(self):
        cases=recipe.oracle_cases();self.assertEqual(len(cases),4)
        main=next(c for c in cases if c['m']==13)
        self.assertEqual((main['groups'],main['k_tiles'],main['logical_outputs'],main['inactive_threads']),([3,2,1],3,273,111))
        for case in cases:
            self.assertEqual(len(case['rounds'][0]['output_bits']),case['m']*case['n'])
            self.assertNotEqual(case['rounds'][0]['input_a'],case['rounds'][1]['input_a'])
            self.assertNotEqual(case['rounds'][0]['input_b'],case['rounds'][1]['input_b'])

if __name__=='__main__':unittest.main()
