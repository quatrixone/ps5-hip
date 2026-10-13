import argparse,copy,hashlib,json,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import native_vector_title as profile
import build_native_vector_title as builder
p=argparse.ArgumentParser();p.add_argument('--compiler',type=pathlib.Path);args,rest=p.parse_known_args()
sys.argv=[sys.argv[0],*rest]

class Profile(unittest.TestCase):
    def test_identity_and_startup(self):
        selected=profile.profile_for('hip_generic')
        self.assertEqual(selected['sequence_launches'],[0,1,3,7]);self.assertEqual(selected['dispatches'],11)
        self.assertEqual(selected['kernel_names'],['affine_words','block_sum'])
        self.assertEqual(profile.provenance_files('hip_generic'),profile.provenance_files('hip_routes'))
        source='extern "C" void ps5hip_install_shell_exit_probe();\n    ps5hip_install_shell_exit_probe();\n#endif\n    _init();'
        patched=builder.patch_startup(source,'hip_generic')
        self.assertIn('PS5HIP_GENERIC_TITLE',patched);self.assertNotIn('PS5HIP_ROUTES_TITLE',patched)
        self.assertLess(patched.index('    ps5hip_install_vector_title_finalizer();'),patched.index('    _init();'))
    def test_real_compiler_and_mutations(self):
        if not args.compiler:self.skipTest('fresh title compiler output required')
        raw=(args.compiler/'receipt.json').read_bytes();obj=(args.compiler/'ps5/routes.o').read_bytes()
        report=profile.validate_generic_compiler(raw,obj,ROOT)
        self.assertTrue(report['title_entry']);self.assertEqual(len(report['cases']),6)
        commands=json.loads((args.compiler/'commands.json').read_text())
        for host in ('linux','ps5','linux-model'):
            command=next(c for c in commands if c['label']==host+'-compile')
            self.assertIn('-Dmain=ps5hip_generic_client_main',command['argv']);self.assertIn('-DPS5HIP_GENERIC_TITLE=1',command['argv'])
        self.assertEqual((args.compiler/'ps5/routes.hsaco').stat().st_size,profile.GENERIC_PROFILE['code_bytes'])
        for a,b in ((raw+b' ',obj),(raw,obj[:-1]),(raw,obj[:100]+bytes([obj[100]^1])+obj[101:])):
            with self.assertRaises(ValueError):profile.validate_generic_compiler(a,b,ROOT)
        # Temporarily repin mutated JSON so semantic checks, not the outer hash,
        # reject wrong compiler mode/host selection/function inventory/source.
        original=profile.GENERIC_PROFILE['compiler_receipt_sha256']
        try:
            for kind in range(4):
                wrong=copy.deepcopy(report)
                if kind==0:wrong['title_entry']=False
                elif kind==1:wrong['cases']=[c for c in wrong['cases'] if c['profile']!='ps5']
                elif kind==2:next(c for c in wrong['cases'] if c['profile']=='ps5')['registration']['register_function_calls']=3
                else:wrong['sources']['tests/hip_generic.hip']='0'*64
                data=json.dumps(wrong).encode();profile.GENERIC_PROFILE['compiler_receipt_sha256']=hashlib.sha256(data).hexdigest()
                with self.assertRaises(ValueError):profile.validate_generic_compiler(data,obj,ROOT)
        finally:profile.GENERIC_PROFILE['compiler_receipt_sha256']=original

if __name__=='__main__':unittest.main()
