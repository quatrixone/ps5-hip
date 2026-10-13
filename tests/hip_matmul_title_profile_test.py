import argparse,copy,hashlib,json,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import native_vector_title as profile
import build_native_vector_title as builder
p=argparse.ArgumentParser();p.add_argument('--compiler',type=pathlib.Path);args,rest=p.parse_known_args()
sys.argv=[sys.argv[0],*rest]

class Profile(unittest.TestCase):
    def test_identity_and_startup(self):
        selected=profile.profile_for('hip_matmul')
        self.assertEqual(selected['shapes_mnk'],[[8,8,8],[13,21,19],[1,1,17],[17,9,1]]);self.assertEqual(selected['dispatches'],8)
        self.assertEqual(selected['kernel_names'],['tiled_matmul'])
        self.assertEqual(profile.provenance_files('hip_matmul'),profile.provenance_files('hip_routes'))
        source='extern "C" void ps5hip_install_shell_exit_probe();\n    ps5hip_install_shell_exit_probe();\n#endif\n    _init();'
        patched=builder.patch_startup(source,'hip_matmul')
        self.assertIn('PS5HIP_MATMUL_TITLE',patched);self.assertNotIn('PS5HIP_ROUTES_TITLE',patched)
        self.assertLess(patched.index('    ps5hip_install_vector_title_finalizer();'),patched.index('    _init();'))
    def test_real_compiler_and_mutations(self):
        if not args.compiler:self.skipTest('fresh title compiler output required')
        raw=(args.compiler/'receipt.json').read_bytes();obj=(args.compiler/'ps5-O2/matmul.o').read_bytes()
        report=profile.validate_matmul_compiler(raw,obj,ROOT)
        self.assertTrue(report['title_entry']);self.assertEqual(len(report['cases']),5)
        commands=json.loads((args.compiler/'commands.json').read_text())
        for host in ('linux-O1','linux-O2','ps5-O1','ps5-O2','linux-model-O2'):
            command=next(c for c in commands if c['label']==host+'-compile')
            self.assertIn('-Dmain=ps5hip_matmul_client_main',command['argv']);self.assertIn('-DPS5HIP_MATMUL_TITLE=1',command['argv'])
        self.assertEqual((args.compiler/'ps5-O2/matmul.hsaco').stat().st_size,profile.MATMUL_PROFILE['code_bytes'])
        for a,b in ((raw+b' ',obj),(raw,obj[:-1]),(raw,obj[:100]+bytes([obj[100]^1])+obj[101:])):
            with self.assertRaises(ValueError):profile.validate_matmul_compiler(a,b,ROOT)
        # Temporarily repin mutated JSON so semantic checks, not the outer hash,
        # reject wrong compiler mode/host selection/function inventory/source.
        original=profile.MATMUL_PROFILE['compiler_receipt_sha256']
        try:
            for kind in range(4):
                wrong=copy.deepcopy(report)
                if kind==0:wrong['title_entry']=False
                elif kind==1:wrong['cases']=[c for c in wrong['cases'] if c['host']!='ps5']
                elif kind==2:next(c for c in wrong['cases'] if c['host']=='ps5')['lifecycle']['functions']=2
                else:wrong['sources']['tests/hip_tiled_matmul_blockdim.hip']='0'*64
                data=json.dumps(wrong).encode();profile.MATMUL_PROFILE['compiler_receipt_sha256']=hashlib.sha256(data).hexdigest()
                with self.assertRaises(ValueError):profile.validate_matmul_compiler(data,obj,ROOT)
        finally:profile.MATMUL_PROFILE['compiler_receipt_sha256']=original

if __name__=='__main__':unittest.main()
