import argparse,copy,hashlib,json,pathlib,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import native_vector_title as profile
import build_native_vector_title as builder
p=argparse.ArgumentParser();p.add_argument('--memory-fill',action='store_true');p.add_argument('--device-copy',action='store_true');p.add_argument('--compiler',type=pathlib.Path);args,rest=p.parse_known_args()
sys.argv=[sys.argv[0],*rest]
KERNEL='hip_device_copy' if args.device_copy else 'hip_memory_fill' if args.memory_fill else 'hip_generic'
SELECTED=profile.profile_for(KERNEL)
validate=profile.validate_device_copy_compiler if args.device_copy else profile.validate_memory_fill_compiler if args.memory_fill else profile.validate_generic_compiler

class Profile(unittest.TestCase):
    def test_compiler_dependency_boundary(self):
        if not args.compiler:self.skipTest('fresh title compiler output required')
        raw=(args.compiler/'receipt.json').read_bytes();obj=(args.compiler/'ps5/routes.o').read_bytes();report=json.loads(raw)
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            for name in report['sources']:
                target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
            for name in ('CMakeLists.txt','src/hip_runtime.cpp','tests/hip_generic_model.cpp'):
                self.assertNotIn(name,report['sources']);target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('independent edit')
            validate(raw,obj,root)
            for name in ('tests/hip_generic.hip','include/ps5hip/hip_runtime_control.h','include/ps5hip/native_memory.h','tools/probe_hip_generic.py',*(['tests/hip_device_copy_client.h'] if args.device_copy else ['tests/hip_memory_fill_client.h'] if args.memory_fill else [])):
                target=root/name;original=target.read_bytes();target.write_bytes(original+b'\n')
                with self.assertRaises(ValueError):validate(raw,obj,root)
                target.write_bytes(original)
    def test_identity_and_startup(self):
        selected=profile.profile_for(KERNEL)
        self.assertEqual(selected['dispatches'],24 if args.device_copy else 6 if args.memory_fill else 11)
        if args.device_copy:self.assertEqual((selected['copies'],selected['readback_words'],selected['exact_values']),(12,5926,2648))
        elif args.memory_fill:self.assertEqual((selected['fills'],selected['output_words'],selected['input_words']),(24,746,746))
        else:self.assertEqual(selected['sequence_launches'],[0,1,3,7])
        self.assertEqual(selected['kernel_names'],['affine_words','block_sum'])
        self.assertEqual(profile.provenance_files('hip_generic'),profile.provenance_files('hip_routes'))
        source='extern "C" void ps5hip_install_shell_exit_probe();\n    ps5hip_install_shell_exit_probe();\n#endif\n    _init();'
        patched=builder.patch_startup(source,KERNEL)
        self.assertIn('PS5HIP_DEVICE_COPY_TITLE' if args.device_copy else 'PS5HIP_MEMORY_FILL_TITLE' if args.memory_fill else 'PS5HIP_GENERIC_TITLE',patched);self.assertNotIn('PS5HIP_ROUTES_TITLE',patched)
        self.assertLess(patched.index('    ps5hip_install_vector_title_finalizer();'),patched.index('    _init();'))
    def test_real_compiler_and_mutations(self):
        if not args.compiler:self.skipTest('fresh title compiler output required')
        raw=(args.compiler/'receipt.json').read_bytes();obj=(args.compiler/'ps5/routes.o').read_bytes()
        report=validate(raw,obj,ROOT)
        self.assertTrue(report['title_entry']);self.assertEqual(len(report['cases']),6)
        commands=json.loads((args.compiler/'commands.json').read_text())
        for host in ('linux','ps5','linux-model'):
            command=next(c for c in commands if c['label']==host+'-compile')
            self.assertIn('-Dmain=ps5hip_device_copy_client_main' if args.device_copy else '-Dmain=ps5hip_memory_fill_client_main' if args.memory_fill else '-Dmain=ps5hip_generic_client_main',command['argv']);self.assertIn('-DPS5HIP_GENERIC_TITLE=1',command['argv'])
        self.assertEqual((args.compiler/'ps5/routes.hsaco').stat().st_size,SELECTED['code_bytes'])
        for a,b in ((raw+b' ',obj),(raw,obj[:-1]),(raw,obj[:100]+bytes([obj[100]^1])+obj[101:])):
            with self.assertRaises(ValueError):validate(a,b,ROOT)
        # Temporarily repin mutated JSON so semantic checks, not the outer hash,
        # reject wrong compiler mode/host selection/function inventory/source.
        original=SELECTED['compiler_receipt_sha256']
        try:
            for kind in range(5):
                wrong=copy.deepcopy(report)
                if kind==0:wrong['title_entry']=False
                elif kind==1:wrong['cases']=[c for c in wrong['cases'] if c['profile']!='ps5']
                elif kind==2:next(c for c in wrong['cases'] if c['profile']=='ps5')['registration']['register_function_calls']=3
                elif kind==3:wrong['sources']['tests/hip_generic.hip']='0'*64
                else:wrong['memory_fill']=not args.memory_fill
                data=json.dumps(wrong).encode();SELECTED['compiler_receipt_sha256']=hashlib.sha256(data).hexdigest()
                with self.assertRaises(ValueError):validate(data,obj,ROOT)
        finally:SELECTED['compiler_receipt_sha256']=original

if __name__=='__main__':unittest.main()
