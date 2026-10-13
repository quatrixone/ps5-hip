#!/usr/bin/env python3
"""Measured HIPIFY native profile boundaries and optional actual native artifact checks."""
import argparse,copy,hashlib,json,pathlib,sys,tempfile,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import native_vector_title as profile
import build_native_vector_title as recipe
import stage_native_title as stager
parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--compiler',type=pathlib.Path);parser.add_argument('--native-build',type=pathlib.Path);args,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]
sha=lambda data:hashlib.sha256(data).hexdigest()

class HipifyNative(unittest.TestCase):
    def compiler(self):
        if not args.compiler:self.skipTest('provide --compiler for actual fixed combined object')
        return (args.compiler/'receipt.json').read_bytes(),(args.compiler/'ps5/translated.o').read_bytes()
    def receipt(self):
        required=['tools/build_native_vector_title.py','tools/native_vector_title.py','tools/hip_vector_title.cpp','src/hip_runtime.cpp','include/ps5hip/hip_runtime_control.h','tools/native_shell_exit_probe.cpp','tools/native_runtime.h','evidence/offline-compatible-hip-headers.json','src/hip_launch_profile.cpp','tests/cuda_vector_add.cu','tools/probe_hipify_client.py','tools/probe_hip_routes.py','docs/HIPIFY_NATIVE_BUILD.md','tests/hipify_native_title_test.py']
        return {'target':'gfx1013','kernel':'hipify','shutdown_mode':'shell-request-probe','source_commit':'a'*40,'source_tree_clean':True,'hardware_validated':False,'hipify_profile':copy.deepcopy(profile.HIPIFY_PROFILE),'source_sha256':{n:sha((ROOT/n).read_bytes()) for n in required},'artifacts':{n:sha(('synthetic '+n).encode()) for n in ('eboot.bin','lapy.elf','sce_module/libc.prx')},'routes_audits':{'raw_pie':{'bundle_sha256':profile.HIPIFY_PROFILE['bundle_sha256']},'converted_eboot':{'bundle_sha256':profile.HIPIFY_PROFILE['bundle_sha256']},'startup':{'install_order':['shell','vector_finalizer','_init','main'],'compiler_atexit_unregister':True},'fself_roundtrip':{'matches_except_unmapped_sie_note':True}},'hipify_provenance':{n:sha(b'synthetic provenance') for n in profile.provenance_files('hipify')}}
    def test_measured_profile_and_ordinary_signature(self):
        p=profile.profile_for('hipify');self.assertEqual(p['client_entry'],'_Z18hipify_client_mainv');self.assertEqual(p['kernel_names'],['translated_add']);self.assertEqual((p['dispatches'],p['exact_values'],p['word_comparisons']),(6,662,2238));self.assertEqual(len(profile.provenance_files('hipify')),10)
    def test_actual_compiler(self):
        receipt,combined=self.compiler();r=profile.validate_hipify_compiler(receipt,combined,ROOT);self.assertEqual(r['generated_source_hand_edits'],0)
    def test_builder_selects_actual_compiler_object(self):
        for kernel,name in (('hip_routes','ps5/routes.o'),('hip_generic','ps5/routes.o'),('hip_matmul','ps5-O2/matmul.o'),('hipify','ps5/translated.o')):
            self.assertEqual(recipe.compiler_object(pathlib.Path('probe'),kernel),pathlib.Path('probe')/name)
        receipt,_=self.compiler()
        combined=recipe.compiler_object(args.compiler,'hipify').read_bytes()
        profile.validate_hipify_compiler(receipt,combined,ROOT)
    def test_changed_receipt_or_combined_rejected(self):
        receipt,combined=self.compiler()
        for a,b in ((receipt+b' ',combined),(receipt,combined[:-1]+bytes([combined[-1]^1]))):
            with self.assertRaises(ValueError):profile.validate_hipify_compiler(a,b,ROOT)
    def test_semantic_descriptor_and_mode_negatives(self):
        receipt,combined=self.compiler();base=json.loads(receipt)
        for change in ('edited','entry','executed','descriptor','registration','artifact'):
            r=copy.deepcopy(base)
            if change=='edited':r['generated_source_hand_edits']=1
            elif change=='entry':r['original_main_renamed_by_flag']=False
            elif change=='executed':r['hardware_executed']=True
            elif change=='descriptor':next(c for c in r['cases'] if c['host']=='ps5')['audit']['descriptors']['translated_add']['private_bytes']=4
            elif change=='registration':next(c for c in r['cases'] if c['host']=='ps5')['lifecycle']['register_function_calls']=2
            else:r['artifacts']['ps5/translated.hsaco']='0'*64
            raw=json.dumps(r).encode()
            with mock.patch.dict(profile.HIPIFY_PROFILE,{'compiler_receipt_sha256':sha(raw)}):
                with self.assertRaises(ValueError):profile.validate_hipify_compiler(raw,combined,ROOT)
    def test_original_source_and_current_compiler_dependencies_bound(self):
        receipt,combined=self.compiler();r=json.loads(receipt)
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder)
            for n in r['sources']:
                if n in ('CMakeLists.txt','tests/hipify_runtime_model.cpp'):continue
                p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
            profile.validate_hipify_compiler(receipt,combined,root)
            (root/'tests/cuda_vector_add.cu').write_bytes(b'edited')
            with self.assertRaises(ValueError):profile.validate_hipify_compiler(receipt,combined,root)
    def test_profile_resources_and_source_rejection(self):
        r=self.receipt();profile.validate_profile(r,ROOT)
        for key,value in (('target','gfx1030'),('shutdown_mode','unsupported'),('hardware_validated',True)):
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):profile.validate_profile(bad,ROOT)
        bad=copy.deepcopy(r);bad['hipify_profile']['dispatches']=5
        with self.assertRaises(ValueError):profile.validate_profile(bad,ROOT)
        bad=copy.deepcopy(r);bad['source_sha256']['tests/cuda_vector_add.cu']='0'*64
        with self.assertRaises(ValueError):profile.validate_profile(bad,ROOT)
    def test_audit_booleans_cannot_authorize_signed_image(self):
        r=self.receipt()
        with self.assertRaises(ValueError):profile.validate_build(r,ROOT,signed=b'arbitrary signed bytes')
    def test_actual_startup_requires_noarg_entry_and_one_registration(self):
        text='''00000000 <_start>:
 call <ps5hip_install_shell_exit_probe>
 call <ps5hip_install_vector_title_finalizer>
 call <_init>
 call <main>
00000100 <__hip_module_ctor>:
 call <__hipRegisterFunction>
 call <atexit>
00000200 <__hip_module_dtor>:
 call <__hipUnregisterFatBinary>
00000300 <ps5hip_install_vector_title_finalizer>:
 call <atexit>
00000400 <main>:
 call <_Z18hipify_client_mainv>
00000500 <_Z18hipify_client_mainv>:
 ret
'''
        self.assertEqual(recipe.audit_startup(text,kernel='hipify')['client_entry'],'_Z18hipify_client_mainv')
        for bad in (text.replace('_Z18hipify_client_mainv','_Z18hipify_client_mainiPPc'),text.replace('call <__hipRegisterFunction>',''),text.replace('call <__hipRegisterFunction>','call <__hipRegisterFunction>\n call <__hipRegisterFunction>')):
            with self.assertRaises(RuntimeError):recipe.audit_startup(bad,kernel='hipify')
    def test_stage_uses_embedded_profile_and_no_external_canary(self):
        r=self.receipt()
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);build=root/'build';(build/'candidate').mkdir(parents=True);(build/'provenance').mkdir()
            for n in r['artifacts']:
                p=build/'candidate'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('synthetic '+n).encode())
            for n in r['hipify_provenance']:(build/'provenance'/n).write_bytes(b'synthetic provenance')
            (build/'receipt.json').write_text(json.dumps(r))
            # This isolated test checks staging shape, not fabricated ELF admission.
            with mock.patch.object(profile,'validate_build',side_effect=lambda a,b,*c:profile.validate_profile(a,b)):
                m=stager.stage(ROOT,build,root/'stage',{'title_id':'PPSA99014','title_name':'ps5-hip'},'a'*40,'/ordinary/results')
            self.assertNotIn('canary.hsaco',m['files']);self.assertEqual(m['hipify_profile'],profile.HIPIFY_PROFILE);self.assertTrue((root/'stage/hipify-provenance/routes-combined.o').is_file())
    def test_actual_native_binding_and_replacement_rejection(self):
        if not args.native_build:self.skipTest('native offline build not yet qualified')
        r=json.loads((args.native_build/'receipt.json').read_text());provenance=args.native_build/'provenance';signed=(args.native_build/'candidate/eboot.bin').read_bytes();profile.validate_build(r,ROOT,provenance,signed)
        replacement=bytearray(signed);replacement[-1]^=1;bad=copy.deepcopy(r);bad['artifacts']['eboot.bin']=sha(replacement)
        with self.assertRaises(ValueError):profile.validate_build(bad,ROOT,provenance,bytes(replacement))
        bad=copy.deepcopy(r);del bad['routes_audits']['fself_roundtrip']['app_crt_sha256']
        with self.assertRaises(ValueError):profile.validate_build(bad,ROOT,provenance,signed)

if __name__=='__main__':unittest.main()
