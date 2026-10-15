"""RMSNorm freeze/freshness/readback controls with no network or console access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_console as control
import freeze_source_candidate as freeze


class RmsnormControlTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.c=control.Control.__new__(control.Control)
        self.c.fixture='rmsnorm';self.c.out=self.root;self.c.record={}
        self.c.manifest={'results_dir':'/data/ps5-hip/results','rmsnorm':{'reference':'frozen-reference'}}

    def test_all_dump_names_must_be_absent(self):
        self.c.listing=lambda directory:[]
        self.c.fresh_rmsnorm_dumps()
        self.assertEqual(len(self.c.record['rmsnorm_raw_absent_before_launch']),12)
        for name in control.RMSNORM_RAW_NAMES:
            self.c.listing=lambda directory,n=name:['-rw-r--r-- 1 root root 1 now '+n]
            with self.subTest(name=name),self.assertRaisesRegex(RuntimeError,'already exists'):
                self.c.fresh_rmsnorm_dumps()

    def test_collect_exact_files_hashes_and_no_overwrite(self):
        payload=b'a'*control.RMSNORM_RAW_BYTES
        calls=[]
        def fetch(path,bound):
            self.assertEqual(bound,len(payload));calls.append(path);return payload
        self.c.fetch=fetch
        self.c.collect_rmsnorm_dumps()
        self.assertEqual(len(calls),12)
        self.assertEqual(set(self.c.record['rmsnorm_raw_files']),set(control.RMSNORM_RAW_NAMES))
        self.c.collect_rmsnorm_dumps()
        self.assertEqual(len(calls),12)
        first=self.root/'raw'/control.RMSNORM_RAW_NAMES[0]
        first.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'changed'):self.c.collect_rmsnorm_dumps()

    def test_partial_or_missing_readback_cannot_pass(self):
        self.c.fetch=lambda *args:b'short'
        with self.assertRaisesRegex(RuntimeError,'length'):self.c.collect_rmsnorm_dumps()
        self.assertFalse(self.c.record['rmsnorm_raw_files'])
        self.assertEqual((self.root/'raw'/control.RMSNORM_RAW_NAMES[0]).read_bytes(),b'short')
        self.c.fetch=lambda *args:(_ for _ in ()).throw(RuntimeError('missing'))
        self.c.collect_rmsnorm_dumps(partial=True)
        self.assertEqual(len(self.c.record['rmsnorm_raw_capture_errors']),12)
        self.assertNotIn('rmsnorm_numerical',self.c.record)

    def test_transcript_alone_never_sets_numerical_success(self):
        self.c.record['application']={'hardware_gate_passed':False,'reported_metrics':{'max_abs':1.23456789e-6,'max_budget_fraction':0.25}}
        calls=[];self.c.verify=lambda:calls.append('frozen')
        with patch('verify_public_hip_rmsnorm.verify',return_value={'status':'FAIL'}):
            with self.assertRaisesRegex(RuntimeError,'verification failed'):self.c.verify_rmsnorm_dumps()
        self.assertNotIn('rmsnorm_numerical',self.c.record)
        def verified(reference,dumps,output):
            self.assertEqual(dumps,self.root/'raw')
            output.mkdir();(output/'receipt.json').write_bytes(b'bound receipt')
            return {'status':'PASS','cases':[{'maximum_absolute_error':1.23456789123e-6,'maximum_budget_fraction':0.25}]}
        with patch('verify_public_hip_rmsnorm.verify',side_effect=verified):self.c.verify_rmsnorm_dumps()
        self.assertEqual(calls,['frozen','frozen'])
        self.assertEqual(self.c.record['rmsnorm_numerical']['receipt_sha256'],control.sha(b'bound receipt'))

    def test_raw_metric_mismatch_rejects_before_numerical_acceptance(self):
        self.c.verify=lambda:None
        self.c.record['application']={'reported_metrics':{'max_abs':0,'max_budget_fraction':0}}
        result={'status':'PASS','cases':[{'maximum_absolute_error':1e-6,'maximum_budget_fraction':0.1}]}
        with patch('verify_public_hip_rmsnorm.verify',return_value=result):
            with self.assertRaisesRegex(RuntimeError,'transcript metrics differ'):self.c.verify_rmsnorm_dumps()
        self.assertNotIn('rmsnorm_numerical',self.c.record)

    def test_native_audit_binds_all_images_and_rejects_stale_or_unqualified(self):
        build=self.root/'build';build.mkdir()
        images=[build/n for n in ('title.elf','eboot.elf','extracted.elf')]
        for i,path in enumerate(images):path.write_bytes(bytes([i]))
        names=('tools/audit_public_hip_rmsnorm_native.py','tools/audit_public_hip_embedded_native.py',
               'tools/probe_host_registration.py','tools/native_multi_query.py',
               'tests/fat_binary_fixture_test.py','tests/native_import_libraries_test.py',
               'tools/build_native_vector_title.py','tools/public_hip_rmsnorm_console_title.cpp',
               'tools/public_hip_app_crt.cpp','tests/public_hip_rmsnorm_native_audit_test.py')
        for n in names:
            path=self.root/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'implementation')
        audit={'status':'PASS','classification':'offline-public-hip-rmsnorm-native-image-audit',
               'native_executed':False,'inputs':{str(p):control.sha(p.read_bytes()) for p in images+[self.root/n for n in names]}}
        receipt={'title_elf_sha256':audit['inputs'][str(images[0])],
                 'fself_roundtrip':{'elf_sha256':audit['inputs'][str(images[1])],'extracted_sha256':audit['inputs'][str(images[2])]}}
        path=build/'rmsnorm-native-audit.json';path.write_text(json.dumps(audit))
        with patch('freeze_source_candidate.ROOT',self.root):
            self.assertEqual(freeze.validate_rmsnorm_audit(build,receipt),path)
            for key,value in [('status','FAIL'),('classification','other'),('native_executed',True),('inputs',{})]:
                changed=dict(audit);changed[key]=value;path.write_text(json.dumps(changed))
                with self.subTest(key=key),self.assertRaises(ValueError):freeze.validate_rmsnorm_audit(build,receipt)
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError,'does not bind'):
                freeze.validate_rmsnorm_audit(build,dict(receipt,title_elf_sha256='0'*64))
            changed=dict(audit,inputs={k:v for k,v in audit['inputs'].items() if k!=str(self.root/names[0])})
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'does not bind'):freeze.validate_rmsnorm_audit(build,receipt)
            path.write_text(json.dumps(audit));images[2].write_bytes(b'stale')
            with self.assertRaisesRegex(ValueError,'input changed'):freeze.validate_rmsnorm_audit(build,receipt)

    def test_freeze_binds_packaged_assets_and_exact_reference_copy(self):
        original=self.root/'original';original.mkdir()
        artifact=original/'run1/x';artifact.parent.mkdir();artifact.write_bytes(b'oracle')
        original_receipt=original/'receipt.json';original_receipt.write_text(json.dumps({'artifacts':{'run1/x':control.sha(b'oracle')}}))
        module=self.root/'norm.hsaco';module.write_bytes(b'module')
        build=self.root/'build';stage=build/'package/app/rmsnorm';stage.mkdir(parents=True)
        (stage/'norm.hsaco').write_bytes(b'module');(stage/'x').write_bytes(b'oracle')
        bound={'reference':{'receipt':str(original_receipt),'receipt_sha256':control.sha(original_receipt.read_bytes())},
               'assets':{'rmsnorm/norm.hsaco':str(module),'rmsnorm/x':str(artifact)}}
        receipt={'rmsnorm':bound,'package_files':{'rmsnorm/norm.hsaco':control.sha(b'module'),'rmsnorm/x':control.sha(b'oracle')}}
        output=self.root/'candidate';output.mkdir()
        with patch('freeze_source_candidate.ROOT',self.root),patch('build_public_hip.rmsnorm_assets',return_value=bound),patch('build_public_hip.rmsnorm_reference_inputs',return_value={'receipt_sha256':bound['reference']['receipt_sha256']}):
            frozen=freeze.freeze_rmsnorm_reference(build,receipt,output)
            self.assertEqual((self.root/frozen['reference']/'receipt.json').read_bytes(),original_receipt.read_bytes())
            self.assertEqual((self.root/frozen['reference']/'run1/x').read_bytes(),b'oracle')
            (stage/'x').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'packaged oracle'):freeze.freeze_rmsnorm_reference(build,receipt,self.root/'candidate2')


if __name__=='__main__':unittest.main()
