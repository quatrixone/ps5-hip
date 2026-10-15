"""Offline candidate identity/copy controls; authenticated corpus generation is tested separately."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import freeze_source_candidate as freeze
from audit_public_hip_q4_native import SOURCES
from verify_public_hip_q4 import ASSET_NAMES


class Q4Freeze(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.build=self.root/'build';self.build.mkdir()
        self.output=self.root/'candidate';self.output.mkdir()
        self.reference=self.root/'reference';self.reference.mkdir()
        self.fixture=self.root/'fixture';self.fixture.mkdir()
        self.assets={};artifacts={}
        for name in sorted(ASSET_NAMES|{'q4/q8.hsaco','q4/q4.hsaco'}):
            source=(self.fixture/Path(name).name if name.endswith('.hsaco') else self.reference/name[3:])
            source.parent.mkdir(parents=True,exist_ok=True);source.write_bytes(name.encode())
            self.assets[name]=str(source)
            dest=self.build/'package/app'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
            if not name.endswith('.hsaco'):artifacts[name[3:]]={'bytes':source.stat().st_size,'sha256':freeze.digest(source)}
        receipt=self.reference/'receipt.json';receipt.write_text(json.dumps({'artifacts':artifacts}))
        self.bound={'assets':self.assets,'reference':{'receipt':str(receipt),'receipt_sha256':freeze.digest(receipt)},'inputs':{},'module_inputs':{}}
        self.receipt={'q4':self.bound,'package_files':{n:freeze.digest(Path(p)) for n,p in self.assets.items()}}
        self.rootpatch=patch.object(freeze,'ROOT',self.root);self.rootpatch.start();self.addCleanup(self.rootpatch.stop)

    def run_freeze(self):
        with patch('public_hip_q4_assets.assets',return_value=self.bound):
            return freeze.freeze_q4_reference(self.build,self.receipt,self.output)

    def test_complete_reference_copy_and_two_module_binding(self):
        result=self.run_freeze()
        self.assertEqual(result['fixture'],'fixture')
        self.assertEqual(result['reference'],'candidate/q4-reference')
        self.assertEqual(set(result['module_sha256']),{'q8','q4'})
        self.assertEqual((self.output/'q4-reference/receipt.json').read_bytes(),(self.reference/'receipt.json').read_bytes())
        self.assertEqual({p.relative_to(self.reference) for p in self.reference.rglob('*') if p.is_file()},
                         {p.relative_to(self.output/'q4-reference') for p in (self.output/'q4-reference').rglob('*') if p.is_file()})

    def test_package_missing_extra_and_byte_change_rejected_before_copy(self):
        initial=copy.deepcopy(self.receipt)
        for change in ('missing','extra','changed'):
            self.receipt=copy.deepcopy(initial)
            if change=='missing':del self.receipt['package_files']['q4/q8.hsaco']
            elif change=='extra':self.receipt['package_files']['q4/unlisted']= '0'*64
            else:(self.build/'package/app/q4/q4.hsaco').write_bytes(b'changed')
            with self.subTest(change=change),self.assertRaises(ValueError):self.run_freeze()
            self.assertFalse((self.output/'q4-reference').exists())

    def test_unsafe_path_hash_size_and_bound_checked_before_copy(self):
        receipt=self.reference/'receipt.json';original=json.loads(receipt.read_text())
        first=next(iter(original['artifacts']))
        for mutation in ('escape','hash','size','bound'):
            report=copy.deepcopy(original)
            if mutation=='escape':report['artifacts']['../outside']={'bytes':1,'sha256':'0'*64}
            if mutation=='hash':report['artifacts'][first]['sha256']='0'*64
            if mutation=='size':report['artifacts'][first]['bytes']+=1
            receipt.write_text(json.dumps(report))
            with self.subTest(mutation=mutation),patch.object(freeze,'Q4_REFERENCE_LIMIT',1 if mutation=='bound' else 16*1024*1024),self.assertRaises(ValueError):self.run_freeze()
            self.assertFalse((self.output/'q4-reference').exists())

    def test_source_drift_and_copied_receipt_mismatch_rejected(self):
        changed=copy.deepcopy(self.bound);changed['reference']['receipt_sha256']='0'*64
        with patch('public_hip_q4_assets.assets',side_effect=[self.bound,changed]):
            with self.assertRaisesRegex(ValueError,'receipt differs'):
                freeze.freeze_q4_reference(self.build,self.receipt,self.output)

    def test_native_audit_exact_images_sources_and_classification(self):
        images=[self.build/n for n in ('title.elf','eboot.elf','extracted.elf')]
        for p in [*images,*(self.root/n for n in SOURCES)]:
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(str(p).encode())
        audit={'status':'PASS','classification':'offline-public-hip-q4-native-image-audit','native_executed':False,
               'inputs':{str(p):freeze.digest(p) for p in [*images,*(self.root/n for n in SOURCES)]}}
        receipt={'title_elf_sha256':freeze.digest(images[0]),'fself_roundtrip':{'elf_sha256':freeze.digest(images[1]),'extracted_sha256':freeze.digest(images[2])}}
        target=self.build/'q4-native-audit.json';target.write_text(json.dumps(audit))
        self.assertEqual(freeze.validate_q4_audit(self.build,receipt),target)
        for key,value in [('status','FAIL'),('classification','wrong'),('native_executed',True),('inputs',{})]:
            bad=copy.deepcopy(audit);bad[key]=value;target.write_text(json.dumps(bad))
            with self.subTest(key=key),self.assertRaises(ValueError):freeze.validate_q4_audit(self.build,receipt)
        bad=copy.deepcopy(audit);del bad['inputs'][str(images[0])];target.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError,'does not bind'):freeze.validate_q4_audit(self.build,receipt)
        target.write_text(json.dumps(audit));images[2].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'input changed'):freeze.validate_q4_audit(self.build,receipt)

if __name__=='__main__':unittest.main()
