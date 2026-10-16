"""Receipt admission controls; no compiler, runtime or GPU execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import qualify_public_hip_rmsnorm as q

class Receipts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'fixture';self.folder.mkdir()
        self.base=self.root/'base';self.base.mkdir()
        self.patch=patch.object(q,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.inputs={}
        for name in set(q.FIXTURE_SOURCES+q.BASE_SOURCES):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source');self.inputs[str(p)]=q.sha(p)
        spec=self.root/'ports/clr/ps5-public-resources.json';spec.write_text(json.dumps({'files':{'hipamd/src/hip_module.cpp':{'after':q.sha(self.root/q.BASE_SOURCES[0])}}}));self.inputs[str(spec)]=q.sha(spec)
        self.order=['ockl-workitem-only','mul_hi','ocml',*(n for n in q.CONSTANTS if n!='oclc_isa_version_1030')]
        for name in self.order:
            p=self.folder/'libs'/(name+'.bc');p.parent.mkdir(exist_ok=True);p.write_text('bitcode');self.inputs[str(p)]=q.sha(p)
        for name in q.FIXTURE_ARTIFACTS:(self.folder/name).write_text('artifact')
        (self.folder/'norm.d').write_text('norm: '+str(self.root/q.FIXTURE_SOURCES[0])+'\n')
        tools={}
        for name in q.FIXTURE_TOOLS:
            p=self.root/'bin'/name;p.parent.mkdir(exist_ok=True);p.write_text('tool');tools[name]={'path':str(p),'sha256':q.sha(p)}
        self.which=patch.object(q.shutil,'which',lambda name:tools[name]['path']);self.which.start();self.addCleanup(self.which.stop)
        self.metadata=patch.object(q,'selected_metadata',return_value={'symbol':q.SYMBOL});self.metadata.start();self.addCleanup(self.metadata.stop)
        self.fixture=dict(status='PASS',native_executed=False,gpu_executed=False,inputs=self.inputs.copy(),tools=tools,artifacts={n:{'sha256':q.sha(self.folder/n),'bytes':(self.folder/n).stat().st_size} for n in q.FIXTURE_ARTIFACTS},library_order=self.order,selected={'symbol':q.SYMBOL},descriptor=dict(group_bytes=0,private_bytes=0,kernarg_bytes=488,rsrc1=0x60af00c3,rsrc2=0x38c,rsrc3=0,properties=0x409,preload=0),kernel_count=16,undefined_symbols=[],commands=[{'exit_code':0}])
        prepared={}
        for n in ('hipamd/src/hip_module.cpp','rocclr/platform/kernel.cpp'):
            p=self.base/'source-public'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source');prepared[str(p)]=q.sha(p)
        self.report=dict(classification='source-only-genuine-public-hip-model',native_executed=False,gpu_executed=False,inputs=self.inputs.copy(),prepared_sources=prepared,included_files={str(self.root/q.BASE_SOURCES[0]):self.inputs[str(self.root/q.BASE_SOURCES[0])]},toolchain={n:tools[n] for n in q.BASE_TOOLS})
    def fixture_check(self,j):
        (self.folder/'receipt.json').write_text(json.dumps(j));return q.fixture_inputs(self.folder)
    def base_check(self,j):
        (self.base/'receipt.json').write_text(json.dumps(j));return q.base_inputs(self.base)
    def test_complete(self):
        self.assertTrue(self.fixture_check(self.fixture));self.assertTrue(self.base_check(self.report))
    def test_weighted_qualification_is_explicit_and_strict(self):
        report={'classification':'source-only-selected-upstream-weighted-rmsnorm-fixture'}
        (self.folder/'receipt.json').write_text(json.dumps(report))
        with patch.dict(sys.modules,{'build_llm_weighted_rmsnorm_fixture':SimpleNamespace(fixture_inputs=lambda p:{'weighted':'digest'})}):
            self.assertEqual(q.qualification_fixture_inputs(self.folder),({'weighted':'digest'},True))
        with self.assertRaises(ValueError):q.fixture_inputs(self.folder)
        def reject(folder):raise ValueError('weighted identity rejected')
        with patch.dict(sys.modules,{'build_llm_weighted_rmsnorm_fixture':SimpleNamespace(fixture_inputs=reject)}):
            with self.assertRaisesRegex(ValueError,'weighted identity rejected'):q.qualification_fixture_inputs(self.folder)
    def test_original_qualification_stays_unweighted(self):
        expected=self.fixture_check(self.fixture)
        self.assertEqual(q.qualification_fixture_inputs(self.folder),(expected,False))
    def test_empty_groups(self):
        for check,report,groups in ((self.fixture_check,self.fixture,('inputs','tools','artifacts')),(self.base_check,self.report,('inputs','prepared_sources','included_files','toolchain'))):
            for group in groups:
                for missing in (False,True):
                    with self.subTest(group=group,missing=missing):
                        j=copy.deepcopy(report)
                        if missing:del j[group]
                        else:j[group]={}
                        with self.assertRaises(ValueError):check(j)
    def test_missing_required_identities(self):
        for check,report,names in ((self.fixture_check,self.fixture,q.FIXTURE_SOURCES),(self.base_check,self.report,q.BASE_SOURCES)):
            for name in names:
                j=copy.deepcopy(report);del j['inputs'][str(self.root/name)]
                with self.subTest(name=name),self.assertRaises(ValueError):check(j)
    def test_stale_and_missing_tools(self):
        for check,report,group in ((self.fixture_check,self.fixture,'tools'),(self.base_check,self.report,'toolchain')):
            j=copy.deepcopy(report);j['inputs'][next(iter(j['inputs']))]='0'*64
            with self.assertRaises(ValueError):check(j)
            j=copy.deepcopy(report);del j[group]['clang++-18']
            with self.assertRaises(ValueError):check(j)
            j=copy.deepcopy(report);j[group]['clang++-18']['sha256']='0'*64
            with self.assertRaises(ValueError):check(j)
    def test_artifact_and_metadata_mutations(self):
        for mutate in (lambda j:j['artifacts']['norm.hsaco'].update(sha256='0'*64),lambda j:j['artifacts']['norm.hsaco'].update(bytes=0),lambda j:j['artifacts'].update({'../escape':{}}),lambda j:j['descriptor'].update(kernarg_bytes=0),lambda j:j.update(library_order=[]),lambda j:j.update(commands=[])):
            j=copy.deepcopy(self.fixture);mutate(j)
            with self.assertRaises(ValueError):self.fixture_check(j)

if __name__=='__main__':unittest.main()
