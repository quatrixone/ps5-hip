"""RMSNorm asset admission and package routing; no native execution."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_public_hip as b

class RmsnormBuild(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.folder=self.root/'reference';self.folder.mkdir()
        self.p=patch.object(b,'ROOT',self.root);self.p.start();self.addCleanup(self.p.stop)
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        save(self.root/'llm-inputs.json',{'client':{'revision':'revision','source_url':'source'}})
        save(self.root/'evidence/offline-llm-model-conversion.json',{'model_revision':'model','outputs':{'tinyllama-q4_0.gguf':{'sha256':'modelhash'}}})
        save(self.root/'tests/fixtures/llm-reference-prompts.json',{'model_revision':'model','cases':[{'tokens':list(range(17))}]})
        self.identities={}
        cpu=self.root/'cpu/receipt.json'
        libs={}
        for n in ('libggml.so','libggml-cpu.so','libggml-base.so'):
            p=cpu.parent/'cmake/bin'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'library');libs['cmake/bin/'+n]=b.sha(p);self.identities[str(p)]=b.sha(p)
        save(cpu,{'classification':'host-cpu-reference-tools-built','source_revision':'revision','identity':{'source_revision':'revision','source_url':'source','flags':['-DGGML_BACKEND_DL=OFF','-DGGML_CPU=ON','-DGGML_HIP=OFF','-DGGML_CUDA=OFF','-DGGML_BLAS=OFF']},'artifacts':libs})
        self.identities[str(cpu)]=b.sha(cpu)
        for p in (Path('/usr/bin/clang++-18').resolve(),Path(sys.executable).resolve()):self.identities[str(p)]='tool'
        self.provenance=patch.object(b,'rmsnorm_cpu_provenance',return_value={});self.provenance.start();self.addCleanup(self.provenance.stop)
        self.check=patch.object(b,'checked_inputs',side_effect=lambda *a:self.identities.copy());self.check.start();self.addCleanup(self.check.stop)
        profile=dict(width=2048,epsilon_f32=b.rms_reference.EPSILON,row_counts=[1,2,17],embedding_tokens=list(range(17)),threads=1,operator='GGML_OP_RMS_NORM unfused F32',tolerance=b.rms_reference.TOLERANCE)
        save(self.folder/'declared-profile.json',profile)
        (self.folder/'tokens.txt').write_text(' '.join(map(str,range(17)))+'\n')
        for n in ('fixture','noop-rejection.txt','corrupt-rejection.txt'):(self.folder/n).write_text('fixture')
        (self.folder/'fixture.d').write_text('fixture: \n')
        save(self.folder/'commands.json',[{'label':n,'exit_code':0} for n in ('dependencies','compile','runtime-libraries','run1','run2')])
        for repeat in (1,2):
            (self.folder/('run%d.stdout'%repeat)).write_text('GGML_RMS_NORM_CPU_PASS cases=6 width=2048 threads=1 gpu_execution=false\n')
            (self.folder/('run%d'%repeat)).mkdir()
        cases=[]
        for family in ('stress','embedding'):
            for rows in (1,2,17):
                name=family+'-'+str(rows);values=[0.0]*(rows*2048)
                for repeat in (1,2):
                    for kind in ('input','cpu'):(self.folder/('run%d'%repeat)/(name+'.'+kind+'.f32le')).write_bytes(struct.pack('<'+'f'*len(values),*values))
                p=self.folder/(name+'.fp64le');p.write_bytes(struct.pack('<'+'d'*len(values),*values))
                cases.append(dict(name=name,rows=rows,input_sha256=b.sha(self.folder/'run1'/(name+'.input.f32le')),cpu_sha256=b.sha(self.folder/'run1'/(name+'.cpu.f32le')),fp64_sha256=b.sha(p),metrics=b.rms_reference.compare(values,values)))
        self.report=dict(classification='host-cpu-rmsnorm-reference',gpu_executed=False,fresh_repeats_bitwise_equal=True,negative_controls=['noop rejected','single-element corruption rejected'],source_revision='revision',model_sha256='modelhash',profile=profile,cases=cases,artifacts={str(p.relative_to(self.folder)):b.sha(p) for p in self.folder.rglob('*') if p.is_file()})
    def verify(self,report=None):
        (self.folder/'receipt.json').write_text(json.dumps(report or self.report));return b.rmsnorm_reference_inputs(self.folder)
    def test_verified_assets(self):
        result=self.verify();self.assertEqual(len(result['assets']),12)
        self.assertTrue(all(n.startswith('rmsnorm/') for n in result['assets']))
    def test_profile_case_and_group_mutations(self):
        for mutate in (lambda j:j.update(artifacts={}),lambda j:j['profile'].update(width=1024),lambda j:j.update(cases=j['cases'][:-1]),lambda j:j.update(fresh_repeats_bitwise_equal=False),lambda j:j.update(model_sha256='bad'),lambda j:j['artifacts'].update({'../escape':'bad'})):
            j=copy.deepcopy(self.report);mutate(j)
            with self.assertRaises(ValueError):self.verify(j)
    def test_stale_tensor_and_dependency(self):
        p=self.folder/'run1/stress-1.cpu.f32le';old=p.read_bytes();p.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.verify()
        p.write_bytes(old)
        dep=self.folder/'fixture.d';dep.write_text('fixture: /missing/header.h\n');self.report['artifacts']['fixture.d']=b.sha(dep)
        with self.assertRaisesRegex(ValueError,'header identity'):self.verify()
    def test_missing_runtime_identity(self):
        del self.identities[str(self.root/'cpu/cmake/bin/libggml-cpu.so')]
        with self.assertRaises(ValueError):self.verify()
    def test_module_and_reference_routing(self):
        with patch.object(b,'rmsnorm_fixture_inputs',return_value={'module':'digest'}) as module,patch.object(b,'rmsnorm_reference_inputs',return_value={'assets':{'rmsnorm/x':'path'}}) as reference:
            result=b.rmsnorm_assets(self.root/'module',self.folder)
            module.assert_called_once_with(self.root/'module');reference.assert_called_once_with(self.folder)
            self.assertEqual(result['assets'],{'rmsnorm/norm.hsaco':str(self.root/'module/norm.hsaco'),'rmsnorm/x':'path'})
    def test_required_native_symbols(self):
        names=['hipInit','hipGetDeviceCount','ps5hip_clr_admit_device','hipMalloc','hipFree','hipMemcpy','hipModuleLoad','hipModuleGetFunction','hipModuleUnload','hipModuleLaunchKernel','hipDeviceSynchronize']
        with patch.object(b,'run'),patch.object(b.subprocess,'check_output',return_value='\n'.join('DEF '+n for n in names)):
            b.link_title(self.root,[],'rmsnorm')
        with patch.object(b,'run'),patch.object(b.subprocess,'check_output',return_value='\n'.join('DEF '+n for n in names[:-1])):
            with self.assertRaisesRegex(RuntimeError,'hipDeviceSynchronize'):b.link_title(self.root,[],'rmsnorm')

class CpuProvenance(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.root_patch=patch.object(b,'ROOT',self.root);self.root_patch.start();self.addCleanup(self.root_patch.stop)
        self.inputs={}
        for n in ('llm-inputs.json','tools/build_llm_cpu_reference.py','tools/fetch_locked_deps.py'):
            p=self.root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source');self.inputs[str(p)]=b.sha(p)
        for n in b.RMSNORM_CPU_IMPLEMENTATIONS:
            p=self.root/'.deps/sources/llama-client'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('implementation')
        self.model=self.root/'model.gguf';self.model.write_bytes(b'model')
        self.cpu={'source_revision':'revision','identity':{'inputs':self.inputs,'tools':{}}}
        self.commands=[{'label':label,'argv':['fixture',str(self.model),'tokens',label]} for label in ('run1','run2')]
        self.tools=patch.object(b,'checked_tools',return_value={});self.tools.start();self.addCleanup(self.tools.stop)
    def verify(self,dirty=False):
        def git(argv,**kwargs):
            if 'rev-parse' in argv:return 'revision'
            if 'status' in argv:return ' M ggml/src/ggml-cpu/ops.cpp' if dirty else ''
            return '\n'.join(b.RMSNORM_CPU_IMPLEMENTATIONS)
        with patch.object(b.subprocess,'check_output',side_effect=git):
            return b.rmsnorm_cpu_provenance(self.cpu,self.commands,b.sha(self.model))
    def test_current_source_and_model_bound(self):
        result=self.verify();self.assertIn(str(self.model),result)
        for n in b.RMSNORM_CPU_IMPLEMENTATIONS:self.assertIn(str(self.root/'.deps/sources/llama-client'/n),result)
    def test_dirty_implementation_rejected(self):
        with self.assertRaisesRegex(ValueError,'tree changed'):self.verify(dirty=True)
    def test_missing_build_identity_rejected(self):
        for n in tuple(self.inputs):
            digest=self.inputs.pop(n)
            with self.assertRaisesRegex(ValueError,'identities missing'):self.verify()
            self.inputs[n]=digest
    def test_deleted_implementation_rejected(self):
        (self.root/'.deps/sources/llama-client'/b.RMSNORM_CPU_IMPLEMENTATIONS[0]).unlink()
        with self.assertRaises(FileNotFoundError):self.verify()
    def test_mismatched_or_absent_model_rejected(self):
        with self.assertRaisesRegex(ValueError,'model bytes differ'):b.rmsnorm_cpu_provenance(self.cpu,self.commands,'0'*64)
        self.commands[0]['argv'].pop(1)
        with self.assertRaisesRegex(ValueError,'invocation identity'):self.verify()

if __name__=='__main__':unittest.main()
