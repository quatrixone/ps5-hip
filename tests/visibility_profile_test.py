"""Offline provenance, stage and strict-log boundaries; synthetic ELF bytes are not GPU evidence."""
import copy,json,sys,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import visibility_fixture as fixture
import stage_native_title as stage
import verify_native_case as case
import check_canary_log as logs
class Contract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        (self.root/'assets').mkdir();shutil.copyfile(ROOT/'assets/icon0.png',self.root/'assets/icon0.png')
        self.code=b'synthetic visibility code'
        self.report={'synthetic_test_only':True}
        names=['kernels/visibility.hip','tools/probe_visibility_compiler.py','tools/build_canaries.py','tools/native_visibility.h','tools/visibility_fixture.py']
        for name in names:
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        sources={n:fixture.digest((self.root/n).read_bytes()) for n in names[:3]}
        self.lock={'report_sha256':fixture.digest((json.dumps(self.report,indent=2)+'\n').encode()),'sources':sources,'objects':{'gfx1013':{'sha256':fixture.digest(self.code),'bytes':len(self.code)}}}
        (self.root/'tools/visibility_inputs.json').write_text(json.dumps(self.lock))
        self.build={'kernel':'dependent_visibility','target':'gfx1013','hardware_validated':False,'device_build':self.report,'visibility_profile':copy.deepcopy(fixture.PROFILE),
                    'source_sha256':{n:fixture.digest((self.root/n).read_bytes()) for n in names+['tools/visibility_inputs.json']}}
    def test_binding_mutations(self):
        fixture.validate_build(self.build,self.code,self.root)
        for key,value in fixture.PROFILE.items():
            bad=copy.deepcopy(self.build);bad['visibility_profile'][key]=value+1 if type(value) is int else value+'bad'
            with self.subTest(key=key),self.assertRaises(ValueError):fixture.validate_build(bad,self.code,self.root)
        for name in self.build['source_sha256']:
            p=self.root/name;before=p.read_bytes();p.write_bytes(before+b'x')
            with self.subTest(name=name),self.assertRaises((ValueError,json.JSONDecodeError)):fixture.validate_build(self.build,self.code,self.root)
            p.write_bytes(before)
        for bad in [dict(self.build,target='gfx1030'),dict(self.build,kernel='vector_add'),dict(self.build,device_build={}),dict(self.build,source_sha256={})]:
            with self.assertRaises(ValueError):fixture.validate_build(bad,self.code,self.root)
        with self.assertRaises(ValueError):fixture.validate_build(self.build,self.code+b'x',self.root)
    def test_stage_and_verifier_propagate_exact_profile(self):
        builddir = self.root / 'build'; candidate = builddir / 'candidate'; candidate.mkdir(parents=True)
        payloads = {'eboot.bin': b'host', 'lapy.elf': b'helper', 'sce_module/libc.prx': b'libc', 'canary.hsaco': self.code}
        for name, data in payloads.items():
            p = candidate / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        self.build['artifacts'] = {n: fixture.digest(d) for n,d in payloads.items()}
        (builddir / 'receipt.json').write_text(json.dumps(self.build))
        output = self.root / 'stage'; title = 'PPSA99014'
        manifest = stage.stage(self.root, builddir, output, {'title_id':title,'title_name':'math test'}, 'test', '/chosen/results')
        self.assertEqual(manifest['visibility_profile'], self.build['visibility_profile'])
        controllers = self.root / 'controllers'; controllers.mkdir()
        for name in ('launch.elf', 'close.elf'): (controllers / name).write_bytes(name.encode())
        control = {'title_id':title,'artifacts': {n:case.fingerprint(controllers/n) for n in ('launch.elf','close.elf')}}
        (controllers / 'receipt.json').write_text(json.dumps(control))
        (output / (title+'.ffpkg')).write_bytes(b'synthetic image')
        package = {'title_id':title, 'manifest_sha256':case.fingerprint(output/'manifest.json')['sha256'],
                   'image':dict(file=title+'.ffpkg', **case.fingerprint(output/(title+'.ffpkg')))}
        (output/'package-receipt.json').write_text(json.dumps(package))
        catalog = {'candidates':[{'manifest':manifest,'package':package}]}
        with patch.object(case, 'ROOT', self.root):
            self.assertFalse(case.verify_case(output,controllers,catalog,{'controller_build':control},title)['hardware_gate_passed'])
            # Re-sign adjacent manifest/catalog hashes to exercise profile equality, not generic hash rejection.
            manifest['visibility_profile']['groups'] = 6
            (output/'manifest.json').write_text(json.dumps(manifest)); package['manifest_sha256']=case.fingerprint(output/'manifest.json')['sha256']
            (output/'package-receipt.json').write_text(json.dumps(package))
            with self.assertRaisesRegex(ValueError, 'visibility manifest/build'): case.verify_case(output,controllers,catalog,{'controller_build':control},title)
        self.build['visibility_profile']['elements'] = 64
        (builddir/'receipt.json').write_text(json.dumps(self.build))
        with self.assertRaises(ValueError): stage.stage(self.root,builddir,self.root/'bad',{'title_id':title,'title_name':'math'},'test','/chosen/results')
        self.assertFalse((self.root/'bad').exists())

    def test_exact_log_contract(self):
        lines=['PS5HIP START target=gfx1013 kernel=dependent_visibility','PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass']
        for r in range(1,4):
            for name in ['visibility_producer','visibility_consumer']:lines.append(f'PS5HIP SUBMIT kernel={name} round={r}/3 count=257 groups=5 timeout_ms=2000')
            lines.append(f'PS5HIP ROUND kernel=dependent_visibility index={r}/3 exact_values=257 inputs=unchanged guards=unchanged')
        lines+=['PS5HIP PASS kernel=dependent_visibility dispatches=6 rounds=3 exact_values=771 persistent_buffers=3 guards=unchanged allocations=closed',
                'PS5HIP SHUTDOWN main-return=0','PS5HIP SHUTDOWN atexit=observed','PS5HIP SHUTDOWN static-dtor=observed',
                'PS5HIP SHELL_EXIT application-fini=completed','PS5HIP SHELL_EXIT request=begin','PS5HIP SHELL_EXIT request-return=0']
        raw=('\n'.join(lines)+'\n').encode()
        self.assertEqual(logs.check_log(raw,'gfx1013','dependent_visibility',shell_exit_probe=True)['application_result'],'application-pass')
        variants=[raw.replace(a,b) for a,b in [(b'round=2/3',b'round=1/3'),(b'persistent_buffers=3',b'persistent_buffers=2'),(b'exact_values=771',b'exact_values=257'),(b'inputs=unchanged',b'inputs=changed'),(b'allocations=closed',b'allocations=open')]]
        for i in range(len(lines)):variants.append(('\n'.join(lines[:i]+lines[i+1:])+'\n').encode())
        swapped=lines.copy();swapped[2],swapped[3]=swapped[3],swapped[2];variants.append(('\n'.join(swapped)+'\n').encode())
        for bad in variants:self.assertNotEqual(logs.check_log(bad,'gfx1013','dependent_visibility',shell_exit_probe=True)['application_result'],'application-pass')
    def test_actual_frozen_inputs(self):
        directory=ROOT/'build/visibility-probes'
        if not directory.exists():self.skipTest('actual compiler fixture directory unavailable')
        for target in ('gfx1013','gfx1030'):
            data,report=fixture.verified_device_object(ROOT,directory,target,'18.1.8')
            build={'kernel':'dependent_visibility','target':target,'device_build':report,'visibility_profile':fixture.PROFILE,
                   'source_sha256':{n:fixture.digest((ROOT/n).read_bytes()) for n in list(fixture.inputs(ROOT)['sources'])+['tools/native_visibility.h','tools/visibility_fixture.py','tools/visibility_inputs.json']}}
            fixture.validate_build(build,data,ROOT)
if __name__=='__main__':unittest.main(verbosity=2)
