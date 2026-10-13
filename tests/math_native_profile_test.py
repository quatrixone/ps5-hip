"""Synthetic boundary mutations; optional genuine frozen compiler-artifact checks."""
import copy
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import math_native_fixture as fixture
import stage_native_title as stage
import verify_native_case as case
import check_canary_log as logs

FIXTURES = None
if '--fixtures' in sys.argv:
    i = sys.argv.index('--fixtures'); FIXTURES = Path(sys.argv[i+1]); del sys.argv[i:i+2]


class Contract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'tools').mkdir(); (self.root / 'assets').mkdir()
        shutil.copyfile(ROOT / 'assets/icon0.png', self.root / 'assets/icon0.png')
        self.lock = copy.deepcopy(fixture.lock())
        for name in set(self.lock['oracle_sources']) | {'tools/math_native_fixture.py'}:
            p = self.root / name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, p)
        self.code = b'explicit synthetic code object fixture'
        self.report = {'sources': {'tools/math_native_fixture.py': fixture.sha((self.root / 'tools/math_native_fixture.py').read_bytes())}}
        self.lock['device_report_sha256'] = fixture.canonical(self.report)
        for p in self.lock['profiles']:
            p['code_sha256'] = fixture.sha(self.code); p['bytes'] = len(self.code)
        (self.root / 'tools/math_native_inputs.json').write_text(json.dumps(self.lock))
        self.build = {'kernel': 'fp32_math', 'target': 'gfx1013', 'hardware_validated': False,
                      'device_build': self.report, 'math_profile': fixture.profile('gfx1013', self.root),
                      'source_sha256': {name: fixture.sha((self.root / name).read_bytes()) for name in
                                       set(self.lock['oracle_sources']) | {'tools/math_native_fixture.py', 'tools/math_native_inputs.json'}}}

    def test_binding_rejects_each_contract_and_source_mutation(self):
        fixture.validate_build(self.build, self.code, self.root)
        for key, value in self.build['math_profile'].items():
            bad = copy.deepcopy(self.build); bad['math_profile'][key] = value + 1 if type(value) is int else str(value) + 'bad'
            with self.subTest(key=key), self.assertRaises(ValueError): fixture.validate_build(bad, self.code, self.root)
        for name in self.build['source_sha256']:
            p = self.root / name; original = p.read_bytes(); p.write_bytes(original + b'changed')
            with self.subTest(source=name), self.assertRaises((ValueError, json.JSONDecodeError)):
                fixture.validate_build(self.build, self.code, self.root)
            p.write_bytes(original)
        for bad in (dict(self.build, target='gfx9999'), dict(self.build, kernel='vector_add'),
                    dict(self.build, device_build={}), dict(self.build, source_sha256={})):
            with self.assertRaises(ValueError): fixture.validate_build(bad, self.code, self.root)
        with self.assertRaises(ValueError): fixture.validate_build(self.build, self.code + b'x', self.root)

    def test_only_two_nonoverlapping_reviewed_immediates(self):
        original = struct.pack('<III', 0xc2ce8ed0, 123, 0xc2ce8ed0)
        expected = struct.pack('<III', 0xc2cff1b4, 123, 0xc2cff1b4)
        self.assertEqual(fixture.corrected_bytes(original, [0,8]), expected)
        for offsets in ([0,0],[0,2],[-1,8],[0,12],[False,8],[0],['0',8]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                fixture.corrected_bytes(original, offsets)
        with self.assertRaises(ValueError): fixture.corrected_bytes(b'x'+original[1:], [0,8])

    def test_stage_and_verifier_propagate_exact_profile(self):
        builddir = self.root / 'build'; candidate = builddir / 'candidate'; candidate.mkdir(parents=True)
        payloads = {'eboot.bin': b'host', 'lapy.elf': b'helper', 'sce_module/libc.prx': b'libc', 'canary.hsaco': self.code}
        for name, data in payloads.items():
            p = candidate / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        self.build['artifacts'] = {n: fixture.sha(d) for n,d in payloads.items()}
        (builddir / 'receipt.json').write_text(json.dumps(self.build))
        output = self.root / 'stage'; title = 'PPSA99014'
        manifest = stage.stage(self.root, builddir, output, {'title_id':title,'title_name':'math test'}, 'test', '/chosen/results')
        self.assertEqual(manifest['math_profile'], self.build['math_profile'])
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
            manifest['math_profile']['groups'] = 6
            (output/'manifest.json').write_text(json.dumps(manifest)); package['manifest_sha256']=case.fingerprint(output/'manifest.json')['sha256']
            (output/'package-receipt.json').write_text(json.dumps(package))
            with self.assertRaisesRegex(ValueError, 'math manifest/build'): case.verify_case(output,controllers,catalog,{'controller_build':control},title)
        self.build['math_profile']['kernarg_bytes'] = 64
        (builddir/'receipt.json').write_text(json.dumps(self.build))
        with self.assertRaises(ValueError): stage.stage(self.root,builddir,self.root/'bad',{'title_id':title,'title_name':'math'},'test','/chosen/results')
        self.assertFalse((self.root/'bad').exists())

    def test_exact_log_contract_and_mutations(self):
        digest = fixture.profile('gfx1013')['contract_sha256']
        lines = ['PS5HIP START target=gfx1013 kernel=fp32_math',
                 'PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass',
                 'PS5HIP SUBMIT target=gfx1013 kernel=fp32_math inputs=257 output_words=1542 groups=5 kernarg_bytes=280 timeout_ms=2000',
                 f'PS5HIP PASS kernel=fp32_math numeric_values=1542 contract={digest} inputs=unchanged guards=unchanged allocations=closed']
        raw = ('\n'.join(lines)+'\n').encode()
        self.assertEqual(logs.check_log(raw,'gfx1013','fp32_math')['application_result'],'application-pass')
        for before, after in [(b'257',b'256'),(b'1542',b'1541'),(b'groups=5',b'groups=4'),(b'280',b'64'),
                              (digest.encode(),b'0'*64),(b'numeric_values',b'exact_values'),(b'allocations=closed',b'allocations=open')]:
            self.assertNotEqual(logs.check_log(raw.replace(before,after),'gfx1013','fp32_math')['application_result'],'application-pass')
        self.assertNotEqual(logs.check_log(raw,'gfx1013','fp32_math',shell_exit_probe=True)['application_result'],'application-pass')

    @unittest.skipUnless(FIXTURES, 'supply frozen --fixtures for full compiler-input rehash')
    def test_real_compiler_binding(self):
        for target in ('gfx1013','gfx1030'):
            data, report = fixture.verified_device_object(ROOT,FIXTURES,target,'18.1.8')
            self.assertEqual(fixture.sha(data), fixture.profile(target)['code_sha256'])
            self.assertEqual(len(report['profiles']),4)
        # Intercept reads, never write to the frozen reference artifacts.
        original = Path.read_bytes
        basis = Path(json.loads((FIXTURES/'receipt.json').read_text())['original_directory'])
        for damaged in (FIXTURES/'receipt.json', ROOT/'tools/probe_ocml_math.py',
                        FIXTURES/'gfx1013-O2.hsaco', basis/'libs/ocml.bc',
                        basis/'libs/oclc_daz_opt_off.bc', basis/'gfx1013-O2.hsaco'):
            damaged = damaged.resolve()
            def altered(path):
                data = original(path)
                return data + b'changed' if path.resolve() == damaged else data
            with self.subTest(damaged=damaged.name), patch.object(Path, 'read_bytes', altered):
                with self.assertRaises(ValueError):
                    fixture.verified_device_object(ROOT,FIXTURES,'gfx1013','18.1.8')


if __name__ == '__main__': unittest.main()
