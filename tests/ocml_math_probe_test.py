"""Offline recipe boundary checks; optional actual compiler artifacts are not GPU tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import probe_ocml_math as probe

FIXTURES = None
if '--fixtures' in sys.argv:
    at = sys.argv.index('--fixtures')
    FIXTURES = Path(sys.argv[at + 1]).resolve()
    del sys.argv[at:at + 2]


class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name, data in {'device-libs-source/ocml/a.cl': 'original OCML',
                           'package-stage/include/hip.h': 'real header',
                           'ockl-workitem-only.bc': 'workitem bytes',
                           'workitem-build.json': '{}'} .items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        self.manifest = {'commit': 'commit', 'subtree': 'tree',
                         'files': [{'path': p, 'sha256': h} for p, h in
                                   probe.inventory(self.root / 'device-libs-source').items()]}
        (self.root / 'device-libs-sources.json').write_text(json.dumps(self.manifest))
        self.pins = {'device_libraries_commit': 'commit', 'device_libraries_subtree': 'tree',
                     'reference_files': {p: probe.sha(self.root / p) for p in
                                         ('device-libs-sources.json', 'ockl-workitem-only.bc', 'workitem-build.json')},
                     'package_stage': {'files': 1, 'inventory_sha256': probe.digest(
                         probe.inventory(self.root / 'package-stage'))}}

    def test_exact_inputs_and_source_header_workitem_substitution(self):
        probe.verify_inputs(self.root, self.pins)
        for name in ('device-libs-source/ocml/a.cl', 'package-stage/include/hip.h',
                     'ockl-workitem-only.bc', 'workitem-build.json', 'device-libs-sources.json'):
            path = self.root / name
            before = path.read_bytes()
            path.write_bytes(before + b'changed')
            with self.subTest(name=name), self.assertRaises(ValueError):
                probe.verify_inputs(self.root, self.pins)
            path.write_bytes(before)

    def test_added_missing_symlink_and_duplicate_inputs(self):
        path = self.root / 'device-libs-source/extra.cl'
        path.write_text('extra')
        with self.assertRaises(ValueError):
            probe.verify_inputs(self.root, self.pins)
        path.unlink()
        path.symlink_to(self.root / 'ockl-workitem-only.bc')
        with self.assertRaises(ValueError):
            probe.verify_inputs(self.root, self.pins)
        path.unlink()
        original = self.root / 'device-libs-source/ocml/a.cl'
        data = original.read_bytes()
        original.unlink()
        with self.assertRaises(ValueError):
            probe.verify_inputs(self.root, self.pins)
        original.write_bytes(data)
        self.manifest['files'] *= 2
        manifest = self.root / 'device-libs-sources.json'
        manifest.write_text(json.dumps(self.manifest))
        self.pins['reference_files'][manifest.name] = probe.sha(manifest)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            probe.verify_inputs(self.root, self.pins)

    def test_output_is_fresh_disjoint_and_local(self):
        output = probe.ROOT / 'results' / 'unused-probe-test-output'
        probe.check_output(output, self.root)
        for bad, reference in ((self.root, output), (output, output),
                                (self.root / 'outside', output)):
            with self.subTest(output=bad), self.assertRaises(ValueError):
                probe.check_output(bad, reference)


class ConstantsAndCaller(unittest.TestCase):
    def test_constant_values_and_duplicate_definition(self):
        for value in probe.CONSTANTS.values():
            name, kind, number = value
            text = f'@{name} = addrspace(4) constant {kind} {number}, align 4\n'
            probe.audit_constant(text, value)
            for bad in (text + text, '', text.replace(f'{kind} {number}', f'{kind} {number + 1}')):
                with self.assertRaises(ValueError):
                    probe.audit_constant(bad, value)

    def test_real_call_declaration_required_not_name_mention(self):
        lines = [f'{prefix} float @__ocml_{name}_f32(float %x)' for name in ('sin', 'log', 'tanh')
                 for prefix in ('declare', '  %v = call')]
        text = '\n'.join(lines)
        probe.audit_caller_ir(text)
        for bad in (text.replace('call', '; mention'), text.replace('declare', 'define'),
                    text.replace('__ocml_log_f32', '__fake_log_f32')):
            with self.assertRaises(ValueError):
                probe.audit_caller_ir(bad)


@unittest.skipUnless(FIXTURES, 'supply --fixtures for actual four-profile compiler artifact checks')
class ActualArtifacts(unittest.TestCase):
    def test_four_actual_profiles_and_mutations(self):
        receipt = json.loads((FIXTURES / 'receipt.json').read_text())
        self.assertEqual(receipt['status'], 'passed_offline_compile_link_only')
        self.assertEqual(len(receipt['profiles']), 4)
        for item in receipt['profiles']:
            label = item['target'] + '-' + item['optimization']
            code = FIXTURES / (label + '.hsaco')
            self.assertEqual(probe.sha(code), item['code_sha256'])
            texts = [(FIXTURES / (label + '-' + suffix + '.stdout')).read_text()
                     for suffix in ('notes', 'isa', 'inspect', 'undefined', 'relocations')]
            probe.audit_object(code, item['target'], *texts)
            mutations = [(0, '.private_segment_fixed_size: 0', '.private_segment_fixed_size: 4'),
                         (0, '.wavefront_size: 32', '.wavefront_size: 64'),
                         (2, 'initial_profile=eligible', 'initial_profile=rejected'),
                         (2, 'hidden_arguments=13', 'hidden_arguments=14'),
                         (1, 'global_store_dword', 'flat_store_dword')]
            for index, before, after in mutations:
                altered = copy.copy(texts)
                self.assertIn(before, altered[index])
                altered[index] = altered[index].replace(before, after)
                with self.subTest(profile=label, mutation=after), self.assertRaises(ValueError):
                    probe.audit_object(code, item['target'], *altered)
            for index, bad in ((3, '__ocml_missing\n'), (4, 'R_AMDGPU_ABS64 unresolved\n')):
                altered = copy.copy(texts)
                altered[index] = bad
                with self.assertRaises(ValueError):
                    probe.audit_object(code, item['target'], *altered)
            caller = (FIXTURES / (label + '-caller.ll')).read_text()
            probe.audit_caller_ir(caller)
            trace = (FIXTURES / (label + '-driver.stderr')).read_text()
            command = next(x['argv'] for x in receipt['commands'] if x['label'] == label + '-compile')
            libs = [FIXTURES / 'libs' / a.split('=', 1)[1] for a in command if a.startswith('--hip-device-lib=')]
            linker = Path(receipt['tools']['ld.lld']['path'])
            probe.audit_driver(trace, linker, item['target'], libs)
            for before, after in (('--no-undefined', '--allow-undefined'),
                                  (item['target'], 'gfx9999'), ('ocml.bc', 'wrong.bc'),
                                  ('"-cc1"', '"-cc1" "-ffast-math"')):
                self.assertIn(before, trace)
                with self.assertRaises(ValueError):
                    probe.audit_driver(trace.replace(before, after), linker, item['target'], libs)


if __name__ == '__main__':
    unittest.main()
