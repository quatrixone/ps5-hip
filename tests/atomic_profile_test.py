"""Real frozen atomic objects and synthetic staging/log receipts; no native build."""
import copy
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import atomic_fixture as fixture
import build_canaries
import build_native_canary as builder
import check_canary_log as checker
import stage_native_title as stager
import verify_native_case as verifier


class AtomicProfiles(unittest.TestCase):
    def test_fixture_tampering_and_unknown_profiles(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        names = ['kernels/atomic_probe.hip', 'build/atomic-probes/gfx1013-O2/atomics.hsaco',
                 'build/atomic-probes/gfx1013-O2/driver.txt']
        for name in names:
            destination = root / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        fixture.verified_object(root, 'gfx1013')
        for name in names:
            path = root / name; original = path.read_bytes(); path.write_bytes(original + b'changed')
            with self.assertRaises(ValueError): fixture.verified_object(root, 'gfx1013')
            path.write_bytes(original)
        for target, optimization in [('gfx900', 'O2'), ('gfx1013', 'O3')]:
            with self.assertRaises(ValueError): fixture.verified_object(root, target, optimization)

    def test_four_real_objects_and_isa(self):
        self.assertEqual(fixture.digest((ROOT / 'kernels/atomic_probe.hip').read_bytes()), fixture.SOURCE_SHA256)
        for target, optimization in fixture.PROFILES:
            code, driver, binding = fixture.verified_object(ROOT, target, optimization)
            build_canaries.require_device_linker(driver.decode(), shutil.which('ld.lld-18'))
            self.assertEqual(len(code), 8272)
            self.assertEqual(binding['optimization'], optimization)
            path = ROOT / 'build/atomic-probes' / (target + '-' + optimization) / 'atomics.hsaco'
            isa = subprocess.check_output(['llvm-objdump-18', '-d', '--mcpu=' + target, str(path)], text=True)
            inventory = build_canaries.isa_inventory(isa, set(fixture.KERNELS))
            blocks = re.split(r'\n[0-9a-fA-F]+ <([^>]+)>:\n', isa)
            bodies = dict(zip(blocks[1::2], blocks[2::2]))
            for index, name in enumerate(fixture.KERNELS):
                operation = 'cmpswap' if index in (2, 3) else 'add'
                opcode = 'global_atomic_' + operation + ('_x2' if index % 2 else '')
                self.assertIn(opcode, inventory[name]['opcodes'])
                self.assertRegex(bodies[name], r'\b' + opcode + r'\b[^\n]*\bglc\b')
                self.assertFalse(inventory[name]['flat_scratch_setup'])
                self.assertFalse(any(op.startswith(('flat_', 'scratch_', 'ds_')) or op == 's_swappc_b64'
                                     for op in inventory[name]['opcodes']))
                if index >= 4:
                    self.assertIn('global_load_dword', inventory[name]['opcodes'])
                    self.assertNotIn('v_readfirstlane_b32', inventory[name]['opcodes'])
            metadata = subprocess.check_output(['llvm-readobj-18', '--notes', str(path)], text=True)
            for key, value in [('group_segment_fixed_size', 0), ('private_segment_fixed_size', 0), ('wavefront_size', 32)]:
                self.assertEqual(re.findall(r'\.' + key + r': (\d+)', metadata), [str(value)] * 6)

    def make_candidate(self, target):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        root = pathlib.Path(temporary.name)
        build = root / 'build'; (build / 'candidate/sce_module').mkdir(parents=True)
        (root / 'assets').mkdir(); shutil.copyfile(ROOT / 'assets/icon0.png', root / 'assets/icon0.png')
        sources = ('kernels/atomic_probe.hip', 'tools/native_atomics.h', 'tools/atomic_fixture.py')
        for source in sources:
            destination = root / source; destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / source, destination)
        data, _, binding = fixture.verified_object(ROOT, target)
        files = {'eboot.bin': b'synthetic host container', 'lapy.elf': b'synthetic helper',
                 'sce_module/libc.prx': b'synthetic companion', 'canary.hsaco': data}
        for name, contents in files.items(): (build / 'candidate' / name).write_bytes(contents)
        receipt = {'kernel': 'atomic_suite', 'target': target, 'hardware_validated': False,
                   'shutdown_mode': 'shell-request-probe', 'dispatches': 6,
                   'dispatch_kernels': fixture.KERNELS[:], 'atomic': copy.deepcopy(fixture.CONTRACT),
                   'device_build': binding, 'source_sha256': {name: fixture.digest((root / name).read_bytes()) for name in sources},
                   'artifacts': {name: fixture.digest(contents) for name, contents in files.items()}}
        return root, build, receipt

    def test_build_stage_and_verifier_contracts(self):
        for target in ('gfx1013', 'gfx1030'):
            root, build, receipt = self.make_candidate(target)
            good = copy.deepcopy(receipt)
            bad_receipts = []
            for value in (True, 6.0, '6', 5, 7):
                bad = copy.deepcopy(good); bad['dispatches'] = value; bad_receipts.append(bad)
            for key, value in fixture.CONTRACT.items():
                bad = copy.deepcopy(good); bad['atomic'][key] = value + 1 if type(value) is int else '0'; bad_receipts.append(bad)
                if type(value) is int:
                    bad = copy.deepcopy(good); bad['atomic'][key] = float(value); bad_receipts.append(bad)
            for field, value in [('optimization', 'O1'), ('code_sha256', '0' * 64), ('bytes', 8272.0),
                                 ('target', 'gfx1030' if target == 'gfx1013' else 'gfx1013')]:
                bad = copy.deepcopy(good); bad['device_build'][field] = value; bad_receipts.append(bad)
            bad = copy.deepcopy(good); bad['dispatch_kernels'].reverse(); bad_receipts.append(bad)
            for source in ('kernels/atomic_probe.hip', 'tools/native_atomics.h', 'tools/atomic_fixture.py'):
                bad = copy.deepcopy(good); del bad['source_sha256'][source]; bad_receipts.append(bad)
            for bad in bad_receipts:
                with self.assertRaises(ValueError): fixture.validate_build(bad)
                (build / 'receipt.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    stager.stage(root, build, root / 'rejected', {'title_id': 'PPSA99014', 'title_name': 'atomic test'}, 'a' * 40, '/chosen/results')
                self.assertFalse((root / 'rejected').exists())
            (build / 'receipt.json').write_text(json.dumps(good))
            stage = root / 'stage'
            manifest = stager.stage(root, build, stage, {'title_id': 'PPSA99014', 'title_name': 'atomic test'}, 'a' * 40, '/chosen/results')
            self.assertEqual(manifest['dispatch_kernels'], fixture.KERNELS)
            self.assertEqual(manifest['atomic_compiler'], fixture.binding(target))
            self.assertEqual(manifest['expected_log'], '/chosen/results/ps5-hip-' + target + '-atomic_suite.log')
            controllers = root / 'controllers'; controllers.mkdir()
            for name in ('launch.elf', 'close.elf'): (controllers / name).write_bytes(name.encode())
            control = {'title_id': 'PPSA99014', 'artifacts': {name: verifier.fingerprint(controllers / name) for name in ('launch.elf', 'close.elf')}}
            (controllers / 'receipt.json').write_text(json.dumps(control))
            (stage / 'PPSA99014.ffpkg').write_bytes(b'synthetic package')
            package = {'title_id': 'PPSA99014', 'image': {'file': 'PPSA99014.ffpkg', **verifier.fingerprint(stage / 'PPSA99014.ffpkg')}}
            def verify(candidate):
                (stage / 'manifest.json').write_text(json.dumps(candidate))
                package['manifest_sha256'] = verifier.fingerprint(stage / 'manifest.json')['sha256']
                (stage / 'package-receipt.json').write_text(json.dumps(package))
                return verifier.verify_case(stage, controllers, {'candidates': [{'manifest': candidate, 'package': package}]}, {'controller_build': control}, 'PPSA99014')
            result = verify(manifest)
            self.assertEqual(result['dispatches'], 6); self.assertFalse(result['hardware_gate_passed'])
            for field, value in [('dispatches', True), ('dispatch_kernels', list(reversed(fixture.KERNELS))),
                                 ('kernel', 'arithmetic_suite'), ('target', 'gfx1030' if target == 'gfx1013' else 'gfx1013')]:
                bad = copy.deepcopy(manifest); bad[field] = value
                with self.assertRaises(ValueError): verify(bad)
            for field, value in [('bytes', 8272.0), ('optimization', 'O1'), ('code_sha256', '0' * 64)]:
                bad = copy.deepcopy(manifest); bad['atomic_compiler'][field] = value
                with self.assertRaises(ValueError): verify(bad)
            bad = copy.deepcopy(manifest); bad['files']['canary.hsaco']['bytes'] = 8272.0
            with self.assertRaises(ValueError): verify(bad)

    def test_strict_six_dispatch_log(self):
        suffix = ('PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\nPS5HIP SHUTDOWN static-dtor=observed\n'
                  'PS5HIP SHELL_EXIT application-fini=completed\nPS5HIP SHELL_EXIT request=begin\nPS5HIP SHELL_EXIT request-return=0\n')
        for target in ('gfx1013', 'gfx1030'):
            start = f'PS5HIP START target={target} kernel=atomic_suite\nPS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n'
            rounds = [f'PS5HIP DISPATCH index={i}/6 kernel={name}\n'
                      f'PS5HIP SUBMIT target={target} kernel={name} count=16448 groups=257 timeout_ms=2000\n'
                      f'PS5HIP PASS kernel={name} exact_values=16449 guards=unchanged offsets={"unchanged" if i >= 5 else "absent"} allocations=closed\n'
                      for i, name in enumerate(fixture.KERNELS, 1)]
            summary = 'PS5HIP PASS kernel=atomic_suite dispatches=6 exact_values=98694 guards=unchanged allocations=closed\n'
            for shell in (False, True):
                good = start + ''.join(rounds) + summary + (suffix if shell else '')
                self.assertEqual(checker.check_log(good.encode(), target, 'atomic_suite', shell_exit_probe=shell)['application_result'], 'application-pass')
                mutations = [good[:-1], good * 2, good.replace('98694', '98693'), good.replace('16449', '16448'),
                             good.replace('16448 groups=257', '16448 groups=258'), good.replace('guards=unchanged', 'guards=modified'),
                             good.replace('offsets=unchanged', 'offsets=absent'), good.replace('allocations=closed', 'allocations=retained'),
                             good.replace(''.join(rounds), ''.join(reversed(rounds))), good.replace('index=6/6', 'index=5/6')]
                for round in rounds: mutations.extend([good.replace(round, ''), good.replace(round, round + round)])
                for bad in mutations:
                    self.assertNotEqual(checker.check_log(bad.encode(), target, 'atomic_suite', shell_exit_probe=shell)['application_result'], 'application-pass')
                for position in range(6):
                    quarantine = start + ''.join(rounds[:position]) + 'PS5HIP QUARANTINED: GPU ownership unresolved; no retry or local teardown\n'
                    self.assertEqual(checker.check_log(quarantine.encode(), target, 'atomic_suite', shell_exit_probe=shell)['application_result'], 'quarantined')

    def test_builder_selects_real_atomic_fixture_and_macro(self):
        source = pathlib.Path(builder.__file__).read_text()
        self.assertIn('verified_atomic_object(ROOT, args.target)', source)
        self.assertIn('-DPS5HIP_CANARY_ATOMIC_SUITE=1', source)
        for name in ('kernels/atomic_probe.hip', 'tools/native_atomics.h', 'tools/atomic_fixture.py'):
            self.assertIn(name, source)


if __name__ == '__main__': unittest.main()
