import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import probe_aperture as probe


class ApertureRecipeTests(unittest.TestCase):
    def test_existing_output_refused_before_any_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            marker = output / 'tools-before.json'
            marker.write_text('preserve existing evidence')
            with patch.object(probe, 'out', output):
                with self.assertRaises(FileExistsError):
                    probe.main()
            self.assertEqual(marker.read_text(), 'preserve existing evidence')
            self.assertEqual(list(output.iterdir()), [marker])

    def test_actual_isa_store_contract_and_corruptions(self):
        for target in ('gfx1013', 'gfx1030'):
            for opt in ('O1', 'O2'):
                isa = (ROOT / f'build/aperture-probes/{target}-{opt}/isa.txt').read_text()
                probe.audit_output_isa(isa, opt)
                with self.assertRaises(ValueError):
                    probe.audit_output_isa(isa.replace('s_waitcnt', 's_code_end // 00000000152A: BF9F0000\n\ts_waitcnt', 1), opt)
                for old, new in [('src_private_base', 'src_private_limit'),
                                 ('src_private_limit', 'src_private_base'),
                                 (', 6,', ', 8,'), ('0x6b18d3a5', '0x6b18d3a4'),
                                 ('s_endpgm', 's_setpc_b64 s[30:31]'),
                                 ('global_store_dword', 'flat_store_dword'),
                                 ('offset:16' if opt == 'O2' else 'offset:8', 'offset:32')]:
                    self.assertIn(old, isa)
                    with self.assertRaises(ValueError):
                        probe.audit_output_isa(isa.replace(old, new), opt)

    def test_selected_actual_object_and_provenance_mutations(self):
        for target in ('gfx1013', 'gfx1030'):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for name in ('tools/probe_aperture.py', 'kernels/private_aperture.hip'):
                    dest = root / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, dest)
                directory = root / 'build/aperture-probes'
                directory.mkdir(parents=True)
                original = json.loads((ROOT / 'build/aperture-probes/comparison.json').read_text())
                for opt in ('O1', 'O2'):
                    shutil.copytree(ROOT / f'build/aperture-probes/{target}-{opt}', directory / f'{target}-{opt}')
                report_path = directory / 'comparison.json'
                report_path.write_text(json.dumps(original))
                data, _ = probe.verified_aperture_object(root, target, '18.1.8')
                self.assertEqual(data, (directory / f'{target}-O2/aperture.hsaco').read_bytes())
                for name in ('tools/probe_aperture.py', 'kernels/private_aperture.hip',
                             f'build/aperture-probes/{target}-O2/aperture.hsaco',
                             f'build/aperture-probes/{target}-O2/driver.txt'):
                    path = root / name
                    previous = path.read_bytes()
                    path.write_bytes(previous + b'changed')
                    with self.assertRaises(ValueError):
                        probe.verified_aperture_object(root, target, '18.1.8')
                    path.write_bytes(previous)
                for mutation in ('duplicate', 'missing-o2', 'version'):
                    report = copy.deepcopy(original)
                    selected = next(p for p in report['cases'] if p['target'] == target and p['optimization'] == 'O2')
                    if mutation == 'duplicate': report['cases'].append(selected)
                    if mutation == 'missing-o2': selected['optimization'] = 'O1'
                    if mutation == 'version': report['tool_versions']['clang++'] = 'clang version 19.0.0'
                    report_path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError):
                        probe.verified_aperture_object(root, target, '18.1.8')
                report_path.write_text(json.dumps(original))
                shutil.copyfile(directory / f'{target}-O1/aperture.hsaco', directory / f'{target}-O2/aperture.hsaco')
                with self.assertRaises(ValueError):
                    probe.verified_aperture_object(root, target, '18.1.8')


if __name__ == '__main__':
    unittest.main()
