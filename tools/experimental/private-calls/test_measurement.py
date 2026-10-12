"""Negative semantic-contract controls; no compiler executed."""
from pathlib import Path
import copy
import importlib.util
import json
import unittest

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('binding', root / 'validation/measurement_binding_v1.py')
binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binding)
reference = json.loads((root / 'measurement/reference-contracts.json').read_text())


class MeasurementTests(unittest.TestCase):
    def test_fresh_provenance_never_changes_semantic_reference(self):
        fresh = copy.deepcopy(reference)
        fresh['compiler']['sha256'] = 'different independent clean build'
        fresh['profiles'][0]['input_sha256'] = 'different path-dependent input digest'
        fresh['profiles'][0]['raw_files'] = {'new object': 'different raw digest'}
        binding.compare_semantics(fresh, reference)

    def test_resource_call_effect_and_observable_mutations_rejected(self):
        root_name = 'nested_private'
        middle = '_Z11call_middlePjj.private5'
        leaf = '_Z9call_leafPjj.private5'
        paths = [
            ('descriptors', root_name, 'private_segment_fixed_size'),
            ('descriptors', root_name, 'next_free_vgpr'),
            ('descriptors', root_name, 'user_sgpr_count'),
            ('nested_frame', 'middle_per_lane'),
            ('nested_frame', 'root_wave_stack_init'),
            ('reachable', middle, 'calls', 0),
            ('reachable', leaf, 'buffer_store'),
            ('reachable', leaf, 'flat_store'),
            ('reachable', root_name, 'flat_init'),
            ('middle_saved_restore_instructions', 0),
            ('middle_postcall_observable_window', 0),
            ('middle_postcall_observable_window', 4),
            ('middle_postcall_observable_window', 5),
        ]
        for path in paths:
            fresh = copy.deepcopy(reference)
            value = fresh['profiles'][0]
            for key in path[:-1]:
                value = value[key]
            old = value[path[-1]]
            value[path[-1]] = old + 1 if isinstance(old, int) else old + ' changed'
            with self.subTest(path=path), self.assertRaises(AssertionError):
                binding.compare_semantics(fresh, reference)

    def test_configuration_or_coverage_change_rejected(self):
        for change in ('ipra', 'missing', 'duplicate'):
            fresh = copy.deepcopy(reference)
            if change == 'ipra':
                fresh['configuration'] = ['-enable-ipra=true', '-verify-machineinstrs']
            elif change == 'missing':
                fresh['profiles'].pop()
            else:
                fresh['profiles'][-1] = fresh['profiles'][0]
            with self.subTest(change=change), self.assertRaises(AssertionError):
                binding.compare_semantics(fresh, reference)


if __name__ == '__main__':
    unittest.main()
