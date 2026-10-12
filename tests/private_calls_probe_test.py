import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from probe_private_calls import inspect_calls, oracle

PROBES = pathlib.Path(sys.argv.pop(1))


class PrivateCallProbeTest(unittest.TestCase):
    def test_real_objects_and_corruptions(self):
        for target in ('gfx1013', 'gfx1030'):
            for optimization in ('O1', 'O2'):
                folder = PROBES / (target + '-' + optimization)
                args = [(folder / name).read_text() for name in
                        ('isa.txt', 'private_calls.s', 'private_calls.ll', 'metadata.txt')]
                with self.subTest(target=target, optimization=optimization):
                    self.assertEqual(len(inspect_calls(*args)['call_graph']), 2)
                isa, assembly, ir, metadata = args
                bad_call = re.sub(r'(s_add_u32 s6, s6, )(0x[0-9a-f]+)',
                                  lambda m: m[1] + hex((int(m[2], 16) + 4) & 0xffffffff), isa, count=1)
                mutants = [
                    ('call target', 0, bad_call),
                    ('call high word', 0, isa.replace('s_addc_u32 s7, s7, -1', 's_addc_u32 s7, s7, 0')),
                    ('call source pair', 0, isa.replace('s_swappc_b64 s[30:31], s[6:7]', 's_swappc_b64 s[30:31], s[8:9]')),
                    ('call return pair', 0, isa.replace('s_swappc_b64 s[30:31], s[6:7]', 's_swappc_b64 s[28:29], s[6:7]')),
                    ('disconnected reload', 0, re.sub(r'(v_add3_u32 v0, )v\d+(, v0, v\d+)', r'\g<1>v255\2', isa)),
                    ('leaf store', 0, isa.replace('flat_store_dword ', 'removed_store ')),
                    ('post-call reload', 0, isa.replace('buffer_load_dword ', 'removed_load ')),
                    ('retained call', 0, isa.replace('s_swappc_b64', 'removed_call')),
                    ('frame', 0, isa.replace('s_movk_i32 s32, 0x1200', 's_movk_i32 s32, 0x1100')),
                    ('flat init', 0, isa.replace('HW_REG_FLAT_SCR_HI', 'REMOVED_FLAT_HI')),
                    ('noinline', 2, ir.replace('noinline', '')),
                    ('relocations', 3, metadata.replace('Relocations [\n]', 'Relocations [\n pending\n]')),
                    ('dynamic stack', 1, re.sub(r'(\.amdhsa_uses_dynamic_stack\s+)0', r'\g<1>1', assembly)),
                ]
                for name, index, mutant in mutants:
                    changed = args.copy()
                    changed[index] = mutant
                    with self.subTest(target=target, optimization=optimization, corruption=name):
                        self.assertNotEqual(mutant, args[index])
                        with self.assertRaises(ValueError):
                            inspect_calls(*changed)

    def test_observable_store_and_unsigned_edges(self):
        for x in list(range(32)) + list(range(0xffffffc0, 0x100000000)):
            expected = (4*x + 5*(x & 31) + 2*((x + 7) & 31) + 6) & 0xffffffff
            self.assertEqual(oracle(x), expected)
            self.assertEqual((expected - oracle(x, drop_store=True)) & 0xffffffff, 3)


if __name__ == '__main__':
    unittest.main()
