#!/usr/bin/env python3
"""Independent exact-rational checks plus actual generated C++ checker boundaries."""
import argparse
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generator', ROOT / 'tools/generate_math_oracle.py')
gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
CONTRACT = json.loads((ROOT / 'tools/math_oracle_contract.json').read_text())
ROWS = [dict(zip(CONTRACT['expected_columns'], row)) for row in CONTRACT['expected']]
GENERATION = None


def power(n):
    return F(2**n) if n >= 0 else F(1, 2**-n)


def value(bits):
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    if exponent == 255:
        raise ValueError('nonfinite rational')
    return (-1 if bits >> 31 else 1) * F(fraction | (0x800000 if exponent else 0)) * power(exponent-150 if exponent else -149)


def rounded(x, square=False):
    """Independent integer binary search and exact midpoint/ties-even rounding."""
    sign = 0x80000000 if x < 0 else 0
    x = abs(x)
    if not square and x >= value(0x7f7fffff) + power(103):
        return sign | 0x7f800000
    lo, hi = 0, 0x7f7fffff
    while lo < hi:
        middle = (lo + hi + 1) // 2
        point = value(middle)
        if (point * point if square else point) <= x:
            lo = middle
        else:
            hi = middle - 1
    if lo == 0x7f7fffff:
        return sign | lo
    midpoint = (value(lo) + value(lo + 1)) / 2
    threshold = midpoint * midpoint if square else midpoint
    return sign | (lo + (x > threshold or (x == threshold and lo & 1)))


def hexadecimal(text):
    sign = -1 if text.startswith('-') else 1
    mantissa, exponent = text.lstrip('-').split('p')
    whole, _, fraction = mantissa[2:].partition('.')
    return sign * F(int(whole + fraction, 16), 16**len(fraction)) * power(int(exponent))


class OracleTests(unittest.TestCase):
    def test_render_and_provenance(self):
        path = ROOT / 'tools/math_oracle_contract.json'
        self.assertEqual(CONTRACT['input_bits'], gen.corpus())
        self.assertEqual(len(set(CONTRACT['input_bits'])), 257)
        self.assertEqual(len(ROWS), 1542)
        self.assertEqual((ROOT / 'tools/math_oracle.h').read_text(), gen.render_header(CONTRACT, gen.sha(path)))
        for name, digest in CONTRACT['generator_sources'].items():
            self.assertEqual(gen.sha(ROOT / name), digest)
        self.assertFalse(CONTRACT['hardware_validated'])

    def test_exact_rational_fma_sqrt(self):
        unfused_differences = 0
        for i, bits in enumerate(CONTRACT['input_bits']):
            if bits & 0x7f800000 == 0x7f800000:
                continue
            x = value(bits)
            fused = rounded(x * F(5, 4) - F(1, 2))
            self.assertEqual(ROWS[6*i]['reference'], fused)
            arg = rounded(abs(x) + F(1, 4))
            self.assertEqual(ROWS[6*i+1]['reference'], rounded(value(arg), square=True))
            product = rounded(x * F(5, 4))
            if product & 0x7f800000 != 0x7f800000:
                separate = rounded(value(product) - F(1, 2))
                if separate != fused:
                    unfused_differences += 1
                    self.assertFalse(gen.accepts(ROWS[6*i], separate))
        self.assertGreater(unfused_differences, 0)

    def test_specials_and_ftz(self):
        def row(bits, op):
            return ROWS[6 * CONTRACT['input_bits'].index(bits) + op]
        for sign in (0, 0x80000000):
            for op in (2, 5):
                self.assertEqual(row(sign, op)['reference'], sign)
                self.assertFalse(gen.accepts(row(sign, op), sign ^ 0x80000000))
                for tiny in (1, 0x007fffff, 0x00800000):
                    self.assertEqual(row(sign | tiny, op)['reference'], sign | tiny)
                    self.assertFalse(gen.accepts(row(sign | tiny, op), sign))
            self.assertEqual(row(sign, 3)['reference'], 0x3f800000)
            self.assertEqual(row(sign, 0)['reference'], 0xbf000000)
            self.assertEqual(row(sign, 1)['reference'], 0x3f000000)
            inf = sign | 0x7f800000
            self.assertEqual(row(inf, 5)['reference'], sign | 0x3f800000)
            self.assertEqual(row(inf, 3)['reference'], 0 if sign else 0x7f800000)
            self.assertEqual(row(inf, 2)['rule'], 1)

    def test_all_boundaries_and_cpp(self):
        checks = []
        def check(i, bits, accepted):
            self.assertEqual(gen.accepts(ROWS[i], bits), accepted, (i, hex(bits)))
            checks.append((i, bits, accepted))
        for i, r in enumerate(ROWS):
            check(i, r['reference'], True)
            check(i, gen.poison(r, i), False)
            if r['rule'] == 0:
                check(i, r['reference'] ^ 1, False)
            elif r['rule'] == 1:
                for bits in (0x7fc00001, 0xffffffff, 0xffc12345): check(i, bits, True)
                for bits in (0x7f800001, 0xff800001, 0x7f800000, 0, 0x3f800000): check(i, bits, False)
            else:
                check(i, r['lo'], True); check(i, r['hi'], True)
                lower = r['lo'] + (1 if r['lo'] >> 31 else -1)
                upper = r['hi'] + (-1 if r['hi'] >> 31 else 1)
                check(i, lower, False); check(i, upper, False)
                for bits in (0, 0x80000000, r['reference'] ^ 0x80000000, 0x7f800000, 0x7fc00000): check(i, bits, False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = '#include "math_oracle.h"\n#include <cstdio>\nint main(){unsigned i,b,a;while(std::scanf("%u %x %u",&i,&b,&a)==3) if(ps5hip_math::check_output(i,b)!=bool(a))return 1;return ps5hip_math::check_output(1542,0)?2:0;}\n'
            (path / 'test.cpp').write_text(source)
            subprocess.run(['clang++-18', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I'+str(ROOT/'tools'), str(path/'test.cpp'), '-o', str(path/'test')], check=True, capture_output=True)
            subprocess.run([str(path/'test')], input=''.join(f'{i} {b:x} {int(a)}\n' for i,b,a in checks), text=True, check=True, capture_output=True)

    def test_directed_certificate(self):
        if GENERATION is None:
            self.skipTest('raw directed certificates require --generation')
        certificate = GENERATION / 'generate.stdout'
        self.assertEqual(gen.sha(certificate), CONTRACT['directed_certificate']['sha256'])
        raw = [json.loads(line) for line in certificate.read_text().splitlines()]
        self.assertEqual(len(raw), len(ROWS))
        for i, (r, detailed) in enumerate(zip(ROWS, raw)):
            self.assertEqual([detailed[k] for k in CONTRACT['expected_columns']], CONTRACT['expected'][i])
            self.assertEqual((detailed['index'], detailed['operation']), divmod(i, 6))
            if detailed['precision'] == 0: continue
            left, right = map(hexadecimal, (detailed['reference_lo'], detailed['reference_hi']))
            self.assertLessEqual(left, right)
            self.assertEqual(rounded(left), r['reference'])
            self.assertEqual(rounded(right), r['reference'])
            if r['rule'] == 2:
                budget = CONTRACT['finite_ulp_budgets'][i % 6] * power(detailed['ulp_exponent'])
                for endpoint in (r['lo'], r['hi']):
                    actual = value(endpoint)
                    self.assertLessEqual(abs(actual-left), budget)
                    self.assertLessEqual(abs(actual-right), budget)
                    self.assertEqual(actual > 0, value(r['reference']) > 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation', type=Path)
    args, remaining = parser.parse_known_args()
    GENERATION = args.generation
    unittest.main(argv=[__file__] + remaining)
