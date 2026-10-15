import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_llm_reference_fixtures as fixture


class ReferenceFixtureTest(unittest.TestCase):
    def test_logit_validator_requires_full_finite_rows_and_exact_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            logits, tokens = Path(directory) / 'logits', Path(directory) / 'tokens'
            rows = [1., 2., 2., 0., 0., -1., 3., 1.]
            logits.write_bytes(struct.pack('<8f', *rows))
            tokens.write_text('1\n')
            result = fixture.validate_logits(logits, tokens, 1, vocabulary=4)
            self.assertEqual(result['rows'], 2)
            self.assertEqual(result['generated_tokens'], [1])
            for invalid in ('2\n', '4\n', '', '1\n2\n'):
                tokens.write_text(invalid)
                with self.assertRaises(ValueError):
                    fixture.validate_logits(logits, tokens, 1, vocabulary=4)
            tokens.write_text('1\n')
            for bad in (struct.pack('<7f', *rows[:7]), struct.pack('<9f', *(rows + [0.])),
                        struct.pack('<8f', *([float('nan')] + rows[1:])),
                        struct.pack('<8f', *(rows[:-1] + [float('inf')]))):
                logits.write_bytes(bad)
                with self.assertRaises(ValueError):
                    fixture.validate_logits(logits, tokens, 1, vocabulary=4)

    def test_locked_cases_and_context_boundary(self):
        profile = json.loads(fixture.PROMPTS.read_text())
        for case in profile['cases']:
            fixture.validate_case(case)
        boundary = next(case for case in profile['cases'] if case['name'] == 'context_boundary')
        self.assertEqual(len(boundary['tokens']) + boundary['decode_steps'], 512)
        self.assertTrue(any(case['decode_steps'] == 64 for case in profile['cases']))
        for replacement in ({'tokens': [True]}, {'tokens': [32000]}, {'tokens': []},
                            {'decode_steps': 65}, {'tokens': [1] * 497}, {'name': '../bad'}):
            with self.assertRaises(ValueError):
                fixture.validate_case({**boundary, **replacement})

    def test_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'fresh directory'):
                fixture.build(root, root, root)


if __name__ == '__main__':
    unittest.main()
