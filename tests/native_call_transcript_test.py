import sys
import unittest
from pathlib import Path

DRAFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRAFT / 'tools'))
from source_public_hip_log import check_log, LLM_CLIENT_EXPECTED, LLM_STORAGE_HIP_CALL_LINES
from source_llm_model_log import check_model_log, EXPECTED


class Transcript(unittest.TestCase):
    def test_vocab_and_model_acceptance_retained(self):
        check_log(LLM_CLIENT_EXPECTED, 'llm_client')
        check_model_log(EXPECTED)
        self.assertNotIn(b'HIP_CALL', EXPECTED)

    def test_every_row_must_appear_exactly_once_and_in_order(self):
        for line in LLM_STORAGE_HIP_CALL_LINES:
            encoded = (line + '\n').encode()
            for altered in (LLM_CLIENT_EXPECTED.replace(encoded, b''),
                            LLM_CLIENT_EXPECTED.replace(encoded, encoded + encoded)):
                with self.subTest(line=line), self.assertRaises(ValueError):
                    check_log(altered, 'llm_client')
        first, second = [line.encode() for line in LLM_STORAGE_HIP_CALL_LINES[:2]]
        with self.assertRaises(ValueError):
            check_log(LLM_CLIENT_EXPECTED.replace(first, b'TEMP').replace(second, first).replace(b'TEMP', second), 'llm_client')

    def test_accounting_and_result_mutations_rejected(self):
        line = LLM_STORAGE_HIP_CALL_LINES[5].encode()
        for old, new in ((b'attempts=1', b'attempts=2'), (b'completions=1', b'completions=0'),
                         (b'successes=1', b'successes=0'), (b'failures=0', b'failures=1'),
                         (b'requested_bytes=64', b'requested_bytes=63'),
                         (b'last_result=0', b'last_result=1'), (b'first_error=0', b'first_error=1')):
            with self.subTest(field=old), self.assertRaises(ValueError):
                check_log(LLM_CLIENT_EXPECTED.replace(line, line.replace(old, new)), 'llm_client')

    def test_no_added_pointers_warnings_or_missing_finalization(self):
        for suffix in (b' pointer=0x1234', b' warning=true'):
            with self.assertRaises(ValueError):
                check_log(LLM_CLIENT_EXPECTED.replace(b'call=h2d ', b'call=h2d' + suffix + b' '), 'llm_client')
        for transcript, checker in ((LLM_CLIENT_EXPECTED, lambda value: check_log(value, 'llm_client')),
                                    (EXPECTED, check_model_log)):
            for line in transcript.splitlines(keepends=True):
                if b'FINAL' in line or b'FINALIZER' in line or b'application_fini' in line:
                    with self.assertRaises(ValueError):
                        checker(transcript.replace(line, b''))


if __name__ == '__main__':
    unittest.main()
