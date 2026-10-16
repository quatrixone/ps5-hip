"""Reject changed or omitted compiler inputs before accepting a native object."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_public_hip as b


class CompileInputs(unittest.TestCase):
    def test_dependency_changes_and_input_mutation(self):
        for mode in ('valid', 'missing-source', 'new-header', 'mutated-header'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                source = out / 'source.cpp'
                header = out / 'header.h'
                source.write_text('#include "header.h"\n')
                header.write_text('original')
                obj = out / 'source.o'

                def run(output, label, args):
                    deps = [source, header]
                    if mode == 'missing-source':
                        deps = [header]
                    if '-MD' in args:
                        obj.write_bytes(b'object')
                        if mode == 'new-header':
                            deps.append(out / 'unexpected.h')
                        if mode == 'mutated-header':
                            header.write_text('changed')
                    Path(args[args.index('-MF') + 1]).write_text(
                        'object: ' + ' '.join(map(str, deps)) + '\n')

                with patch.object(b, 'run', side_effect=run):
                    if mode == 'valid':
                        result = b.compile_bound_unit(out, source, obj, ['compiler'], 'compile')
                        self.assertEqual(result['headers'][str(header.resolve())], b.sha(header))
                        self.assertEqual(result['object_sha256'], b.sha(obj))
                    else:
                        with self.assertRaises(RuntimeError):
                            b.compile_bound_unit(out, source, obj, ['compiler'], 'compile')


if __name__ == '__main__':
    unittest.main()
