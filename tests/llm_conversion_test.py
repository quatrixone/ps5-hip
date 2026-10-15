import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import convert_llm_model as convert
import prepare_llm_conversion as prepare


class ConversionTest(unittest.TestCase):
    def test_inventory_rejects_wrong_shape_count_dtype_and_duplicates(self):
        def tensor(name='weight', dtype='Q4_0', elements=32):
            return SimpleNamespace(name=name, tensor_type=SimpleNamespace(name=dtype),
                                   shape=[32], n_elements=elements, n_bytes=18)
        expected = dict(tensor_count=1, parameters=32)
        config = dict(max_position_embeddings=512, hidden_size=32, num_hidden_layers=1,
                      intermediate_size=64, num_attention_heads=2, num_key_value_heads=1)
        values = {'general.architecture': 'llama', 'llama.context_length': 512,
                  'llama.embedding_length': 32, 'llama.block_count': 1, 'llama.feed_forward_length': 64,
                  'llama.attention.head_count': 2, 'llama.attention.head_count_kv': 1}
        fields = {k: SimpleNamespace(contents=lambda v=v: v) for k, v in values.items()}
        reader = lambda tensors: SimpleNamespace(tensors=tensors, fields=fields)
        result = convert.inventory(reader([tensor()]), expected, ['Q4_0'], config)
        self.assertEqual(result['dtype_counts'], {'Q4_0': 1})
        for tensors in ([], [tensor(dtype='Q6_K')], [tensor(elements=64)], [tensor(), tensor()]):
            with self.assertRaises(ValueError):
                convert.inventory(reader(tensors), expected, ['Q4_0'], config)
        with self.assertRaises(ValueError):
            convert.inventory(reader([tensor(), tensor()]), dict(tensor_count=2, parameters=64), ['Q4_0'], config)
        with self.assertRaises(ValueError):
            convert.inventory(reader([tensor(dtype='F32')]), expected, ['Q4_0', 'F32'], config)
        fields['general.architecture'] = SimpleNamespace(contents=lambda: 'other')
        with self.assertRaises(ValueError):
            convert.inventory(reader([tensor()]), expected, ['Q4_0'], config)

    def test_preserves_existing_output_and_rejects_unprepared_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'fresh directory'):
                convert.convert(root, root, root, root)
            with self.assertRaisesRegex(ValueError, 'prepared conversion environment'):
                convert.convert(root, root / 'output', root, root)
            self.assertFalse((root / 'output').exists())

    def test_environment_rejects_wrong_platform(self):
        with patch.object(prepare.platform, 'system', return_value='Windows'):
            with self.assertRaisesRegex(ValueError, 'Linux x86-64'):
                prepare.prepare(Path('unused'))

    def test_environment_reuse_preserves_receipt_and_checks_tool_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            helper = root / 'tools/fetch_llm_model.py'
            helper.write_text('original helper')
            requirements = root / 'requirements.txt'
            requirements.write_text('locked requirements')
            lock = root / 'lock.json'
            lock.write_text(json.dumps(dict(requirements_sha256=hashlib.sha256(requirements.read_bytes()).hexdigest(),
                python=dict(sha256='a' * 64), python_version='Python 3.12.15', packages={'test': '1'})))
            lock_sha = hashlib.sha256(lock.read_bytes()).hexdigest()
            output = root / 'env'
            (output / 'venv').mkdir(parents=True)
            (output / 'runtime-ready').write_text('a' * 64)
            (output / 'configuration.json').write_text(json.dumps(dict(lock_sha256=lock_sha, tool_sha256='b' * 64)))
            (output / 'receipt.json').write_text(json.dumps(dict(classification='isolated-converter-installed', lock_sha256=lock_sha)))
            def command(argv, **kwargs):
                return 'Python 3.12.15' if argv[-1] == '--version' else '[{"name":"test","version":"1"}]'
            with patch.object(prepare, 'ROOT', root), patch.object(prepare, 'LOCK', lock), patch.object(prepare, 'REQUIREMENTS', requirements), patch.object(prepare, 'fetch_file'), patch.object(prepare.subprocess, 'check_output', side_effect=command), patch.object(prepare.subprocess, 'run') as run:
                prepare.prepare(output)
                self.assertTrue(json.loads((output / 'receipt.json').read_text())['reused_installed_packages'])
                self.assertEqual(len(list((output / 'receipts').iterdir())), 1)
                self.assertTrue(all('install' not in call.args[0] for call in run.call_args_list))
                run.side_effect = lambda *a, **k: helper.write_text('changed during verification')
                with self.assertRaisesRegex(ValueError, 'setup tool changed'):
                    prepare.prepare(output)
                self.assertFalse((output / 'receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
