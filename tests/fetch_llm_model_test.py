import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('fetch', Path(__file__).resolve().parents[1] / 'tools/fetch_llm_model.py')
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


class Response(io.BytesIO):
    status = 200

    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = {} if headers is None else headers


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b'locked weights'
        self.spec = dict(url='https://huggingface.co/test', bytes=len(self.data),
                         sha256=hashlib.sha256(self.data).hexdigest())
        self.path = self.root / 'model.safetensors'

    def test_verified_reuse_and_corrupt_preservation(self):
        with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(self.data)) as call:
            fetch.fetch_file(self.path, self.spec)
            fetch.fetch_file(self.path, self.spec)
            self.assertEqual(call.call_count, 1)
        self.path.write_bytes(b'bad')
        with self.assertRaises(ValueError):
            fetch.fetch_file(self.path, self.spec)
        self.assertEqual(self.path.read_bytes(), b'bad')

    def test_failure_never_publishes(self):
        for data in (self.data[:-1], self.data + b'x', b'x' * len(self.data)):
            with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(data)):
                with self.assertRaises(ValueError):
                    fetch.fetch_file(self.path, self.spec)
            self.assertEqual(list(self.root.iterdir()), [])
        response = Response(self.data, {'Content-Length': '100'})
        with patch.object(fetch.urllib.request, 'urlopen', return_value=response):
            with self.assertRaises(ValueError):
                fetch.fetch_file(self.path, self.spec)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_publication_race_preserves_other_file(self):
        def collide(source, target):
            target.write_bytes(b'other')
            raise FileExistsError(target)
        with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(self.data)), patch.object(fetch.os, 'link', side_effect=collide):
            with self.assertRaises(FileExistsError):
                fetch.fetch_file(self.path, self.spec)
        self.assertEqual(self.path.read_bytes(), b'other')
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_receipt_only_after_full_set(self):
        lock = self.root / 'lock.json'
        lock.write_text(json.dumps(dict(model='test', revision='test', metadata={'config.json': self.spec},
                                       weights=dict(url=self.spec['url'], file_bytes=self.spec['bytes'],
                                                    expected_file_sha256=self.spec['sha256']))))
        receipt = self.root / 'receipt.json'
        output = self.root / 'source'
        with patch.object(fetch.urllib.request, 'urlopen', side_effect=[Response(self.data), Response(b'bad')]):
            with self.assertRaises(ValueError):
                fetch.download(output, receipt, lock)
        self.assertFalse(receipt.exists())
        with patch.object(fetch.urllib.request, 'urlopen', return_value=Response(self.data)):
            fetch.download(output, receipt, lock)
        self.assertFalse(json.loads(receipt.read_text())['model_executed'])
        with self.assertRaises(ValueError):
            fetch.download(output, receipt, lock)


if __name__ == '__main__':
    unittest.main()
