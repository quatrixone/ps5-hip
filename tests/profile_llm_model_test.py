import io
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from profile_llm_model import bounded_read, inventory, derive, parse_lfs_pointer


class Response(io.BytesIO):
    def __init__(self, status, headers, data=b'12345678'):
        super().__init__(data)
        self.status, self.headers, self.reads = status, headers, 0

    def read(self, size=-1):
        self.reads += 1
        return super().read(size)


class ProfileTest(unittest.TestCase):
    def test_lfs_pointer(self):
        valid = b'version https://git-lfs.github.com/spec/v1\noid sha256:' + b'a' * 64 + b'\nsize 2200119864\n'
        self.assertEqual(parse_lfs_pointer(valid), dict(sha256='a' * 64, size=2200119864))
        for data in (valid.replace(b'sha256:', b'sha1:'), valid.replace(b'size 2200119864', b'size -1'),
                     valid.replace(b'a' * 64, b'z' * 64), valid + b'unknown field\n', b'weight data'):
            with self.assertRaises(ValueError):
                parse_lfs_pointer(data)

    def test_reject_before_read(self):
        for status, headers in ((200, {}), (206, {'Content-Range': 'bytes 1-8/100'}),
                                (206, {'Content-Range': 'bytes 0-7/100', 'Content-Length': '3000000000'}),
                                (206, {'Content-Range': 'bytes 0-7/100', 'Content-Encoding': 'gzip'})):
            response = Response(status, headers)
            with self.assertRaises(ValueError):
                bounded_read(response, 8, (0, 7))
            self.assertEqual(response.reads, 0)

    def test_exact_and_truncated_range(self):
        self.assertEqual(bounded_read(Response(206, {'Content-Range': 'bytes 0-7/100'}), 8, (0, 7)), (b'12345678', 100))
        for data in (b'123', b'123456789'):
            with self.assertRaises(ValueError):
                bounded_read(Response(206, {'Content-Range': 'bytes 0-7/100'}, data), 8, (0, 7))

    def test_inventory_and_estimates(self):
        header = {'matrix': {'dtype': 'BF16', 'shape': [2, 32], 'data_offsets': [0, 128]},
                  'norm': {'dtype': 'F32', 'shape': [32], 'data_offsets': [128, 256]}}
        tensors = inventory(header, 256)
        config = dict(num_attention_heads=4, num_key_value_heads=2, hidden_size=64, num_hidden_layers=3)
        stats = derive(config, tensors, 8)
        self.assertEqual((stats['parameters'], stats['f16_weight_bytes'], stats['q4_0_estimate_bytes']), (96, 192, 164))
        self.assertEqual(stats['kv_cache']['bytes'], 3072)
        header['norm']['data_offsets'] = [64, 192]
        with self.assertRaises(ValueError):
            inventory(header, 256)
        header['matrix']['shape'] = [True, 32]
        with self.assertRaises(ValueError):
            inventory(header, 256)


if __name__ == '__main__':
    unittest.main()
