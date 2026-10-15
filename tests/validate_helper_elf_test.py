import struct
import unittest

from tools.validate_helper_elf import validate


class HelperElfTest(unittest.TestCase):
    def test_accepts_bounded_elf64(self):
        data = bytearray(128)
        data[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<Q", data, 40, 64)
        struct.pack_into("<HH", data, 58, 64, 1)
        validate(data)

    def test_rejects_trailing_and_out_of_bounds_sections(self):
        data = bytearray(128)
        data[:6] = b"\x7fELF\x02\x01"
        struct.pack_into("<Q", data, 40, 64)
        struct.pack_into("<HH", data, 58, 64, 1)
        with self.assertRaises(ValueError):
            validate(data + b"x")
        struct.pack_into("<Q", data, 64 + 24, 120)
        struct.pack_into("<Q", data, 64 + 32, 16)
        with self.assertRaises(ValueError):
            validate(data)


if __name__ == "__main__":
    unittest.main()
