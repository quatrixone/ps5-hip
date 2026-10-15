#!/usr/bin/env python3
"""Reject a truncated or trailing-data Lapy ELF before packaging."""
import pathlib
import struct
import sys


def validate(data):
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("helper must be little-endian ELF64")
    section_offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count = struct.unpack_from("<HH", data, 58)
    if entry_size != 64 or count == 0 or section_offset < 64:
        raise ValueError("helper needs ordinary ELF64 section headers")
    if section_offset + count * entry_size != len(data):
        raise ValueError("helper has truncated sections or trailing bytes")
    for index in range(count):
        section = struct.unpack_from("<IIQQQQIIQQ", data, section_offset + index * entry_size)
        if section[1] != 8 and section[4] + section[5] > len(data):
            raise ValueError("helper section extends beyond ELF stream")


if __name__ == "__main__":
    validate(pathlib.Path(sys.argv[1]).read_bytes())
    print("Lapy helper ELF validated")
