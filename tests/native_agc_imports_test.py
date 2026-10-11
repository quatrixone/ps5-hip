"""Check AGC provider ownership in a converted native ELF, without executing it."""
import argparse
import pathlib

from native_import_libraries_test import nid, read_imports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=pathlib.Path)
    args = parser.parse_args()
    _, _, symbols = read_imports(args.elf)
    expected = {
        "sceAgcInit": ("libSceAgc", "libSceAgc"),
        "sceAgcSuspendPoint": ("libSceAgc", "libSceAgc"),
        "sceAgcDriverSubmitDcb": ("libSceAgcDriver", "libSceAgcDriver"),
    }
    failed = False
    for name, provider in expected.items():
        actual = symbols.get(nid(name))
        matches = actual == provider
        print(f"{'PASS' if matches else 'FAIL'} {name} ({nid(name)}): "
              f"module/library={actual!r}, expected={provider!r}")
        failed |= not matches
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
