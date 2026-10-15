"""Inspect serialized SCE module/library identities, not just linker stub owners."""
import argparse
from pathlib import Path
from native_import_libraries_test import nid, read_imports


def verify(path):
    _, _, imports = read_imports(path, verify_hash=True)
    expected = {
        'sceShellCoreUtilExitApp': ('libSceSystemService', 'libSceShellCoreUtil'),
        'sceAgcInit': ('libSceAgc', 'libSceAgc'),
        'sceAgcSuspendPoint': ('libSceAgc', 'libSceAgc'),
        'sceAgcDriverSubmitDcb': ('libSceAgcDriver', 'libSceAgcDriver'),
    }
    for symbol, binding in expected.items():
        actual = imports.get(nid(symbol))
        if actual != binding:
            raise ValueError(f'{symbol}: serialized module/library {actual!r}, expected {binding!r}')
    return expected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('elf', type=Path)
    print(verify(parser.parse_args().elf))
