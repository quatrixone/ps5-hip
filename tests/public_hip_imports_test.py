"""Inspect serialized SCE module/library identities, not just linker stub owners."""
import argparse
from pathlib import Path
from native_import_libraries_test import nid, read_imports


def verify(path, *, require_gpu=True):
    modules, libraries, imports = read_imports(path, verify_hash=True)
    expected = {
        'sceShellCoreUtilExitApp': ('libSceSystemService', 'libSceShellCoreUtil'),
        'sceAgcInit': ('libSceAgc', 'libSceAgc'),
        'sceAgcSuspendPoint': ('libSceAgc', 'libSceAgc'),
        'sceAgcDriverSubmitDcb': ('libSceAgcDriver', 'libSceAgcDriver'),
    }
    if not require_gpu:
        gpu_symbols = set(expected) - {'sceShellCoreUtilExitApp'}
        if (any(nid(symbol) in imports for symbol in gpu_symbols) or
                any(name.startswith('libSceAgc') for name in (*modules.values(), *libraries.values()))):
            raise ValueError('loader-only fixture unexpectedly imports GPU services')
        expected = {'sceShellCoreUtilExitApp': expected['sceShellCoreUtilExitApp']}
    for symbol, binding in expected.items():
        actual = imports.get(nid(symbol))
        if actual != binding:
            raise ValueError(f'{symbol}: serialized module/library {actual!r}, expected {binding!r}')
    return expected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('elf', type=Path)
    parser.add_argument('--no-gpu-imports', action='store_true')
    args = parser.parse_args()
    print(verify(args.elf, require_gpu=not args.no_gpu_imports))
