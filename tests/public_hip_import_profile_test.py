"""Keep GPU and loader-only import profiles fail-closed; ELF parsing is tested separately."""
from pathlib import Path
import unittest
from unittest.mock import patch
from public_hip_imports_test import verify, nid

SHELL = {'sceShellCoreUtilExitApp': ('libSceSystemService', 'libSceShellCoreUtil')}
GPU = {'sceAgcInit': ('libSceAgc', 'libSceAgc'),
       'sceAgcSuspendPoint': ('libSceAgc', 'libSceAgc'),
       'sceAgcDriverSubmitDcb': ('libSceAgcDriver', 'libSceAgcDriver')}

class ImportProfileTest(unittest.TestCase):
    def verify(self, symbols, require_gpu, modules=None, libraries=None):
        with patch('public_hip_imports_test.read_imports', return_value=(
                modules or {}, libraries or {}, {nid(k): v for k, v in symbols.items()})) as read:
            result = verify(Path('fixture.elf'), require_gpu=require_gpu)
            read.assert_called_once_with(Path('fixture.elf'), verify_hash=True)
            return result

    def test_gpu_profile_retains_all_bindings(self):
        self.assertEqual(self.verify(SHELL | GPU, True), SHELL | GPU)
        for missing in (SHELL | GPU):
            with self.assertRaises(ValueError):
                self.verify({k: v for k, v in (SHELL | GPU).items() if k != missing}, True)

    def test_loader_profile_requires_shell(self):
        self.assertEqual(self.verify(SHELL, False), SHELL)
        for symbols in ({}, {'sceShellCoreUtilExitApp': ('libSceSystemService', 'wrong')}):
            with self.assertRaises(ValueError):
                self.verify(symbols, False)

    def test_loader_rejects_gpu_symbols_even_with_false_owners(self):
        for symbol in GPU:
            with self.assertRaises(ValueError):
                self.verify(SHELL | {symbol: ('other', 'other')}, False)

    def test_loader_rejects_gpu_module_or_library_dependencies(self):
        for name in ('libSceAgc', 'libSceAgcDriver', 'libSceAgcUnexpected'):
            for modules, libraries in (({1: name}, {}), ({}, {1: name})):
                with self.assertRaises(ValueError):
                    self.verify(SHELL, False, modules, libraries)

if __name__ == '__main__':
    unittest.main()
