"""Pure import-audit rejection checks; no payload execution."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'tools/build_teardown_snapshot.py'
spec = importlib.util.spec_from_file_location('recipe', path)
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def symbol(name, undefined=True):
    return '  1: 0000000000000000 0 FUNC GLOBAL DEFAULT ' + ('UND' if undefined else '5') + ' ' + name + '\n'


class Tests(unittest.TestCase):
    def test_exact_import_boundary(self):
        text = 'NeededLibraries [\n' + ''.join('  ' + name + '\n' for name in sorted(recipe.NEEDED)) + ']\n'
        text += ''.join(symbol(name) for name in sorted(recipe.ALLOWED_IMPORTS))
        providers = {name: '' for name in recipe.NEEDED}
        providers['libkernel_sys.sprx'] = symbol('getfsstat', False)
        self.assertEqual(recipe.audit(text, providers)['getfsstat_provider'], 'libkernel_sys.sprx')
        for name in ('mount', 'unmount', 'nmount', 'syscall', 'ioctl', 'sceSystemServiceKillApp',
                     'sceSystemServiceLaunchApp', 'setuid', 'unlink', 'rename'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                recipe.audit(text + symbol(name), providers)
        for mutation in (text.replace('libkernel_sys.sprx', 'libkernel_web.sprx'),
                         text.replace(symbol('getfsstat'), ''),
                         text.replace(symbol('kill'), ''), text.replace(symbol('sceSystemServiceGetAppId'), '')):
            with self.assertRaises(ValueError): recipe.audit(mutation, providers)
        wrong = dict(providers, **{'libkernel_sys.sprx': ''})
        with self.assertRaises(ValueError): recipe.audit(text, wrong)
        duplicate = dict(providers, **{'libSceLibcInternal.sprx': symbol('getfsstat', False)})
        with self.assertRaises(ValueError): recipe.audit(text, duplicate)


if __name__ == '__main__': unittest.main()
