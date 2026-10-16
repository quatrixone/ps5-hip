import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
helper = ROOT / 'tools/native_ccache.py'
spec = importlib.util.spec_from_file_location('native_ccache', helper)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class Cache(unittest.TestCase):
    def test_disabled_and_missing_auto(self):
        self.assertEqual(c.configuration(Path('/repo'), {'PS5_HIP_CCACHE': '0'}),
                         ({'enabled': False, 'requested': '0'}, []))
        with patch.object(c.shutil, 'which', return_value=None):
            self.assertEqual(c.configuration(Path('/repo'), {})[1], [])
            with self.assertRaises(ValueError):
                c.configuration(Path('/repo'), {'PS5_HIP_CCACHE': '1'})

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            c.configuration(Path('/repo'), {'PS5_HIP_CCACHE': 'yes'})

    def test_dependency_link_and_driver_bypass(self):
        for args in (['compiler', '-c', '-M'], ['compiler', '-c', '-E'],
                     ['compiler', '-###'], ['compiler', 'a.o', '-o', 'app']):
            with patch.object(c, 'configuration', side_effect=AssertionError('no cache for this command')):
                command, env = c.invocation(Path('/repo'), args, {'CCACHE_SLOPPINESS': 'time_macros', 'PATH': '/bin'})
            self.assertEqual(command, args)
            self.assertNotIn('CCACHE_SLOPPINESS', env)
            self.assertEqual(env['CCACHE_CONFIGPATH'], '/dev/null')

    def test_compile_preserves_exact_arguments(self):
        args = ['compiler', '-MD', '-MF', '/fresh/a.d', '-c', 'unit.cpp', '-o', '/fresh/a.o']
        with patch.object(c, 'configuration', return_value=({}, ['/bin/ccache', 'sloppiness='])):
            command, env = c.invocation(Path('/repo'), args, {'CCACHE_PREFIX': 'untrusted', 'PS5_HIP_CCACHE': 'auto'})
        self.assertEqual(command, ['/bin/ccache', 'sloppiness=', args[0],
            '--start-no-unused-arguments', '-nostartfiles', '--end-no-unused-arguments', *args[1:]])
        self.assertNotIn('CCACHE_PREFIX', env)
        with patch.object(c, 'configuration', return_value=({}, [])):
            self.assertEqual(c.invocation(Path('/repo'), args, {})[0], args)

    def test_actual_compiler_and_sdk_chain_bound(self):
        with patch.object(c.shutil, 'which', return_value='/usr/bin/ccache'), \
             patch.object(c.subprocess, 'check_output', return_value='/llvm/bin\n'), \
             patch.object(c, 'shared_libraries', return_value=[Path('/usr/bin/ldd'), Path('/lib/libtest.so')]), \
             patch.object(c, 'sha', return_value='digest'):
            record, prefix = c.configuration(Path('/repo'), {'PS5_HIP_CCACHE_DIR': '/cache'})
        self.assertIn('/llvm/bin/clang++', record['inputs'])
        self.assertIn('/repo/.deps/sdk/bin/prospero-clang++', record['inputs'])
        self.assertIn('/repo/.deps/sdk/bin/clang++', record['inputs'])
        self.assertIn('/lib/libtest.so', record['inputs'])
        self.assertIn('/usr/bin/ldd', record['inputs'])
        self.assertIn('sloppiness=', prefix)
        self.assertIn('max_size=1GiB', prefix)
        self.assertIn('remote_storage=', prefix)

    def test_disabled_historical_record_ignores_current_auto_environment(self):
        with patch.object(c, 'configuration', side_effect=AssertionError('must not re-detect disabled history')):
            c.verify(Path('/repo'), {'enabled': False, 'requested': 'auto'})
            c.verify(Path('/repo'), {'enabled': False, 'requested': '0'})
        with self.assertRaises(ValueError):
            c.verify(Path('/repo'), {'enabled': False, 'requested': '1'})

    def test_enabled_record_identity_and_policy_tampering_rejected(self):
        record = {'enabled': True, 'requested': 'auto', 'directory': '/cache', 'options': ['sloppiness='],
                  'tool': '/bin/ccache', 'tool_sha256': 'expected', 'inputs': {'/llvm/clang': 'expected'}}
        actual = {**record, 'requested': '1'}
        with patch.object(c, 'configuration', return_value=(actual, [])):
            c.verify(Path('/repo'), record)
            for field, value in [('tool_sha256', 'changed'), ('options', ['sloppiness=time_macros'])]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    c.verify(Path('/repo'), {**record, field: value})

    def test_changed_shared_library_rejects_prior_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            library = Path(temporary) / 'libcache.so'
            library.write_bytes(b'original shared library')
            original_sha = c.sha
            def output(argv, **kwargs):
                return ('libcache.so => ' + str(library) + ' (0x1000)\n'
                        if argv[0] == '/usr/bin/ldd' else '/llvm/bin\n')
            with patch.object(c.shutil, 'which', return_value='/usr/bin/ccache'), \
                 patch.object(c.subprocess, 'check_output', side_effect=output), \
                 patch.object(c, 'sha', side_effect=lambda p: original_sha(p) if Path(p) == library else 'unchanged'):
                record, _ = c.configuration(Path('/repo'), {})
                c.verify(Path('/repo'), record)
                library.write_bytes(b'changed shared library')
                with self.assertRaisesRegex(ValueError, 'identity or safe configuration changed'):
                    c.verify(Path('/repo'), record)

    def test_missing_shared_dependency_rejected(self):
        with patch.object(c.subprocess, 'check_output', return_value='libcache.so => not found\n'):
            with self.assertRaisesRegex(ValueError, 'shared library missing'):
                c.shared_libraries('/usr/bin/ccache')

    def test_launcher_does_not_rehash_shared_libraries_per_compile(self):
        with patch.object(c.shutil, 'which', return_value='/usr/bin/ccache'), \
             patch.object(c.subprocess, 'check_output', return_value='/llvm/bin\n'), \
             patch.object(c, 'shared_libraries', side_effect=AssertionError('build identity work only')):
            c.configuration(Path('/repo'), {}, identities=False)

    @unittest.skipUnless(os.environ.get('PS5_HIP_CACHE_INTEGRATION') == '1', 'opt-in real SDK cache test')
    def test_real_wrapper_cache_and_dependency_identity(self):
        for language in ('c', 'c++'):
            with self.subTest(language=language):
                self.check_real_wrapper_cache(language)

    def check_real_wrapper_cache(self, language):
        root = ROOT
        if shutil.which('ccache') is None or not (root / '.deps/sdk/bin/prospero-clang++').is_file():
            self.skipTest('ccache and prepared PS5 SDK required')
        wrapper = root / 'tools/ps5-clangxx'
        with tempfile.TemporaryDirectory(prefix='ps5hip-cache-test-') as directory:
            work = Path(directory)
            (work / 'tools').mkdir()
            (work / '.deps').mkdir()
            (work / '.deps/sdk').symlink_to(root / '.deps/sdk', target_is_directory=True)
            for name, source in [('ps5-clangxx', wrapper), ('native_ccache.py', helper)]:
                target = work / 'tools' / name
                target.write_text(source.read_text())
                target.chmod(0o755)
            source = work / 'source.cpp'
            header = work / 'value.h'
            source.write_text('#include "value.h"\nint value(){return VALUE;}\n')
            header.write_text('#define VALUE 7\n')
            environment = dict(os.environ, PS5_HIP_CCACHE='1', PS5_HIP_CCACHE_DIR=str(work / 'cache'),
                               CCACHE_SLOPPINESS='time_macros', CCACHE_PREFIX='/must/not/execute')
            counts, objects, dependencies = [], [], []
            for name in ('first', 'fresh', 'header', 'source', 'disabled'):
                output = work / name
                output.mkdir()
                if name == 'header':
                    header.write_text('#define VALUE 8\n')
                if name == 'source':
                    source.write_text('#include "value.h"\nint value(){return VALUE+1;}\n')
                if name == 'disabled':
                    environment['PS5_HIP_CCACHE'] = '0'
                compiled = subprocess.run([str(work / 'tools/ps5-clangxx'), '-x', language, '-O0', '-fPIC', '-I' + str(output),
                    '-MD', '-MF', str(output / 'unit.d'), '-c', str(source), '-o', str(output / 'unit.o')],
                    env=environment, check=True, capture_output=True, timeout=60)
                self.assertEqual(compiled.stderr, b'')
                clean = {key: value for key, value in os.environ.items() if not key.startswith('CCACHE_')}
                stats_text = subprocess.check_output(['ccache', '--dir', str(work / 'cache'),
                    '--print-stats'], env=clean, text=True, timeout=10)
                stats = {key: int(value) for key, value in (line.split() for line in stats_text.splitlines())}
                self.assertEqual(stats.get('preprocessor_error', 0), 0)
                counts.append((stats['cache_miss'], stats['preprocessed_cache_hit']))
                objects.append(c.sha(output / 'unit.o'))
                dependencies.append((output / 'unit.d').read_text().split(':', 1)[1])
            self.assertEqual(counts, [(1, 0), (1, 1), (2, 1), (3, 1), (3, 1)])
            self.assertEqual(objects[0], objects[1])
            self.assertEqual(objects[3], objects[4])
            self.assertEqual(dependencies[0], dependencies[1])
            self.assertTrue(all(str(header) in row for row in dependencies))


if __name__ == '__main__':
    unittest.main()
