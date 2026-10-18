"""CPU launcher policy and actual producer control flow, without compilation."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_llm_cpu_reference as b

class CPUCache(unittest.TestCase):
    def test_disabled_missing_and_required(self):
        with patch.object(b.shutil, 'which', return_value=None):
            for mode in ('0', 'auto'):
                record, prefix = b.compiler_cache({'PS5_HIP_CCACHE': mode})
                self.assertEqual(record, {'enabled': False, 'requested': mode})
                self.assertEqual(prefix, [])
                self.assertEqual(b.cache_flags(prefix), ['-DCMAKE_C_COMPILER_LAUNCHER=', '-DCMAKE_CXX_COMPILER_LAUNCHER='])
            for mode in ('1', 'invalid'):
                with self.assertRaises(ValueError): b.compiler_cache({'PS5_HIP_CCACHE': mode})

    def test_enabled_policy_identity_and_delimiters(self):
        with patch.object(b.shutil, 'which', return_value='/usr/bin/ccache'), patch.object(b, 'sha', return_value='digest'), patch.object(b, 'shared_libraries', return_value=[Path('/lib/cache.so')]):
            record, prefix = b.compiler_cache({'PS5_HIP_CCACHE': '1', 'PS5_HIP_CCACHE_DIR': '/cache with spaces'})
            self.assertEqual(record['tool_sha256'], 'digest')
            self.assertEqual(record['inputs'], {'/lib/cache.so': 'digest'})
            for option in ('max_size=1GiB', 'compiler_check=content', 'direct_mode=false', 'sloppiness=', 'remote_storage=', 'namespace=ps5-hip-cpu-v1'):
                self.assertIn(option, prefix)
            self.assertTrue(all(flag.endswith(';'.join(prefix)) for flag in b.cache_flags(prefix)))
            for path in ('/bad;path', '/bad\npath', '/bad\rpath'):
                with self.assertRaises(ValueError): b.compiler_cache({'PS5_HIP_CCACHE_DIR': path})
            with patch.object(b, 'sha', return_value='changed'):
                self.assertNotEqual(record, b.compiler_cache({'PS5_HIP_CCACHE': '1', 'PS5_HIP_CCACHE_DIR': '/cache with spaces'})[0])

    def producer(self, stale=False, final_change=False):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp);client=root/'client';out=root/'build/cpu'
            for name in ('ggml/src/ggml-cpu/CMakeLists.txt','src/llama-model-loader.cpp'):
                p=client/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source')
            if stale:
                out.mkdir(parents=True);(out/'configuration.json').write_text('{}')
            record={'enabled':False,'requested':'0'};calls=[]
            def run(argv, **kwargs):
                calls.append((argv,kwargs['env']))
                (out/'cmake/bin').mkdir(parents=True,exist_ok=True)
                if 'test-quantize-fns' in str(argv[0]):kwargs['stdout'].write('0 tests failed\n')
                import subprocess
                return subprocess.CompletedProcess(argv,0)
            for name,value in [('ROOT',root),('build_jobs',lambda:3),('read',lambda _: {'client':{'revision':'rev','source_url':'url'}}),('source',lambda *a:client),('sha',lambda _: 'digest'),('git',lambda *a:'rev')]:stack.enter_context(patch.object(b,name,value))
            stack.enter_context(patch.object(b.shutil,'which',side_effect=lambda name:'/tools/'+name))
            stack.enter_context(patch.object(b.subprocess,'check_output',return_value='version 18.1.8\n'))
            stack.enter_context(patch.object(b.subprocess,'run',side_effect=run))
            stack.enter_context(patch.object(b.loader,'render_user',return_value='rendered',create=True))
            stack.enter_context(patch.object(b,'compiler_cache',side_effect=[(record,[]),({'enabled':False,'requested':'auto'} if final_change else record,[])]))
            stack.enter_context(patch.object(sys,'argv',['builder','--output',str(out)]))
            stack.enter_context(patch.dict(os.environ,{'CCACHE_SLOPPINESS':'time_macros','CCACHE_PREFIX':'bad','PS5_HIP_CCACHE':'0'}))
            if stale or final_change:
                with self.assertRaisesRegex(RuntimeError,'reference inputs changed' if stale else 'cache identity changed'):b.main()
                self.assertFalse((out/'receipt.json').exists())
                if stale:self.assertEqual(calls,[])
            else:
                b.main();receipt=json.loads((out/'receipt.json').read_text());self.assertEqual(receipt['identity']['compiler_cache'],record)
                self.assertIn('tests/llm_cpu_ccache_test.py',receipt['identity']['inputs'])
                self.assertTrue(all(env.get('CCACHE_CONFIGPATH')=='/dev/null' and 'CCACHE_PREFIX' not in env and 'CCACHE_SLOPPINESS' not in env for _,env in calls))
                self.assertIn('-DCMAKE_C_COMPILER_LAUNCHER=',calls[0][0]);self.assertIn('-DCMAKE_CXX_COMPILER_LAUNCHER=',calls[0][0])
                self.assertIn('3',calls[1][0])
    def test_main_environment_flags_receipt(self):self.producer()
    def test_main_rejects_stale_configuration_before_build(self):self.producer(stale=True)
    def test_main_rejects_cache_change_before_receipt(self):self.producer(final_change=True)

if __name__=='__main__':unittest.main()
