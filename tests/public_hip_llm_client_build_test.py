"""Native vocabulary package boundaries and mandatory scheduler wrapper linkage."""
from pathlib import Path
import shutil
import subprocess
import re
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_public_hip as b


class ClientBuild(unittest.TestCase):
    def test_asset_paths_follow_elevation_namespace(self):
        source=(b.ROOT/'tools/public_hip_llm_client_console_title.cpp').read_text()
        title=json.loads((b.ROOT/'title.json').read_text())['title_id']
        self.assertIn('title_root = "/system_ex/app/'+title+'/"',source)
        self.assertEqual(re.findall(r'"(/app0/[^"\n]+)"',source),
                         ['/app0/results-dir.txt','/app0/lapy.elf'])
        main=source[source.index('int main()'):]
        self.assertLess(main.index('"/app0/results-dir.txt"'),main.index('elevation::request'))
        self.assertLess(main.index('elevation::request'),main.index('configured_model_phase()'))
        self.assertLess(main.index('elevation::request'),main.index('configured_model_path('))
        for asset in ('client-phase.txt','vocab.gguf','model-path.txt'):
            self.assertIn('std::string(title_root)+"'+asset+'"',source)

    def test_actual_phase_parser_valid_and_failure_boundaries(self):
        source=(b.ROOT/'tools/public_hip_llm_client_console_title.cpp').read_text()
        start=source.index('static bool configured_model_phase()')
        function=source[start:source.index('\nint main()',start)]
        harness = ('#include <fstream>\n#include <string>\n#include <cstring>\n'
                   '#include <cstdio>\n#include <stdexcept>\n'
                   'static const char* title_root;\n'
                   '[[noreturn]] void ps5hip_clr_gpu_fail(const char* why){throw std::runtime_error(why); }\n'
                   +function+'\nint main(int argc,char**argv){title_root=argv[1];try{'
                   'std::printf("phase=%d\\n",configured_model_phase());return 0;}'
                   'catch(const std::exception&e){std::printf("reason=%s\\n",e.what());return 1;}}\n')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cpp=root/'parser.cpp';exe=root/'parser';cpp.write_text(harness)
            subprocess.run(['c++','-std=c++17',str(cpp),'-o',str(exe)],check=True,capture_output=True)
            cases=[('vocab',b'vocab\n',0,'phase=0'),('model',b'model-gate\n',0,'phase=1'),
                   ('missing',None,1,'reason=client-phase-open'),
                   ('empty',b'',1,'reason=client-phase-config'),
                   ('trailing',b'vocab\nx',1,'reason=client-phase-config'),
                   ('long',b'x'*16,1,'reason=client-phase-config'),
                   ('invalid',b'bad\n',1,'reason=client-phase-config'),
                   ('directory','directory',1,'reason=client-phase-read')]
            for name,data,code,message in cases:
                with self.subTest(name=name):
                    directory=root/name;directory.mkdir();phase=directory/'client-phase.txt'
                    if data=='directory':phase.mkdir()
                    elif data is not None:phase.write_bytes(data)
                    result=subprocess.run([str(exe),str(directory)+'/'],capture_output=True,text=True)
                    self.assertEqual((result.returncode,result.stdout.strip()),(code,message))

    def test_wrapper_link_and_required_symbols(self):
        names = ['hipInit', 'hipGetDeviceCount', 'ps5hip_clr_admit_device',
                 'hipMalloc', 'hipFree', 'hipMemcpy', 'hipMemset', 'hipDeviceSynchronize',
                 'llama_model_load_from_file', 'llama_tokenize',
                 '__wrap_ggml_backend_sched_graph_compute_async']
        with tempfile.TemporaryDirectory() as tmp, patch.object(b, 'run') as run:
            for absent in (None, *names[3:]):
                symbols = '\n'.join('DEF ' + n for n in names if n != absent)
                with patch.object(b.subprocess, 'check_output', return_value=symbols):
                    if absent:
                        with self.assertRaisesRegex(RuntimeError, absent):
                            b.link_title(Path(tmp), [], 'llm_client')
                    else:
                        b.link_title(Path(tmp), [], 'llm_client')
                        args = run.call_args.args[2]
                        self.assertIn('--wrap=ggml_backend_sched_graph_compute_async', args)
                        self.assertIn('--undefined=__wrap_ggml_backend_sched_graph_compute_async', args)

    def test_native_allocator_link_is_explicit_and_paired(self):
        names=['hipInit','hipGetDeviceCount','ps5hip_clr_admit_device','hipMalloc','hipFree','hipMemcpy','hipMemset','hipDeviceSynchronize','llama_model_load_from_file','llama_tokenize','__wrap_ggml_backend_sched_graph_compute_async']
        wrappers=['__wrap_ggml_aligned_malloc','__wrap_ggml_aligned_free']
        with tempfile.TemporaryDirectory() as tmp, patch.object(b,'run') as run:
            for enabled in (False,True):
                with patch.object(b.subprocess,'check_output',return_value='\n'.join('DEF '+n for n in names+wrappers)):
                    b.link_title(Path(tmp),[],'llm_client',native_host_allocator=enabled)
                args=run.call_args.args[2]
                for symbol in ('ggml_aligned_malloc','ggml_aligned_free'):
                    self.assertEqual('--wrap='+symbol in args,enabled)
                    self.assertEqual('--undefined=__wrap_'+symbol in args,enabled)
            for absent in wrappers:
                with patch.object(b.subprocess,'check_output',return_value='\n'.join('DEF '+n for n in names+wrappers if n!=absent)):
                    with self.assertRaisesRegex(RuntimeError,absent):b.link_title(Path(tmp),[],'llm_client',native_host_allocator=True)
            with self.assertRaisesRegex(ValueError,'requires llm_client'):b.link_title(Path(tmp),[],'bootstrap',native_host_allocator=True)

    def test_vocab_package_exact_inventory_and_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'out'
            out.mkdir()

            def save(name, data):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
                return p

            save('title.json', b'{"title_id":"PPSA99014"}')
            save('dependencies.json', b'{"ufs2tool":{"revision":"test"}}')
            save('assets/param.json', b'{"titleId":"PPSA99014"}')
            save('assets/icon0.png', b'icon')
            save('deps/elevation/lapy/build/owned_root_helper-PPSA99014/lapy.elf', b'helper')
            image, libc = save('image', b'image'), save('libc', b'libc')
            assets = {'vocab.gguf': save('vocab', b'vocab'),
                      'client-phase.txt': save('phase', b'vocab\n')}

            def run(output, label, args):
                if label == 'makefs':
                    data = bytearray(0x10560)
                    data[0x1055c:] = bytes.fromhex('19015419')
                    Path(args[-2]).write_bytes(data)
                elif label == 'package-extract':
                    shutil.copytree(out / 'package/app', args[-1])

            with patch.object(b, 'ROOT', root), patch.object(b, 'DEPS', root / 'deps'), patch.object(b, 'run', side_effect=run):
                for bad in ({'../escape': assets['vocab.gguf']},
                            {**assets, 'model.gguf': assets['vocab.gguf']}):
                    with self.assertRaises(ValueError):
                        b.package_title(out, image, libc, '/user/results', client_assets=bad)
                assets['client-phase.txt'].write_bytes(b'model-gate\n')
                with self.assertRaises(ValueError):
                    b.package_title(out, image, libc, '/user/results', client_assets=assets)
                assets['client-phase.txt'].write_bytes(b'vocab\n')
                _, files = b.package_title(out, image, libc, '/user/results', client_assets=assets)
                self.assertEqual(files['vocab.gguf'], b.sha(assets['vocab.gguf']))
                self.assertEqual(files['client-phase.txt'], b.sha(assets['client-phase.txt']))
                self.assertNotIn('model.gguf', files)


if __name__ == '__main__':
    unittest.main()
