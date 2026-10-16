"""Native vocabulary package boundaries and mandatory scheduler wrapper linkage."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_public_hip as b


class ClientBuild(unittest.TestCase):
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
