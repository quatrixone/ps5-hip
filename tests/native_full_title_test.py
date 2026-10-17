"""Exercise production phase parsing, model-full asset inventory and compile flags."""
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import build_public_hip as builder


class FullTitle(unittest.TestCase):
    def test_phase_reader_exact_bytes(self):
        text = (ROOT/'tools/public_hip_llm_client_console_title.cpp').read_text()
        reader = text[text.index('enum class ClientPhase'):text.index('static std::string configured_model_path')]
        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            cpp, binary = folder/'reader.cpp', folder/'reader'
            cpp.write_text('#include <fstream>\n#include <string>\n#include <cstring>\n#include <stdexcept>\n'
                           'static const char* title_root;\n'
                           '[[noreturn]] void ps5hip_clr_gpu_fail(const char* p){throw std::runtime_error(p);}\n'+reader+
                           '\nint main(int,char** argv){title_root=argv[1];try{return int(configured_model_phase());}catch(...){return 99;}}\n')
            subprocess.run(['c++', '-std=c++17', str(cpp), '-o', str(binary)], check=True)
            cases = [(b'vocab\n',0),(b'model-gate\n',1),(b'model-prefix\n',2),(b'model-layer\n',3),(b'model-full\n',4),
                     (b'model-full',99),(b'model-full\r\n',99),(b'model-full\nextra',99),(b'model-full\x00\n',99)]
            for data, expected in cases:
                (folder/'client-phase.txt').write_bytes(data)
                with self.subTest(data=data):
                    self.assertEqual(subprocess.run([str(binary),str(folder)+'/']).returncode,expected)

    def test_full_assets_require_external_path_and_exact_six_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = pathlib.Path(folder)
            assets = {}
            for name, data in [('vocab.gguf',b'vocab'),('client-phase.txt',b'model-full\n'),
                               ('model-path.txt',b'/data/hip/models/tinyllama-q4_0.gguf')]:
                path=folder/name;path.write_bytes(data);assets[name]=path
            for name in ('getrows','normalized','weighted','q8','q4','layer'):
                path=folder/(name+'.hsaco');path.write_bytes(b'fixture');assets['model-layer/'+path.name]=path
            builder.validate_client_assets(assets)
            for name in ('model-path.txt','model-layer/layer.hsaco'):
                with self.subTest(missing=name),self.assertRaises(ValueError):
                    builder.validate_client_assets({k:v for k,v in assets.items() if k!=name})
            with self.assertRaises(ValueError):
                builder.validate_client_assets({**assets,'model.gguf':assets['vocab.gguf']})

    def test_macro_is_scoped_to_full_title(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=pathlib.Path(folder);source=folder/'source';hip=source/'hipamd/src';hip.mkdir(parents=True)
            for i in range(30):(hip/(str(i)+'.cpp')).write_text('')
            for phase in ('vocab','model-gate','model-prefix','model-layer','model-full'):
                calls=[];out=folder/phase;out.mkdir()
                def compile_unit(output,path,obj,flags,label,source_root=None):
                    calls.append((path.name,flags));return {'object':str(obj.relative_to(output))}
                with patch.object(builder,'compiler_flags',return_value=[]),patch.object(builder,'compile_bound_unit',side_effect=compile_unit):
                    self.assertEqual(len(builder.compile_units(out,source,'llm_client',client_phase=phase)),76)
                marked=[name for name,flags in calls if '-DPS5HIP_LLM_AUTHENTICATED_MODEL_FULL=1' in flags]
                self.assertEqual(marked,['public_hip_llm_client_console_title.cpp'] if phase=='model-full' else [])
                if phase=='model-full':
                    flags=next(flags for name,flags in calls if name=='public_hip_llm_client_console_title.cpp')
                    self.assertIn('-DPS5HIP_LLM_AUTHENTICATED_MODEL_GATE=1',flags)
                    self.assertNotIn('-DPS5HIP_LLM_AUTHENTICATED_MODEL_LAYER=1',flags)


if __name__ == '__main__':
    unittest.main()
