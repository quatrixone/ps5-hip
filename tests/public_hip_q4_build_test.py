"""Q4 native build routing and real package inventory boundaries; no GPU execution."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_public_hip as b

class Q4Build(unittest.TestCase):
    def test_argument_pairs(self):
        for args in (["--fixture","q4"], ["--fixture","q4","--q4-fixture","fixture"],
                     ["--q4-reference","reference"], ["--q4-fixture","fixture","--q4-reference","reference"]):
            with patch.object(sys,'argv',['build',*args]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:b.main()
                self.assertEqual(error.exception.code,2)
    def test_required_native_symbols(self):
        names=['hipInit','hipGetDeviceCount','ps5hip_clr_admit_device','hipMalloc','hipFree','hipMemcpy','hipModuleLoad','hipModuleGetFunction','hipModuleUnload','hipModuleLaunchKernel','hipDeviceSynchronize']
        with tempfile.TemporaryDirectory() as tmp,patch.object(b,'run'):
            for absent in [None,*names[3:]]:
                symbols='\n'.join('DEF '+n for n in names if n!=absent)
                with patch.object(b.subprocess,'check_output',return_value=symbols):
                    if absent:
                        with self.assertRaisesRegex(RuntimeError,absent):b.link_title(Path(tmp),[],'q4')
                    else:b.link_title(Path(tmp),[],'q4')
    def test_package_assets_and_stale_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);deps=root/'deps';out=root/'out';out.mkdir()
            def save(name,data):
                f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(data);return f
            save('title.json',b'{"title_id":"PPSA99014"}')
            save('dependencies.json',b'{"ufs2tool":{"revision":"test"}}')
            save('assets/param.json',b'{"titleId":"PPSA99014"}');save('assets/icon0.png',b'icon')
            save('deps/elevation/lapy/build/owned_root_helper-PPSA99014/lapy.elf',b'helper')
            image=save('image',b'image');libc=save('libc',b'libc');asset=save('asset',b'quantized')
            def run(output,label,args):
                if label=='makefs':
                    data=bytearray(0x10560);data[0x1055c:]=bytes.fromhex('19015419');Path(args[-2]).write_bytes(data)
                elif label=='package-extract':shutil.copytree(out/'package/app',args[-1])
            with patch.object(b,'ROOT',root),patch.object(b,'DEPS',deps),patch.object(b,'run',side_effect=run):
                for name in ['../escape','q4/../escape','other/file','q4/dir\\escape']:
                    with self.assertRaises(ValueError):b.package_title(out,image,libc,'/user/results',extra_assets={name:asset})
                package,files=b.package_title(out,image,libc,'/user/results',extra_assets={'q4/case/q8.bin':asset})
                self.assertTrue(package.is_file());self.assertEqual(files['q4/case/q8.bin'],b.sha(asset))
                (out/'package/app/q4/case/q8.bin').write_bytes(b'corrupted')
                with self.assertRaisesRegex(RuntimeError,'stage differs'):
                    b.package_title(out,image,libc,'/user/results',extra_assets={'q4/case/q8.bin':asset})

if __name__=='__main__':unittest.main()
