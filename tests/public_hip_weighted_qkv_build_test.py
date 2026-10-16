"""Weighted QKV native build routing and real package inventory boundaries; no GPU execution."""
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

class WeightedQkvBuild(unittest.TestCase):
    def test_argument_pairs(self):
        complete=["--fixture","weighted_qkv","--weighted-rmsnorm-fixture","weighted","--q4-fixture","halfword","--weighted-qkv-reference","reference"]
        for index in (2,4,6):
            args=complete[:index]+complete[index+2:]
            with patch.object(sys,'argv',['build',*args]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:b.main()
                self.assertEqual(error.exception.code,2)
        for args in (["--weighted-rmsnorm-fixture","weighted"],["--weighted-qkv-reference","reference"],complete+["--q4-reference","other"]):
            with patch.object(sys,'argv',['build',*args]),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:b.main()
                self.assertEqual(error.exception.code,2)
        with patch.object(sys,'argv',['build',*complete]),patch.object(b,'weighted_qkv_assets',side_effect=ValueError('authenticated helper reached')):
            with self.assertRaisesRegex(ValueError,'authenticated helper reached'):b.main()
    def test_required_native_symbols(self):
        names=['hipInit','hipGetDeviceCount','ps5hip_clr_admit_device','hipMalloc','hipFree','hipMemcpy','hipModuleLoad','hipModuleGetFunction','hipModuleUnload','hipModuleLaunchKernel','hipDeviceSynchronize']
        with tempfile.TemporaryDirectory() as tmp,patch.object(b,'run'):
            for absent in [None,*names[3:]]:
                symbols='\n'.join('DEF '+n for n in names if n!=absent)
                with patch.object(b.subprocess,'check_output',return_value=symbols):
                    if absent:
                        with self.assertRaisesRegex(RuntimeError,absent):b.link_title(Path(tmp),[],'weighted_qkv')
                    else:b.link_title(Path(tmp),[],'weighted_qkv')
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
                for name in ['../escape','weighted-qkv/../escape','other/file','weighted-qkv/dir\\escape']:
                    with self.assertRaises(ValueError):b.package_title(out,image,libc,'/user/results',extra_assets={name:asset})
                package,files=b.package_title(out,image,libc,'/user/results',extra_assets={'weighted-qkv/q8.hsaco':asset})
                self.assertTrue(package.is_file());self.assertEqual(files['weighted-qkv/q8.hsaco'],b.sha(asset))
                (out/'package/app/weighted-qkv/q8.hsaco').write_bytes(b'corrupted')
                with self.assertRaisesRegex(RuntimeError,'stage differs'):
                    b.package_title(out,image,libc,'/user/results',extra_assets={'weighted-qkv/q8.hsaco':asset})

    def test_audit_exact_images_sources_and_cache_identity(self):
        from audit_public_hip_weighted_qkv_native import SOURCES
        with tempfile.TemporaryDirectory() as tmp,patch.object(b,'ROOT',Path(tmp)):
            root=Path(tmp);out=root/'build';out.mkdir()
            paths=[out/n for n in ('title.elf','eboot.elf','extracted.elf')]+[root/n for n in SOURCES]
            for p in paths:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(p.name.encode())
            report=dict(status='PASS',classification='offline-public-hip-weighted-qkv-native-image-audit',native_executed=False,audited_function_count=21,inputs={str(p):b.sha(p) for p in paths})
            audit=out/'weighted-qkv-native-audit.json';audit.write_text(json.dumps(report));digest=b.validate_weighted_qkv_audit(out);self.assertEqual(digest,b.sha(audit))
            for key,value in [('status','FAIL'),('native_executed',True),('audited_function_count',18),('inputs',{})]:
                bad=dict(report);bad[key]=value;audit.write_text(json.dumps(bad))
                with self.assertRaises(RuntimeError):b.validate_weighted_qkv_audit(out)
            audit.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'identity changed'):b.validate_weighted_qkv_audit(out,'0'*64)
            paths[1].write_bytes(b'converted image mutation')
            with self.assertRaises(RuntimeError):b.validate_weighted_qkv_audit(out)

    def test_new_sources_bound(self):
        inputs=b.build_inputs()
        for name in ('tools/public_hip_weighted_qkv_console_title.cpp','tools/public_hip_weighted_qkv_assets.py','tools/audit_public_hip_weighted_qkv_native.py','tools/verify_weighted_qkv_readbacks.py','tests/public_hip_weighted_qkv_build_test.py'):
            self.assertEqual(inputs[name],b.sha(b.ROOT/name))

if __name__=='__main__':unittest.main()
