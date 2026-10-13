#!/usr/bin/env python3
import pathlib,sys,argparse,json,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from native_vector_title import validate_reduction_compiler,profile_for
from build_native_vector_title import patch_startup,audit_startup,compiler_source_inputs,compiler_object
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=pathlib.Path,required=True);args,rest=p.parse_known_args();ref=args.reference_root.resolve();cp=ROOT/'results/companion-compiler-v2'
class ReductionTitleTests(unittest.TestCase):
 def test_compiler_and_sources(self):
  r=validate_reduction_compiler((cp/'receipt.json').read_bytes(),(cp/'ps5/combined.o').read_bytes(),ROOT)
  self.assertEqual(len(compiler_source_inputs(r,ROOT)),len(r['sources']));self.assertEqual(compiler_object(cp,'hip_reduction'),cp/'ps5/combined.o')
 def test_mutated_authority_rejected(self):
  raw=(cp/'receipt.json').read_bytes();obj=(cp/'ps5/combined.o').read_bytes()
  with self.assertRaises(ValueError):validate_reduction_compiler(raw+b' ',obj,ROOT)
  with self.assertRaises(ValueError):validate_reduction_compiler(raw,obj[:-1]+bytes([obj[-1]^1]),ROOT)
  r=json.loads(raw);r['inputs']={}
  with self.assertRaises(ValueError):validate_reduction_compiler(json.dumps(r).encode(),obj,ROOT)
 def test_startup_redirect(self):
  lock=json.loads((ROOT/'dependencies.json').read_text());raw=(ref/'.deps'/('native-tools-'+lock['native_tools']['revision'])/'app_crt.cpp').read_text();patched=patch_startup(raw,'hip_reduction')
  self.assertNotIn('__asm__("main")',patched);self.assertEqual(patched.count('__asm__("ps5hip_reduction_title_main")'),1)
  with self.assertRaises(RuntimeError):patch_startup(raw.replace('__asm__("main")','__asm__("wrong")'),'hip_reduction')
 def test_linked_startup_oracle(self):
  text='''00000000 <_start>:
 call <ps5hip_install_shell_exit_probe>
 call <ps5hip_install_vector_title_finalizer>
 call <_init>
 call <ps5hip_reduction_title_main>
00000100 <__hip_module_ctor>:
 call <__hipRegisterFunction>
 call <atexit>
00000200 <__hip_module_dtor>:
 call <__hipUnregisterFatBinary>
00000300 <ps5hip_install_vector_title_finalizer>:
 call <atexit>
00000400 <ps5hip_reduction_title_main>:
 call <ps5hip_reduction_companion_main>
00000500 <ps5hip_reduction_companion_main>:
 ret
'''
  result=audit_startup(text,kernel='hip_reduction');self.assertEqual(result['install_order'],['shell','vector_finalizer','_init','reduction_title'])
  with self.assertRaises(RuntimeError):audit_startup(text.replace('call <ps5hip_reduction_title_main>','call <main>'),kernel='hip_reduction')
  with self.assertRaises(RuntimeError):audit_startup(text.replace('call <ps5hip_reduction_companion_main>','call <main>'),kernel='hip_reduction')
if __name__=='__main__':unittest.main(argv=[sys.argv[0],*rest])
