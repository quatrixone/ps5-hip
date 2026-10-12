"""Tool binding and unchanged audit checks; no LLVM processes invoked."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import link_alias_audit_v1 as original
import link_alias_portable_v2 as portable


class ToolBindingTests(unittest.TestCase):
 def setUp(self):
  self.temporary=tempfile.TemporaryDirectory(prefix='alias tools with spaces ')
  self.addCleanup(self.temporary.cleanup)
  self.root=Path(self.temporary.name).resolve();self.prepared=self.root/'prepared.json'
  self.linker=self.root/'custom linker';self.objdump=self.root/'custom inspector';self.library=self.root/'shared library'
  for path in (self.linker,self.objdump,self.library):path.write_text(path.name)
  self.report={'tools':{name:{'path':str(path),'sha256':portable.sha(path),'libraries':{str(self.library):portable.sha(self.library)}} for name,path in [('ld.lld-18',self.linker),('llvm-objdump-18',self.objdump)]}}
  self.prepared.write_text(json.dumps(self.report))

 def test_explicit_paths_with_spaces(self):
  self.assertEqual(portable.verified_tools(self.prepared,self.linker,self.objdump),{'linker':self.linker,'objdump':self.objdump})

 def test_changed_tool_and_library_rejected(self):
  for path in (self.linker,self.objdump,self.library):
   with self.subTest(path=path.name):
    before=path.read_bytes();path.write_bytes(before+b'changed')
    with self.assertRaises(ValueError):portable.verified_tools(self.prepared,self.linker,self.objdump)
    path.write_bytes(before)

 def test_unbound_identical_binary_and_relative_path_rejected(self):
  other=self.root/'different linker';other.write_bytes(self.linker.read_bytes())
  for path in (other,Path(self.linker.name)):
   with self.subTest(path=path):
    with self.assertRaises(ValueError):portable.verified_tools(self.prepared,path,self.objdump)

 def test_missing_prepared_tool_rejected(self):
  del self.report['tools']['llvm-objdump-18'];self.prepared.write_text(json.dumps(self.report))
  with self.assertRaises(KeyError):portable.verified_tools(self.prepared,self.linker,self.objdump)

 def test_existing_output_refused(self):
  with patch.object(portable.subprocess,'run',side_effect=AssertionError('No tool may run')):
   with self.assertRaises(FileExistsError):portable.run(self.root,self.root,self.prepared,self.linker,self.objdump)

 def test_bound_path_reaches_argv_without_shell(self):
  source=self.root/'validation'/'negative-runs'/'baseline'/'alias-gfx1013';source.mkdir(parents=True)
  (source/'output.o').write_bytes(b'object')
  (source.parent/'result.json').write_text(json.dumps({'cases':[{'name':'alias-gfx1013','files':{'output.o':portable.sha(source/'output.o')}}]}))
  with patch.object(portable.subprocess,'run',side_effect=RuntimeError('stop before execution')) as process:
   with self.assertRaisesRegex(RuntimeError,'stop before execution'):
    portable.run(self.root/'validation',self.root/'audit',self.prepared,self.linker,self.objdump)
   self.assertEqual(process.call_args.args[0],[str(self.linker),'-flavor','gnu','-shared','--no-undefined',str(source/'output.o'),'-o',str(self.root/'audit/gfx1013/baseline/linked.hsaco')])
   self.assertFalse(process.call_args.kwargs.get('shell',False))

 def test_parser_and_complete_audit_loop_unchanged(self):
  self.assertIs(portable.parse,original.parse)
  self.assertIs(portable.positive_equal,original.positive_equal)
  def audit_loop(module):
   tree=ast.parse(Path(module.__file__).read_text())
   run=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run')
   return ast.dump(next(node for node in run.body if isinstance(node,ast.For)),include_attributes=False)
  self.assertEqual(audit_loop(portable),audit_loop(original))

 def test_cli_requires_explicit_tools(self):
  args=[sys.executable,portable.__file__,'--validation',str(self.root),'--output',str(self.root/'new'),'--prepared',str(self.prepared)]
  result=subprocess.run(args,capture_output=True,text=True)
  self.assertEqual(result.returncode,2);self.assertIn('--linker',result.stderr);self.assertIn('--objdump',result.stderr)
  self.assertFalse((self.root/'new').exists())


if __name__=='__main__':unittest.main()
