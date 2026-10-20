"""Qualify the dense normal-control-flow audit on synthetic disassembly only.

No image, console or numerical evidence is involved: these bodies are written
here to check that the ordering proof accepts the intended shape and rejects
bypasses, unresolved branches, dead calls and repeated calls.
"""
import sys,unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_public_hip_llm_client_native as auditor
from audit_public_hip_llm_client_native import dense_dominates,dense_order,symbol_calls

FATAL='ps5hip_clr_gpu_fail'

def body(rows,base=0x1000,stride=6):
 """Render one function body in the inspected llvm-objdump-18 shape."""
 text=[]
 for index,(op,arg) in enumerate(rows):
  if op in ('je','jne','jmp'):arg='0x'+format(base+int(arg)*stride,'x')+' <m0+0x'+format(int(arg)*stride,'x')+'>'
  elif op=='callq':arg='0x5000 <'+arg+'>'
  text.append(f'{base+index*stride:x}: {op} {arg}'.rstrip())
 return '\n'.join(text)+'\n'

def only(text,name):
 found=symbol_calls(text,name);assert len(found)==1,name;return found

class DenseOrdering(unittest.TestCase):
 def test_linear_order_accepted(self):
  text=body([('callq','open'),('callq','run'),('callq','finish'),('retq','')])
  dense_order(text,['open','run','finish'])
  dense_dominates(text,only(text,'open'),only(text,'finish'))

 def test_conditional_bypass_rejected(self):
  # The branch reaches the later call without the earlier one.
  text=body([('je','3'),('callq','open'),('nop',''),('callq','finish'),('retq','')])
  with self.assertRaisesRegex(RuntimeError,'dense required call bypass'):
   dense_dominates(text,only(text,'open'),only(text,'finish'))

 def test_loop_back_edge_still_proves_order(self):
  text=body([('callq','open'),('callq','finish'),('je','0'),('retq','')])
  dense_dominates(text,only(text,'open'),only(text,'finish'))

 def test_unresolved_or_indirect_branch_rejected(self):
  for row in (('jmpq','*%rax'),('je','0x7fff <elsewhere>')):
   text=body([('callq','open')])+f'1100: {row[0]} {row[1]}\n'+body([('callq','finish'),('retq','')],base=0x1106)
   with self.assertRaisesRegex(RuntimeError,'dense unresolved branch'):
    dense_dominates(text,only(text,'open'),only(text,'finish'))

 def test_unreachable_required_call_rejected(self):
  # The later call sits behind an unconditional return with no edge to it.
  text=body([('callq','open'),('retq',''),('callq','finish'),('retq','')])
  with self.assertRaisesRegex(RuntimeError,'dense required call unreachable'):
   dense_dominates(text,only(text,'open'),only(text,'finish'))

 def test_only_declared_fatal_calls_terminate_a_path(self):
  rows=lambda name:[('je','4'),('callq','open'),('callq','finish'),('retq',''),('callq',name),('jmp','2')]
  text=body(rows(FATAL))
  dense_dominates(text,only(text,'open'),only(text,'finish'))
  ordinary=body(rows('ggml_free'))
  with self.assertRaisesRegex(RuntimeError,'dense required call bypass'):
   dense_dominates(ordinary,only(ordinary,'open'),only(ordinary,'finish'))

 def test_repeated_missing_or_empty_rejected(self):
  repeated=body([('callq','open'),('callq','open'),('callq','finish'),('retq','')])
  with self.assertRaisesRegex(RuntimeError,'dense missing/repeated call: open'):
   dense_order(repeated,['open','finish'])
  with self.assertRaisesRegex(RuntimeError,'dense missing/repeated call: absent'):
   dense_order(body([('callq','open'),('retq','')]),['open','absent'])
  with self.assertRaisesRegex(RuntimeError,'dense empty CFG'):
   dense_dominates('',[0],[1])

 def test_duplicate_instruction_address_rejected(self):
  text=body([('callq','open'),('callq','finish'),('retq','')])
  duplicated=text+'1000: retq\n'
  with self.assertRaisesRegex(RuntimeError,'dense duplicate instruction address'):
   dense_dominates(duplicated,only(text,'open'),only(text,'finish'))

 def test_call_position_must_be_an_instruction(self):
  text=body([('callq','open'),('retq','')])
  with self.assertRaisesRegex(RuntimeError,'dense call position is not an instruction'):
   dense_dominates(text,[len(text)+8],only(text,'open'))

class DenseRouting(unittest.TestCase):
 def test_phase_routes_to_dense_audit_with_checked_sources(self):
  with patch.object(auditor.shutil,'which',return_value=sys.executable),\
       patch.object(auditor,'symbol_values',side_effect=RuntimeError('reached image inspection')):
   with self.assertRaisesRegex(RuntimeError,'reached image inspection'):
    auditor.audit(__file__,__file__,__file__,'dense-layer')
   for profile,fmt in (('generation','gguf'),('smoke','original_safetensors')):
    with self.assertRaisesRegex(RuntimeError,'dense mixed audit profile'):
     auditor.audit(__file__,__file__,__file__,'dense-layer',profile,fmt)

 def test_dense_source_identities_exist(self):
  root=auditor.ROOT
  for name in ('tests/llm_dense_native_audit_test.py','tests/native_dense_title_test.py',
               'ports/ggml/native_dense_layer.h','ports/ggml/dense_layer_qualification.h',
               'ports/ggml/dense_layer_fixture.h','ports/ggml/dense_layer_cases.def'):
   self.assertTrue((root/name).is_file(),name)

if __name__=='__main__':unittest.main()
