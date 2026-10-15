import sys,unittest,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_public_hip_rmsnorm_native import bound_calls,ordered_calls,symbol_calls

class NativeAudit(unittest.TestCase):
 def test_actual_import_binding(self):
  def read(address,size):
   return (b'\xff\x25'+struct.pack('<i',0x300-address-6),None)
  self.assertEqual(len(bound_calls(' callq *0x42(%rip) # 0x300\n callq 0x100 <label>\n',{0x300},read)),2)
  self.assertEqual(bound_calls(' callq *0x42(%rip) # 0x301\n',{0x300},read),[])
  self.assertEqual(bound_calls(' callq 0x100 <atexit>\n',{0x301},read),[])
 def test_lea_call_requires_same_unclobbered_register(self):
  good=' leaq 0x42(%rip), %rax # 0x50 <target>\n movl %ecx, %edx\n callq *%rax\n'
  self.assertEqual(len(symbol_calls(good,'target')),1)
  self.assertFalse(symbol_calls(good.replace('movl %ecx, %edx','movq %rcx, %rax'),'target'))
  self.assertFalse(symbol_calls(good.replace('callq *%rax','callq *%rbx'),'target'))
  for instruction in ('xorl %eax, %eax', 'incq %rax', 'popq %rax', 'movb $0, %al', 'jne 0x80 <elsewhere>'):
   with self.subTest(instruction=instruction):
    self.assertFalse(symbol_calls(good.replace('movl %ecx, %edx',instruction),'target'))
 def test_order_requires_unique_actual_calls(self):
  good=' callq 0x10 <a>\n callq 0x20 <b>\n'
  ordered_calls(good,['a','b'])
  for bad in (good.replace('callq','lea'),good+good,good.replace('<a>','<x>'),good):
   names=['b','a'] if bad==good else ['a','b']
   with self.assertRaises(Exception):ordered_calls(bad,names)

if __name__=='__main__':unittest.main()
