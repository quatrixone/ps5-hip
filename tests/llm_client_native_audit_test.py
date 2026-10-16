import sys,unittest,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_public_hip_llm_client_native import (bounded_body,symbol_calls,WRAPPER_TEMPLATE,DEVICE,SUPPORTS,wrapper_contract,normalized_wrapper,load_layout,audit_text,lifecycle_imports,nid)

def fixture():
 base=0x1000;lines=[]
 for i,line in enumerate(WRAPPER_TEMPLATE.splitlines()):
  op,*tail=line.split(' ',1);arg=tail[0] if tail else ''
  if op.startswith('j'):arg='0x'+format(base+int(arg,16),'x')
  elif op=='call':op='callq';arg='0x5000 <target>'
  else:arg=arg.replace('RIP','0x450(%rip)')
  lines.append(f'{base+i*6:x}: {op} {arg}')
 support='0: movq %rdi, -0x8(%rsp)\n5: movq %rsi, -0x10(%rsp)\na: xorl %eax, %eax\nc: andb $0x1, %al\ne: movzbl %al, %eax\n11: retq\n'
 names=['ggml_graph_n_nodes','ggml_graph_node','ggml_backend_sched_get_tensor_backend','ggml_backend_get_device',DEVICE,DEVICE,SUPPORTS,'ggml_op_desc','ggml_backend_name','printf','ggml_backend_sched_graph_compute_async']
 return '\n'.join(lines),base,support,list(enumerate(names))
class Audit(unittest.TestCase):
 def test_actual_lifecycle_import_owners(self):
  valid={nid(name):('libSceLibcInternal','libSceLibcInternal') for name in ('atexit','_init_env','printf')}
  self.assertEqual(len(lifecycle_imports(valid)),3)
  for name in ('atexit','_init_env','printf'):
   for binding in (None,('libkernel','libSceLibcInternal'),('libSceLibcInternal','libkernel')):
    bad=dict(valid)
    if binding is None:del bad[nid(name)]
    else:bad[nid(name)]=binding
    with self.subTest(name=name,binding=binding),self.assertRaises(RuntimeError):lifecycle_imports(bad)
 def test_reviewed_wrapper(self):wrapper_contract(*fixture())
 def test_early_or_inverted_forward_rejected(self):
  body,base,support,calls=fixture()
  for bad in [body.replace('jge ','jmp ',1),body.replace('jge ','jl ',1),body.replace('$0x0, -0x1c','$0x1, -0x1c',1),body+'\n2000: callq 0x5000 <bypass>',body.replace('0x1137','0x114e',1)]:
   with self.subTest(bad=bad[-50:]),self.assertRaises(RuntimeError):wrapper_contract(bad,base,support,calls)
 def test_changed_predicate_or_calls_rejected(self):
  body,base,support,calls=fixture()
  with self.assertRaises(RuntimeError):wrapper_contract(body,base,support.replace('xorl %eax, %eax','movl $0x1, %eax'),calls)
  for bad in [calls[:-1],calls+[(999,'bypass')],[(i,'wrong' if n==SUPPORTS else n) for i,n in calls],list(reversed([(i,n) for i,n in enumerate(reversed([n for _,n in calls]))]))]:
   with self.assertRaises(RuntimeError):wrapper_contract(body,base,support,bad)
 def test_relocation_addresses_only_are_normalized(self):
  body,base,_,_=fixture();self.assertEqual(normalized_wrapper(body.replace('0x450(%rip)','-0x800(%rip)'),base),WRAPPER_TEMPLATE)
  self.assertNotEqual(normalized_wrapper(body.replace('%rsi, -0x18','%rdi, -0x18',1),base),WRAPPER_TEMPLATE)
 def test_layout_rejects_wx_and_truncation(self):
  data=bytearray(120);data[:6]=b'\x7fELF\x02\x01';struct.pack_into('<Q',data,32,64);struct.pack_into('<HH',data,54,56,1);struct.pack_into('<IIQQQQQQ',data,64,1,5,0,0,0,120,120,0x4000)
  self.assertEqual(load_layout(data)[0]['flags'],5)
  bad=bytearray(data);struct.pack_into('<I',bad,68,7)
  with self.assertRaises(RuntimeError):load_layout(bad)
  with self.assertRaises(RuntimeError):load_layout(data[:-1])
 def test_converted_text_corruption_and_flags_rejected(self):
  def read(data,flags):return lambda address,size:(data,(0,flags))
  audit_text([read(b'abc',5),read(b'abc',1),read(b'abc',1)],{10},{10:3})
  for middle in [read(b'abd',1),read(b'abc',5),read(b'abc',3)]:
   with self.assertRaises(RuntimeError):audit_text([read(b'abc',5),middle,read(b'abc',1)],{10},{10:3})
class Relax(unittest.TestCase):
 def test_exact_direct(self):
  self.assertEqual(normalized_wrapper('1000: addr32 callq 0x1234 <ggml_graph_node>',0x1000),'call\n')
  self.assertEqual(len(symbol_calls('1000: addr32 callq 0x1234 <ggml_graph_node>','ggml_graph_node')),1)
  self.assertEqual(symbol_calls('1000: addr32 callq 0x1234 <wrong>','ggml_graph_node'),[])
 def test_symbol_extent_not_arbitrary_padding(self):
  body='1000: xorl %eax, %eax\n1002: retq\n1003: int3\n'
  self.assertEqual(bounded_body(body,0x1000,3),'1000: xorl %eax, %eax\n1002: retq\n')
  self.assertIn('int3',bounded_body(body,0x1000,4))
  self.assertIn('int3',bounded_body(body.replace('retq','int3'),0x1000,3))
  with self.assertRaises(RuntimeError):bounded_body(body,0x1000,0)
 def test_bad_prefix_or_indirect(self):
  for arg in ['callq *%rax','callq *0x1234(%rip)','movq %rax, %rbx','addr32 callq 0x1234 <name>','callq 0x1234 <name> extra']:
   with self.subTest(arg=arg),self.assertRaises(RuntimeError):normalized_wrapper('1000: addr32 '+arg,0x1000)
  for prefix in ('data16','rep','lock'):
   self.assertNotEqual(normalized_wrapper('1000: '+prefix+' callq 0x1234 <name>',0x1000),'call\n')

if __name__=='__main__':unittest.main()
