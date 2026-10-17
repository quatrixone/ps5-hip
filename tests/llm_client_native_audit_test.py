import sys,unittest,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_public_hip_llm_client_native import (WRAPPER_TEMPLATE,DEVICE,SUPPORTS,wrapper_contract,normalized_wrapper,load_layout,audit_text,lifecycle_imports,nid,phase_contract)

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
 def test_production_audit_input_paths(self):
  from unittest.mock import patch
  import audit_public_hip_llm_client_native as auditor
  # Reach image inspection using only checked-in source identities, with no staged test file.
  with patch.object(auditor.shutil,'which',return_value=sys.executable),patch.object(auditor,'symbol_values',side_effect=RuntimeError('reached image inspection')):
   with self.assertRaisesRegex(RuntimeError,'reached image inspection'):
    auditor.audit(__file__,__file__,__file__,'model-gate')
  self.assertIn('tests/llm_client_native_audit_test.py',auditor.SOURCES)
  self.assertIn('tools/audit_public_hip_llm_client_native.py',auditor.SOURCES)
  self.assertIn('tools/prepare_llm_native_scheduler.py',auditor.ALLOCATOR_SOURCES)
  self.assertTrue(all((auditor.ROOT/p).is_file() for p in auditor.SOURCES))
 def test_storage_snapshot_symbols_are_optional_only_in_model_phase(self):
  import ast
  tree=ast.parse((a.ROOT/'tools/audit_public_hip_llm_client_native.py').read_text())
  function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
  def assignment(node,name):
   return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
  start=next(i for i,n in enumerate(function.body) if assignment(n,'snapshot_names'))
  end=next(i for i,n in enumerate(function.body) if assignment(n,'model_names'))
  code=compile(ast.Module(body=function.body[start:end],type_ignores=[]),'<actual-snapshot-selection>','exec')
  for phase in ('vocab','model-gate'):
   for present in (False,True):
    values={'native_storage_call_snapshot':1,'native_hip_call_statistics':2} if present else {}
    env=dict(values=values,phase=phase,names=[],require=a.require,match=lambda part:next(n for n in values if part in n))
    if phase=='vocab' and not present:
     with self.assertRaisesRegex(RuntimeError,'snapshot missing'):exec(code,env)
    else:
     exec(code,env)
     self.assertEqual(len(env['names']),2 if present else 0)
 def test_model_phase_chain(self):
  chain=['main','model','llama_process','decode','process_ubatch','context', '__wrap_ggml_backend_sched_graph_compute_async']
  bodies={a:f'10: callq 0x20 <{b}>\n' for a,b in zip(chain,chain[1:])};bodies[chain[-1]]=''
  phase_contract('model-gate','model',chain,bodies)
  phase_contract('vocab',None,[],{})
  for phase,gate in [('invalid','model'),('vocab','model'),('model-gate',None)]:
   with self.assertRaises(RuntimeError):phase_contract(phase,gate,chain,bodies)
  for caller in chain[:-1]:
   bad=dict(bodies);bad[caller]=''
   with self.assertRaises(RuntimeError):phase_contract('model-gate','model',chain,bad)
  bad=dict(bodies);bad['context']+='20: callq 0x40 <ggml_backend_sched_graph_compute_async>\n'
  with self.assertRaises(RuntimeError):phase_contract('model-gate','model',chain,bad)
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
  for bad in [body.replace('jge ','jmp ',1),body.replace('jge ','jl ',1),body.replace('$0x0, -0x1c','$0x1, -0x1c',1),body+'\n2000: callq 0x5000 <bypass>',body.replace('0x1142','0x1159',1)]:
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
import audit_public_hip_llm_client_native as a
class Allocator(unittest.TestCase):
 def fixture(self):
  roles=dict(buffer_alloc='cpu_alloc',buffer_free='cpu_free',finalized='finalized',stats='stats',
   constructors=['ctor1','ctor2'],destructors=['dtor1','dtor2'],close='close',cpu='cpu',malloc_address=0x800)
  edges={'cpu_alloc':['__wrap_ggml_aligned_malloc'],'cpu_free':['__wrap_ggml_aligned_free'],
   'ggml_init':['__wrap_ggml_aligned_malloc'],'ggml_free':['__wrap_ggml_aligned_free'],
   'ggml_backend_sched_new':['__wrap_ggml_aligned_malloc'],
   'ggml_backend_sched_free':['__wrap_ggml_aligned_free'],
   'ggml_threadpool_new_impl':['__wrap_ggml_aligned_malloc'],
   'ggml_threadpool_free':['__wrap_ggml_aligned_free','__wrap_ggml_aligned_free'],
   'main':['stats'],'finalized':['stats'],
   '__wrap_ggml_aligned_malloc':['ctor2','cpu','ps5hip_llm_fail'],
   '__wrap_ggml_aligned_free':['cpu','close','dtor2','ps5hip_llm_fail']}
  bodies={caller:'\n'.join(f'{16+i*8:x}: addr32 callq 0x800 <{callee}>' for i,callee in enumerate(callees)) for caller,callees in edges.items()}
  bodies['ggml_threadpool_new_impl']+='\n100: leaq 0x6f9(%rip), %rax\n107: movl $0x100, %edi\n10c: callq *%rax'
  return bodies,roles
 def test_paired_external_and_original_internal(self):
  a.allocator_contract(*self.fixture())
 def test_missing_ownership_or_observation_edge(self):
  bodies,roles=self.fixture()
  for caller,body in bodies.items():
   for line in body.splitlines():
    bad=dict(bodies);bad[caller]=body.replace(line,'')
    with self.subTest(caller=caller,line=line),self.assertRaises(RuntimeError):a.allocator_contract(bad,roles)
 def test_mixed_allocator_pair_rejected(self):
  bodies,roles=self.fixture()
  for caller,target in [('cpu_alloc','ggml_aligned_malloc'),('cpu_free','ggml_aligned_free'),
   ('ggml_init','ggml_aligned_malloc'),('ggml_free','ggml_aligned_free'),
   ('ggml_backend_sched_new','ggml_aligned_malloc'),('ggml_backend_sched_free','ggml_aligned_free'),
   ('__wrap_ggml_aligned_malloc','ggml_aligned_malloc'),('__wrap_ggml_aligned_free','ggml_aligned_free')]:
   bad=dict(bodies);bad[caller]+=f'\n90: callq 0x800 <{target}>'
   with self.subTest(caller=caller),self.assertRaises(RuntimeError):a.allocator_contract(bad,roles)
 def test_indirect_target_and_register_flow_rejected(self):
  bodies,roles=self.fixture();original=bodies['ggml_threadpool_new_impl']
  for bad in [original.replace('0x6f9','0x6f8'),original.replace('%rax','%rcx'),
   original.replace('$0x100','$0x200'),original.replace('10c: callq','109: xorq %rax, %rax\n10c: callq'),
   original.replace('leaq','movq'),original+'\n200: callq *%rax']:
   altered=dict(bodies);altered['ggml_threadpool_new_impl']=bad
   with self.subTest(body=bad),self.assertRaises(RuntimeError):a.allocator_contract(altered,roles)
 def test_duplicate_routing_rejected(self):
  bodies,roles=self.fixture();bodies['cpu_alloc']+='\n90: callq 0x800 <__wrap_ggml_aligned_malloc>'
  with self.assertRaises(RuntimeError):a.allocator_contract(bodies,roles)
 def test_scheduler_pair_cannot_revert_to_libc_or_duplicate(self):
  bodies,roles=self.fixture()
  for caller,wrapped,libc in [('ggml_backend_sched_new','__wrap_ggml_aligned_malloc','malloc'),
                             ('ggml_backend_sched_free','__wrap_ggml_aligned_free','free')]:
   for altered in [bodies[caller].replace(wrapped,libc),bodies[caller]+'\n90: callq 0x800 <'+wrapped+'>']:
    bad=dict(bodies);bad[caller]=altered
    with self.subTest(caller=caller,body=altered),self.assertRaises(RuntimeError):a.allocator_contract(bad,roles)
class Relax(unittest.TestCase):
 def test_exact_direct(self):
  self.assertEqual(a.normalized_wrapper('1000: addr32 callq 0x1234 <ggml_graph_node>',0x1000),'call\n')
  self.assertEqual(len(a.symbol_calls('1000: addr32 callq 0x1234 <ggml_graph_node>','ggml_graph_node')),1)
  self.assertEqual(a.symbol_calls('1000: addr32 callq 0x1234 <wrong>','ggml_graph_node'),[])
 def test_symbol_extent_not_arbitrary_padding(self):
  body='1000: xorl %eax, %eax\n1002: retq\n1003: int3\n'
  self.assertEqual(a.bounded_body(body,0x1000,3),'1000: xorl %eax, %eax\n1002: retq\n')
  self.assertIn('int3',a.bounded_body(body,0x1000,4))
  self.assertIn('int3',a.bounded_body(body.replace('retq','int3'),0x1000,3))
  with self.assertRaises(RuntimeError):a.bounded_body(body,0x1000,0)
 def test_bad_prefix_or_indirect(self):
  for arg in ['callq *%rax','callq *0x1234(%rip)','movq %rax, %rbx','addr32 callq 0x1234 <name>','callq 0x1234 <name> extra']:
   with self.subTest(arg=arg),self.assertRaises(RuntimeError):a.normalized_wrapper('1000: addr32 '+arg,0x1000)
  for prefix in ('data16','rep','lock'):
   self.assertNotEqual(a.normalized_wrapper('1000: '+prefix+' callq 0x1234 <name>',0x1000),'call\n')

class PrefixWrapper(unittest.TestCase):
 def test_writer_got_register_binding(self):
  reader=lambda address,size:(bytes(size),(0,1))
  text='10: movq 0xe9(%rip), %rcx # 0x100 <misleading_name>\n17: xorl %eax, %eax\n19: movl $0xa01, %esi\n1e: movl $0x180, %edx\n23: callq *%rcx\n'
  self.assertEqual(len(a.prefix_writer_import_calls(text,{0x100},reader)),1)
  self.assertEqual(a.prefix_writer_import_calls(text,{0x101},reader),[])
  for bad in (text.replace('0xe9','0xe8'),text.replace('callq *%rcx','callq *%rax'),text.replace('xorl %eax, %eax','xorl %ecx, %ecx'),text.replace('xorl %eax, %eax','jmp 0x23')):
   self.assertEqual(a.prefix_writer_import_calls(bad,{0x100},reader),[])
  with self.assertRaises(RuntimeError):a.prefix_writer_import_calls(text.replace('# 0x100','# 0x101'),{0x100},reader)
  self.assertEqual(len(a.prefix_writer_import_calls(text+text,{0x100},reader)),2)
 def test_scope_aliases_require_identical_addresses(self):
  roles={'scope_constructor':'ActualPrefixQualificationC1args','scope_destructor':'ActualPrefixQualificationD1Ev'}
  values={'ActualPrefixQualificationC1args':{0x100},'ActualPrefixQualificationC2args':{0x100},'ActualPrefixQualificationD1Ev':{0x200},'ActualPrefixQualificationD2Ev':{0x200}}
  for ctor in ('C1','C2'):
   for dtor in ('D1','D2'):
    text=f'10: addr32 callq 0x100 <ActualPrefixQualification{ctor}args>\n20: callq 0x200 <ActualPrefixQualification{dtor}Ev>'
    normalized=a.prefix_scope_aliases(text,roles,values)
    self.assertEqual(len(a.symbol_calls(normalized,roles['scope_constructor'])),1)
    self.assertEqual(len(a.symbol_calls(normalized,roles['scope_destructor'])),1)
  for name in values:
   for addresses in (set(),{0x300},{0x100,0x200}):
    bad=dict(values);bad[name]=addresses
    with self.subTest(name=name,addresses=addresses),self.assertRaises(RuntimeError):a.prefix_scope_aliases('',roles,bad)
 def test_scope_alias_call_operand_must_match(self):
  roles={'scope_constructor':'ActualPrefixQualificationC1args','scope_destructor':'ActualPrefixQualificationD1Ev'}
  values={name:{address} for name,address in [('ActualPrefixQualificationC1args',0x100),('ActualPrefixQualificationC2args',0x100),('ActualPrefixQualificationD1Ev',0x200),('ActualPrefixQualificationD2Ev',0x200)]}
  for name in values:
   with self.subTest(name=name),self.assertRaises(RuntimeError):a.prefix_scope_aliases('10: callq 0x300 <'+name+'>',roles,values)
  text='10: callq 0x100 <unrelated>\n20: callq *%rax'
  self.assertEqual(a.prefix_scope_aliases(text,roles,values),text)
 def test_scope_alias_indirect_address_and_register_flow(self):
  roles={'scope_constructor':'ActualPrefixQualificationC1args','scope_destructor':'ActualPrefixQualificationD1Ev'}
  values={name:{address} for name,address in [('ActualPrefixQualificationC1args',0x100),('ActualPrefixQualificationC2args',0x100),('ActualPrefixQualificationD1Ev',0x200),('ActualPrefixQualificationD2Ev',0x200)]}
  text='10: leaq 0xe9(%rip), %rax # 0x100 <ActualPrefixQualificationC2args>\n17: leaq -0x20(%rbp), %rdi\n1b: callq *%rax\n'
  self.assertEqual(len(a.symbol_calls(a.prefix_scope_aliases(text,roles,values),roles['scope_constructor'])),1)
  for bad in (text.replace('0xe9','0xe8'),text.replace('# 0x100','# 0x101'),text.replace('17:','18:')):
   with self.assertRaises(RuntimeError):a.prefix_scope_aliases(bad,roles,values)
  for bad in (text.replace('1b:','19: xorl %eax, %eax\n1b:'),text.replace('callq *%rax','callq *%rcx')):
   self.assertEqual(a.symbol_calls(a.prefix_scope_aliases(bad,roles,values),roles['scope_constructor']),[])
 def fixture(self):
  body,base,support,calls=fixture();lines=[]
  for index,line in enumerate(a.PREFIX_WRAPPER_TEMPLATE.splitlines()):
   op,*tail=line.split(' ',1);arg=tail[0] if tail else ''
   if op.startswith('j'):arg='0x'+format(base+int(arg,16),'x')
   elif op=='call':op='callq';arg='0x5000 <target>'
   else:arg=arg.replace('RIP','0x450(%rip)')
   lines.append(f'{base+index*6:x}: {op} {arg}')
  return '\n'.join(lines),base,support,list(enumerate([a.PREFIX_HOOK,*[n for _,n in calls]]))
 def test_reviewed_prefix_wrapper(self):a.wrapper_contract(*self.fixture())
 def test_changed_hook_and_rejection_branches(self):
  body,base,support,calls=self.fixture()
  for bad in [body.replace('je 0x1041','jne 0x1041'),body.replace('jge 0x1159','jge 0x1142'),body.replace('jmp 0x1048','jmp 0x1031'),body.replace('cmpq $0x0, 0x450(%rip)','cmpq $0x1, 0x450(%rip)')]:
   with self.subTest(body=bad),self.assertRaises(RuntimeError):a.wrapper_contract(bad,base,support,calls)
  for bad in [calls[1:],[(i,'wrong' if n==a.PREFIX_HOOK else n) for i,n in calls]]:
   with self.assertRaises(RuntimeError):a.wrapper_contract(body,base,support,bad)
 def test_prefix_model_chain_keeps_wrapped_boundary(self):
  chain=['main','prefix','llama_process','decode','process_ubatch','context',a.WRAPPER]
  bodies={caller:f'10: callq 0x20 <{callee}>\n' for caller,callee in zip(chain,chain[1:])};bodies[chain[-1]]=''
  a.phase_contract('model-prefix','prefix',chain,bodies)
  for caller in chain[:-1]:
   bad=dict(bodies);bad[caller]=''
   with self.assertRaises(RuntimeError):a.phase_contract('model-prefix','prefix',chain,bad)
  bad=dict(bodies);bad['context']+='20: callq 0x40 <ggml_backend_sched_graph_compute_async>\n'
  with self.assertRaises(RuntimeError):a.phase_contract('model-prefix','prefix',chain,bad)

class LayerAudit(unittest.TestCase):
 def fixture(self):
  _,base,support,calls=fixture();lines=[]
  for i,line in enumerate(a.LAYER_WRAPPER_TEMPLATE.splitlines()):
   op,*tail=line.split(' ',1);arg=tail[0] if tail else ''
   if op.startswith('j'):arg='0x'+format(base+int(arg,16),'x')
   elif op=='call':op='callq';arg='0x5000 <target>'
   else:arg=arg.replace('RIP','0x450(%rip)')
   lines.append(f'{base+i*6:x}: {op} {arg}')
  return '\n'.join(lines),base,support,list(enumerate([a.PREFIX_HOOK,a.LAYER_HOOK,*[n for _,n in calls]]))
 def test_exact_two_hooks_and_rejection(self):
  body,base,support,calls=self.fixture();a.wrapper_contract(body,base,support,calls)
  for old,new in [('je 0x1058','jmp 0x1170'),('jge 0x1170','jge 0x1190'),('jne 0x115d','je 0x115d'),('jmp 0x1190','jmp 0x1170')]:
   self.assertIn(old,body)
   with self.subTest(branch=old),self.assertRaises(RuntimeError):a.wrapper_contract(body.replace(old,new),base,support,calls)
  for bad in (calls[1:],calls[:1]+calls[2:],calls+[(999,a.LAYER_HOOK)],[(i,a.PREFIX_HOOK if n==a.LAYER_HOOK else n) for i,n in calls],[(0,a.LAYER_HOOK),(1,a.PREFIX_HOOK),*calls[2:]]):
   with self.assertRaises(RuntimeError):a.wrapper_contract(body,base,support,bad)
  with self.assertRaises(RuntimeError):a.wrapper_contract(body,base,support.replace('xorl %eax, %eax','movl $0x1, %eax'),calls)
 def test_layer_aliases_exact_addresses(self):
  roles={'scope_constructor':'ActualLayerQualificationC1args','scope_destructor':'ActualLayerQualificationD1Ev'}
  values={n:{v} for n,v in [('ActualLayerQualificationC1args',0x100),('ActualLayerQualificationC2args',0x100),('ActualLayerQualificationD1Ev',0x200),('ActualLayerQualificationD2Ev',0x200)]}
  text='10: callq 0x100 <ActualLayerQualificationC2args>\n20: callq 0x200 <ActualLayerQualificationD2Ev>'
  normalized=a.prefix_scope_aliases(text,roles,values,'ActualLayerQualification');self.assertIn('C1args',normalized)
  for badtext,badvalues in [(text.replace('0x100','0x101'),values),(text,{**values,'ActualLayerQualificationC2args':{0x300}})]:
   with self.assertRaises(RuntimeError):a.prefix_scope_aliases(badtext,roles,badvalues,'ActualLayerQualification')
 def contract_fixture(self):
  roles=a.layer_roles(lambda x:x);edges=a.layer_edges(roles)
  for role,apis in [('initialize',['hipModuleLoad','hipModuleGetFunction','hipMalloc']),('shutdown',['hipFree','hipModuleUnload','hipDeviceSynchronize']),('launch',['hipModuleLaunchKernel','hipDeviceSynchronize']),('read',['hipMemcpy'])]:
   edges[roles[role]]=apis
  bodies={n:'\n'.join(f'{16+i*8:x}: callq 0x5000 <{target}>' for i,target in enumerate(targets)) for n,targets in edges.items()}
  return bodies,roles,[roles[n] for n in ('initialize','shutdown','launch','read','execute','preflight')]
 def test_layer_direct_edges_and_connected_public_apis(self):
  bodies,roles,dispatch=self.contract_fixture();a.layer_contract(bodies,roles,dispatch)
  for caller,text in bodies.items():
   for line in text.splitlines():
    bad=dict(bodies);bad[caller]=text.replace(line,'')
    with self.subTest(caller=caller,line=line),self.assertRaises(RuntimeError):a.layer_contract(bad,roles,dispatch)
  for caller in [a.LAYER_HOOK,roles['graph'],*dispatch]:
   bad=dict(bodies);bad[caller]+='\n900: callq 0x5000 <ggml_backend_cpu_init>'
   with self.assertRaises(RuntimeError):a.layer_contract(bad,roles,dispatch)
 def test_counter_lambda_path_must_be_connected(self):
  bodies,roles,dispatch=self.contract_fixture();api='hipModuleLaunchKernel';bodies[roles['launch']]=bodies[roles['launch']].replace(api,'counter');bodies['counter']='10: callq 0x5000 <'+api+'>';dispatch.append('counter');a.layer_contract(bodies,roles,dispatch)
  bad=dict(bodies);bad[roles['launch']]=bad[roles['launch']].replace('counter','unconnected')
  with self.assertRaises(RuntimeError):a.layer_contract(bad,roles,dispatch)

if __name__=='__main__':unittest.main()
