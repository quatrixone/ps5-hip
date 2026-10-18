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

class Full(unittest.TestCase):
 @staticmethod
 def instructions(template,base,offsets=None,targets=None):
  source=template.splitlines();offsets=offsets or [i*6 for i in range(len(source))];targets=targets or {}
  lines=[]
  for i,line in enumerate(source):
   op,*tail=line.split(' ',1);arg=tail[0] if tail else ''
   if op.startswith('j') and arg!='*%rax':arg='0x'+format(base+int(arg,16),'x')
   elif op=='call':op='callq';arg='0x9000 <target>'
   elif 'RIP' in arg:
    target=targets.get(i,0x8000);disp=target-base-offsets[i+1]
    arg=arg.replace('RIP',('-' if disp<0 else '')+hex(abs(disp))+'(%rip)')+' # '+hex(target)
   lines.append(f'{base+offsets[i]:x}: {op} {arg}')
  return '\n'.join(lines)+'\n'
 def wrapper_fixture(self):
  lines=a.FULL_WRAPPER_TEMPLATE.splitlines();index=lines.index('testb $0x1, RIP');offsets=[i*6 for i in range(len(lines))];offsets[index:index+2]=[0x2d,0x34]
  return self.instructions(a.FULL_WRAPPER_TEMPLATE,0x1000,offsets,{index:0x8000}),list(enumerate(a.FULL_WRAPPER_CALLS))
 def test_full_wrapper_requires_exact_control_flow_and_calls(self):
  body,calls=self.wrapper_fixture();a.full_wrapper_contract(body,0x1000,calls,0x8000)
  for before,after in [('jge ','jl '),('sete ','setne '),('$0x31e','$0x31f'),('movl $0x0, -0x4','movl $0x1, -0x4')]:
   with self.subTest(before=before),self.assertRaises(RuntimeError):a.full_wrapper_contract(body.replace(before,after,1),0x1000,calls,0x8000)
  for changed in (calls[:-1],calls+[(1000,'bypass')],list(enumerate(reversed(a.FULL_WRAPPER_CALLS)))):
   with self.assertRaises(RuntimeError):a.full_wrapper_contract(body,0x1000,changed,0x8000)
  with self.assertRaises(RuntimeError):a.full_wrapper_contract(body,0x1000,calls,0x8001)
 def test_full_supported_opcode_table(self):
  offsets=[0,5,0xa,0x11,0x17,0x1d,0x23,0x28,0x2d,0x32,0x35,0x3a,0x3e,0x44,0x49,0x50,0x54,0x57,0x59,0x5e,0x63,0x68,0x6c,0x6e,0x71]
  body=self.instructions(a.FULL_SUPPORT_TEMPLATE,0x2000,offsets,{2:0x8000,14:0x9000})
  table=struct.pack('<101i',*[0x2000+(0x59 if i in a.FULL_OPS else 0x63)-0x9000 for i in range(101)])
  def reader(addr,size):self.assertEqual((addr,size),(0x9000,404));return table,None
  self.assertEqual(a.full_support_contract(body,0x2000,reader,0x8000),0x9000)
  for op in range(101):
   changed=bytearray(table);struct.pack_into('<i',changed,4*op,0x2000+(0x63 if op in a.FULL_OPS else 0x59)-0x9000)
   with self.subTest(op=op),self.assertRaises(RuntimeError):a.full_support_contract(body,0x2000,lambda *_:(bytes(changed),None),0x8000)
  with self.assertRaises(RuntimeError):a.full_support_contract(body.replace('ja ','jge ',1),0x2000,reader,0x8000)
 def test_full_global_and_mode_store_targets(self):
  body='1000: movb $0x1, 0xff9(%rip) # 0x2000\n1007: retq\n'
  a.full_global_bindings(body,0x1000,{0:('enabled',0)},lambda _:0x2000)
  a.full_mode_store(body,0x2000,1)
  with self.assertRaises(RuntimeError):a.full_global_bindings(body,0x1000,{0:('enabled',0)},lambda _:0x3000)
  with self.assertRaises(RuntimeError):a.full_mode_store(body,0x2000,0)
  with self.assertRaises(RuntimeError):a.full_mode_store(body+body,0x2000,1)
 def test_full_initial_mode_flag_cannot_enable_data_only_bypass(self):
  header=(1,6,0,0x1000,0,16,32,4096)
  a.zero_initialized_flag(lambda *_:(b'\0',None),[header],0x1000)
  a.zero_initialized_flag(lambda *_:self.fail('BSS must not read stored bytes'),[header],0x1010)
  with self.assertRaises(RuntimeError):a.zero_initialized_flag(lambda *_:(b'\1',None),[header],0x1000)
  with self.assertRaises(RuntimeError):a.zero_initialized_flag(lambda *_:(b'\0',None),[header],0x1020)
  with self.assertRaises(RuntimeError):a.zero_initialized_flag(lambda *_:(b'\0',None),[(1,7,*header[2:])],0x1000)
 @staticmethod
 def smoke_body():
  return '10: movq %rsp, %rax\n13: movq %rcx, (%rax)\n16: movl $0x1, 0x8(%rax)\n1d: leaq 0x30(%rip), %rcx\n24: movq %rbx, %r9\n27: leaq 0xc0(%rsp), %rdi\n2f: leaq 0xf0(%rsp), %rdx\n37: leaq 0x218(%rsp), %r8\n40: callq 0x9000 <run>\n'
 def test_smoke_argument_not_long_or_overwritten(self):
  body=self.smoke_body();a.full_profile_operand(body,'run')
  for changed in (body.replace('$0x1','$0x0'),body.replace('0x8(%rax)','0x10(%rax)'),body.replace('40: callq','38: movl $0x0, 0x8(%rsp)\n40: callq'),body+body):
   with self.assertRaises(RuntimeError):a.full_profile_operand(changed,'run')
 def contract_fixture(self):
  names=('run enable disable reset context_reset model_reset readback writer graph backend_init release begin finish execute initialize shutdown controls select node launch read dispatch_reset').split();roles={n:n for n in names}
  bodies={n:'' for n in names};bodies['main']=self.smoke_body();bodies[a.WRAPPER]=''
  for caller,targets in a.full_edges(roles).items():
   if caller!='main':bodies[caller]=''.join(f'{i*8+10:x}: callq 0x9000 <{t}>\n' for i,t in enumerate(targets))
  for role,apis in [('initialize',('hipModuleLoad','hipModuleGetFunction','hipMalloc')),('shutdown',('hipFree','hipModuleUnload','hipDeviceSynchronize')),('launch',('hipModuleLaunchKernel','hipDeviceSynchronize')),('read',('hipMemcpy',))]:
   bodies[role]+=''.join(f'{i*8+100:x}: callq 0x9000 <{t}>\n' for i,t in enumerate(apis))
  return bodies,roles,[roles[n] for n in ('begin','finish','execute','initialize','shutdown','controls','node','launch','read','dispatch_reset')]
 def test_full_real_scheduler_reset_cleanup_and_hip_edges(self):
  bodies,roles,dispatch=self.contract_fixture();a.full_contract(bodies,roles,dispatch)
  for caller,targets in a.full_edges(roles).items():
   for target in targets:
    bad=dict(bodies);bad[caller]=bad[caller].replace('<'+target+'>','<wrong>')
    with self.subTest(caller=caller,target=target),self.assertRaises(RuntimeError):a.full_contract(bad,roles,dispatch)
  bad=dict(bodies);bad['run']=bad['run'].replace('<llama_memory_clear>','<temporary>').replace('<reset>','<llama_memory_clear>').replace('<temporary>','<reset>')
  with self.assertRaises(RuntimeError):a.full_contract(bad,roles,dispatch)
  for caller in dispatch:
   bad=dict(bodies);bad[caller]+='900: callq 0x9000 <ggml_backend_cpu_init>\n'
   with self.assertRaises(RuntimeError):a.full_contract(bad,roles,dispatch)

class Generation(unittest.TestCase):
 def fixture(self):
  r={n:n for n in ('helper','sampler','process','decode','eog','greedy')};full={'run':'run','reset':'reset'}
  bodies={n:'' for n in (*r.values(),*full.values())}
  for caller,targets in a.generation_edges(full,r).items():
   bodies[caller]=''.join(f'{10+i*8:x}: callq 0x9000 <{t}>\n' for i,t in enumerate(targets))
  bodies['sampler']+='80: callq 0x9000 <process>\n'
  return bodies,full,r,list(r.values())
 def test_generation_abi_is_distinct_and_not_overwritten(self):
  body=Full.smoke_body().replace('$0x1','$0x2');a.full_profile_operand(body,'run','generation')
  for profile,changed in [('smoke',body),('generation',Full.smoke_body()),('generation',body.replace('40: callq','38: movl $0x1, 0x8(%rsp)\n40: callq')),('long',body)]:
   with self.subTest(profile=profile),self.assertRaises(RuntimeError):a.full_profile_operand(changed,'run',profile)
 def test_all_actual_generation_edges_required(self):
  bodies,full,r,scope=self.fixture();a.generation_contract(bodies,full,r,scope)
  for caller,targets in a.generation_edges(full,r).items():
   for callee in targets:
    changed=dict(bodies);changed[caller]=changed[caller].replace('<'+callee+'>','<wrong>')
    with self.subTest(caller=caller,callee=callee),self.assertRaises(RuntimeError):a.generation_contract(changed,full,r,scope)
 def test_sampling_order_feedback_and_no_fallback(self):
  bodies,full,r,scope=self.fixture()
  for caller,changed in [('process',bodies['process'].replace('<decode>','<temp>').replace('<greedy>','<decode>').replace('<temp>','<greedy>')),('sampler',bodies['sampler'].replace('80: callq 0x9000 <process>','')),('decode',bodies['decode'].replace('<llama_get_logits_ith>','<wrong>'))]:
   modified=dict(bodies);modified[caller]=changed
   with self.assertRaises(RuntimeError):a.generation_contract(modified,full,r,scope)
  for caller in scope:
   modified=dict(bodies);modified[caller]+='900: callq 0x9000 <ggml_backend_cpu_init>\n'
   with self.subTest(caller=caller),self.assertRaises(RuntimeError):a.generation_contract(modified,full,r,scope)
 def test_generation_symbols_must_be_unique_functions(self):
  names=['_ZN2m019run_full_generationEargs','_ZN17ps5hip_generation8generateIargs','_ZZN17ps5hip_generation8generateIargsENKUliE_clEi','_ZZN2m019run_full_generationEargsENKUliiE_clEii','_ZZN2m019run_full_generationEargsENKUliE_clEi','_ZN17ps5hip_generation6greedyEargs']
  values={n:[i+1] for i,n in enumerate(names)};addresses=set(range(1,7));a.generation_roles(values,addresses)
  for name in names:
   with self.subTest(name=name),self.assertRaises(RuntimeError):a.generation_roles({n:v for n,v in values.items() if n!=name},addresses)
  with self.assertRaises(RuntimeError):a.generation_roles({**values,names[3]+'junk_clEii':[7]},addresses|{7})
  with self.assertRaises(RuntimeError):a.generation_roles(values,addresses-{4})

class NativeLoader(unittest.TestCase):
 def fixture(self):
  roles=dict(load='loader',constructor='construct',destructor='destroy',data='data')
  calls=['construct','data','read_raw','data','ggml_backend_tensor_set','data','ggml_validate_row_data','destroy','destroy']
  body=lambda targets:'\n'.join(f'{10+i*8:x}: callq 0x9000 <{t}>' for i,t in enumerate(targets))
  return roles,{'loader':body(calls),'construct':body(['__wrap_ggml_aligned_malloc']),'destroy':body(['__wrap_ggml_aligned_free']),'data':'10: retq'}
 def test_native_pair_and_normal_exception_release_edges(self):
  roles,bodies=self.fixture();a.loader_contract(bodies,roles)
  for role,target in [('load','construct'),('load','destroy'),('load','data'),('load','ggml_backend_tensor_set'),('constructor','__wrap_ggml_aligned_malloc'),('destructor','__wrap_ggml_aligned_free')]:
   bad=dict(bodies);name=roles[role];bad[name]=bad[name].replace('<'+target+'>','<wrong>',1)
   with self.subTest(role=role,target=target),self.assertRaises(RuntimeError):a.loader_contract(bad,roles)
 def test_original_pair_or_staging_vector_cannot_bypass(self):
  roles,bodies=self.fixture()
  for role in roles:
   for target in ['ggml_aligned_malloc','ggml_aligned_free','malloc','calloc','realloc','free']:
    bad=dict(bodies);bad[roles[role]]+='\n100: callq 0x9000 <'+target+'>'
    with self.subTest(role=role,target=target),self.assertRaises(RuntimeError):a.loader_contract(bad,roles)
  bad=dict(bodies);bad['loader']+='\n100: callq 0x9000 <stdvector_no_init_ctor>'
  with self.assertRaises(RuntimeError):a.loader_contract(bad,roles)
 def test_owner_symbols_are_local_and_unambiguous(self):
  values={n:1 for n in ['_ZN18llama_model_loader13load_all_dataargs','_ZZload18native_read_bufferC2Em','_ZZload18native_read_bufferD2Ev','_ZZload18native_read_buffer4dataEv']}
  a.loader_roles(values)
  for name in list(values):
   with self.subTest(name=name),self.assertRaises(RuntimeError):a.loader_roles({n:v for n,v in values.items() if n!=name})
  with self.assertRaises(RuntimeError):a.loader_roles({**values,'_ZZother18native_read_bufferD2Ev':2})
 def machine_fixture(self):
  roles,bodies=self.fixture();bases={k:0x1000 for k in roles}
  for role,template in a.LOADER_TEMPLATES.items():
   text=Full.instructions(template,0x1000)
   targets={'constructor':['__wrap_ggml_aligned_malloc','__cxa_allocate_exception','bad_alloc_ctor','__cxa_throw'],
            'destructor':['__wrap_ggml_aligned_free','__clang_call_terminate'],'data':[]}[role]
   for target in targets:text=text.replace('<target>','<'+target+'>',1)
   bodies[roles[role]]=text
  rows=[]
  for line in bodies['loader'].splitlines():
   if '<construct>' in line:rows.append('1: movq -0x1a0(%rbp), %rsi')
   if any('<'+roles[k]+'>' in line for k in ('constructor','destructor','data')):rows.append('2: leaq -0x2b0(%rbp), %rdi')
   rows.append(line)
  bodies['loader']='\n'.join(rows)
  return roles,bodies,bases
 def test_native_size_pointer_null_branch_and_owner_abi(self):
  roles,bodies,bases=self.machine_fixture();a.loader_machine_contract(bodies,roles,bases)
  for role,before,after in [('constructor','jne ','je '),('constructor','0x8(%rax)','0x10(%rax)'),
    ('constructor','-0x10(%rbp), %rdi','-0x8(%rbp), %rdi'),('destructor','(%rax), %rsi','0x8(%rax), %rsi'),
    ('destructor','jmp ','jne '),('data','0x8(%rax)','(%rax)'),('load','-0x1a0(%rbp)','-0x1a8(%rbp)'),
    ('load','-0x2b0(%rbp)','-0x2c0(%rbp)')]:
   bad=dict(bodies);name=roles[role];self.assertIn(before,bad[name]);bad[name]=bad[name].replace(before,after,1)
   with self.subTest(role=role,before=before),self.assertRaises(RuntimeError):a.loader_machine_contract(bad,roles,bases)
  for target in ['__cxa_allocate_exception','__cxa_throw']:
   bad=dict(bodies);bad['construct']=bad['construct'].replace('<'+target+'>','<wrong>')
   with self.assertRaises(RuntimeError):a.loader_machine_contract(bad,roles,bases)

if __name__=='__main__':unittest.main()
