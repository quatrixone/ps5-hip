import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_public_hip_llm_client_native as a
RUN='_ZN2m015run_native_fullENativeFullModelSourceE'
# Actual LLVM18 PS5 proposed title object, main+0x4c8..0x502.
def callsite(source=1,profile=2):
 return '\n'.join(['4c8: movq %rsp, %rax','4cb: movq %rcx, (%rax)',f'4ce: movl $0x{source:x}, 0x10(%rax)',f'4d5: movl $0x{profile:x}, 0x8(%rax)','4dc: leaq 0x123(%rip), %rcx','4e3: movq 0x123(%rip), %r9','4ea: leaq 0xe0(%rsp), %rdi','4f2: leaq 0x110(%rsp), %rdx','4fa: leaq 0x1b8(%rsp), %r8',f'502: callq 0x600 <{RUN}>'])
class OriginalAudit(unittest.TestCase):
 def test_observed_native_operands(self):
  a.original_source_operand(callsite(),RUN)
  a.full_profile_operand(callsite(0),RUN,'generation')
  for bad in (callsite(0),callsite(2),callsite(1,1),callsite().replace('0x10(%rax)','0x18(%rax)'),callsite().replace('502:','500: movl $0x0, 0x10(%rsp)\n502:')):
   with self.subTest(bad=bad),self.assertRaises(RuntimeError):a.original_source_operand(bad,RUN)
  with self.assertRaises(RuntimeError):a.full_profile_operand(callsite(),RUN,'generation')
 def test_invalid_scope_rejected_before_image_io(self):
  for phase,profile,fmt in [('vocab','smoke','original_safetensors'),('model-full','smoke','original_safetensors'),('model-full','corpus','original_safetensors'),('model-full','generation','pt')]:
   with self.assertRaisesRegex(RuntimeError,'unsupported original'):a.audit('missing','missing','missing',phase,profile,fmt)
 def fixture(self):
  r=a.original_roles(lambda s:s,'load_data');f={'run':RUN};edges=a.original_edges(f,r)
  bodies={k:'\n'.join(f'{i+1:x}: callq 0x900 <{v}>' for i,v in enumerate(vs)) for k,vs in edges.items()}
  bodies[RUN]+='\nf0: callq 0x900 <llama_init_from_model>\nf1: leaq 0xabc(%rip), %rsi'
  bodies[r['callback']]+='\nf1: leaq 0xabd(%rip), %rcx'
  bodies[r['backend_allocate']]='f1: leaq 0xabe(%rip), %rcx'
  bodies[r['upload']]+='\nf1: callq *%rax'
  bodies['main']=callsite()
  return bodies,f,r,{r['callback']:0xabc,'ggml_backend_tensor_set':0xabd,r['backend_set']:0xabe}
 def test_each_required_edge_and_address(self):
  b,f,r,addresses=self.fixture();a.original_contract(b,f,r,addresses)
  for caller,targets in a.original_edges(f,r).items():
   for target in targets:
    bad=dict(b);bad[caller]=bad[caller].replace('<'+target+'>','<wrong>')
    with self.subTest(caller=caller,target=target),self.assertRaises(RuntimeError):a.original_contract(bad,f,r,addresses)
  for target in addresses:
   bad=dict(addresses);bad[target]=0xffff
   with self.assertRaises(RuntimeError):a.original_contract(b,f,r,bad)
  bad=dict(b);bad[r['upload']]=bad[r['upload']].replace('callq *%rax','nop')
  with self.assertRaises(RuntimeError):a.original_contract(bad,f,r,addresses)
 def test_real_loader_prefix_not_nested_template(self):
  actual='_ZN18llama_model_loader13load_all_dataEP12ggml_contextRealSignature'
  nested='_ZNSt3__111lower_boundB8ne180100INS_11__wrap_iterIPP11ggml_tensorEES3_ZN18llama_model_loader13load_all_dataEP12ggml_contextComparator'
  values={actual:[1],nested:[2],'_ZZnative18native_read_bufferC2':[3],'_ZZnative18native_read_bufferD2':[4],'_ZZnative18native_read_buffer4data':[5]}
  selected=a.loader_roles(values)['load']
  self.assertEqual(a.original_roles(lambda s:s,selected)['load_data'],actual)
  for bad in ({k:v for k,v in values.items() if k!=actual},{**values,actual+'OtherOverload':[6]}):
   with self.assertRaises(RuntimeError):a.loader_roles(bad)
 def test_strict_model_reset_lifecycle(self):
  def fixture(mode,modern=True):
   r=dict(run=RUN if modern else 'legacy_run',model_reset='model_reset',context_reset='context_reset',disable='disable')
   names=(['llama_model_init_from_user','model_reset'] if mode=='original_safetensors' else [])+(['llama_model_load_from_file','model_reset'] if modern else ['llama_model_load_from_file'])+['llama_init_from_model','decode','context_reset','model_reset','disable']
   return r,names
  def body(names):return '\n'.join(f'{i+1:x}: callq 0x900 <{n}>' for i,n in enumerate(names))
  for mode,modern in [('original_safetensors',True),('gguf',True),('gguf',False)]:
   r,names=fixture(mode,modern);a.full_model_cleanup_contract(body(names),r,mode)
   for i,n in enumerate(names):
    if n=='model_reset':
     with self.subTest(mode=mode,modern=modern,missing=i),self.assertRaises(RuntimeError):a.full_model_cleanup_contract(body(names[:i]+names[i+1:]),r,mode)
   bad=names+['model_reset']
   with self.assertRaises(RuntimeError):a.full_model_cleanup_contract(body(bad),r,mode)
   bad=names[:];i=bad.index('context_reset');bad[i],bad[i+1]=bad[i+1],bad[i]
   with self.assertRaises(RuntimeError):a.full_model_cleanup_contract(body(bad),r,mode)
   if modern:
    bad=names[:];bad.remove('model_reset');bad.insert(bad.index('decode')+1,'model_reset')
    with self.assertRaises(RuntimeError):a.full_model_cleanup_contract(body(bad),r,mode)
   wrong=dict(r,run='legacy_run' if modern else RUN)
   with self.assertRaises(RuntimeError):a.full_model_cleanup_contract(body(names),wrong,mode)
 def test_original_sources_bound(self):
  self.assertIn('ports/ggml/native_original_config.h',a.ORIGINAL_SOURCES)
  self.assertTrue(all((a.ROOT/p).is_file() for p in a.ORIGINAL_SOURCES if p not in ('tests/llm_original_native_audit_test.py','ports/ggml/native_original_config.h')))
if __name__=='__main__':unittest.main()
