"""Bounded static audit of the vocabulary-phase native client application."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from audit_public_hip_rmsnorm_native import (audit_text,load_regions,require,symbol_values,blocks,section,
    import_slots,bound_calls,symbol_calls,ordered_calls)
from public_hip_imports_test import verify as verify_imports
from native_import_libraries_test import read_imports,nid
WRAPPER='__wrap_ggml_backend_sched_graph_compute_async'
SUPPORTS='_ZN2m0L8supportsEP19ggml_backend_devicePK11ggml_tensor'
DEVICE='_ZN2m013device_handleEv'
# Reviewed LLVM18 O0 native wrapper: every nonempty graph reaches the false
# supports predicate or an earlier rejection. Only the loop exit can forward.
# Preserve all instruction operands/branch edges except relocated RIP addresses.
WRAPPER_TEMPLATE='pushq %rbp\nmovq %rsp, %rbp\nsubq $0x60, %rsp\nmovq %rdi, -0x10(%rbp)\nmovq %rsi, -0x18(%rbp)\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl $0x0, -0x1c(%rbp)\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x34(%rbp)\nmovq -0x18(%rbp), %rdi\ncall\nmovl %eax, %ecx\nmovl -0x34(%rbp), %eax\ncmpl %ecx, %eax\njge 137\nmovq -0x18(%rbp), %rdi\nmovl -0x1c(%rbp), %esi\ncall\nmovq %rax, -0x28(%rbp)\nmovq -0x10(%rbp), %rdi\nmovq -0x28(%rbp), %rsi\ncall\nmovq %rax, -0x30(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje ad\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x40(%rbp)\ncall\nmovq %rax, %rcx\nmovq -0x40(%rbp), %rax\ncmpq %rcx, %rax\njne ad\ncall\nmovq %rax, %rdi\nmovq -0x28(%rbp), %rsi\ncall\ntestb $0x1, %al\njne 124\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x4c(%rbp)\nmovq -0x28(%rbp), %rdi\ncall\nmovq %rax, -0x48(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje ee\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x58(%rbp)\njmp fe\nleaq RIP, %rax\nmovq %rax, -0x58(%rbp)\njmp fe\nmovq -0x48(%rbp), %rdx\nmovl -0x4c(%rbp), %esi\nmovq -0x58(%rbp), %rcx\nleaq RIP, %rdi\nmovb $0x0, %al\ncall\nmovl $0xffffffff, -0x4(%rbp)\njmp 157\njmp 129\nmovl -0x1c(%rbp), %eax\naddl $0x1, %eax\nmovl %eax, -0x1c(%rbp)\njmp 26\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovq -0x10(%rbp), %rdi\nmovq -0x18(%rbp), %rsi\ncall\nmovl %eax, -0x4(%rbp)\nmovl -0x4(%rbp), %eax\naddq $0x60, %rsp\npopq %rbp\nretq\n'
SOURCES=['tools/audit_public_hip_llm_client_native.py','tests/llm_client_native_audit_test.py',
 'tools/audit_public_hip_rmsnorm_native.py','tools/audit_public_hip_embedded_native.py',
 'tools/probe_host_registration.py','tools/native_multi_query.py','tests/fat_binary_fixture_test.py',
 'tests/public_hip_imports_test.py','tests/native_import_libraries_test.py','tools/build_native_vector_title.py',
 'tools/public_hip_llm_client_console_title.cpp','tools/public_hip_app_crt.cpp',
 'ports/ggml/ps5_backend.cpp','ports/ggml/ps5_backend.h','ports/ggml/model_gate.h']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized_wrapper(body,base):
 lines=[]
 for line in body.splitlines():
  m=re.match(r'^\s*([0-9a-f]+):\s+(\w+)\s*(.*)$',line)
  if not m:continue
  _,op,arg=m.groups();arg=arg.split('#')[0].strip()
  if op == 'addr32':
   # lld preserves a six-byte GOTPCRELX call footprint as 67 e8 rel32.
   # Only a direct call is admitted; the existing exact callee/order audit remains mandatory.
   require(re.fullmatch(r'callq?\s+0x[0-9a-f]+\s+<[^<>\s]+>',arg) is not None,'unsupported addr32 instruction')
   value='call'
  elif op.startswith('call'):value='call'
  elif op.startswith('j'):
   target=re.match(r'0x([0-9a-f]+)',arg);require(target is not None,'unresolved wrapper branch')
   value=op+' '+format(int(target[1],16)-base,'x')
  else:value=op+' '+re.sub(r'(?:-?0x[0-9a-f]+)?\(%rip\)','RIP',arg)
  lines.append(value.rstrip())
 return '\n'.join(lines)+'\n'
def wrapper_contract(body,base,support_body,call_positions):
 require(normalized_wrapper(body,base)==WRAPPER_TEMPLATE,'wrapper instruction/control-flow contract differs')
 require(normalized_wrapper(support_body,0)=='movq %rdi, -0x8(%rsp)\nmovq %rsi, -0x10(%rsp)\nxorl %eax, %eax\nandb $0x1, %al\nmovzbl %al, %eax\nretq\n','zero-op supports predicate changed')
 expected=['ggml_graph_n_nodes','ggml_graph_node','ggml_backend_sched_get_tensor_backend',
  'ggml_backend_get_device',DEVICE,DEVICE,SUPPORTS,'ggml_op_desc','ggml_backend_name','printf','ggml_backend_sched_graph_compute_async']
 require([name for _,name in sorted(call_positions)]==expected,'wrapper call targets/order differ')
def bounded_body(body,address,size):
 require(size>0,'missing function extent')
 lines=[]
 for line in body.splitlines(keepends=True):
  match=re.match(r'^\s*([0-9a-f]+):',line)
  if match is None or address<=int(match[1],16)<address+size:lines.append(line)
 return ''.join(lines)

def load_layout(data):
 require(len(data)>=64 and data[:6]==b'\x7fELF\x02\x01','not ELF64LE')
 off=struct.unpack_from('<Q',data,32)[0];ents,count=struct.unpack_from('<HH',data,54)
 require(ents==56 and count>0 and off<=len(data) and count<= (len(data)-off)//ents,'invalid program headers')
 segments=[]
 for i in range(count):
  typ,flags,offset,va,_,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',data,off+i*ents)
  if typ==1:
   require(flags&3!=3,'writable executable segment');require(filesz<=memsz and offset<=len(data) and filesz<=len(data)-offset,'load range invalid')
   segments.append(dict(flags=flags,file_size=filesz,memory_size=memsz,address=va))
 require(segments,'missing load segments');return segments

def lifecycle_imports(imports):
 expected=('libSceLibcInternal','libSceLibcInternal')
 for name in ('atexit','_init_env','printf'):
  require(imports.get(nid(name))==expected,'lifecycle import owner differs: '+name)
 return {name:expected for name in ('atexit','_init_env','printf')}

def audit(raw,converted,extracted):
 paths=[Path(p).resolve() for p in (raw,converted,extracted)]
 tool=Path(shutil.which('llvm-objdump-18') or '');require(tool.is_file(),'objdump unavailable')
 identities={str(p):sha(p) for p in [*paths,tool.resolve(),*(ROOT/n for n in SOURCES)]}
 data=paths[0].read_bytes();values=symbol_values(data)
 def one(n):
  require(n in values and len(values[n])==1,'missing/ambiguous symbol '+n);return values[n][0]
 def match(fragment):
  names=[n for n in values if fragment in n and not n.startswith('_ZZ')];require(len(names)==1,'ambiguous symbol '+fragment);return names[0]
 finalized=match('finalizedEv');vocab=match('vocabulary_check');storage=match('storage_and_gate_check');gate=match('release_compute');registration=match('registrationEv')
 names=['_start','_init','main','ps5_fp_ieee','ps5hip_install_shell_exit_probe','ps5hip_install_vector_title_finalizer',finalized,vocab,storage,gate,registration,'llama_model_load_from_file','llama_tokenize','llama_model_free','hipInit','hipMalloc','hipMemcpy','hipFree','hipDeviceSynchronize','ggml_backend_sched_graph_compute','ggml_backend_graph_compute',WRAPPER,SUPPORTS,DEVICE]
 text=subprocess.check_output([str(tool),'-d','--no-show-raw-insn','--print-imm-hex','--disassemble-symbols='+','.join(names),str(paths[0])],text=True)
 sizes={};symtab=section(data,b'.symtab')
 for at in range(0,len(symtab),24):
  _,info,_,index,value,size=struct.unpack_from('<IBBHQQ',symtab,at)
  if index and info&15==2 and size:sizes[value]=max(sizes.get(value,0),size)
 bodies={a:t for a,n,t in blocks(text)}
 def body(n):return bounded_body(bodies[one(n)],one(n),sizes.get(one(n),0))
 positions=ordered_calls(body('_start'),['ps5_fp_ieee','ps5hip_install_shell_exit_probe','ps5hip_install_vector_title_finalizer','_init','main'])
 reader=load_regions(paths[0])[0];slots={n:import_slots(data,n) for n in ['atexit','_init_env','printf']}
 env=bound_calls(body('_start'),slots['_init_env'],reader);require(len(env)==1 and positions[0]<env[0]<positions[1],'startup env ordering')
 require(len(bound_calls(body('ps5hip_install_vector_title_finalizer'),slots['atexit'],reader))==1,'atexit callback')
 require(one(finalized) in {int(s,16) for s in re.findall(r'0x([0-9a-f]+)',body('ps5hip_install_vector_title_finalizer'))},'finalizer identity')
 ordered_calls(body('main'),[vocab,storage]);ordered_calls(body(vocab),['llama_model_load_from_file','llama_tokenize','llama_model_free'])
 require(symbol_calls(body(storage),gate),'storage missing gate');require(symbol_calls(body(gate),'ggml_backend_sched_graph_compute'),'gate scheduler boundary missing')
 wrapper=body(WRAPPER);calls=[]
 for name in ['ggml_graph_n_nodes','ggml_graph_node','ggml_backend_sched_get_tensor_backend','ggml_backend_get_device',DEVICE,SUPPORTS,'ggml_op_desc','ggml_backend_name','ggml_backend_sched_graph_compute_async']:
  calls.extend((pos,name) for pos in symbol_calls(wrapper,name))
 calls.extend((pos,'printf') for pos in bound_calls(wrapper,slots['printf'],reader))
 wrapper_contract(wrapper,one(WRAPPER),body(SUPPORTS),calls)
 audit_text([load_regions(p)[0] for p in paths],{one(n) for n in names},sizes)
 layouts=[dict(file=str(p),segments=load_layout(p.read_bytes())) for p in paths]
 imports=[{**verify_imports(p),**lifecycle_imports(read_imports(p,verify_hash=True)[2])} for p in paths[1:]]
 require(imports[0]==imports[1],'converted/extracted import owners differ')
 require(all(sha(p)==h for p,h in identities.items()),'audit inputs changed')
 return dict(status='PASS',classification='offline-public-hip-llm-client-native-image-audit',phase='vocab',
  inputs=identities,audited_functions=names,audited_function_count=len(names),layouts=layouts,imports=imports[0],
  wrapper_retained=True,wrapper_no_bypass=True,actual_client_gate_executed=False,native_executed=False,gpu_executed=False,
  limits=['Vocabulary-only image; actual llama_context caller is not required or proven here.',
  'Reviewed LLVM18 O0 wrapper template rejects nonempty graphs with current zero-op predicate; any codegen change requires review.',
  'Static preservation/import/lifecycle audit only; no native loading, tokenization, computation or teardown execution proof.'])
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for n in ['raw','converted','extracted','output']:parser.add_argument('--'+n,type=Path,required=True)
 args=parser.parse_args()
 if args.output.exists():parser.error('fresh output required')
 report=audit(args.raw,args.converted,args.extracted);args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
