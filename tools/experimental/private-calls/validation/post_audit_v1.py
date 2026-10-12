"""Exact linked ordinary-graph delta audit. No compiler invocation or GPU claim."""
from pathlib import Path
import argparse,copy,hashlib,json,re,struct
ROOT='nested_private';MIDDLE='_Z11call_middlePjj.private5';LEAF='_Z9call_leafPjj.private5'
GRAPH=[ROOT,MIDDLE,LEAF]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
INIT=['s_add_u32 s6, s6, s9','s_addc_u32 s7, s7, 0','s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s6','s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s7']
FRAME='s_movk_i32 s32, 0x1200'

class Elf:
 def __init__(self,data):
  self.data=data
  assert data[:7]==b'\x7fELF\x02\x01\x01' and struct.unpack_from('<HH',data,16)==(3,224)
  phoff,shoff=struct.unpack_from('<QQ',data,32);phsize,phnum,shsize,shnum=struct.unpack_from('<HHHH',data,54)
  assert phsize==56 and shsize==64 and phoff+phsize*phnum<=len(data) and shoff+shsize*shnum<=len(data)
  self.loads=[]
  for i in range(phnum):
   typ,flags,off,addr,_,size,_,_=struct.unpack_from('<IIQQQQQQ',data,phoff+i*56)
   assert off+size<=len(data)
   if typ==1:self.loads.append((addr,size,off))
  sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*64) for i in range(shnum)]
  self.symbols={}
  for s in sections:
   if s[1]!=2:continue
   assert s[9]==24 and s[5]%24==0 and s[6]<len(sections)
   strings=sections[s[6]];table=data[strings[4]:strings[4]+strings[5]]
   assert s[4]+s[5]<=len(data)
   for offset in range(s[4],s[4]+s[5],24):
    name,info,_,section,value,size=struct.unpack_from('<IBBHQQ',data,offset)
    assert name<len(table);end=table.find(b'\0',name);assert end>=0
    name=table[name:end].decode()
    if name and section and (info&15) in (1,2):
     assert name not in self.symbols;self.symbols[name]={'type':info&15,'address':value,'size':size}
 def at(self,address,size):
  found=[self.data[off+address-base:off+address-base+size] for base,length,off in self.loads if base<=address and size<=length-(address-base)]
  assert len(found)==1,'ELF address not uniquely backed';return found[0]
 def descriptor(self):
  symbol=self.symbols[ROOT+'.kd'];assert symbol['type']==1 and symbol['size']==64
  data=self.at(symbol['address'],64)
  entry=symbol['address']+struct.unpack_from('<q',data,16)[0]
  assert entry==self.symbols[ROOT]['address']
  assert struct.unpack_from('<III',data)==(0,288,16)
  assert struct.unpack_from('<I',data,52)[0]==0x91 and struct.unpack_from('<H',data,56)[0]==0x429
  return data[:16]+bytes(8)+data[24:]

def parse_linked(data,isa):
 elf=Elf(data);functions={}
 for match in re.finditer(r'(?ms)^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',isa):
  name=match[2]
  if name not in GRAPH:continue
  sym=elf.symbols[name];assert sym['type']==2 and sym['address']==int(match[1],16)
  ops=[];cursor=sym['address'];padding=False
  for line in match[3].splitlines():
   if not line.strip():continue
   parsed=re.fullmatch(r'\s*(.*?)\s+//\s+([0-9A-Fa-f]+):\s+([0-9A-Fa-f ]+?)(?:\s+<[^>]+>)?\s*',line)
   assert parsed,('unparsed disassembly',line)
   text=' '.join(parsed[1].split());address=int(parsed[2],16);words=parsed[3].split()
   assert words and all(len(word)==8 for word in words)
   encoded=b''.join(struct.pack('<I',int(word,16)) for word in words)
   assert address==cursor and elf.at(address,len(encoded))==encoded,'ISA/linked bytes mismatch'
   cursor+=len(encoded)
   if address>=sym['address']+sym['size']:
    assert ops and ops[-1]['text']==('s_endpgm' if name==ROOT else 's_setpc_b64 s[30:31]')
    assert text in ('s_code_end','s_nop 0'),('unexpected padding outside ELF function extent',name,text,hex(address),sym)
    padding=True;continue
   assert not padding and cursor<=sym['address']+sym['size']
   ops.append({'text':text,'address':address,'bytes':len(encoded)})
  assert ops and ops[-1]['address']+ops[-1]['bytes']==sym['address']+sym['size']
  functions[name]={'address':sym['address'],'ops':ops}
 assert set(functions)==set(GRAPH)
 return functions,elf.descriptor()

def canonical(functions):
 output={};edges={};address_names={f['address']:name for name,f in functions.items()}
 for name,function in functions.items():
  ops=function['ops'];texts=[op['text'] for op in ops];edges[name]=[];used=set()
  for i,op in enumerate(ops):
   if not op['text'].startswith('s_swappc'):continue
   call=re.fullmatch(r's_swappc_b64 s\[30:31\], s\[(\d+):(\d+)\]',op['text']);assert call,'unrecognized call'
   lo,hi=call[1],call[2];assert int(hi)==int(lo)+1
   get=[j for j in range(i) if ops[j]['text']==f's_getpc_b64 s[{lo}:{hi}]'];assert get
   j=get[-1];assert j not in used and j+2<i;used.add(j)
   low=re.fullmatch(r's_add_u32 s'+lo+', s'+lo+r', (0x[0-9a-f]+|-?\d+)',ops[j+1]['text'])
   high=re.fullmatch(r's_addc_u32 s'+hi+', s'+hi+r', (0x[0-9a-f]+|-?\d+)',ops[j+2]['text'])
   assert low and high and ops[j]['bytes']==4
   delta=int(low[1],0)&0xffffffff;delta=delta-(1<<32) if delta&(1<<31) else delta
   assert (int(high[1],0)&0xffffffff)==(0xffffffff if delta<0 else 0)
   target=ops[j]['address']+4+delta;assert target in address_names,'unknown linked call target'
   callee=address_names[target];edges[name].append(callee)
   texts[j+1]=f's_add_u32 s{lo}, s{lo}, rel32lo({callee})'
   texts[j+2]=f's_addc_u32 s{hi}, s{hi}, rel32hi({callee})'
  assert len(used)==sum(op['text'].startswith('s_getpc_b64') for op in ops),'unaccounted PC-relative calculation'
  output[name]=texts
 assert edges=={ROOT:[MIDDLE],MIDDLE:[LEAF],LEAF:[]},edges
 return output

def compare_graphs(baseline,enabled):
 before=canonical(baseline);after=canonical(enabled)
 # Exact four-instruction emitted initializer; the independent frame setup stays.
 allowed_prefixes=[[INIT[0],INIT[1],FRAME,INIT[2],INIT[3]],[INIT[0],FRAME,INIT[1],INIT[2],INIT[3]]]
 assert before[ROOT][:5] in allowed_prefixes,'unexpected baseline initializer'
 before[ROOT]=[FRAME]+before[ROOT][5:]
 for name in GRAPH:
  for instruction in after[name]:
   opcode=instruction.split()[0]
   assert not opcode.startswith(('flat_','scratch_','s_getreg','s_setreg')),'forbidden memory/HW-state instruction'
   assert not re.search(r'\b(?:src_private_\w*|flat_scr\w*|flat_scratch\w*)\b',instruction,re.I),'forbidden aperture/flat register'
  assert before[name]==after[name],('unexpected enabled instruction delta',name,[(i,a,b) for i,(a,b) in enumerate(zip(before[name],after[name])) if a!=b][:4],len(before[name]),len(after[name]))
 return {'functions':GRAPH,'remaining_instruction_counts':{n:len(after[n]) for n in GRAPH},'removed_instructions':INIT,'normalization':'Only linked call low/high displacements mapped to their verified target symbol; every other instruction and operand retained.'}

def main():
 root=Path(__file__).resolve().parent
 parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,default=root/'runs/baseline');parser.add_argument('--enabled',type=Path,default=root/'runs/patched-on');parser.add_argument('--default-off',type=Path,default=root/'runs/patched-off');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 assert not args.output.exists()
 reports=[json.loads((d/'result.json').read_text()) for d in (args.baseline,args.enabled,args.default_off)]
 assert [r['mode'] for r in reports]==['baseline','patched-on','patched-off']
 assert reports[2]['same_build_baseline_identical'] is True and reports[1]['compiler_sha256']==reports[2]['compiler_sha256']
 assert all(r['matrix_sha256']==sha(root/'matrix.json') and len(r['cases'])==24 for r in reports)
 matrix=json.loads((root/'matrix.json').read_text());results=[]
 for case in matrix['cases']:
  if case['kind']!='ordinary':continue
  parsed=[];descriptors=[]
  for folder,report in zip((args.baseline,args.enabled,args.default_off),reports):
   record=next(c for c in report['cases'] if c['id']==case['id']);directory=folder/case['id']
   for file,digest in record['files'].items():assert sha(directory/file)==digest,('changed recorded artifact',directory/file)
   graph,descriptor=parse_linked((directory/'output.hsaco').read_bytes(),(directory/'isa.txt').read_text());parsed.append(graph);descriptors.append(descriptor)
  assert descriptors[0]==descriptors[1]==descriptors[2],'linked descriptor resources changed'
  for file in ('output.s','output.o','output.hsaco'):
   assert sha(args.baseline/case['id']/file)==sha(args.default_off/case['id']/file),'default-off bytes changed'
  results.append({'case':case['id'],**compare_graphs(parsed[0],parsed[1]),'linked_descriptor_without_entry_relocation_sha256':hashlib.sha256(descriptors[0]).hexdigest()})
 assert len(results)==8
 report={'ordinary_cases':results,'hardware_validated':False,'accepted_compiler':False,'post_audit_passed':True,'baseline_receipt_sha256':sha(args.baseline/'result.json'),'enabled_receipt_sha256':sha(args.enabled/'result.json'),'default_off_receipt_sha256':sha(args.default_off/'result.json'),'script_sha256':sha(Path(__file__)),'limits':'Exact ordinary linked graph delta only. Other matrices, build provenance, private-memory hardware and generic/volatile qualification remain separate.'}
 args.output.write_text(json.dumps(report,indent=2)+'\n');print('PASS8actualenabledgraphs',sha(args.output))
if __name__=='__main__':main()
