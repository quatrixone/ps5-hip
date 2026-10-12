"""Versioned paired-IPRA-off 304-byte graph audit. No compiler or GPU invocation."""
from pathlib import Path
import argparse,copy,hashlib,json,re,struct
ROOT='nested_private';MIDDLE='_Z11call_middlePjj.private5';LEAF='_Z9call_leafPjj.private5'
GRAPH=[ROOT,MIDDLE,LEAF]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
INIT=['s_add_u32 s6, s6, s9','s_addc_u32 s7, s7, 0','s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s6','s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s7']
FRAME='s_movk_i32 s32, 0x1200'

def frame_move_delta(before,after,frame=FRAME):
 """Only one immediate s32 initialization may commute past an entry prefix."""
 assert frame in (FRAME,'s_mov_b32 s32, 0')
 assert before.count(frame)==after.count(frame)==1,'frame initializer missing/duplicated'
 a,b=before.index(frame),after.index(frame)
 assert a==0 and 0<=b<=8,'frame initializer moved outside bounded entry prefix'
 assert before[1:]==after[:b]+after[b+1:],'change beyond frame move'
 crossed=after[:b]
 for instruction in crossed:
  scalar=set()
  for lo,hi in re.findall(r'\bs\[(\d+):(\d+)\]',instruction):
   assert int(lo)<=int(hi)<=127;scalar.update(range(int(lo),int(hi)+1))
  scalar.update(int(r) for r in re.findall(r'\bs(\d+)\b',instruction))
  assert 32 not in scalar,'frame register dependency or alias'
  # Fixed emitted entry operations only. No calls, branches, stores, waits,
  # HW state or other implicit stack-pointer consumers are admitted.
  load=re.fullmatch(r's_load_dwordx(2|4) s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], (?:null|0x0)',instruction)
  if load:
   assert int(load[3])-int(load[2])+1==int(load[1]) and int(load[5])-int(load[4])==1
   continue
  assert any(re.fullmatch(pattern,instruction) for pattern in (
   r'v_mov_b32_e32 v\d+, 0',
   r'v_lshl_add_u32 v0, s8, 6, v0',
   r's_add_u32 s0, s0, s9',
   r's_addc_u32 s1, s1, 0',
   r's_mov_b32 s8, 0',
   r'v_lshlrev_b64 v\[1:2\], 2, v\[0:1\]',
  )),('unreviewed crossed instruction',instruction)
 return {'frame':frame,'from_index':a,'to_index':b,'crossed':crossed,'proof':'Only s32 is written by immediate MOV; every crossed known instruction has disjoint explicit registers and no implicit s32 effect. MOV does not read/write SCC, EXEC, VCC or memory.'}

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
  assert struct.unpack_from('<III',data)==(0,304,16)
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

def validate_frame_contract(graph,tail):
 root,middle,leaf=(graph[n] for n in GRAPH)
 assert root.count(FRAME)==1
 assert middle.count('s_addk_i32 s32, 0x1400')==1 and middle.count('s_addk_i32 s32, 0xec00')==1
 assert not any(re.search(r'\bs(?:32|33)\b',instruction) for instruction in leaf)
 for op in ('buffer_store_dword v42, off, s[0:3], s33 offset:144','buffer_load_dword v42, off, s[0:3], s33 offset:144',
            'buffer_store_dword v40, off, s[0:3], s33 offset:4','buffer_store_dword v41, off, s[0:3], s33'):
  assert middle.count(op)==1,('missing ABI spill or WWM slot',op)
 assert len(tail)==6
 loaded=[re.fullmatch(r'buffer_load_dword (v\d+), v\d+, s\[0:3\], 0 offen',op) for op in tail[:2]]
 assert all(loaded) and loaded[0][1]!=loaded[1][1]
 assert tail[2:5]==['buffer_load_dword v41, off, s[0:3], s33','buffer_load_dword v40, off, s[0:3], s33 offset:4','s_waitcnt vmcnt(2)']
 assert tail[5]==f'v_add3_u32 v0, {loaded[0][1]}, v0, {loaded[1][1]}'
 indices=[i for i in range(len(middle)-len(tail)+1) if middle[i:i+len(tail)]==tail]
 assert len(indices)==1,'observable target reloads/ABI restoration/wait/dataflow changed'
 call=next(i for i,op in enumerate(middle) if op.startswith('s_swappc_b64'))
 assert indices[0]>call
 return {'root_bytes':144,'middle_bytes':160,'leaf_bytes':0,'total_private_bytes':304,'wwm_slot_offset':144,'postcall_target_loads':2,'postcall_abi_restore_loads':2,'wait_vmcnt':2}

def compare_graphs(baseline,enabled,tail,allow_frame_move=False):
 before=canonical(baseline);after=canonical(enabled)
 frame_contract=validate_frame_contract(before,tail);assert validate_frame_contract(after,tail)==frame_contract
 # Exact four-instruction emitted initializer; the independent frame setup stays.
 allowed_prefixes=[[INIT[0],INIT[1],FRAME,INIT[2],INIT[3]],[INIT[0],FRAME,INIT[1],INIT[2],INIT[3]]]
 assert before[ROOT][:5] in allowed_prefixes,'unexpected baseline initializer'
 before[ROOT]=[FRAME]+before[ROOT][5:]
 movement=None
 for name in GRAPH:
  for instruction in after[name]:
   opcode=instruction.split()[0]
   assert not opcode.startswith(('flat_','scratch_','s_getreg','s_setreg')),'forbidden memory/HW-state instruction'
   assert not re.search(r'\b(?:src_private_\w*|flat_scr\w*|flat_scratch\w*)\b',instruction,re.I),'forbidden aperture/flat register'
  if name==ROOT and before[name]!=after[name] and allow_frame_move:
   movement=frame_move_delta(before[name],after[name])
  else:
   assert before[name]==after[name],('unexpected enabled instruction delta',name,[(i,a,b) for i,(a,b) in enumerate(zip(before[name],after[name])) if a!=b][:4],len(before[name]),len(after[name]))
 return {'functions':GRAPH,'remaining_instruction_counts':{n:len(after[n]) for n in GRAPH},'removed_instructions':INIT,'frame_contract':frame_contract,'frame_move':movement,'normalization':'Only linked call low/high displacements mapped to their verified target symbol; optional bounded immediate-frame-MOV commute is independently checked. All other instruction operands/order and full callee state remain exact.'}

def main():
 root=Path(__file__).resolve().parent
 parser=argparse.ArgumentParser();parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--enabled',type=Path,required=True);parser.add_argument('--default-off',type=Path,required=True);parser.add_argument('--matrix',type=Path,required=True);parser.add_argument('--frame-contract',type=Path,required=True);parser.add_argument('--allow-frame-move',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 assert not args.output.exists()
 reports=[json.loads((d/'result.json').read_text()) for d in (args.baseline,args.enabled,args.default_off)]
 assert [r['mode'] for r in reports]==['baseline','patched-on','patched-off']
 assert reports[2]['same_build_baseline_identical'] is True and reports[1]['compiler_sha256']==reports[2]['compiler_sha256']
 assert all(r['matrix_sha256']==sha(args.matrix) and len(r['cases'])==24 for r in reports)
 for folder in (args.baseline,args.enabled,args.default_off):
  commands=json.loads((folder/'commands.json').read_text())
  llc_commands=[c['command'] for c in commands if Path(c['command'][0]).name=='llc']
  assert len(llc_commands)==48 and all('-enable-ipra=false' in command for command in llc_commands),'not a paired explicit-IPRA-off matrix'
 matrix=json.loads(args.matrix.read_text());results=[]
 assert sha(args.frame_contract)=='33179c0a773c1072602b5c6b6842a430a9a81a4e9a1ad4eb65751f7680cf5699'
 measured=json.loads(args.frame_contract.read_text())
 for profile in measured['profiles']:
  for file,digest in profile['raw_files'].items():assert sha(args.frame_contract.parent/profile['id']/file)==digest
 contracts={profile['id']:profile for profile in measured['profiles']}
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
  contract=contracts[f"ordinary-{case['target']}-{case['optimization']}-original"]
  results.append({'case':case['id'],**compare_graphs(parsed[0],parsed[1],contract['middle_postcall_observable_window'],args.allow_frame_move),'linked_descriptor_without_entry_relocation_sha256':hashlib.sha256(descriptors[0]).hexdigest()})
 assert len(results)==8
 report={'ordinary_cases':results,'hardware_validated':False,'accepted_compiler':False,'post_audit_passed':True,'frame_contract_sha256':sha(args.frame_contract),'bounded_frame_move_enabled':args.allow_frame_move,'command_receipts':{str(folder/'commands.json'):sha(folder/'commands.json') for folder in (args.baseline,args.enabled,args.default_off)},'baseline_receipt_sha256':sha(args.baseline/'result.json'),'enabled_receipt_sha256':sha(args.enabled/'result.json'),'default_off_receipt_sha256':sha(args.default_off/'result.json'),'script_sha256':sha(Path(__file__)),'limits':'Exact ordinary linked graph delta only. Other matrices, build provenance, private-memory hardware and generic/volatile qualification remain separate.'}
 args.output.write_text(json.dumps(report,indent=2)+'\n');print('PASS8actualenabledgraphs',sha(args.output))
if __name__=='__main__':main()
