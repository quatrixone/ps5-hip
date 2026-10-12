"""Versioned IPRA-off audit; measurements are independent clean-baseline evidence."""
import re

def check_frames(asm,root,reachable,desc):
 functions={m[1]:(m[2],int(m[3])) for m in re.finditer(r'(?ms)^([_A-Za-z][\w.$]*):[^\n]*\n(.*?)^; ScratchSize: (\d+)',asm)}
 assert all(n in functions for n in reachable), 'missing assembly frame inventory'
 init=re.findall(r'^\s*s_movk_i32 s32, (0x[0-9a-f]+)',functions[root][0],re.M)
 assert len(init)==1 and int(init[0],16)%32==0, 'unknown root frame initialization'
 root_bytes=int(init[0],16)//32
 local=[root_bytes]
 for name in reachable[1:]:
  body,size=functions[name];adjustments=re.findall(r'^\s*s_addk_i32 s32, (0x[0-9a-f]+)',body,re.M)
  if size:
   assert len(adjustments)==2 and int(adjustments[0],16)==size*32 and int(adjustments[1],16)==(-size*32)&0xffff, 'frame increment/decrement mismatch'
  else:assert not adjustments, 'zero-frame leaf changes stack'
  local.append(size)
 assert sum(local)==desc['private_segment_fixed_size']==functions[root][1], 'cumulative frame mismatch'
 if root=='volatile_mixed_private':assert local==[144,0]
 if root in ('nested_private','volatile_nested_private'):
  assert len(local)==3 and local[0]==144 and local[2]==0
  # Generic profiles are independently measured by the paired baseline run;
  # ordinary/volatile must also match the pinned eight-profile measurement.
  if reachable[1] in ('_Z11call_middlePjj.private5','volatile_middle'):assert local==[144,160,0]
 return {'reachable_local_bytes':local,'cumulative_bytes':sum(local),'root_wave_stack_init':init[0]}

def inspect_case(case,folder,positive,measured):
 isa=(folder/'isa.txt').read_text();asm=(folder/'output.s').read_text();metadata=(folder/'metadata.txt').read_text()
 assert 'Relocations [\n]' in metadata,'unresolved relocations'
 symbols={m[2]:(int(m[1],16),m[3]) for m in re.finditer(r'(?ms)^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',isa)}
 address_names={value[0]:name for name,value in symbols.items()}
 def calls(name):
  body=symbols[name][1];edges=[]
  for call in re.finditer(r's_swappc_b64 s\[(\d+):(\d+)\], s\[(\d+):(\d+)\]',body):
   lo,hi=call[3],call[4];prefix=body[:call.start()]
   pcs=list(re.finditer(r's_getpc_b64 s\['+lo+':'+hi+r'\].*?// ([0-9A-F]+):',prefix));assert pcs,(name,'missing directPC')
   pc=pcs[-1];tail=prefix[pc.end():]
   add=re.search(r's_add_u32 s'+lo+', s'+lo+r', (0x[0-9a-f]+|-?\d+)\s',tail);assert add,(name,'missing displacement')
   delta=int(add[1],0)&0xffffffff;delta=delta-(1<<32) if delta&(1<<31) else delta
   high=re.search(r's_addc_u32 s'+hi+', s'+hi+r', (-?\d+)\s',tail);assert high and int(high[1])==(-1 if delta<0 else 0)
   address=int(pc[1],16)+4+delta;assert address in address_names,(name,hex(address))
   assert (call[1],call[2])==('30','31'),(name,'returnregisterpair')
   edges.append({'caller':name,'callee':address_names[address],'address':address})
  assert len(edges)==body.count('s_swappc_b64'),(name,'unknowncallform')
  return edges
 def walk(name,seen):
  assert name not in seen,'recursive graph';seen=seen|{name};edges=calls(name);result=[name]
  for edge in edges:result+=walk(edge['callee'],seen)
  return result
 descriptors={m[1]:{k:int(v,0) for k,v in re.findall(r'^\s*\.amdhsa_(\w+)\s+(\w+)',m[2],re.M)} for m in re.finditer(r'\.amdhsa_kernel (\S+)\n(.*?)\.end_amdhsa_kernel',asm,re.S)}
 roots=[]
 for root in case['roots']:
  assert root in symbols and root in descriptors
  reachable=walk(root,set());inventory={}
  for name in reachable:
   body=symbols[name][1]
   inventory[name]={'flat':len(re.findall(r'^\s*flat_\w+',body,re.M)),'scratch':len(re.findall(r'^\s*scratch_\w+',body,re.M)),'aperture':body.count('src_private_'),'flat_init':body.count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_'),'buffer_load':len(re.findall(r'^\s*buffer_load_\w+',body,re.M)),'buffer_store':len(re.findall(r'^\s*buffer_store_\w+',body,re.M)),'calls':calls(name)}
  desc=descriptors[root];assert desc['user_sgpr_count']==8 and desc['user_sgpr_flat_scratch_init']==1
  if root in ('nested_private','volatile_nested_private'):assert len(reachable)==3
  if root=='volatile_mixed_private':assert len(reachable)==2
  profile=next((p for p in measured['profiles'] if (p['kind'],p['target'],p['optimization'])==(case['kind'],case['target'],case['optimization'])),None)
  if case['kind'] in ('ordinary','volatile'):
   assert profile is not None and desc==profile['descriptors'][root], (root,'measured baseline descriptor differs')
  frames=check_frames(asm,root,reachable,desc)
  if case['kind']=='ordinary':
   assert reachable==['nested_private','_Z11call_middlePjj.private5','_Z9call_leafPjj.private5']
   assert all(not v['flat'] and not v['scratch'] and not v['aperture'] for v in inventory.values())
   assert all(inventory[name]['buffer_load'] and inventory[name]['buffer_store'] for name in reachable[1:])
   middle=symbols[reachable[1]][1];post=middle.split('s_swappc_b64',1)[1]
   ops=[line.split('//')[0].strip() for line in post.splitlines() if '//' in line]
   load_pattern=r'buffer_load_dword (v\d+), (v\d+), s\[0:3\], 0 offen'
   first=next((i for i,op in enumerate(ops) if re.fullmatch(load_pattern,op)),None)
   assert first is not None and first+5<len(ops),'missingpostcallloads'
   first_load=re.fullmatch(load_pattern,ops[first]);second_load=re.fullmatch(load_pattern,ops[first+1])
   assert second_load and ops[first+2:first+5]==['buffer_load_dword v41, off, s[0:3], s33','buffer_load_dword v40, off, s[0:3], s33 offset:4','s_waitcnt vmcnt(2)']
   assert ops[first+5]==f'v_add3_u32 v0, {first_load[1]}, v0, {second_load[1]}','disconnectedpostcallreload'
   assert ops[first:first+6]==profile['middle_postcall_observable_window'], 'measured postcall operand/wait window differs'
   leaf=symbols[reachable[2]][1]
   loads=re.findall(load_pattern,leaf);stores=re.findall(r'buffer_store_dword (v\d+), (v\d+), s\[0:3\], 0 offen',leaf)
   assert len(loads)==2 and stores==[loads[0]],'leafprivateaccesscontract'
   assert f'v_add_nc_u32_e32 {loads[0][0]}, 3, {loads[0][0]}' in leaf
   assert f'v_add_nc_u32_e32 v0, v0, {loads[0][0]}' in leaf and 's_setpc_b64 s[30:31]' in leaf
   assert re.search(r'global_store_dword v\[\d+:\d+\], v0, off',symbols[root][1]),'returnnotstored'
   assert 's_movk_i32 s32, 0x1200' in symbols[root][1] and 's_addk_i32 s32, 0x1400' in middle and 's_addk_i32 s32, 0xec00' in middle
  else:assert any(v['flat'] for v in inventory.values()),'negative no longer exercises generic FLAT'
  assert inventory[root]['flat_init']==(0 if positive else 2),(root,'unexpectedflatinit',inventory[root])
  roots.append({'root':root,'reachable':reachable,'descriptor':desc,'inventory':inventory,'frames':frames})
 return roots

