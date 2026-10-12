from pathlib import Path
import hashlib,json,re
out=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=read(out/'compile-receipt.json');profiles=[]
for item in receipt['cases']:
 case=item['case'];d=out/case['id'];asm=(d/'output.s').read_text();isa=(d/'isa.txt').read_text();meta=(d/'metadata.txt').read_text()
 for n,h in item['files'].items():assert sha(d/n)==h
 assert 'Relocations [\n]' in meta
 symbols={m[2]:(int(m[1],16),m[3]) for m in re.finditer(r'(?ms)^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',isa)};addresses={v[0]:n for n,v in symbols.items()}
 def calls(name):
  body=symbols[name][1];edges=[]
  for call in re.finditer(r's_swappc_b64 s\[(\d+):(\d+)\], s\[(\d+):(\d+)\]',body):
   lo,hi=call[3],call[4];prefix=body[:call.start()];pcs=list(re.finditer(r's_getpc_b64 s\['+lo+':'+hi+r'\].*?// ([0-9A-F]+):',prefix));assert pcs
   pc=pcs[-1];tail=prefix[pc.end():];add=re.search(r's_add_u32 s'+lo+', s'+lo+r', (0x[0-9a-f]+|-?\d+)\s',tail);assert add
   delta=int(add[1],0)&0xffffffff;delta=delta-(1<<32) if delta&(1<<31) else delta
   high=re.search(r's_addc_u32 s'+hi+', s'+hi+r', (-?\d+)\s',tail);assert high and int(high[1])==(-1 if delta<0 else 0)
   target=int(pc[1],16)+4+delta;assert target in addresses and (call[1],call[2])==('30','31');edges.append(addresses[target])
  assert len(edges)==body.count('s_swappc_b64');return edges
 functions={m[1]:{'body':m[2],'scratch_info_bytes':int(m[3])} for m in re.finditer(r'(?ms)^([_A-Za-z][\w.$]*):[^\n]*\n(.*?)^; ScratchSize: (\d+)',asm)}
 descriptors={m[1]:{k:int(v,0) for k,v in re.findall(r'^\s*\.amdhsa_(\w+)\s+(\w+)',m[2],re.M)} for m in re.finditer(r'\.amdhsa_kernel (\S+)\n(.*?)\.end_amdhsa_kernel',asm,re.S)}
 ordinary=case['kind']=='ordinary';root='nested_private' if ordinary else 'volatile_nested_private';middle='_Z11call_middlePjj.private5' if ordinary else 'volatile_middle';leaf='_Z9call_leafPjj.private5' if ordinary else 'volatile_leaf'
 assert calls(root)==[middle] and calls(middle)==[leaf] and calls(leaf)==[]
 assert descriptors[root]['private_segment_fixed_size']==304 and descriptors[root]['user_sgpr_count']==8 and descriptors[root]['user_sgpr_flat_scratch_init']==1
 assert functions[middle]['scratch_info_bytes']==160 and functions[leaf]['scratch_info_bytes']==0
 assert 's_movk_i32 s32, 0x1200' in functions[root]['body'] and 's_addk_i32 s32, 0x1400' in functions[middle]['body'] and 's_addk_i32 s32, 0xec00' in functions[middle]['body']
 inventories={}
 for name in [root,middle,leaf]:
  body=symbols[name][1];inventories[name]={'calls':calls(name),'flat_init':body.count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_'),'flat_load':len(re.findall(r'^\s*flat_load_',body,re.M)),'flat_store':len(re.findall(r'^\s*flat_store_',body,re.M)),'buffer_load':len(re.findall(r'^\s*buffer_load_',body,re.M)),'buffer_store':len(re.findall(r'^\s*buffer_store_',body,re.M))}
 assert inventories[root]['flat_init']==2
 post=symbols[middle][1].split('s_swappc_b64',1)[1];ops=[line.split('//')[0].strip() for line in post.splitlines() if '//' in line]
 if ordinary:
  pattern=r'buffer_load_dword (v\d+), (v\d+), s\[0:3\], 0 offen';at=next(i for i,op in enumerate(ops) if re.fullmatch(pattern,op));window=ops[at:at+6];a=re.fullmatch(pattern,window[0]);b=re.fullmatch(pattern,window[1]);assert a and b
  assert window[2:5]==['buffer_load_dword v41, off, s[0:3], s33','buffer_load_dword v40, off, s[0:3], s33 offset:4','s_waitcnt vmcnt(2)']
  assert window[5]==f'v_add3_u32 v0, {a[1]}, v0, {b[1]}'
  leafbody=symbols[leaf][1];loads=re.findall(pattern,leafbody);stores=re.findall(r'buffer_store_dword (v\d+), (v\d+), s\[0:3\], 0 offen',leafbody)
  assert len(loads)==2 and stores==[loads[0]] and f'v_add_nc_u32_e32 {loads[0][0]}, 3, {loads[0][0]}' in leafbody and f'v_add_nc_u32_e32 v0, v0, {loads[0][0]}' in leafbody
  assert all(v['flat_load']==v['flat_store']==0 for v in inventories.values())
 else:
  assert inventories[leaf]['flat_load']==3 and inventories[leaf]['flat_store']==1
  assert len(re.findall(r'^\s*flat_load_dword',post,re.M))==2
  window=[op for op in ops if op.startswith(('flat_load','s_waitcnt vmcnt','v_add3'))]
  assert calls('volatile_mixed_private')==[leaf] and descriptors['volatile_mixed_private']['private_segment_fixed_size']==144
 assert re.search(r'global_store_dword v\[\d+:\d+\], v0, off',symbols[root][1])
 saved=[line.strip() for line in functions[middle]['body'].splitlines() if 'Folded Spill' in line or 'Folded Reload' in line]
 assert any('offset:144' in line and 'Folded Spill' in line for line in saved)
 old=out.parent/'g3-private-backend-validation-v3/runs/baseline'/case['id']/'output.s';oldasm=old.read_text();oldbody=re.search(r'(?ms)^'+re.escape(middle)+r':[^\n]*\n(.*?)^; ScratchSize: (\d+)',oldasm)
 assert oldbody and oldbody[2]=='144' and 'offset:128 ; 4-byte Folded Spill' in oldbody[1]
 profiles.append({'id':case['id'],'kind':case['kind'],'target':case['target'],'optimization':case['optimization'],'input_sha256':case['sha256'],'descriptors':{name:descriptors[name] for name in case['roots']},'nested_frame':{'root_per_lane':144,'middle_per_lane':160,'leaf_per_lane':0,'cumulative_per_lane':304,'root_wave_stack_init':'0x1200','middle_wave_stack_increment':'0x1400','middle_wave_stack_decrement_encoded':'0xec00'},'reachable':inventories,'middle_saved_restore_instructions':saved,'middle_postcall_observable_window':window,'comparison_old_ipra_on':{'assembly_sha256':sha(old),'middle_per_lane':144,'lane_save_spill_offset':128,'new_lane_save_spill_offset':144},'raw_files':item['files']})
report={'schema':1,'classification':'independent measured clean-baseline IPRA-off frame contracts; not patched compiler acceptance','compiler':receipt['compiler'],'configuration':['-enable-ipra=false','-verify-machineinstrs'],'compile_receipt_sha256':sha(out/'compile-receipt.json'),'profiles':profiles,'contract_reason':'IPRA-off retains standard call-preserved VGPRs: ordinary v40/v41 plus lane-save v42; volatile v40/v41/v42 plus lane-save v43. Low save slots occupy offsets0/4[/8], moving array origin to16 and lane-save spill from128 to144; 16-byte frame rounding changes middle144 to160. Root local144 and leaf0 are unchanged.','wait_reason':'Ordinary postcall issues two observable data loads then two independent ABI restore loads. vmcnt(2) waits for the earlier two data loads before v_add3; final vmcnt(0) precedes return. Preserve exact per-profile operand connections rather than requiring old adjacent vmcnt(0).','proposal':['Introduce a new versioned matrix/frame audit bound to this independent baseline receipt and exact input/target/optimization; do not replace historical288 contracts.','Require baseline/configuration IPRAoff and enabled compiler byte-identical to paired disabled compiler; disabled outputs must equal independent baseline.','Compare full root descriptors to selected304/144 profile; exact root/middle frame init/increment/decrement remain checked.','Preserve linked reachable nested calls, leaf store/load/add operands and root output store; use the measured six-instruction postcall window per profile including exact restore loads and vmcnt2.','Retain strict no-FLAT/private-aperture acceptance for ordinary reachable code and rejection evidence for volatile generic FLAT.','Do not normalize arbitrary MOVs or instructions; any independently established harmless frame-base MOV rescheduling needs a separately bounded dataflow check.'],'hardware_validated':False,'patched_acceptance':False,'analysis_script_sha256':sha(Path(__file__))}
(out/'measured-contracts.json').write_text(json.dumps(report,indent=2)+'\n');print('measured',len(profiles),'profiles');print(sha(out/'measured-contracts.json'))
