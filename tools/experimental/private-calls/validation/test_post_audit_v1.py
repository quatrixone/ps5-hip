"""Read frozen baseline artifacts; test synthetic graph mutations in memory only."""
from pathlib import Path
import copy,hashlib,json,re
import post_audit_v1 as audit
root=Path(__file__).resolve().parent
matrix=json.loads((root/'matrix.json').read_text());records=[]
for case in matrix['cases']:
 if case['kind']!='ordinary':continue
 directory=root/'runs/baseline'/case['id']
 data=(directory/'output.hsaco').read_bytes();isa=(directory/'isa.txt').read_text()
 baseline,descriptor=audit.parse_linked(data,isa)
 enabled=copy.deepcopy(baseline)
 enabled[audit.ROOT]['ops']=[op for op in enabled[audit.ROOT]['ops'] if op['text'] not in audit.INIT]
 audit.compare_graphs(baseline,enabled)
 mutations={}
 state=copy.deepcopy(enabled);state[audit.LEAF]['ops'].insert(0,{'text':'s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s6','address':0,'bytes':4});mutations['callee_hw_write']=state
 read=copy.deepcopy(enabled);read[audit.MIDDLE]['ops'].insert(0,{'text':'s_getreg_b32 s6, hwreg(HW_REG_MODE)','address':0,'bytes':4});mutations['callee_hw_read']=read
 aperture=copy.deepcopy(enabled);aperture[audit.LEAF]['ops'].insert(0,{'text':'s_mov_b64 s[6:7], src_private_base','address':0,'bytes':4});mutations['callee_aperture']=aperture
 index=copy.deepcopy(enabled)
 selected=next(op for op in index[audit.LEAF]['ops'] if op['text'].startswith('v_and_b32_e32') and ', 31,' in op['text'])
 selected['text']=selected['text'].replace(', 31,',', 30,');mutations['leaf_wrong_index']=index
 address=copy.deepcopy(enabled);selected=next(op for op in address[audit.LEAF]['ops'] if op['text'].startswith('v_lshl_add_u32'))
 selected['text']=selected['text'].replace(', 2,',', 3,');mutations['leaf_wrong_scale']=address
 frame=copy.deepcopy(enabled);selected=next(op for op in frame[audit.MIDDLE]['ops'] if op['text']=='s_addk_i32 s32, 0x1200')
 selected['text']='s_addk_i32 s32, 0x1000';mutations['middle_wrong_frame']=frame
 for name,graph in mutations.items():
  try:audit.compare_graphs(baseline,graph)
  except AssertionError:pass
  else:raise AssertionError('accepted '+name)
 relocated=copy.deepcopy(enabled)
 for i,name in enumerate(audit.GRAPH):
  shift=(i+1)*0x1000;relocated[name]['address']+=shift
  for op in relocated[name]['ops']:op['address']+=shift
 for name,callee in [(audit.ROOT,audit.MIDDLE),(audit.MIDDLE,audit.LEAF)]:
  ops=relocated[name]['ops'];j=next(i for i,op in enumerate(ops) if op['text'].startswith('s_getpc_b64'))
  delta=relocated[callee]['address']-(ops[j]['address']+4)
  ops[j+1]['text']=re.sub(r', (?:0x[0-9a-f]+|-?\d+)$',', '+hex(delta&0xffffffff),ops[j+1]['text'])
  ops[j+2]['text']=re.sub(r', (?:0x[0-9a-f]+|-?\d+)$',', '+str(-1 if delta<0 else 0),ops[j+2]['text'])
 audit.compare_graphs(baseline,relocated)
 leaf_begin=isa.index('<'+audit.LEAF+'>:')
 bad_isa=isa[:leaf_begin]+isa[leaf_begin:].replace('BF8C0000','BF8C0001',1);assert bad_isa!=isa
 try:audit.parse_linked(data,bad_isa)
 except AssertionError:pass
 else:raise AssertionError('ISA byte mismatch accepted')
 records.append({'case':case['id'],'actual_linked_baseline_parsed':True,'synthetic_initializer_removal_accepted':True,'synthetic_semantic_call_relocation_accepted':True,'rejected_mutations':list(mutations)+['ISA_bytes_mismatch'],'baseline_hsaco_sha256':hashlib.sha256(data).hexdigest()})
source=Path(json.loads((root.parents[1]/'prepare-inputs.json').read_text())['llvm_source'])/'llvm/lib/Target/AMDGPU/SIFrameLowering.cpp'
report={'actual_enabled_compiler_output_tested':False,'compiler_invocations':0,'hardware_validated':False,'cases':records,'removable_slice_source':{'file':'llvm/lib/Target/AMDGPU/SIFrameLowering.cpp','commit':'3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff','sha256':audit.sha(source),'lines':'473-498','description':'GFX10 pointer add low/high then FLAT_SCR_LO/HI writes; independent frame setup retained.'},'audit_sha256':audit.sha(root/'post_audit_v1.py'),'test_sha256':audit.sha(Path(__file__))}
destination=root/'post-audit-v1-tests.json';assert not destination.exists();destination.write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(records),'baseline cases;',7*len(records),'rejected mutations;',audit.sha(destination))
