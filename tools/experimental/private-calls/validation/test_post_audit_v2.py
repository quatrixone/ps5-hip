"""Four real baseline ELFs plus synthetic graph/commute negatives; no enabled claim."""
from pathlib import Path
import copy,hashlib,json
from post_audit_v2 import parse_linked,canonical,validate_frame_contract,compare_graphs,frame_move_delta,ROOT,MIDDLE,LEAF,FRAME
p=Path(__file__).resolve().parent;base=p.parent/'g3-private-backend-ipra-frame-baseline'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
contract=base/'measured-contracts.json';assert sha(contract)=='33179c0a773c1072602b5c6b6842a430a9a81a4e9a1ad4eb65751f7680cf5699'
def reject(fn):
 try:fn()
 except AssertionError:return
 raise AssertionError('invalid mutation accepted')
rows=[]
for profile in json.loads(contract.read_text())['profiles']:
 for file,digest in profile['raw_files'].items():assert sha(base/profile['id']/file)==digest
 if profile['kind']!='ordinary':continue
 d=base/profile['id'];graph,descriptor=parse_linked((d/'output.hsaco').read_bytes(),(d/'isa.txt').read_text())
 tail=profile['middle_postcall_observable_window'];validate_frame_contract(canonical(graph),tail)
 after=copy.deepcopy(graph);ops=after[ROOT]['ops'];frame=next(op for op in ops[:5] if op['text']==FRAME)
 after[ROOT]['ops']=[frame]+ops[5:]
 compare_graphs(graph,after,tail)
 moved=copy.deepcopy(after);root=moved[ROOT]['ops'];root.insert(5,root.pop(0))
 reject(lambda:compare_graphs(graph,moved,tail))
 compare_graphs(graph,moved,tail,True)
 count=1
 for name,needle,replacement in [
  (MIDDLE,'s_addk_i32 s32, 0x1400','s_addk_i32 s32, 0x1200'),
  (MIDDLE,'s_addk_i32 s32, 0xec00','s_addk_i32 s32, 0xee00'),
  (MIDDLE,'buffer_store_dword v42, off, s[0:3], s33 offset:144','buffer_store_dword v42, off, s[0:3], s33 offset:128'),
  (MIDDLE,'buffer_load_dword v40, off, s[0:3], s33 offset:4','buffer_load_dword v40, off, s[0:3], s33 offset:8'),
  (MIDDLE,'s_waitcnt vmcnt(2)','s_waitcnt vmcnt(3)'),
  (MIDDLE,tail[5],tail[5].replace(', v0,',', v41,')),
  (LEAF,'v_and_b32_e32 v1, 31, v1','v_and_b32_e32 v1, 30, v1'),
 ]:
  changed=copy.deepcopy(moved);hits=[op for op in changed[name]['ops'] if op['text']==needle]
  if name==LEAF and not hits:
   hits=[op for op in changed[name]['ops'] if '31,' in op['text']]
   assert hits;replacement=hits[0]['text'].replace('31,','30,')
  assert hits,(profile['id'],needle);hits[0]['text']=replacement
  reject(lambda:compare_graphs(graph,changed,tail,True));count+=1
 rows.append({'profile':profile['id'],'baseline_elf_sha256':sha(d/'output.hsaco'),'baseline_frame_contract_verified':True,'synthetic_strict_and_bounded_move_passed':True,'graph_negatives_rejected':count})
before=[FRAME,'s_load_dwordx2 s[6:7], s[4:5], null','v_mov_b32_e32 v1, 0','s_add_u32 s0, s0, s9','s_addc_u32 s1, s1, 0','s_endpgm']
after=before[1:5]+[FRAME]+before[5:];frame_move_delta(before,after)
for op in ['s_mov_b32 s0, s32','s_load_dwordx2 s[31:32], s[4:5], null','s_load_dwordx2 s[6:7], s[32:33], null','s_swappc_b64 s[30:31], s[4:5]','s_branch 1','s_waitcnt lgkmcnt(0)','buffer_store_dword v0, off, s[0:3], s32','s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s6','s_getreg_b32 s5, hwreg(HW_REG_FLAT_SCR_LO)','s_load_dwordx2 s[6:9], s[4:5], null']:
 reject(lambda:frame_move_delta([FRAME,op,'s_endpgm'],[op,FRAME,'s_endpgm']))
reject(lambda:frame_move_delta(before,after+[FRAME]))
reject(lambda:frame_move_delta(before,after[:-1]+['s_endpgm 1']))
reject(lambda:frame_move_delta([FRAME]+['v_mov_b32_e32 v1, 0']*9,['v_mov_b32_e32 v1, 0']*9+[FRAME]))
out=p/'post-audit-v2-tests.json';assert not out.exists()
out.write_text(json.dumps({'real_baseline_profiles':rows,'pure_commute_negative_count':13,'actual_enabled_tested':False,'hardware_validated':False,'frame_contract_sha256':sha(contract),'checker_sha256':sha(p/'post_audit_v2.py'),'recipe_sha256':sha(Path(__file__))},indent=2)+'\n');print('PASS4baselineELFs32graph+13commute negatives',sha(out))
