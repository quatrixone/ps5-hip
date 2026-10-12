"""Mutate actual paired assembly; never run compiler or claim execution equivalence."""
from pathlib import Path
import hashlib,json
from post_audit_extensions_v2 import positive_equal,rejected_equal
p=Path(__file__).resolve().parent;v=p;m=p.parent/'g3-private-backend-mir-controls'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def reject(fn):
 try:fn()
 except AssertionError:return
 raise AssertionError('bad mutation accepted')
rows=[]
for target in ('gfx1013','gfx1030'):
 for kind,folder,name,frame,root,calls in [('alias',v/'negative-runs','alias-'+target,'s_movk_i32 s32, 0x400','negative',1),('small-stress',v/'stress-runs',target,'s_mov_b32 s32, 0','stress',2),('MIR',m/'runs-v2','unchanged-'+target,'s_mov_b32 s32, 0','stress',2)]:
  suffix='-late-callee-unbounded' if kind=='small-stress' else ''
  before=folder/('baseline'+suffix)/name/'output.s';after=folder/('patched-on'+suffix)/name/'output.s'
  b,e=before.read_text(),after.read_text();positive_equal(b,e,root,frame,calls)
  mutations={'frame-immediate':e.replace(frame,'s_mov_b32 s32, 123',1),'callee-state':e.replace('\ts_setpc_b64','\ts_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s6\n\ts_setpc_b64',1),'resource':e.replace('.amdhsa_user_sgpr_count 8','.amdhsa_user_sgpr_count 6',1),'extra-directive':e+'\t.p2align 2\n','s32-dependency':e.replace('s_add_u32 s0, s0, s9','s_add_u32 s0, s32, s9',1),'wait':e.replace(frame,frame+'\n\ts_waitcnt vmcnt(0)',1)}
  if kind=='alias':
   mutations.update({
    'pc-target':e.replace('safe@rel32@lo+4','unknown@rel32@lo+4',1),
    'pc-addend':e.replace('safe@rel32@hi+12','safe@rel32@hi+8',1),
    'pc-register-overlap':e.replace('s_getpc_b64 s[4:5]','s_getpc_b64 s[0:1]',1),
    'store-descriptor-overlap':e.replace('buffer_store_dword v0, off, s[0:3], 0 offset:16','buffer_store_dword v0, off, s[4:7], 0 offset:16',1),
    'store-offset':e.replace('0 offset:16','0 offset:20',1),
    'store-order':e.replace('\tbuffer_store_dword v0, off, s[0:3], 0 offset:16\n\tv_mov_b32_e32 v0, 16\n','\tv_mov_b32_e32 v0, 16\n\tbuffer_store_dword v0, off, s[0:3], 0 offset:16\n',1),
    'SCC-consumer':e.replace('v_mov_b32_e32 v0, 16','s_cmov_b32 s8, 16',1),
    'extra-op':e.replace('s_getpc_b64 s[4:5]','s_nop 0\n\ts_getpc_b64 s[4:5]',1),
    'EXEC-write':e.replace('v_mov_b32_e32 v0, 16','s_mov_b32 exec_lo, 16',1),
    'PC-order':e.replace('s_add_u32 s4, s4, safe@rel32@lo+4','s_addc_u32 s4, s4, safe@rel32@lo+4',1),
   })
  for label,changed in mutations.items():
   assert changed!=e,(kind,label,'mutation not applied');reject(lambda:positive_equal(b,changed,root,frame,calls))
  reject(lambda:rejected_equal(b.encode(),e.encode()))
  rows.append({'case':kind+'-'+target,'baseline_sha256':sha(before),'enabled_sha256':sha(after),'actual_positive_passed':True,'mutations_rejected':len(mutations)+1})
out=p/'post-audit-extensions-v2-tests.json';assert not out.exists();out.write_text(json.dumps({'cases':rows,'hardware_validated':False,'accepted_compiler':False,'script_sha256':sha(Path(__file__)),'checker_sha256':sha(p/'post_audit_extensions_v2.py'),'frame_helper_sha256':sha(p/'post_audit_v2.py')},indent=2)+'\n');print('PASS6actualpositives',sum(row['mutations_rejected'] for row in rows),'mutations',sha(out))
