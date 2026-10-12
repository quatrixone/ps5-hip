"""Synthetic output mutations, backed by six actual baseline assembly controls."""
from pathlib import Path
import json,hashlib
from post_audit_extensions_v1 import remove_exact_init,positive_equal,rejected_equal
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
rows=[]
for target in ('gfx1013','gfx1030'):
 for kind,source,root,frame,calls in [
  ('alias',p/'negative-runs/baseline'/('alias-'+target)/'output.s','negative','s_movk_i32 s32, 0x400',1),
  ('stress',p/'stress-runs/baseline-late-callee-unbounded'/target/'output.s','stress','s_mov_b32 s32, 0',2),
  ('MIR',p.parent/'g3-private-backend-mir-controls/runs-v2/baseline'/('unchanged-'+target)/'output.s','stress','s_mov_b32 s32, 0',2),
 ]:
  baseline=source.read_text();synthetic=remove_exact_init(baseline,root,frame,calls)
  positive_equal(baseline,synthetic,root,frame,calls)
  mutations={
   'callee-state-write':synthetic.replace('\ts_setpc_b64','\ts_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s6\n\ts_setpc_b64',1),
   'callee-address':synthetic.replace('buffer_load_dword v0, v0' if kind=='alias' else 'buffer_load_dword v1, v0','buffer_load_dword v0, v1' if kind=='alias' else 'buffer_load_dword v1, v2',1),
   'resource-count':synthetic.replace('.amdhsa_user_sgpr_count 8','.amdhsa_user_sgpr_count 6',1),
   'stack-accounting':synthetic.replace('; ScratchSize: 0','; ScratchSize: 4',1),
   'call-target':synthetic.replace('safe@rel32@lo' if kind=='alias' else 'leaf@rel32@lo','unknown@rel32@lo',1),
   'frame':synthetic.replace(frame,'s_mov_b32 s32, 123',1),
   'code-length':synthetic.replace('; codeLenInByte = 20' if kind=='alias' else '; codeLenInByte = 40','; codeLenInByte = 16',1),
  }
  for name,changed in mutations.items():
   assert changed!=synthetic,(kind,name,'mutation not applied')
   try:positive_equal(baseline,changed,root,frame,calls)
   except AssertionError:pass
   else:raise AssertionError((kind,name,'mutation accepted'))
  try:positive_equal(baseline,baseline,root,frame,calls)
  except AssertionError:pass
  else:raise AssertionError('unchanged enabled output accepted')
  rejected_equal(baseline.encode(),baseline.encode())
  try:rejected_equal(baseline.encode(),synthetic.encode())
  except AssertionError:pass
  else:raise AssertionError('rejected changed code accepted')
  rows.append({'fixture':str(source),'sha256':sha(source),'synthetic_positive_passed':True,'negative_mutations_rejected':9})
report={'actual_enabled_tested':False,'cases':rows,'recipe_sha256':sha(Path(__file__)),'checker_sha256':sha(p/'post_audit_extensions_v1.py'),'hardware_validated':False}
output=p/'post-audit-extensions-v1-tests.json';assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print('PASS six baseline controls,54 mutation rejects',sha(output))
