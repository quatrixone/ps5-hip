from pathlib import Path
import ast,hashlib,json,shutil,tempfile
from frame_audit_v3 import inspect_case
out=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
measured=read(out/'measured-contracts.json');receipt=read(out/'compile-receipt.json');passed=[];rejected=[]
for item in receipt['cases']:
 case=item['case'];folder=out/case['id'];inspect_case(case,folder,False,measured);passed.append(case['id'])
 if case['kind']!='ordinary':continue
 mutations=[('old-private288','output.s','.amdhsa_private_segment_fixed_size 304','.amdhsa_private_segment_fixed_size 288'),('wrong-root-frame','output.s','s_movk_i32 s32, 0x1200','s_movk_i32 s32, 0x1000'),('wrong-middle-increment','output.s','s_addk_i32 s32, 0x1400','s_addk_i32 s32, 0x1200'),('wrong-middle-decrement','output.s','s_addk_i32 s32, 0xec00','s_addk_i32 s32, 0xee00'),('missing-middle-call','isa.txt','s_swappc_b64 s[30:31]','s_nop'),('wrong-wait','isa.txt','s_waitcnt vmcnt(2)','s_waitcnt vmcnt(3)'),('wrong-abi-restore','isa.txt','buffer_load_dword v40, off, s[0:3], s33 offset:4','buffer_load_dword v40, off, s[0:3], s33 offset:8'),('missing-observable-add','isa.txt','v_add3_u32 v0,','v_add3_u32 v7,'),('lost-leaf-store','isa.txt','buffer_store_dword v3, v2, s[0:3], 0 offen','buffer_store_dword v3, v4, s[0:3], 0 offen')]
 for name,file,before,after in mutations:
  original=(folder/file).read_text()
  if before not in original:
   # Locate leaf store operands in this profile; preserve mutation semantics.
   assert name=='lost-leaf-store'
   import re
   m=re.search(r'buffer_store_dword (v\d+), (v\d+), s\[0:3\], 0 offen',original[original.index('<_Z9call_leafPjj.private5>:'):]);assert m
   before=m[0];after=before.replace('s[0:3]','s[4:7]')
  with tempfile.TemporaryDirectory(prefix='frame-audit-') as temporary:
   d=Path(temporary)
   for n in ['output.s','isa.txt','metadata.txt']:shutil.copyfile(folder/n,d/n)
   (d/file).write_text(original.replace(before,after))
   try:inspect_case(case,d,False,measured)
   except (AssertionError,StopIteration,KeyError,ValueError):rejected.append(case['id']+':'+name)
   else:raise AssertionError('mutation accepted '+case['id']+':'+name)
for n in ['frame_audit_v3.py','run_matrix_v3.py']:ast.parse((out/n).read_text())
report={'classification':'saved clean baseline frame audit; no additional compiler invocation','actual_profiles_passed':passed,'mutations_rejected':rejected,'mutation_count':len(rejected),'measured_receipt_sha256':sha(out/'measured-contracts.json'),'auditor_sha256':sha(out/'frame_audit_v3.py'),'runner_sha256':sha(out/'run_matrix_v3.py'),'test_sha256':sha(Path(__file__)),'hardware_validated':False}
p=out/'frame-audit-v3-tests.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(passed),'profiles',len(rejected),'mutations');print(sha(p))
