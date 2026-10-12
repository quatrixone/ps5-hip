"""No compiler execution. Exact rejected codegen and narrow positive assembly deltas."""
from pathlib import Path
import argparse,hashlib,json,re
from diagnostics import check_diagnostics
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
INIT=['s_add_u32 s6, s6, s9','s_addc_u32 s7, s7, 0',
      's_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s6',
      's_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s7']

def remove_exact_init(assembly, root, frame, calls):
 lines=assembly.splitlines(keepends=True)
 positions=[i for i,line in enumerate(lines) if re.match(r'^'+re.escape(root)+r':(?:\s|$)',line)]
 assert len(positions)==1,'missing/duplicate root'
 start=positions[0]
 instructions=[i for i in range(start+1,len(lines)) if re.match(r'^\s+[a-z][a-z0-9_]*\s',lines[i])]
 first=instructions[:5]
 assert [lines[i].strip() for i in first]==[INIT[0],frame,*INIT[1:]],'unexpected exact initialization slice'
 removed={first[i] for i in (0,2,3,4)}
 sizes=[i for i in range(start+1,len(lines)) if re.fullmatch(r'; codeLenInByte = \d+\n?',lines[i])]
 assert len(sizes)==1,'root must be final function with exactly one code-length comment'
 size=sizes[0];old=int(re.search(r'\d+',lines[size])[0]);assert old>=16
 lines[size]=lines[size].replace(str(old),str(old-16),1)
 result=''.join(line for i,line in enumerate(lines) if i not in removed)
 # All machine instructions across every function, not just root.
 ops=[line.strip().split(';',1)[0].strip() for line in result.splitlines() if re.match(r'^\s+[a-z][a-z0-9_]*\s',line)]
 assert sum(op.startswith('s_swappc_b64 ') for op in ops)==calls,'call count changed'
 assert not any(re.search(r'^(?:flat_|scratch_|s_getreg|s_setreg)|\b(?:flat_scr\w*|flat_scratch\w*|src_private_\w*)\b',op) for op in ops),'forbidden state/flat/aperture instruction'
 return result

def positive_equal(baseline,enabled,root,frame,calls):
 expected=remove_exact_init(baseline,root,frame,calls)
 assert expected==enabled,'instructions, symbols, directives, resource fields or non-size comments changed beyond exact four-op slice'

def rejected_equal(baseline,enabled):
 assert baseline==enabled,'rejected codegen changed'

def verified_report(folder, mode, count):
 report=json.loads((folder/'result.json').read_text())
 assert report['mode']==mode and len(report['cases'])==count
 for case in report['cases']:
  name=case.get('id',case.get('name',case.get('target')))
  assert name and Path(name).name==name
  for file,digest in case['files'].items():
   assert Path(file).name==file and sha(folder/name/file)==digest,('changed receipt artifact',folder,name,file)
 return report

def audit(validation,mir,output):
 assert not output.exists(),'output already exists'
 reports={};rows=[];sources={};baseline_hash=None;patched_hash=None
 def triplet(relative,count,identity):
  nonlocal baseline_hash,patched_hash
  root=mir if relative=='MIR' else validation
  stem='runs-v2' if relative=='MIR' else relative
  suffix='-late-callee-unbounded' if relative=='stress-small' else ''
  if relative=='stress-small':stem='stress-runs'
  folders=[root/stem/(mode+suffix) for mode in ('baseline','patched-off','patched-on')]
  loaded=[verified_report(folder,mode,count) for folder,mode in zip(folders,('baseline','patched-off','patched-on'))]
  b,o,e=loaded
  if baseline_hash is None:baseline_hash=b['compiler_sha256'];patched_hash=o['compiler_sha256']
  assert b['compiler_sha256']==baseline_hash and o['compiler_sha256']==e['compiler_sha256']==patched_hash
  assert len({report[identity] for report in loaded})==1,('fixture identity mismatch',relative)
  for folder in folders:reports[str(folder/'result.json')]=sha(folder/'result.json')
  names=lambda report:[c.get('id',c.get('name',c.get('target'))) for c in report['cases']]
  assert names(b)==names(o)==names(e) and len(set(names(b)))==count
  return folders,loaded
 def compare(folders,name,positive=None,object_file=True):
  b,o,e=[folder/name for folder in folders]
  files=['output.s','output.o'] if object_file else ['output.s']
  for file in files:assert (b/file).read_bytes()==(o/file).read_bytes(),('default-off changed',name,file)
  if positive:
   positive_equal((b/files[0]).read_text(),(e/files[0]).read_text(),*positive)
  else:
   for file in files:rejected_equal((b/file).read_bytes(),(e/file).read_bytes())
  rows.append({'case':name,'suite':str(folders[0].parent),'comparison':'exact four-init-op removal, root code-length comment -16; all other text identical' if positive else 'byte-identical assembly/object' if object_file else 'byte-identical assembly','files':{str(path/file):sha(path/file) for path in (b,o,e) for file in files}})
 def diagnostic(folder,log,source,expected,present):
  sources[str(source)]=sha(source)
  return check_diagnostics((folder/log).read_text(),source.read_text().replace('\n  define','\ndefine'),expected,present)
 try:
  folders,loaded=triplet('runs',24,'matrix_sha256')
  matrix=json.loads((validation/'matrix.json').read_text())
  assert loaded[0]['matrix_sha256']==sha(validation/'matrix.json')
  for case in loaded[2]['cases']:
   name=case['id']
   if name.startswith('ordinary-'):continue # Covered by stronger linked-ELF post_audit_v1.
   assert name.startswith(('generic-','volatile-'))
   expected={root['root']:'rejected' for root in case['roots']}
   reachable={name for root in case['roots'] for name in root['reachable']}
   check_diagnostics((folders[2]/name/'asm.log').read_text(),'',expected,reachable)
   assert all(root['inventory'][root['root']]['flat_init']==2 for root in case['roots'])
   compare(folders,name)
  folders,loaded=triplet('negative-runs',20,'matrix_sha256')
  matrix=json.loads((validation/'negative-matrix.json').read_text());assert loaded[0]['matrix_sha256']==sha(validation/'negative-matrix.json')
  fixtures={c['name']:validation/c['input'] for c in matrix['cases']}
  for case in loaded[2]['cases']:
   name=case['name'];kind=name.rsplit('-',1)[0]
   if kind=='dynamic':
    assert all(next(c for c in report['cases'] if c['name']==name)['compiler_failed'] for report in loaded)
    continue # Explicit unsupported coverage, never a passed proof rejection.
   assert case['compiler_failed'] is None
   positive=('negative','s_movk_i32 s32, 0x400',1) if kind=='alias' else None
   diagnostic(folders[2]/name,'assembly.log',fixtures[kind],{'negative':'proven' if positive else 'rejected'},['negative','safe'] if positive else ['negative'])
   compare(folders,name,positive)
  for small in (False,True):
   relative='stress-small' if small else 'stress-runs';folders,loaded=triplet(relative,2,'source_sha256')
   source=validation/'stress-inputs'/('late-callee-unbounded.ll' if small else 'late-spill.ll')
   assert loaded[0]['source_sha256']==sha(source)
   for case in loaded[2]['cases']:
    name=case['target'];expected={key:'proven' for key in ('stress','middle','leaf')} if small else {'stress':'rejected'}
    diagnostic(folders[2]/name,'output.s.log',source,expected,['stress','middle','leaf'])
    compare(folders,name,('stress','s_mov_b32 s32, 0',2) if small else None)
  folders,loaded=triplet('MIR',10,'matrix_sha256');matrix=json.loads((mir/'matrix.json').read_text())
  assert all(report['all_controls_passed'] for report in loaded) and loaded[0]['matrix_sha256']==sha(mir/'matrix.json')
  fixtures={c['name']:c for c in matrix['cases']}
  for case in loaded[2]['cases']:
   name=case['name'];fixture=fixtures[name];positive=fixture['kind']=='unchanged'
   assert case['stages']=={'pre-pei':0,'assembly':0}
   diagnostic(folders[2]/name,'assembly.log',mir/fixture['file'],{'stress':'proven' if positive else 'rejected'},['stress','middle','leaf'])
   compare(folders,name,('stress','s_mov_b32 s32, 0',2) if positive else None,False)
  assert len(rows)==48
  report={'extensions_passed':True,'accepted_compiler':False,'hardware_validated':False,'cases':rows,'compiler_sha256':{'baseline':baseline_hash,'patched':patched_hash},'receipts':reports,'sources':sources,'script_sha256':sha(Path(__file__)),'limits':['Ordinary linked-ELF audit is separate and required.','Dynamic alloca remains unsupported before proof, not a passed rejection.','Source alias canonicalizes to direct safe call; only separate MIR tests exercise residual alias.','Rejected root diagnostics alone do not establish the rejection reason or presence of every intended source callee.','Positive assembly preserves all symbols, operands, directives, resource fields and comments except exact root code-length decrement; object-byte comparison is reserved for rejected cases.']}
 except Exception as error:
  output.write_text(json.dumps({'extensions_passed':False,'error':repr(error),'cases':rows,'receipts':reports},indent=2)+'\n');raise
 output.write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(rows),sha(output))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,default=HERE);p.add_argument('--mir',type=Path,default=HERE.parent/'g3-private-backend-mir-controls');p.add_argument('--output',type=Path,required=True);args=p.parse_args();audit(args.validation,args.mir,args.output)
