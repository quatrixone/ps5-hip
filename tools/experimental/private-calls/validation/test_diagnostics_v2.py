from pathlib import Path
import argparse,hashlib,json,re
from diagnostics_v2 import check_diagnostics,linked_functions,DEAD_ORDINARY_ORIGINALS
p=argparse.ArgumentParser();p.add_argument('--records',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--output',type=Path);args=p.parse_args()
root=args.records;case='ordinary-gfx1013-O1-original'
paths={'log':root/'runs/patched-on'/case/'asm.log','source':root/'inputs/ordinary-gfx1013-O1.ll','linked':root/'runs/patched-on'/case/'metadata.txt','baseline':root/'runs/baseline'/case/'metadata.txt'}
data={k:v.read_text() for k,v in paths.items()}
required=['nested_private','_Z11call_middlePjj.private5','_Z9call_leafPjj.private5'];expected=dict.fromkeys(required,'proven')
def check(d,allow=DEAD_ORDINARY_ORIGINALS):return check_diagnostics(d['log'],d['source'],expected,required,linked_metadata=d['linked'],baseline_metadata=d['baseline'],allowed_dead=allow)
actual=check(data);assert len(actual['linked_functions'])==11
mutations=[]
def reject(name,field,text):
 d=dict(data);d[field]=text
 try:check(d)
 except AssertionError:mutations.append(name)
 else:raise AssertionError('accepted mutation: '+name)
for name in actual['linked_functions']:
 reject('omit-inventory:'+name,'log',data['log'].replace('buffer-call-proof inventory '+name+' present\n',''))
 value=actual['results'][name]
 reject('omit-result:'+name,'log',data['log'].replace('buffer-call-proof result '+name+' '+value+'\n',''))
 reject('mark-missing:'+name,'log',data['log'].replace('inventory '+name+' present','inventory '+name+' missing'))
 # Omitting both records for an unrelated function must still be rejected.
 reject('omit-both:'+name,'log',data['log'].replace('buffer-call-proof inventory '+name+' present\n','').replace('buffer-call-proof result '+name+' '+value+'\n',''))
 for field in ['linked','baseline']:
  blocks=list(re.finditer(r'(?ms)^  Symbol \{\n(.*?)^  \}\n',data[field]));block=next(m for m in blocks if re.search(r'^    Name: '+re.escape(name)+r' \(',m[1],re.M))
  reject('omit-'+field+':'+name,field,data[field][:block.start()]+data[field][block.end():])
for name in DEAD_ORDINARY_ORIGINALS:
 reject('noninternal:'+name,'source',re.sub(r'(?m)^define internal([^\n]*@'+re.escape(name)+r'\()',r'define external\1',data['source']))
 reject('escaping-reference:'+name,'source',data['source']+'\n@keep = global ptr @'+name+'\n')
 reject('dead-inventory:'+name,'log','buffer-call-proof inventory '+name+' missing\n'+data['log'])
 reject('extra-source:'+name,'source',data['source']+'\ndefine internal void @extra_dead() {\n ret void\n}\n')
reject('late-inventory','log',data['log']+'buffer-call-proof inventory late missing\n')
reject('inventory-after-register-check','log',data['log'].replace('buffer-call-proof register-use _Z13volatile_leafPVjj explicit 1 with-mask 1','buffer-call-proof register-use _Z13volatile_leafPVjj explicit 1 with-mask 1\nbuffer-call-proof inventory late missing'))
reject('duplicate-result','log',data['log']+'buffer-call-proof result nested_private proven\n')
reject('wrong-root-result','log',data['log'].replace('result nested_private proven','result nested_private rejected'))
reject('malformed-record','log',data['log']+'buffer-call-proof inventory bad bad extra\n')
reject('unknown-machine','log','buffer-call-proof inventory extra present\n'+data['log']+'buffer-call-proof result extra proven\n')
try:check(data,())
except AssertionError:mutations.append('no-dead-exception')
else:raise AssertionError('dead exceptions not enforced')
try:check(data,DEAD_ORDINARY_ORIGINALS|{'extra'})
except AssertionError:mutations.append('unapproved-dead-exception')
else:raise AssertionError('unapproved exception')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'classification':'saved actual V3 positive record diagnostic regression; no compiler rerun','hardware_validated':False,'positive_record':actual,'rejected_mutations':mutations,'rejected_count':len(mutations),'inputs':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},'parser_sha256':sha(Path(__file__).with_name('diagnostics_v2.py')),'test_sha256':sha(Path(__file__))}
output=args.output or Path(__file__).with_name('diagnostic-parser-v2-tests.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print('PASS real record; rejected',len(mutations),'mutations');print(sha(output))
