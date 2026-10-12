from pathlib import Path
import subprocess,json,re,hashlib
out=Path(__file__).resolve().parent
cases=[]
for attr in ['nest','inreg','byref(i32)','nonnull','align 4','dereferenceable(4)','dereferenceable_or_null(4)','noalias','"research-unreviewed"']:
 for location in ['definition','callsite','both']:
  formal=(attr+' ') if location in ['definition','both'] else ''
  actual=(attr+' ') if location in ['callsite','both'] else ''
  source=f'''target triple = "amdgcn-amd-amdhsa"
define internal i32 @f(ptr {formal}nocapture %p) noinline norecurse {{
 %v=load i32,ptr %p,align 4
 ret i32 %v
}}
define i32 @caller(ptr addrspace(5) %p) {{
 %g=addrspacecast ptr addrspace(5) %p to ptr
 %r=call i32 @f(ptr {actual}%g)
 ret i32 %r
}}
'''
  name=f'attr-{len(cases):02d}';(out/(name+'.ll')).write_text(source)
  cmd=['opt-18','-load-pass-plugin',str(out/'PrivateSpecialize.so'),'-passes=research-private-specialize,verify','-S',str(out/(name+'.ll')),'-o',str(out/(name+'-out.ll'))]
  r=subprocess.run(cmd,capture_output=True,text=True);(out/(name+'.log')).write_text(r.stdout+r.stderr)
  r.check_returncode();text=(out/(name+'-out.ll')).read_text();assert 'f.private5' not in text
  cases.append({'attribute':attr,'location':location,'result':'unchanged generic callee'})
for kind in ['notail','tail']:
 source='''target triple = "amdgcn-amd-amdhsa"
define internal i32 @f(ptr nocapture %p) noinline norecurse {
 %v=load i32,ptr %p,align 4
 ret i32 %v
}
define i32 @caller(ptr addrspace(5) %p) {
 %g=addrspacecast ptr addrspace(5) %p to ptr
 %r=TAIL call i32 @f(ptr %g)
 ret i32 %r
}
'''.replace('TAIL',kind)
 (out/(kind+'.ll')).write_text(source)
 cmd=['opt-18','-load-pass-plugin',str(out/'PrivateSpecialize.so'),'-passes=research-private-specialize,verify','-S',str(out/(kind+'.ll')),'-o',str(out/(kind+'-out.ll'))]
 r=subprocess.run(cmd,capture_output=True,text=True);(out/(kind+'.log')).write_text(r.stdout+r.stderr);r.check_returncode();assert kind+' call i32 @f.private5(ptr addrspace(5)' in (out/(kind+'-out.ll')).read_text()
 cases.append({'tail_kind':kind,'result':'specialized with exact tail kind preserved'})
(out/'attribute-results.json').write_text(json.dumps({'cases':cases,'count':len(cases)},indent=2)+'\n');print(len(cases),'attribute/tail-kind cases passed')
