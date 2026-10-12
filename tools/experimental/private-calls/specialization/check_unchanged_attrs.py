from pathlib import Path
import hashlib,json,re,subprocess
out=Path(__file__).resolve().parent
plugin=out/'PrivateSpecialize.so'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
text='target triple = "amdgcn-amd-amdhsa"\n'
cases={'integer_inreg':('i32 inreg','i32'),'integer_zext':('i8 zeroext','i8'),'integer_sext':('i8 signext','i8'),'pointer_inreg':('ptr inreg','ptr'),'pointer_nest':('ptr nest','ptr'),'pointer_byref':('ptr byref(i32)','ptr'),'pointer_byval':('ptr byval(i32)','ptr')}
for name,(attrs,ty) in cases.items():
 text+=f'''define internal fastcc zeroext i32 @{name}(ptr nocapture %p, {attrs} %other) noinline norecurse {{
 %x=load i32, ptr %p, align 4
 ret i32 %x
}}
define i32 @caller_{name}(ptr addrspace(5) %p, {ty} %other) {{
 %a=addrspacecast ptr addrspace(5) %p to ptr
 %r=call fastcc zeroext i32 @{name}(ptr %a, {attrs} %other)
 ret i32 %r
}}
'''
source=out/'unchanged-attrs.ll';source.write_text(text)
cmd=['opt-18','-load-pass-plugin',str(plugin),'-passes=research-private-specialize,verify','-S',str(source),'-o',str(out/'unchanged-attrs-out.ll')]
before=sha(plugin);r=subprocess.run(cmd,capture_output=True,text=True);(out/'unchanged-attrs.log').write_text(r.stdout+r.stderr);r.check_returncode()
result=(out/'unchanged-attrs-out.ll').read_text();checked={}
for name,(attrs,ty) in cases.items():
 definition=re.search(r'^define internal fastcc zeroext i32 @'+name+r'\.private5\(([^\n]*)',result,re.M)
 assert definition and attrs+' %' in definition[1],(name,definition[1] if definition else None)
 call=re.search(r'call fastcc zeroext i32 @'+name+r'\.private5\(([^\n]*)',result)
 assert call and attrs+' %other' in call[1],name
 checked[name]={'clone':definition[0],'call':call[0]}
assert sha(plugin)==before
receipt={'plugin_sha256':before,'command':cmd,'exit_code':r.returncode,'all_unchanged_parameter_attrs_preserved':True,'callingconv_return_zeroext_preserved':True,'cases':checked,'source_sha256':sha(source),'output_sha256':sha(out/'unchanged-attrs-out.ll'),'script_sha256':sha(Path(__file__))}
(out/'unchanged-attrs-review.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS seven unchanged-argument ABI attributes and fastcc/zeroext return kept in clone+call')
print(sha(out/'unchanged-attrs-review.json'))
