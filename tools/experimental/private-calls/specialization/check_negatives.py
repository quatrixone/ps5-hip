from pathlib import Path
import subprocess,json,re,hashlib
out=Path(__file__).resolve().parent
prefix='target triple = "amdgcn-amd-amdhsa"\n'
fns='''
define internal i32 @ordinary(ptr nocapture %p) noinline norecurse {
 %x=load i32, ptr %p, align 4
 ret i32 %x
}
define internal i32 @volatile_fn(ptr nocapture %p) noinline norecurse {
 %x=load volatile i32, ptr %p, align 4
 store volatile i32 %x, ptr %p, align 4
 ret i32 %x
}
define internal i32 @atomic_fn(ptr nocapture %p) noinline norecurse {
 %x=load atomic i32, ptr %p monotonic, align 4
 ret i32 %x
}
define internal i32 @int_cast(ptr nocapture %p) noinline norecurse {
 %x=ptrtoint ptr %p to i32
 ret i32 %x
}
define internal i32 @escape(ptr %p,ptr %dst) noinline norecurse {
 store ptr %p,ptr %dst
 ret i32 0
}
define internal i32 @recursive(ptr nocapture %p) noinline {
 %x=call i32 @recursive(ptr %p)
 ret i32 %x
}
declare i32 @external(ptr)
declare i32 @vararg(ptr,...)
define i32 @cases(ptr addrspace(5) %p,ptr addrspace(1) %g,ptr %fn,ptr %dst,i1 %which) {
 %a=addrspacecast ptr addrspace(5) %p to ptr
 %b=addrspacecast ptr addrspace(1) %g to ptr
 %c=select i1 %which,ptr %a,ptr %b
 %yes=call i32 @ordinary(ptr %a)
 %global=call i32 @ordinary(ptr %b)
 %ambiguous=call i32 @ordinary(ptr %c)
 %volatile=call i32 @volatile_fn(ptr %a)
 %atomic=call i32 @atomic_fn(ptr %a)
 %integer=call i32 @int_cast(ptr %a)
 %escapes=call i32 @escape(ptr %a,ptr %dst)
 %recursive=call i32 @recursive(ptr %a)
 %external=call i32 @external(ptr %a)
 %indirect=call i32 %fn(ptr %a)
 %varargs=call i32 (ptr,...) @vararg(ptr %a,ptr %a)
 ret i32 %yes
}
'''
source=out/'negatives.ll';source.write_text(prefix+fns)
cmd=['opt-18','-load-pass-plugin',str(out/'PrivateSpecialize.so'),'-passes=research-private-specialize,function(infer-address-spaces,instcombine,simplifycfg),verify','-S',str(source),'-o',str(out/'negatives-out.ll')]
r=subprocess.run(cmd,capture_output=True,text=True);(out/'negatives.log').write_text(r.stdout+r.stderr);r.check_returncode();text=(out/'negatives-out.ll').read_text()
assert '@ordinary.private5(ptr addrspace(5)' in text and '@volatile_fn.private5(ptr addrspace(5)' in text
for name in ['atomic_fn','int_cast','escape','recursive','external','vararg']:assert name+'.private5' not in text
assert len(re.findall(r'call i32 @ordinary\(ptr',text))==2 and 'call i32 %fn(ptr' in text
vol=re.search(r'define internal i32 @volatile_fn.private5[^\n]*\n(.*?)^}',text,re.M|re.S)[1]
assert len(re.findall('load volatile i32, ptr ',vol))==1 and len(re.findall('store volatile i32 .* ptr ',vol))==1 and 'addrspacecast ptr addrspace(5)' in vol
# Historical v1 equality belongs to the original receipt, not this fresh run.
report={'pass':'verified','command':cmd,'positive_private_specialization':True,'generic_global_and_ambiguous_calls_preserved':True,'indirect_call_preserved':True,'rejected':['atomic','pointer-to-integer','pointer escape','recursive','external','vararg'],'volatile_clone_exists_but_memory_stays_generic':True,'volatile_operation_count_preserved_per_clone':{'load':1,'store':1},'historical_v1_output_equality_retested':False,'limits':'Small verifier/structural checks, not formal semantics proof or executable GPU equivalence; original generic fallback remains, including unsupported flat operations.'}
(out/'negative-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

