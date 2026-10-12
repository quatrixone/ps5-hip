from pathlib import Path
import hashlib,json,subprocess
out=Path(__file__).resolve().parent;dest=out/'negative-inputs';dest.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prefix='''target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-i64:64-n32:64-S32-A5-G1"
target triple = "amdgcn-amd-amdhsa"
'''
safe='''define internal fastcc i32 @safe(ptr addrspace(5) nocapture %p) noinline norecurse {
 %v=load i32, ptr addrspace(5) %p, align 4
 ret i32 %v
}
'''
bad='''define internal fastcc i32 @bad(ptr addrspace(5) %p) noinline norecurse {
 %g=addrspacecast ptr addrspace(5) %p to ptr
 %v=load volatile i32, ptr %g, align 4
 ret i32 %v
}
'''
cases={
 'external':('declare i32 @unknown(ptr addrspace(5))\n','%r=call i32 @unknown(ptr addrspace(5) %p)', ''),
 'indirect':('', '%r=call i32 %fn(ptr addrspace(5) %p)', ''),
 'alias':(safe+'@alias = internal alias i32 (ptr addrspace(5)), ptr @safe\n', '%r=call fastcc i32 @alias(ptr addrspace(5) %p)', ''),
 'dynamic':('''define internal fastcc i32 @dynamic(ptr addrspace(5) %p, i32 %n) noinline norecurse {
 %a=alloca i32, i32 %n, align 4, addrspace(5)
 store volatile i32 7, ptr addrspace(5) %a, align 4
 %v=load volatile i32, ptr addrspace(5) %a, align 4
 ret i32 %v
}
''', '%r=call fastcc i32 @dynamic(ptr addrspace(5) %p, i32 %n)', ''),
 'realigned':('''define internal fastcc i32 @aligned(ptr addrspace(5) %p) noinline norecurse {
 %a=alloca i32, align 4096, addrspace(5)
 store volatile i32 7, ptr addrspace(5) %a, align 4096
 %v=load volatile i32, ptr addrspace(5) %a, align 4096
 ret i32 %v
}
''', '%r=call fastcc i32 @aligned(ptr addrspace(5) %p)', ''),
 'inlineasm':('''define internal fastcc i32 @asm_body(ptr addrspace(5) %p) noinline norecurse {
 %v=call i32 asm sideeffect "s_mov_b32 $0, 7", "=s"()
 ret i32 %v
}
''', '%r=call fastcc i32 @asm_body(ptr addrspace(5) %p)', ''),
 'hwreg':('''define internal fastcc i32 @hwreg(ptr addrspace(5) %p) noinline norecurse {
 %v=call i32 @llvm.amdgcn.s.getreg(i32 63492)
 ret i32 %v
}
declare i32 @llvm.amdgcn.s.getreg(i32 immarg)
''', '%r=call fastcc i32 @hwreg(ptr addrspace(5) %p)', ''),
 'recursive':('''define internal fastcc i32 @rec(ptr addrspace(5) %p, i32 %n) noinline {
 %done=icmp eq i32 %n, 0
 br i1 %done, label %exit, label %again
again:
 %m=sub i32 %n, 1
 %x=call fastcc i32 @rec(ptr addrspace(5) %p, i32 %m)
 %v=add i32 %x, 1
 ret i32 %v
exit:
 %loaded=load i32, ptr addrspace(5) %p
 ret i32 %loaded
}
''','%r=call fastcc i32 @rec(ptr addrspace(5) %p, i32 %n)', ''),
 'forgedattrs':(bad,'%r=call fastcc i32 @bad(ptr addrspace(5) %p)', '"amdgpu-calls"="false" "amdgpu-proven-buffer-only"="true" "ps5hip-proven-buffer-only"="true"'),
 'deepbad':(bad+'''define internal fastcc i32 @level2(ptr addrspace(5) %p) noinline norecurse {
 %r=call fastcc i32 @bad(ptr addrspace(5) %p)
 ret i32 %r
}
define internal fastcc i32 @level1(ptr addrspace(5) %p) noinline norecurse {
 %r=call fastcc i32 @level2(ptr addrspace(5) %p)
 ret i32 %r
}
''','%r=call fastcc i32 @level1(ptr addrspace(5) %p)', '')}
report=[]
for name,(functions,call,attrs) in cases.items():
 text=prefix+functions+f'''define amdgpu_kernel void @negative(ptr addrspace(1) %out, ptr %fn, i32 %n) {attrs} {{
 %p=alloca i32, align 16, addrspace(5)
 store i32 11, ptr addrspace(5) %p, align 16
 {call}
 store i32 %r, ptr addrspace(1) %out, align 4
 ret void
}}
'''
 path=dest/(name+'.ll');path.write_text(text)
 subprocess.run(['opt-18','-passes=verify','-disable-output',str(path)],check=True)
 report.append({'name':name,'input':str(path.relative_to(out)),'sha256':sha(path),'patched_expectation':'negative root rejected; flat initialization retained','link_required':False})
(out/'negative-matrix.json').write_text(json.dumps({'cases':report,'targets':['gfx1013','gfx1030'],'optimization':'O2','hardware_validated':False,'source':'fresh isolated IR; original fixtures untouched','limits':'Some backend targets may reject unsupported IR before code generation; preserve that as unsupported, never pass it as proof rejection. Alias lowering must be inspected rather than assumed to preserve a machine alias.','script_sha256':sha(Path(__file__))},indent=2)+'\n')
print('Prepared10freshnegativeIRcases')
