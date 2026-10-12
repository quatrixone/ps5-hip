from pathlib import Path
import hashlib,json,subprocess
import argparse
p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=96);p.add_argument('--label',default='late-spill');args=p.parse_args()
out=Path(__file__).resolve().parent; dest=out/'stress-inputs';dest.mkdir(exist_ok=True)
prefix='''target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-i64:64-n32:64-S32-A5-G1"
target triple = "amdgcn-amd-amdhsa"
define internal fastcc i32 @leaf(ptr addrspace(5) %p) noinline norecurse {
 %a=load volatile i32, ptr addrspace(5) %p, align 4
 %b=add i32 %a, 3
 store volatile i32 %b, ptr addrspace(5) %p, align 4
 ret i32 %b
}
'''
def body(kernel, count):
    name='stress' if kernel else 'middle'
    signature='amdgpu_kernel void' if kernel else 'internal fastcc i32'
    lines=[f'define {signature} @{name}(ptr addrspace(1) %out, ptr addrspace(1) %in) noinline norecurse "amdgpu-num-vgpr"="40" {{']
    if not kernel:
        lines+=[' %frame=alloca [8192 x i32], align 16, addrspace(5)', ' %p=getelementptr [8192 x i32], ptr addrspace(5) %frame, i32 0, i32 8191', ' store volatile i32 17, ptr addrspace(5) %p, align 4']
    for i in range(count):
        lines += [f' %ptr{i}=getelementptr i32, ptr addrspace(1) %in, i32 {i}', f' %v{i}=load volatile i32, ptr addrspace(1) %ptr{i}, align 4']
    lines += [' %r=call fastcc i32 @middle(ptr addrspace(1) %out, ptr addrspace(1) %in)' if kernel else ' %r=call fastcc i32 @leaf(ptr addrspace(5) %p)']
    for i in range(count):
        lines += [f' %x{i}=xor i32 %v{i}, %r']
        if i: lines += [f' %s{i}=add i32 '+('%x0' if i==1 else f'%s{i-1}')+f', %x{i}']
    lines += [f' store i32 %s{count-1}, ptr addrspace(1) %out, align 4', ' ret void'] if kernel else [f' ret i32 %s{count-1}']
    return '\n'.join(lines+['}'])+'\n'
text=prefix+body(False,args.count)+body(True,args.count)
path=dest/(args.label+'.ll');assert not path.exists();path.write_text(text)
subprocess.run(['opt-18','-passes=verify','-disable-output',str(path)],check=True)
(out/(args.label+'-input.json')).write_text(json.dumps({'file':str(path.relative_to(out)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'purpose':'Force live values across nested calls and large frame offsets; actual late spill coverage requires measured pre/post-PEI differences, not source intent.','hardware_validated':False},indent=2)+'\n')
