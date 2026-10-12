from pathlib import Path
import hashlib,json,re,subprocess
out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plugin=out/'PrivateSpecialize.so';before=sha(plugin);reports=[]

def execute(name,functions,arguments,calls):
    source='target triple = "amdgcn-amd-amdhsa"\n'+functions
    source+='define i32 @caller(ptr addrspace(5) %p, ptr addrspace(1) %global) {\n %g=addrspacecast ptr addrspace(5) %p to ptr\n %h=addrspacecast ptr addrspace(1) %global to ptr\n'
    source+='\n'.join(calls)+'\n ret i32 7\n}\n'
    path=out/(name+'.ll');path.write_text(source)
    outputs=[];commands=[]
    for repeat in (1,2):
        dest=out/(name+f'-out{repeat}.ll')
        cmd=['opt-18','-load-pass-plugin',str(plugin),'-passes=research-private-specialize,verify','-S',str(path),'-o',str(dest)]
        r=subprocess.run(cmd,capture_output=True,text=True);(out/(name+f'-{repeat}.log')).write_text(r.stdout+r.stderr);r.check_returncode()
        outputs.append(dest.read_text());commands.append(cmd)
    assert outputs[0]==outputs[1],name
    text=outputs[0]
    clones=re.findall(r'^define internal fastcc zeroext i32 @(f\d+\.private5[^ (]*)\(([^\n]*)',text,re.M)
    counts={}
    for clone,_ in clones:
        origin=clone.split('.')[0];counts[origin]=counts.get(origin,0)+1
    assert len(clones)<=64 and all(n<=8 for n in counts.values()),counts
    reports.append({'case':name,'total_clones':len(clones),'descendants_by_origin':counts,'deterministic':True,'commands':commands,'source_sha256':sha(path),'output_sha256':sha(out/(name+'-out1.ll'))})
    return text,clones

def definition(name,n):
    args=', '.join('ptr nocapture %p'+str(i) for i in range(n))
    # Real private loads make an untransformed fallback observably necessary.
    body='\n'.join(f' %v{i}=load i32, ptr %p{i}, align 4' for i in range(n))
    return f'define internal fastcc zeroext i32 @{name}({args}) noinline norecurse {{\n{body}\n ret i32 %v0\n}}\n'

functions=''.join(definition('f'+str(i),1) for i in range(70))
calls=[f' %r{i}=notail call fastcc zeroext i32 @f{i}(ptr %g)' for i in range(70)]
calls.append(' %reuse=tail call fastcc zeroext i32 @f0(ptr %g)')
text,clones=execute('module-budget',functions,1,calls)
assert len(clones)==64 and set(n.split('.')[0] for n,_ in clones)=={'f'+str(i) for i in range(64)}
for i in range(64,70):
    assert f'%r{i} = notail call fastcc zeroext i32 @f{i}(ptr %g)' in text
    assert re.search(r'define internal fastcc zeroext i32 @f'+str(i)+r'\(ptr nocapture %p0\)[^\n]*\n\s*%v0 = load i32, ptr %p0, align 4\n\s*ret i32 %v0',text)
assert 'tail call fastcc zeroext i32 @f0.private5(ptr addrspace(5) %p)' in text
reports[-1]['unchanged_generic_calls']=6
reports[-1]['cached_clone_reused_after_exhaustion']=True

calls=[]
for mask in range(16):
    args=', '.join('ptr %'+('g' if mask&(1<<i) else 'h') for i in range(4))
    calls.append(f' %r{mask}=notail call fastcc zeroext i32 @f0({args})')
text,clones=execute('origin-budget',definition('f0',4),4,calls)
assert len(clones)==8 and any(n.count('.private5')>=2 for n,_ in clones)
actual_calls=re.findall(r'notail call fastcc zeroext i32 @(f0[^ (]*)\(([^\n]*)',text)
assert len(actual_calls)==16
assert any('ptr %g' in args for _,args in actual_calls)
assert 'notail call fastcc zeroext i32 @f0(ptr %h, ptr %h, ptr %h, ptr %h)' in text
assert all('ptr %g' in args or args.count('ptr addrspace(5)')==bin(mask).count('1') for mask,(_,args) in enumerate(actual_calls))
reports[-1]['remaining_generic_private_arguments']=sum(args.count('ptr %g') for _,args in actual_calls)
reports[-1]['generic_global_fallback_retained']=True

functions=''.join(definition('f'+str(i),4) for i in range(10));calls=[]
for fn in range(10):
    for mask in range(1,16):
        args=', '.join('ptr %'+('g' if mask&(1<<i) else 'h') for i in range(4))
        calls.append(f' %r{fn}_{mask}=notail call fastcc zeroext i32 @f{fn}({args})')
text,clones=execute('combined-budgets',functions,4,calls)
assert len(clones)==64
assert len(re.findall(r'notail call fastcc zeroext i32 @',text))==150
reports[-1]['all_150_calls_retained']=True
assert sha(plugin)==before
receipt={'plugin_sha256':before,'limits':{'new_clones_per_module_per_pass_invocation':64,'descendants_per_original_generic_function_per_pass_invocation':8},'cases':reports,'boundary':'Budget exhaustion keeps the current call and its generic pointer conversions, or reuses an existing clone without adding one. Structural and verifier evidence only; no GPU execution or safe platform fallback claim. Reinvoking the pass has a fresh budget; not a lifetime/module-size quota.','script_sha256':sha(Path(__file__))}
(out/'budget-results.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('PASS module64/origin8/combined64; repeated runs byte-identical; skippedcalls and cache reuse retained')
print('receipt',sha(out/'budget-results.json'))
