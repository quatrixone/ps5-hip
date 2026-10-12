from pathlib import Path
import hashlib,json,re,subprocess
out=Path(__file__).resolve().parent
plugin=out/'PrivateSpecialize.so'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=sha(plugin);commands=[]
for name,pipeline in [('adversarial-specialized','research-private-specialize,verify'),('adversarial-lowered','research-private-specialize,function(infer-address-spaces,instcombine,simplifycfg),verify')]:
    command=['opt-18','-load-pass-plugin',str(plugin),'-passes='+pipeline,'-S',str(out/'cases.ll'),'-o',str(out/(name+'.ll'))]
    r=subprocess.run(command,capture_output=True,text=True);(out/(name+'.log')).write_text(r.stdout+r.stderr);r.check_returncode()
    commands.append({'command':command,'returncode':r.returncode})
raw=(out/'adversarial-specialized.ll').read_text();lower=(out/'adversarial-lowered.ll').read_text()
for name in ('forward','escape','ptrcompare','selectptr'):assert name+'.private5' not in raw,name
defs=re.findall(r'^define internal fastcc zeroext i32 @(pair[^ (]*)\(([^\n]*)',raw,re.M)
assert len(defs)==4,defs
assert all('#0' in signature for _,signature in defs),defs
assert 'call fastcc zeroext i32 @pair(ptr %c, ptr %c, i32 %i)' in raw
full=[name for name,sig in defs if sig.count('ptr addrspace(5)')==2];assert len(full)==1
assert len(re.findall(r'call fastcc zeroext i32 @'+re.escape(full[0])+r'\(',raw))==2
for name,sig in defs:
    body=re.search(r'define internal fastcc zeroext i32 @'+re.escape(name)+r'\([^\n]*\n(.*?)^}',raw,re.M|re.S)[1]
    assert body.count('load volatile i32')==1 and body.count('store volatile i32')==1
assert 'call i32 @unused.private5(ptr addrspace(5) poison)' in raw
assert 'call i32 @unused.private5(ptr addrspace(5) null)' in raw
assert 'call i32 @unused.private5(ptr addrspace(5) undef)' in raw
assert raw.count('define internal i32 @unused.private5(')==1
offset=re.search(r'define internal i32 @offset.private5[^\n]*\n(.*?)^}',lower,re.M|re.S)[1]
assert 'getelementptr i8, ptr addrspace(5)' in offset and 'getelementptr inbounds i32, ptr addrspace(5)' in offset
assert sha(plugin)==before
report={'plugin_sha256':before,'commands':commands,'verified':['fastcc and zeroext return attrs preserved','noinline/norecurse/nounwind preserved on clones','two independently specialized parameters; three deduplicated specialized pair definitions for three address-space combinations','repeat private/private calls share clone; generic global/global call retained','poison/null/undef values preserved without noundef insertion in unused-pointer callee','no specialization of forwarding to external, storing pointer, comparing pointer, or selected mixed pointer','volatile load/store count preserved; unsupported volatile lowering remains generic','inbounds/non-inbounds GEP flags retained by downstream lowering'],'semantic_limits':['LLVM verification and IR structure only, not executable equivalence proof','Pointer arithmetic is delegated to upstream InferAddressSpaces; no new object-bounds guarantee','This does not retest known changed-parameter ABI attributes being fixed separately'],'files':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.ll','.log')},'script_sha256':sha(Path(__file__))}
(out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS',json.dumps(report['verified']))
print('receipt',sha(out/'review.json'))
