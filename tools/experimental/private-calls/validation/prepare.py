from pathlib import Path
import hashlib,json,re,subprocess
out=Path(__file__).resolve().parent;root=out.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plugin=root/'results/g3-private-specialization-v3/PrivateSpecialize.so'
assert sha(plugin)==json.loads((root/'prepare-inputs.json').read_text())['plugin_sha256']
inputs=out/'inputs';inputs.mkdir()
cases=[];originals={};commands=[]
for target in ('gfx1013','gfx1030'):
 for opt in ('O1','O2'):
  for kind in ('ordinary','generic','volatile'):
   parent=root/'results'/('g3-volatile-call-observable' if kind=='volatile' else 'g3-flat-call-observable')/(target+'-'+opt)
   source=parent/('volatile_calls.ll' if kind=='volatile' else 'probes.ll')
   originals[str(source.relative_to(root))]=sha(source)
   stem=f'{kind}-{target}-{opt}'
   dest=inputs/(stem+'.ll')
   if kind=='ordinary':
    cmd=['opt-18','-load-pass-plugin',str(plugin),'-passes=research-private-specialize,function(infer-address-spaces,instcombine,simplifycfg),verify','-S',str(source),'-o',str(dest)]
    r=subprocess.run(cmd,capture_output=True,text=True);(inputs/(stem+'.log')).write_text(r.stdout+r.stderr);r.check_returncode();commands.append(cmd)
   else:dest.write_bytes(source.read_bytes())
   text=dest.read_text();pattern=r'(?ms)^define [^\n]+\{\n.*?^}'
   blocks=re.findall(pattern,text);assert len(blocks)>=3
   reordered=iter(reversed(blocks));reverse=re.sub(pattern,lambda m:next(reordered),text)
   assert sorted(re.findall(pattern,reverse))==sorted(blocks) and reverse!=text
   rev=inputs/(stem+'-reverse.ll');rev.write_text(reverse)
   roots=['volatile_nested_private','volatile_mixed_private'] if kind=='volatile' else ['nested_private']
   for order,path in [('original',dest),('reverse',rev)]:
    cmd=['opt-18','-passes=verify','-disable-output',str(path)];subprocess.run(cmd,check=True);commands.append(cmd)
    cases.append({'id':stem+'-'+order,'kind':kind,'target':target,'optimization':opt,'order':order,'input':str(path.relative_to(out)),'sha256':sha(path),'roots':roots})
for path,digest in originals.items():assert sha(root/path)==digest
report={'schema':1,'hardware_validated':False,'plugin_sha256':sha(plugin),'backend_patch_sha256':sha(root/'results/g3-private-backend-design/buffer-call-proof-v4.patch'),'source_inputs':originals,'cases':cases,'commands':commands,'scope':'24 IR cases: ordinary specialized, generic original, fresh volatile nested+mixed, each target/O1/O2 and original/reversed definition order. Function bodies and original fixtures unchanged.','script_sha256':sha(Path(__file__))}
(out/'matrix.json').write_text(json.dumps(report,indent=2)+'\n')
print('Prepared',len(cases),'verified IR cases',sha(out/'matrix.json'))
