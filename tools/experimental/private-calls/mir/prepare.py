from pathlib import Path
import hashlib,json,re
p=Path(__file__).resolve().parent;repo=p.parents[1]
source_root=Path(json.loads((repo/'prepare-inputs.json').read_text())['llvm_source'])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
refs=['llvm/lib/Target/AMDGPU/SIFixVGPRCopies.cpp','llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp','llvm/include/llvm/CodeGen/MIRYamlMapping.h','llvm/lib/CodeGen/MIRParser/MIRParser.cpp']
assert '#define DEBUG_TYPE "si-fix-vgpr-copies"' in (source_root/refs[0]).read_text()
out=p/'inputs';out.mkdir(exist_ok=False)
cases=[]
for target in ['gfx1013','gfx1030']:
 original=repo/'results/g3-private-backend-validation/stress-runs/baseline-late-callee-unbounded'/target/'pre-pei.mir'
 text=original.read_text();start=text.index('name:            leaf\n');end=text.index('\n...\n',start)
 leaf=text[start:end]
 assert 'stack:           []' in leaf
 variable=leaf.replace('stack:           []','stack:\n  - { id: 0, type: variable-sized, alignment: 4 }',1)
 bundle=leaf.replace('    renamable ', '    BUNDLE {\n      S_NOP 0\n    }\n    renamable ',1)
 assert bundle!=leaf
 alias=text.replace('  define internal fastcc i32 @leaf(', '  @late_alias = internal alias i32 (ptr addrspace(5)), ptr @leaf\n\n  define internal fastcc i32 @leaf(',1)
 alias=re.sub(r'(SI_CALL[^\n]*), @leaf,',r'\1, @late_alias,',alias)
 assert '@late_alias, csr_amdgpu' in alias
 variants={'unchanged':text,'variable-frame':text[:start]+variable+text[end:],'bundle':text[:start]+bundle+text[end:],'residual-alias':alias}
 large=repo/'results/g3-private-backend-validation/stress-runs/baseline'/target/'pre-pei.mir'
 variants['unknown-spill-pseudo']=large.read_text();assert 'SI_SPILL_V32_SAVE' in variants['unknown-spill-pseudo']
 for kind,body in variants.items():
  dest=out/(kind+'-'+target+'.mir');dest.write_text(body)
  cases.append({'name':kind+'-'+target,'kind':kind,'target':target,'file':str(dest.relative_to(p)),'sha256':sha(dest),'source_sha256':sha(large if kind=='unknown-spill-pseudo' else original),'expected_root_proof':'proven' if kind=='unchanged' else 'rejected'})
report={'cases':cases,'start_before':'si-fix-vgpr-copies','stop_before':'prologepilog','source_commit':'3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff','source_references':{f:sha(source_root/f) for f in refs},'limits':['Diagnostic MIR mutations only: no production IR/device/descriptor change or semantic execution claim.','MIR verifier/parser success is not proof rejection; require actual compiled pass diagnostic and retained initialization separately.'],'recipe_sha256':sha(Path(__file__))}
(p/'matrix.json').write_text(json.dumps(report,indent=2)+'\n');print('Prepared',len(cases),'MIR diagnostic cases')
