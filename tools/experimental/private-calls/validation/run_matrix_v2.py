from pathlib import Path
import argparse,hashlib,json,re,subprocess
from diagnostics_v2 import check_diagnostics, DEAD_ORDINARY_ORIGINALS
out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())

def inspect_case(case,folder,positive):
 isa=(folder/'isa.txt').read_text();asm=(folder/'output.s').read_text();metadata=(folder/'metadata.txt').read_text()
 assert 'Relocations [\n]' in metadata,'unresolved relocations'
 symbols={m[2]:(int(m[1],16),m[3]) for m in re.finditer(r'(?ms)^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',isa)}
 address_names={value[0]:name for name,value in symbols.items()}
 def calls(name):
  body=symbols[name][1];edges=[]
  for call in re.finditer(r's_swappc_b64 s\[(\d+):(\d+)\], s\[(\d+):(\d+)\]',body):
   lo,hi=call[3],call[4];prefix=body[:call.start()]
   pcs=list(re.finditer(r's_getpc_b64 s\['+lo+':'+hi+r'\].*?// ([0-9A-F]+):',prefix));assert pcs,(name,'missing directPC')
   pc=pcs[-1];tail=prefix[pc.end():]
   add=re.search(r's_add_u32 s'+lo+', s'+lo+r', (0x[0-9a-f]+|-?\d+)\s',tail);assert add,(name,'missing displacement')
   delta=int(add[1],0)&0xffffffff;delta=delta-(1<<32) if delta&(1<<31) else delta
   high=re.search(r's_addc_u32 s'+hi+', s'+hi+r', (-?\d+)\s',tail);assert high and int(high[1])==(-1 if delta<0 else 0)
   address=int(pc[1],16)+4+delta;assert address in address_names,(name,hex(address))
   assert (call[1],call[2])==('30','31'),(name,'returnregisterpair')
   edges.append({'caller':name,'callee':address_names[address],'address':address})
  assert len(edges)==body.count('s_swappc_b64'),(name,'unknowncallform')
  return edges
 def walk(name,seen):
  assert name not in seen,'recursive graph';seen=seen|{name};edges=calls(name);result=[name]
  for edge in edges:result+=walk(edge['callee'],seen)
  return result
 descriptors={m[1]:{k:int(v,0) for k,v in re.findall(r'^\s*\.amdhsa_(\w+)\s+(\w+)',m[2],re.M)} for m in re.finditer(r'\.amdhsa_kernel (\S+)\n(.*?)\.end_amdhsa_kernel',asm,re.S)}
 roots=[]
 for root in case['roots']:
  assert root in symbols and root in descriptors
  reachable=walk(root,set());inventory={}
  for name in reachable:
   body=symbols[name][1]
   inventory[name]={'flat':len(re.findall(r'^\s*flat_\w+',body,re.M)),'scratch':len(re.findall(r'^\s*scratch_\w+',body,re.M)),'aperture':body.count('src_private_'),'flat_init':body.count('s_setreg_b32 hwreg(HW_REG_FLAT_SCR_'),'buffer_load':len(re.findall(r'^\s*buffer_load_\w+',body,re.M)),'buffer_store':len(re.findall(r'^\s*buffer_store_\w+',body,re.M)),'calls':calls(name)}
  desc=descriptors[root];assert desc['user_sgpr_count']==8 and desc['user_sgpr_flat_scratch_init']==1
  if root in ('nested_private','volatile_nested_private'):assert len(reachable)==3 and desc['private_segment_fixed_size']==288
  if root=='volatile_mixed_private':assert len(reachable)==2
  if case['kind']=='ordinary':
   assert reachable==['nested_private','_Z11call_middlePjj.private5','_Z9call_leafPjj.private5']
   assert all(not v['flat'] and not v['scratch'] and not v['aperture'] for v in inventory.values())
   assert all(inventory[name]['buffer_load'] and inventory[name]['buffer_store'] for name in reachable[1:])
   middle=symbols[reachable[1]][1];post=middle.split('s_swappc_b64',1)[1]
   ops=[line.split('//')[0].strip() for line in post.splitlines() if '//' in line]
   load_pattern=r'buffer_load_dword (v\d+), (v\d+), s\[0:3\], 0 offen'
   first=next((i for i,op in enumerate(ops) if re.fullmatch(load_pattern,op)),None)
   assert first is not None and first+3<len(ops),'missingpostcallloads'
   first_load=re.fullmatch(load_pattern,ops[first]);second_load=re.fullmatch(load_pattern,ops[first+1])
   assert second_load and ops[first+2].startswith('s_waitcnt vmcnt(0)')
   assert ops[first+3]==f'v_add3_u32 v0, {first_load[1]}, v0, {second_load[1]}','disconnectedpostcallreload'
   leaf=symbols[reachable[2]][1]
   loads=re.findall(load_pattern,leaf);stores=re.findall(r'buffer_store_dword (v\d+), (v\d+), s\[0:3\], 0 offen',leaf)
   assert len(loads)==2 and stores==[loads[0]],'leafprivateaccesscontract'
   assert f'v_add_nc_u32_e32 {loads[0][0]}, 3, {loads[0][0]}' in leaf
   assert f'v_add_nc_u32_e32 v0, v0, {loads[0][0]}' in leaf and 's_setpc_b64 s[30:31]' in leaf
   assert re.search(r'global_store_dword v\[\d+:\d+\], v0, off',symbols[root][1]),'returnnotstored'
   assert 's_movk_i32 s32, 0x1200' in symbols[root][1] and 's_addk_i32 s32, 0x1200' in middle and 's_addk_i32 s32, 0xee00' in middle
  else:assert any(v['flat'] for v in inventory.values()),'negative no longer exercises generic FLAT'
  assert inventory[root]['flat_init']==(0 if positive else 2),(root,'unexpectedflatinit',inventory[root])
  roots.append({'root':root,'reachable':reachable,'descriptor':desc,'inventory':inventory})
 return roots

def main():
 p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--mode',choices=['installed','baseline','patched-off','patched-on'],required=True);p.add_argument('--run-label');args=p.parse_args()
 matrix=read(out/'matrix.json');folder=out/'runs'/(args.run_label or args.mode);folder.mkdir(parents=True,exist_ok=False)
 compiler=args.compiler.resolve();before=sha(compiler);results=[];commands=[]
 baseline=read(out/'runs/baseline/result.json') if args.mode.startswith('patched') else None
 if args.mode=='patched-on':
  off=read(out/'runs/patched-off/result.json')
  assert off['same_build_baseline_identical'] is True
  assert off['compiler_sha256']==before, 'enabled compiler differs from verified disabled compiler'
  assert off['matrix_sha256']==baseline['matrix_sha256']==sha(out/'matrix.json'), 'control matrix differs'
 try:
  for case in matrix['cases']:
   print(args.mode,case['id'],flush=True);d=folder/case['id'];d.mkdir();source=out/case['input'];assert sha(source)==case['sha256']
   base=[str(compiler),'-mtriple=amdgcn-amd-amdhsa','-mcpu='+case['target'],'-'+case['optimization'],'-verify-machineinstrs']
   if args.mode=='patched-on':base+=['-amdgpu-experimental-prove-buffer-calls','-debug-only=amdgpu-prove-buffer-calls']
   for name,cmd in [('asm',base+['-filetype=asm',str(source),'-o',str(d/'output.s')]),('object',base+['-filetype=obj',str(source),'-o',str(d/'output.o')]),('link',['ld.lld-18','-shared','--no-undefined',str(d/'output.o'),'-o',str(d/'output.hsaco')]),('isa',['llvm-objdump-18','-d','--mcpu='+case['target'],str(d/'output.hsaco')]),('metadata',['llvm-readobj-18','--notes','--symbols','--relocations',str(d/'output.hsaco')])]:
    r=subprocess.run(cmd,capture_output=True,text=True);(d/(name+'.log' if name not in ('isa','metadata') else name+'.txt')).write_text(r.stdout+r.stderr)
    commands.append({'case':case['id'],'command':cmd,'exit_code':r.returncode});(folder/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');r.check_returncode()
   roots=inspect_case(case,d,args.mode=='patched-on' and case['kind']=='ordinary')
   if baseline:
    basecase=next(c for c in baseline['cases'] if c['id']==case['id']);basedir=out/'runs/baseline'/case['id']
    assert sha(basedir/'metadata.txt')==basecase['files']['metadata.txt'], 'baseline metadata changed'
   diagnostics=None
   if args.mode=='patched-on':
    expected={root['root']:('proven' if case['kind']=='ordinary' else 'rejected') for root in roots}
    reachable={name for root in roots for name in root['reachable']}
    if case['kind']=='ordinary':expected.update({name:'proven' for name in reachable})
    diagnostics=check_diagnostics((d/'asm.log').read_text(),source.read_text(),expected,reachable,
      linked_metadata=(d/'metadata.txt').read_text(),baseline_metadata=(basedir/'metadata.txt').read_text(),
      allowed_dead=DEAD_ORDINARY_ORIGINALS if case['kind']=='ordinary' else ())
   if baseline:
    basecase=next(c for c in baseline['cases'] if c['id']==case['id']);basedir=out/'runs/baseline'/case['id']
    if args.mode=='patched-off':
     for name in ('output.s','output.o','output.hsaco'):assert sha(d/name)==sha(basedir/name),(case['id'],name,'defaultoffdiff')
    else:
     assert [r['descriptor'] for r in roots]==[r['descriptor'] for r in basecase['roots']],(case['id'],'resourcechange')
   results.append({'id':case['id'],'roots':roots,'diagnostics':diagnostics,'files':{p.name:sha(p) for p in sorted(d.iterdir())}})
  assert sha(compiler)==before
  report={'mode':args.mode,'compiler':str(compiler),'compiler_sha256':before,'matrix_sha256':sha(out/'matrix.json'),'hardware_validated':False,'patched_acceptance_complete':False,'same_build_baseline_identical':args.mode=='patched-off','cases':results,'limits':['Installed compiler controls are not evidence for patched compiler behavior.','Matrix checks selected observable ordinary/generic/volatile graphs only. Full unknown/external/alias/recursive/inlineasm/dynamicframe/forgedattr/MIR reload negative acceptance remains separate.','Even passing patched matrix does not alone establish pre-PEI all-machine-functions barrier instrumentation or authorize GPU use.'],'script_sha256':sha(Path(__file__)), 'diagnostic_parser_sha256':sha(out/'diagnostics_v2.py')}
  (folder/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(results),'cases',sha(folder/'result.json'))
 except Exception as e:
  (folder/'failure.json').write_text(json.dumps({'error':repr(e),'completed_cases':len(results),'commands':commands},indent=2)+'\n');raise
if __name__=='__main__':main()
