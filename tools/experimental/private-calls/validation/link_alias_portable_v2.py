"""Run the unchanged alias byte audit with explicitly prepared LLVM tools."""
from pathlib import Path
import argparse,json,subprocess
from link_alias_audit_v1 import parse,positive_equal,sha


def verified_tools(prepared,linker,objdump):
 report=json.loads(prepared.read_text());tools={}
 for key,name,path in [('linker','ld.lld-18',linker),('objdump','llvm-objdump-18',objdump)]:
  identity=report['tools'][name]
  if not path.is_absolute():raise ValueError('Tool path must be absolute: '+key)
  path=path.resolve(strict=True)
  if str(path)!=identity['path'] or sha(path)!=identity['sha256']:
   raise ValueError('Tool differs from preparation: '+key)
  for library,digest in identity['libraries'].items():
   if sha(Path(library))!=digest:raise ValueError('Tool library changed: '+library)
  tools[key]=path
 return tools


def run(validation,output,prepared,linker,objdump):
 prepared_hash=sha(prepared);tools=verified_tools(prepared,linker,objdump)
 output.mkdir(exist_ok=False);cases=[];commands=[]
 identities={key:sha(value) for key,value in tools.items()}
 for target in ('gfx1013','gfx1030'):
  row={'target':target,'modes':{}};cases.append(row);graphs=[];descriptors=[]
  for mode in ('baseline','patched-off','patched-on'):
   source=validation/'negative-runs'/mode/('alias-'+target);d=output/target/mode;d.mkdir(parents=True)
   report=json.loads((source.parent/'result.json').read_text());case=next(c for c in report['cases'] if c['name']=='alias-'+target)
   assert sha(source/'output.o')==case['files']['output.o']
   for stage,cmd in [('link',[str(tools['linker']),'-flavor','gnu','-shared','--no-undefined',str(source/'output.o'),'-o',str(d/'linked.hsaco')]),('isa',[str(tools['objdump']),'-d','--mcpu='+target,str(d/'linked.hsaco')])]:
    result=subprocess.run(cmd,capture_output=True,text=True);(d/('isa.txt' if stage=='isa' else 'link.log')).write_text(result.stdout+result.stderr);commands.append({'command':cmd,'exit_code':result.returncode});assert result.returncode==0
   graph,descriptor,call=parse((d/'linked.hsaco').read_bytes(),(d/'isa.txt').read_text());graphs.append(graph);descriptors.append(descriptor)
   row['modes'][mode]={'input_object_sha256':sha(source/'output.o'),'receipt_sha256':sha(source.parent/'result.json'),'resolved_call':call,'files':{f.name:sha(f) for f in d.iterdir()}}
  assert descriptors[0]==descriptors[1]==descriptors[2] and graphs[0]==graphs[1]
  assert (output/target/'baseline/linked.hsaco').read_bytes()==(output/target/'patched-off/linked.hsaco').read_bytes()
  assert graphs[0]['safe']==graphs[2]['safe'] and not any(op.startswith(('flat_','scratch_','s_setreg','s_getreg')) for body in graphs[2].values() for op in body)
  # Synthetic formatting wraps actual byte-bound linked instructions to reuse
  # the exact literal schedule checker; it does not replace saved real assembly.
  def assembly(graph):return ''.join(name+':\n'+''.join('\t'+op+'\n' for op in body)+'; codeLenInByte = '+str(92 if name=='negative' else 20)+'\n' for name,body in graph.items())
  before=assembly(graphs[0]);after=assembly(graphs[2]).replace('; codeLenInByte = 92','; codeLenInByte = 76')
  row['linked_delta']=positive_equal(before,after,'negative','s_movk_i32 s32, 0x400',1)
 assert sha(prepared)==prepared_hash and verified_tools(prepared,linker,objdump)==tools
 report={'linked_alias_audit_passed':True,'accepted_compiler':False,'hardware_validated':False,'cases':cases,'commands':commands,'tools_sha256':identities,'tool_paths':{key:str(value) for key,value in tools.items()},'prepared_sha256':prepared_hash,'script_sha256':sha(Path(__file__)),'checker_sha256':sha(Path(__file__).with_name('link_alias_audit_v1.py')),'limits':'Only two alias-control graphs linked from existing objects. Complete compiler acceptance and ordinary graph audit remain separate.'}
 (output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',sha(output/'result.json'))


if __name__=='__main__':
 if not __debug__:raise RuntimeError('Assertions are required; do not use Python -O or PYTHONOPTIMIZE')
 p=argparse.ArgumentParser()
 for name in ('validation','output','prepared','linker','objdump'):p.add_argument('--'+name,type=Path,required=True)
 args=p.parse_args();run(args.validation,args.output,args.prepared,args.linker,args.objdump)
