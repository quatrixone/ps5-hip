"""Link existing alias objects only; verify bytes and exact resolved call targets."""
from pathlib import Path
import argparse,hashlib,json,re,struct,subprocess
from post_audit_v2 import Elf
from post_audit_extensions_v2 import positive_equal
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def parse(data,isa):
 elf=Elf(data);assert {name for name,value in elf.symbols.items() if value['type']==2}=={'negative','safe'}
 functions={}
 for match in re.finditer(r'(?ms)^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',isa):
  name=match[2]
  if name not in ('negative','safe'):continue
  symbol=elf.symbols[name];assert symbol['address']==int(match[1],16)
  cursor=symbol['address'];ops=[]
  for line in match[3].splitlines():
   if not line.strip():continue
   m=re.fullmatch(r'\s*(.*?)\s+//\s+([0-9A-Fa-f]+):\s+([0-9A-Fa-f ]+?)(?:\s+<[^>]+>)?\s*',line);assert m
   op=' '.join(m[1].split());address=int(m[2],16);words=m[3].split();assert all(len(word)==8 for word in words)
   encoded=b''.join(struct.pack('<I',int(word,16)) for word in words)
   assert address==cursor and elf.at(address,len(encoded))==encoded;cursor+=len(encoded)
   if address>=symbol['address']+symbol['size']:
    assert ops and ops[-1]['text']==('s_endpgm' if name=='negative' else 's_setpc_b64 s[30:31]') and op in ('s_code_end','s_nop 0')
    continue
   assert cursor<=symbol['address']+symbol['size'];ops.append({'text':op,'address':address,'bytes':len(encoded)})
  assert ops[-1]['address']+ops[-1]['bytes']==symbol['address']+symbol['size'];functions[name]=ops
 assert set(functions)=={'negative','safe'}
 calls=[i for i,op in enumerate(functions['negative']) if op['text'].startswith('s_swappc')];assert len(calls)==1
 ops=functions['negative'];i=calls[0];assert ops[i]['text']=='s_swappc_b64 s[30:31], s[4:5]'
 pcs=[j for j in range(i) if ops[j]['text']=='s_getpc_b64 s[4:5]'];assert len(pcs)==1;j=pcs[0];assert j+2<i
 low=re.fullmatch(r's_add_u32 s4, s4, (0x[0-9a-f]+|-?\d+)',ops[j+1]['text']);high=re.fullmatch(r's_addc_u32 s5, s5, (0x[0-9a-f]+|-?\d+)',ops[j+2]['text']);assert low and high and ops[j]['bytes']==4
 delta=int(low[1],0)&0xffffffff;delta=delta-(1<<32) if delta&(1<<31) else delta
 assert int(high[1],0)&0xffffffff==(0xffffffff if delta<0 else 0)
 target=ops[j]['address']+4+delta;assert target==elf.symbols['safe']['address'],'linked call does not reach safe'
 canonical={name:[op['text'] for op in body] for name,body in functions.items()}
 canonical['negative'][j+1]='s_add_u32 s4, s4, safe@rel32@lo+4';canonical['negative'][j+2]='s_addc_u32 s5, s5, safe@rel32@hi+12'
 # Only spelling differences of the encoded zero SMEM operand are canonicalized.
 for name,body in canonical.items():
  canonical[name]=[op.replace('s_load_dwordx2 s[34:35], s[4:5], null','s_load_dwordx2 s[34:35], s[4:5], 0x0') for op in body]
 kd=elf.symbols['negative.kd'];descriptor=elf.at(kd['address'],64)
 assert kd['size']==64 and kd['address']+struct.unpack_from('<q',descriptor,16)[0]==elf.symbols['negative']['address']
 assert struct.unpack_from('<III',descriptor)==(0,32,20)
 return canonical,descriptor[:16]+bytes(8)+descriptor[24:],{'pc':ops[j]['address'],'delta':delta,'safe_address':target,'return_pair':[30,31],'source_pair':[4,5]}

def run(validation,output):
 output.mkdir(exist_ok=False);cases=[];commands=[]
 tools={'linker':Path('/usr/lib/llvm-18/bin/lld'),'objdump':Path('/usr/lib/llvm-18/bin/llvm-objdump')};identities={key:sha(value) for key,value in tools.items()}
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
 assert all(sha(path)==identities[key] for key,path in tools.items())
 report={'linked_alias_audit_passed':True,'accepted_compiler':False,'hardware_validated':False,'cases':cases,'commands':commands,'tools_sha256':identities,'script_sha256':sha(Path(__file__)),'limits':'Only two alias-control graphs linked from existing objects. Complete compiler acceptance and ordinary graph audit remain separate.'}
 (output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',sha(output/'result.json'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.validation,args.output)
