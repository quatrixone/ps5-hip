"""Check actual PS5 PIE module offset tables and wrapper dynamic relocations."""
import pathlib,json,hashlib,subprocess,struct,argparse
R=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--proof',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
P=a.proof.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((P/'receipt.json').read_text())
assert all(H(R/k)==v for k,v in receipt['sources'].items())
assert all(H(pathlib.Path(k))==v for k,v in receipt['inputs'].items())
assert all(H(P/k)==v for k,v in receipt['artifacts'].items())
before={str(x):H(x) for x in (pathlib.Path(__file__),P/'receipt.json',P/'ps5/a.o',P/'ps5/b.o',pathlib.Path('/usr/bin/ld.lld-18'),pathlib.Path('/usr/bin/llvm-readobj-18'))}
def elf(data):
 off=struct.unpack_from('<Q',data,40)[0];width,count,si=struct.unpack_from('<HHH',data,58);assert width==64
 entries=[struct.unpack_from('<IIQQQQIIQQ',data,off+i*width) for i in range(count)]
 nh=entries[si];names=data[nh[4]:nh[4]+nh[5]]
 return {names[h[0]:names.find(b'\0',h[0])].decode():(h,data[h[4]:h[4]+h[5]]) for h in entries}
cases=[];commands=[]
for order in ('ab','ba'):
 ld=O/(order+'.ld');binary=O/(order+'.elf')
 parts=['SECTIONS { . = 0x10000;']
 for name in ('a','b'):
  obj=P/'ps5'/(name+'.o')
  parts += [f'.bundle_{name} ALIGN(4096) : {{ {obj}(.hip_fatbin) }}',f'.wrapper_{name} ALIGN(8) : {{ {obj}(.hipFatBinSegment) }}']
 # LLD18 derives output permissions from input sections; READONLY is unsupported.
 parts+=['.module_table ALIGN(8) : { *(.rodata .rodata.*) . = ALIGN(8);']
 for name in ('a','b'):
  parts += [f'QUAD(ADDR(.wrapper_{name})-ADDR(.wrapper_a)); QUAD(ADDR(.wrapper_{name})+SIZEOF(.wrapper_{name})-ADDR(.wrapper_a)); QUAD(ADDR(.bundle_{name})-ADDR(.bundle_a)); QUAD(ADDR(.bundle_{name})+SIZEOF(.bundle_{name})-ADDR(.bundle_a));']
 parts+=['} }'];ld.write_text('\n'.join(parts)+'\n')
 argv=['/usr/bin/ld.lld-18','-pie','--unresolved-symbols=ignore-all','-e','launch_a','-T',str(ld),*[str(P/'ps5'/(n+'.o')) for n in order],'-o',str(binary)]
 q=subprocess.run(argv,capture_output=True,text=True);(O/(order+'-link.txt')).write_text(q.stdout+q.stderr);assert q.returncode==0,q.stderr;commands.append(argv)
 q=subprocess.run(['/usr/bin/llvm-readobj-18','--sections','--relocations','--symbols',str(binary)],capture_output=True,text=True);assert q.returncode==0;(O/(order+'-readobj.txt')).write_text(q.stdout)
 data=binary.read_bytes();assert struct.unpack_from('<H',data,16)[0]==3;s=elf(data);table=s['.module_table'][1][-64:];assert len(table)==64 and not (s['.module_table'][0][2]&1)
 rela=s['.rela.dyn'][1];records=[struct.unpack_from('<QQq',rela,i) for i in range(0,len(rela),24)]
 for i,n in enumerate(('a','b')):
  w=s['.wrapper_'+n][0];b=s['.bundle_'+n][0]
  assert struct.unpack_from('<QQQQ',table,32*i)==(w[3]-s['.wrapper_a'][0][3],w[3]+24-s['.wrapper_a'][0][3],b[3]-s['.bundle_a'][0][3],b[3]+b[5]-s['.bundle_a'][0][3])
  assert w[5]==24 and (w[3]+8,8,b[3]) in records
  assert s['.bundle_'+n][1]==elf((P/'ps5'/(n+'.o')).read_bytes())['.hip_fatbin'][1]
 assert not any(s['.module_table'][0][3]<=offset<s['.module_table'][0][3]+s['.module_table'][0][5] for offset,_,_ in records)
 assert len(s['.ctors'][1])==16
 cases.append(dict(order=order,relative_wrapper_relocations=2,offset_table_bytes=64,constructors=2))
assert before=={k:H(pathlib.Path(k)) for k in before}
(O/'receipt.json').write_text(json.dumps(dict(classification='offline-ps5-pie-module-offset-link-pass',hardware_validated=False,inputs=before,cases=cases,commands=commands,artifacts={str(x.relative_to(O)):H(x) for x in O.iterdir() if x.is_file()},limits=['Unresolved host imports allowed; diagnostic PIE is not a runnable native title.','Two real R_X86_64_RELATIVE wrapper pointers and relocation-free offset table verified.']),indent=2)+'\n')
print('PASS PS5 PIE: two relative wrapper relocations and exact offset table in both object orders')
