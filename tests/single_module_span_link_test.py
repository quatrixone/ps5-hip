"""Prove actual native runtime weak-table references remain null after linking."""
import argparse,hashlib,json,pathlib,struct,sys,tempfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
from fat_binary_fixture_test import section
from probe_host_registration import load_regions
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
NAMES={'__ps5hip_module_spans_begin','__ps5hip_module_spans_end'}

def symbols(data,table=b'.symtab',strings=b'.strtab'):
 raw=section(data,table);names=section(data,strings);result=[]
 for at in range(0,len(raw),24):
  n,info,other,index,value,size=struct.unpack_from('<IBBHQQ',raw,at)
  name=names[n:names.index(0,n)].decode() if n else ''
  result.append((name,info,other,index,value,size))
 return result

def slots(image,runtime):
 data=image.read_bytes();syms=symbols(data)
 selected=[s for s in syms if s[0] in NAMES]
 assert len(selected)==2 and {s[0] for s in selected}==NAMES
 assert all(s[2]==2 and s[3]==0xfff1 and s[4]==0 for s in selected),'optional symbols must be hidden absolute zero'
 assert not any(s[0] in NAMES for s in symbols(data,b'.dynsym',b'.dynstr')),'optional symbols must not be imports'
 starts=[s[4] for s in syms if s[0]=='__hipRegisterFatBinary'];assert len(starts)==1
 obj=runtime.read_bytes();os=symbols(obj);rela=section(obj,b'.rela.text.__hipRegisterFatBinary');read,_,_=load_regions(image)
 refs=[]
 for at in range(0,len(rela),24):
  offset,info,addend=struct.unpack_from('<QQq',rela,at);name=os[info>>32][0]
  if name not in NAMES or (info&0xffffffff)!=9:continue
  # Non-relaxable GOTPCREL references in the actual optional-table null tests.
  assert addend==-5
  place=starts[0]+offset;raw,_=read(place,4)
  target=place+struct.unpack('<i',raw)[0]-addend
  refs.append((name,target))
 assert len(refs)==4 and all(sum(n==name for n,_ in refs)==2 for name in NAMES)
 result={name:{target for n,target in refs if n==name} for name in NAMES}
 assert all(len(v)==1 for v in result.values()) and len(set.union(*result.values()))==2
 return {n:next(iter(v)) for n,v in result.items()}

def check_storage(image,addresses):
 read,_,relas=load_regions(image)
 for address in addresses.values():
  raw,_=read(address,8);assert raw==bytes(8),'optional table GOT is nonzero'
  assert not any(address<=r[0]<address+8 or r[0]<=address<r[0]+8 for r in relas),'loader may relocate optional table GOT'

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',type=pathlib.Path,required=True);p.add_argument('--runtime-object',type=pathlib.Path,required=True);p.add_argument('--converted',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 inputs={str(v.resolve()):H(v) for v in (a.image,a.runtime_object,*([a.converted] if a.converted else []))}
 sources={str(path.relative_to(ROOT)):H(path) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and (path:=pathlib.Path(m.__file__).resolve()).is_relative_to(ROOT) and path.is_file()}
 addresses=slots(a.image,a.runtime_object);check_storage(a.image,addresses)
 if a.converted:check_storage(a.converted,addresses)
 mutations=0
 with tempfile.TemporaryDirectory() as temp:
  bad=pathlib.Path(temp)/'bad.elf'
  for address in addresses.values():
   data=bytearray(a.image.read_bytes());_,headers,_=load_regions(a.image);load=next(h for h in headers if h[0]==1 and h[3]<=address< h[3]+h[5]);at=load[2]+address-load[3]
   data[at]=1;bad.write_bytes(data)
   try:check_storage(bad,addresses)
   except AssertionError:mutations+=1
   else:raise AssertionError('nonzero GOT accepted')
   data=bytearray(a.image.read_bytes());relas=section(bytes(data),b'.rela.dyn');assert relas and data.count(relas)==1
   struct.pack_into('<Q',data,data.index(relas),address);bad.write_bytes(data)
   try:check_storage(bad,addresses)
   except AssertionError:mutations+=1
   else:raise AssertionError('relocated GOT accepted')
 assert mutations==4 and all(H(pathlib.Path(n))==h for n,h in inputs.items())
 assert all(H(ROOT/n)==h for n,h in sources.items())
 a.output.mkdir(parents=True,exist_ok=False)
 (a.output/'receipt.json').write_text(json.dumps(dict(classification='actual-single-module-hidden-zero-got-pass',sources=sources,inputs=inputs,got_slots=addresses,references=4,mutations_rejected=mutations,converted_checked=bool(a.converted),hardware_executed=False),indent=2)+'\n')
 print('PASS actual hidden ABS0 symbols, four runtime references, two zero nonrelocated GOT slots; four mutations rejected')
if __name__=='__main__':main()
