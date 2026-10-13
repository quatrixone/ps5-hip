"""Exact two-module native profile and linked ownership audits."""
import hashlib,json,pathlib,re,struct
from fat_binary_fixture_test import section
from probe_host_registration import load_regions,audit_image
H=lambda data:hashlib.sha256(data).hexdigest()
PROFILE={
 'target':'gfx1013','kernel_names':['multi_a','multi_b'],'client_entry':'_Z30ps5hip_multi_query_client_mainiPPc',
 'compiler_receipt_sha256':'31503807f755ee127e0b75b2bff3ae1a07de9c66397238695e8faf6053e9d013',
 'bundle_sha256':'d0a7c636661ee689481f962c490a446128b3f381fef199000918decbff546442',
 'dispatches':3,'exact_values':219,'comparisons':522,'guards':84,'queries':32,'rejections':8,'logical_bytes':348,
 'modules':[
  {'module':'a','kernel':'multi_a','combined_object_sha256':'aa8959ba60f768c94abe2398a3ccaf05acc2b9d867ebe54806aff7fcbb445df3','bundle_sha256':'1cde39ab847d5feccc518e481ab48a5a99dd055b227bb96992e8e557d3eee3a9','code_sha256':'49829beee057dcbaea0bb06a7b3b0ea5cb058f6bde52c33c9d3a743ac23644c8','code_bytes':3752,'bundle_bytes':7848},
  {'module':'b','kernel':'multi_b','combined_object_sha256':'876aff255be5c82b771bc4949d313e6b929569f5f1732adcba77d90fac6f7407','bundle_sha256':'8ea414f361c5b1374f438eeeaa4fad69948ac7c675499b1ec395d59cd44accb4','code_sha256':'45acbfc55ef0dae9de909c1bf3fd4b1a85d2b06fd4b4e76cf40743f0c6377f6d','code_bytes':3752,'bundle_bytes':7848}]
}
SOURCES=['tools/native_multi_query.py','tools/probe_hip_multi_compiler.py','tools/hip_multi_query_client.cpp','tools/hip_multi_query_client.h','tests/hip_multi_a.hip','tests/hip_multi_b.hip','tests/hip_multi_query_model.cpp','tests/hip_multi_query_client_test.py','tests/hip_multi_query_title_model.cpp','tests/hip_multi_query_title_test.py','tests/hip_multi_native_profile_test.py','docs/HIP_MULTI_QUERY_TITLE.md']
SOURCES += ['tests/hip_multi_query_title_receipt_test.py']
def require(ok,message):
 if not ok:raise ValueError(message)
def validate_compiler(raw,objects,root):
 require(H(raw)==PROFILE['compiler_receipt_sha256'] and len(objects)==2,'multi compiler identity')
 r=json.loads(raw);require(r['classification']=='offline-two-module-compiler-inspection-pass' and not r['hardware_executed'],'multi compiler classification')
 for name,digest in r['sources'].items():require(H((root/name).read_bytes())==digest,'multi compiler source changed: '+name)
 for name,digest in r['inputs'].items():require(H(pathlib.Path(name).read_bytes())==digest,'multi compiler input changed: '+name)
 for m,obj in zip(PROFILE['modules'],objects):
  require(H(obj)==m['combined_object_sha256'] and H(section(obj,b'.hip_fatbin'))==m['bundle_sha256'],'multi object/bundle mismatch')
 return r
def aggregate(bundles):
 require(len(bundles)==2,'two bundles required');return bundles[0]+bytes((-len(bundles[0]))%4096)+bundles[1]
def patch_linker(source):
 ro='    .rodata : { *(.rodata .rodata.*) } : ro';rw='        *(.data.rel.ro .data.rel.ro.*)'
 require(source.count(ro)==1 and source.count(rw)==1,'native multi linker input changed')
 read=['    .rodata : {','        HIDDEN(__ps5hip_names_begin = .);','        *(.rodata .rodata.*)','        HIDDEN(__ps5hip_names_end = .);','        . = ALIGN(4096);','        HIDDEN(__ps5hip_bundle_begin = .);']
 write=['        HIDDEN(__ps5hip_relro_begin = .);',rw,'        . = ALIGN(8);','        HIDDEN(__ps5hip_wrappers_begin = .);']
 for m in 'ab':
  read += ['        . = ALIGN(4096);',f'        HIDDEN(multi_{m}_bundle_begin = .);',f'        KEEP(*multi-{m}.o(.hip_fatbin))',f'        HIDDEN(multi_{m}_bundle_end = .);']
  write += [f'        HIDDEN(multi_{m}_wrapper_begin = .);',f'        KEEP(*multi-{m}.o(.hipFatBinSegment))',f'        HIDDEN(multi_{m}_wrapper_end = .);']
 read+=['        HIDDEN(__ps5hip_bundle_end = .);','        . = ALIGN(8);','        HIDDEN(__ps5hip_module_spans_begin = .);']
 for m in 'ab':
  for kind,base in (('wrapper','wrappers'),('bundle','bundle')):
   for edge in ('begin','end'):read.append(f'        QUAD(multi_{m}_{kind}_{edge}-__ps5hip_{base}_begin);')
 read+=['        HIDDEN(__ps5hip_module_spans_end = .);','    } : ro']
 write+=['        HIDDEN(__ps5hip_wrappers_end = .);','        HIDDEN(__ps5hip_relro_end = .);']
 return source.replace(ro,'\n'.join(read)).replace(rw,'\n'.join(write))
def blocks(disassembly):
 result=[]
 for m in re.finditer(r'^([0-9a-f]+) <([^>]+)>:\n(.*?)(?=^[0-9a-f]+ <|\Z)',disassembly,re.M|re.S):result.append((int(m[1],16),m[2],m[3]))
 return result
def symbol_values(data):
 raw=section(data,b'.symtab');names=section(data,b'.strtab');result={}
 for at in range(0,len(raw),24):
  n,info,other,index,value,size=struct.unpack_from('<IBBHQQ',raw,at)
  if index and n:
   name=names[n:names.index(0,n)].decode();result.setdefault(name,[]).append(value)
 return result
def module_constructors(disassembly,symbols):
 found=[b for b in blocks(disassembly) if b[1]=='__hip_module_ctor'];require(len(found)==2,'two actual compiler constructors required')
 result=[]
 for actor in 'ab':
  address=symbols['multi_'+actor+'_wrapper_begin'];matches=[]
  for at,name,body in found:
   targets={int(v,16) for v in re.findall(r'(?:0x)([0-9a-f]+)',body)}
   if address in targets:matches.append(at)
  require(len(matches)==1,'compiler constructor does not reference exact own wrapper');result.append(matches[0])
 require(len(set(result))==2,'compiler constructor ownership ambiguous');return result
def audit_modules(image,symbols,bundles,disassembly,native=False):
 read,headers,relas=load_regions(image);tb=symbols['__ps5hip_module_spans_begin'];te=symbols['__ps5hip_module_spans_end']
 require(tb>0 and te-tb==64,'optional module table symbols missing')
 table,segment=read(tb,64);require(segment[1]==4,'module table not read-only load')
 lo=tb&~0x3fff;hi=(te+0x3fff)&~0x3fff
 for h in headers:
  if h[0]==1 and h[1]&2:require((h[3]+h[6]+0x3fff)&~0x3fff<=lo or (h[3]&~0x3fff)>=hi,'writable load overlaps module table page')
 require(not any(tb<=r[0]<te for r in relas),'relative-offset module table has relocations')
 ctors=module_constructors(disassembly,symbols);audits=[];expected=[]
 for i,m in enumerate(PROFILE['modules']):
  actor=m['module'];own=dict(symbols)
  for kind,aggregate_name in (('wrapper','wrappers'),('bundle','bundle')):
   for edge in ('begin','end'):
    value=symbols['multi_'+actor+'_'+kind+'_'+edge];own['__ps5hip_'+aggregate_name+'_'+edge]=value
    expected.append(value-symbols['__ps5hip_'+aggregate_name+'_begin'])
  own['__hip_module_ctor']=ctors[i]
  require(H(bundles[i])==m['bundle_sha256'],'linked own bundle identity')
  audits.append(audit_image(image,own,bundles[i],native))
 require(list(struct.unpack('<8Q',table))==expected,'linked module span ownership mismatch')
 whole,_=read(symbols['__ps5hip_bundle_begin'],symbols['__ps5hip_bundle_end']-symbols['__ps5hip_bundle_begin'])
 require(whole==aggregate(bundles) and H(whole)==PROFILE['bundle_sha256'],'aggregate linked bundle identity')
 require(symbols['__ps5hip_wrappers_end']-symbols['__ps5hip_wrappers_begin']==48,'unexpected wrapper outside two-module table')
 return {'modules':audits,'table':[tb,te],'table_sha256':H(table),'read_only_page_bytes':16384,'bundle_sha256':H(whole),'native_executed':False}
def audit_startup(disassembly,image):
 from build_native_vector_title import audit_startup as single_startup
 # Retain the existing shell/finalizer/first-ctor actual-PLT checks, then prove
 # every distinct compiler constructor, handle and destructor by address.
 result=single_startup(disassembly,image,'hip_vector')
 values=symbol_values(image.read_bytes());unique={n:v[0] for n,v in values.items() if len(v)==1}
 ctors=module_constructors(disassembly,unique);allblocks=blocks(disassembly)
 bodies={at:(name,body) for at,name,body in allblocks}
 def body(at):require(at in bodies,'missing constructor/dtor disassembly address');return bodies[at][1]
 def targets(text):return {int(v,16) for v in re.findall(r'0x([0-9a-f]+)',text)}
 require(set(values.get('__hip_module_ctor',[]))==set(ctors),'constructor symbol/address inventory')
 dtors=values.get('__hip_module_dtor',[]);handles=values.get('__hip_gpubin_handle',[])
 require(len(dtors)==2 and len(handles)==2 and len(set(dtors))==len(set(handles))==2,'two distinct compiler handles/destructors required')
 reader,_,_=load_regions(image);proof=[]
 for actor,ctor in zip('ab',ctors):
  text=body(ctor);refs=targets(text)
  require(text.count('<__hipRegisterFatBinary>')==text.count('<__hipRegisterFunction>')==1,'each compiler ctor must bind exactly one module/function')
  require(unique['multi_'+actor] in refs,'own genuine host token missing')
  own_dtor=set(dtors)&refs;own_handle=set(handles)&refs
  require(len(own_dtor)==len(own_handle)==1,'own compiler dtor/handle missing')
  dtor=next(iter(own_dtor));handle=next(iter(own_handle));dtext=body(dtor)
  require(dtext.count('<__hipUnregisterFatBinary>')==1 and set(handles)&targets(dtext)=={handle},'compiler destructor does not retire own handle')
  forwards=[]
  for target in re.findall(r'\b(?:jmp|callq?)\s+0x([0-9a-f]+)',text):
   address=int(target,16);raw,_=reader(address,6)
   if raw[:2]==b'\xff\x25' and address+6+struct.unpack_from('<i',raw,2)[0] in result['atexit_got']:forwards.append(address)
  require(len(forwards)==1,'each compiler destructor needs exactly one actual atexit registration')
  proof.append({'module':actor,'ctor':ctor,'dtor':dtor,'handle':handle,'atexit_plt':forwards[0]})
 require(len({p['dtor'] for p in proof})==len({p['handle'] for p in proof})==2,'compiler ownership aliases')
 main=body(unique['main']);entry=PROFILE['client_entry']
 require('<'+entry+'>' in main and entry in unique,'title does not call exact plain client entry')
 result.update(modules=proof,client_entry=entry);return result
