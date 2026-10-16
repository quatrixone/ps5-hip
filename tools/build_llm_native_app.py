"""Build a fresh authenticated vocabulary-only native application offline."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, threading, shlex, re
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dependencies.json').is_file())
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
import build_public_hip as b
import build_llm_native_client as client_builder
import build_llm_executed_graph as graph
from fetch_locked_deps import source,sources,read


def file_inputs(argv):
 """Existing explicit file operands, excluding compiler/converter output operands."""
 skipped={i+1 for i,a in enumerate(argv) if a in ('-o','--out','-MF')}
 return [Path(a) for i,a in enumerate(argv[1:],1) if i not in skipped and Path(a).is_file()]


def command_file_inputs(output,label,argv,title):
 persistent=[];scratch=[]
 for p in file_inputs(argv):
  ephemeral=(label in ('fsck','package-extract') and p.name==title+'.ffpkg' and p.parent.parent==Path(output)/'package' and p.parent.name.startswith('pack-'))
  (scratch if ephemeral else persistent).append(p)
 return persistent,scratch


def write_phase(path):
 Path(path).write_bytes(b'vocab\n')


def validate_vocab_asset(asset,evidence):
 spec=json.loads(Path(evidence).read_text())
 if spec.get('status')!='PASS_PRODUCTION_VOCABULARY_FIXTURE' or spec.get('tensor_count')!=0 or spec.get('metadata_fields')!=45 or spec.get('tokenization_cases')!=32:
  raise ValueError('unqualified vocabulary evidence')
 if Path(asset).stat().st_size!=spec['vocab_bytes'] or b.sha(asset)!=spec['asset_sha256']:raise ValueError('vocabulary differs from reviewed bytes')


def host_link_inputs(text):
 """Resolve the actual host driver's ELF inputs, including GNU INPUT/GROUP scripts."""
 plans=[shlex.split(line) for line in text.splitlines() if line.lstrip().startswith('"')]
 if len(plans)!=1 or not Path(plans[0][0]).is_file():raise ValueError('unexpected host linker driver plan')
 plan=plans[0];dirs=[Path(a[2:]).resolve() for a in plan if a.startswith('-L')];found=set()
 def resolve(name):
  candidates=([d/('lib'+name[2:]+suffix) for d in dirs for suffix in ('.so','.a')] if name.startswith('-l') else [Path(name)] if Path(name).is_absolute() else [d/name for d in dirs])
  p=next((p.resolve() for p in candidates if p.is_file()),None)
  if p is None:raise ValueError('unresolved host link input: '+name)
  if p in found:return
  found.add(p)
  with p.open('rb') as stream:magic=stream.read(8)
  if magic.startswith(b'\x7fELF') or magic==b'!<arch>\n':return
  script=re.sub(r'/\*.*?\*/','',p.read_text(),flags=re.S)
  script=re.sub(r'OUTPUT_FORMAT\s*\([^)]*\)','',script)
  tokens=re.findall(r'[^\s(),;]+',script)
  if not tokens or tokens[0] not in ('GROUP','INPUT'):raise ValueError('unsupported host linker script')
  for token in tokens:
   if token not in ('GROUP','INPUT','AS_NEEDED'):resolve(token)
 found.add(Path(plan[0]).resolve())
 skip={i+1 for i,a in enumerate(plan) if a=='-o'}
 for i,a in enumerate(plan[1:],1):
  if i not in skip and (a.startswith('-l') or Path(a).is_absolute()):resolve(a)
 return sorted(found)


def vocab_inputs(folder):
 folder=Path(folder).resolve();receipt=folder/'receipt.json';digest=b.sha(receipt);r=json.loads(receipt.read_text())
 if (r.get('schema')!=1 or r.get('status')!='PASS' or r.get('classification')!='upstream-vocabulary-only-gguf-tokenization' or r.get('tensor_count')!=0 or r.get('metadata_fields')!=45 or r.get('tokenization_cases')!=32 or r.get('original_and_vocabulary_tokens_bitwise_equal') is not True or r.get('native_executed') is not False or r.get('gpu_executed') is not False):raise ValueError('unsupported vocabulary fixture')
 inputs=r.get('inputs');artifacts=r.get('artifacts')
 if not isinstance(inputs,dict) or not inputs or not isinstance(artifacts,dict) or not artifacts:raise ValueError('missing vocabulary identities')
 for name,h in inputs.items():
  if not Path(name).is_absolute() or b.sha(name)!=h:raise ValueError('stale vocabulary input: '+name)
 for name in ('tools/build_llm_vocab_fixture.py','tests/llm_vocab_tokenize.cpp','tests/llm_vocab_fixture_test.py','tools/build_llm_executed_graph.py','llm-inputs.json','llm-conversion-inputs.json'):
  if inputs.get(str(ROOT/name))!=b.sha(ROOT/name):raise ValueError('missing vocabulary source: '+name)
 for name,h in artifacts.items():
  p=folder/name
  if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or not p.resolve().is_relative_to(folder) or b.sha(p)!=h:raise ValueError('vocabulary artifact differs')
 if set(artifacts)!={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=receipt}:raise ValueError('vocabulary artifact inventory differs')
 # Re-authenticate the actual model/CPU closure, not arbitrary reference bytes.
 models=[Path(p) for p in inputs if Path(p).name=='tinyllama-q4_0.gguf']
 cpus=[Path(p).parent for p in inputs if Path(p).name=='receipt.json' and (Path(p).parent/'cmake/bin/libllama.so').is_file()]
 if len(models)!=1 or len(cpus)!=1:raise ValueError('ambiguous vocabulary model/CPU identity')
 for p,h in graph.cpu_inputs(cpus[0],models[0]).items():
  if inputs.get(p)!=h:raise ValueError('vocabulary CPU/model closure differs')
 from build_llm_vocab_fixture import validate_tokens
 validate_tokens((folder/'original.stdout').read_text(),(folder/'vocabulary-only.stdout').read_text())
 asset=folder/'tinyllama-vocab.gguf'
 evidence=ROOT/'evidence/offline-llm-vocab-fixture.json';evidence_sha=b.sha(evidence);validate_vocab_asset(asset,evidence)
 inputs={**inputs,str(evidence):evidence_sha}
 if asset.name not in artifacts or not 0<asset.stat().st_size<16*1024*1024 or r['vocab_bytes']!=asset.stat().st_size:raise ValueError('invalid vocabulary asset size')
 if b.sha(receipt)!=digest:raise ValueError('vocabulary receipt changed')
 return {'asset':str(asset),'receipt':str(receipt),'receipt_sha256':digest,'inputs':{**inputs,**{str(folder/n):h for n,h in artifacts.items()},str(receipt):digest}}


def merged_unit_headers(units):
 headers={}
 for unit in units:
  for path,digest in unit['headers'].items():
   if path in headers and headers[path]!=digest:raise ValueError('conflicting compiled header identity: '+path)
   headers[path]=digest
 return headers


def build(output,native_client,vocab_fixture,results_dir):
 output=Path(output).resolve()
 if output.exists() or not (output.is_relative_to(ROOT/'build') or output.is_relative_to(ROOT/'results')):raise ValueError('fresh build/results output required')
 b.validate_results_directory(results_dir)
 if not __debug__:raise ValueError('ordinary Python required for upstream source preparers')
 for key in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX','SDKROOT','CFLAGS','CXXFLAGS','LDFLAGS'):os.environ.pop(key,None)
 os.environ['PATH']='/usr/bin:/bin';os.environ['LLVM_CONFIG']='/usr/bin/llvm-config-18'
 inputs={};links={};inventories={};lock=threading.RLock();commands=[];host_units=[]
 def bind(path,expected=None):
  path=Path(path).absolute();digest=b.sha(path)
  with lock:
   if expected is not None and digest!=expected:raise ValueError('identity differs: '+str(path))
   if str(path) in inputs and inputs[str(path)]!=digest:raise ValueError('input changed: '+str(path))
   inputs[str(path)]=digest
   if path.is_symlink():links[str(path)]=str(path.resolve())
  return path
 def merge(values):
  for p,h in values.items():bind(p,h)
 def tree(path):
  path=Path(path);files=[p for p in path.rglob('*') if p.is_file() and '.git' not in p.relative_to(path).parts and '__pycache__' not in p.parts]
  if not files:raise ValueError('empty dependency tree: '+str(path))
  for p in files:bind(p)
  inventories[str(path)]=sorted(str(p) for p in files)
 def recheck():
  for p,h in list(inputs.items()):
   if b.sha(p)!=h:raise ValueError('input changed before publication: '+p)
  for p,target in links.items():
   if str(Path(p).resolve())!=target:raise ValueError('input symlink changed: '+p)
  for path,names in inventories.items():
   if sorted(str(p) for p in Path(path).rglob('*') if p.is_file() and '.git' not in p.relative_to(path).parts and '__pycache__' not in p.parts)!=names:raise ValueError('dependency inventory changed: '+path)
 bind(Path(__file__));bind(ROOT/'tests/llm_native_app_build_test.py')
 for name,h in b.build_inputs().items():bind(ROOT/name,h)
 for module in list(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name and Path(name).is_file():bind(Path(name).resolve())
 required_sources=['tools/public_hip_llm_client_console_title.cpp','ports/ggml/ps5_backend.cpp','ports/ggml/ps5_backend.h','ports/ggml/model_gate.h','tools/audit_public_hip_llm_client_native.py']
 for name in required_sources:bind(ROOT/name)
 specs=sources()
 for name in ('clr','hip-headers','sdk','native-tools','ufs2tool','eden','lapy','ps5log'):source(name,specs[name],False)
 upstream_client=source('llama-client',read('llm-inputs.json')['client'],False)
 for name in ('clr','hip-headers'):tree(ROOT/'.deps/sources'/name)
 tree(ROOT/'.deps/sources/native-tools/tooling/native')
 tree(ROOT/'.deps/sdk');tree(ROOT/'.deps/elevation/client')
 deps=read('dependencies.json');title=read('title.json')['title_id']
 for p in [ROOT/'.deps/elevation/receipt.json',ROOT/'.deps/elevation/lapy/build'/('owned_root_helper-'+title)/'lapy.elf',ROOT/'.deps/sources/native-tools/runtime/libc.prx.sha256']:bind(p)
 tree(ROOT/'.deps'/('host-zlib-'+deps['host_zlib']['version'])/'root')
 ufs=ROOT/'.deps'/('ufs2tool-'+deps['ufs2tool']['revision']);tree(ufs/'bin');bind(ufs/'receipt.json')
 # Bind managed runtime used by UFS2Tool, as well as native executable/DSO bytes.
 tool_names=('clang++-18','ld.lld-18','llvm-config-18','llvm-readelf-18','llvm-objdump-18','llvm-nm-18','llvm-readobj-18','patch','git','dotnet','bash','sh','readlink','realpath','dirname','basename','cut','ldd','env')
 tools={}
 for name in tool_names:
  found=shutil.which(name)
  if found is None:raise ValueError('missing tool: '+name)
  p=bind(found);tools[name]={'path':str(p),'resolved':str(p.resolve()),'sha256':inputs[str(p)]}
  if name.endswith('-18'):
   version=subprocess.check_output([p,'--version'],text=True)
   if '18.1.8' not in version:raise ValueError('LLVM18.1.8 required')
   tools[name]['version']=version
  dsos=subprocess.run(['/usr/bin/ldd',p],capture_output=True,text=True)
  for line in dsos.stdout.splitlines():
   fields=line.split();lib=fields[2] if len(fields)>2 and fields[1]=='=>' else fields[0] if fields else ''
   if lib.startswith('/') and Path(lib).is_file():bind(lib)
 bind(sys.executable)
 dotnet=Path(tools['dotnet']['resolved']).parent
 tree(dotnet/'host/fxr');tree(dotnet/'shared/Microsoft.NETCore.App')
 def runtime(path):
  bind(path)
  result=subprocess.run(['/usr/bin/ldd',str(path)],capture_output=True,text=True)
  if 'not found' in result.stdout:raise ValueError('missing runtime library for '+str(path))
  if result.returncode and 'not a dynamic executable' not in result.stderr+result.stdout and 'statically linked' not in result.stderr+result.stdout:raise ValueError('cannot inspect runtime libraries: '+str(path))
  for line in result.stdout.splitlines():
   fields=line.split();name=fields[2] if len(fields)>2 and fields[1]=='=>' else fields[0] if fields else ''
   if name.startswith('/') and Path(name).is_file():bind(name)
 client=client_builder.app_objects(native_client);merge(client['inputs'])
 vocab=vocab_inputs(vocab_fixture);merge(vocab['inputs'])
 output.mkdir(parents=True)
 original_run=b.run
 def recorded_run(out,label,argv):
  argv=list(map(str,argv));executable=Path(argv[0]) if '/' in argv[0] else Path(shutil.which(argv[0]));bind(executable)
  if executable.is_relative_to(output):
   with executable.open('rb') as stream:magic=stream.read(4)
   if magic==b'\x7fELF':runtime(executable)
  # Helpers create converter/stub sources immediately before their compiler call.
  # Compile those TUs separately to obtain actual dependency files, then link.
  if label in ('native-tool','libc-builder') or label in ('stub-compile-libSceAgc','stub-compile-libSceAgcDriver'):
   sources_=[Path(a) for a in argv[1:] if Path(a).suffix in ('.cpp','.c') and Path(a).is_file()]
   outindex=argv.index('-o');destination=Path(argv[outindex+1]);flags=[];i=0
   while i<len(argv):
    a=argv[i]
    if a=='-o':i+=2;continue
    if a=='-c' or a in list(map(str,sources_)) or a.endswith('.a'):i+=1;continue
    flags.append(a);i+=1
   objects=[]
   for index,p in enumerate(sources_):
    bind(p);obj=destination if label in ('stub-compile-libSceAgc','stub-compile-libSceAgcDriver') else output/(label+'-'+str(index)+'.o')
    record=b.compile_bound_unit(output,p,obj,flags,label+'-tu-'+str(index));merge(record['headers']);bind(obj);host_units.append(record);objects.append(obj)
   if not label in ('stub-compile-libSceAgc','stub-compile-libSceAgcDriver'):
    recorded_run(out,label+'-link',[argv[0],*[str(p) for p in objects],*[a for a in argv if a.endswith('.a')],'-o',destination])
   return
  host_link=label in ('native-tool-link','libc-builder-link')
  if host_link:
   plan=subprocess.run([argv[0],'-###',*argv[1:]],capture_output=True,text=True)
   (output/(label+'.plan.stdout')).write_text(plan.stdout);(output/(label+'.plan.stderr')).write_text(plan.stderr)
   if plan.returncode:raise ValueError('host linker plan failed')
   planned=host_link_inputs(plan.stderr)
   linker=Path(shlex.split(next(line for line in plan.stderr.splitlines() if line.lstrip().startswith('"')))[0]).resolve()
   for p in planned:bind(p)
   runtime(linker)
   argv.insert(1,'-Wl,--trace')
  if label=='link' and '--trace' not in argv:argv.insert(1,'--trace')
  row={'label':label,'argv':argv}
  # Bind generated binaries, raw ELF inputs, linker scripts and package executables before use.
  persistent,scratch=command_file_inputs(output,label,argv,title)
  for p in persistent:bind(p)
  row['temporary_package_inputs']={str(p):b.sha(p) for p in scratch}
  try:
   original_run(out,label,argv);row['exit_code']=0
   for p,h in row['temporary_package_inputs'].items():
    if b.sha(p)!=h:raise ValueError('temporary package input changed during command')
   if label=='link' or host_link:
    selected=[]
    authenticated_paths={name:Path(name).resolve() for name in inputs}
    authenticated_targets=set(authenticated_paths.values())
    for line in (out/(label+'.stdout')).read_text().splitlines():
     name=line.split('(',1)[0].strip();p=Path(name)
     if not p.is_absolute():p=ROOT/p
     if p.is_file():
      if p.resolve() not in authenticated_targets:raise ValueError('unbound selected linker input: '+str(p))
      bind(p);selected.append(str(p))
    if not selected:raise ValueError('missing actual linker trace')
    for name,target in authenticated_paths.items():
     if Path(name).resolve()!=target:raise ValueError('link input resolution changed during validation')
    row['selected_inputs']=selected
  except BaseException as e:
   row['error']=type(e).__name__;raise
  finally:
   with lock:commands.append(row);(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 b.run=recorded_run
 try:
  public=b.source_tree(output,'llm_client');tree(public)
  units=b.compile_units(output,public,'llm_client')
  merge(merged_unit_headers(units))
  for unit in units:bind(output/unit['object'],unit['object_sha256'])
  flags=b.compiler_flags(public)+['-I'+str(upstream_client/n) for n in ('include','ggml/include','ggml/src')]+['-I'+str(ROOT/'ports/ggml')]
  backend=b.compile_bound_unit(output,ROOT/'ports/ggml/ps5_backend.cpp',output/'ps5_backend.o',flags,'compile-ps5-backend');merge(backend['headers']);bind(output/backend['object'],backend['object_sha256']);units.append(backend)
  units.extend({'object':p,'object_sha256':inputs[p],'source':'authenticated-native-client'} for p in client['objects'])
  if len(units)!=147:raise ValueError('expected147 app objects')
  b.native_stubs(output);recheck();b.link_title(output,units,'llm_client')
  bind(output/'title.elf');tool=b.native_tool(output);bind(tool);libc=b.clean_room_libc(output,tool);bind(libc)
  image,roundtrip=b.convert_title(output,tool,'llm_client')
  for name in ('eboot.elf','eboot.bin','extracted.elf'):bind(output/name)
  phase=output/'client-phase.txt';write_phase(phase);bind(phase)
  package,files=b.package_title(output,image,libc,results_dir,client_assets={'vocab.gguf':vocab['asset'],'client-phase.txt':phase})
  bind(package)
  scratch_rows=[c for c in commands if c.get('temporary_package_inputs')]
  if [c['label'] for c in scratch_rows]!=['fsck','package-extract'] or any(len(c['temporary_package_inputs'])!=1 or next(iter(c['temporary_package_inputs'].values()))!=b.sha(package) for c in scratch_rows):raise ValueError('temporary package evidence differs from final package')
  audit=output/'llm-client-native-audit.json'
  recorded_run(output,'audit-client',[sys.executable,ROOT/'tools/audit_public_hip_llm_client_native.py','--raw',output/'title.elf','--converted',output/'eboot.elf','--extracted',output/'extracted.elf','--output',audit])
  ar=json.loads(audit.read_text())
  if ar.get('status')!='PASS' or ar.get('classification')!='offline-public-hip-llm-client-native-image-audit' or ar.get('native_executed') is not False or ar.get('gpu_executed') is not False or ar.get('phase')!='vocab' or ar.get('actual_client_gate_executed') is not False:raise ValueError('native app audit not accepted')
  for p in [output/n for n in ('title.elf','eboot.elf','extracted.elf')]+[ROOT/'tools/audit_public_hip_llm_client_native.py']:
   if ar.get('inputs',{}).get(str(p))!=b.sha(p):raise ValueError('audit omits required identity')
  merge(ar['inputs']);bind(audit)
  # Re-run consumers to reject source or dependency changes during compilation.
  if client_builder.app_objects(native_client)!=client or vocab_inputs(vocab_fixture)!=vocab:raise ValueError('app prerequisite changed')
  for name in ('clr','hip-headers','sdk','native-tools','ufs2tool','eden','lapy','ps5log'):source(name,specs[name],False)
  recheck()
  artifacts={str(p.relative_to(output)):b.sha(p) for p in output.rglob('*') if p.is_file()}
  receipt={'schema':1,'status':'PASS','classification':'source-only-vocabulary-native-client-app','fixture':'llm_client','results_dir':results_dir,'phase':'vocab','llm_client':{'phase':'vocab','vocabulary':{k:vocab[k] for k in ('receipt','receipt_sha256','asset')},'native_client':{k:client[k] for k in ('receipt','receipt_sha256')},'audit':{'path':str(audit),'sha256':b.sha(audit)}},'inputs':inputs,'symlinks':links,'tools':tools,'units':units,'host_units':host_units,'object_count':147,'commands':commands,'roundtrip':roundtrip,'package_files':files,'package_sha256':b.sha(package),'artifacts':artifacts,'native_executed':False,'gpu_executed':False,'limits':['Vocabulary-only offline app; no complete model graph or inference.','Installed SDK, elevation and packaging runtime bytes bound; historical prerequisite build provenance inherited from deps-build.','No console execution or model operation support claim.']}
  (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
 finally:b.run=original_run

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('output','native-client','vocab-fixture'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--results-dir',required=True);a=parser.parse_args();build(a.output,a.native_client,a.vocab_fixture,a.results_dir)
