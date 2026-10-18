"""Build a fresh authenticated vocabulary-only native application offline."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys, threading, shlex, re
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dependencies.json').is_file())
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'tools'),str(ROOT/'tests')]
import build_public_hip as b
import build_llm_native_client as client_builder
import build_llm_executed_graph as graph
from fetch_locked_deps import source,sources,read
from external_model import DEFAULT_MODEL_PATH, validate_external_model, encode_model_path
from source_public_hip_log import package_limit


def validate_full_profile(info,model):
 if info['model']['sha256']!=model['sha256'] or info['model']['bytes']!=model['bytes'] or info['profile'] not in ('smoke','generation','corpus','extended'):raise ValueError('full/model profile identity differs')
 if info['profile']=='generation' and (len(info['raw_files'])!=386 or info['raw_files'].get('generation.json')!=590 or info['raw_files'].get('steps.jsonl')!=1108):raise ValueError('generation title metadata contract differs')
 if info['profile']=='corpus':
  # The authenticated inventory must fit the fixed native title's output contract.
  from build_llm_generation_reference import corpus_cases
  files=info['raw_files'];metadata={'corpus.json',*(case+'/'+name for case,_ in corpus_cases() for name in ('generation.json','steps.jsonl'))}
  if len(files)!=6539 or not metadata<=files.keys() or sum(files.values())!=86091236 or sum(files[n] for n in metadata)!=26084:raise ValueError('corpus title metadata contract differs')
 if info['profile']=='extended':
  raise ValueError('extended native watchdog not qualified from measured corpus timing')
 return info['profile']


def validate_dense_audit(audit):
 if audit.get('phase')!='dense-layer' or audit.get('actual_dense_call_chain_audited') is not True:raise ValueError('dense call chain not audited')


def file_inputs(argv,cwd=None):
 """Existing explicit file operands, excluding compiler/converter output operands."""
 skipped={i+1 for i,a in enumerate(argv) if a in ('-o','--out','-MF')}
 base=ROOT if cwd is None else Path(cwd)
 return [p for i,a in enumerate(argv[1:],1) if i not in skipped
         for p in [Path(a) if Path(a).is_absolute() else base/a] if p.is_file()]


def command_file_inputs(output,label,argv,title,cwd=None):
 persistent=[];scratch=[]
 for p in file_inputs(argv,cwd):
  ephemeral=(label in ('fsck','package-extract') and p.name==title+'.ffpkg' and p.parent.parent==Path(output)/'package' and p.parent.name.startswith('pack-'))
  (scratch if ephemeral else persistent).append(p)
 return persistent,scratch


def write_phase(path,phase='vocab'):
 if phase not in ('vocab','model-gate','model-prefix','model-layer','model-full','dense-layer'):raise ValueError('invalid app phase')
 Path(path).write_bytes((phase+'\n').encode())


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


def validate_phase_arguments(phase,model,model_source,cpu_build,prefix_fixture=None,layer_fixture=None,full_fixture=None,dense_fixture=None):
 if (phase=='dense-layer') != (dense_fixture is not None):raise ValueError('dense fixture does not match phase')
 if (phase=='model-full') != (full_fixture is not None):raise ValueError('full fixture does not match phase')
 if (phase=='model-layer') != (layer_fixture is not None):raise ValueError('layer fixture does not match phase')
 if (phase=='model-prefix') != (prefix_fixture is not None):raise ValueError('prefix fixture does not match phase')
 if phase not in ('vocab','model-gate','model-prefix','model-layer','model-full','dense-layer'):raise ValueError('invalid app phase')
 supplied=(model is not None,model_source is not None,cpu_build is not None)
 if (phase in ('vocab','dense-layer') and any(supplied)) or (phase in ('model-gate','model-prefix','model-layer','model-full') and not all(supplied)):raise ValueError('model prerequisites do not match phase')


def model_inputs(model,source_dir,cpu_build,*,authenticated_original_inputs=None):
 from fetch_llm_model import verify
 model=Path(model).absolute();source_dir=Path(source_dir).absolute();cpu_build=Path(cpu_build).resolve()
 # Check before resolving so aliases cannot conceal a symlink asset.
 if model.is_symlink() or not model.is_file() or model.resolve()!=model:raise ValueError('model must be a regular non-aliased file')
 evidence_path=ROOT/'evidence/offline-llm-model-conversion.json';evidence=json.loads(evidence_path.read_text());expected=evidence['outputs']['tinyllama-q4_0.gguf']
 if expected['bytes']!=619831552 or expected['sha256']!='c42ff3ebdfc0b8a0aa0c35a499ccf81303fb9fbdc1a0798c096a098df85003aa':raise ValueError('unexpected model profile')
 if model.stat().st_size!=expected['bytes']:raise ValueError('model size differs')
 receipt=model.parent/'receipt.json';receipt_hash=b.sha(receipt);r=json.loads(receipt.read_text())
 model_lock_path=ROOT/'models/tinyllama-1.1b-chat-v1.0.json';model_lock=json.loads(model_lock_path.read_text());client_lock=read('llm-inputs.json')['client']
 if r.get('schema')!=1 or r.get('classification')!='host-model-conversion-verified' or r.get('model_inference_executed') is not False or r.get('native_executed') is not False or r.get('model_revision')!=model_lock['revision'] or r.get('source_revision')!=client_lock['revision']:raise ValueError('invalid model conversion receipt')
 identities=r.get('identities')
 if not isinstance(identities,dict) or not identities:raise ValueError('missing conversion identities')
 for name,h in identities.items():
  if not Path(name).is_absolute() or b.sha(name)!=h:raise ValueError('stale conversion identity')
 required=[ROOT/n for n in ('llm-inputs.json','llm-conversion-inputs.json','tools/convert_llm_model.py','tools/fetch_llm_model.py')]+[model_lock_path,cpu_build/'receipt.json']
 for p in required:
  if identities.get(str(p))!=b.sha(p):raise ValueError('missing conversion prerequisite')
 environment_receipts=[Path(p) for p in identities if Path(p).name=='receipt.json' and Path(p)!=cpu_build/'receipt.json']
 if len(environment_receipts)!=1:raise ValueError('conversion environment receipt missing')
 environment=json.loads(environment_receipts[0].read_text())
 if environment.get('lock_sha256')!=b.sha(ROOT/'llm-conversion-inputs.json'):raise ValueError('conversion environment lock differs')
 output=r.get('outputs',{}).get('tinyllama-q4_0.gguf',{})
 for key in ('bytes','sha256','metadata','dtype_counts','tensor_bytes'):
  if output.get(key)!=expected[key]:raise ValueError('converted model metadata/profile differs')
 tensors=output.get('tensors',[])
 if len(tensors)!=expected['tensor_count'] or len({t['name'] for t in tensors})!=len(tensors) or sum(t['elements'] for t in tensors)!=expected['parameters']:raise ValueError('converted tensor inventory differs')
 commands=r.get('commands',[])
 if [c.get('label') for c in commands]!=['convert-f16','quantize-q4_0'] or any(c.get('exit_code')!=0 or c.get('timed_out') for c in commands):raise ValueError('conversion did not succeed')
 if Path(commands[0]['argv'][2])!=source_dir or Path(commands[1]['argv'][0])!=cpu_build/'cmake/bin/llama-quantize' or Path(commands[1]['argv'][3])!=model:raise ValueError('conversion command identities differ')
 inputs={**identities,str(receipt):receipt_hash,str(evidence_path):b.sha(evidence_path)}
 # Existing helper verifies clean locked client, complete CPU/tool closure and exact GGUF digest.
 for p,h in graph.cpu_inputs(cpu_build,model).items():
  if p in inputs and inputs[p]!=h:raise ValueError('model prerequisite identity conflict')
  inputs[p]=h
 # Only the enclosing admission may supply its freshly byte-verified snapshot;
 # it must still finish its own complete fresh sweep after this semantic check.
 def original_file(path,size,digest):
  if authenticated_original_inputs is None:
   verify(path,size,digest)
  else:
   if path.is_symlink() or not path.is_file() or path.stat().st_size!=size:raise ValueError('incorrect original file type or size: '+str(path))
   if authenticated_original_inputs.get(str(path))!=digest:raise ValueError('original file absent or different in authenticated inputs: '+str(path))
  inputs[str(path)]=digest
 for name,spec in model_lock['metadata'].items():
  original_file(source_dir/name,spec['bytes'],spec['sha256'])
 weights=model_lock['weights'];original_file(source_dir/'model.safetensors',weights['file_bytes'],weights['expected_file_sha256'])
 if b.sha(receipt)!=receipt_hash:raise ValueError('conversion receipt changed')
 return {'asset':str(model),'source_dir':str(source_dir),'cpu_build':str(cpu_build),'bytes':expected['bytes'],'sha256':expected['sha256'],'receipt':str(receipt),'receipt_sha256':receipt_hash,'inputs':inputs}


def merged_unit_headers(units):
 headers={}
 for unit in units:
  for path,digest in unit['headers'].items():
   if path in headers and headers[path]!=digest:raise ValueError('conflicting compiled header identity: '+path)
   headers[path]=digest
 return headers


def select_prepared_source(output, fresh, previous):
 """Reuse only a byte-equivalent prepared tree, preserving its compiler paths."""
 if previous is None:return fresh
 previous=Path(previous).absolute()
 if not any(previous.is_relative_to(ROOT/n) for n in ('build','results')) or previous==output:
  raise ValueError('prepared source build must be a distinct build/results directory')
 if any(p.is_symlink() for p in (previous,*previous.parents)) or previous.resolve()!=previous:
  raise ValueError('prepared source build path is ambiguous')
 old=previous/'source-public'
 for folder in (previous,output):
  receipt=folder/'source-base/receipt.json'
  if any(p.is_symlink() for p in (receipt,*receipt.parents)) or receipt.resolve()!=receipt:raise ValueError('ambiguous prepared source base receipt')
  if not receipt.is_file():raise ValueError('prepared source base receipt missing')
  data=json.loads(receipt.read_text())
  if data.get('classification')!='exact-combined-native-clr-gpu-source':raise ValueError('prepared source base receipt differs')
 if (previous/'source-base/receipt.json').read_bytes()!=(output/'source-base/receipt.json').read_bytes():
  raise ValueError('prepared source base identity differs')
 def inventory(folder):
  if not folder.is_dir():raise ValueError('prepared source tree missing')
  paths=list(folder.rglob('*'))
  if any(p.is_symlink() for p in [folder,*paths]):raise ValueError('prepared source tree symlink')
  return {str(p.relative_to(folder)):b.sha(p) for p in paths if p.is_file()}
 old_files,new_files=inventory(old),inventory(fresh)
 name='public-hip-preparation.json'
 if set(old_files)!=set(new_files) or name not in old_files:raise ValueError('prepared source inventory differs')
 old_receipt=json.loads((old/name).read_text());new_receipt=json.loads((fresh/name).read_text())
 if old_receipt.get('classification')!='exact-public-hip-patch-roundtrip':raise ValueError('prepared public receipt differs')
 # Only this producer's source-base input keys legitimately depend on output location.
 inputs={}
 for path,digest in old_receipt['inputs'].items():
  candidate=Path(path)
  if candidate.is_relative_to(previous/'source-base/source'):
   path=str(output/'source-base/source'/candidate.relative_to(previous/'source-base/source'))
  elif str(previous) in path:raise ValueError('unexpected prepared receipt reference')
  if path in inputs:raise ValueError('ambiguous prepared receipt reference')
  inputs[path]=digest
 old_receipt['inputs']=inputs
 if old_receipt!=new_receipt:raise ValueError('prepared public receipt identity differs')
 if any(old_files[n]!=new_files[n] for n in old_files if n!=name):raise ValueError('prepared source content differs')
 return old


def build(output,native_client,vocab_fixture,results_dir,phase='vocab',model=None,model_source=None,cpu_build=None,model_remote_path=DEFAULT_MODEL_PATH,prefix_fixture=None,prepared_source_build=None,layer_fixture=None,full_fixture=None,model_format="gguf",dense_fixture=None):
 if model_format not in ('gguf','original_safetensors') or (model_format!='gguf' and (phase!='model-full' or model_source is None)):
  raise ValueError('original format requires full model and original source directory')
 validate_phase_arguments(phase,model,model_source,cpu_build,prefix_fixture,layer_fixture,full_fixture,dense_fixture)
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
 def merge_authenticated(values):
  # Successful prerequisite consumers already hashed these bytes; final recheck stays fresh.
  for name,digest in values.items():
   path=Path(name)
   if not path.is_absolute() or not path.is_file():raise ValueError('invalid authenticated input path: '+name)
   with lock:
    if name in inputs and inputs[name]!=digest:raise ValueError('input changed: '+name)
    inputs[name]=digest
    if path.is_symlink():links[name]=str(path.resolve())
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
 bind(Path(__file__));bind(ROOT/'tests/llm_native_app_build_test.py');bind(ROOT/'tests/watchdog_test.py')
 bind(ROOT/'tests/native_model_path_test.py');bind(ROOT/'tests/source_external_model_test.py')
 model_test=Path(__file__).with_name('test_model_package.py')
 bind(model_test if model_test.is_file() else ROOT/'tests/llm_native_model_package_test.py')
 cache_record,_=b.cache.configuration(ROOT,os.environ)
 for name,h in b.cache.bound_inputs(cache_record).items():bind(name,h)
 bind("/usr/bin/python3")
 for name,h in b.build_inputs().items():bind(ROOT/name,h)
 for module in list(sys.modules.values()):
  name=getattr(module,'__file__',None)
  if name and Path(name).is_file():bind(Path(name).resolve())
 required_sources=['tools/public_hip_llm_client_console_title.cpp','ports/ggml/ps5_backend.cpp','ports/ggml/ps5_backend.h','ports/ggml/model_gate.h','tools/audit_public_hip_llm_client_native.py']
 if phase in ('model-gate','model-prefix','model-layer','model-full','dense-layer'):
  required_sources+=['ports/ggml/native_host_allocator.cpp','ports/ggml/native_host_allocator.h',
   'tests/llm_native_host_allocator_test.py','tests/llm_native_scheduler_test.py',
   'src/native_memory.cpp','include/ps5hip/native_memory.h']
 if phase=='dense-layer':required_sources += ['ports/ggml/native_dense_layer.h','ports/ggml/dense_layer_qualification.h','ports/ggml/dense_layer_fixture.h','ports/ggml/dense_layer_cases.def','ports/ggml/layer_qualification.h','ports/ggml/model_layer.h','ports/ggml/layer_dispatch.h','ports/ggml/prefix_dispatch.h','ports/ggml/prefix_calls.h','tools/public_hip_dense_layer_assets.py','tests/native_dense_app_build_test.py']
 if phase=='model-prefix':required_sources += ['ports/ggml/native_prefix.h','ports/ggml/prefix_qualification.h','ports/ggml/model_prefix.h','ports/ggml/prefix_dispatch.h','ports/ggml/prefix_calls.h','tools/public_hip_prefix_assets.py']
 if phase=='model-layer':required_sources += ['ports/ggml/native_layer.h','ports/ggml/layer_qualification.h','ports/ggml/model_layer.h','ports/ggml/layer_dispatch.h','ports/ggml/prefix_dispatch.h','ports/ggml/prefix_calls.h','tools/public_hip_layer_assets.py']
 if phase=='model-full':required_sources += ['ports/ggml/extended_generation.h','ports/ggml/extended_generation_cases.def','ports/ggml/extended_title_output.h','ports/ggml/native_full.h','ports/ggml/full_qualification.h','ports/ggml/full_dispatch.h','ports/ggml/full_graph.h','ports/ggml/layer_dispatch.h','ports/ggml/model_layer.h','ports/ggml/native_prefix.h','ports/ggml/prefix_dispatch.h','ports/ggml/prefix_calls.h','ports/ggml/model_prefix.h','tools/public_hip_full_assets.py']
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
 client=client_builder.app_objects(native_client);merge_authenticated(client['inputs'])
 vocab=vocab_inputs(vocab_fixture);merge_authenticated(vocab['inputs'])
 model_info=model_inputs(model,model_source,cpu_build) if phase in ('model-gate','model-prefix','model-layer','model-full') else None
 external_model={'path':model_remote_path,'bytes':model_info['bytes'],'sha256':model_info['sha256']} if model_info and model_format=='gguf' else None
 if external_model:validate_external_model(external_model)
 external_original_model=None
 if model_format=='original_safetensors':
  from external_original_model import contract
  external_original_model=contract(model_source,model_remote_path,read('models/tinyllama-1.1b-chat-v1.0.json'))
  for name in external_original_model['files']:bind(Path(model_source)/name)
  for name in ('models/tinyllama-1.1b-chat-v1.0.json','tools/external_original_model.py','ports/ggml/native_original_config.h','ports/ggml/tinyllama_source_metadata.h','ports/ggml/tinyllama_tensor_callback.h','ports/ggml/tinyllama_target_reader.h','ports/ggml/tinyllama_source_tensors.h','ports/ggml/safetensors_reader.h','tests/native_original_title_test.py'):bind(ROOT/name)

 if model_info:merge_authenticated(model_info['inputs'])
 prefix_info=None
 if phase=='model-prefix':
  from public_hip_prefix_assets import prefix_inputs
  prefix_info=prefix_inputs(prefix_fixture);merge_authenticated(prefix_info['inputs'])
  if prefix_info['model']['sha256']!=model_info['sha256'] or prefix_info['model']['bytes']!=model_info['bytes']:raise ValueError('prefix/model identity differs')
 layer_info=None
 if phase=='model-layer':
  from public_hip_layer_assets import layer_inputs
  layer_info=layer_inputs(layer_fixture);merge_authenticated(layer_info['inputs'])
  if layer_info['model']['sha256']!=model_info['sha256'] or layer_info['model']['bytes']!=model_info['bytes']:raise ValueError('layer/model identity differs')
 full_info=None
 if phase=='model-full':
  from public_hip_full_assets import full_inputs
  full_info=full_inputs(full_fixture);merge_authenticated(full_info['inputs'])
  validate_full_profile(full_info,model_info)
  if model_format!='gguf' and full_info['profile']!='generation':raise ValueError('original format requires short generation')
 dense_info=None
 if phase=='dense-layer':
  from public_hip_dense_layer_assets import dense_inputs
  dense_info=dense_inputs(dense_fixture);merge_authenticated(dense_info['inputs'])
 output.mkdir(parents=True)
 original_run=b.run
 def recorded_run(out,label,argv,cwd=None):
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
  row={'label':label,'argv':argv,'cwd':str(ROOT if cwd is None else cwd)}
  # Bind generated binaries, raw ELF inputs, linker scripts and package executables before use.
  persistent,scratch=command_file_inputs(output,label,argv,title,cwd=cwd)
  for p in persistent:bind(p)
  row['temporary_package_inputs']={str(p):b.sha(p) for p in scratch}
  try:
   original_run(out,label,argv,**({} if cwd is None else {'cwd':cwd}));row['exit_code']=0
   for p,h in row['temporary_package_inputs'].items():
    if b.sha(p)!=h:raise ValueError('temporary package input changed during command')
   if label=='link' or host_link:
    selected=[];authenticated=dict(inputs);authenticated_paths={};checked=set()
    def resolved(name):
     if name not in authenticated_paths:authenticated_paths[name]=Path(name).resolve()
     return authenticated_paths[name]
    for line in (out/(label+'.stdout')).read_text().splitlines():
     name=line.split('(',1)[0].strip();p=Path(name)
     if not p.is_absolute():p=ROOT/p
     if p.is_file():
      name=str(p);target=resolved(name)
      if name not in checked:
       identity=name if name in authenticated else str(target) if str(target) in authenticated else None
       if identity is None:
        identity=next((n for n in authenticated if resolved(n)==target),None)
       if identity is None:raise ValueError('unbound selected linker input: '+str(p))
       resolved(identity)
       bind(p,authenticated[identity]);checked.add(name)
      selected.append(name)
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
  public=select_prepared_source(output,public,prepared_source_build)
  reused_source=public!=output/'source-public'
  if reused_source:
   tree(public);bind(public.parent/'source-base/receipt.json');bind(output/'source-base/receipt.json')
  units=b.compile_units(output,public,'llm_client',client_phase=phase,full_profile=full_info['profile'] if full_info else 'smoke',**({'model_format':model_format} if model_format!='gguf' else {}))
  merge(merged_unit_headers(units))
  for unit in units:bind(output/unit['object'],unit['object_sha256'])
  flags=b.compiler_flags(public)+['-I'+str(upstream_client/n) for n in ('include','ggml/include','ggml/src')]+['-I'+str(ROOT/'ports/ggml')]
  backend=b.compile_bound_unit(output,ROOT/'ports/ggml/ps5_backend.cpp',output/'ps5_backend.o',flags,'compile-ps5-backend');merge(backend['headers']);bind(output/backend['object'],backend['object_sha256']);units.append(backend)
  if phase in ('model-gate','model-prefix','model-layer','model-full','dense-layer'):
   # llama_context destruction references the upstream optimizer cleanup even without training.
   opt=b.compile_bound_unit(output,upstream_client/'ggml/src/ggml-opt.cpp',output/'ggml-opt.o',flags,'compile-ggml-opt');merge(opt['headers']);bind(output/opt['object'],opt['object_sha256']);units.append(opt)
   allocator=b.compile_bound_unit(output,ROOT/'ports/ggml/native_host_allocator.cpp',output/'native_host_allocator.o',flags,'compile-native-host-allocator');merge(allocator['headers']);bind(output/allocator['object'],allocator['object_sha256']);units.append(allocator)
  units.extend({'object':p,'object_sha256':inputs[p],'source':'authenticated-native-client'} for p in client['objects'])
  if len(units)!=(149 if phase in ('model-gate','model-prefix','model-layer','model-full','dense-layer') else 147):raise ValueError('unexpected app object count')
  b.native_stubs(output);recheck();b.link_title(output,units,'llm_client',native_host_allocator=(phase in ('model-gate','model-prefix','model-layer','model-full','dense-layer')))
  bind(output/'title.elf');tool=b.native_tool(output);bind(tool);libc=b.clean_room_libc(output,tool);bind(libc)
  image,roundtrip=b.convert_title(output,tool,'llm_client')
  for name in ('eboot.elf','eboot.bin','extracted.elf'):bind(output/name)
  phase_file=output/'client-phase.txt';write_phase(phase_file,phase);bind(phase_file)
  assets={'client-phase.txt':phase_file}
  if phase!='dense-layer':assets['vocab.gguf']=vocab['asset']
  if dense_info:assets.update(dense_info['assets'])
  if external_original_model:
   from external_original_model import encode
   original_config=output/'original-model.json';original_config.write_bytes(encode(external_original_model));bind(original_config);assets['original-model.json']=original_config
  if external_model:
   model_path_file=output/'model-path.txt';model_path_file.write_bytes(encode_model_path(external_model['path']));bind(model_path_file)
   assets['model-path.txt']=model_path_file
  if prefix_info:assets.update(prefix_info['assets'])
  if layer_info:assets.update(layer_info['assets'])
  if full_info:assets.update(full_info['assets'])
  package,files=b.package_title(output,image,libc,results_dir,client_assets=assets)
  if not 0<package.stat().st_size<=package_limit('llm_client'):raise ValueError('native client package exceeds small package limit')
  bind(package)
  scratch_rows=[c for c in commands if c.get('temporary_package_inputs')]
  if [c['label'] for c in scratch_rows]!=['fsck','package-extract'] or any(len(c['temporary_package_inputs'])!=1 or next(iter(c['temporary_package_inputs'].values()))!=b.sha(package) for c in scratch_rows):raise ValueError('temporary package evidence differs from final package')
  audit=output/'llm-client-native-audit.json'
  recorded_run(output,'audit-client',[sys.executable,ROOT/'tools/audit_public_hip_llm_client_native.py','--raw',output/'title.elf','--converted',output/'eboot.elf','--extracted',output/'extracted.elf','--output',audit,*(['--phase',phase] if phase in ('model-gate','model-prefix','model-layer','model-full','dense-layer') else []),*(['--full-profile',full_info['profile']] if full_info else []),*(['--model-format',model_format] if model_format!='gguf' else [])])
  ar=json.loads(audit.read_text())
  if ar.get('status')!='PASS' or ar.get('classification')!='offline-public-hip-llm-client-native-image-audit' or ar.get('native_executed') is not False or ar.get('gpu_executed') is not False or ar.get('phase')!=phase or ar.get('actual_client_gate_executed') is not False:raise ValueError('native app audit not accepted')
  if phase in ('model-gate','model-prefix','model-layer','model-full') and ar.get('actual_client_call_chain_audited') is not True:raise ValueError('actual model call chain not audited')
  if phase=='dense-layer':validate_dense_audit(ar)
  if phase=='model-prefix' and ar.get('actual_prefix_call_chain_audited') is not True:raise ValueError('actual prefix call chain not audited')
  if phase=='model-layer' and ar.get('actual_layer_call_chain_audited') is not True:raise ValueError('actual layer call chain not audited')
  if phase=='model-full' and (ar.get('actual_full_call_chain_audited') is not True or ar.get('full_profile')!=full_info['profile']):raise ValueError('actual full profile/call chain not audited')
  if full_info and full_info['profile'] in ('generation','corpus','extended') and ar.get('actual_generation_call_chain_audited') is not True:raise ValueError('actual generation call chain not audited')
  for p in [output/n for n in ('title.elf','eboot.elf','extracted.elf')]+[ROOT/'tools/audit_public_hip_llm_client_native.py']:
   if ar.get('inputs',{}).get(str(p))!=b.sha(p):raise ValueError('audit omits required identity')
  if external_original_model and (ar.get('model_format')!=model_format or ar.get('original_model_call_chain_audited') is not True):raise ValueError('original model call chain not audited')
  merge(ar['inputs']);bind(audit)
  # Source-closure authentication must inspect the original runner, not our instrumentation.
  b.run=original_run
  # Re-run consumers to reject source or dependency changes during compilation.
  if client_builder.app_objects(native_client)!=client or vocab_inputs(vocab_fixture)!=vocab or (model_info and model_inputs(model,model_source,cpu_build)!=model_info):raise ValueError('app prerequisite changed')
  if prefix_info and prefix_inputs(prefix_fixture)!=prefix_info:raise ValueError('prefix prerequisite changed')
  if layer_info and layer_inputs(layer_fixture)!=layer_info:raise ValueError('layer prerequisite changed')
  if dense_info and dense_inputs(dense_fixture)!=dense_info:raise ValueError('dense prerequisite changed')
  if full_info and full_inputs(full_fixture)!=full_info:raise ValueError('full prerequisite changed')
  for name in ('clr','hip-headers','sdk','native-tools','ufs2tool','eden','lapy','ps5log'):source(name,specs[name],False)
  if reused_source:select_prepared_source(output,output/'source-public',public.parent)
  recheck()
  artifacts={str(p.relative_to(output)):b.sha(p) for p in output.rglob('*') if p.is_file()}
  receipt={'schema':1,'status':'PASS','classification':'source-only-synthetic-dense-layer-native-client-app' if phase=='dense-layer' else 'source-only-vocabulary-native-client-app' if phase=='vocab' else 'source-only-model-prefix-native-client-app' if phase=='model-prefix' else 'source-only-model-layer-native-client-app' if phase=='model-layer' else 'source-only-model-full-native-client-app' if phase=='model-full' else 'source-only-model-rejection-native-client-app','fixture':'llm_client','results_dir':results_dir,'phase':phase,'llm_client':{'phase':phase,'model':{k:model_info[k] for k in ('asset','source_dir','cpu_build','bytes','sha256','receipt','receipt_sha256')} if model_info else None,'vocabulary':{k:vocab[k] for k in ('receipt','receipt_sha256','asset')},'native_client':{k:client[k] for k in ('receipt','receipt_sha256')},'audit':{'path':str(audit),'sha256':b.sha(audit)}},'inputs':inputs,'symlinks':links,'tools':tools,'units':units,'host_units':host_units,'object_count':len(units),'commands':commands,'roundtrip':roundtrip,'package_files':files,'package_sha256':b.sha(package),'package_bytes':package.stat().st_size,'artifacts':artifacts,'native_executed':False,'gpu_executed':False,'limits':['Offline '+phase+' app; model-gate tests rejection only, not inference.','Installed SDK, elevation and packaging runtime bytes bound; historical prerequisite build provenance inherited from deps-build.','No console execution or model operation support claim.']}
  b.cache.verify(ROOT,cache_record)
  receipt['compiler_cache']=cache_record
  if reused_source:receipt['prepared_source_build']=str(public.parent)
  if dense_info:receipt['llm_client']['dense']={k:dense_info[k] for k in ('receipt','receipt_sha256','reference')}
  if prefix_info:receipt['llm_client']['prefix']={k:prefix_info[k] for k in ('receipt','receipt_sha256','reference')}
  if layer_info:receipt['llm_client']['layer']={k:layer_info[k] for k in ('receipt','receipt_sha256','reference')}
  if full_info:receipt['llm_client']['full']={k:full_info[k] for k in ('receipt','receipt_sha256','reference','profile')}
  if external_original_model:
   if contract(model_source,model_remote_path,read('models/tinyllama-1.1b-chat-v1.0.json'))!=external_original_model:raise ValueError('original source contract changed')
   receipt['external_original_model']=external_original_model
   receipt['model_format']='original_safetensors'
  if external_model:receipt['external_model']=external_model
  (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  return receipt
 finally:b.run=original_run

if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('output','native-client','vocab-fixture'):parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--results-dir',required=True);parser.add_argument('--phase',choices=('vocab','model-gate','model-prefix','model-layer','model-full','dense-layer'),default='vocab')
 for name in ('model','model-source','cpu-build'):parser.add_argument('--'+name,type=Path)
 parser.add_argument('--prefix-fixture',type=Path)
 parser.add_argument('--layer-fixture',type=Path)
 parser.add_argument('--full-fixture',type=Path)
 parser.add_argument('--dense-fixture',type=Path)
 parser.add_argument('--prepared-source-build',type=Path,help='Reuse compiler paths only after fresh preparation proves identical source bytes')
 parser.add_argument('--model-remote-path',default=DEFAULT_MODEL_PATH)
 parser.add_argument('--model-format',choices=('gguf','original_safetensors'),default='gguf')
 a=parser.parse_args();build(a.output,a.native_client,a.vocab_fixture,a.results_dir,a.phase,a.model,a.model_source,a.cpu_build,a.model_remote_path,a.prefix_fixture,a.prepared_source_build,a.layer_fixture,a.full_fixture,a.model_format,a.dense_fixture)
