"""Authenticate model-rejection packages and frozen controller dependencies."""
import hashlib,json,re
from pathlib import Path
from external_model import validate_external_model, encode_model_path
from source_public_hip_log import package_limit
from source_model_streaming import require,CHUNK
MODEL_BYTES=619831552
MODEL_PACKAGE_LIMIT=1024*1024*1024  # UFS metadata and 20% free space; exact image size/hash still required.
MODEL_SHA='c42ff3ebdfc0b8a0aa0c35a499ccf81303fb9fbdc1a0798c096a098df85003aa'
def stream_digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
 return h.hexdigest()
def model_contract(manifest):
 require(manifest.get('fixture')=='llm_client' and manifest.get('llm_client',{}).get('phase') in ('model-gate','model-prefix'),'wrong model phase')
 model=manifest['llm_client']['model']
 require(model['bytes']==MODEL_BYTES and model['sha256']==MODEL_SHA,'wrong locked model')
 t=manifest['model_transfer'];b=manifest['baseline_transfer']
 external=manifest.get('external_model')
 if external is not None:validate_external_model(external)
 # Historical embedded transfer contracts remain readable; new receipt admission requires external storage.
 require(type(t['bytes']) is int and (0<t['bytes']<=package_limit('llm_client') if external is not None else MODEL_BYTES<t['bytes']<=MODEL_PACKAGE_LIMIT),'model package bound')
 for v in (t,b):
  require(type(v['bytes']) is int and v['bytes']>0 and re.fullmatch('[0-9a-f]{64}',v['sha256']) and 0<v['deadline_seconds']<=1800,'invalid exact transfer')
 require(b['sha256']==manifest['baseline_image_sha256'],'baseline digest differs')
 # Legacy/vocab baselines remain bounded; larger baseline needs independently authenticated model phase.
 if b['bytes']>package_limit(manifest.get('baseline_log_fixture','bootstrap')):
  path=Path(b['receipt']);require(stream_digest(path)==b['receipt_sha256'],'large baseline receipt differs')
  receipt=validate_historical_baseline(path)
  require(receipt['package_bytes']==b['bytes'] and receipt['package_sha256']==b['sha256'],'large baseline identity differs')
ROOT=Path(__file__).resolve().parents[1]
def validate_historical_baseline(path):
 """Authenticate a previously reviewed deployment for recovery, never execution."""
 import freeze_source_candidate as legacy
 path=Path(path).resolve();folder=path.parent
 require(path.name=='build-receipt.json','baseline requires frozen candidate build receipt')
 receipt=json.loads(path.read_text());manifest_path=folder/'manifest.json'
 manifest=json.loads(manifest_path.read_text());deployment_path=folder/'deploy/receipt.json'
 deployment=json.loads(deployment_path.read_text());review=json.loads((folder/'review.json').read_text())
 require(review.get('approved') is True and review.get('manifest_sha256')==stream_digest(manifest_path) and review.get('deployment_receipt_sha256')==stream_digest(deployment_path),'baseline deployment review differs')
 require(deployment.get('phase')=='deploy' and all(deployment.get(k) is True for k in ('accepted','promoted','healthy_after','lock_released')) and deployment.get('launched') is False,'baseline deployment not accepted')
 require(deployment.get('host')==manifest.get('host')=='192.168.4.40' and deployment.get('title_id')==manifest.get('title_id')=='PPSA99014' and deployment.get('manifest_sha256')==review['manifest_sha256'],'baseline environment differs')
 require(receipt.get('schema')==1 and receipt.get('status')=='PASS' and receipt.get('classification')=='source-only-model-rejection-native-client-app' and receipt.get('fixture')=='llm_client' and receipt.get('phase')=='model-gate','baseline receipt phase differs')
 require(receipt.get('native_executed') is False and receipt.get('gpu_executed') is False,'baseline receipt execution claims')
 bound=receipt['llm_client'];model=bound['model']
 require(bound.get('phase')=='model-gate' and model['bytes']==MODEL_BYTES and model['sha256']==MODEL_SHA and manifest.get('llm_client')==bound,'baseline locked model differs')
 files=manifest['files']
 def frozen(p,expected):
  require(p.resolve().is_relative_to(ROOT) and not p.is_symlink(),'baseline artifact path')
  require(files.get(str(p.relative_to(ROOT)))==expected and stream_digest(p)==expected,'baseline frozen artifact differs')
 frozen(path,stream_digest(path))
 package=folder/'package/PPSA99014.ffpkg';size=receipt['package_bytes'];digest=receipt['package_sha256']
 external=receipt.get('external_model')
 if external is not None:
  validate_external_model(external);require(manifest.get('external_model')==external,'baseline external model differs')
 require(type(size) is int and (0<size<=package_limit('llm_client') if external is not None else MODEL_BYTES<size<=MODEL_PACKAGE_LIMIT) and package.stat().st_size==size and deployment.get('package_sha256')==digest,'baseline package identity differs')
 require(manifest.get('package')==str(package.relative_to(ROOT)) and manifest.get('model_transfer',{}).get('bytes')==size and manifest['model_transfer'].get('sha256')==digest,'baseline transfer differs')
 frozen(package,digest)
 app=folder/'package/app';names=legacy.LLM_CLIENT_PACKAGE_FILES|({'model-path.txt'} if external is not None else {'model.gguf'})
 require(set(receipt['package_files'])==names and manifest.get('package_files')==receipt['package_files'] and {str(p.relative_to(app)) for p in app.rglob('*') if p.is_file()}==names,'baseline nine assets required')
 for name,h in receipt['package_files'].items():frozen(app/name,h)
 require((app/'client-phase.txt').read_bytes()==b'model-gate\n','baseline phase differs')
 if external is not None:require((app/'model-path.txt').read_bytes()==encode_model_path(external['path']),'baseline model path differs')
 else:require((app/'model.gguf').stat().st_size==MODEL_BYTES and receipt['package_files']['model.gguf']==MODEL_SHA,'baseline model asset differs')
 require((app/'results-dir.txt').read_text()==receipt['results_dir']==manifest.get('results_dir'),'baseline results directory differs')
 audit_path=folder/'llm-client-native-audit.json';frozen(audit_path,bound['audit']['sha256']);audit=json.loads(audit_path.read_text())
 require(audit.get('status')=='PASS' and audit.get('classification')=='offline-public-hip-llm-client-native-image-audit' and audit.get('phase')=='model-gate','baseline audit differs')
 require(all(audit.get(k) is True for k in ('wrapper_retained','wrapper_no_bypass','actual_client_call_chain_audited')) and all(audit.get(k) is False for k in ('actual_client_gate_executed','native_executed','gpu_executed')),'baseline audit claims differ')
 original=Path(bound['audit']['path']).parent
 for name in ('title.elf','eboot.elf','extracted.elf'):
  image=original/name;h=audit['inputs'].get(str(image));require(h is not None and receipt['inputs'].get(str(image))==h and stream_digest(image)==h,'baseline image provenance differs')
 return receipt

def prerequisites(bound):
 import build_llm_native_app as driver
 model=driver.model_inputs(Path(bound['model']['asset']),Path(bound['model']['source_dir']),Path(bound['model']['cpu_build']))
 return driver.vocab_inputs(Path(bound['vocabulary']['receipt']).parent),driver.client_builder.app_objects(Path(bound['native_client']['receipt']).parent),model

def audit_sources():
 from audit_public_hip_llm_client_native import SOURCES
 return [ROOT/n for n in SOURCES]

def validate_model_receipt(build,receipt):
 import sys
 sys.path.insert(0,str(ROOT/'tools'))
 import freeze_source_candidate as legacy
 phase=receipt.get('phase');prefix=phase=='model-prefix'
 classification='source-only-model-prefix-native-client-app' if prefix else 'source-only-model-rejection-native-client-app'
 require(receipt.get('schema')==1 and receipt.get('status')=='PASS' and receipt.get('classification')==classification,'wrong model receipt')
 require(receipt.get('fixture')=='llm_client' and phase in ('model-gate','model-prefix') and receipt.get('native_executed') is False and receipt.get('gpu_executed') is False,'wrong phase/execution')
 external=receipt.get('external_model');validate_external_model(external)
 bound=receipt['llm_client'];require(bound.get('phase')==phase,'wrong bound phase')
 inputs=receipt.get('inputs');require(isinstance(inputs,dict) and inputs,'missing input closure')
 import native_ccache as cache
 cache.verify(ROOT,receipt.get('compiler_cache'))
 for name,h in cache.bound_inputs(receipt['compiler_cache']).items():require(inputs.get(name)==h,'unbound compiler cache identity')
 for name,h in inputs.items():require(Path(name).is_absolute() and stream_digest(name)==h,'input changed')
 for name,target in receipt.get('symlinks',{}).items():require(str(Path(name).resolve())==target,'symlink changed')
 vocab,client,model=prerequisites(bound)
 for key,actual,fields in [('vocabulary',vocab,('receipt','receipt_sha256','asset')),('native_client',client,('receipt','receipt_sha256')),('model',model,('asset','bytes','sha256','receipt','receipt_sha256','source_dir','cpu_build'))]:
  require({k:actual[k] for k in fields}==bound[key],'prerequisite differs')
  require(all(inputs.get(p)==h for p,h in actual['inputs'].items()),'prerequisite closure omitted')
 require(model['bytes']==MODEL_BYTES and model['sha256']==MODEL_SHA,'wrong model profile')
 require(receipt.get('object_count')==149 and [u['object'] for u in receipt['units'][-70:]]==client['objects'],'object inventory differs')
 original=Path(bound['audit']['path']).parent
 legacy.validate_client_units(original,receipt);legacy.validate_client_image_assets(original,receipt)
 audit_path=build/'llm-client-native-audit.json';require(stream_digest(audit_path)==bound['audit']['sha256'],'audit hash differs')
 audit=json.loads(audit_path.read_text())
 require(audit.get('status')=='PASS' and audit.get('classification')=='offline-public-hip-llm-client-native-image-audit' and audit.get('phase')==phase,'wrong native audit phase')
 for field in ('wrapper_retained','wrapper_no_bypass','actual_client_call_chain_audited'):require(audit.get(field) is True,'missing native call-chain proof')
 if prefix:require(audit.get('actual_prefix_call_chain_audited') is True,'missing actual prefix call-chain proof')
 for field in ('actual_client_gate_executed','native_executed','gpu_executed'):require(audit.get(field) is False,'unearned execution claim')
 audit_inputs=audit.get('inputs',{})
 for path in [original/n for n in ('title.elf','eboot.elf','extracted.elf')]+audit_sources():
  h=stream_digest(path);require(audit_inputs.get(str(path))==h and inputs.get(str(path))==h,'missing audit identity')
 for name,h in audit_inputs.items():require(Path(name).is_absolute() and stream_digest(name)==h,'audit input changed')
 app=build/'package/app';package=build/'package/PPSA99014.ffpkg'
 prefix_data=prefix_binding(bound,inputs) if prefix else None
 names=legacy.LLM_CLIENT_PACKAGE_FILES|{'model-path.txt'}|(set(prefix_data['assets']) if prefix else set())
 require(set(receipt['package_files'])==names and {str(p.relative_to(app)) for p in app.rglob('*') if p.is_file()}==names,'exact model package assets required')
 require(type(receipt['package_bytes']) is int and 0<receipt['package_bytes']<=package_limit('llm_client') and package.stat().st_size==receipt['package_bytes'] and stream_digest(package)==receipt['package_sha256'],'package identity/bound differs')
 for name,h in receipt['package_files'].items():require(not (app/name).is_symlink() and stream_digest(app/name)==h,'package asset differs')
 require((app/'client-phase.txt').read_bytes()==(phase+'\n').encode() and (app/'results-dir.txt').read_text()==receipt['results_dir'],'package configuration differs')
 require((app/'model-path.txt').read_bytes()==encode_model_path(external['path']) and receipt['package_files']['vocab.gguf']==stream_digest(vocab['asset']),'packaged model path/vocabulary differs')
 if prefix:
  require({n:receipt['package_files'][n] for n in prefix_data['assets']}=={n:stream_digest(p) for n,p in prefix_data['assets'].items()},'packaged prefix modules differ')
 for name,h in inputs.items():require(stream_digest(name)==h,'input changed during admission')
 return bound

def validator_identities():
 import sys,types,importlib
 driver=importlib.import_module('build_llm_native_app')
 pending=[importlib.import_module('public_hip_prefix_assets'),importlib.import_module('build_llm_prefix_reference'),driver,importlib.import_module('freeze_source_candidate'),importlib.import_module('fetch_llm_model')]
 paths={Path(__file__).resolve(),ROOT/'tests/llm_native_app_build_test.py'}
 paths.update(ROOT/n for n in ('tools/build_public_hip_prefix_assets.py','tests/public_hip_prefix_assets_test.py','tests/source_model_prefix_freeze_test.py','tests/llm_prefix_reference_test.py','tests/source_prefix_console_test.py','tests/source_prefix_authentication_test.py','tools/source_llm_prefix_log.py','tools/verify_llm_prefix_readbacks.py','tests/native_prefix_transcript_test.py','tests/llm_prefix_readbacks_test.py'))
 paths.update(ROOT/n for n in ('tests/native_model_path_test.py','tests/source_external_model_test.py'))
 paths.update(ROOT/n for n in ('tools/source_console.py', 'tools/source_model_streaming.py', 'tools/source_llm_model_log.py', 'tests/source_model_auth_test.py', 'tests/source_model_callsites_test.py', 'tests/source_model_cleanup_budget_test.py', 'tests/source_model_execute_test.py', 'tests/source_model_phase_budget_test.py', 'tests/source_model_storage_edges_test.py', 'tests/source_model_streaming_test.py'))
 seen=set()
 while pending:
  module=pending.pop()
  if id(module) in seen:continue
  seen.add(id(module));name=getattr(module,'__file__',None)
  if not name:continue
  path=Path(name).resolve()
  if not path.is_relative_to(ROOT) or path.suffix!='.py':continue
  paths.add(path)
  for value in vars(module).values():
   if isinstance(value,types.ModuleType):pending.append(value)
   elif isinstance(value,(types.FunctionType,type)):
    target=sys.modules.get(getattr(value,'__module__',''))
    if target is not None:pending.append(target)
 return {str(p.relative_to(ROOT)):stream_digest(p) for p in paths}

def validate_model_candidate(candidate,manifest):
 require(manifest.get('deadline_seconds')==150 and manifest.get('control_phase_seconds')==1800,'model phase budget differs')
 initial=validator_identities()
 require(all(manifest['files'].get(n)==h for n,h in initial.items()),'live validator identity not frozen')
 model_contract(manifest)
 receipt=json.loads((candidate/'build-receipt.json').read_text())
 bound=validate_model_receipt(candidate,receipt)
 require(bound==manifest['llm_client'] and receipt['external_model']==manifest.get('external_model'),'manifest provenance differs')
 require(manifest['model_transfer']['bytes']==receipt['package_bytes'] and manifest['model_transfer']['sha256']==receipt['package_sha256'],'transfer identity differs')
 require(manifest['package_files']==receipt['package_files'],'manifest package inventory differs')
 if bound['phase']=='model-prefix':
  expected=frozen_prefix_manifest(candidate,bound,receipt['inputs'])
  require(manifest.get('model_prefix')==expected,'frozen prefix manifest differs')
  copied=candidate/'prefix-reference'
  for path in copied.rglob('*'):
   if path.is_file():require(manifest['files'].get(str(path.relative_to(ROOT)))==stream_digest(path),'frozen prefix file identity missing')
 require(validator_identities()==initial,'validator changed during authentication')
 return bound



PREFIX_MODULES=('getrows','normalized','weighted','q8','q4')
PREFIX_RAW_FILES={f'token-{token}-{name}.f32le':width*4 for token in (0,1,2,31999)
                  for name,width in (('embedding',2048),('normalized',2048),('weighted',2048),('q',2048),('k',256),('v',256))}
PREFIX_REFERENCE_LIMIT=16*1024*1024


def prefix_binding(bound,inputs):
 from public_hip_prefix_assets import prefix_inputs
 link=bound.get('prefix',{})
 require(set(link)=={'receipt','receipt_sha256','reference'},'prefix binding fields differ')
 receipt=Path(link['receipt'])
 require(receipt.is_absolute() and receipt.name=='receipt.json' and receipt.is_relative_to(ROOT),'prefix aggregate path differs')
 data=prefix_inputs(receipt.parent)
 require({k:data[k] for k in link}==link,'prefix aggregate binding differs')
 require(set(data['assets'])=={'model-prefix/'+n+'.hsaco' for n in PREFIX_MODULES},'prefix exact five modules required')
 model=bound['model']
 require(data['model']=={'path':str(Path(model['asset']).resolve()),'bytes':model['bytes'],'sha256':model['sha256']},'prefix model identity differs')
 require(all(inputs.get(p)==h for p,h in data['inputs'].items()),'prefix closure omitted')
 return data


def frozen_prefix_manifest(candidate,bound,inputs):
 from build_llm_prefix_reference import authenticate_prefix_reference
 data=prefix_binding(bound,inputs)
 reference=candidate/'prefix-reference';copied,closure=authenticate_prefix_reference(reference)
 original=Path(data['reference'])/'receipt.json'
 require(copied['receipt_sha256']==stream_digest(original) and copied['model']==data['model'],'frozen prefix reference differs')
 require(sum(p.stat().st_size for p in reference.rglob('*') if p.is_file())<=PREFIX_REFERENCE_LIMIT,'prefix reference too large')
 return {'aggregate':str(Path(data['receipt']).relative_to(ROOT)),'aggregate_sha256':data['receipt_sha256'],
         'reference':str(reference.relative_to(ROOT)),'reference_receipt_sha256':copied['receipt_sha256'],
         'module_sha256':{n:stream_digest(data['assets']['model-prefix/'+n+'.hsaco']) for n in PREFIX_MODULES},
         'raw_files':dict(PREFIX_RAW_FILES)}
