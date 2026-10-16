"""Authenticate model-rejection packages and frozen controller dependencies."""
import hashlib,json,re
from pathlib import Path
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
 require(manifest.get('fixture')=='llm_client' and manifest.get('llm_client',{}).get('phase')=='model-gate','wrong model phase')
 model=manifest['llm_client']['model']
 require(model['bytes']==MODEL_BYTES and model['sha256']==MODEL_SHA,'wrong locked model')
 t=manifest['model_transfer'];b=manifest['baseline_transfer']
 require(type(t['bytes']) is int and MODEL_BYTES<t['bytes']<=MODEL_PACKAGE_LIMIT,'model package bound')
 for v in (t,b):
  require(type(v['bytes']) is int and v['bytes']>0 and re.fullmatch('[0-9a-f]{64}',v['sha256']) and 0<v['deadline_seconds']<=1800,'invalid exact transfer')
 require(b['sha256']==manifest['baseline_image_sha256'],'baseline digest differs')
 # Legacy/vocab baselines remain bounded; larger baseline needs independently authenticated model phase.
 from source_public_hip_log import package_limit
 if b['bytes']>package_limit(manifest.get('baseline_log_fixture','bootstrap')):
  path=Path(b['receipt']);require(stream_digest(path)==b['receipt_sha256'],'large baseline receipt differs')
  receipt=json.loads(path.read_text());validate_model_receipt(path.parent,receipt)
  require(receipt['package_bytes']==b['bytes'] and receipt['package_sha256']==b['sha256'],'large baseline identity differs')
ROOT=Path(__file__).resolve().parents[1]
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
 require(receipt.get('schema')==1 and receipt.get('status')=='PASS' and receipt.get('classification')=='source-only-model-rejection-native-client-app','wrong model receipt')
 require(receipt.get('fixture')=='llm_client' and receipt.get('phase')=='model-gate' and receipt.get('native_executed') is False and receipt.get('gpu_executed') is False,'wrong phase/execution')
 bound=receipt['llm_client'];require(bound.get('phase')=='model-gate','wrong bound phase')
 inputs=receipt.get('inputs');require(isinstance(inputs,dict) and inputs,'missing input closure')
 for name,h in inputs.items():require(Path(name).is_absolute() and stream_digest(name)==h,'input changed')
 for name,target in receipt.get('symlinks',{}).items():require(str(Path(name).resolve())==target,'symlink changed')
 vocab,client,model=prerequisites(bound)
 for key,actual,fields in [('vocabulary',vocab,('receipt','receipt_sha256','asset')),('native_client',client,('receipt','receipt_sha256')),('model',model,('asset','bytes','sha256','receipt','receipt_sha256','source_dir','cpu_build'))]:
  require({k:actual[k] for k in fields}==bound[key],'prerequisite differs')
  require(all(inputs.get(p)==h for p,h in actual['inputs'].items()),'prerequisite closure omitted')
 require(model['bytes']==MODEL_BYTES and model['sha256']==MODEL_SHA,'wrong model profile')
 require(receipt.get('object_count')==148 and [u['object'] for u in receipt['units'][-70:]]==client['objects'],'object inventory differs')
 original=Path(bound['audit']['path']).parent
 legacy.validate_client_units(original,receipt);legacy.validate_client_image_assets(original,receipt)
 audit_path=build/'llm-client-native-audit.json';require(stream_digest(audit_path)==bound['audit']['sha256'],'audit hash differs')
 audit=json.loads(audit_path.read_text())
 require(audit.get('status')=='PASS' and audit.get('classification')=='offline-public-hip-llm-client-native-image-audit' and audit.get('phase')=='model-gate','wrong native audit phase')
 for field in ('wrapper_retained','wrapper_no_bypass','actual_client_call_chain_audited'):require(audit.get(field) is True,'missing native call-chain proof')
 for field in ('actual_client_gate_executed','native_executed','gpu_executed'):require(audit.get(field) is False,'unearned execution claim')
 audit_inputs=audit.get('inputs',{})
 for path in [original/n for n in ('title.elf','eboot.elf','extracted.elf')]+audit_sources():
  h=stream_digest(path);require(audit_inputs.get(str(path))==h and inputs.get(str(path))==h,'missing audit identity')
 for name,h in audit_inputs.items():require(Path(name).is_absolute() and stream_digest(name)==h,'audit input changed')
 app=build/'package/app';package=build/'package/PPSA99014.ffpkg'
 names=legacy.LLM_CLIENT_PACKAGE_FILES|{'model.gguf'}
 require(set(receipt['package_files'])==names and {str(p.relative_to(app)) for p in app.rglob('*') if p.is_file()}==names,'exact9 assets required')
 require(MODEL_BYTES<receipt['package_bytes']<=MODEL_PACKAGE_LIMIT and package.stat().st_size==receipt['package_bytes'] and stream_digest(package)==receipt['package_sha256'],'package identity/bound differs')
 for name,h in receipt['package_files'].items():require(not (app/name).is_symlink() and stream_digest(app/name)==h,'package asset differs')
 require((app/'client-phase.txt').read_bytes()==b'model-gate\n' and (app/'results-dir.txt').read_text()==receipt['results_dir'],'package configuration differs')
 require((app/'model.gguf').stat().st_size==MODEL_BYTES and receipt['package_files']['model.gguf']==model['sha256'] and receipt['package_files']['vocab.gguf']==stream_digest(vocab['asset']),'packaged model/vocabulary differs')
 for name,h in inputs.items():require(stream_digest(name)==h,'input changed during admission')
 return bound

def validator_identities():
 import sys,types,importlib
 driver=importlib.import_module('build_llm_native_app')
 pending=[driver,importlib.import_module('freeze_source_candidate'),importlib.import_module('fetch_llm_model')]
 paths={Path(__file__).resolve(),ROOT/'tests/llm_native_app_build_test.py'}
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
 require(bound==manifest['llm_client'],'manifest provenance differs')
 require(manifest['model_transfer']['bytes']==receipt['package_bytes'] and manifest['model_transfer']['sha256']==receipt['package_sha256'],'transfer identity differs')
 require(manifest['package_files']==receipt['package_files'],'manifest package inventory differs')
 require(validator_identities()==initial,'validator changed during authentication')
 return bound

