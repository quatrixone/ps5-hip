"""Authenticate six existing modules and the full-model CPU oracle for native smoke."""
import json
from pathlib import Path
from public_hip_layer_assets import KINDS, FILES, validate_module, check_inputs
from public_hip_weighted_qkv_assets import sha, merge
ROOT=Path(__file__).resolve().parents[1]
SOURCES=('tools/public_hip_full_assets.py','tools/build_public_hip_full_assets.py','tests/public_hip_full_assets_test.py','tools/public_hip_layer_assets.py')
CLASSIFICATION='authenticated-full-model-smoke-assets'
RAW_FILES={f'run-{run}-pos-{pos}-{stage}.f32le':128000 if stage=='logits' else 8192
           for run,pos in ((0,0),(0,1),(1,0))
           for stage in [*(f'layer-{i}' for i in range(22)),'norm','logits']}

def validate_reference(folder):
 from build_llm_full_model_reference import authenticate_full_model_reference
 return authenticate_full_model_reference(folder)

def assemble(parents,reference,profile='smoke'):
 if profile not in ('smoke','generation','corpus'):raise ValueError('altered or unsupported full profile')
 if set(parents)!=set(KINDS):raise ValueError('exact five module parents required')
 inputs={str(ROOT/n):sha(ROOT/n) for n in SOURCES};assets={};resolved={}
 for kind in KINDS:
  folder=Path(parents[kind]).resolve();resolved[kind]=str(folder);receipt=folder/'receipt.json'
  if receipt.is_symlink():raise ValueError('symlink parent receipt')
  data=json.loads(receipt.read_text())
  if KINDS[kind] and data.get('classification')!=KINDS[kind]:raise ValueError('wrong module classification: '+kind)
  closure=validate_module(kind,folder)
  if closure.get(str(receipt))!=sha(receipt):raise ValueError('parent receipt absent from closure')
  merge(inputs,closure)
  for name in FILES[kind]:
   path=folder/name
   if path.is_symlink() or not path.is_file() or closure.get(str(path))!=sha(path):raise ValueError('module absent from authenticated closure: '+name)
   assets['model-layer/'+('normalized.hsaco' if name=='norm.hsaco' else name)]=str(path)
 reference=Path(reference).resolve();receipt=reference/'receipt.json'
 raw_files=RAW_FILES
 if profile=='smoke':
  data,closure=validate_reference(reference)
  if data.get('status')!='PASS' or data.get('classification')!='successful-full-model-cpu-boundary-reference' or data.get('requests')!=514 or data.get('raw_files')!=168:raise ValueError('full reference contract')
 else:
  from build_llm_generation_reference import authenticate_generation_reference,loads,STAGES,corpus_cases
  data,closure=authenticate_generation_reference(reference)
  if data.get('classification')!='successful-greedy-generation-cpu-reference' or data.get('profile','short')!=('corpus' if profile=='corpus' else 'short'):raise ValueError('generation reference profile')
  policies=[('',loads((reference/'raw/generation.json').read_text()))] if profile=='generation' else [(key+'/',loads((reference/'raw'/key/'generation.json').read_text())) for key,_ in corpus_cases()]
  raw_files={prefix+f'run-{row["run"]}-pos-{pos}-{stage}.f32le':128000 if stage=='logits' else 8192
             for prefix,policy in policies for row in policy['runs'] for pos in range(len(row['processed'])) for stage in STAGES}
  requests=sum(len(row['processed']) for _,policy in policies for row in policy['runs'])
  if data.get('requests')!=requests or data.get('raw_files')!=len(raw_files) or (profile=='generation' and (requests!=16 or len(raw_files)!=384)):raise ValueError('generation reference contract')
  raw_files.update({prefix+name:(reference/'raw'/(prefix+name)).stat().st_size for prefix,_ in policies for name in ('generation.json','steps.jsonl')})
  if profile=='corpus':
   cases=corpus_cases();index=loads((reference/'raw/corpus.json').read_text())
   if index!=dict(schema=1,profile='corpus',max_new_tokens=16,cases=[key for key,_ in cases]):raise ValueError('corpus index')
   for (key,prompt),(prefix,policy) in zip(cases,policies):
    if policy.get('prompt')!=prompt or policy.get('max_new_tokens')!=16 or len(policy.get('runs',[]))!=2:raise ValueError('corpus policy')
    for run,row in enumerate(policy['runs']):
     if row.get('run')!=run or len(row.get('generated',[]))!=16 or row.get('stop')!='step_limit' or row.get('processed')!=policy.get('prompt_tokens',[])+row['generated'][:-1]:raise ValueError('corpus feedback')
   raw_files['corpus.json']=(reference/'raw/corpus.json').stat().st_size
   entries=list((reference/'raw').rglob('*'))
   if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries) or {str(p.relative_to(reference/'raw')) for p in entries if p.is_file()}!=set(raw_files) or {str(p.relative_to(reference/'raw')) for p in entries if p.is_dir()}!={key for key,_ in cases}:raise ValueError('corpus raw inventory')
 if data.get('receipt')!=str(receipt) or data.get('receipt_sha256')!=closure.get(str(receipt)):raise ValueError('reference receipt identity mismatch')
 model=data.get('model',{})
 if set(model)!={'path','bytes','sha256'}:raise ValueError('model identity')
 path=Path(model['path'])
 if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size!=model['bytes'] or closure.get(str(path))!=model['sha256']:raise ValueError('model absent from reference closure')
 for name,size in raw_files.items():
  path=reference/'raw'/name
  if path.is_symlink() or not path.is_file() or path.stat().st_size!=size or closure.get(str(path))!=sha(path):raise ValueError('smoke reference output absent from closure: '+name)
 merge(inputs,closure);check_inputs(inputs)
 return {'schema':1,'status':'PASS','classification':CLASSIFICATION if profile=='smoke' else 'authenticated-full-model-'+profile+'-assets','profile':profile,'parents':resolved,
         'reference':str(reference),'reference_data':str(reference/'raw'),'model':model,'inputs':inputs,'assets':assets,
         'raw_files':raw_files,'native_execution':False,'gpu_execution':False,'numerical_acceptance':False}

def full_inputs(folder):
 receipt=Path(folder).resolve()/'receipt.json'
 if receipt.is_symlink():raise ValueError('symlink aggregate receipt')
 digest=sha(receipt);data=json.loads(receipt.read_text());expected=assemble(data['parents'],data['reference'],data.get('profile'))
 if data!=expected:raise ValueError('stale or altered full aggregate receipt')
 if sha(receipt)!=digest:raise ValueError('aggregate receipt changed during authentication')
 inputs=dict(data['inputs']);merge(inputs,{str(receipt):digest})
 return {**data,'inputs':inputs,'receipt':str(receipt),'receipt_sha256':digest}
