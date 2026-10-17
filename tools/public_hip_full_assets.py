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

def assemble(parents,reference):
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
 reference=Path(reference).resolve();data,closure=validate_reference(reference);receipt=reference/'receipt.json'
 if data.get('status')!='PASS' or data.get('classification')!='successful-full-model-cpu-boundary-reference' or data.get('requests')!=514 or data.get('raw_files')!=168:raise ValueError('full reference contract')
 if data.get('receipt')!=str(receipt) or data.get('receipt_sha256')!=closure.get(str(receipt)):raise ValueError('reference receipt identity mismatch')
 model=data.get('model',{})
 if set(model)!={'path','bytes','sha256'}:raise ValueError('model identity')
 path=Path(model['path'])
 if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size!=model['bytes'] or closure.get(str(path))!=model['sha256']:raise ValueError('model absent from reference closure')
 for name,size in RAW_FILES.items():
  path=reference/'raw'/name
  if path.is_symlink() or not path.is_file() or path.stat().st_size!=size or closure.get(str(path))!=sha(path):raise ValueError('smoke reference output absent from closure: '+name)
 merge(inputs,closure);check_inputs(inputs)
 return {'schema':1,'status':'PASS','classification':CLASSIFICATION,'profile':'smoke','parents':resolved,
         'reference':str(reference),'reference_data':str(reference/'raw'),'model':model,'inputs':inputs,'assets':assets,
         'raw_files':RAW_FILES,'native_execution':False,'gpu_execution':False,'numerical_acceptance':False}

def full_inputs(folder):
 receipt=Path(folder).resolve()/'receipt.json'
 if receipt.is_symlink():raise ValueError('symlink aggregate receipt')
 digest=sha(receipt);data=json.loads(receipt.read_text());expected=assemble(data['parents'],data['reference'])
 if data!=expected:raise ValueError('stale or altered full aggregate receipt')
 if sha(receipt)!=digest:raise ValueError('aggregate receipt changed during authentication')
 inputs=dict(data['inputs']);merge(inputs,{str(receipt):digest})
 return {**data,'inputs':inputs,'receipt':str(receipt),'receipt_sha256':digest}
