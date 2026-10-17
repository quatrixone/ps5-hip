"""Authenticate the exact six modules consumed by the bounded real-model layer."""
import json
from pathlib import Path
from public_hip_weighted_qkv_assets import sha, merge
ROOT=Path(__file__).resolve().parents[1]
SOURCES=('tools/public_hip_layer_assets.py','tools/build_public_hip_layer_assets.py','tests/public_hip_layer_assets_test.py','tools/public_hip_weighted_qkv_assets.py')
CLASSIFICATION='authenticated-real-model-layer-assets'
KINDS={'getrows':'source-only-selected-upstream-getrows-fixture',
       'normalized':None,'weighted':'source-only-selected-upstream-weighted-rmsnorm-fixture',
       'q4':'diagnostic-q4-halfword-fixture','layer':'source-only-bounded-layer-fixture'}
FILES={'getrows':('getrows.hsaco',),'normalized':('norm.hsaco',),'weighted':('weighted.hsaco',),'q4':('q8.hsaco','q4.hsaco'),'layer':('layer.hsaco',)}


def validate_module(kind,folder):
 from qualify_public_hip_q4 import fixture_inputs as getrows
 from qualify_public_hip_rmsnorm import fixture_inputs as normalized
 from build_llm_weighted_rmsnorm_fixture import fixture_inputs as weighted
 from build_llm_q4_halfword_fixture import fixture_inputs as q4
 from build_llm_layer_fixture import fixture_inputs as layer
 return {'getrows':getrows,'normalized':normalized,'weighted':weighted,'q4':q4,'layer':layer}[kind](folder)


def validate_reference(folder):
 from build_llm_layer_reference import authenticate_layer_reference
 return authenticate_layer_reference(folder)


def check_inputs(inputs):
 for name,digest in inputs.items():
  path=Path(name)
  if not path.is_absolute() or not path.is_file() or sha(path)!=digest:
   raise ValueError('layer input changed or missing: '+name)


def assemble(parents,reference):
 if set(parents)!=set(KINDS):raise ValueError('exact five module parents required')
 inputs={str(ROOT/n):sha(ROOT/n) for n in SOURCES};assets={};resolved={}
 for kind in KINDS:
  folder=Path(parents[kind]).resolve();resolved[kind]=str(folder)
  receipt=folder/'receipt.json'
  if receipt.is_symlink():raise ValueError('symlink parent receipt')
  data=json.loads(receipt.read_text())
  if KINDS[kind] and data.get('classification')!=KINDS[kind]:raise ValueError('wrong module classification: '+kind)
  closure=validate_module(kind,folder)
  if closure.get(str(receipt))!=sha(receipt):raise ValueError('parent receipt absent from closure')
  merge(inputs,closure)
  for name in FILES[kind]:
   path=folder/name
   if path.is_symlink() or not path.is_file() or closure.get(str(path))!=sha(path):raise ValueError('module absent from authenticated closure: '+name)
   package='normalized.hsaco' if name=='norm.hsaco' else name
   assets['model-layer/'+package]=str(path)
 reference=Path(reference).resolve();data,closure=validate_reference(reference)
 receipt=reference/'receipt.json'
 if data.get('receipt')!=str(receipt) or data.get('receipt_sha256')!=closure.get(str(receipt)):
  raise ValueError('reference receipt identity mismatch')
 model=data.get('model',{})
 if set(model)!= {'path','bytes','sha256'}:raise ValueError('invalid model identity')
 modelpath=Path(model['path'])
 if not modelpath.is_absolute() or modelpath.is_symlink() or not modelpath.is_file() or modelpath.stat().st_size!=model['bytes'] or closure.get(str(modelpath))!=model['sha256']:
  raise ValueError('model absent from reference closure')
 if Path(data['reference']).resolve()!=reference/'run-1':raise ValueError('unexpected reference data directory')
 merge(inputs,closure);check_inputs(inputs)
 return {'schema':1,'status':'PASS','classification':CLASSIFICATION,'parents':resolved,
         'reference':str(reference),'reference_data':data['reference'],'model':model,'inputs':inputs,'assets':assets,
         'native_execution':False,'gpu_execution':False}


def layer_inputs(folder):
 receipt=Path(folder).resolve()/'receipt.json'
 if receipt.is_symlink():raise ValueError('symlink aggregate receipt')
 digest=sha(receipt);data=json.loads(receipt.read_text())
 expected=assemble(data['parents'],data['reference'])
 if data!=expected:raise ValueError('stale or altered layer aggregate receipt')
 if sha(receipt)!=digest:raise ValueError('aggregate receipt changed during authentication')
 inputs=dict(data['inputs']);merge(inputs,{str(receipt):digest})
 return {**data,'inputs':inputs,'receipt':str(receipt),'receipt_sha256':digest}
