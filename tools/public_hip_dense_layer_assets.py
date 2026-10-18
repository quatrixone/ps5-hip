"""Authenticate six modules and the synthetic dense reference without model assets."""
import argparse,hashlib,json
from pathlib import Path
from public_hip_layer_assets import FILES,KINDS,check_inputs,validate_module as legacy_module
from public_hip_weighted_qkv_assets import sha,merge

ROOT=Path(__file__).resolve().parents[1]
CLASSIFICATION='authenticated-synthetic-dense-layer-assets'
SOURCES=('tools/public_hip_dense_layer_assets.py','tools/public_hip_layer_assets.py',
         'tools/public_hip_weighted_qkv_assets.py','tests/public_hip_dense_layer_assets_test.py',
         'tools/verify_llm_dense_readbacks.py','tests/llm_dense_readbacks_test.py')

def validate_module(kind,folder):
 if kind=='layer':
  from build_llm_layer_fixture import fixture_inputs
  return fixture_inputs(folder,'dense-f16')
 return legacy_module(kind,folder)

def validate_reference(folder):
 from verify_llm_dense_readbacks import authenticate_reference
 return authenticate_reference(folder)

def assemble(parents,reference):
 if set(parents)!=set(FILES):raise ValueError('exact five module parents required')
 inputs={str(ROOT/n):sha(ROOT/n) for n in SOURCES};assets={};resolved={}
 for kind,names in FILES.items():
  folder=Path(parents[kind]).resolve();resolved[kind]=str(folder);receipt=folder/'receipt.json'
  if receipt.is_symlink():raise ValueError('symlink parent receipt')
  expected='source-only-bounded-dense-f16-layer-fixture' if kind=='layer' else KINDS[kind]
  if expected and json.loads(receipt.read_text()).get('classification')!=expected:raise ValueError('wrong module classification: '+kind)
  closure=validate_module(kind,folder)
  if closure.get(str(receipt))!=sha(receipt):raise ValueError('parent receipt absent from closure')
  merge(inputs,closure)
  for name in names:
   path=folder/name
   if path.is_symlink() or not path.is_file() or closure.get(str(path))!=sha(path):raise ValueError('module absent from closure')
   package='normalized.hsaco' if name=='norm.hsaco' else name
   assets['dense-layer/'+package]=str(path)
 reference=Path(reference).resolve();validated=validate_reference(reference)
 closure=validated['inputs'];receipt=reference/'receipt.json'
 if receipt.is_symlink() or closure.get(str(receipt))!=sha(receipt):raise ValueError('reference receipt absent from closure')
 merge(inputs,closure);check_inputs(inputs)
 return dict(schema=1,status='PASS',classification=CLASSIFICATION,parents=resolved,
             reference=str(reference),inputs=inputs,assets=assets,native_execution=False,gpu_execution=False)

def dense_inputs(folder):
 receipt=Path(folder).resolve()/'receipt.json'
 if receipt.is_symlink():raise ValueError('symlink aggregate receipt')
 digest=sha(receipt);data=json.loads(receipt.read_text())
 if data!=assemble(data['parents'],data['reference']):raise ValueError('stale or altered dense aggregate')
 if sha(receipt)!=digest:raise ValueError('aggregate changed during authentication')
 inputs=dict(data['inputs']);merge(inputs,{str(receipt):digest})
 return {**data,'inputs':inputs,'receipt':str(receipt),'receipt_sha256':digest}

def build(parents,reference,output):
 data=assemble(parents,reference);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
 receipt=output/'receipt.json';encoded=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode()
 digest=hashlib.sha256(encoded).hexdigest();receipt.write_bytes(encoded)
 if sha(receipt)!=digest:raise ValueError('new aggregate receipt changed')
 return {**data,'inputs':{**data['inputs'],str(receipt):digest},'receipt':str(receipt),'receipt_sha256':digest}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in (*FILES,'reference','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();result=build({n:getattr(a,n) for n in FILES},a.reference,a.output)
 print(json.dumps({k:result[k] for k in ('receipt','receipt_sha256','assets')},indent=2))
