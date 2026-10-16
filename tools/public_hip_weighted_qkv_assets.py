"""Authenticate exactly three modules and seven numerical assets for weighted QKV."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REFERENCE_FILES={'input.f32le':'input.f32le','norm.f32le':'run1/norm.f32le',
 'weighted.fp64le':'weighted.fp64le','weighted.budget.f64le':'weighted.budget.f64le',
 **{n+'.weights.q4_0':'run1/'+n+'.weights.q4_0' for n in ('q','k','v')}}


def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def merge(inputs,extra):
 for name,digest in extra.items():
  if name in inputs and inputs[name]!=digest:raise ValueError('conflicting weighted QKV identity: '+name)
  inputs[name]=digest


def assets(weighted_fixture,q4_fixture,reference):
 # Lazy imports avoid a native-builder/reference-authentication import cycle.
 own={str(ROOT/n):sha(ROOT/n) for n in ('tools/public_hip_weighted_qkv_assets.py','tests/public_hip_weighted_qkv_assets_test.py',
                                      'tools/verify_weighted_qkv_readbacks.py','tests/weighted_qkv_readbacks_test.py')}
 from build_llm_weighted_rmsnorm_fixture import fixture_inputs as weighted_inputs
 from build_llm_q4_halfword_fixture import fixture_inputs as halfword_inputs
 from verify_weighted_qkv_readbacks import authenticate
 weighted_fixture,q4_fixture,reference=(Path(p).resolve() for p in (weighted_fixture,q4_fixture,reference))
 modules={};merge(modules,weighted_inputs(weighted_fixture));merge(modules,halfword_inputs(q4_fixture))
 data,reference_inputs=authenticate(reference)
 if set(data)!=set(REFERENCE_FILES.values()):raise ValueError('exact seven weighted QKV assets required')
 inputs=dict(own);merge(inputs,modules);merge(inputs,reference_inputs)
 selected={'weighted.hsaco':weighted_fixture/'weighted.hsaco','q8.hsaco':q4_fixture/'q8.hsaco','q4.hsaco':q4_fixture/'q4.hsaco'}
 selected.update({name:reference/relative for name,relative in REFERENCE_FILES.items()})
 output={}
 for name,path in selected.items():
  if path.is_symlink() or not path.is_file() or inputs.get(str(path))!=sha(path):raise ValueError('package asset absent from authenticated closure: '+name)
  if name in REFERENCE_FILES and path.read_bytes()!=data[REFERENCE_FILES[name]]:raise ValueError('reference bytes changed during selection')
  output['weighted-qkv/'+name]=str(path)
 for name,digest in inputs.items():
  if sha(name)!=digest:raise ValueError('weighted QKV input changed during admission: '+name)
 return {'module_inputs':modules,'reference':{'receipt':str(reference/'receipt.json'),'receipt_sha256':reference_inputs[str(reference/'receipt.json')],'inputs':reference_inputs},
         'inputs':inputs,'assets':output}
