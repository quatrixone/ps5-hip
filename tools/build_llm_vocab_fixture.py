"""Build an authenticated zero-tensor vocabulary GGUF using the locked upstream writer."""
from pathlib import Path
import argparse,json,sys,subprocess,os
import build_llm_executed_graph as g
ROOT=g.ROOT
HARNESS=Path(__file__).resolve().parents[1]/'tests/llm_vocab_tokenize.cpp'
TEST=Path(__file__).resolve().parents[1]/'tests/llm_vocab_fixture_test.py'
def validate_metadata(original,derived):
 fields={n:f for n,f in original.fields.items() if not n.startswith('GGUF.')}
 if derived.tensors or {n for n in derived.fields if not n.startswith('GGUF.')}!=set(fields):raise ValueError('metadata/tensor inventory mismatch')
 for n,f in fields.items():
  other=derived.get_field(n)
  if other.types!=f.types or other.contents()!=f.contents():raise ValueError('metadata differs: '+n)
 return len(fields)
def validate_tokens(original,derived):
 if original!=derived:raise ValueError('token sequences differ')
 rows=[json.loads(s) for s in original.splitlines()]
 if len(rows)!=32 or {(r['case'],r['add_special'],r['parse_special']) for r in rows}!={(c,a,s) for c in range(8) for a in (False,True) for s in (False,True)}:raise ValueError('token cases incomplete')
 if any(not isinstance(r['tokens'],list) or any(type(t)is not int or not 0<=t<32000 for t in r['tokens']) for r in rows):raise ValueError('token IDs invalid')
def verify_asset(path,expected):
 if g.sha(path)!=expected:raise ValueError('generated vocabulary changed during execution')
def build(out,cpu,model,environment):
 root=ROOT
 if out.exists() or not out.is_relative_to(root/'results'):raise ValueError('fresh results output required')
 client=root/'.deps/sources/llama-client';lib=cpu/'cmake/bin'
 sys.path.insert(0,str(client/'gguf-py'))
 import gguf
 out.mkdir(parents=True)
 inputs=g.cpu_inputs(cpu,model)
 for p in [Path(__file__).resolve(),HARNESS,Path(sys.executable).resolve(),root/'tools/build_llm_executed_graph.py',Path('/usr/bin/clang++-18').resolve(),Path('/usr/bin/ldd'),environment/'receipt.json']+list((client/'gguf-py').rglob('*.py')):g.merge(inputs,str(p),g.sha(p))
 # Bind imported Python extension/package files too, not a new global installation.
 for module in list(sys.modules.values()):
  p=getattr(module,'__file__',None)
  if p and Path(p).is_file():g.merge(inputs,str(Path(p).resolve()),g.sha(p))
 if Path(sys.prefix).resolve() != (environment/'venv').resolve():raise ValueError('prepared conversion environment Python required')
 lock=json.loads((root/'llm-conversion-inputs.json').read_text());env_receipt=json.loads((environment/'receipt.json').read_text())
 if env_receipt['lock_sha256']!=g.sha(root/'llm-conversion-inputs.json'):raise ValueError('environment lock mismatch')
 import importlib.metadata
 for name,version in lock['packages'].items():
  if importlib.metadata.version(name)!=version:raise ValueError('environment package differs: '+name)
 g.merge(inputs,str(root/'llm-conversion-inputs.json'),g.sha(root/'llm-conversion-inputs.json'))
 g.merge(inputs,str(TEST),g.sha(TEST))
 r=gguf.GGUFReader(str(model));target=out/'tinyllama-vocab.gguf'
 w=gguf.GGUFWriter(str(target),arch=r.get_field('general.architecture').contents(),endianess=r.endianess)
 for f in r.fields.values():
  if f.name.startswith('GGUF.') or f.name=='general.architecture':continue
  w.add_key_value(f.name,f.contents(),f.types[0],sub_type=f.types[-1] if f.types[0]==gguf.GGUFValueType.ARRAY else None)
 w.write_header_to_file();w.write_kv_data_to_file();w.write_ti_data_to_file();w.close()
 v=gguf.GGUFReader(str(target))
 metadata_count=validate_metadata(r,v)
 asset_hash=g.sha(target)
 commands=[];env={k:v for k,v in os.environ.items() if k not in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 def run(label,args):
  p=subprocess.run(list(map(str,args)),capture_output=True,text=True,env=env,timeout=120);(out/(label+'.stdout')).write_text(p.stdout);(out/(label+'.stderr')).write_text(p.stderr);commands.append(dict(label=label,argv=list(map(str,args)),exit_code=p.returncode));
  if p.returncode:raise ValueError(label+' failed: '+p.stderr)
  return p.stdout
 if '18.1.8' not in run('compiler-version',['/usr/bin/clang++-18','--version']):raise ValueError('LLVM 18.1.8 required')
 flags=['/usr/bin/clang++-18','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(client/'include'),'-I'+str(client/'ggml/include')]
 run('headers',[*flags,'-M','-MT','tokenize','-MF',out/'headers.d',HARNESS])
 for p in g.dependencies(out/'headers.d'):g.merge(inputs,p,g.sha(p))
 run('compile',[*flags,'-MD','-MT','tokenize','-MF',out/'compiled.d',HARNESS,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-lggml','-lggml-base','-o',out/'tokenize'])
 if g.dependencies(out/'headers.d')!=g.dependencies(out/'compiled.d'):raise ValueError('compiler header closure changed')
 ldd=run('runtime-libraries',['/usr/bin/ldd',out/'tokenize'])
 for p,h in g.runtime_inputs(ldd,lib,inputs).items():g.merge(inputs,p,h)
 binaryhash=g.sha(out/'tokenize');full=run('original',[out/'tokenize',model]);small=run('vocabulary-only',[out/'tokenize',target]);validate_tokens(full,small)
 if g.sha(out/'tokenize')!=binaryhash:raise ValueError('tokenizer executable changed')
 verify_asset(target,asset_hash)
 g.recheck(inputs)
 receipt=dict(schema=1,status='PASS',classification='upstream-vocabulary-only-gguf-tokenization',inputs=inputs,artifacts={p.name:g.sha(p) for p in out.iterdir() if p.is_file()},commands=commands,metadata_fields=metadata_count,tensor_count=0,vocab_bytes=target.stat().st_size,tokenization_cases=32,original_and_vocabulary_tokens_bitwise_equal=True,native_executed=False,gpu_executed=False,limits=['Host upstream vocab_only loader/tokenizer only; native linking/execution unproven.','No weights, inference or custom GGUF parser; final app separately enforces package size.'])
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS',out/'receipt.json')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('output','cpu-build','model','environment'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();build(a.output.resolve(),a.cpu_build.resolve(),a.model.resolve(),a.environment.resolve())
