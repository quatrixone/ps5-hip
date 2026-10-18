"""Deterministic dense layer fixture: existing CPU backend versus host HIP arithmetic."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def build(cpu,output,numeric):
 features=Path('/proc/cpuinfo').read_text().split()
 if any(k not in features for k in ('avx2','fma','f16c')):raise ValueError('host AVX2/FMA/F16C required')
 cpu=cpu.resolve();output=output.resolve();lib=cpu/'cmake/bin';cache=cpu/'cmake/CMakeCache.txt'
 if output.exists():raise ValueError('fresh output directory required')
 config={s.split('=',1)[0]:s.split('=',1)[1] for s in cache.read_text().splitlines() if '=' in s and not s.startswith(('#','//'))}
 if any(config.get(k)!=v for k,v in {'GGML_AVX2:BOOL':'ON','GGML_FMA:BOOL':'ON','GGML_AVX512:BOOL':'OFF'}.items()):raise ValueError('AVX2/FMA reference configuration required')
 names=['tests/llm_dense_layer_default_test.cpp','tests/llm_dense_layer_test.cpp','tests/llm_dense_layer_failure_test.cpp','tests/llm_dense_layer_checks_test.cpp','tests/llm_dense_layer_checks.h','tests/llm_layer_numeric_mock.h','kernels/llm_layer.hip','kernels/llm_cpu_math.h','ports/ggml/model_layer.h','ports/ggml/layer_dispatch.h','ports/ggml/prefix_dispatch.h','ports/ggml/model_prefix.h','ports/ggml/prefix_calls.h','tools/build_llm_dense_layer_test.py']
 files=[ROOT/n for n in names]+[cache,Path('/usr/bin/clang++-18'),(lib/'libggml-cpu.so').resolve(),(lib/'libggml-base.so').resolve()]
 inputs={str(p.resolve()):sha(p) for p in files};output.mkdir(parents=True)
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_','ASAN_','UBSAN_','LSAN_')) and k not in ('CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 env.update(OMP_NUM_THREADS='2',ASAN_OPTIONS='halt_on_error=1:detect_leaks=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
 flags=['-mavx2','-mfma','-mf16c','-std=c++17','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-D__HIP_PLATFORM_AMD__']
 flags+=['-I'+str(ROOT/n) for n in ('ports/ggml','ports/hip','.deps/sources/hip-headers/include','.deps/sources/clr/hipamd/include','.deps/sources/llama-client/ggml/include')]
 commands=[]
 def run(label,cmd,timeout=120):
  start=time.monotonic()
  with (output/(label+'.stdout')).open('wb') as out,(output/(label+'.stderr')).open('wb') as err:
   p=subprocess.run(list(map(str,cmd)),env=env,stdout=out,stderr=err,timeout=timeout)
  commands.append(dict(label=label,argv=list(map(str,cmd)),exit_code=p.returncode,seconds=time.monotonic()-start));(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  p.check_returncode();stderr=(output/(label+'.stderr')).read_text()
  if any(s in stderr for s in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer','LeakSanitizer')):raise RuntimeError('sanitizer diagnostics')
 def compile_unit(name,source,sanitize):
  binary=output/name;options=['-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer'] if sanitize else ['-O2']
  run('compile-'+name,['/usr/bin/clang++-18',*flags,*options,ROOT/source,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lggml-cpu','-lggml-base','-o',binary]);run('linkage-'+name,['/usr/bin/ldd',binary])
  linkage=(output/('linkage-'+name+'.stdout')).read_text()
  if any(str(lib/n) not in linkage for n in ('libggml-cpu.so','libggml-base.so')):raise ValueError('unexpected reference library resolution')
  return binary
 for name,source in [('default','tests/llm_dense_layer_default_test.cpp'),('checks','tests/llm_dense_layer_checks_test.cpp'),('failures','tests/llm_dense_layer_failure_test.cpp')]:run(name,[compile_unit(name,source,True)])
 run('admission',[compile_unit('admission','tests/llm_dense_layer_test.cpp',True),'admission'])
 if numeric:run('numeric',[compile_unit('numeric','tests/llm_dense_layer_test.cpp',False),'numeric'],900)
 if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('fixture inputs changed')
 result=dict(status='PASS',classification='synthetic-dense-layer-host-arithmetic' if numeric else 'synthetic-dense-layer-admission',inputs=inputs,commands=commands,gpu_executed=False,native_executed=False,limits=['Existing CPU libraries are not instrumented; no complete producer closure reauthentication','Seven fixed deterministic F16 learned matrices; F32 vector replaces embedding lookup','Host kernel interpretation; native ISA and complete dense model admission remain unqualified'])
 (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(result['classification']+' PASS')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--numeric',action='store_true');a=p.parse_args();build(a.cpu_build,a.output,a.numeric)
