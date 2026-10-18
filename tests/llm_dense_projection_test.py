"""Exercise the exact bounded dense kernel against existing ggml CPU F16 dot/conversion."""
import argparse,hashlib,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(cpu):
 cpu=Path(cpu).resolve();lib=cpu/'cmake/bin';cache=cpu/'cmake/CMakeCache.txt'
 config={s.split('=',1)[0]:s.split('=',1)[1] for s in cache.read_text().splitlines() if '=' in s and not s.startswith(('#','//'))}
 if any(config.get(k)!=v for k,v in {'GGML_AVX2:BOOL':'ON','GGML_FMA:BOOL':'ON','GGML_AVX512:BOOL':'OFF'}.items()):raise ValueError('pinned AVX2/FMA CPU configuration required')
 sources=[Path(__file__).resolve(),ROOT/'tests/llm_dense_projection_test.cpp',ROOT/'kernels/llm_layer.hip',ROOT/'kernels/llm_cpu_math.h',cache,Path('/usr/bin/clang++-18'),(lib/'libggml-cpu.so').resolve(),(lib/'libggml-base.so').resolve()]
 inputs={str(p):sha(p) for p in sources}
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_','ASAN_','UBSAN_','LSAN_')) and k not in ('CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 env.update(ASAN_OPTIONS='halt_on_error=1:detect_leaks=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',OMP_NUM_THREADS='1')
 rows=[]
 with tempfile.TemporaryDirectory(prefix='ps5hip-dense-f16-') as d:
  for mode,flags in [('sanitized',['-O1','-fsanitize=address,undefined,float-cast-overflow','-fno-sanitize-recover=all','-fno-omit-frame-pointer']),('optimized',['-O2'])]:
   binary=Path(d)/mode;cmd=['/usr/bin/clang++-18','-std=c++17','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off',*flags,str(ROOT/'tests/llm_dense_projection_test.cpp'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--no-as-needed','-lggml-cpu','-lggml-base','-o',str(binary)]
   built=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=60)
   if built.returncode:raise RuntimeError(built.stderr)
   linkage=subprocess.check_output(['/usr/bin/ldd',binary],env=env,text=True)
   if str(lib/'libggml-cpu.so') not in linkage:raise ValueError('unexpected CPU library resolution')
   tested=subprocess.run([str(binary)],env=env,capture_output=True,text=True,timeout=120)
   if tested.returncode or any(s in tested.stderr for s in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer','LeakSanitizer')):raise RuntimeError(tested.stdout+tested.stderr)
   if tested.stdout!='DENSE_F16_PRIMITIVE_PASS shapes=4 widths=2048,5632 all_rows=9984 actual_cpu_dot=true GPU_execution=false\n':raise ValueError('completion marker')
   rows.append(dict(mode=mode,command=cmd,exit_code=tested.returncode,stdout=tested.stdout,stderr=tested.stderr,linkage=linkage));print(mode+': '+tested.stdout.strip(),flush=True)
 if any(sha(Path(p))!=h for p,h in inputs.items()):raise ValueError('primitive inputs changed')
 return dict(status='PASS',inputs=inputs,runs=rows,native_executed=False,gpu_executed=False,limits=['Actual existing AVX2/FMA CPU dot and converter reference; no CPU build closure reauthentication','No graph admission, learned-projection dispatch or whole-layer qualification','Host execution of exact kernel body; native ISA/FTZ unqualified','Existing reference libraries uninstrumented'])
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-build',type=Path,required=True);p.add_argument('--report',type=Path);a=p.parse_args()
 if a.report and a.report.exists():p.error('fresh report path required')
 result=run(a.cpu_build)
 if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
