"""Compile and test the native dense helper with HIP transport mocked on the host."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from dense_layer_reference import validate
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def capture_inputs(files,reference,oracle):
 inputs={str(p.resolve()):sha(p) for p in files}
 if inputs[str((reference/'reference.bin').resolve())]!=oracle['sha256'] or inputs[str((reference/'reference.json').resolve())]!=oracle['manifest_sha256']:raise ValueError('validated reference changed')
 return inputs
def build(cpu,reference,output):
 cpu=cpu.resolve();reference=reference.resolve();output=output.resolve();lib=cpu/'cmake/bin'
 if output.exists():raise ValueError('fresh output required')
 if not (cpu/'receipt.json').is_file():raise ValueError('existing CPU build receipt required; no dependency rebuild')
 if not all(x in Path('/proc/cpuinfo').read_text().split() for x in ('avx2','fma','f16c')):raise ValueError('AVX2/FMA/F16C required')
 oracle=validate(reference/'reference.bin',reference/'reference.json')
 files=[Path(__file__),ROOT/'ports/ggml/ps5_backend.cpp',ROOT/'tests/llm_native_dense_layer_test.cpp',ROOT/'tests/llm_layer_numeric_mock.h',ROOT/'tools/dense_layer_reference.py',cpu/'receipt.json',Path('/usr/bin/clang++-18').resolve(),reference/'reference.bin',reference/'reference.json']
 files+=list((ROOT/'ports/ggml').glob('*.h'))+list((ROOT/'ports/ggml').glob('*.def'))+list((ROOT/'kernels').glob('llm*'))
 files += [(lib/n).resolve() for n in ('libggml.so','libggml-base.so')]
 inputs=capture_inputs(files,reference,oracle);output.mkdir(parents=True)
 (output/'inputs.json').write_text(json.dumps(inputs,indent=2))
 env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_','ASAN_','UBSAN_','LSAN_')) and k not in ('CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
 env.update(OMP_NUM_THREADS='2',ASAN_OPTIONS='halt_on_error=1:detect_leaks=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
 commands=[]
 def run(label,cmd,timeout=180,leaks=True):
  e=dict(env);e['ASAN_OPTIONS']='halt_on_error=1:detect_leaks='+str(int(leaks));start=time.monotonic()
  with (output/(label+'.stdout')).open('wb') as out,(output/(label+'.stderr')).open('wb') as err:p=subprocess.run(list(map(str,cmd)),stdout=out,stderr=err,env=e,timeout=timeout)
  commands.append(dict(label=label,argv=list(map(str,cmd)),exit_code=p.returncode,seconds=time.monotonic()-start,leak_detection=leaks));(output/'commands.json').write_text(json.dumps(commands,indent=2));p.check_returncode()
  if any(x in (output/(label+'.stderr')).read_text() for x in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:','LeakSanitizer')):raise ValueError('sanitizer diagnostic')
 flags=['-std=c++17','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-mavx2','-mfma','-mf16c','-fno-fast-math','-ffp-contract=off','-D__HIP_PLATFORM_AMD__']
 flags+=['-I'+str(ROOT/n) for n in ('ports/ggml','ports/hip','.deps/sources/hip-headers/include','.deps/sources/clr/hipamd/include','.deps/sources/llama-client/ggml/include','.deps/sources/llama-client/ggml/src')]
 binary=output/'test'
 run('compile',['/usr/bin/clang++-18',*flags,'-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer',ROOT/'tests/llm_native_dense_layer_test.cpp',ROOT/'ports/ggml/ps5_backend.cpp','-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lggml','-lggml-base','-o',binary])
 run('linkage',['/usr/bin/ldd',binary]);linkage=(output/'linkage.stdout').read_text()
 if any(str(lib/n) not in linkage for n in ('libggml.so','libggml-base.so')):raise ValueError('library resolution mismatch')
 for mode in ('missing-symbol','launch','sync','writer','readback','numeric'):
  run(mode,[binary,mode,reference/'reference.bin'],600,mode=='numeric')
  marker='NATIVE_DENSE_HOST_PASS' if mode=='numeric' else 'NATIVE_DENSE_QUARANTINE_PASS mode='+mode+' frees=0'
  if marker not in (output/(mode+'.stdout')).read_text():raise ValueError('missing completion marker')
 if any(sha(Path(p))!=v for p,v in inputs.items()):raise ValueError('inputs changed')
 receipt=dict(status='PASS',classification='synthetic-native-dense-helper-host-transport',native_executed=False,gpu_executed=False,inputs=inputs,binary_sha256=sha(binary),reference=oracle,commands=commands,requests=526,records=11084,bytes=154392576,limits=['Existing libraries are uninstrumented and no full CPU producer closure is reauthenticated','Failure-path leak checks disabled because fatal quarantine intentionally retains metadata/state','HIP kernel interpretation, not native execution; no model loading'])
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Native dense helper host qualification PASS')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-build',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.cpu_build,a.reference,a.output)
