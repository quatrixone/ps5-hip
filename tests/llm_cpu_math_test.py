"""Compare standalone GPU-compatible math to the existing pinned AVX2 CPU library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def run(cpu, output=None):
    cpu=Path(cpu).resolve();lib=cpu/'cmake/bin';cache=cpu/'cmake/CMakeCache.txt'
    if output is not None:
        output=Path(output).resolve()
        if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results output required')
    configuration={line.split('=',1)[0]:line.split('=',1)[1] for line in cache.read_text().splitlines() if '=' in line and not line.startswith(('#','//'))}
    if any(configuration.get(key)!=value for key,value in {'GGML_AVX2:BOOL':'ON','GGML_FMA:BOOL':'ON','GGML_AVX512:BOOL':'OFF'}.items()):raise ValueError('pinned AVX2/FMA CPU configuration required')
    inputs={str(p):sha(p) for p in [Path(__file__).resolve(),ROOT/'tests/llm_cpu_math_test.cpp',ROOT/'kernels/llm_cpu_math.h',cache,Path('/usr/bin/clang++-18'),*sorted({p.resolve() for p in lib.glob('libggml*.so*')})]}
    env={k:v for k,v in os.environ.items() if not k.startswith(('LD_','GGML_','OMP_')) and k not in ('CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX')}
    env.update(ASAN_OPTIONS='halt_on_error=1:detect_leaks=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
    results=[]
    with tempfile.TemporaryDirectory(prefix='ps5-hip-cpu-math-') as directory:
        for mode,options in [('sanitized',['-O1','-fsanitize=address,undefined','-fno-omit-frame-pointer']),('optimized',['-O2'])]:
            binary=Path(directory)/mode
            command=['/usr/bin/clang++-18','-std=c++17','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror',*options,'-I'+str(ROOT/'kernels'),str(ROOT/'tests/llm_cpu_math_test.cpp'),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--no-as-needed','-lggml-cpu','-lggml-base','-o',str(binary)]
            built=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
            if built.returncode:raise RuntimeError(built.stderr)
            tested=subprocess.run([str(binary)],env=env,capture_output=True,text=True,timeout=30)
            results.append(dict(mode=mode,command=command,exit_code=tested.returncode,stdout=tested.stdout,stderr=tested.stderr))
            if tested.returncode or any(s in tested.stderr for s in ('AddressSanitizer','runtime error:','UndefinedBehaviorSanitizer','LeakSanitizer')):raise RuntimeError(tested.stdout+tested.stderr)
    if any(sha(Path(p))!=digest for p,digest in inputs.items()):raise ValueError('math test inputs changed')
    receipt=dict(status='PASS',inputs=inputs,runs=results,native_executed=False,gpu_executed=False,limits=['Bitwise comparison against existing AVX2/FMA CPU library; no full CPU-build provenance reauthentication','Device ISA, denormal modes and native execution remain unqualified','Softmax requires positive width multiple8, finite row maximum; -Inf mask allowed, all-masked/NaN rows unsupported'])
    if output is not None:
        output.mkdir();(output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for result in results:print(result['mode']+': '+result['stdout'].strip())
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-build',type=Path,default=ROOT/'build/llm-cpu-reference');parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.cpu_build,args.output)
