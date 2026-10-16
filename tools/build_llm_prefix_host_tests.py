"""Reproduce prefix selector/dispatch/backend/native-helper tests with mocked HIP."""
from pathlib import Path
import argparse,json,os,re,subprocess,sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dependencies.json').is_file())
SOURCE=ROOT
sys.path[:0]=[str(SOURCE/'tools'),str(ROOT/'tools')]
import build_llm_executed_graph as provenance
import source_llm_prefix_log as contract


def build(cpu,model,output):
 cpu=Path(cpu).resolve();model=Path(model).resolve();output=Path(output).resolve()
 if output.exists() or not output.is_relative_to(ROOT/'results'):raise ValueError('fresh results output required')
 compiler=Path('/usr/bin/clang++-18');lib=cpu/'cmake/bin';client=ROOT/'.deps/sources/llama-client'
 inputs=provenance.cpu_inputs(cpu,model)
 units={'backend':SOURCE/'ports/ggml/ps5_backend.cpp','dispatch':SOURCE/'tests/llm_prefix_dispatch_test.cpp','selector':SOURCE/'tests/llm_model_prefix_test.cpp','integration':SOURCE/'tests/llm_prefix_backend_test.cpp','helper':SOURCE/'tests/llm_native_prefix_test.cpp'}
 for p in [Path(__file__).resolve(),Path(provenance.__file__).resolve(),Path(contract.__file__).resolve(),compiler,Path('/usr/bin/ldd'),*units.values()]:provenance.merge(inputs,p,provenance.sha(p))
 output.mkdir(parents=True);(output/'raw').mkdir()
 scrub=('LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS','ASAN_SYMBOLIZER_PATH')
 env={k:v for k,v in os.environ.items() if k not in scrub};env['ASAN_OPTIONS']='halt_on_error=1:detect_leaks=1';env['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
 commands=[]
 def run(label,argv):
  args=list(map(str,argv));result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,timeout=600)
  (output/(label+'.stdout')).write_bytes(result.stdout);(output/(label+'.stderr')).write_bytes(result.stderr)
  commands.append(dict(label=label,argv=args,exit_code=result.returncode))
  (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  if result.returncode or re.search(rb'AddressSanitizer|UndefinedBehaviorSanitizer|LeakSanitizer|runtime error:',result.stderr):raise ValueError(label+' failed; inspect retained diagnostics')
  return result.stdout.decode()
 flags=['-std=c++17','-O1','-gline-tables-only','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-fsanitize=address,undefined','-D__HIP_PLATFORM_AMD__']
 flags += ['-I'+str(p) for p in (SOURCE/'ports/ggml',ROOT/'ports/ggml',ROOT/'ports/hip',ROOT/'.deps/sources/hip-headers/include',ROOT/'.deps/sources/clr/hipamd/include',client/'include',client/'ggml/include')]
 flags += ['-isystem',str(client/'ggml/src')]
 for name,source in units.items():
  run(name+'-headers',[compiler,*flags,'-M','-MF',output/(name+'.headers.d'),source])
  headers=provenance.dependencies(output/(name+'.headers.d'))
  for p in headers:provenance.merge(inputs,p,provenance.sha(p))
  run(name+'-compile',[compiler,*flags,'-MD','-MF',output/(name+'.d'),'-c',source,'-o',output/(name+'.o')])
  if provenance.dependencies(output/(name+'.d'))!=headers:raise ValueError('compile header closure changed')
 provenance.recheck(inputs)
 binaries={}
 for name in ('dispatch','selector','integration','helper'):
  objects=[output/(name+'.o')]+([output/'backend.o'] if name in ('integration','helper') else [])
  binary=output/name
  run(name+'-link',[compiler,'-fsanitize=address,undefined',*objects,'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-Wl,--no-as-needed','-lllama','-lggml','-lggml-cpu','-lggml-base','-o',binary])
  for p,h in provenance.runtime_inputs(run(name+'-ldd',['/usr/bin/ldd',binary]),lib,inputs).items():provenance.merge(inputs,p,h)
  binaries[str(binary)]=provenance.sha(binary)
 run('dispatch',[output/'dispatch']);run('selector',[output/'selector'])
 run('integration',[output/'integration',model]);run('helper',[output/'helper',model,output/'raw'])
 fragment=contract.check_fragment((output/'helper.stdout').read_bytes())
 if {p.name for p in (output/'raw').iterdir()}!=set(contract.FILES):raise ValueError('helper output inventory differs')
 for name,size in contract.FILES.items():
  data=(output/'raw'/name).read_bytes()
  if len(data)!=size or any(data):raise ValueError('mock helper produced unexpected tensor arithmetic')
 provenance.recheck({**inputs,**binaries})
 receipt=dict(schema=1,status='PASS',classification='source-only-prefix-mocked-hip-integration-tests',inputs=inputs,
  commands=commands,artifacts={str(p.relative_to(output)):provenance.sha(p) for p in output.rglob('*') if p.is_file()},
  cpu_build=str(cpu),model={'path':str(model),'bytes':model.stat().st_size,'sha256':inputs[str(model)]},
  tokens=[0,1,2,31999],original_full_graph_forwarded=False,prefix_graphs=4,mock_hip_launches=36,
  helper_fragment=fragment,sanitizer_environment={k:env[k] for k in ('ASAN_OPTIONS','UBSAN_OPTIONS')},
  native_executed=False,gpu_executed=False,numerical_verified=False,
  limits=['Public HIP API and memory-query mocks; no actual CLR/GPU transport proof',
  'Only new test/backend translation units sanitizer-instrumented; existing CPU shared libraries unchanged',
  'Host shared-library symbol interception; production native title retains mandatory linker wrapper',
  'Zero mock outputs deliberately cannot establish numerical success'])
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('PASS prefix dispatch, selector, actual-model backend route and native-helper host tests')
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('cpu-build','model','output'):parser.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args();build(a.cpu_build,a.model,a.output)
