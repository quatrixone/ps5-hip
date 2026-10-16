"""Rebuild a locked native SDK payload loader/tokenizer client. No execution claim."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dependencies.json').is_file())
sys.path.insert(0,str(ROOT/'tools'))
from fetch_locked_deps import source,read
BASE=('ggml.c','ggml.cpp','ggml-alloc.c','ggml-backend.cpp','ggml-backend-meta.cpp','ggml-threading.cpp','ggml-quants.c','gguf.cpp','ggml-backend-reg.cpp','ggml-backend-dl.cpp')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def bind(inputs,p):
 p=str(Path(p).absolute());h=sha(p)
 if p in inputs and inputs[p]!=h:raise ValueError('input changed: '+p)
 inputs[p]=h

def selection(client):
 text=(client/'src/CMakeLists.txt').read_text();match=re.search(r'set\(LLAMA_CORE_SOURCES(.*?)\)',text,re.S)
 if not match:raise ValueError('core selection missing')
 core=match.group(1).split()
 if not core or any(not n.endswith('.cpp') or '/' in n for n in core):raise ValueError('unexpected core source selection')
 cpu_text=(client/'ggml/src/ggml-cpu/CMakeLists.txt').read_text();match=re.search(r'list\s*\(APPEND GGML_CPU_SOURCES(.*?)\)',cpu_text,re.S)
 if not match:raise ValueError('CPU selection missing')
 cpu=[n for n in match.group(1).split() if n.endswith(('.c','.cpp'))]
 cpu+=['ggml-cpu/arch/x86/quants.c','ggml-cpu/arch/x86/repack.cpp']
 units=[client/'ggml/src'/n for n in BASE]+[client/'src'/n for n in core]+[client/'ggml/src'/n for n in cpu]
 models=sorted((client/'src/models').glob('*.cpp'))
 if len(units)!=59 or len(models)!=161 or len(set(units))!=len(units):raise ValueError('locked source selection differs')
 if any(not p.is_file() for p in units):raise ValueError('source missing')
 return units,models

def build(output):
 output=Path(output).resolve()
 if output.exists() or not (output.is_relative_to(ROOT/'build') or output.is_relative_to(ROOT/'results')):raise ValueError('fresh build/results output required')
 deps=read('dependencies.json');lock=read('llm-inputs.json')['client'];client=source('llama-client',lock,False);sdk_source=source('sdk',deps['sdk'],False);sdk=ROOT/'.deps/sdk'
 if (sdk/'.ps5-sdk-revision').read_text().strip()!=deps['sdk']['revision']:raise ValueError('SDK revision marker mismatch')
 units,models=selection(client);inputs={};symlinks={};mutex=threading.Lock();commands=[]
 for p in [Path(__file__),Path(__file__).with_name('llm_native_client_main.cpp'),ROOT/'tools/fetch_locked_deps.py',ROOT/'tools/ps5-clangxx',ROOT/'dependencies.json',ROOT/'llm-inputs.json',client/'src/CMakeLists.txt',client/'ggml/src/CMakeLists.txt',client/'ggml/src/ggml-cpu/CMakeLists.txt',*units,*models]:bind(inputs,p)
 sdk_files=sorted(p for p in sdk.rglob('*') if p.is_file())
 for p in sdk_files:
  bind(inputs,p)
  if p.is_symlink():symlinks[str(p)]=str(p.resolve())
 tools={};env={k:v for k,v in os.environ.items() if k not in ('LD_PRELOAD','LD_AUDIT','LD_LIBRARY_PATH','CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','LIBRARY_PATH','COMPILER_PATH','GCC_EXEC_PREFIX','SDKROOT','CFLAGS','CXXFLAGS','LDFLAGS')};env['PATH']='/usr/bin:/bin';env['LLVM_CONFIG']='/usr/bin/llvm-config-18'
 for name in ('clang++-18','ld.lld-18','llvm-config-18','llvm-readobj-18','llvm-nm-18','bash','sh','readlink','realpath','dirname','basename','cut'):
  p=Path(shutil.which(name,path=env['PATH']));bind(inputs,p);tools[name]={'path':str(p),'resolved':str(p.resolve()),'sha256':sha(p)}
  if name.endswith('-18'):
   tools[name]['version']=subprocess.check_output([p,'--version'],text=True,env=env)
   if '18.1.8' not in tools[name]['version']:raise ValueError('LLVM18.1.8 required')
  shared=subprocess.run(['ldd',str(p)],capture_output=True,text=True,env=env)
  for line in shared.stdout.splitlines():
   parts=line.split();name=parts[2] if len(parts)>2 and parts[1]=='=>' else parts[0] if parts else ''
   if name.startswith('/') and Path(name).is_file():bind(inputs,name)
 bind(inputs,sys.executable)
 output.mkdir(parents=True)
 def run(label,argv,timeout=240):
  row={'label':label,'argv':list(map(str,argv))}
  try:
   p=subprocess.run(row['argv'],cwd=ROOT,capture_output=True,text=True,timeout=timeout,env=env);row['exit_code']=p.returncode
   (output/(label+'.stdout')).write_text(p.stdout);(output/(label+'.stderr')).write_text(p.stderr)
  except subprocess.TimeoutExpired as e:
   row.update(timed_out=True,deadline_seconds=timeout)
   for ext,val in [('stdout',e.stdout),('stderr',e.stderr)]:(output/(label+'.'+ext)).write_bytes(val.encode() if isinstance(val,str) else val or b'')
   raise
  finally:
   with mutex:
    commands.append(row);(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
  if p.returncode:raise RuntimeError(label+': '+p.stderr[-1600:])
  return p.stdout
 for lib in ('llama','ggml'):
  template=client/('src/llama-version.h.in' if lib=='llama' else 'ggml/src/ggml-version.h.in');bind(inputs,template)
  generated=output/(lib+'-version.h');generated.write_text(template.read_text().replace('@'+lib.upper()+'_VERSION@','ps5-hip-native-link').replace('@'+lib.upper()+'_BUILD_COMMIT@',lock['revision']));bind(inputs,generated)
 for i in range(0,len(models),16):
  p=output/('unity-'+str(i//16)+'.cpp');p.write_text('\n'.join('#include "'+str(s)+'"' for s in models[i:i+16])+'\n');bind(inputs,p);units.append(p)
 units.append(Path(__file__).with_name('llm_native_client_main.cpp'))
 if len(units)!=71:raise ValueError('expected exact71 object closure')
 flags=['-O0','-fPIC','-ffunction-sections','-fdata-sections','-pthread','-DGGML_SCHED_MAX_COPIES=4','-DGGML_USE_CPU']+['-I'+str(p) for p in [output,client/'include',client/'src',client/'ggml/include',client/'ggml/src',client/'ggml/src/ggml-cpu']]
 compiler=ROOT/'tools/ps5-clangxx'
 def compile(pair):
  i,p=pair;obj=output/(str(i)+'.o');dep=output/(str(i)+'.d');args=[compiler,*flags,*(['-x','c','-std=c11'] if p.suffix=='.c' else ['-std=c++17'])]
  run('headers-'+str(i),[*args,'-c','-M',p,'-MF',dep])
  paths=[Path(n).resolve() for n in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])]
  with mutex:
   for header in paths:bind(inputs,header)
  actual_dep=output/(str(i)+'.compiled.d')
  run('compile-'+str(i),[*args,'-MD','-MF',actual_dep,'-c',p,'-o',obj])
  actual_paths={Path(n).resolve() for n in shlex.split(actual_dep.read_text().replace('\\\n',' ').split(':',1)[1])}
  if actual_paths!=set(paths):raise ValueError('actual compile dependencies differ')
  print('compiled',i,p.name,flush=True)
  with mutex:bind(inputs,obj)
  return obj
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:objects=list(pool.map(compile,enumerate(units)))
 run('driver-plan',[compiler,'-###','-Wl,--gc-sections','-Wl,--trace',*objects,'-o',output/'client.elf'])
 plan_lines=[shlex.split(line) for line in (output/'driver-plan.stderr').read_text().splitlines() if line.lstrip().startswith('\"')]
 if len(plan_lines)!=1:raise ValueError('unexpected linker driver plan')
 plan=plan_lines[0]
 if Path(plan[0]).resolve()!=(sdk/'bin/prospero-lld').resolve():raise ValueError('unexpected linker wrapper')
 llvm_dir=Path(subprocess.check_output([tools['llvm-config-18']['path'],'--bindir'],text=True,env=env).strip())
 if sha(llvm_dir/'ld.lld')!=tools['ld.lld-18']['sha256']:raise ValueError('unexpected actual LLVM linker')
 expected_dirs=[sdk/'target/lib',sdk/'target/user/homebrew/lib']
 if [Path(a[2:]) for a in plan if a.startswith('-L')]!=expected_dirs:raise ValueError('unexpected link search directory')
 if [a[2:] for a in plan if a.startswith('-l')]!=['unwind','c++abi','c++','c','kernel_web','SceLibcInternal','SceNet']:raise ValueError('unexpected requested link libraries')
 for i,arg in enumerate(plan):
  if Path(arg).is_absolute() and (i==0 or plan[i-1]!='-o'):
   if str(Path(arg).absolute()) not in inputs:raise ValueError('unbound planned link input: '+arg)
 # Every possible library in the only permitted search roots was authenticated before compilation.
 for path,digest in inputs.items():
  if sha(path)!=digest:raise ValueError('input changed before link: '+path)
 run('link',[compiler,'-Wl,--gc-sections','-Wl,--trace',*objects,'-o',output/'client.elf'])
 # Linker trace enumerates every selected CRT/archive/import DSO (archive members keep their container identity).
 trace=(output/'link.stdout').read_text()
 link_inputs=[]
 for line in trace.splitlines():
  name=line.split('(',1)[0].strip();p=Path(name)
  if not p.is_absolute():p=ROOT/p
  if p.is_file():
   if str(p.absolute()) not in inputs:raise ValueError('unbound actual linker input: '+str(p))
   bind(inputs,p);link_inputs.append(str(p))
 if not link_inputs:raise ValueError('missing linker trace')
 run('headers',['llvm-readobj-18','--file-headers','--program-headers','--needed-libs',output/'client.elf'])
 imports=run('imports',['llvm-nm-18','--dynamic','--undefined-only',output/'client.elf'])
 names=[line.split()[-1] for line in imports.splitlines() if line.strip()]
 if len(names)!=116 or any(n.startswith(('ggml_','llama_','_Z')) for n in names):raise ValueError('unexpected unresolved client imports')
 source('llama-client',lock,False);source('sdk',deps['sdk'],False)
 if sorted(str(p) for p in sdk.rglob('*') if p.is_file())!=sorted(map(str,sdk_files)):raise ValueError('SDK inventory changed')
 for p,target in symlinks.items():
  if str(Path(p).resolve())!=target:raise ValueError('SDK link changed')
 for p,h in inputs.items():
  if sha(p)!=h:raise ValueError('input changed before publication: '+p)
 artifacts={str(p.relative_to(output)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in output.rglob('*') if p.is_file()}
 receipt={'status':'PASS','classification':'source-only-native-client-payload-link','client_revision':lock['revision'],'sdk_revision':deps['sdk']['revision'],'inputs':inputs,'sdk_symlinks':symlinks,'tools':tools,'units':[str(p) for p in units],'objects':{str(p):sha(p) for p in objects},'object_count':71,'model_sources':[str(p) for p in models],'link_inputs':link_inputs,'commands':commands,'artifacts':artifacts,'dynamic_imports':names,'native_executed':False,'gpu_executed':False,'limits':['SDK payload ELF only; RWX segment is not accepted app packaging.','Vocab-only native caller; no model loaded or tokenized here and no public HIP backend integrated.','Installed SDK bytes and clean source revision bound; SDK source-build provenance is inherited from deps-build, not recreated by this target.','No general model support or inference claim from retaining all upstream factory implementations.']}
 (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS',output/'receipt.json',flush=True);return receipt
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();build(args.output)
