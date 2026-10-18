import os, copy, hashlib, json, pathlib, subprocess, sys, tempfile, unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO=ROOT if (ROOT/'models').exists() else ROOT.parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(REPO/'tools')]
import external_original_model as original
import build_public_hip as builder
class OriginalTitle(unittest.TestCase):
 def fixture(self,folder):
  identities={}
  for name in original.NAMES:
   data=name.encode();(folder/name).write_bytes(data);identities[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
  lock={'metadata':{k:v for k,v in identities.items() if k!='model.safetensors'},'weights':{'file_bytes':identities['model.safetensors']['bytes'],'expected_file_sha256':identities['model.safetensors']['sha256']}}
  return original.contract(folder,'/data/hip/models/tinyllama-source',lock),lock
 def test_locked_contract_and_mutations(self):
  with tempfile.TemporaryDirectory() as d:
   folder=pathlib.Path(d);value,lock=self.fixture(folder)
   self.assertEqual(original.validate(json.loads(original.encode(value)),lock),value)
   for field,bad in [('schema',True),('directory','/data/hip/models/../x'),('directory','/data/hip/models/x/'),('format','gguf')]:
    v=copy.deepcopy(value);v[field]=bad
    with self.subTest(field=field,bad=bad),self.assertRaises(ValueError):original.validate(v,lock)
   for field,bad in [('bytes',1),('sha256','0'*64)]:
    v=copy.deepcopy(value);v['files']['model.safetensors'][field]=bad
    with self.assertRaises(ValueError):original.validate(v,lock)
   (folder/'model.safetensors').write_bytes(b'x'*len('model.safetensors'))
   with self.assertRaises(ValueError):original.contract(folder,value['directory'],lock)
 def test_exact_package_assets(self):
  with tempfile.TemporaryDirectory() as d:
   folder=pathlib.Path(d);value,_=self.fixture(folder);assets={}
   for name,data in [('vocab.gguf',b'vocab'),('client-phase.txt',b'model-full\n'),('original-model.json',original.encode(value))]+[('model-layer/'+n+'.hsaco',b'module') for n in ('getrows','normalized','weighted','q8','q4','layer')]:
    p=folder/name;p.parent.mkdir(exist_ok=True);p.write_bytes(data);assets[name]=p
   builder.validate_client_assets(assets)
   with self.assertRaises(ValueError):builder.validate_client_assets({**assets,'model-path.txt':assets['vocab.gguf']})
   with self.assertRaises(ValueError):builder.validate_client_assets({k:v for k,v in assets.items() if k!='original-model.json'})
 def test_compile_scope(self):
  with tempfile.TemporaryDirectory() as d:
   folder=pathlib.Path(d);source=folder/'source';hip=source/'hipamd/src';hip.mkdir(parents=True)
   for i in range(30):(hip/(str(i)+'.cpp')).write_text('')
   calls=[]
   def compile_unit(output,path,obj,flags,label,source_root=None):
    calls.append((path.name,flags));return {'object':str(obj.relative_to(output))}
   with patch.object(builder,'compiler_flags',return_value=[]),patch.object(builder,'compile_bound_unit',side_effect=compile_unit),patch.object(builder,'_plan_bound_unit',side_effect=lambda *a,**kw: ({},lambda _: compile_unit(*a,**kw))):
    builder.compile_units(folder,source,'llm_client',client_phase='model-full',full_profile='generation',model_format='original_safetensors')
   marked=[n for n,f in calls if '-DPS5HIP_NATIVE_ORIGINAL_MODEL=1' in f]
   self.assertEqual(marked,['public_hip_llm_client_console_title.cpp'])
   for phase,profile in [('model-full','corpus'),('model-full','smoke'),('vocab','generation')]:
    with self.assertRaises(ValueError):builder.compile_units(folder,source,'llm_client',client_phase=phase,full_profile=profile,model_format='original_safetensors')
 def test_native_path_inventory(self):
  with tempfile.TemporaryDirectory() as d:
   folder=pathlib.Path(d);value,_=self.fixture(folder);config=folder/'original.json';cpp=folder/'test.cpp';binary=folder/'test'
   cpp.write_text('''#include <sys/stat.h>
#include <string>
#include <cstring>
#include <cstdlib>
static int fake_lstat(const char* p,struct stat* st){std::string s(p);st->st_mode=S_IFDIR;st->st_size=0;if(std::getenv("PS5HIP_TEST_SYMLINK")&&s=="/data/hip/models"){st->st_mode=S_IFLNK;return 0;}if(s.find(".",s.find_last_of('/'))!=std::string::npos){st->st_mode=S_IFREG;st->st_size=s.size()-s.find_last_of('/')-1;}return 0;}
#define lstat fake_lstat
#include "native_original_config.h"
#undef lstat
int main(int,char** argv){try{return ps5hip::original_model_directory(argv[1])=="/data/hip/models/tinyllama-source"?0:2;}catch(...){return 1;}}
''')
   subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(ROOT/'ports/ggml'),'-isystem',str(REPO/'ports/ggml'),'-I'+str(REPO/'.deps/sources/llama-client/ggml/include'),'-I'+str(REPO/'.deps/sources/llama-client/include'),'-I'+str(REPO/'.deps/sources/llama-client/vendor'),str(cpp),'-o',str(binary)],check=True)
   def run(v):config.write_bytes(original.encode(v));return subprocess.run([str(binary),str(config)]).returncode
   self.assertEqual(run(value),0)
   self.assertEqual(subprocess.run([str(binary),str(config)],env={**os.environ,"PS5HIP_TEST_SYMLINK":"1"}).returncode,1)
   v=copy.deepcopy(value);v['files']['model.safetensors']['bytes']+=1;self.assertEqual(run(v),1)
   v=copy.deepcopy(value);v['files']['extra']={'bytes':1,'sha256':'0'*64};config.write_text(json.dumps(v));self.assertEqual(subprocess.run([str(binary),str(config)]).returncode,1)
if __name__=='__main__':unittest.main()
