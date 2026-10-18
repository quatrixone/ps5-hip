"""Extracted production optional-tensor branch; no patched API/runtime claim."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import prepare_llm_native_loader as loader
SOURCE=ROOT/'.deps/sources/llama-client/src/llama-model-loader.cpp'
class UserLoader(unittest.TestCase):
 def test_narrow_shared_render(self):
  original=SOURCE.read_text();user=loader.render_user(original)
  self.assertEqual(user.replace(loader.USER_GUARD,''),original)
  self.assertEqual(loader.render(original).count(loader.USER_GUARD),1)
  for value in (user,original.replace(loader.USER_ANCHOR,''),original+loader.USER_ANCHOR):
   with self.assertRaises(ValueError):loader.render_user(value)
 def test_extracted_production_optional_branch(self):
  source=loader.render_user(SOURCE.read_text());start=source.index(loader.USER_ANCHOR);end=source.index('        ggml_tensor t_meta;',start)
  body=source[start:end]+'        selected=type;return &selected;\n    }\n    return &file_backed;'
  code=r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
enum {TENSOR_SKIP_IF_VIRTUAL=1,TENSOR_NOT_REQUIRED=2,GGML_TYPE_F32=0,GGML_TYPE_Q4_0=2};using ggml_type=int;
int selected=-1,file_backed=9,lookup=0,tid=-1;void* metadata=nullptr;
int64_t gguf_find_tensor(void*,const char*){++lookup;return tid;}
int gguf_get_tensor_type(void*,int64_t){return GGML_TYPE_Q4_0;}
struct Name{std::string str(){return "weight";}}tn;
int* run(bool user,int flags,std::initializer_list<int64_t> ne){std::vector<int> files;if(!user)files.push_back(1);@BODY@}
int main(){
 for(bool user:{false,true})for(int flags=0;flags<4;++flags)for(int descriptor:{-1,0}){
  tid=descriptor;lookup=0;selected=-1;auto p=run(user,flags,{4,4});
  if(!user){assert(p==&file_backed&&lookup==0);continue;}
  if(flags&TENSOR_SKIP_IF_VIRTUAL){assert(p==nullptr&&lookup==0);continue;}
  assert(lookup==1);
  if((flags&TENSOR_NOT_REQUIRED)&&tid==-1)assert(p==nullptr&&selected==-1);
  else assert(p==&selected&&selected==(tid==-1?GGML_TYPE_F32:GGML_TYPE_Q4_0));
 }
 tid=0;assert(run(true,TENSOR_NOT_REQUIRED,{0,4})==nullptr);
}
'''.replace('@BODY@',body)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'test.cpp').write_text(code)
   subprocess.run(['clang++-18','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',str(root/'test.cpp'),'-o',str(root/'test')],check=True,timeout=30)
   subprocess.run([str(root/'test')],check=True,timeout=10)
 def test_cpu_cmake_substitution_without_build(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'source';src.mkdir();(src/'src').mkdir();generated=root/'generated.cpp';generated.write_text('int generated;\n')
   (src/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.19)\nproject("llama.cpp" CXX)\nadd_subdirectory(src)\n')
   (src/'src/CMakeLists.txt').write_text('add_library(llama llama-model-loader.cpp keep.cpp)\nset_target_properties(llama PROPERTIES UNITY_BUILD ON)\nset_source_files_properties(llama-model-loader.cpp keep.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)\n')
   for name in ('llama-model-loader.cpp','keep.cpp'):(src/'src'/name).write_text('int keep;\n')
   command=['cmake','-S',str(src),'-B',str(root/'build'),'-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DCMAKE_PROJECT_INCLUDE='+str(ROOT/'tools/llm_user_loader.cmake'),'-DPS5HIP_USER_LOADER='+str(generated)]
   subprocess.run(command,check=True,capture_output=True,timeout=30)
   files={row['file'] for row in json.loads((root/'build/compile_commands.json').read_text())}
   self.assertEqual(files,{str(generated),str(src/'src/keep.cpp')})
   (src/'src/CMakeLists.txt').write_text('add_library(llama keep.cpp)\n')
   result=subprocess.run(command,capture_output=True,text=True,timeout=30)
   self.assertNotEqual(result.returncode,0);self.assertIn('upstream loader target changed',result.stderr)
if __name__=='__main__':unittest.main()
