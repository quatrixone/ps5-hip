"""Compile the actual native helper load selection with tiny API lifetime doubles."""
from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
class OriginalLoad(unittest.TestCase):
 def test_loader_selection_and_lifetime(self):
  source=(ROOT/'ports/ggml/native_full.h').read_text()
  block=source[source.index(' // Keep source descriptors'):source.index(' if(generation){clock.report("load"')]
  gate=source[source.index(' require(model_source=='):source.index(' const bool smoke=')]
  code=r'''
#include <memory>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <cassert>
int mode=0,live=0,file_calls=0,user_calls=0;bool original=false;
void require(bool ok){if(!ok)throw std::runtime_error("reject");}
struct gguf_context{};void gguf_free(gguf_context* p){delete p;--live;}
namespace direct_tiny{using Metadata=std::unique_ptr<gguf_context,decltype(&gguf_free)>;Metadata construct(const char*){if(mode==3)throw std::runtime_error("read");++live;return Metadata(new gguf_context,gguf_free);}std::string read(const std::filesystem::path&,int){return "config";}int json(const std::string&){return 1;}}
namespace ps5hip{struct SafetensorsReader{SafetensorsReader(const std::string&){++live;}~SafetensorsReader(){--live;}};
struct TinyLlamaTensorCallback{TinyLlamaTensorCallback(SafetensorsReader&,int){++live;}~TinyLlamaTensorCallback(){--live;}void add_descriptors(gguf_context*){}static void callback(void*,void*){}bool complete(){return mode!=1;}};}
struct llama_model{};struct Params{int n_gpu_layers=99;};
void llama_model_free(llama_model* p){if(original)assert(live==3);delete p;}
llama_model* llama_model_load_from_file(const char*,Params p){assert(p.n_gpu_layers==99);++file_calls;return new llama_model;}
llama_model* llama_model_init_from_user(gguf_context*,void(*)(void*,void*),void*,Params p){assert(p.n_gpu_layers==99&&live==3);++user_calls;return mode==2?nullptr:new llama_model;}
struct Flag{bool load(){return mode==4;}};struct Logs{Flag unexpected;};
enum class NativeFullModelSource{gguf,original_safetensors};
void run(NativeFullModelSource model_source){@GATE@ const char* model_path="explicit/path";Params mp;Logs logs;@BLOCK@}
int main(){run(NativeFullModelSource::gguf);assert(file_calls==1&&user_calls==0&&live==0);
 original=true;
#ifdef PS5HIP_NATIVE_ORIGINAL_MODEL
 run(NativeFullModelSource::original_safetensors);assert(file_calls==1&&user_calls==1&&live==0);
 for(mode=1;mode<=4;++mode){bool caught=false;try{run(NativeFullModelSource::original_safetensors);}catch(...){caught=true;}assert(caught&&live==0);}
#else
 bool caught=false;try{run(NativeFullModelSource::original_safetensors);}catch(...){caught=true;}assert(caught&&file_calls==1&&user_calls==0&&live==0);
#endif
 bool invalid=false;try{run(static_cast<NativeFullModelSource>(99));}catch(...){invalid=true;}assert(invalid&&live==0);
}
'''.replace('@BLOCK@',block).replace('@GATE@',gate)
  with tempfile.TemporaryDirectory() as tmp:
   tmp=Path(tmp);cpp=tmp/'test.cpp';cpp.write_text(code)
   for enabled in (False,True):
    exe=tmp/('test'+str(enabled));command=['clang++-18','-std=c++17','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all',*(['-DPS5HIP_NATIVE_ORIGINAL_MODEL=1'] if enabled else []),str(cpp),'-o',str(exe)]
    subprocess.run(command,check=True,timeout=60);subprocess.run([str(exe)],check=True,timeout=10)
 def test_existing_gpu_configuration_and_decoder_unchanged(self):
  text=(ROOT/'ports/ggml/native_full.h').read_text()
  self.assertIn('mp.n_gpu_layers=99;mp.use_extra_bufts=false;mp.check_tensors=true;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF',text)
  self.assertIn('mp.devices=devices',text);self.assertIn('mp.tensor_buft_overrides=overrides',text)
if __name__=='__main__':unittest.main()
