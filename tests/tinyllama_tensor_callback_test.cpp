#include "tinyllama_tensor_callback.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <sys/stat.h>
using namespace ps5hip;static unsigned checks=0,calls=0,fail_at=0;
static void reject(const std::function<void()>& fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}assert(caught);++checks;}
static void upload(ggml_tensor*,const void*,size_t offset,size_t count){assert(count>0&&count<=256*1024);assert(offset==calls*256*1024);if(++calls==fail_at)throw std::runtime_error("injected upload");}
int main(int argc,char** argv){assert(argc==4);nlohmann::json config,header;{std::ifstream f(argv[1]);f>>config;}{std::ifstream f(argv[2]);f>>header;}const auto strict_header=header;
 std::map<std::string,SafetensorsReader::Tensor> inventory;for(auto it=header.begin();it!=header.end();++it){if(it.key()=="__metadata__")continue;auto& t=it.value();inventory.emplace(it.key(),SafetensorsReader::Tensor{t["dtype"],t["shape"].get<std::vector<uint64_t>>(),t["data_offsets"][0],t["data_offsets"][1]});}
 const auto mapping=tinyllama_tensor_mapping(config,inventory);std::unique_ptr<gguf_context,decltype(&gguf_free)> metadata(gguf_init_empty(),gguf_free);add_tinyllama_descriptors(metadata.get(),mapping);assert(gguf_get_n_tensors(metadata.get())==201);++checks;
 for(const auto& map:mapping){auto id=gguf_find_tensor(metadata.get(),map.target.c_str());assert(id>=0);assert(gguf_get_tensor_type(metadata.get(),id)==tinyllama_target_type(map));auto* ne=gguf_get_tensor_ne(metadata.get(),id);assert(uint64_t(ne[0])==map.shape.back()&&uint64_t(ne[1])==(map.shape.size()==1?1:map.shape[0]));++checks;}
 assert(gguf_find_tensor(metadata.get(),"output.weight")>=0);++checks;reject([&]{add_tinyllama_descriptors(metadata.get(),mapping);});
 auto invalid=mapping;invalid.back().target=invalid.front().target;std::unique_ptr<gguf_context,decltype(&gguf_free)> empty(gguf_init_empty(),gguf_free);reject([&]{add_tinyllama_descriptors(empty.get(),invalid);});assert(gguf_get_n_tensors(empty.get())==0);
 invalid=mapping;invalid.front().shape[0]=UINT64_MAX;reject([&]{add_tinyllama_descriptors(empty.get(),invalid);});
 const std::string path=argv[3];header={{"matrix",{{"dtype","F16"},{"shape",{256,2048}},{"data_offsets",{0,1048576}}}},{"norm",{{"dtype","F16"},{"shape",{2048}},{"data_offsets",{1048576,1052672}}}}};
 auto write=[&](){std::ofstream f(path,std::ios::binary);auto text=header.dump();uint64_t n=text.size();f.write(reinterpret_cast<char*>(&n),8);f<<text;std::vector<uint16_t> values(526336,0x3c00);f.write(reinterpret_cast<char*>(values.data()),values.size()*2);};write();SafetensorsReader source(path);
 SourceTensorMapping matrix{"matrix","blk.0.attn_k.weight",{256,2048},4},norm{"norm","output_norm.weight",{2048},0};
 std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({ggml_tensor_overhead()*2,nullptr,true}),ggml_free);assert(context);
 auto* t=ggml_new_tensor_2d(context.get(),GGML_TYPE_Q4_0,2048,256);ggml_set_name(t,matrix.target.c_str());auto* n=ggml_new_tensor_1d(context.get(),GGML_TYPE_F32,2048);ggml_set_name(n,norm.target.c_str());
 std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors_from_buft(context.get(),ggml_backend_cpu_buffer_type()),ggml_backend_buffer_free);assert(buffer);
 for(auto pair:{std::make_pair(t,matrix),std::make_pair(n,norm)}){upload_tinyllama_tensor(source,pair.second,pair.first);TinyLlamaTargetReader target(source,pair.second,pair.first->type);std::vector<unsigned char> actual(size_t(target.size())),expected(actual.size());ggml_backend_tensor_get(pair.first,actual.data(),0,actual.size());target.read(0,expected.data(),expected.size());assert(actual==expected);++checks;}
 calls=0;upload_tinyllama_tensor(source,matrix,t,upload);assert(calls==2);++checks;
 for(unsigned failed:{1,2}){calls=0;fail_at=failed;reject([&]{upload_tinyllama_tensor(source,matrix,t,upload);});assert(calls==failed);}fail_at=0;
 ggml_set_name(t,"unexpected");reject([&]{upload_tinyllama_tensor(source,matrix,t,upload);});ggml_set_name(t,matrix.target.c_str());
 auto type=t->type;t->type=GGML_TYPE_F32;reject([&]{upload_tinyllama_tensor(source,matrix,t,upload);});t->type=type;
 t->ne[1]++;reject([&]{upload_tinyllama_tensor(source,matrix,t,upload);});t->ne[1]--;
 t->nb[1]++;reject([&]{upload_tinyllama_tensor(source,matrix,t,upload);});t->nb[1]--;
 reject([&]{TinyLlamaTensorCallback callback(source,config);});reject([&]{TinyLlamaTensorCallback::callback(t,nullptr);});
 {std::ofstream truncated(path,std::ios::binary);}calls=0;reject([&]{upload_tinyllama_tensor(source,matrix,t,upload);});assert(calls==0);
 // Strict full inventory backed by a sparse file; only the small norm payload is read.
 const auto strict_text=strict_header.dump();uint64_t payload=0;for(auto it=strict_header.begin();it!=strict_header.end();++it)if(it.key()!="__metadata__")payload=std::max(payload,it.value()["data_offsets"][1].get<uint64_t>());
 const auto norm_offset=strict_header.at("model.norm.weight")["data_offsets"][0].get<uint64_t>();
 const uint64_t virtual_bytes=8+strict_text.size()+payload;
 auto restore=[&](){std::ofstream file(path,std::ios::binary|std::ios::trunc);uint64_t length=strict_text.size();file.write(reinterpret_cast<char*>(&length),8);file<<strict_text;file.seekp(std::streamoff(virtual_bytes-1));file.put(0);file.seekp(std::streamoff(8+length+norm_offset));std::vector<uint16_t> values(2048,0x3f80);file.write(reinterpret_cast<char*>(values.data()),4096);assert(file);};restore();
 SafetensorsReader strict_source(path);TinyLlamaTensorCallback callback(strict_source,config);assert(!callback.complete());++checks;
 ggml_set_name(n,"unexpected");reject([&]{TinyLlamaTensorCallback::callback(n,&callback);});ggml_set_name(n,norm.target.c_str());assert(!callback.complete());
 {std::ofstream truncated(path,std::ios::binary);}reject([&]{TinyLlamaTensorCallback::callback(n,&callback);});assert(!callback.complete());restore();
 TinyLlamaTensorCallback::callback(n,&callback);assert(!callback.complete());std::vector<float> norm_values(2048);ggml_backend_tensor_get(n,norm_values.data(),0,8192);for(float value:norm_values)assert(value==1.f);++checks;
 reject([&]{TinyLlamaTensorCallback::callback(n,&callback);});assert(!callback.complete());
 struct stat info{};assert(stat(path.c_str(),&info)==0);const auto allocated=uint64_t(info.st_blocks)*512;assert(allocated<1024*1024);++checks;
 std::cout<<"sparse virtual_bytes="<<virtual_bytes<<" allocated_bytes="<<allocated<<" successful_callback_source_payload=4096\n";
 std::cout<<"PASS "<<checks<<" descriptor/callback checks; 201 descriptors, bounded CPU uploads, no model loading\n";
}
