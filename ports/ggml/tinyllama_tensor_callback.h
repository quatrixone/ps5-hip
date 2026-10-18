#pragma once
#include "tinyllama_target_reader.h"
#include "ggml-backend.h"
#include "gguf.h"
namespace ps5hip {
inline ggml_type tinyllama_target_type(const SourceTensorMapping& map){return map.shape.size()==1?GGML_TYPE_F32:GGML_TYPE_Q4_0;}
// Mapping must come from tinyllama_tensor_mapping(config, source.inventory()).
inline void add_tinyllama_descriptors(gguf_context* metadata,const std::vector<SourceTensorMapping>& mapping) {
 using safetensors_detail::check;check(metadata&&gguf_get_n_tensors(metadata)==0&&mapping.size()==201,"descriptor inventory differs");
 std::set<std::string> names;
 for(const auto& map:mapping)check(!map.target.empty()&&map.target.find('\0')==std::string::npos&&map.target.size()<GGML_MAX_NAME&&names.insert(map.target).second&&(map.shape.size()==1||map.shape.size()==2)&&map.shape.back()>0&&map.shape.back()<=5632&&(map.shape.size()==1||(map.shape[0]>0&&map.shape[0]<=32000&&map.shape.back()%32==0)),"descriptor mapping differs");
 check(names.count("output.weight")==1,"independent output descriptor required");
 std::unique_ptr<ggml_context,decltype(&ggml_free)> context(ggml_init({ggml_tensor_overhead()*mapping.size(),nullptr,true}),ggml_free);
 check(bool(context),"descriptor context allocation");
 for(const auto& map:mapping) {
  int64_t ne[2]={int64_t(map.shape.back()),map.shape.size()==1?1:int64_t(map.shape[0])};
  auto* tensor=ggml_new_tensor(context.get(),tinyllama_target_type(map),int(map.shape.size()),ne);
  check(tensor!=nullptr,"descriptor allocation");ggml_set_name(tensor,map.target.c_str());gguf_add_tensor(metadata,tensor);
 }
}
using TensorUpload=void(*)(ggml_tensor*,const void*,size_t,size_t);
// The uploader must throw on observable failure. ggml's default setter returns void.
inline void upload_tinyllama_tensor(SafetensorsReader& source,const SourceTensorMapping& map,ggml_tensor* tensor,TensorUpload upload=ggml_backend_tensor_set) {
 using safetensors_detail::check;check(tensor&&upload,"null callback tensor/uploader");
 check(map.target==ggml_get_name(tensor)&&tensor->type==tinyllama_target_type(map),"callback name/type differs");
 check(map.shape.size()==1||map.shape.size()==2,"callback rank differs");
 for(unsigned dim=0;dim<GGML_MAX_DIMS;++dim){const uint64_t wanted=dim==0?map.shape.back():dim==1&&map.shape.size()==2?map.shape[0]:1;check(tensor->ne[dim]>0&&uint64_t(tensor->ne[dim])==wanted,"callback shape differs");}
 check(ggml_is_contiguous(tensor),"callback layout differs");
 TinyLlamaTargetReader reader(source,map,tinyllama_target_type(map));check(ggml_nbytes(tensor)==reader.size(),"callback bytes differ");
 constexpr size_t max_transfer=256*1024;
 std::vector<unsigned char> transfer(size_t(std::min<uint64_t>(max_transfer,reader.size())));
 for(uint64_t offset=0;offset<reader.size();) {
  const size_t count=size_t(std::min<uint64_t>(transfer.size(),reader.size()-offset));
  reader.read(offset,transfer.data(),count);upload(tensor,transfer.data(),size_t(offset),count);offset+=count;
 }
}
class TinyLlamaTensorCallback {
public:
 TinyLlamaTensorCallback(SafetensorsReader& source,const nlohmann::json& config):source_(source),mapping_(tinyllama_tensor_mapping(config,source.inventory())){}
 void add_descriptors(gguf_context* metadata) const {add_tinyllama_descriptors(metadata,mapping_);}
 static void callback(ggml_tensor* tensor,void* userdata) {
  using safetensors_detail::check;check(userdata&&tensor,"null callback input");auto& self=*static_cast<TinyLlamaTensorCallback*>(userdata);
  const std::string name=ggml_get_name(tensor);check(!self.completed_.count(name),"duplicate tensor callback");
  auto found=std::find_if(self.mapping_.begin(),self.mapping_.end(),[&](const auto& map){return map.target==name;});
  check(found!=self.mapping_.end(),"unexpected tensor callback");upload_tinyllama_tensor(self.source_,*found,tensor);self.completed_.insert(name);
 }
 bool complete() const {return completed_.size()==mapping_.size();}
private:
 SafetensorsReader& source_;std::vector<SourceTensorMapping> mapping_;std::set<std::string> completed_;
};
}
