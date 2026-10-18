#pragma once
#include "safetensors_reader.h"
namespace ps5hip {
struct SourceTensorMapping {
 std::string source,target;std::vector<uint64_t> shape;uint64_t permutation_heads;
};
// Strict locked TinyLlama profile; metadata/tokenizer construction remains separate.
inline std::vector<SourceTensorMapping> tinyllama_tensor_mapping(const nlohmann::json& config,
 const std::map<std::string,SafetensorsReader::Tensor>& inventory) {
 using namespace safetensors_detail;
 const nlohmann::json required={{"architectures",{"LlamaForCausalLM"}},{"model_type","llama"},
  {"hidden_size",2048},{"intermediate_size",5632},{"num_hidden_layers",22},{"num_attention_heads",32},
  {"num_key_value_heads",4},{"vocab_size",32000},{"tie_word_embeddings",false},{"attention_bias",false},
  {"hidden_act","silu"},{"rope_scaling",nullptr},{"rope_theta",10000.0},{"rms_norm_eps",1e-5},{"pretraining_tp",1}};
 for(auto it=required.begin();it!=required.end();++it)check(config.contains(it.key())&&config[it.key()]==it.value(),"unsupported TinyLlama configuration");
 check(!config.value("mlp_bias",false),"unsupported MLP bias");
 check(!config.contains("head_dim")||config["head_dim"]==64,"unsupported head dimension");
 std::vector<SourceTensorMapping> result;
 auto add=[&](std::string source,std::string target,std::vector<uint64_t> shape,uint64_t heads=0){
  result.push_back({std::move(source),std::move(target),std::move(shape),heads});
 };
 add("model.embed_tokens.weight","token_embd.weight",{32000,2048});
 add("model.norm.weight","output_norm.weight",{2048});
 add("lm_head.weight","output.weight",{32000,2048});
 for(unsigned layer=0;layer<22;++layer) {
  auto src="model.layers."+std::to_string(layer)+".";auto dst="blk."+std::to_string(layer)+".";
  add(src+"input_layernorm.weight",dst+"attn_norm.weight",{2048});
  add(src+"post_attention_layernorm.weight",dst+"ffn_norm.weight",{2048});
  add(src+"self_attn.q_proj.weight",dst+"attn_q.weight",{2048,2048},32);
  add(src+"self_attn.k_proj.weight",dst+"attn_k.weight",{256,2048},4);
  add(src+"self_attn.v_proj.weight",dst+"attn_v.weight",{256,2048});
  add(src+"self_attn.o_proj.weight",dst+"attn_output.weight",{2048,2048});
  add(src+"mlp.gate_proj.weight",dst+"ffn_gate.weight",{5632,2048});
  add(src+"mlp.up_proj.weight",dst+"ffn_up.weight",{5632,2048});
  add(src+"mlp.down_proj.weight",dst+"ffn_down.weight",{2048,5632});
 }
 check(inventory.size()==result.size(),"TinyLlama tensor count differs");
 for(const auto& item:result) {
  auto found=inventory.find(item.source);check(found!=inventory.end(),"TinyLlama tensor missing");const auto& tensor=found->second;
  check(tensor.shape==item.shape&&(tensor.dtype=="F16"||tensor.dtype=="BF16"),"TinyLlama tensor shape/dtype differs");
 }
 return result;
}

// Preserves original little-endian F16/BF16 bytes; no conversion or whole-tensor staging.
inline void read_llama_permuted16(SafetensorsReader& reader,const std::string& name,uint64_t heads,
 uint64_t offset,void* destination,size_t bytes) {
 using namespace safetensors_detail;const auto& tensor=reader.inventory().at(name);
 check(tensor.shape.size()==2&&(tensor.dtype=="F16"||tensor.dtype=="BF16"),"permutation requires 16-bit matrix");
 auto rows=tensor.shape[0],columns=tensor.shape[1];
 check(heads>0&&rows>0&&columns>0&&rows%heads==0&&(rows/heads)%2==0,"invalid permutation heads");
 check(columns<=UINT64_MAX/2,"row byte overflow");auto row_bytes=columns*2;
 check(offset<=tensor.end-tensor.begin&&bytes<=tensor.end-tensor.begin-offset&&bytes<=SafetensorsReader::max_chunk,"permutation range differs");
 check(destination||!bytes,"null destination");auto* output=static_cast<unsigned char*>(destination);
 auto head_rows=rows/heads;
 while(bytes) {
  auto row=offset/row_bytes,within=offset%row_bytes,local=row%head_rows;
  auto source_row=(row/head_rows)*head_rows+local/2+(local%2)*(head_rows/2);
  auto count=std::min<uint64_t>(bytes,row_bytes-within);
  reader.read(name,source_row*row_bytes+within,output,size_t(count));
  offset+=count;output+=count;bytes-=size_t(count);
 }
}
}
