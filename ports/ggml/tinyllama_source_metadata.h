#pragma once
// Exact TinyLlama profile: SentencePiece defines segmentation; HF JSON is cross-checked.
// No generic JSON BPE merges, pair templates, weight loading or source authentication.
#include "gguf.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <string_view>
#include <set>
#include <map>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
namespace direct_tiny {
using Json=nlohmann::json;
inline void need(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
inline std::string read(const std::filesystem::path& p,size_t limit){need(!std::filesystem::is_symlink(p)&&std::filesystem::is_regular_file(p),"source file kind");const auto n=std::filesystem::file_size(p);need(n>0&&n<=limit,"source file bound");std::ifstream f(p,std::ios::binary);std::string data(size_t(n),'\0');f.read(data.data(),std::streamsize(n));need(size_t(f.gcount())==n&&f.peek()==std::char_traits<char>::eof(),"source file changed/read");return data;}
inline Json json(std::string_view data){std::vector<std::set<std::string>> keys;return Json::parse(data,[&](int depth,Json::parse_event_t event,Json& value){need(depth<=32,"JSON nesting bound");if(event==Json::parse_event_t::object_start)keys.emplace_back();else if(event==Json::parse_event_t::key)need(keys.back().insert(value.get<std::string>()).second,"duplicate JSON key");else if(event==Json::parse_event_t::object_end)keys.pop_back();return true;});}
// Only the wire forms used by the fixed SentencePiece ModelProto are accepted.
struct Field{unsigned number,wire;uint64_t integer=0;std::string_view bytes;};
struct Message {
 std::string_view data;size_t at=0;
 uint64_t var(){uint64_t n=0;for(unsigned i=0;i<10;++i){need(at<data.size(),"truncated protobuf varint");const auto c=uint8_t(data[at++]);need(i!=9||c<=1,"protobuf varint overflow");n|=uint64_t(c&127)<<(7*i);if(!(c&128))return n;}throw std::runtime_error("protobuf varint overflow");}
 bool next(Field& f){if(at==data.size())return false;const auto key=var();need(key>>3&&key>>3<=(1u<<29)-1,"protobuf field number");f={unsigned(key>>3),unsigned(key&7),0,{}};if(f.wire==0)f.integer=var();else{size_t n=0;if(f.wire==2){const auto length=var();need(length<=data.size()-at,"protobuf length");n=size_t(length);}else if(f.wire==5)n=4;else if(f.wire==1)n=8;else throw std::runtime_error("protobuf wire type");need(n<=data.size()-at,"truncated protobuf field");f.bytes=data.substr(at,n);at+=n;}return true;}
};
struct Vocabulary{std::vector<std::string> pieces;std::vector<float> scores;std::vector<int32_t> types;};
inline std::map<unsigned,Field> scalar_fields(std::string_view raw){Message m{raw};Field f;std::map<unsigned,Field> fields;while(m.next(f))need(fields.emplace(f.number,f).second,"duplicate protobuf field");return fields;}
inline void integer(const std::map<unsigned,Field>& f,unsigned key,uint64_t expected){const auto i=f.find(key);need(i!=f.end()&&i->second.wire==0&&i->second.integer==expected,"SentencePiece integer profile");}
inline void text(const std::map<unsigned,Field>& f,unsigned key,std::string_view expected,bool optional=false){const auto i=f.find(key);if(optional&&i==f.end())return;need(i!=f.end()&&i->second.wire==2&&i->second.bytes==expected,"SentencePiece text profile");}
inline Vocabulary sentencepiece(std::string_view raw){need(!raw.empty()&&raw.size()<=1024*1024,"SentencePiece size");Message m{raw};Field f;Vocabulary v;std::string_view trainer,normalizer;bool trainer_seen=false,normalizer_seen=false;std::set<std::string> unique;
 while(m.next(f)){
  need(f.wire==2,"SentencePiece top wire");
  if(f.number==1){need(v.pieces.size()<32000,"SentencePiece vocab bound");const auto p=scalar_fields(f.bytes);need(p.size()>=2&&p.size()<=3&&(p.size()==2||p.count(3))&&p.count(1)&&p.count(2)&&(!p.count(3)||p.at(3).wire==0),"SentencePiece piece fields");need(p.at(1).wire==2&&p.at(2).wire==5,"SentencePiece piece wire");const auto piece=p.at(1).bytes;need(!piece.empty()&&piece.size()<=4096&&piece.find('\0')==piece.npos,"SentencePiece piece bytes");const auto type=p.count(3)?p.at(3).integer:1;need(type==1||type==2||type==3||type==6,"SentencePiece unsupported type");uint32_t bits=0;for(unsigned i=0;i<4;++i)bits|=uint32_t(uint8_t(p.at(2).bytes[i]))<<(8*i);float score;std::memcpy(&score,&bits,4);need(std::isfinite(score),"SentencePiece nonfinite score");v.pieces.emplace_back(piece);(void)Json(v.pieces.back()).dump();need(unique.insert(v.pieces.back()).second,"SentencePiece duplicate piece");v.scores.push_back(score);v.types.push_back(int32_t(type));
  }else if(f.number==2){need(!trainer_seen,"duplicate trainer");trainer_seen=true;trainer=f.bytes;}else if(f.number==3){need(!normalizer_seen,"duplicate normalizer");normalizer_seen=true;normalizer=f.bytes;}else throw std::runtime_error("unsupported SentencePiece top field");
 }
 need(v.pieces.size()==32000&&!trainer.empty()&&!normalizer.empty(),"SentencePiece required profile");const auto tr=scalar_fields(trainer),norm=scalar_fields(normalizer);for(const auto& item:norm)need(item.first>=1&&item.first<=6,"unsupported normalizer field");integer(tr,3,2);integer(tr,4,32000);integer(tr,35,1);integer(tr,40,0);integer(tr,41,1);integer(tr,42,2);integer(tr,43,UINT64_MAX);text(norm,1,"identity");text(norm,2,"");integer(norm,3,1);integer(norm,4,0);if(norm.count(5))integer(norm,5,1);text(norm,6,"",true);
 need(v.pieces[0]=="<unk>"&&v.pieces[1]=="<s>"&&v.pieces[2]=="</s>"&&v.types[0]==2&&v.types[1]==3&&v.types[2]==3,"SentencePiece special identity");
 for(unsigned i=0;i<256;++i){char name[10];std::snprintf(name,sizeof(name),"<0x%02X>",i);need(v.pieces[i+3]==name&&v.types[i+3]==6,"SentencePiece byte fallback");}for(unsigned i=259;i<32000;++i)need(v.types[i]==1,"SentencePiece normal inventory");return v;
}
inline void equal(const Json& j,const char* key,const Json& wanted){need(j.contains(key)&&j.at(key)==wanted,"HF profile differs");}
inline void keys(const Json& j,std::initializer_list<const char*> allowed){need(j.is_object(),"HF object required");std::set<std::string> names(allowed.begin(),allowed.end());for(auto it=j.begin();it!=j.end();++it)need(names.count(it.key()),"unsupported HF field");}
using Metadata=std::unique_ptr<gguf_context,decltype(&gguf_free)>;
// Caller supplies one authenticated immutable sidecar set for the entire call.
// Named-file structural checks do not authenticate bytes or parent directories.
inline Metadata construct(const std::filesystem::path& folder){need(std::filesystem::is_directory(folder)&&!std::filesystem::is_symlink(folder),"source directory");auto config=json(read(folder/"config.json",65536)),tc=json(read(folder/"tokenizer_config.json",65536)),tj=json(read(folder/"tokenizer.json",4*1024*1024)),special=json(read(folder/"special_tokens_map.json",65536));auto vocab=sentencepiece(read(folder/"tokenizer.model",1024*1024));
 keys(config,{"architectures","attention_bias","bos_token_id","eos_token_id","hidden_act","hidden_size","initializer_range","intermediate_size","max_position_embeddings","model_type","num_attention_heads","num_hidden_layers","num_key_value_heads","pretraining_tp","rms_norm_eps","rope_scaling","rope_theta","tie_word_embeddings","torch_dtype","transformers_version","use_cache","vocab_size","mlp_bias","head_dim"});
 if(config.contains("mlp_bias")){need(config.at("mlp_bias").is_boolean(),"HF mlp_bias type");equal(config,"mlp_bias",false);}if(config.contains("head_dim")){need(config.at("head_dim").is_number_integer(),"HF head_dim type");equal(config,"head_dim",64);}
 keys(tc,{"added_tokens_decoder","bos_token","chat_template","clean_up_tokenization_spaces","eos_token","legacy","model_max_length","pad_token","padding_side","sp_model_kwargs","tokenizer_class","unk_token","use_default_system_prompt","add_eos_token","add_sep_token","remove_extra_whitespaces","add_bos_token","add_prefix_space","additional_special_tokens"});
 keys(special,{"bos_token","eos_token","unk_token","pad_token","additional_special_tokens"});for(const auto* sidecar:{&tc,&special})if(sidecar->contains("additional_special_tokens"))equal(*sidecar,"additional_special_tokens",Json::array());
 for(auto kv:std::vector<std::pair<const char*,Json>>{{"architectures",Json::array({"LlamaForCausalLM"})},{"model_type","llama"},{"hidden_act","silu"},{"attention_bias",false},{"tie_word_embeddings",false},{"rope_scaling",nullptr},{"torch_dtype","bfloat16"},{"bos_token_id",1},{"eos_token_id",2}})equal(config,kv.first,kv.second);
 for(auto kv:std::vector<std::pair<const char*,unsigned>>{{"num_hidden_layers",22},{"max_position_embeddings",2048},{"hidden_size",2048},{"intermediate_size",5632},{"num_attention_heads",32},{"num_key_value_heads",4},{"vocab_size",32000},{"pretraining_tp",1}}){need(config.at(kv.first).is_number_integer(),"HF integer type");equal(config,kv.first,kv.second);}equal(config,"rms_norm_eps",1e-5);equal(config,"rope_theta",10000.0);
 equal(tc,"tokenizer_class","LlamaTokenizer");equal(tc,"clean_up_tokenization_spaces",false);equal(tc,"legacy",false);equal(tc,"sp_model_kwargs",Json::object());equal(tc,"use_default_system_prompt",false);equal(tc,"padding_side","right");equal(tc,"model_max_length",2048);for(auto kv:std::vector<std::pair<const char*,const char*>>{{"bos_token","<s>"},{"eos_token","</s>"},{"unk_token","<unk>"},{"pad_token","</s>"}}){equal(tc,kv.first,kv.second);const auto& item=special.at(kv.first);keys(item,{"content","lstrip","normalized","rstrip","single_word"});equal(item,"content",kv.second);for(auto flag:{"lstrip","normalized","rstrip","single_word"})equal(item,flag,false);}
 need(tc.at("added_tokens_decoder").is_object()&&tc.at("added_tokens_decoder").size()==3,"HF added token count");for(unsigned id=0;id<3;++id){const auto& a=tc.at("added_tokens_decoder").at(std::to_string(id));equal(a,"content",vocab.pieces[id]);for(auto name:{"lstrip","rstrip","normalized","single_word"})equal(a,name,false);equal(a,"special",true);vocab.scores[id]=-1000.f;vocab.types[id]=3;}
 for(auto name:{"add_eos_token","add_sep_token","remove_extra_whitespaces"})if(tc.contains(name))equal(tc,name,false);for(auto name:{"add_bos_token","add_prefix_space"})if(tc.contains(name))equal(tc,name,true);
 equal(tj.at("model"),"type","BPE");equal(tj.at("model"),"byte_fallback",true);equal(tj.at("model"),"dropout",nullptr);equal(tj.at("model"),"unk_token","<unk>");equal(tj.at("model"),"fuse_unk",true);equal(tj.at("model"),"continuing_subword_prefix",nullptr);equal(tj.at("model"),"end_of_word_suffix",nullptr);need(tj.at("model").at("vocab").size()==32000,"HF tokenizer vocabulary size");for(unsigned i=0;i<32000;++i)equal(tj.at("model").at("vocab"),vocab.pieces[i].c_str(),i);
 equal(tj,"pre_tokenizer",nullptr);equal(tj,"normalizer",json(R"({"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"▁"},{"type":"Replace","pattern":{"String":" "},"content":"▁"}]})"));
 equal(tj,"decoder",json(R"({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]})"));
 equal(tj.at("post_processor"),"type","TemplateProcessing");equal(tj.at("post_processor"),"special_tokens",json(R"({"<s>":{"id":"<s>","ids":[1],"tokens":["<s>"]}})"));need(tj.at("added_tokens").is_array()&&tj.at("added_tokens").size()==3,"HF tokenizer added inventory");for(unsigned id=0;id<3;++id){auto a=tj.at("added_tokens").at(id);equal(a,"id",id);a.erase("id");need(a==tc.at("added_tokens_decoder").at(std::to_string(id)),"HF tokenizer added identity");}equal(tj.at("post_processor"),"single",json(R"([{"SpecialToken":{"id":"<s>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}])"));
 Metadata meta(gguf_init_empty(),gguf_free);need(bool(meta),"metadata allocation");gguf_set_val_str(meta.get(),"general.architecture","llama");
 for(auto kv:std::vector<std::pair<const char*,const char*>>{{"llama.block_count","num_hidden_layers"},{"llama.context_length","max_position_embeddings"},{"llama.embedding_length","hidden_size"},{"llama.feed_forward_length","intermediate_size"},{"llama.attention.head_count","num_attention_heads"},{"llama.attention.head_count_kv","num_key_value_heads"},{"llama.vocab_size","vocab_size"}})gguf_set_val_u32(meta.get(),kv.first,config.at(kv.second).get<uint32_t>());
 for(auto key:{"llama.attention.key_length","llama.attention.value_length","llama.rope.dimension_count"})gguf_set_val_u32(meta.get(),key,64);gguf_set_val_f32(meta.get(),"llama.rope.freq_base",config.at("rope_theta").get<float>());gguf_set_val_f32(meta.get(),"llama.attention.layer_norm_rms_epsilon",config.at("rms_norm_eps").get<float>());
 gguf_set_val_str(meta.get(),"tokenizer.ggml.model","llama");gguf_set_val_str(meta.get(),"tokenizer.ggml.pre","default");std::vector<const char*> pieces;for(auto& s:vocab.pieces)pieces.push_back(s.c_str());gguf_set_arr_str(meta.get(),"tokenizer.ggml.tokens",pieces.data(),pieces.size());gguf_set_arr_data(meta.get(),"tokenizer.ggml.scores",GGUF_TYPE_FLOAT32,vocab.scores.data(),vocab.scores.size());gguf_set_arr_data(meta.get(),"tokenizer.ggml.token_type",GGUF_TYPE_INT32,vocab.types.data(),vocab.types.size());
 for(auto kv:std::vector<std::pair<const char*,unsigned>>{{"tokenizer.ggml.bos_token_id",1},{"tokenizer.ggml.eos_token_id",2},{"tokenizer.ggml.unknown_token_id",0},{"tokenizer.ggml.padding_token_id",2}})gguf_set_val_u32(meta.get(),kv.first,kv.second);
 for(auto key:{"tokenizer.ggml.add_bos_token","tokenizer.ggml.add_space_prefix"})gguf_set_val_bool(meta.get(),key,true);for(auto key:{"tokenizer.ggml.add_eos_token","tokenizer.ggml.add_sep_token","tokenizer.ggml.remove_extra_whitespaces"})gguf_set_val_bool(meta.get(),key,false);
 need(tc.at("chat_template").is_string(),"HF chat template type");need(tc.at("chat_template").get_ref<const std::string&>().find('\0')==std::string::npos,"HF chat template NUL");gguf_set_val_str(meta.get(),"tokenizer.chat_template",tc.at("chat_template").get_ref<const std::string&>().c_str());return meta;
}
}
