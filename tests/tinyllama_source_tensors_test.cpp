#include "tinyllama_source_tensors.h"
#include <cassert>
#include <functional>
#include <iostream>
using Json=nlohmann::json;
using Reader=ps5hip::SafetensorsReader;
static unsigned checks=0;
static void reject(std::function<void()> call){bool failed=false;try{call();}catch(const std::exception&){failed=true;}assert(failed);++checks;}
int main(int argc,char** argv) {
 assert(argc==5);std::ifstream config_file(argv[1]),header_file(argv[2]);
 auto config=ps5hip::safetensors_detail::parse(std::string(std::istreambuf_iterator<char>(config_file),{}));
 auto header=ps5hip::safetensors_detail::parse(std::string(std::istreambuf_iterator<char>(header_file),{}));
 std::map<std::string,Reader::Tensor> inventory;
 for(auto it=header.begin();it!=header.end();++it)if(it.key()!="__metadata__") {
  const auto& value=it.value();inventory.emplace(it.key(),Reader::Tensor{value["dtype"],value["shape"],value["data_offsets"][0],value["data_offsets"][1]});
 }
 auto mapping=ps5hip::tinyllama_tensor_mapping(config,inventory);assert(mapping.size()==201);++checks;
 {Json names=Json::object();for(const auto& item:mapping)names[item.source]=item.target;std::ofstream out(argv[4]);out<<names;}
 std::set<std::string> targets;unsigned q=0,k=0;
 for(const auto& item:mapping){assert(targets.insert(item.target).second);if(item.permutation_heads==32)++q;if(item.permutation_heads==4)++k;}
 assert(q==22&&k==22&&targets.count("blk.21.ffn_down.weight"));++checks;
 auto removed=inventory;removed.erase("lm_head.weight");reject([&]{ps5hip::tinyllama_tensor_mapping(config,removed);});
 auto extra=inventory;extra.emplace("extra",inventory.begin()->second);reject([&]{ps5hip::tinyllama_tensor_mapping(config,extra);});
 auto renamed=inventory;renamed.emplace("extra",renamed.begin()->second);renamed.erase("lm_head.weight");reject([&]{ps5hip::tinyllama_tensor_mapping(config,renamed);});
 auto wrong=inventory;wrong.begin()->second.shape[0]++;reject([&]{ps5hip::tinyllama_tensor_mapping(config,wrong);});
 wrong=inventory;wrong.begin()->second.dtype="F32";reject([&]{ps5hip::tinyllama_tensor_mapping(config,wrong);});
 wrong=inventory;for(auto& item:wrong)item.second.dtype="F16";assert(ps5hip::tinyllama_tensor_mapping(config,wrong).size()==201);++checks;
 for(auto key:{"hidden_size","num_hidden_layers","num_attention_heads","num_key_value_heads","vocab_size"}) {
  auto invalid=config;invalid[key]=1;reject([&]{ps5hip::tinyllama_tensor_mapping(invalid,inventory);});
 }
 {auto invalid=config;invalid["tie_word_embeddings"]=true;reject([&]{ps5hip::tinyllama_tensor_mapping(invalid,inventory);});}
 reject([&]{ps5hip::safetensors_detail::parse("{\"weight\":{},\"weight\":{}}");});
 // Independent reshape(head,2,half,columns).swapaxes(1,2) oracle.
 for(const auto* dtype:{"F16","BF16"})for(uint64_t heads:{32,4}) {
  uint64_t half=3,columns=3,rows=heads*2*half;std::string data(rows*columns*2,'\0'),expected;
  for(size_t i=0;i<data.size();++i)data[i]=char((i*17+13)%256);
  for(uint64_t h=0;h<heads;++h)for(uint64_t a=0;a<half;++a)for(uint64_t b=0;b<2;++b)
   expected.append(data,(h*2*half+b*half+a)*columns*2,columns*2);
  Json document={{"weight",{{"dtype",dtype},{"shape",{rows,columns}},{"data_offsets",{0,data.size()}}}}};auto text=document.dump();
  {std::ofstream file(argv[3],std::ios::binary);uint64_t length=text.size();for(unsigned i=0;i<8;++i){file.put(char(length>>(8*i)));}file<<text;file.write(data.data(),data.size());}
  Reader reader(argv[3]);std::string output(data.size(),'\0');
  // Deliberately split in the middle of 16-bit elements and row boundaries.
  for(size_t offset=0;offset<output.size();offset+=7)ps5hip::read_llama_permuted16(reader,"weight",heads,offset,&output[offset],std::min<size_t>(7,output.size()-offset));
  assert(output==expected);++checks;
  reject([&]{ps5hip::read_llama_permuted16(reader,"weight",0,0,&output[0],1);});
  reject([&]{ps5hip::read_llama_permuted16(reader,"weight",heads,output.size(),&output[0],1);});
  reject([&]{ps5hip::read_llama_permuted16(reader,"weight",heads,0,nullptr,1);});
 }
 std::remove(argv[3]);std::cout<<checks<<" checks PASS; actual header inventory=201; Q heads=32 K heads=4\n";
}
