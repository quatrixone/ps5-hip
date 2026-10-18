#include "tinyllama_target_reader.h"
#include <cassert>
#include <functional>
#include <iostream>
using namespace ps5hip;
static unsigned checks=0;
void reject(const std::function<void()>& fn){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}assert(caught);++checks;}
int main(int argc,char** argv){
 assert(argc==2);const std::string path=argv[1];
 for(const std::string dtype:{"F16","BF16"}) {
  nlohmann::json header={{"matrix",{{"dtype",dtype},{"shape",{8,64}},{"data_offsets",{0,1024}}}},
                         {"norm",{{"dtype",dtype},{"shape",{64}},{"data_offsets",{1024,1152}}}}};
  std::vector<uint16_t> bits(576);std::vector<float> original(576);
  for(size_t i=0;i<bits.size();++i){float value=float(int(i%131)-65)/17;bits[i]=dtype=="BF16"?ggml_fp32_to_bf16(value).bits:ggml_fp32_to_fp16(value);original[i]=dtype=="BF16"?ggml_bf16_to_fp32(ggml_bf16_t{bits[i]}):ggml_fp16_to_fp32(bits[i]);}
  auto write=[&](){std::ofstream file(path,std::ios::binary);const auto text=header.dump();uint64_t length=text.size();file.write(reinterpret_cast<char*>(&length),8);file<<text;file.write(reinterpret_cast<char*>(bits.data()),bits.size()*2);};write();
  SafetensorsReader source(path);SourceTensorMapping matrix{"matrix","target",{8,64},2},norm{"norm","norm",{64},0};
  for(const auto& map:{matrix,norm}) {
   bool is_norm=map.shape.size()==1;TinyLlamaTargetReader reader(source,map,is_norm?GGML_TYPE_F32:GGML_TYPE_Q4_0);
   std::vector<unsigned char> expected(size_t(reader.size())),full(expected.size());
   if(is_norm)std::memcpy(expected.data(),original.data()+512,expected.size());
   else for(size_t row=0;row<8;++row){size_t local=row%4;size_t src=(row/4)*4+local/2+(local%2)*2;float values[64];for(unsigned i=0;i<64;++i)values[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(original[src*64+i]));assert(ggml_quantize_chunk(GGML_TYPE_Q4_0,values,expected.data()+row*36,0,1,64,nullptr)==36);}
   reader.read(0,full.data(),full.size());assert(full==expected);++checks;
   for(size_t offset=0;offset<=expected.size();++offset)for(size_t count:{size_t(0),size_t(1),size_t(17),size_t(37),size_t(73)})if(count<=expected.size()-offset){std::vector<unsigned char> actual(count);reader.read(offset,actual.data(),count);assert(std::equal(actual.begin(),actual.end(),expected.begin()+offset));++checks;}
   reader.read(reader.size(),nullptr,0);++checks;
   reject([&]{reader.read(UINT64_MAX,nullptr,0);});reject([&]{reader.read(reader.size(),full.data(),1);});reject([&]{reader.read(0,nullptr,1);});reject([&]{reader.read(1,full.data(),SIZE_MAX);});
   assert(reader.scratch_bytes()<=5632*10);++checks;
  }
  reject([&]{TinyLlamaTargetReader bad(source,matrix,GGML_TYPE_F32);});
  auto bad=matrix;bad.shape={UINT64_MAX,64};reject([&]{TinyLlamaTargetReader reader(source,bad,GGML_TYPE_Q4_0);});
  bad=matrix;bad.permutation_heads=3;reject([&]{TinyLlamaTargetReader reader(source,bad,GGML_TYPE_Q4_0);});
  bad=norm;bad.permutation_heads=2;reject([&]{TinyLlamaTargetReader reader(source,bad,GGML_TYPE_F32);});
  bad=matrix;bad.source="absent";reject([&]{TinyLlamaTargetReader reader(source,bad,GGML_TYPE_Q4_0);});
  bits[0]=dtype=="BF16"?0x7f80:0x7c00;write();SafetensorsReader invalid(path);TinyLlamaTargetReader nonfinite(invalid,matrix,GGML_TYPE_Q4_0);unsigned char byte;reject([&]{nonfinite.read(0,&byte,1);});
  if(dtype=="BF16"){bits[0]=0x7f7f;write();SafetensorsReader finite(path);TinyLlamaTargetReader overflow(finite,matrix,GGML_TYPE_Q4_0);reject([&]{overflow.read(0,&byte,1);});}
  header["matrix"]["dtype"]="U16";write();SafetensorsReader wrong_type(path);reject([&]{TinyLlamaTargetReader reader(wrong_type,matrix,GGML_TYPE_Q4_0);});
  header["matrix"]["dtype"]=dtype;header["matrix"]["shape"]={32,16};write();SafetensorsReader unaligned(path);bad=matrix;bad.shape={32,16};reject([&]{TinyLlamaTargetReader reader(unaligned,bad,GGML_TYPE_Q4_0);});
  header["matrix"]["shape"]={8,64};bits[0]=0;write();SafetensorsReader truncated(path);TinyLlamaTargetReader reader(truncated,matrix,GGML_TYPE_Q4_0);{std::ofstream empty(path,std::ios::binary);}reject([&]{reader.read(0,&byte,1);});
 }
 std::cout<<"PASS "<<checks<<" bounded target-reader checks\n";
}
