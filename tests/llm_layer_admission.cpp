#include "ps5hip/code_object.h"
#include <fstream>
#include <iterator>
#include <iostream>
#include <map>
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::ifstream f(argv[1],std::ios::binary);
 std::vector<unsigned char>b((std::istreambuf_iterator<char>(f)),{});
 auto o=ps5hip::inspect_code_object(b.data(),b.size());
 const std::map<std::string,std::vector<unsigned>> expected={
  {"layer_rms_cpu",{8,8}},{"layer_quantize_q8_cpu",{8,8,4}},{"layer_q4_cpu",{8,8,8,4,4}},{"layer_copy",{8,8,4}},{"layer_to_f16",{8,8,4}},{"layer_silu",{8,8,4}},
  {"layer_add",{8,8,8,4}},{"layer_mul",{8,8,8,4}},
  {"layer_rope",{8,8,4,4}},{"layer_set_rows",{8,8,8,4,4}},
  {"layer_scores",{8,8,8,4}},{"layer_softmax",{8,8,8,4}},{"layer_values",{8,8,8,4}}};
 if(o.kernels.size()!=expected.size())return 3;
 for(auto&k:o.kernels) {
  auto r=ps5hip::initial_launch_rejection(k);auto it=expected.find(k.name);
  if(it==expected.end()||it->second.size()!=k.arguments.size()||!k.hidden_arguments.empty())return 4;
  unsigned offset=0;
  for(size_t i=0;i<k.arguments.size();++i){const auto&a=k.arguments[i];unsigned size=it->second[i];if(a.offset!=offset||a.size!=size||a.kind!=(size==8?"global_buffer":"by_value"))return 5;offset+=size;}
  if(k.kernarg_size!=offset||k.rsrc2!=0x8c||k.private_bytes||k.group_bytes||k.sgpr_spills||k.vgpr_spills)return 6;
  std::cout<<k.name<<" rsrc1="<<std::hex<<k.rsrc1<<" rsrc2="<<k.rsrc2<<std::dec<<" args="<<k.kernarg_size<<" hidden="<<k.hidden_arguments.size()<<" rejection="<<r<<"\n";
  if(!r.empty())return 1;
 }
 return 0;
}
