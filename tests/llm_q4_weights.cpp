// Extract exact Q4_0 row bytes through genuine locked GGUF/GGML metadata APIs.
#include "ggml.h"
#include "gguf.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
int main(int argc,char**argv){try{
 require(argc==3,"usage: extractor model output-directory");
 ggml_context* metadata=nullptr;gguf_init_params params{true,&metadata};
 auto* file=gguf_init_from_file(argv[1],params);require(file&&metadata,"GGUF metadata failure");
 struct Case{const char* tensor;const char* name;int width,rows,total_rows;};
 const Case cases[]={{"blk.0.attn_k.weight","attn-k",2048,256,256},{"blk.0.ffn_down.weight","ffn-down",5632,17,2048}};
 std::ifstream source(argv[1],std::ios::binary);require(bool(source),"model open");
 for(const auto& c:cases){
  auto* tensor=ggml_get_tensor(metadata,c.tensor);require(tensor&&tensor->type==GGML_TYPE_Q4_0&&tensor->ne[0]==c.width&&tensor->ne[1]==c.total_rows&&tensor->ne[2]==1&&tensor->ne[3]==1,"unexpected tensor layout");
  auto id=gguf_find_tensor(file,c.tensor);require(id>=0,"missing tensor");
  const size_t stride=ggml_row_size(tensor->type,c.width);require(stride==size_t(c.width/32)*18,"Q4 block layout");
  const size_t offset=gguf_get_data_offset(file)+gguf_get_tensor_offset(file,id);
  std::vector<char> bytes(stride*c.rows);source.seekg(offset);source.read(bytes.data(),bytes.size());require(source.good(),"short tensor read");
  auto path=std::filesystem::path(argv[2])/(std::string(c.name)+".weights.q4_0");require(!std::filesystem::exists(path),"output exists");
  std::ofstream output(path,std::ios::binary);output.write(bytes.data(),bytes.size());output.close();require(output.good(),"tensor write failed");
  std::cout<<c.name<<' '<<c.width<<' '<<c.rows<<' '<<bytes.size()<<' '<<offset<<'\n';
 }
 gguf_free(file);ggml_free(metadata);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
