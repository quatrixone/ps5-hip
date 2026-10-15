#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
static void write(const std::filesystem::path& path,const std::vector<float>& values){
 std::ofstream stream(path,std::ios::binary);stream.write(reinterpret_cast<const char*>(values.data()),values.size()*4);require(stream.good(),"write failed");
}
static std::vector<float> embeddings(const char* model,const std::vector<int>& tokens){
 ggml_context* metadata=nullptr;gguf_init_params params{true,&metadata};
 gguf_context* file=gguf_init_from_file(model,params);require(file&&metadata,"GGUF metadata load failed");
 auto* tensor=ggml_get_tensor(metadata,"token_embd.weight");require(tensor&&tensor->type==GGML_TYPE_Q4_0&&tensor->ne[0]==2048&&tensor->ne[1]==32000,"unexpected embedding layout");
 auto id=gguf_find_tensor(file,"token_embd.weight");require(id>=0,"missing embedding tensor");
 const size_t stride=ggml_row_size(tensor->type,2048),offset=gguf_get_data_offset(file)+gguf_get_tensor_offset(file,id);
 const auto* traits=ggml_get_type_traits(tensor->type);require(traits&&traits->to_float,"missing genuine dequantizer");
 std::ifstream stream(model,std::ios::binary);std::vector<unsigned char> bytes(stride);std::vector<float> values(tokens.size()*2048);
 for(size_t i=0;i<tokens.size();++i){require(tokens[i]>=0&&tokens[i]<32000,"invalid token");stream.seekg(offset+stride*tokens[i]);stream.read(reinterpret_cast<char*>(bytes.data()),stride);require(stream.good(),"embedding read failed");traits->to_float(bytes.data(),values.data()+i*2048,2048);}
 gguf_free(file);ggml_free(metadata);return values;
}
static std::vector<float> stress(){
 std::vector<float> values(17*2048);uint32_t random=0x12345678;
 for(int row=0;row<17;++row)for(int col=0;col<2048;++col){float x=0;
  switch(row){
   case 0:x=0;break;case 1:x=1;break;case 2:x=-1;break;
   case 3:x=col==2047?10:0;break;case 4:x=col==0?-3:0;break;
   case 5:x=col%2?-1:1;break;case 6:x=col%2?-1e-20f:1e-20f;break;
   case 7:x=float(col%17-8)/8;break;case 8:x=(col%2?-1:1)*(col%3==0?100.0f:0.001f);break;
   default:random=1664525u*random+1013904223u;x=(float(int(random>>8)-8388608)/8388608)*float(row-7);break;
  }values[row*2048+col]=x;
 }
 return values;
}
static void evaluate(ggml_backend_t backend,const std::vector<float>& source,int rows,const std::filesystem::path& prefix){
 ggml_init_params params{2*1024*1024,nullptr,true};auto* ctx=ggml_init(params);require(ctx,"context allocation failed");
 auto* input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,2048,rows);ggml_set_input(input);
 auto* output=ggml_rms_norm(ctx,input,1e-5f);require(output->op==GGML_OP_RMS_NORM,"wrong operator");ggml_set_output(output);
 auto* graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,output);
 auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"backend tensor allocation failed");
 std::vector<float> values(source.begin(),source.begin()+rows*2048),result(rows*2048);
 ggml_backend_tensor_set(input,values.data(),0,values.size()*4);
 require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"CPU graph computation failed");
 ggml_backend_tensor_get(output,result.data(),0,result.size()*4);
 for(float x:result)require(std::isfinite(x),"nonfinite CPU result");
 write(prefix.string()+".input.f32le",values);write(prefix.string()+".cpu.f32le",result);
 ggml_backend_buffer_free(buffer);ggml_free(ctx);
}
int main(int argc,char**argv){try{
 require(argc==4,"usage: rmsnorm model.gguf tokens.txt output-directory");require(sizeof(float)==4,"F32 required");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "fixture files require little endian"
#endif
 std::ifstream input(argv[2]);std::vector<int> tokens;for(int token;input>>token;)tokens.push_back(token);require(tokens.size()==17,"exactly17 tokens required");
 auto embed=embeddings(argv[1],tokens),cases=stress();auto backend=ggml_backend_cpu_init();require(backend,"CPU backend init failed");ggml_backend_cpu_set_n_threads(backend,1);
 for(int rows:{1,2,17}){evaluate(backend,cases,rows,std::filesystem::path(argv[3])/("stress-"+std::to_string(rows)));evaluate(backend,embed,rows,std::filesystem::path(argv[3])/("embedding-"+std::to_string(rows)));}
 ggml_backend_free(backend);std::cout<<"GGML_RMS_NORM_CPU_PASS cases=6 width=2048 threads=1 gpu_execution=false\n";
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
