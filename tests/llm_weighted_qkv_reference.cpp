// Genuine locked GGUF extraction and CPU RMS_NORM -> MUL graph; no GPU execution.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static void write(const std::filesystem::path& p,const std::vector<char>& b){
 require(!std::filesystem::exists(p),"output exists");std::ofstream f(p,std::ios::binary);f.write(b.data(),b.size());f.close();require(f.good(),"output write");
}
int main(int argc,char**argv){try{
 require(argc==4,"usage: fixture model input-f32 output-directory");
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "requires little endian"
#endif
 ggml_context* metadata=nullptr;gguf_init_params params{true,&metadata};auto* file=gguf_init_from_file(argv[1],params);require(file&&metadata,"GGUF metadata");
 auto epsilon=gguf_find_key(file,"llama.attention.layer_norm_rms_epsilon");
 require(epsilon>=0&&gguf_get_kv_type(file,epsilon)==GGUF_TYPE_FLOAT32&&gguf_get_val_f32(file,epsilon)==1e-5f,"model RMS epsilon");
 std::ifstream model(argv[1],std::ios::binary);require(bool(model),"model open");std::vector<char> norm;
 struct Tensor{const char* name;const char* output;ggml_type type;int rows;};
 for(const Tensor c: {Tensor{"blk.0.attn_norm.weight","norm.f32le",GGML_TYPE_F32,1},
      Tensor{"blk.0.attn_q.weight","q.weights.q4_0",GGML_TYPE_Q4_0,2048},
      Tensor{"blk.0.attn_k.weight","k.weights.q4_0",GGML_TYPE_Q4_0,256},
      Tensor{"blk.0.attn_v.weight","v.weights.q4_0",GGML_TYPE_Q4_0,256}}){
  auto* t=ggml_get_tensor(metadata,c.name);require(t&&t->type==c.type&&t->ne[0]==2048&&t->ne[1]==c.rows&&t->ne[2]==1&&t->ne[3]==1,"tensor layout");
  auto id=gguf_find_tensor(file,c.name);require(id>=0,"missing tensor");
  size_t bytes=ggml_nbytes(t),offset=gguf_get_data_offset(file)+gguf_get_tensor_offset(file,id);
  require(bytes==(c.type==GGML_TYPE_F32?8192:size_t(c.rows)*64*18),"tensor byte count");
  std::vector<char> data(bytes);model.seekg(offset);model.read(data.data(),data.size());require(model.good(),"short model tensor");
  write(std::filesystem::path(argv[3])/c.output,data);if(c.type==GGML_TYPE_F32)norm=data;
  std::cout<<c.output<<' '<<bytes<<' '<<offset<<'\n';
 }
 std::ifstream input(argv[2],std::ios::binary|std::ios::ate);require(input&&input.tellg()==8192,"input extent");input.seekg(0);
 std::vector<float> values(2048),weights(2048),result(2048);input.read(reinterpret_cast<char*>(values.data()),8192);require(input.good(),"input read");std::memcpy(weights.data(),norm.data(),8192);
 for(size_t i=0;i<2048;++i)require(std::isfinite(values[i])&&std::isfinite(weights[i]),"nonfinite tensor");
 auto backend=ggml_backend_cpu_init();require(backend,"CPU backend");ggml_backend_cpu_set_n_threads(backend,1);
 ggml_init_params cfg{2*1024*1024,nullptr,true};auto* ctx=ggml_init(cfg);require(ctx,"graph context");
 auto* x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2048);auto* w=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2048);ggml_set_input(x);ggml_set_input(w);
 auto* normalized=ggml_rms_norm(ctx,x,1e-5f);auto* y=ggml_mul(ctx,normalized,w);require(normalized->op==GGML_OP_RMS_NORM&&y->op==GGML_OP_MUL,"operator graph");ggml_set_output(y);
 auto* graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);auto storage=ggml_backend_alloc_ctx_tensors(ctx,backend);require(storage,"CPU tensor allocation");
 ggml_backend_tensor_set(x,values.data(),0,8192);ggml_backend_tensor_set(w,weights.data(),0,8192);
 require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"CPU graph compute");ggml_backend_tensor_get(y,result.data(),0,8192);
 for(float v:result)require(std::isfinite(v),"nonfinite CPU result");std::vector<char> bytes(8192);std::memcpy(bytes.data(),result.data(),8192);write(std::filesystem::path(argv[3])/"weighted.cpu.f32le",bytes);
 ggml_backend_buffer_free(storage);ggml_free(ctx);ggml_backend_free(backend);gguf_free(file);ggml_free(metadata);
 std::cout<<"WEIGHTED_QKV_CPU_PASS width=2048 q=2048 k=256 v=256 threads=1 gpu_execution=false\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
