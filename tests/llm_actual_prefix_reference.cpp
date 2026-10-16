#include "model_prefix.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>
#include <cmath>
static void write_file(const std::string& path,const void* data,size_t size){
 std::ifstream prior(path);ps5hip_prefix::check(!prior.good(),"oracle output exists");
 std::ofstream file(path,std::ios::binary);file.write(static_cast<const char*>(data),size);file.close();ps5hip_prefix::check(file.good(),"oracle output write");
}
struct State {
    std::array<ggml_tensor*,3> qkv{};
    bool captured=false;
    int32_t expected_token=1;
    const char * output_dir=nullptr;
    static bool callback(ggml_tensor * t,bool ask,void * data) {
        if(ask) return true;
        auto & state=*static_cast<State*>(data);
        const char * names[]={"Qcur-0","Kcur-0","Vcur-0"};
        for(int i=0;i<3;++i) if(t->op==GGML_OP_MUL_MAT && std::strcmp(t->name,names[i])==0) state.qkv[i]=t;
        if(!state.captured && state.qkv[0] && state.qkv[1] && state.qkv[2]) {
            auto * ctx=ggml_init({1024*1024,nullptr,true}); ps5hip_prefix::check(ctx,"prefix context");
            auto * graph=ggml_new_graph_custom(ctx,64,false);
            for(auto * q:state.qkv) ggml_build_forward_expand(graph,q);
            auto selected=ps5hip_prefix::select(graph);
            auto clone=ps5hip_prefix::materialize(ctx,selected);
            int32_t token=-1; ggml_backend_tensor_get(selected.leaves[1],&token,0,4);
            ps5hip_prefix::check(token==state.expected_token,"actual token value");
            size_t bytes=0; for(auto * t:selected.leaves) { ps5hip_prefix::check(t->data && t->buffer,"actual loaded leaf"); bytes+=ggml_nbytes(t); }
            ps5hip_prefix::check(bytes==39821316,"actual selected input bytes");
            std::array<char,1152> row{};ggml_backend_tensor_get(selected.leaves[0],row.data(),size_t(token)*row.size(),row.size());
            write_file(std::string(state.output_dir)+"/token-"+std::to_string(token)+"-embedding.q4_0",row.data(),row.size());
            if(token==0){
                const char* names[]={"norm.f32le","q.weights.q4_0","k.weights.q4_0","v.weights.q4_0"};
                for(size_t i=2;i<selected.leaves.size();++i){
                    std::vector<char> data(ggml_nbytes(selected.leaves[i]));ggml_backend_tensor_get(selected.leaves[i],data.data(),0,data.size());
                    write_file(std::string(state.output_dir)+"/"+names[i-2],data.data(),data.size());
                }
                std::string metadata="{\"classification\":\"actual-selected-model-leaves\",\"tensors\":[";
                for(size_t i=0;i<selected.leaves.size();++i){
                    auto* t=selected.leaves[i];if(i)metadata+=",";
                    metadata+="{\"name\":\""+std::string(t->name)+"\",\"dtype\":\""+ggml_type_name(t->type)+"\",\"bytes\":"+std::to_string(ggml_nbytes(t))+",\"ne\":[";
                    for(int j=0;j<4;++j){if(j)metadata+=",";metadata+=std::to_string(t->ne[j]);}
                    metadata+="],\"nb\":[";for(int j=0;j<4;++j){if(j)metadata+=",";metadata+=std::to_string(t->nb[j]);}metadata+="]}";
                }
                metadata+="],\"embedding_row_bytes\":1152,\"embedding_row_offset_rule\":\"token_id * 1152 within token_embd.weight\"}\n";
                write_file(std::string(state.output_dir)+"/source-tensors.json",metadata.data(),metadata.size());
            }
            auto backend=ggml_backend_cpu_init(); ps5hip_prefix::check(backend,"CPU oracle backend");
            ggml_backend_cpu_set_n_threads(backend,1);
            auto storage=ggml_backend_alloc_ctx_tensors(ctx,backend); ps5hip_prefix::check(storage,"CPU oracle allocation");
            for(size_t i=0;i<selected.leaves.size();++i) {
                std::vector<char> data(ggml_nbytes(selected.leaves[i]));
                ggml_backend_tensor_get(selected.leaves[i],data.data(),0,data.size());
                ggml_backend_tensor_set(clone.leaves[i],data.data(),0,data.size());
            }
            ps5hip_prefix::check(ggml_backend_graph_compute(backend,clone.graph)==GGML_STATUS_SUCCESS,"CPU prefix oracle compute");
            const char * outputs[]={"embedding","normalized","weighted","q","k","v"};
            for(size_t i=0;i<clone.outputs.size();++i) {
                auto * out=clone.outputs[i]; std::vector<float> data(ggml_nelements(out));
                ggml_backend_tensor_get(out,data.data(),0,data.size()*sizeof(float));
                for(float v:data) ps5hip_prefix::check(std::isfinite(v),"nonfinite CPU oracle");
                std::string path=std::string(state.output_dir)+"/token-"+std::to_string(token)+"-"+outputs[i]+".f32le";
                std::ifstream prior(path); ps5hip_prefix::check(!prior.good(),"oracle output exists");
                std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float));file.close();ps5hip_prefix::check(file.good(),"oracle output write");
            }
            ggml_backend_buffer_free(storage);ggml_backend_free(backend);
            std::printf("ACTUAL_PREFIX_PASS token=%d nodes=%d source_bytes=%zu Q=2048 K=256 V=256 GPU_execution=false\n",token,ggml_graph_n_nodes(clone.graph),bytes);
            state.captured=true; ggml_free(ctx);
        }
        return true;
    }
};
int main(int argc,char ** argv) {
    if(argc!=3) return 2;
    llama_backend_init(); auto mp=llama_model_default_params(); mp.n_gpu_layers=0; mp.use_extra_bufts=false;
    mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
    auto * model=llama_model_load_from_file(argv[1],mp); ps5hip_prefix::check(model,"load model");
    State state; state.output_dir=argv[2]; auto cp=llama_context_default_params(); cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;
    cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv=false;cp.op_offload=false;cp.cb_eval=State::callback;cp.cb_eval_user_data=&state;
    auto * context=llama_init_from_model(model,cp);ps5hip_prefix::check(context,"create context");
    const int tokens[]={0,1,2,31999};
    for(int step=0;step<4;++step) {
    state.qkv={};state.captured=false;state.expected_token=tokens[step];
    auto * batch=llama_batch_ext_init(context);ps5hip_prefix::check(batch,"batch");
    const int index=llama_batch_ext_add_token(batch,0,state.expected_token); const llama_pos position=step;
    ps5hip_prefix::check(index>=0 && llama_batch_ext_set_pos(batch,index,&position) && llama_batch_ext_set_output_logits(batch,index,true),"token setup");
    int result=llama_process(context,LLAMA_PROCESS_TYPE_DECODE,batch);
    llama_batch_ext_free(batch);
    ps5hip_prefix::check(result==0 && state.captured,"actual model selection not observed");
    }
    llama_free(context);llama_model_free(model);llama_backend_free();
}
