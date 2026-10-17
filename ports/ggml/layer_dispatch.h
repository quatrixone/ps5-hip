#pragma once
#include "model_layer.h"
#include "prefix_dispatch.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace ps5hip_layer {
// Only the selected one-token layer is admitted. Control reads never substitute
// host arithmetic for activations; all tensor operations stay on the GPU.
struct Dispatch {
    ps5hip_prefix::Dispatch base;
    hipModule_t module=nullptr;
    std::array<hipFunction_t,12> functions{};
    void* half=nullptr;
    void* full_scratch=nullptr; // Owned and span-validated by the full-model dispatcher.
    unsigned expected_position=0;
    static constexpr size_t half_bytes=32*512*2;
    static constexpr const char* symbols[]={"layer_add","layer_mul","layer_silu","layer_rope",
        "layer_set_rows","layer_scores","layer_softmax","layer_values","layer_copy","layer_to_f16","layer_quantize_q8_cpu","layer_q4_cpu"};
    using Call=ps5hip_prefix::Call;
    Dispatch(ps5hip_prefix::Dispatch::Span owned,ps5hip_prefix::Dispatch::Fail fail):base(owned,fail,5632){}
    void initialize(const std::array<const char*,6>& paths) {
        ps5hip_prefix::check(paths[5]&&paths[5][0]&&!module&&!half,"layer initialization");
        base.initialize({paths[0],paths[1],paths[2],paths[3],paths[4]});
        base.call(Call::module_load,0,"layer_module",[&]{return hipModuleLoad(&module,paths[5]);});
        base.ok(module?hipSuccess:hipErrorInvalidValue,"null_layer_module");
        for(size_t i=0;i<functions.size();++i) {
            base.call(Call::module_function,0,"layer_function",[&]{return hipModuleGetFunction(&functions[i],module,symbols[i]);});
            base.ok(functions[i]?hipSuccess:hipErrorInvalidValue,"null_layer_function");
        }
        base.call(Call::scratch_allocate,half_bytes,"layer_half_allocate",[&]{return hipMalloc(&half,half_bytes);});
        base.ok(half?hipSuccess:hipErrorInvalidValue,"null_layer_half");
    }
    void shutdown() {
        base.shutdown();
        base.call(Call::scratch_free,half_bytes,"layer_half_free",[&]{return hipFree(half);});half=nullptr;
        base.call(Call::module_unload,0,"layer_unload",[&]{return hipModuleUnload(module);});module=nullptr;functions={};
    }
    void reset(){ps5hip_prefix::check(base.ready&&!base.terminal,"layer reset state");expected_position=0;}
    void launch(unsigned f,unsigned count,void** args) {
        ps5hip_prefix::check(count&&f<functions.size(),"layer launch bounds");
        base.call(Call::launch,0,"layer_launch",[&]{return hipModuleLaunchKernel(functions[f],(count+63)/64,1,1,64,1,1,0,nullptr,args,nullptr);});
        base.call(Call::completion_sync,0,"layer_sync",[]{return hipDeviceSynchronize();});++base.completed_launches;
    }
    static const ggml_tensor* input(ggml_cgraph* g,const char* name) {
        const ggml_tensor* found=nullptr;
        for(int i=0;i<ggml_graph_n_nodes(g);++i)for(auto* s:ggml_graph_node(g,i)->src)
            if(s&&s->op==GGML_OP_NONE&&std::strcmp(s->name,name)==0) {
                ps5hip_prefix::check(!found||found==s,"ambiguous layer input");found=s;
            }
        ps5hip_prefix::check(found,"missing layer input");return found;
    }
    void read(const ggml_tensor* t,void* out,size_t size) {
        ps5hip_prefix::check(t&&base.owned(t)&&ggml_nbytes(t)==size,"layer control span");
        base.call(Call::token_read,size,"layer_control",[&]{return hipMemcpy(out,t->data,size,hipMemcpyDeviceToHost);});
    }
    bool preflight(ggml_cgraph* g) {
        try {
            if(ggml_graph_n_nodes(g)!=37)return false;
            (void)select(g); // Exact operation, layout, source and parameter inventory.
            if(expected_position>=512)return false;
            std::vector<const ggml_tensor*> owners;
            auto check_tensor=[&](const ggml_tensor* t) {
                ps5hip_prefix::check(t&&base.owned(t),"layer unowned tensor");
                const ggml_tensor* root=t;
                for(unsigned depth=0;root->view_src;++depth) {
                    ps5hip_prefix::check(depth<16&&base.owned(root->view_src),"layer view chain");
                    ps5hip_prefix::check(static_cast<const char*>(root->data)==static_cast<const char*>(root->view_src->data)+root->view_offs,"layer view address");
                    root=root->view_src;
                }
                if(std::find(owners.begin(),owners.end(),root)==owners.end())owners.push_back(root);
            };
            for(int i=0;i<ggml_graph_n_nodes(g);++i) {
                auto* t=ggml_graph_node(g,i);check_tensor(t);
                for(auto* s:t->src)if(s)check_tensor(s);
            }
            for(size_t i=0;i<owners.size();++i)for(size_t j=0;j<i;++j)
                if(!ps5hip_prefix::Dispatch::disjoint(owners[i],owners[j]))return false;
            int32_t token=-1,position=-1;int64_t k=-1;std::array<int64_t,256> v{};
            read(input(g,"inp_tokens"),&token,sizeof(token));read(input(g,"inp_pos"),&position,sizeof(position));
            read(input(g,"attn_inp_k_idxs"),&k,sizeof(k));read(input(g,"attn_inp_v_idxs"),v.data(),sizeof(v));
            if(token<0||token>=32000||position!=int32_t(expected_position)||k!=position)return false;
            for(size_t i=0;i<v.size();++i)if(v[i]!=int64_t(i*512+expected_position))return false;
            const unsigned extent=expected_position<256?256:512;
            std::array<float,512> mask{};read(input(g,"attn_inp_kq_mask"),mask.data(),extent*sizeof(float));
            for(unsigned i=0;i<extent;++i)if(i<=expected_position?mask[i]!=0:!(std::isinf(mask[i])&&mask[i]<0))return false;
            return true;
        } catch(const std::runtime_error&) {return false;}
    }
    static bool exact_or_disjoint(const ggml_tensor* a,const ggml_tensor* b) {
        return a->data==b->data||ps5hip_prefix::Dispatch::disjoint(a,b);
    }
    bool vector(const ggml_tensor* t) const {
        return t&&t->type==GGML_TYPE_F32&&t->ne[0]==2048&&t->ne[1]==1&&t->ne[2]==1&&t->ne[3]==1&&
               ggml_is_contiguous(t)&&base.owned(t);
    }
    // Caller must admit the complete graph and live aliases/controls before any
    // node executes. This primitive does not advance the request position.
    bool execute_node(ggml_tensor* t,bool direct_inputs) {
        ps5hip_prefix::check(base.ready&&!base.terminal&&half&&module,"layer not ready");
        if(!t)return false;
            switch(t->op) {
            case GGML_OP_RESHAPE:case GGML_OP_VIEW:case GGML_OP_PERMUTE:break;
            case GGML_OP_GET_ROWS: {
                if(t->src[0]&&t->src[0]->type==GGML_TYPE_F32) {
                    auto* ids=t->src[1];
                    if(!direct_inputs||!vector(t)||!vector(t->src[0])||!ids||ids->type!=GGML_TYPE_I32||
                       ggml_nelements(ids)!=1||!ggml_is_contiguous(ids)||!base.owned(ids)||
                       !exact_or_disjoint(t,t->src[0]))return false;
                    int32_t index=-1;read(ids,&index,sizeof(index));if(index!=0)return false;
                    void* x=t->src[0]->data;void* out=t->data;unsigned n=2048;
                    void* args[]={&x,&out,&n};launch(8,n,args);break;
                }
                return base.execute(t);
            }
            case GGML_OP_MUL: {
                if(!direct_inputs)return base.execute(t);
                if(!vector(t)||!vector(t->src[0])||!vector(t->src[1])||
                   !exact_or_disjoint(t,t->src[0])||
                   !exact_or_disjoint(t,t->src[1]))return false;
                void* x=t->src[0]->data;void* y=t->src[1]->data;void* out=t->data;unsigned n=2048;
                void* args[]={&x,&y,&out,&n};launch(1,n,args);break;
            }
            case GGML_OP_RMS_NORM: {
                if(!direct_inputs)return base.execute(t);
                if(!vector(t)||!vector(t->src[0])||!exact_or_disjoint(t,t->src[0]))return false;
                for(int i=1;i<GGML_MAX_SRC;++i)if(t->src[i])return false;
                float epsilon;std::memcpy(&epsilon,t->op_params,4);if(epsilon!=1e-5f)return false;
                // The qualified 1024-thread kernel reduces all input squares
                // across a block barrier before each thread writes its own columns.
                void* input=t->src[0]->data;void* out=t->data;void* absent=nullptr;
                int ncols=2048,channels=1,samples=1;int64_t stride=2048,zero_stride=0;float scale=1.f;
                ps5hip_prefix::Dispatch::Uint3 columns{1,11,2048},one{1,0,1},zero{};
                void* args[]={&input,&out,&ncols,&channels,&samples,&stride,&stride,&stride,&epsilon,
                    &absent,&stride,&stride,&stride,&columns,&one,&one,&one,
                    &absent,&zero_stride,&zero_stride,&zero_stride,&zero,&zero,&zero,&zero,&scale};
                base.launch(1,1,1,1024,1,128,args);break;
            }
            case GGML_OP_MUL_MAT: {
                if(t->src[0]->type==GGML_TYPE_Q4_0) {
                    void* x=t->src[1]->data;void* weight=t->src[0]->data;void* out=t->data;
                    unsigned width=unsigned(t->src[0]->ne[0]),rows=unsigned(t->src[0]->ne[1]);
                    void* quantize[]={&x,&base.q8,&width};launch(10,width/32,quantize);
                    void* project[]={&weight,&base.q8,&out,&width,&rows};launch(11,rows,project);break;
                }
                const bool score=t->src[0]->ne[0]==64;
                unsigned n=unsigned(score?t->ne[0]:t->src[0]->ne[0]);
                unsigned count=score?2048:32*n;
                void* x=t->src[1]->data;void* convert[]={&x,&half,&count};launch(9,count,convert);
                void* cache=t->src[0]->data;void* out=t->data;void* args[]={&cache,&half,&out,&n};
                launch(score?5:7,score?32*n:2048,args);break;
            }
            case GGML_OP_ROPE: {
                void* x=t->src[0]->data;void* out=t->data;unsigned heads=unsigned(t->ne[1]),position=expected_position;
                void* args[]={&x,&out,&heads,&position};launch(3,heads*32,args);break;
            }
            case GGML_OP_SET_ROWS: {
                void* x=t->src[0]->data;void* ids=t->src[1]->data;void* out=t->data;
                unsigned width=unsigned(t->src[0]->ne[0]),rows=unsigned(t->src[0]->ne[1]);
                void* args[]={&x,&ids,&out,&width,&rows};launch(4,width*rows,args);break;
            }
            case GGML_OP_SOFT_MAX: {
                void* x=t->src[0]->data;void* mask=t->src[1]->data;void* out=t->data;unsigned n=unsigned(t->ne[0]);
                const bool aliased=direct_inputs&&x==out;
                if(aliased&&!full_scratch)return false;
                if(aliased)out=full_scratch;
                void* args[]={&x,&mask,&out,&n};launch(6,32,args);
                // Softmax rereads the entire row for each output; exact input alias is unsafe.
                if(aliased){void* destination=t->data;unsigned count=n*32;
                    void* copy[]={&out,&destination,&count};launch(8,count,copy);}
                break;
            }
            case GGML_OP_CONT: {
                void* x=t->src[0]->data;void* out=t->data;unsigned n=2048;
                void* args[]={&x,&out,&n};launch(8,n,args);break;
            }
            case GGML_OP_ADD: {
                void* x=t->src[0]->data;void* y=t->src[1]->data;void* out=t->data;unsigned n=2048;
                void* args[]={&x,&y,&out,&n};launch(0,n,args);break;
            }
            case GGML_OP_GLU: {
                void* x=t->src[0]->data;void* y=t->src[1]->data;void* out=t->data;unsigned n=5632;
                void* silu[]={&x,&out,&n};launch(2,n,silu);
                void* mul[]={&out,&y,&out,&n};launch(1,n,mul);break;
            }
            default:return false;
            }
        return true;
    }
    bool execute_graph(ggml_cgraph* g) {
        ps5hip_prefix::check(base.ready&&!base.terminal&&half&&module,"layer not ready");
        if(!g||!preflight(g))return false;
        for(int i=0;i<ggml_graph_n_nodes(g);++i)
            ps5hip_prefix::check(execute_node(ggml_graph_node(g,i),false),"layer node dispatch");
        ++expected_position;return true;
    }
};
}
