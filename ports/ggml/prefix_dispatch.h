#pragma once
#include "model_prefix.h"
#include "prefix_calls.h"
#include <hip/hip_runtime_api.h>
#include <cstdint>
#include <cstdlib>

namespace ps5hip_prefix {
// The backend supplies its ownership/span validator and terminal failure policy.
// A HIP failure quarantines this instance; no potentially live resource is freed.
struct Dispatch {
    using Span = bool (*)(const ggml_tensor *);
    using Fail = void (*)(const char *, hipError_t);
    struct Uint3 { uint32_t x,y,z; };
    struct alignas(8) Fusion { uint8_t bytes[80]; };
    static_assert(sizeof(Uint3)==12 && sizeof(Fusion)==80, "kernel argument ABI");
    static constexpr const char * symbols[5] = {
        "_ZL10k_get_rowsILi32ELi2EXadL_ZL15dequantize_q4_0PKvliR15HIP_vector_typeIfLj2EEEEfEvS1_PKiPT2_llS2_IjLj3EEmmmmmmmmm",
        "_ZL12rms_norm_f32ILi1024ELb0ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f",
        "_ZL12rms_norm_f32ILi1024ELb1ELb0ELb0EEvPKfPfiiilllfS1_lll15HIP_vector_typeIjLj3EES4_S4_S4_S1_lllS4_S4_S4_S4_f",
        "_ZL13quantize_q8_1PKfPvlllllj15HIP_vector_typeIjLj3EE",
        "_ZL13mul_mat_vec_qIL9ggml_type2ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj15HIP_vector_typeIjLj3EEjjjS8_jjjS8_jjjj"
    };
    std::array<hipModule_t,5> modules{};
    std::array<hipFunction_t,5> functions{};
    void * q8 = nullptr;
    const unsigned max_width;
    const size_t scratch_bytes;
    Span owned;
    Fail fail;
    bool ready=false, terminal=false;
    unsigned completed_launches=0;
    CallStatistics calls{};
    Dispatch(Span span, Fail failure, unsigned maximum_width=2048):max_width(maximum_width),scratch_bytes(size_t(maximum_width/32)*36),owned(span),fail(failure) {
        check(owned && fail,"dispatch callbacks");
        check(max_width==2048 || max_width==5632,"unsupported dispatch width");
    }
    Dispatch(const Dispatch&)=delete;
    Dispatch& operator=(const Dispatch&)=delete;
    void ok(hipError_t e, const char * site) {
        if(e==hipSuccess) return;
        terminal=true; fail(site,e); std::abort();
    }
    template<class F> void call(Call site,size_t bytes,const char * name,F operation) {
        auto& row=calls[static_cast<unsigned>(site)];
        check(row.attempts<UINT64_MAX && bytes<=UINT64_MAX-row.requested_bytes,"compute call counter overflow");
        ++row.attempts;row.requested_bytes+=bytes;
        hipError_t result;
        try{result=operation();}catch(...){terminal=true;throw;}
        ++row.completions;row.last_result=int(result);
        if(result==hipSuccess)++row.successes;
        else {++row.failures;if(!row.first_error)row.first_error=int(result);}
        ok(result,name);
    }
    void initialize(const std::array<const char*,5>& paths) {
        check(!ready && !terminal && !q8 && !modules[0],"dispatch initialized twice");
        for(auto path:paths) check(path && path[0],"missing module path");
        for(size_t i=0;i<modules.size();++i) {
            call(Call::module_load,0,"module_load",[&]{return hipModuleLoad(&modules[i],paths[i]);}); ok(modules[i]?hipSuccess:hipErrorInvalidValue,"null_module");
            call(Call::module_function,0,"module_function",[&]{return hipModuleGetFunction(&functions[i],modules[i],symbols[i]);}); ok(functions[i]?hipSuccess:hipErrorInvalidValue,"null_function");
        }
        call(Call::scratch_allocate,scratch_bytes,"scratch_allocate",[&]{return hipMalloc(&q8,scratch_bytes);}); ok(q8?hipSuccess:hipErrorInvalidValue,"null_scratch"); ready=true;
    }
    void shutdown() {
        check(ready && !terminal,"dispatch cannot release uncertain work");
        call(Call::release_sync,0,"shutdown_sync",[]{return hipDeviceSynchronize();});
        call(Call::scratch_free,scratch_bytes,"scratch_free",[&]{return hipFree(q8);}); q8=nullptr;
        for(size_t i=modules.size();i-- >0;) { call(Call::module_unload,0,"module_unload",[&]{return hipModuleUnload(modules[i]);}); modules[i]=nullptr; functions[i]=nullptr; }
        ready=false;
    }
    bool valid(const ggml_tensor * t,ggml_type type,int width,int rows) const {
        return shape(t,type,width,rows) && t->data && owned(t);
    }
    static bool disjoint(const ggml_tensor * a,const ggml_tensor * b) {
        if(!a || !b || !a->data || !b->data) return false;
        const auto x=reinterpret_cast<uintptr_t>(a->data),y=reinterpret_cast<uintptr_t>(b->data);
        return x<=y ? ggml_nbytes(a)<=y-x : ggml_nbytes(b)<=x-y;
    }
    bool admit(const ggml_tensor * t) const {
        if(!t || t->view_src) return false;
        const int count=t->op==GGML_OP_RMS_NORM?1:2;
        for(int i=0;i<GGML_MAX_SRC;++i) {
            if((t->src[i]!=nullptr)!=(i<count)) return false;
            if(i<count && !disjoint(t,t->src[i])) return false;
        }
        if(t->op==GGML_OP_GET_ROWS) return valid(t,GGML_TYPE_F32,2048,1) && valid(t->src[0],GGML_TYPE_Q4_0,2048,32000) && valid(t->src[1],GGML_TYPE_I32,1,1);
        if(t->op==GGML_OP_MUL_MAT) {
            const auto width=t->src[0]->ne[0], rows=t->ne[0];
            const bool prefix=width==2048 && (rows==2048 || rows==256);
            const bool ffn=max_width==5632 && ((width==2048 && rows==5632) || (width==5632 && rows==2048));
            return (prefix || ffn) && valid(t,GGML_TYPE_F32,rows,1) && valid(t->src[0],GGML_TYPE_Q4_0,width,rows) && valid(t->src[1],GGML_TYPE_F32,width,1) && t->op_params[0]==GGML_PREC_DEFAULT;
        }
        if(t->op!=GGML_OP_RMS_NORM && t->op!=GGML_OP_MUL) return false;
        const bool weighted=t->op==GGML_OP_MUL;
        auto * norm=weighted?t->src[0]:t;
        if(!valid(t,GGML_TYPE_F32,2048,1) || !norm || norm->op!=GGML_OP_RMS_NORM || !valid(norm,GGML_TYPE_F32,2048,1) || !valid(norm->src[0],GGML_TYPE_F32,2048,1) || !disjoint(t,norm->src[0])) return false;
        for(int i=1;i<GGML_MAX_SRC;++i) if(norm->src[i]) return false;
        if(weighted && !valid(t->src[1],GGML_TYPE_F32,2048,1)) return false;
        float epsilon;std::memcpy(&epsilon,norm->op_params,4);return epsilon==1e-5f;
    }
    bool execute_graph(ggml_cgraph * graph) {
        check(ready && !terminal,"dispatch not ready");
        if(!graph) return false;
        // Reject the entire candidate before the first GPU launch.
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * t=ggml_graph_node(graph,i);if(!admit(t)) return false;
            for(int j=0;j<i;++j) if(ggml_graph_node(graph,j)==t) return false;
            for(auto * src:t->src) if(src && src->op!=GGML_OP_NONE) {
                bool earlier=false;
                for(int j=0;j<i;++j) if(ggml_graph_node(graph,j)==src) earlier=true;
                if(!earlier) return false;
            }
            if(t->op==GGML_OP_GET_ROWS) {
                int32_t token=-1;call(Call::token_read,4,"token_preflight",[&]{return hipMemcpy(&token,t->src[1]->data,4,hipMemcpyDeviceToHost);});
                if(token<0 || token>=32000) return false;
            }
            for(int j=0;j<ggml_graph_n_nodes(graph);++j) {
                auto * other=ggml_graph_node(graph,j);
                if(t!=other && !disjoint(t,other)) return false;
                for(auto * src:other->src) if(src && src!=t && !disjoint(t,src)) return false;
            }
        }
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) if(!execute(ggml_graph_node(graph,i))) return false;
        return true;
    }
    void launch(unsigned module,unsigned gx,unsigned gy,unsigned bx,unsigned by,unsigned lds,void ** args) {
        call(Call::launch,0,"launch",[&]{return hipModuleLaunchKernel(functions[module],gx,gy,1,bx,by,1,lds,nullptr,args,nullptr);});
        call(Call::completion_sync,0,"completion_sync",[]{return hipDeviceSynchronize();}); ++completed_launches;
    }
    bool execute(ggml_tensor * t) {
        check(ready && !terminal,"dispatch not ready");
        if(!admit(t)) return false;
        Uint3 one{1,0,1};
        if(t->op==GGML_OP_GET_ROWS) {
            if(!valid(t,GGML_TYPE_F32,2048,1) || !valid(t->src[0],GGML_TYPE_Q4_0,2048,32000) || !valid(t->src[1],GGML_TYPE_I32,1,1)) return false;
            for(int i=2;i<GGML_MAX_SRC;++i) if(t->src[i]) return false;
            int32_t token=-1; call(Call::token_read,4,"token_validate",[&]{return hipMemcpy(&token,t->src[1]->data,sizeof(token),hipMemcpyDeviceToHost);});
            if(token<0 || token>=32000) return false;
            void * table=t->src[0]->data, * ids=t->src[1]->data, * out=t->data;
            int64_t width=2048,ne11=1; size_t stride=2048,rowbytes=1152,tablebytes=36864000,index_stride=1;
            void * args[]={&table,&ids,&out,&width,&ne11,&one,&stride,&stride,&stride,&rowbytes,&tablebytes,&tablebytes,&index_stride,&index_stride,&index_stride};
            static_assert(sizeof(args)/sizeof(args[0])==15,"GET_ROWS arguments");
            launch(0,1,4,256,1,0,args); return true;
        }
        if(t->op==GGML_OP_RMS_NORM || t->op==GGML_OP_MUL) {
            if(!valid(t,GGML_TYPE_F32,2048,1)) return false;
            const bool weighted=t->op==GGML_OP_MUL;
            auto * norm=weighted?t->src[0]:t;
            if(!norm || norm->op!=GGML_OP_RMS_NORM || !valid(norm,GGML_TYPE_F32,2048,1) || !valid(norm->src[0],GGML_TYPE_F32,2048,1)) return false;
            for(int i=1;i<GGML_MAX_SRC;++i) if(norm->src[i]) return false;
            if(weighted) {
                if(!valid(t->src[1],GGML_TYPE_F32,2048,1)) return false;
                for(int i=2;i<GGML_MAX_SRC;++i) if(t->src[i]) return false;
            }
            float epsilon; std::memcpy(&epsilon,norm->op_params,4); if(epsilon!=1e-5f) return false;
            void * input=norm->src[0]->data,* out=t->data,* weight=weighted?t->src[1]->data:nullptr,* absent=nullptr;
            int ncols=2048,channels=1,samples=1; int64_t stride=2048,zero_stride=0; float scale=1.f; Uint3 columns{1,11,2048},zero{};
            void * args[]={&input,&out,&ncols,&channels,&samples,&stride,&stride,&stride,&epsilon,
                &weight,&stride,&stride,&stride,&columns,&one,&one,&one,
                &absent,&zero_stride,&zero_stride,&zero_stride,&zero,&zero,&zero,&zero,&scale};
            launch(weighted?2:1,1,1,1024,1,128,args); return true;
        }
        if(t->op==GGML_OP_MUL_MAT) {
            // admit() has already restricted this to the four exact model matrix shapes.
            uint32_t width=uint32_t(t->src[0]->ne[0]),rows=uint32_t(t->ne[0]);
            void * input=t->src[1]->data; int64_t size=width; uint32_t ne1=1;
            void * quantize[]={&input,&q8,&size,&size,&size,&size,&size,&ne1,&one};
            launch(3,width/256,1,256,1,0,quantize);
            void * weight=t->src[0]->data,* ids=nullptr,* out=t->data; Fusion fusion{};
            uint32_t blocks=width/32,matrix_blocks=rows*blocks,ids_stride=0,output_rows=rows;
            void * project[]={&weight,&q8,&ids,&fusion,&out,&width,&one,&blocks,&blocks,&output_rows,&one,&matrix_blocks,&blocks,&output_rows,&one,&matrix_blocks,&blocks,&output_rows,&ids_stride};
            launch(4,rows,1,32,4,0,project); return true;
        }
        return false;
    }
};
}
