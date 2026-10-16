#include "model_prefix.h"
#include <cassert>
#include <cstdio>
using namespace ps5hip_prefix;
template<class F> void rejects(F f) { bool failed=false; try { f(); } catch (const std::runtime_error &) { failed=true; } assert(failed); }
int main() {
    auto * ctx = ggml_init({1024*1024, nullptr, true}); assert(ctx);
    auto tensor = [&](ggml_type type, int n, int m, const char * name) {
        auto * t = ggml_new_tensor_2d(ctx, type, n, m); ggml_set_name(t, name); return t;
    };
    auto * table=tensor(GGML_TYPE_Q4_0,2048,32000,"token_embd.weight");
    auto * token=tensor(GGML_TYPE_I32,1,1,"inp_tokens");
    auto * emb=ggml_get_rows(ctx,table,token);
    auto * norm=ggml_rms_norm(ctx,emb,1e-5f);
    auto * weight=tensor(GGML_TYPE_F32,2048,1,"blk.0.attn_norm.weight");
    auto * weighted=ggml_mul(ctx,norm,weight);
    auto * q=ggml_mul_mat(ctx,tensor(GGML_TYPE_Q4_0,2048,2048,"blk.0.attn_q.weight"),weighted); ggml_set_name(q,"Qcur-0");
    auto * k=ggml_mul_mat(ctx,tensor(GGML_TYPE_Q4_0,2048,256,"blk.0.attn_k.weight"),weighted); ggml_set_name(k,"Kcur-0");
    auto * v=ggml_mul_mat(ctx,tensor(GGML_TYPE_Q4_0,2048,256,"blk.0.attn_v.weight"),weighted); ggml_set_name(v,"Vcur-0");
    auto * g=ggml_new_graph_custom(ctx,64,false);
    ggml_build_forward_expand(g,ggml_add(ctx,q,q)); // Unsupported graph node between Q and K/V.
    ggml_build_forward_expand(g,ggml_reshape_2d(ctx,q,64,32));
    ggml_build_forward_expand(g,v); ggml_build_forward_expand(g,k);
    auto selected=select(g); assert(selected.projections[0]==q && selected.projections[1]==k && selected.projections[2]==v);
    assert(selected.leaves[0]==table && selected.leaves[1]==token);
    const int original_nodes=ggml_graph_n_nodes(g);
    auto * clone_ctx=ggml_init({1024*1024,nullptr,true}); assert(clone_ctx);
    auto cloned=materialize(clone_ctx,selected); assert(ggml_graph_n_nodes(cloned.graph)==6);
    assert(ggml_graph_n_nodes(g)==original_nodes && q->src[1]==weighted);
    for(auto * leaf:cloned.leaves) assert(!leaf->data && !leaf->buffer);
    assert(cloned.outputs[3]->src[1]==cloned.outputs[4]->src[1] && cloned.outputs[4]->src[1]==cloned.outputs[5]->src[1]);
    auto old=table->type; table->type=GGML_TYPE_F16; rejects([&]{select(g);}); table->type=old;
    table->nb[1]++; rejects([&]{select(g);}); table->nb[1]--;
    k->src[1]=norm; rejects([&]{select(g);}); k->src[1]=weighted;
    auto p=norm->op_params[0]; norm->op_params[0]=0; rejects([&]{select(g);}); norm->op_params[0]=p;
    token->type=GGML_TYPE_I64; rejects([&]{select(g);}); token->type=GGML_TYPE_I32;
    ggml_set_name(k,"wrong"); rejects([&]{select(g);}); ggml_set_name(k,"Kcur-0");
    q->src[2]=token; rejects([&]{select(g);}); q->src[2]=nullptr;
    auto * duplicate=ggml_mul_mat(ctx,q->src[0],weighted); ggml_set_name(duplicate,"Qcur-0"); ggml_build_forward_expand(g,duplicate);
    rejects([&]{select(g);});
    ggml_free(clone_ctx); ggml_free(ctx);
    std::puts("PASS actual-prefix selection/materialization: closure, source immutability, shared activation, eight rejection cases; GPU execution=false");
}
