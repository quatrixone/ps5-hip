#pragma once
#include "ggml.h"
#include <array>
#include <cstring>
#include <stdexcept>

// Bounded selection only: execution remains the normal PS5HIP backend's job.
namespace ps5hip_prefix {
inline void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
inline bool shape(const ggml_tensor * t, ggml_type type, int64_t width, int64_t rows) {
    return t && t->type == type && t->ne[0] == width && t->ne[1] == rows &&
        t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t) && !t->view_src;
}
inline ggml_tensor * unique(ggml_cgraph * g, const char * name, ggml_op op) {
    ggml_tensor * result = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
        auto * t = ggml_graph_node(g, i);
        if (t->op == op && std::strcmp(t->name, name) == 0) {
            check(!result, "ambiguous layer-zero projection"); result = t;
        }
    }
    check(result, "missing layer-zero projection"); return result;
}
inline void present(ggml_cgraph * g, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(g); ++i) if (ggml_graph_node(g, i) == t) return;
    check(false, "dependency missing from actual graph");
}
inline void sources(const ggml_tensor * t, int count) {
    for (int i = 0; i < GGML_MAX_SRC; ++i) check((t->src[i] != nullptr) == (i < count), "unexpected source slots");
}
struct Selection {
    std::array<ggml_tensor *, 3> projections; // Q, K, V before RoPE
    ggml_tensor * weighted;
    ggml_tensor * normalized;
    ggml_tensor * embedding;
    // Original loaded model leaves plus actual request token: never fixture weights.
    std::array<ggml_tensor *, 6> leaves; // embedding table, token, norm weight, Q, K, V
    float epsilon;
};
inline Selection select(ggml_cgraph * graph) {
    check(graph, "missing actual graph");
    Selection s{};
    const char * names[] = {"Qcur-0", "Kcur-0", "Vcur-0"};
    const char * weights[] = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"};
    for (int i = 0; i < 3; ++i) {
        auto * t = s.projections[i] = unique(graph, names[i], GGML_OP_MUL_MAT);
        sources(t, 2);
        const int rows = i == 0 ? 2048 : 256;
        check(shape(t, GGML_TYPE_F32, rows, 1), "projection output layout");
        check(shape(t->src[0], GGML_TYPE_Q4_0, 2048, rows), "projection weight layout");
        check(std::strcmp(t->src[0]->name, weights[i]) == 0, "projection weight identity");
        check(t->op_params[0] == GGML_PREC_DEFAULT, "projection precision");
        s.leaves[i+3] = t->src[0];
        if (i == 0) s.weighted = t->src[1];
        else check(t->src[1] == s.weighted, "QKV must share actual weighted activation");
    }
    check(shape(s.weighted, GGML_TYPE_F32, 2048, 1) && s.weighted->op == GGML_OP_MUL, "weighted norm layout/op");
    sources(s.weighted, 2);
    s.normalized = s.weighted->src[0]; s.leaves[2] = s.weighted->src[1];
    check(shape(s.leaves[2], GGML_TYPE_F32, 2048, 1) && std::strcmp(s.leaves[2]->name, "blk.0.attn_norm.weight") == 0, "norm weight identity/layout");
    check(shape(s.normalized, GGML_TYPE_F32, 2048, 1) && s.normalized->op == GGML_OP_RMS_NORM, "norm layout/op");
    sources(s.normalized, 1);
    std::memcpy(&s.epsilon, s.normalized->op_params, sizeof(s.epsilon));
    check(s.epsilon == 1e-5f, "norm epsilon");
    s.embedding = s.normalized->src[0];
    check(shape(s.embedding, GGML_TYPE_F32, 2048, 1) && s.embedding->op == GGML_OP_GET_ROWS, "embedding layout/op");
    sources(s.embedding, 2);
    s.leaves[0] = s.embedding->src[0]; s.leaves[1] = s.embedding->src[1];
    check(shape(s.leaves[0], GGML_TYPE_Q4_0, 2048, 32000) && std::strcmp(s.leaves[0]->name, "token_embd.weight") == 0, "embedding table identity/layout");
    check(shape(s.leaves[1], GGML_TYPE_I32, 1, 1), "token input layout");
    for (auto * t : {s.embedding, s.normalized, s.weighted}) present(graph, t);
    for (auto * t : s.leaves) { check(t->op == GGML_OP_NONE, "leaf is computed"); sources(t, 0); }
    return s;
}
struct Graph {
    ggml_cgraph * graph;
    std::array<ggml_tensor *, 6> leaves;
    std::array<ggml_tensor *, 6> outputs; // embedding, unweighted, weighted, Q, K, V
};
// Caller owns ctx and must allocate/stage leaves using its backend. No CPU compute.
inline Graph materialize(ggml_context * ctx, const Selection & source) {
    Graph result{};
    for (size_t i = 0; i < result.leaves.size(); ++i) {
        auto * t = result.leaves[i] = ggml_dup_tensor(ctx, source.leaves[i]);
        check(t, "prefix leaf allocation"); ggml_set_name(t, source.leaves[i]->name); ggml_set_input(t);
    }
    result.outputs[0] = ggml_get_rows(ctx, result.leaves[0], result.leaves[1]);
    result.outputs[1] = ggml_rms_norm(ctx, result.outputs[0], source.epsilon);
    result.outputs[2] = ggml_mul(ctx, result.outputs[1], result.leaves[2]);
    result.graph = ggml_new_graph_custom(ctx, 32, false);
    for (int i = 0; i < 3; ++i) {
        result.outputs[i+3] = ggml_mul_mat(ctx, result.leaves[i+3], result.outputs[2]);
        ggml_set_name(result.outputs[i+3], source.projections[i]->name);
        ggml_build_forward_expand(result.graph, result.outputs[i+3]);
    }
    for (auto * t : result.outputs) ggml_set_output(t);
    check(ggml_graph_n_nodes(result.graph) == 6, "unexpected prefix node inventory");
    return result;
}
}
