#include "full_graph.h"
#include "ggml-impl.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <cassert>
#include <iostream>
using json=nlohmann::json;
static ggml_type type(const std::string& name) {
    for(auto v:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q4_0,GGML_TYPE_I32,GGML_TYPE_I64})if(name==ggml_type_name(v))return v;
    throw std::runtime_error("unexpected trace type");
}
static ggml_op op(const std::string& name) {
    for(int i=0;i<GGML_OP_COUNT;++i)if(name==ggml_op_name(ggml_op(i)))return ggml_op(i);
    throw std::runtime_error("unexpected trace op");
}
int main(int argc,char** argv) {
    assert(argc>1);
    for(int file=1;file<argc;++file) {
        std::ifstream stream(argv[file]);json trace;stream>>trace;
        std::vector<ggml_tensor> tensors(trace.at("tensors").size());
        auto ref=[&](int id)->ggml_tensor* {assert(id>=-1);return id<0?nullptr:&tensors.at(size_t(id));};
        for(const auto& row:trace.at("tensors")) {
            auto& t=tensors.at(row.at("id").get<size_t>());t.type=type(row.at("type"));t.op=op(row.at("op"));
            t.flags=row.at("flags");const std::string name=row.at("name");assert(name.size()<sizeof(t.name));std::strcpy(t.name,name.c_str());
            for(unsigned i=0;i<4;++i){t.ne[i]=row.at("ne").at(i);t.nb[i]=row.at("nb").at(i);}
            for(unsigned i=0;i<GGML_MAX_SRC;++i)t.src[i]=ref(row.at("sources").at(i));
            t.view_src=ref(row.at("view_source"));t.view_offs=row.at("view_offset");
            const std::string bytes=row.at("op_params_hex");assert(bytes.size()==2*sizeof(t.op_params));
            for(size_t i=0;i<sizeof(t.op_params);++i)reinterpret_cast<unsigned char*>(t.op_params)[i]=static_cast<unsigned char>(std::stoul(bytes.substr(2*i,2),nullptr,16));
        }
        std::vector<ggml_tensor*> nodes;for(int id:trace.at("nodes"))nodes.push_back(ref(id));
        ggml_cgraph graph{};graph.n_nodes=int(nodes.size());graph.nodes=nodes.data();
        auto selection=ps5hip_full::select(&graph);assert(selection.logits==nodes.back());
        auto reject=[&]{bool rejected=false;try{ps5hip_full::select(&graph);}catch(const std::runtime_error&){rejected=true;}assert(rejected);};
        --graph.n_nodes;reject();++graph.n_nodes;
        auto* saved=nodes[10];nodes[10]=nodes[9];reject();nodes[10]=saved;
        nodes[1]->flags|=GGML_TENSOR_FLAG_COMPUTE;reject();nodes[1]->flags&=~GGML_TENSOR_FLAG_COMPUTE;
        nodes[4]->flags&=~GGML_TENSOR_FLAG_COMPUTE;reject();nodes[4]->flags|=GGML_TENSOR_FLAG_COMPUTE;
        auto* norm=selection.layers[1].nodes[1];auto* original=norm->src[0];norm->src[0]=selection.embedding;reject();norm->src[0]=original;
        auto* store=selection.layers[1].nodes[12];original=store->src[2];store->src[2]=selection.layers[0].caches[0];reject();store->src[2]=original;
        auto* weight=selection.layers[10].weights[2];auto kind=weight->type;weight->type=GGML_TYPE_F32;reject();weight->type=kind;
        auto* logits=selection.logits;--logits->ne[0];reject();++logits->ne[0];
        auto* view=selection.layers[21].nodes[19];view->view_offs=2;reject();view->view_offs=0;
        norm->op_params[0]^=1;reject();norm->op_params[0]^=1;
        assert(ps5hip_full::select(&graph).logits==logits);
    }
    std::cout<<"PASS full graph metadata/topology and ten mutation classes; no computation authorized\n";
}
