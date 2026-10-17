#pragma once
#include "model_layer.h"
#include <string>
#include <cstdio>

namespace ps5hip_full {
using ps5hip_layer::check;
using Spec=ps5hip_layer::detail::Spec;
struct Selection {
    std::array<ps5hip_layer::Selection,22> layers{};
    ggml_tensor* embedding=nullptr;
    ggml_tensor* logits=nullptr;
    ggml_tensor* output_ids=nullptr;
};
inline std::string layer_name(const char* value,unsigned layer) {
    std::string s=value;
    for(const auto& pair: {std::pair<std::string,std::string>{"blk.0.","blk."+std::to_string(layer)+"."},
                          {"_l0","_l"+std::to_string(layer)}, {"-0","-"+std::to_string(layer)}}) {
        const auto at=s.find(pair.first);if(at!=std::string::npos)s.replace(at,pair.first.size(),pair.second);
    }
    return s;
}
inline void match(ggml_tensor* t,Spec spec,const std::string& name) {
    if(t&&(std::strcmp(t->name,name.c_str())||t->type!=spec.type||t->op!=spec.op))
        std::fprintf(stderr,"full tensor mismatch: got %s/%s/%s expected %s/%s/%s\n",t->name,ggml_type_name(t->type),ggml_op_name(t->op),name.c_str(),ggml_type_name(spec.type),ggml_op_name(spec.op));
    spec.name=name.c_str();ps5hip_layer::detail::match(t,spec);
}
inline void sources(ggml_tensor* t,std::initializer_list<ggml_tensor*> expected,ggml_tensor* owner=nullptr) {
    unsigned i=0;for(auto* p:expected)check(t->src[i++]==p,"full graph source topology");
    for(;i<GGML_MAX_SRC;++i)check(!t->src[i],"full graph extra source");
    check(t->view_src==owner,"full graph view owner");
}
inline void leaf(ggml_tensor* t,Spec spec,const std::string& name) {
    match(t,spec,name);sources(t,{});
}
// Metadata/topology only. Controls, actual GPU spans and scheduler split ownership
// must be validated separately before this selection may authorize computation.
inline Selection select(ggml_cgraph* graph,bool scheduler_copies=false) {
    auto control=[&](ggml_tensor* t,Spec spec,const std::string& name){
        leaf(t,spec,scheduler_copies?"PS5HIP-FULL#"+name+"#0":name);
    };
    check(graph&&ggml_graph_n_nodes(graph)==801,"full graph node count");
    std::set<ggml_tensor*> unique;
    for(int i=0;i<801;++i) {
        auto* t=ggml_graph_node(graph,i);
        check(t&&unique.insert(t).second,"full graph duplicate node");
        check(bool(t->flags&GGML_TENSOR_FLAG_COMPUTE)==(i==0||i>3),"full graph active branch");
    }
    auto node=[&](unsigned i){return ggml_graph_node(graph,int(i));};
    Selection result{};result.embedding=node(0);
    const auto& ns=ps5hip_layer::detail::node_specs;
    const auto& ls=ps5hip_layer::detail::leaf_specs;
    match(node(0),ns[0],"embd");
    leaf(node(0)->src[0],ls[0],"token_embd.weight");control(node(0)->src[1],ls[10],"inp_tokens");
    sources(node(0),{node(0)->src[0],node(0)->src[1]});
    match(node(1),ns[0],"node_1");control(node(1)->src[1],ls[10],"inp_mixed_tokens");
    sources(node(1),{node(0)->src[0],node(1)->src[1]});
    auto dense=ls[1];dense.op=GGML_OP_DUP;match(node(2),dense,"node_2");
    control(node(2)->src[0],ls[1],"inp_mixed_embd");sources(node(2),{node(2)->src[0]});
    dense.op=GGML_OP_SET_ROWS;match(node(3),dense," (view)");
    control(node(3)->src[1],ls[12],"inp_mixed_slots");sources(node(3),{node(1),node(3)->src[1],node(2)},node(2));
    unsigned cursor=4;ggml_tensor* input=node(0);
    for(unsigned l=0;l<22;++l) {
        auto& s=result.layers[l];s.nodes[0]=input;
        s.weights[0]=node(0)->src[0];s.requests[0]=node(0)->src[1];
        // Attention extent is already visible in the fixed score-node position.
        s.extent=node(cursor+22)->ne[0];check(s.extent==256||s.extent==512,"full graph attention extent");
        if(l)check(s.extent==result.layers[0].extent,"full graph mixed extents");
        std::array<ggml_tensor*,2> gathered{};
        for(unsigned i=1;i<37;++i) {
            if(l==21&&i==29)for(unsigned j=0;j<2;++j) {
                auto* t=node(cursor++);gathered[j]=t;auto spec=ns[0];
                match(t,spec,j?"node_789":"node_788");
                if(!result.output_ids)result.output_ids=t->src[1];
                control(result.output_ids,ls[10],"out_ids");
                sources(t,{j?input:s.nodes[28],result.output_ids});
            }
            auto* t=node(cursor++);s.nodes[i]=t;const auto spec=ps5hip_layer::detail::spec(i,s.extent);
            match(t,spec,layer_name(spec.name,l));
            for(unsigned j=0;j<GGML_MAX_SRC;++j) {
                int ref=j<4?spec.refs[j]:-100;auto* actual=t->src[j];
                if(ref==-100){check(!actual,"full graph unexpected source");continue;}
                if(ref>=0) {
                    check(unsigned(ref)<i,"full graph forward reference");
                    auto* expected=l==21&&i==29?gathered.at(j):s.nodes.at(unsigned(ref));
                    check(actual==expected,"full graph node source");
                } else {
                    const unsigned k=unsigned(-1-ref);ggml_tensor** slot=k<10?&s.weights.at(k):k<15?&s.requests.at(k-10):&s.caches.at(k-15);
                    check(actual&&(!*slot||*slot==actual),"full graph leaf identity");*slot=actual;
                }
            }
        }
        for(unsigned k=1;k<17;++k) {
            auto* t=ps5hip_layer::detail::leaf(s,k);auto spec=ps5hip_layer::detail::spec(k,s.extent,true);
            if(k>=10&&k<15)control(t,spec,layer_name(spec.name,l));else leaf(t,spec,layer_name(spec.name,l));
            if(l&&k>=10&&k<15)check(t==result.layers[0].requests[k-10],"full graph shared control identity");
        }
        for(unsigned i=1;i<37;++i) {
            const int owner=ns[i].owner;
            check(s.nodes[i]->view_src==(owner==-100?nullptr:ps5hip_layer::detail::ref(s,owner)),"full graph layer owner");
        }
        input=s.nodes[36];
    }
    auto* norm=node(cursor++);match(norm,ns[1],"norm");sources(norm,{input});
    auto* weighted=node(cursor++);match(weighted,ns[2],"result_norm");
    leaf(weighted->src[1],ls[1],"output_norm.weight");sources(weighted,{norm,weighted->src[1]});
    auto* logits=node(cursor++);auto projection=ns[3];projection.ne[0]=32000;
    projection.nb[1]=projection.nb[2]=projection.nb[3]=128000;
    match(logits,projection,"result_output");leaf(logits->src[0],ls[0],"output.weight");sources(logits,{logits->src[0],weighted});
    check(cursor==801,"full graph trailing nodes");result.logits=logits;return result;
}
}
