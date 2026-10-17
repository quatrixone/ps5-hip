#include "full_dispatch.h"
#include "llm_layer_hip_mock.h"
#include "ggml-impl.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <cassert>
#include <iostream>
using json=nlohmann::json;
extern "C" enum ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*) {assert(false&&"dispatcher test must not invoke scheduler wrapper");return GGML_STATUS_FAILED;}
static ggml_type type(const std::string& name) {
    for(auto v:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q4_0,GGML_TYPE_I32,GGML_TYPE_I64})if(name==ggml_type_name(v))return v;
    throw std::runtime_error("unexpected trace type");
}
static ggml_op op(const std::string& name) {
    for(int i=0;i<GGML_OP_COUNT;++i)if(name==ggml_op_name(ggml_op(i)))return ggml_op(i);
    throw std::runtime_error("unexpected trace op");
}
static std::vector<std::pair<uintptr_t,size_t>> owned_ranges;
static bool owned(const ggml_tensor* t){
    if(!t||!t->buffer||!t->data)return false;
    const uintptr_t address=reinterpret_cast<uintptr_t>(t->data);
    for(const auto& range:owned_ranges)if(address>=range.first&&address-range.first<=range.second&&ggml_nbytes(t)<=range.second-(address-range.first))return true;
    return false;
}
static void fail(const char*,hipError_t){throw std::runtime_error("mock HIP failure");}
static void release_mock(ps5hip_full::Dispatch& d){d.layer.base.terminal=false;d.active=false;d.shutdown();}
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
        auto selected=ps5hip_full::select(&graph);
        std::vector<void*> memory;
        for(auto& t:tensors)if(!t.view_src){t.data=std::calloc(1,ggml_nbytes(&t));assert(t.data);memory.push_back(t.data);owned_ranges.emplace_back(reinterpret_cast<uintptr_t>(t.data),ggml_nbytes(&t));t.buffer=reinterpret_cast<ggml_backend_buffer_t>(&t);}
        for(unsigned pass=0;pass<20;++pass)for(auto& t:tensors)if(t.view_src&&t.view_src->data){t.data=static_cast<char*>(t.view_src->data)+t.view_offs;t.buffer=t.view_src->buffer;}
        for(auto* t:selected.layers[0].requests){auto name=std::string("PS5HIP-FULL#")+t->name+"#0";std::strcpy(t->name,name.c_str());}
        std::strcpy(selected.output_ids->name,"PS5HIP-FULL#out_ids#0");
        for(auto* t:{nodes[1]->src[1],nodes[2]->src[0],nodes[3]->src[1]}){auto name=std::string("PS5HIP-FULL#")+t->name+"#0";std::strcpy(t->name,name.c_str());}
        const auto paths=std::array<const char*,6>{"a","b","c","d","e","f"};
        auto controls=[&](unsigned pos){
            auto& r=selected.layers[0].requests;*static_cast<int32_t*>(r[0]->data)=2;*static_cast<int32_t*>(r[1]->data)=int32_t(pos);
            *static_cast<int64_t*>(r[2]->data)=pos;for(unsigned i=0;i<256;++i)static_cast<int64_t*>(r[3]->data)[i]=i*512+pos;
            for(int64_t i=0;i<selected.layers[0].extent;++i)static_cast<float*>(r[4]->data)[i]=i<=pos?0:-INFINITY;
            *static_cast<int32_t*>(selected.output_ids->data)=0;
        };
        controls(0);
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);assert(d.begin(&graph));
         assert(d.execute_graph(&graph));assert(d.expected_position==0);assert(d.finish(true));assert(d.expected_position==1);
         assert(!d.execute_graph(&graph));assert(d.expected_position==1);d.reset();assert(d.expected_position==0);release_mock(d);}
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);assert(d.begin(&graph));
         for(int i=0;i<801;++i)if(i==0||i>3){ggml_cgraph split{};split.n_nodes=1;split.nodes=&nodes[size_t(i)];assert(d.execute_graph(&split));}
         assert(d.finish(true)&&d.expected_position==1);release_mock(d);}
        for(unsigned which=0;which<13;++which){
         controls(0);ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);unsigned before=launches;
         assert(d.begin(&graph));ggml_cgraph split{};split.n_nodes=1;split.nodes=&nodes[0];
         if(which==0){split.nodes=&nodes[4];assert(!d.execute_graph(&split));assert(launches==before);}
         if(which==1){assert(!d.finish(true));assert(launches==before);}
         if(which==2){assert(d.execute_graph(&split));assert(!d.execute_graph(&split));assert(d.layer.base.terminal);}
         if(which==3){assert(d.execute_graph(&split));assert(!d.finish(false));assert(d.layer.base.terminal);}
         if(which==4){*static_cast<int32_t*>(selected.output_ids->data)=1;assert(!d.execute_graph(&split));assert(launches==before);}
         if(which==5){*static_cast<int32_t*>(selected.layers[0].requests[0]->data)=-1;assert(!d.execute_graph(&split));assert(launches==before);}
         if(which==6){*static_cast<int32_t*>(selected.layers[0].requests[1]->data)=1;assert(!d.execute_graph(&split));assert(launches==before);}
         if(which==7){*static_cast<int64_t*>(selected.layers[0].requests[2]->data)=1;assert(!d.execute_graph(&split));assert(launches==before);}
         if(which==8){static_cast<int64_t*>(selected.layers[0].requests[3]->data)[255]=0;assert(!d.execute_graph(&split));assert(launches==before);}
         if(which==9){static_cast<float*>(selected.layers[0].requests[4]->data)[1]=0;assert(!d.execute_graph(&split));assert(launches==before);}
         if(which==10){auto* old=nodes[4]->data;nodes[4]->data=nodes[0]->data;assert(!d.execute_graph(&split));assert(launches==before);nodes[4]->data=old;}
         if(which==11){assert(d.execute_graph(&split));assert(!d.finish(true));assert(d.layer.base.terminal);}
         if(which==12){split.n_nodes=0;assert(!d.execute_graph(&split));assert(launches==before);}
         if(d.layer.base.terminal){bool rejected=false;try{d.reset();}catch(const std::exception&){rejected=true;}assert(rejected);assert(!d.begin(&graph));}
         assert(d.expected_position==0);release_mock(d);
        }
        controls(0);
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);auto* t=selected.logits;auto* old=t->data;t->data=t->src[0]->data;assert(!d.begin(&graph));t->data=old;release_mock(d);}
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);auto* t=selected.layers[0].requests[4];auto* old=t->data;t->data=nodes[4]->data;assert(!d.begin(&graph));t->data=old;release_mock(d);}
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);auto* old=d.layer.full_scratch;d.layer.full_scratch=nodes[0]->data;assert(!d.begin(&graph));d.layer.full_scratch=old;release_mock(d);}
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);auto* t=nodes[4];auto* old=t->data;auto* input=nodes[0]->data;
         void* shared=std::calloc(1,8196);assert(shared);owned_ranges.emplace_back(reinterpret_cast<uintptr_t>(shared),8196);
         nodes[0]->data=shared;t->data=static_cast<char*>(shared)+4;assert(owned(t)&&owned(nodes[0]));assert(!d.begin(&graph));
         t->data=old;nodes[0]->data=input;owned_ranges.pop_back();std::free(shared);release_mock(d);}
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);auto* t=nodes[4];auto* old=t->data;t->data=nodes[0]->data;assert(d.begin(&graph));assert(d.execute_graph(&graph));assert(d.finish(true));t->data=old;release_mock(d);}
        {ps5hip_full::Dispatch d(owned,fail);d.initialize(paths);auto* token=selected.layers[0].requests[0];auto* old=token->data;
         token->data=selected.logits->data;controls(0);assert(d.begin(&graph));assert(d.execute_graph(&graph));assert(d.finish(true));token->data=old;release_mock(d);}
        {auto* t=selected.layers[0].requests[0];const std::string old=t->name;std::strcpy(t->name,"OTHER#inp_tokens#0");bool rejected=false;try{ps5hip_full::select(&graph,true);}catch(const std::exception&){rejected=true;}assert(rejected);std::strcpy(t->name,old.c_str());}
        for(auto* p:memory)std::free(p);
        owned_ranges.clear();
    }
    assert(allocations.empty());std::cout<<"PASS full split order/finish/live-control preflight; mocked HIP only\n";
}
