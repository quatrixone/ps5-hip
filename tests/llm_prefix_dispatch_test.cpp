#include "prefix_dispatch.h"
#include "ggml-impl.h"
#include <cassert>
#include <map>
#include <string>
#include <vector>
using namespace ps5hip_prefix;
static std::map<const ggml_tensor*,std::vector<char>> buffers;
static std::vector<char> scratch(2304);
static std::vector<unsigned> launches;
static unsigned loads=0,unloads=0,frees=0,syncs=0;
static std::string failure;
static Dispatch * active=nullptr;
static hipError_t status(const char * site) {
    if(active){uint64_t pending=0;for(const auto& row:active->calls)pending+=row.attempts-row.completions;assert(pending==1);}
    if(failure=="nonreturn" && std::strcmp(site,"launch")==0)throw std::runtime_error("nonreturn mock");
    return failure==site?hipErrorInvalidValue:hipSuccess;
}
extern "C" hipError_t hipModuleLoad(hipModule_t * m,const char*) { auto e=status("load"); if(e==hipSuccess)*m=reinterpret_cast<hipModule_t>(++loads);return e; }
extern "C" hipError_t hipModuleGetFunction(hipFunction_t * f,hipModule_t,const char * name) { auto e=status("function");if(e==hipSuccess){for(unsigned i=0;i<5;++i)if(std::strcmp(name,Dispatch::symbols[i])==0)*f=reinterpret_cast<hipFunction_t>(i+1);}return e; }
extern "C" hipError_t hipModuleUnload(hipModule_t) { auto e=status("unload"); if(e==hipSuccess)++unloads;return e; }
extern "C" hipError_t hipMalloc(void ** p,size_t n) { assert(n==2304);auto e=status("malloc");if(e==hipSuccess)*p=scratch.data();return e; }
extern "C" hipError_t hipFree(void * p) { assert(p==scratch.data());auto e=status("free");if(e==hipSuccess)++frees;return e; }
extern "C" hipError_t hipMemcpy(void * dst,const void * src,size_t n,hipMemcpyKind kind) {assert(kind==hipMemcpyDeviceToHost && n==4);auto e=status("copy");if(e==hipSuccess)std::memcpy(dst,src,n);return e;}
extern "C" hipError_t hipDeviceSynchronize() {++syncs;return status("sync");}
template<class T> T arg(void ** a,int i) {T value;std::memcpy(&value,a[i],sizeof(T));return value;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned lds,hipStream_t stream,void ** args,void ** extra) {
    auto e=status("launch");if(e!=hipSuccess)return e;
    unsigned index=reinterpret_cast<uintptr_t>(f)-1;assert(index<5 && gz==1 && bz==1 && !stream && !extra);launches.push_back(index);
    if(index==0) {assert(gx==1 && gy==4 && bx==256 && by==1 && lds==0);assert(arg<int64_t>(args,3)==2048 && arg<int64_t>(args,4)==1);assert(arg<size_t>(args,9)==1152 && arg<size_t>(args,10)==36864000 && arg<size_t>(args,14)==1);auto fd=arg<Dispatch::Uint3>(args,5);assert(fd.x==1 && fd.y==0 && fd.z==1);}
    if(index==1 || index==2) {assert(gx==1 && gy==1 && bx==1024 && by==1 && lds==128);assert(arg<int>(args,2)==2048 && arg<float>(args,8)==1e-5f);assert((arg<void*>(args,9)!=nullptr)==(index==2));assert(arg<void*>(args,17)==nullptr && arg<float>(args,25)==1.f);}
    if(index==3) {assert(gx==8 && gy==1 && bx==256 && by==1 && lds==0);assert(arg<void*>(args,1)==scratch.data() && arg<int64_t>(args,6)==2048);}
    if(index==4) {assert((gx==2048 || gx==256) && gy==1 && bx==32 && by==4 && lds==0);assert(arg<void*>(args,1)==scratch.data() && !arg<void*>(args,2));assert(arg<uint32_t>(args,5)==2048 && arg<uint32_t>(args,9)==gx && arg<uint32_t>(args,11)==gx*64 && arg<uint32_t>(args,18)==0);auto fusion=arg<Dispatch::Fusion>(args,3);for(auto b:fusion.bytes)assert(b==0);}
    return hipSuccess;
}
struct Terminal{};
static void fail(const char*,hipError_t) {throw Terminal{};}
static bool owned(const ggml_tensor * t) {auto i=buffers.find(t);return i!=buffers.end() && t->data==i->second.data() && ggml_nbytes(t)<=i->second.size();}
int main() {
    auto * ctx=ggml_init({1024*1024,nullptr,true});assert(ctx);
    auto tensor=[&](ggml_type type,int n,int m){return ggml_new_tensor_2d(ctx,type,n,m);};
    auto * table=tensor(GGML_TYPE_Q4_0,2048,32000),* token=tensor(GGML_TYPE_I32,1,1);
    auto * emb=ggml_get_rows(ctx,table,token),* norm=ggml_rms_norm(ctx,emb,1e-5f),* weight=tensor(GGML_TYPE_F32,2048,1),* weighted=ggml_mul(ctx,norm,weight);
    auto * q=ggml_mul_mat(ctx,tensor(GGML_TYPE_Q4_0,2048,2048),weighted);
    auto * k=ggml_mul_mat(ctx,tensor(GGML_TYPE_Q4_0,2048,256),weighted);
    auto * v=ggml_mul_mat(ctx,tensor(GGML_TYPE_Q4_0,2048,256),weighted);
    auto * graph=ggml_new_graph_custom(ctx,32,false);for(auto * t:{q,k,v})ggml_build_forward_expand(graph,t);
    for(auto * t:{table,token,emb,norm,weight,weighted,q->src[0],k->src[0],v->src[0],q,k,v}) {buffers[t].resize(ggml_nbytes(t));t->data=buffers[t].data();}
    std::array<const char*,5> paths{"getrows","norm","weighted","q8","q4"};
    Dispatch d(owned,fail);active=&d;d.initialize(paths);
    int32_t value=1;std::memcpy(token->data,&value,4);assert(d.execute_graph(graph));assert((launches==std::vector<unsigned>{0,1,2,3,4,3,4,3,4}));assert(d.completed_launches==9);
    auto counter=[&](Call call)->const CallCounter&{return d.calls[static_cast<unsigned>(call)];};
    assert(counter(Call::module_load).successes==5 && counter(Call::module_function).successes==5 && counter(Call::scratch_allocate).requested_bytes==2304);
    assert(counter(Call::launch).attempts==9 && counter(Call::launch).completions==9 && counter(Call::completion_sync).successes==9 && counter(Call::token_read).requested_bytes==8);
    for(int32_t index:{0,31999}) {std::memcpy(token->data,&index,4);assert(d.execute(emb));}
    const auto before=launches.size();
    for(int32_t index:{-1,32000}) {std::memcpy(token->data,&index,4);assert(!d.execute_graph(graph));assert(launches.size()==before);}
    std::memcpy(token->data,&value,4);auto saved=q->op;q->op=GGML_OP_ADD;assert(!d.execute_graph(graph) && launches.size()==before);q->op=saved;
    std::swap(graph->nodes[0],graph->nodes[1]);assert(!d.execute_graph(graph) && launches.size()==before);std::swap(graph->nodes[0],graph->nodes[1]);
    auto missing=*graph;missing.nodes++;missing.n_nodes--;assert(!d.execute_graph(&missing) && launches.size()==before);
    auto pointer=emb->data;emb->data=table->data;assert(!d.execute(emb) && launches.size()==before);emb->data=pointer;
    table->nb[1]++;assert(!d.execute(emb) && launches.size()==before);table->nb[1]--;
    d.shutdown();assert(frees==1 && unloads==5);
    assert(counter(Call::module_unload).successes==5 && counter(Call::scratch_free).requested_bytes==2304 && counter(Call::release_sync).successes==1);
    d.initialize(paths);assert(d.execute(emb));d.shutdown();assert(frees==2 && unloads==10);
    for(const char * site:{"load","function","malloc","launch","sync","unload","free","copy"}) {
        Dispatch broken(owned,fail);active=&broken;bool thrown=false;failure.clear();
        try {
            if(std::strcmp(site,"load")==0 || std::strcmp(site,"function")==0 || std::strcmp(site,"malloc")==0) failure=site;
            broken.initialize(paths);failure=site;
            if(failure=="unload" || failure=="free") broken.shutdown();else broken.execute(emb);
        } catch(const Terminal &) {thrown=true;}
        assert(thrown && broken.terminal);const auto released=frees;
        unsigned errors=0;for(const auto& row:broken.calls){assert(row.completions==row.successes+row.failures && row.completions<=row.attempts);errors+=row.failures;if(row.failures)assert(row.first_error==hipErrorInvalidValue && row.last_result==hipErrorInvalidValue);}
        assert(errors==1);
        try {broken.shutdown();assert(false);}catch(const std::runtime_error&){}
        assert(frees==released);
    }
    Dispatch nonreturn(owned,fail);active=&nonreturn;failure.clear();nonreturn.initialize(paths);failure="nonreturn";
    bool interrupted=false;try{nonreturn.execute(emb);}catch(const std::runtime_error&){interrupted=true;}
    assert(interrupted&&nonreturn.terminal);
    const auto& attempt=nonreturn.calls[static_cast<unsigned>(Call::launch)];assert(attempt.attempts==1&&attempt.completions==0&&attempt.successes==0&&attempt.failures==0);
    active=nullptr;
    failure.clear();buffers.clear();ggml_free(ctx);
    std::puts("PASS dispatch argument/geometry/order, whole-graph preflight, token bounds, spans, reload, eight HIP failure sites; mocked GPU only");
}
