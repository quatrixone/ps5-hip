#define PS5HIP_ENABLE_DENSE_F16 1
#include "llm_layer_numeric_mock.h"
#include "llm_dense_layer_checks.h"
#include "dense_layer_fixture.h"
#include "layer_dispatch.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <limits>
static std::map<void*,size_t> allocations;
static unsigned loads=0,unloads=0,launches=0,clears=0,dense_launches=0,attention_launches=0;
static size_t uploaded=0;
extern "C" hipError_t hipInit(unsigned){return hipSuccess;}
extern "C" hipError_t hipGetDeviceCount(int* n){*n=1;return hipSuccess;}
extern "C" hipError_t hipModuleLoad(hipModule_t* m,const char*){*m=reinterpret_cast<hipModule_t>(++loads);return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f,hipModule_t,const char* name){
 for(unsigned i=0;i<5;++i)if(std::strcmp(name,ps5hip_prefix::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+1);return hipSuccess;}
 for(unsigned i=0;i<16;++i)if(std::strcmp(name,ps5hip_layer::Dispatch::symbols[i])==0){*f=reinterpret_cast<hipFunction_t>(i+6);return hipSuccess;}
 if(!std::strcmp(name,"layer_dense_f16")){*f=reinterpret_cast<hipFunction_t>(22);return hipSuccess;}
 return hipErrorInvalidValue;
}
extern "C" hipError_t hipModuleUnload(hipModule_t){++unloads;return hipSuccess;}
extern "C" hipError_t hipMalloc(void** p,size_t n){*p=std::aligned_alloc(256,(n+255)&~size_t(255));assert(*p);std::memset(*p,0,n);assert(allocations.emplace(*p,n).second);return hipSuccess;}
extern "C" hipError_t hipFree(void* p){assert(allocations.erase(p)==1);std::free(p);return hipSuccess;}
extern "C" hipError_t hipMemcpy(void* d,const void* s,size_t n,hipMemcpyKind kind){if(kind==hipMemcpyHostToDevice)uploaded+=n;std::memmove(d,s,n);return hipSuccess;}
extern "C" hipError_t hipMemset(void* p,int value,size_t n){++clears;std::memset(p,value,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize(){return hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t stream,void** args,void** extra){assert(uintptr_t(f)>=1&&uintptr_t(f)<=22&&gx&&gy&&gz==1&&bx&&by&&bz==1&&!stream&&!extra);++launches;if(uintptr_t(f)==11||uintptr_t(f)==13)++attention_launches;if(uintptr_t(f)==22){++dense_launches;for(layer_numeric::lane=0;layer_numeric::lane<gx*bx;layer_numeric::lane++)layer_numeric::layer_dense_f16(layer_numeric::arg<const _Float16*>(args,0),layer_numeric::arg<const _Float16*>(args,1),layer_numeric::arg<float*>(args,2),layer_numeric::arg<unsigned>(args,3),layer_numeric::arg<unsigned>(args,4));}else layer_numeric::execute(unsigned(uintptr_t(f)),gx*bx,args);return hipSuccess;}
static std::vector<std::pair<void*,size_t>> spans;
static bool owned(const ggml_tensor* t){if(!t||!t->data)return false;const auto p=uintptr_t(t->data);for(const auto& item:spans){const auto base=uintptr_t(item.first);if(p>=base&&p-base<=item.second&&ggml_nbytes(t)<=item.second-(p-base))return true;}return false;}
static void fail(const char*,hipError_t){assert(false);}
static void remember(ggml_tensor* t){if(t&&!t->view_src&&t->data)spans.emplace_back(t->data,ggml_nbytes(t));}
struct Context {ggml_context* p;explicit Context(size_t bytes){p=ggml_init({bytes,nullptr,false});assert(p);}~Context(){ggml_free(p);}};
static void controls(ps5hip_layer::Graph& g,unsigned position){
 *static_cast<int32_t*>(g.uploads[0].clone->data)=0;
 *static_cast<int32_t*>(g.uploads[1].clone->data)=int32_t(position);
 *static_cast<int64_t*>(g.uploads[2].clone->data)=position;
 auto* ids=static_cast<int64_t*>(g.uploads[3].clone->data);for(unsigned i=0;i<256;i++)ids[i]=int64_t(i*512+position);
 auto* mask=static_cast<float*>(g.uploads[4].clone->data);for(unsigned i=0;i<unsigned(g.extent);i++)mask[i]=i<=position?0:-std::numeric_limits<float>::infinity();
}
static unsigned checked=0;static double maximum_error=0;
static void compare(const ggml_tensor* a,const ggml_tensor* b,bool exact=false,bool probabilities=false){
 assert(a->type==b->type&&ggml_nbytes(a)==ggml_nbytes(b));
 try{maximum_error=std::max(maximum_error,dense_check_values(a->data,b->data,size_t(ggml_nelements(a)),a->type==GGML_TYPE_F16,exact,probabilities));}
 catch(const std::runtime_error& error){std::fprintf(stderr,"boundary %s: %s\n",a->name,error.what());throw;}++checked;
}
static void score_oracle(const ps5hip_layer::Graph& g,const ps5hip_layer::Cache& cache){
 const auto* q=static_cast<const float*>(g.nodes[5]->data);const auto* k=static_cast<const _Float16*>(cache.k->data);const auto* scores=static_cast<const float*>(g.nodes[23]->data);
 const double epsilon=std::ldexp(1.,-24),gamma=64*epsilon/(1-64*epsilon);
 for(unsigned head=0;head<32;head++)for(unsigned slot=0;slot<unsigned(g.extent);slot++){
  double sum=0,magnitude=0;for(unsigned i=0;i<64;i++){const double product=double(float((_Float16)q[head*64+i]))*double(float(k[slot*256+(head/8)*64+i]));sum+=product;magnitude+=std::abs(product);}
  if(std::abs(double(scores[head*unsigned(g.extent)+slot])-sum)>gamma*magnitude+std::ldexp(1.,-126))throw std::runtime_error("device Q/K gamma64 score bound");
 }
}
static DenseSnapshot snapshot(const ps5hip_layer::Graph& g){DenseSnapshot result;for(const auto* t:g.outputs){const auto* p=static_cast<const unsigned char*>(t->data);result.emplace_back(p,p+ggml_nbytes(t));}return result;}
int main(int argc,char**argv){
 assert(argc==2||argc==3);const bool admission=std::string(argv[1])=="admission";assert(admission||std::string(argv[1])=="numeric");
 assert(!admission||argc==2);
 FILE* exported=argc==3?std::fopen(argv[2],"wbx"):nullptr;if(argc==3&&!exported)throw std::runtime_error("exclusive reference creation");
 size_t export_bytes=0;unsigned export_records=0;
 auto export_tensor=[&](const ggml_tensor* t){if(!exported)return;const size_t n=ggml_nbytes(t);if(std::fwrite(t->data,1,n,exported)!=n)throw std::runtime_error("reference write");export_bytes+=n;++export_records;};
 Context resident(100*1024*1024);ps5hip_layer::Selection selection{};selection.dense_fixture=true;selection.extent=256;
 ps5hip_layer::Weights weights{};size_t matrix_bytes=0;
 for(unsigned i=0;i<15;i++){
  auto spec=ps5hip_layer::detail::spec(i,256,true,true);auto* t=ggml_new_tensor(resident.p,spec.type,4,spec.ne);ggml_set_name(t,spec.name);
  if(i<10){selection.weights[i]=t;weights.uploads[i]={t,t};remember(t);
   if(t->type==GGML_TYPE_F16){matrix_bytes+=ggml_nbytes(t);auto* data=static_cast<_Float16*>(t->data);for(int64_t n=0;n<ggml_nelements(t);n++)data[n]=(_Float16)(ps5hip_dense_fixture::weight(i,uint64_t(n)));}
   else {auto* data=static_cast<float*>(t->data);for(int64_t n=0;n<ggml_nelements(t);n++)data[n]=i==0?0.f:ps5hip_dense_fixture::norm(uint64_t(n));}
  }else selection.requests[i-10]=t;
 }
 assert(matrix_bytes==88080384);
 auto immutable_weights=[&]{for(unsigned role=1;role<10;role++){auto* t=selection.weights[role];for(int64_t n=0;n<ggml_nelements(t);n++){
  if(t->type==GGML_TYPE_F16){const _Float16 expected=(_Float16)(ps5hip_dense_fixture::weight(role,uint64_t(n)));assert(!std::memcmp(static_cast<_Float16*>(t->data)+n,&expected,2));}
  else assert(static_cast<float*>(t->data)[n]==ps5hip_dense_fixture::norm(uint64_t(n)));
 }}};immutable_weights();
 Context cache_context(2*1024*1024);auto hostcache=ps5hip_layer::make_cache(cache_context.p),devicecache=ps5hip_layer::make_cache(cache_context.p);
 for(auto* t:{hostcache.k,hostcache.v,devicecache.k,devicecache.v}){std::memset(t->data,0,ggml_nbytes(t));remember(t);}
 Context requests(12*1024*1024);std::array<ps5hip_layer::Graph,2> host,device;
 for(unsigned i=0;i<2;i++){
  host[i]=ps5hip_layer::materialize(requests.p,selection,weights,hostcache,i?512:256);
  device[i]=ps5hip_layer::materialize(requests.p,selection,weights,devicecache,i?512:256);
  for(auto* g:{&host[i],&device[i]}){for(auto* t:g->nodes){remember(t);ggml_set_output(t);}for(auto& u:g->uploads)remember(u.clone);controls(*g,0);}
 }
 ps5hip_layer::Dispatch dispatch(owned,fail);dispatch.initialize_dense_fixture({"getrows","norm","weighted","q8","q4","layer"});
 assert(loads==6&&dispatch.dense_function);
 auto& g=device[0];assert(dispatch.preflight(g.graph));
 auto reject=[&]{auto before=launches;assert(!dispatch.execute_graph(g.graph));assert(launches==before&&dispatch.expected_position==0);};
 ps5hip_layer::Dispatch legacy(owned,fail);assert(!legacy.preflight(g.graph));
 for(unsigned role:{2u,3u,4u,5u,7u,8u,9u}){
  auto* w=selection.weights[role];auto type=w->type;w->type=GGML_TYPE_Q4_0;reject();w->type=type;
  auto stride=w->nb[1];w->nb[1]+=2;reject();w->nb[1]=stride;
  auto size=w->ne[0];w->ne[0]--;reject();w->ne[0]=size;
  auto* ptr=w->data;w->data=selection.weights[0]->data;reject();w->data=ptr;
  char name[GGML_MAX_NAME];std::memcpy(name,w->name,sizeof(name));ggml_set_name(w,"cache_k_l0");reject();std::memcpy(w->name,name,sizeof(name));
 }
 auto* input=selection.weights[0];void* saved=input->data;input->data=dispatch.half;spans.emplace_back(dispatch.half,ps5hip_layer::Dispatch::half_bytes);reject();spans.pop_back();input->data=saved;
 *static_cast<int32_t*>(g.uploads[0].clone->data)=1;reject();controls(g,0);assert(dispatch.preflight(g.graph));
 if(admission){dispatch.shutdown();assert(allocations.empty()&&unloads==6);std::puts("DENSE_LAYER_ADMISSION_PASS roles=7 mutations=37 default_rejected=true launches=0");return 0;}
 auto* cpu=ggml_backend_cpu_init();assert(cpu);ggml_backend_cpu_set_n_threads(cpu,2);
 unsigned requests_done=0;std::vector<DenseSnapshot> reset_reference;
 auto step=[&](unsigned position,float scale,bool save_reset){
  auto* data=static_cast<float*>(input->data);for(unsigned i=0;i<2048;i++)data[i]=ps5hip_dense_fixture::input(i,position,scale);
  auto& h=host[position<256?0:1];auto& d=device[position<256?0:1];controls(h,position);controls(d,position);
  assert(ggml_backend_graph_compute(cpu,h.graph)==GGML_STATUS_SUCCESS);assert(dispatch.execute_graph(d.graph));
  for(unsigned i=0;i<21;i++){compare(d.outputs[i],h.outputs[i],i==0,i==9);export_tensor(h.outputs[i]);}
  score_oracle(d,devicecache);
  if(ps5hip_dense_fixture::cache_checkpoint(position)){compare(devicecache.k,hostcache.k);compare(devicecache.v,hostcache.v);export_tensor(hostcache.k);export_tensor(hostcache.v);}
  for(auto* t:{devicecache.k,devicecache.v}){const auto* p=static_cast<const _Float16*>(t->data);for(unsigned f=0;f<256;f++)for(unsigned slot=position+1;slot<512;slot++)assert(p[t==devicecache.k?slot*256+f:f*512+slot]==0);}
  if(position<2&&save_reset)reset_reference.push_back(snapshot(d));
  else if(position<2&&scale==1.f)dense_check_reset(snapshot(d),reset_reference[position]);
  ++requests_done;
 };
 for(unsigned p=0;p<ps5hip_dense_fixture::cases[0].count;p++){step(p,ps5hip_dense_fixture::cases[0].scale,true);if(p==0||p==1||p==255||p==256||p==511){std::printf("checkpoint %u checked=%u\n",p,checked);std::fflush(stdout);}}
 auto reset=[&]{for(auto* t:{hostcache.k,hostcache.v,devicecache.k,devicecache.v})std::memset(t->data,0,ggml_nbytes(t));dispatch.reset();};
 for(unsigned c=1;c<sizeof(ps5hip_dense_fixture::cases)/sizeof(ps5hip_dense_fixture::cases[0]);++c){reset();const auto& item=ps5hip_dense_fixture::cases[c];for(unsigned p=0;p<item.count;++p)step(p,item.scale,false);}
 if(exported){const bool flushed=std::fflush(exported)==0;const bool closed=std::fclose(exported)==0;if(!flushed||!closed)throw std::runtime_error("reference close");assert(export_records==11084&&export_bytes==154392576);}
 immutable_weights();ggml_backend_free(cpu);dispatch.shutdown();assert(allocations.empty()&&loads==6&&unloads==6);
 assert(dense_launches==requests_done*7&&attention_launches==requests_done*2);
 std::printf("DENSE_LAYER_NUMERIC_PASS requests=%u checked=%u resident_f16_bytes=%zu max_abs=%.12g GPU_execution=false\n",requests_done,checked,matrix_bytes,maximum_error);
}
