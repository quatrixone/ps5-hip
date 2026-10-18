#pragma once
#include "dense_layer_qualification.h"
#include "dense_layer_fixture.h"
#include "model_layer.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace m0 {
// Append-only packed stream. Caller enforces exclusive creation and fsync/finalization.
using DensePackedWrite=bool (*)(uint64_t offset,const void*,size_t,void*);
using DenseLogSync=bool (*)();
struct NativeDenseResult {
 LayerStatistics layer;ps5hip_prefix::CallStatistics calls;HipCallStatistics storage;
 unsigned requests=0,records=0;uint64_t bytes=0;
};
inline NativeDenseResult run_native_dense_layer(const std::array<const char*,6>& modules,
 DensePackedWrite writer,void* writer_state,DenseLogSync log_sync,void (*snapshot)(const char*)=nullptr){
 require(writer&&log_sync);const auto before=hip_call_statistics();
 std::puts("M2_DENSE PHASE prepare_begin");require(log_sync());if(snapshot)snapshot("before_dense");
 ggml_backend_register(registration());auto backend=ggml_backend_dev_init(device_handle(),nullptr);require(backend);configure_dense_layer(backend,modules);
 // No RAII resource cleanup: fatal HIP errors must retain possibly-live storage.
 auto* persistent=ggml_init({1024*1024,nullptr,true});require(persistent);
 ps5hip_layer::Selection selected{};selected.dense_fixture=true;selected.extent=256;ps5hip_layer::Weights weights{};
 for(unsigned i=0;i<15;i++){
  auto spec=ps5hip_layer::detail::spec(i,256,true,true);auto* t=ggml_new_tensor(persistent,spec.type,4,spec.ne);require(t);ggml_set_name(t,spec.name);
  if(i<10){selected.weights[i]=t;weights.uploads[i]={t,t};}else selected.requests[i-10]=t;
 }
 auto cache=ps5hip_layer::make_cache(persistent);std::array<ggml_backend_buffer_t,12> resident{};
 auto buft=ggml_backend_get_default_buffer_type(backend);
 for(unsigned i=0;i<12;i++){
  auto* t=i<10?selected.weights[i]:i==10?cache.k:cache.v;
  resident[i]=ggml_backend_alloc_buffer(backend,ggml_backend_buft_get_alloc_size(buft,t));require(resident[i]);
  require(ggml_backend_tensor_alloc(resident[i],t,ggml_backend_buffer_get_base(resident[i]))==GGML_STATUS_SUCCESS);
 }
 uint64_t weight_bytes=0,request_bytes=0;std::vector<ggml_fp16_t> half(32768);std::vector<float> values(65536);
 for(unsigned role=1;role<10;role++){
  auto* t=selected.weights[role];const bool f16=t->type==GGML_TYPE_F16;const size_t count=size_t(ggml_nelements(t));
  for(size_t start=0;start<count;){const size_t n=std::min(size_t(16384),count-start);
   if(f16){for(size_t j=0;j<n;j++)half[j]=ggml_fp32_to_fp16(ps5hip_dense_fixture::weight(role,start+j));ggml_backend_tensor_set(t,half.data(),start*2,n*2);}
   else {for(size_t j=0;j<n;j++)values[j]=ps5hip_dense_fixture::norm(start+j);ggml_backend_tensor_set(t,values.data(),start*4,n*4);}start+=n;
  }weight_bytes+=ggml_nbytes(t);
 }
 require(weight_bytes==88096768);
 std::array<ggml_context*,2> contexts{};std::array<ggml_backend_buffer_t,2> buffers{};std::array<ps5hip_layer::Graph,2> graphs{};
 for(unsigned i=0;i<2;i++){contexts[i]=ggml_init({1024*1024,nullptr,true});require(contexts[i]);graphs[i]=ps5hip_layer::materialize(contexts[i],selected,weights,cache,i?512:256);buffers[i]=ggml_backend_alloc_ctx_tensors(contexts[i],backend);require(buffers[i]);}
 std::puts("M2_DENSE PHASE prepare_complete weights=88096768 cases=8");require(log_sync());if(snapshot)snapshot("after_dense_prepare");
 unsigned requests=0,records=0;uint64_t bytes=0;
 auto dump=[&](const ggml_tensor* t){const size_t n=ggml_nbytes(t);require(n&&n<=values.size()*4&&bytes<=154392576&&n<=154392576-bytes);ggml_backend_tensor_get(t,values.data(),0,n);
  if(t->type==GGML_TYPE_F32)for(size_t i=0;i<n/4;i++)require(std::isfinite(values[i]));
  require(writer(bytes,values.data(),n,writer_state));bytes+=n;++records;
 };
 for(unsigned ci=0;ci<8;ci++){
  const auto& c=ps5hip_dense_fixture::cases[ci];if(ci)reset_layer_cache(backend);
  for(auto* t:{cache.k,cache.v})ggml_backend_tensor_memset(t,0,0,ggml_nbytes(t));ggml_backend_synchronize(backend);
  std::printf("M2_DENSE CASE index=%u name=%s count=%u reset=true\n",ci,c.name,c.count);require(log_sync());
  for(unsigned position=0;position<c.count;position++){
   auto& graph=graphs[position<256?0:1];for(unsigned i=0;i<2048;i++)values[i]=ps5hip_dense_fixture::input(i,position,c.scale);
   ggml_backend_tensor_set(selected.weights[0],values.data(),0,8192);request_bytes+=8192;
   const int32_t token=0,pos=int32_t(position);const int64_t slot=position;std::array<int64_t,256> vslots{};for(unsigned i=0;i<256;i++)vslots[i]=int64_t(i*512+position);
   const void* controls[]={&token,&pos,&slot,vslots.data()};const size_t sizes[]={4,4,8,sizeof(vslots)};
   for(unsigned i=0;i<4;i++){ggml_backend_tensor_set(graph.uploads[i].clone,controls[i],0,sizes[i]);request_bytes+=sizes[i];}
   for(unsigned i=0;i<unsigned(graph.extent);i++)values[i]=i<=position?0.f:-std::numeric_limits<float>::infinity();
   const size_t mask_bytes=size_t(graph.extent)*4;ggml_backend_tensor_set(graph.uploads[4].clone,values.data(),0,mask_bytes);request_bytes+=mask_bytes;
   execute_dense_layer(backend,graph.graph);for(auto* t:graph.outputs)dump(t);
   if(ps5hip_dense_fixture::cache_checkpoint(position)){dump(cache.k);dump(cache.v);}
   ++requests;std::printf("M2_DENSE REQUEST index=%u case=%s position=%u records=%u bytes=%llu\n",requests-1,c.name,position,records,static_cast<unsigned long long>(bytes));require(log_sync());
  }
 }
 auto state=layer_statistics(backend);state.uploaded_weight_bytes=weight_bytes;state.uploaded_request_bytes=request_bytes;
 require(requests==526&&records==11084&&bytes==154392576&&state.invocations==526&&state.cache_resets==7&&state.completed_launches==17358);
 std::puts("M2_DENSE PHASE cleanup_begin");require(log_sync());auto calls=disable_layer(backend);
 // Successful disable synchronizes dispatch before any resident/request storage frees.
 for(unsigned i=0;i<2;i++){ggml_backend_buffer_free(buffers[i]);ggml_free(contexts[i]);}
 for(auto b:resident)ggml_backend_buffer_free(b);ggml_free(persistent);ggml_backend_free(backend);
 auto storage=hip_call_statistics();for(unsigned i=0;i<unsigned(HipCall::count);i++){auto& a=storage.calls[i];const auto& b=before.calls[i];require(a.attempts>=b.attempts&&a.completions>=b.completions&&a.successes>=b.successes&&a.failures==b.failures&&a.requested_bytes>=b.requested_bytes);a.attempts-=b.attempts;a.completions-=b.completions;a.successes-=b.successes;a.failures-=b.failures;a.requested_bytes-=b.requested_bytes;}
 for(unsigned i=0;i<calls.size();i++){const auto& c=calls[i];require(c.attempts==c.completions&&c.completions==c.successes&&!c.failures);std::printf("M2_DENSE DISPATCH kind=%u attempts=%llu completions=%llu successes=%llu failures=%llu bytes=%llu\n",i,(unsigned long long)c.attempts,(unsigned long long)c.completions,(unsigned long long)c.successes,(unsigned long long)c.failures,(unsigned long long)c.requested_bytes);}
 for(unsigned i=0;i<unsigned(HipCall::count);i++){const auto& c=storage.calls[i];require(c.attempts==c.completions&&c.completions==c.successes&&!c.failures);std::printf("M2_DENSE STORAGE kind=%u attempts=%llu completions=%llu successes=%llu failures=%llu bytes=%llu\n",i,(unsigned long long)c.attempts,(unsigned long long)c.completions,(unsigned long long)c.successes,(unsigned long long)c.failures,(unsigned long long)c.requested_bytes);}
 require(calls[unsigned(ps5hip_prefix::Call::module_function)].successes==22&&calls[unsigned(ps5hip_prefix::Call::release_sync)].successes==1&&storage.calls[unsigned(HipCall::allocate)].successes==14&&storage.calls[unsigned(HipCall::free)].successes==14&&storage.calls[unsigned(HipCall::d2h)].requested_bytes==154392576);
 // Release host staging buffers before observing restored memory.
 std::vector<ggml_fp16_t>().swap(half);std::vector<float>().swap(values);
 std::puts("M2_DENSE PHASE cleanup_complete");if(snapshot)snapshot("after_dense_cleanup");
 std::printf("M2_DENSE COMPUTE_COMPLETE requests=526 records=11084 bytes=154392576 launches=17358 numerical=pending synthetic=true finalization=pending\n");require(log_sync());
 return {state,calls,storage,requests,records,bytes};
}
}
