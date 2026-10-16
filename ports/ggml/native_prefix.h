#pragma once
// Separate phase helper: existing reject_actual_model semantics are unchanged.
#include "model_gate.h"
#include "prefix_qualification.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>

namespace m0 {
struct NativePrefixResult {
 PrefixStatistics prefix;
 PrefixCallStatistics calls;
 unsigned files=0;
 size_t raw_bytes=0;
};
// Title supplies its existing exclusive/fsync writer and log-sync helper.
using PrefixWrite=void (*)(const char* filename,const void* bytes,size_t size,void*);
using PrefixLogSync=bool (*)();
struct NativePrefixSink {
 PrefixWrite write;void* user;PrefixLogSync sync;
 int32_t expected=-1;unsigned files=0;size_t bytes=0;
 alignas(16) std::array<float,2048> storage{};
 static void readback(ggml_tensor* const* tensors,size_t count,int32_t token,void* pointer){
  auto& state=*static_cast<NativePrefixSink*>(pointer);require(count==6&&token==state.expected);
  const char* names[]={"embedding","normalized","weighted","q","k","v"};
  for(size_t i=0;i<count;++i){
   auto* tensor=tensors[i];const size_t values=i<4?2048:256;
   require(tensor&&tensor->type==GGML_TYPE_F32&&ggml_nelements(tensor)==int64_t(values)&&ggml_is_contiguous(tensor));
   ggml_backend_tensor_get(tensor,state.storage.data(),0,values*sizeof(float));
   for(size_t j=0;j<values;++j)require(std::isfinite(state.storage[j]));
   char name[96];const int length=std::snprintf(name,sizeof(name),"token-%d-%s.f32le",int(token),names[i]);require(length>0&&size_t(length)<sizeof(name));
   state.write(name,state.storage.data(),values*sizeof(float),state.user);
   ++state.files;state.bytes+=values*sizeof(float);
   std::printf("M2_PREFIX RAW token=%d tensor=%s file=%s bytes=%zu exclusive=true synced=true\n",int(token),names[i],name,values*sizeof(float));
   require(state.sync());
  }
 }
};
inline NativePrefixResult run_native_prefix(const char* model_path,const std::array<const char*,5>& modules,
 PrefixWrite writer,void* writer_state,PrefixLogSync log_sync,void (*memory_snapshot)(const char*)=nullptr){
 require(model_path&&writer&&log_sync);
 std::puts("M2_PREFIX PHASE load_begin");if(memory_snapshot)memory_snapshot("before_load");
 ModelGateLog logs;llama_log_set(ModelGateLog::callback,&logs);
 llama_backend_init();ggml_backend_register(registration());
 auto backend=ggml_backend_dev_init(device_handle(),nullptr);require(backend);
 configure_prefix(backend,modules);
 auto mp=llama_model_default_params();mp.n_gpu_layers=0;mp.use_extra_bufts=false;mp.check_tensors=true;
 mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
 ggml_backend_dev_t devices[]={device_handle(),nullptr};mp.devices=devices;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(model_path,mp),llama_model_free);require(bool(model)&&!logs.unexpected.load());
 std::puts("M2_PREFIX PHASE load_complete");if(memory_snapshot)memory_snapshot("after_load");
 std::puts("M2_PREFIX PHASE context_begin");
 auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;
 cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 cp.offload_kqv=false;cp.op_offload=false;cp.no_perf=false;
 unsigned completed=0;cp.cb_eval=count_completed;cp.cb_eval_user_data=&completed;
 std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);require(bool(context)&&!logs.unexpected.load());
 std::puts("M2_PREFIX PHASE context_complete");if(memory_snapshot)memory_snapshot("after_context");
 NativePrefixSink sink{writer,writer_state,log_sync};
 const ggml_cgraph* first=nullptr;unsigned attempt=0;
 for(int32_t token:{0,1,2,31999}){
  auto* batch=llama_batch_ext_init(context.get());require(batch);const int index=llama_batch_ext_add_token(batch,0,token);const llama_pos pos=0;
  require(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true));
  sink.expected=token;completed=0;reset_statistics();const int reused_before=llama_perf_context(context.get()).n_reused;
  const auto prefix_before=prefix_statistics(backend);logs.begin();int result;
  {ActualPrefixQualification qualification(backend,NativePrefixSink::readback,&sink);result=llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch);}
  logs.finish();llama_batch_ext_free(batch);
  const auto gate=statistics();const auto prefix_after=prefix_statistics(backend);const int reused_after=llama_perf_context(context.get()).n_reused;
  require(result==-3&&completed==0&&gate.calls==1&&gate.forwarded==0&&gate.rejected==1&&gate.direct_graph_calls==1&&gate.last_graph);
  require(prefix_after.invocations==prefix_before.invocations+1&&prefix_after.completed_launches==prefix_before.completed_launches+9&&prefix_after.uploaded_leaf_bytes==prefix_before.uploaded_leaf_bytes+39821316&&prefix_after.last_token==token);
  const bool reused=attempt&&gate.last_graph==first&&reused_after==reused_before+1;
  if(attempt==0){first=gate.last_graph;require(reused_after==reused_before);}else require(reused);
  std::printf("M2_PREFIX ACTUAL_CLIENT attempt=%u token=%d position=0 result=-3 calls=1 rejected=1 forwarded=0 post_compute=0 expected_errors=2 graph_reused=%s prefix_nodes=6 launches=9 uploaded_leaf_bytes=39821316 inference=false\n",attempt,int(token),reused?"true":"false");require(log_sync());++attempt;
 }
 const auto prefix=prefix_statistics(backend);require(prefix.invocations==4&&prefix.completed_launches==36&&prefix.uploaded_leaf_bytes==159285264&&sink.files==24&&sink.bytes==139264);
 std::puts("M2_PREFIX PHASE cleanup_begin");context.reset();model.reset();
 const auto calls=disable_prefix(backend);ggml_backend_free(backend);llama_backend_free();llama_log_set(nullptr,nullptr);require(!logs.unexpected.load());
 const char* sites[]={"module_load","module_function","scratch_allocate","token_read","launch","completion_sync","release_sync","scratch_free","module_unload"};
 const uint64_t expected[]={5,5,1,8,36,36,1,1,5};
 for(size_t i=0;i<calls.size();++i){const auto& row=calls[i];const uint64_t bytes=i==2||i==7?2304:i==3?32:0;
  require(row.attempts==expected[i]&&row.completions==expected[i]&&row.successes==expected[i]&&!row.failures&&row.requested_bytes==bytes&&!row.first_error&&!row.last_result);
  std::printf("M2_PREFIX HIP_CALL site=%s attempts=%llu completions=%llu successes=%llu failures=0 bytes=%llu last_result=0 first_error=0\n",sites[i],static_cast<unsigned long long>(row.attempts),static_cast<unsigned long long>(row.completions),static_cast<unsigned long long>(row.successes),static_cast<unsigned long long>(row.requested_bytes));
 }
 std::puts("M2_PREFIX PHASE cleanup_complete");if(memory_snapshot)memory_snapshot("after_cleanup");
 std::puts("M2_PREFIX TRANSPORT_COMPLETE tokens=4 prefix_nodes=6 launches=36 raw_files=24 raw_bytes=139264 uploaded_leaf_bytes=159285264 modules_unloaded=5 scratch_allocations=1 scratch_frees=1 numerical_verification=pending inference=false finalization=pending");require(log_sync());
 return {prefix,calls,sink.files,sink.bytes};
}
}
