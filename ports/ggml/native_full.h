#pragma once
#include "native_prefix.h"
#include "full_qualification.h"
#include <vector>
#include <string>
#include <time.h>
#include "generation.h"
#include "extended_generation.h"
#ifdef PS5HIP_NATIVE_ORIGINAL_MODEL
#include "tinyllama_source_metadata.h"
#include "tinyllama_tensor_callback.h"
#endif

namespace m0 {
enum class NativeFullProfile { sequence, smoke, generation, corpus, extended };
static_assert(unsigned(NativeFullProfile::extended)==4);
enum class NativeFullModelSource { gguf, original_safetensors };
struct NativeGenerationCase {const char* id;const char* prompt;};
inline constexpr NativeGenerationCase native_generation_cases[]={
#define PS5HIP_GENERATION_CASE(id,prompt) {id,prompt},
#include "generation_cases.def"
#undef PS5HIP_GENERATION_CASE
};
static_assert(sizeof(native_generation_cases)/sizeof(native_generation_cases[0])==5);
struct NativeFullResult {FullStatistics full;unsigned files;size_t bytes;};
struct NativeFullClock {
 uint64_t previous=0;std::string case_tag;
 uint64_t now(){timespec t{};require(clock_gettime(CLOCK_MONOTONIC,&t)==0&&t.tv_sec>=0&&t.tv_nsec>=0&&t.tv_nsec<1000000000L&&uint64_t(t.tv_sec)<=UINT64_MAX/1000000000ULL-1);
  const uint64_t value=uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec);require(value>=previous);previous=value;return value;}
 void report(const char* scope,uint64_t begin,unsigned run=0,unsigned position=0){const auto end=now();std::printf("M4_GEN TIMING%s scope=%s run=%u position=%u begin_ns=%llu end_ns=%llu elapsed_ns=%llu\n",case_tag.c_str(),scope,run,position,(unsigned long long)begin,(unsigned long long)end,(unsigned long long)(end-begin));}
};
struct NativeFullSink {
 PrefixWrite write;void* user;PrefixLogSync sync;unsigned run=0,position=0,files=0;size_t bytes=0;
 const char* case_id=nullptr;
 const ps5hip_generation::ExtendedCase* extended=nullptr;
 void output(const char* name,const void* data,size_t size){
  if(!case_id){write(name,data,size,user);return;}
  char path[128];const int n=std::snprintf(path,sizeof(path),"%s/%s",case_id,name);require(n>0&&size_t(n)<sizeof(path));write(path,data,size,user);
 }
 bool all=false;unsigned seen=0;int last_boundary=-1;NativeFullClock* clock=nullptr;uint64_t readback_ns=0;
 std::vector<float> values=std::vector<float>(32000);
 bool checkpoint() const{return all||run||position==0||position==1||position==255||position==256||position==511;}
 bool capture(int index) const{return index>=0&&(extended?(index==23||ps5hip_generation::extended_checkpoint(*extended,position)):checkpoint());}
 static int boundary(const char* name){
  for(unsigned l=0;l<22;++l){char expected[24];std::snprintf(expected,sizeof(expected),"l_out-%u",l);if(!std::strcmp(name,expected))return int(l);}
  if(!std::strcmp(name,"result_norm"))return 22;
  if(!std::strcmp(name,"result_output"))return 23;
  return -1;
 }
 static bool callback(ggml_tensor* t,bool ask,void* opaque){
  auto& s=*static_cast<NativeFullSink*>(opaque);const int index=boundary(t->name);
  if(ask)return s.capture(index);
  require(s.capture(index)&&t->type==GGML_TYPE_F32&&ggml_is_contiguous(t));
  const uint64_t began=s.clock?s.clock->now():0;
  if(s.all){require(!(s.seen&(1u<<index)));s.seen|=1u<<index;s.last_boundary=index;}
  const size_t size=ggml_nbytes(t);require(size==(index==23?128000:8192)&&size<=s.values.size()*sizeof(float));
  ggml_backend_tensor_get(t,s.values.data(),0,size);for(size_t i=0;i<size/4;++i)require(std::isfinite(s.values[i]));
  char stage[24];if(index<22)std::snprintf(stage,sizeof(stage),"layer-%d",index);else std::snprintf(stage,sizeof(stage),"%s",index==22?"norm":"logits");
  char name[96];const int n=std::snprintf(name,sizeof(name),"run-%u-pos-%u-%s.f32le",s.run,s.position,stage);require(n>0&&size_t(n)<sizeof(name));
  s.output(name,s.values.data(),size);++s.files;s.bytes+=size;
  std::printf("M3_FULL RAW file=%s%s%s bytes=%zu synced=true\n",s.case_id?s.case_id:"",s.case_id?"/":"",name,size);require(s.sync());
  if(s.clock){const auto elapsed=s.clock->now()-began;require(elapsed<=UINT64_MAX-s.readback_ns);s.readback_ns+=elapsed;}
  return true;
 }
};
struct NativeFullGenerationData {unsigned requests;size_t metadata_bytes;};
inline NativeFullGenerationData run_full_generation(llama_model* model,llama_context* context,NativeFullSink& sink,NativeFullClock& clock,ModelGateLog& logs,const NativeGenerationCase* selected=nullptr,const ps5hip_generation::ExtendedCase* extended=nullptr){
 unsigned actual_requests=0;size_t metadata_bytes=0;const auto log_sync=sink.sync;
 require(!(selected&&extended));sink.extended=extended;
 const unsigned budget=extended?unsigned(extended->budget):selected?16:4;sink.case_id=extended?extended->id:selected?selected->id:nullptr;
 if(selected){require(*selected->id);for(const char* c=selected->id;*c;++c)require(*c>='a'&&*c<='z');}
 clock.case_tag=sink.case_id?std::string(" case=")+sink.case_id:"";
  sink.all=true;sink.clock=&clock;const auto* vocab=llama_model_get_vocab(model);require(vocab&&llama_vocab_n_tokens(vocab)==32000);
  const char* prompt=extended?extended->prompt:selected?selected->prompt:"Once upon a time";int32_t prompt_tokens[ps5hip_generation::context_size];const int count=llama_tokenize(vocab,prompt,int(std::strlen(prompt)),prompt_tokens,int(ps5hip_generation::context_size),true,false);require(count>0&&size_t(count)<=ps5hip_generation::context_size-budget);require(!extended||size_t(count)==extended->prompt_count);
  auto ids=[](const int32_t* tokens,size_t n){std::string text="[";for(size_t i=0;i<n;++i){if(i)text+=',';text+=std::to_string(tokens[i]);}return text+"]";};
  auto hex=[](const char* bytes,size_t n){static const char* digits="0123456789abcdef";std::string text;for(size_t i=0;i<n;++i){const auto c=static_cast<unsigned char>(bytes[i]);text+=digits[c>>4];text+=digits[c&15];}return text;};
  std::string escaped_prompt;for(const char* c=prompt;*c;++c){if(*c=='\n')escaped_prompt+="\\n";else{if(*c=='"'||*c=='\\')escaped_prompt+='\\';escaped_prompt+=*c;}}
  std::string evidence="{\"schema\":1,\"prompt\":\""+escaped_prompt+"\",\"add_special\":true,\"parse_special\":false,\"max_new_tokens\":"+std::to_string(budget)+",\"vocab_size\":32000,\"prompt_tokens\":"+ids(prompt_tokens,size_t(count))+",\"eog_tokens\":[";bool first_eog=true;
  for(int token=0;token<32000;++token)if(llama_vocab_is_eog(vocab,token)){if(!first_eog)evidence+=',';evidence+=std::to_string(token);first_eog=false;}evidence+="],\"runs\":[";
  std::string steps;ps5hip_generation::Result first;
  for(unsigned run=0;run<2;++run){
   sink.run=run;auto began=clock.now();llama_synchronize(context);llama_memory_clear(llama_get_memory(context),true);reset_full();require(llama_memory_seq_pos_max(llama_get_memory(context),0)==-1);clock.report("reset",began,run);
   began=clock.now();uint64_t decode_ns=0;
   auto decode=[&](int32_t token,int32_t position)->ps5hip_generation::Logits{
    sink.position=unsigned(position);sink.seen=0;sink.last_boundary=-1;const auto before=sink.files;const auto before_readback=sink.readback_ns;const auto begin=clock.now();
    auto* batch=llama_batch_ext_init(context);require(batch);const int index=llama_batch_ext_add_token(batch,0,token);const llama_pos pos=position;
    require(index>=0&&llama_batch_ext_set_pos(batch,index,&pos)&&llama_batch_ext_set_output_logits(batch,index,true));const int status=llama_process(context,LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);
    require(status==0&&!logs.unexpected.load()&&llama_memory_seq_pos_min(llama_get_memory(context),0)==0&&llama_memory_seq_pos_max(llama_get_memory(context),0)==position);
    const bool intermediates=!extended||ps5hip_generation::extended_checkpoint(*extended,unsigned(position));
    require(sink.files-before==(intermediates?24u:1u)&&sink.seen==(intermediates?0xffffffu:1u<<23)&&sink.last_boundary==23);const float* logits=llama_get_logits_ith(context,-1);require(logits&&std::memcmp(logits,sink.values.data(),128000)==0);
    const auto end=clock.now(),elapsed=end-begin,readback=sink.readback_ns-before_readback;require(readback<=elapsed&&elapsed<=UINT64_MAX-decode_ns);decode_ns+=elapsed;
    std::printf("M4_GEN DECODE%s run=%u position=%d token=%d committed_position=%d layers=22 logits=32000 begin_ns=%llu end_ns=%llu elapsed_ns=%llu readback_ns=%llu inclusive_readback=true\n",clock.case_tag.c_str(),run,position,token,position,(unsigned long long)begin,(unsigned long long)end,(unsigned long long)elapsed,(unsigned long long)readback);require(log_sync());
    steps+="{\"run\":"+std::to_string(run)+",\"position\":"+std::to_string(position)+",\"token\":"+std::to_string(token)+",\"status\":0,\"committed_position\":"+std::to_string(position)+"}\n";++actual_requests;return {logits,32000};
   };
   const auto result=ps5hip_generation::generate(prompt_tokens,size_t(count),budget,decode,[&](int32_t token){return llama_vocab_is_eog(vocab,token);});
   require(!selected||(result.generated_count==16&&result.stop==ps5hip_generation::StopReason::step_limit));
   require(!extended||(result.generated_count==extended->generated&&std::strcmp(extended->stop,result.stop==ps5hip_generation::StopReason::eos?"eos":"step_limit")==0));
   const auto end=clock.now();require(end-began>=decode_ns);std::printf("M4_GEN RUN_TIME%s run=%u begin_ns=%llu end_ns=%llu elapsed_ns=%llu decode_ns=%llu scheduling_sampling_log_ns=%llu\n",clock.case_tag.c_str(),run,(unsigned long long)began,(unsigned long long)end,(unsigned long long)(end-began),(unsigned long long)decode_ns,(unsigned long long)(end-began-decode_ns));require(log_sync());
   if(!run)first=result;else require(result.processed==first.processed&&result.generated==first.generated&&result.processed_count==first.processed_count&&result.generated_count==first.generated_count&&result.stop==first.stop);
   if(run)evidence+=',';evidence+="{\"run\":"+std::to_string(run)+",\"processed\":"+ids(result.processed.data(),result.processed_count)+",\"generated\":"+ids(result.generated.data(),result.generated_count)+",\"stop\":\""+(result.stop==ps5hip_generation::StopReason::eos?"eos":"step_limit")+"\",\"pieces_hex\":[";
   std::string text;for(size_t i=0;i<result.generated_count;++i){char piece[4096];const int n=llama_token_to_piece(vocab,result.generated[i],piece,sizeof(piece),0,true);require(n>=0&&size_t(n)<=sizeof(piece));const auto encoded=hex(piece,size_t(n));if(i)evidence+=',';evidence+='\"';evidence+=encoded;evidence+='\"';text+=encoded;}
   evidence+="],\"text_hex\":\""+text+"\"}";std::printf("M4_GEN RESULT%s run=%u processed=%zu generated=%zu stop=%s text_hex=%s\n",clock.case_tag.c_str(),run,result.processed_count,result.generated_count,result.stop==ps5hip_generation::StopReason::eos?"eos":"step_limit",text.c_str());require(log_sync());
  }
  evidence+="]}\n";sink.output("generation.json",evidence.data(),evidence.size());sink.output("steps.jsonl",steps.data(),steps.size());metadata_bytes=evidence.size()+steps.size();
  std::printf("M4_GEN METADATA%s files=2 generation_bytes=%zu steps_bytes=%zu synced=true\n",clock.case_tag.c_str(),evidence.size(),steps.size());require(log_sync());
 return {actual_requests,metadata_bytes};
}
inline NativeFullResult run_native_full(const char* model_path,const std::array<const char*,6>& modules,
 PrefixWrite writer,void* writer_state,PrefixLogSync log_sync,void (*memory_snapshot)(const char*)=nullptr,
 NativeFullProfile profile=NativeFullProfile::sequence,NativeFullModelSource model_source=NativeFullModelSource::gguf){
 require(model_path&&writer&&log_sync&&(profile==NativeFullProfile::sequence||profile==NativeFullProfile::smoke||profile==NativeFullProfile::generation||profile==NativeFullProfile::corpus||profile==NativeFullProfile::extended));
 require(model_source==NativeFullModelSource::gguf||model_source==NativeFullModelSource::original_safetensors);
#ifndef PS5HIP_NATIVE_ORIGINAL_MODEL
 require(model_source==NativeFullModelSource::gguf);
#endif
 const bool extended=profile==NativeFullProfile::extended;
 const bool smoke=profile==NativeFullProfile::smoke,generation=extended||profile==NativeFullProfile::generation||profile==NativeFullProfile::corpus,corpus=profile==NativeFullProfile::corpus;
 NativeFullClock clock;uint64_t phase_begin=generation?clock.now():0;
 const unsigned requests=smoke?3:514,raw_files=smoke?72:168;
 const size_t raw_bytes=smoke?949248:2214912;
 const uint64_t nodes=smoke?2394:410172,launches=smoke?2199:376762;
 if(extended)std::puts("M4_GEN PROFILE name=greedy-extended cases=3 runs=2 max_new_tokens=64 context=512");
 else if(corpus)std::puts("M4_GEN PROFILE name=greedy-corpus cases=5 runs=2 max_new_tokens=16 context=512");
 else if(generation)std::puts("M4_GEN PROFILE name=greedy-short runs=2 max_new_tokens=4 context=512");
 else std::printf("M3_FULL PROFILE name=%s requests=%u raw_files=%u raw_bytes=%zu\n",smoke?"smoke":"sequence",requests,raw_files,raw_bytes);require(log_sync());
 std::puts("M3_FULL PHASE load_begin");if(memory_snapshot)memory_snapshot("before_load");
 ModelGateLog logs;llama_log_set(ModelGateLog::callback,&logs);llama_backend_init();ggml_backend_register(registration());enable_full(modules);
 auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.use_extra_bufts=false;mp.check_tensors=true;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_OFF;
 ggml_backend_dev_t devices[]={device_handle(),nullptr};mp.devices=devices;
 llama_model_tensor_buft_override overrides[]={{"^token_embd\\.weight$",ggml_backend_dev_buffer_type(devices[0])},{nullptr,nullptr}};mp.tensor_buft_overrides=overrides;
 // Keep source descriptors and callback state alive through model destruction.
#ifdef PS5HIP_NATIVE_ORIGINAL_MODEL
 direct_tiny::Metadata original_metadata(nullptr,gguf_free);
 std::unique_ptr<ps5hip::SafetensorsReader> original_reader;
 std::unique_ptr<ps5hip::TinyLlamaTensorCallback> original_callback;
#endif
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(nullptr,llama_model_free);
#ifdef PS5HIP_NATIVE_ORIGINAL_MODEL
 if(model_source==NativeFullModelSource::original_safetensors){
  original_metadata=direct_tiny::construct(model_path);
  const auto config=direct_tiny::json(direct_tiny::read(std::filesystem::path(model_path)/"config.json",65536));
  original_reader=std::make_unique<ps5hip::SafetensorsReader>((std::filesystem::path(model_path)/"model.safetensors").string());
  original_callback=std::make_unique<ps5hip::TinyLlamaTensorCallback>(*original_reader,config);
  original_callback->add_descriptors(original_metadata.get());
  model.reset(llama_model_init_from_user(original_metadata.get(),ps5hip::TinyLlamaTensorCallback::callback,original_callback.get(),mp));
  require(bool(model)&&original_callback->complete());
 }else{
#endif
  model.reset(llama_model_load_from_file(model_path,mp));
#ifdef PS5HIP_NATIVE_ORIGINAL_MODEL
 }
#endif
 require(bool(model)&&!logs.unexpected.load());
 if(generation){clock.report("load",phase_begin);phase_begin=clock.now();}
 std::puts("M3_FULL PHASE load_complete");if(memory_snapshot)memory_snapshot("after_load");
 NativeFullSink sink{writer,writer_state,log_sync};auto cp=llama_context_default_params();cp.n_ctx=512;cp.n_batch=cp.n_ubatch=cp.n_seq_max=1;cp.n_threads=cp.n_threads_batch=4;
 cp.type_k=cp.type_v=GGML_TYPE_F16;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=true;cp.cb_eval=NativeFullSink::callback;cp.cb_eval_user_data=&sink;
 std::unique_ptr<llama_context,decltype(&llama_free)> context(llama_init_from_model(model.get(),cp),llama_free);require(bool(context)&&!logs.unexpected.load());
 if(generation)clock.report("context",phase_begin);
 std::puts("M3_FULL PHASE context_complete");if(memory_snapshot)memory_snapshot("after_context");reset_statistics();
 unsigned actual_requests=0,metadata_files=0;size_t metadata_bytes=0;
 if(generation){
  for(unsigned i=0;i<(extended?3u:corpus?5u:1u);++i){
   const auto generated=run_full_generation(model.get(),context.get(),sink,clock,logs,corpus?&native_generation_cases[i]:nullptr,extended?&ps5hip_generation::extended_cases[i]:nullptr);actual_requests+=generated.requests;metadata_files+=2;metadata_bytes+=generated.metadata_bytes;
  }
  sink.case_id=nullptr;sink.extended=nullptr;clock.case_tag.clear();
  if(extended){constexpr char index[]=R"({"schema":1,"profile":"extended","max_new_tokens":64,"cases":["short64","eos","boundary"],"capture":"logits-all-checkpoints-v1"})" "\n";sink.output("extended.json",index,sizeof(index)-1);++metadata_files;metadata_bytes+=sizeof(index)-1;std::puts("M4_GEN EXTENDED cases=3 runs=2 max_new_tokens=64 synced=true");require(log_sync());}
  if(corpus){std::string index="{\"schema\":1,\"profile\":\"corpus\",\"max_new_tokens\":16,\"cases\":[";for(unsigned i=0;i<5;++i){if(i)index+=',';index+='\"';index+=native_generation_cases[i].id;index+='\"';}index+="]}\n";sink.output("corpus.json",index.data(),index.size());++metadata_files;metadata_bytes+=index.size();std::printf("M4_GEN CORPUS cases=5 runs=2 max_new_tokens=16 index_bytes=%zu synced=true\n",index.size());require(log_sync());}
 }else{
 constexpr int32_t tokens[]={0,1,2,31999};
 for(unsigned run=0;run<2;++run){
  sink.run=run;if(run){llama_synchronize(context.get());llama_memory_clear(llama_get_memory(context.get()),true);reset_full();}
  require(llama_memory_seq_pos_max(llama_get_memory(context.get()),0)==-1);
  for(unsigned p=0;p<(smoke?(run?1u:2u):(run?2u:512u));++p){
   sink.position=p;auto* batch=llama_batch_ext_init(context.get());require(batch);
   const int index=llama_batch_ext_add_token(batch,0,tokens[p%4]);const llama_pos position=p;
   require(index>=0&&llama_batch_ext_set_pos(batch,index,&position)&&llama_batch_ext_set_output_logits(batch,index,true));
   const unsigned before=sink.files;const int result=llama_process(context.get(),LLAMA_PROCESS_TYPE_DECODE,batch);llama_batch_ext_free(batch);
   require(result==0&&!logs.unexpected.load()&&llama_memory_seq_pos_min(llama_get_memory(context.get()),0)==0&&llama_memory_seq_pos_max(llama_get_memory(context.get()),0)==position);
   require(sink.files-before==(sink.checkpoint()?24u:0u));const float* logits=llama_get_logits_ith(context.get(),-1);require(logits);for(unsigned i=0;i<32000;++i)require(std::isfinite(logits[i]));
   if(sink.checkpoint()){std::printf("M3_FULL CHECKPOINT run=%u position=%u token=%d committed_position=%u layers=22 logits=32000\n",run,p,tokens[p%4],p);require(log_sync());}
  }
 }
 }
 const auto gate=statistics();const unsigned completed=generation?actual_requests:requests;
 require(gate.calls==completed&&gate.forwarded==completed&&gate.rejected==0);
 require(!extended||(completed==1176&&metadata_files==7));
 require(sink.files==(extended?2096:generation?completed*24:raw_files)&&sink.bytes==(extended?158064640:generation?size_t(completed)*316416:raw_bytes));
 if(generation)phase_begin=clock.now();std::puts("M3_FULL PHASE cleanup_begin");context.reset();model.reset();const auto state=full_statistics();
 require(state.requests==completed&&state.completed_nodes==(generation?uint64_t(completed)*798:nodes)&&state.launches==(generation?uint64_t(completed)*733:launches)&&state.instances_created>0&&state.instances_created==state.instances_freed);
 require(!extended||state.instances_created==1);
 disable_full();llama_backend_free();llama_log_set(nullptr,nullptr);require(!logs.unexpected.load());if(memory_snapshot)memory_snapshot("after_cleanup");
 if(generation){clock.report("cleanup",phase_begin);std::printf("M4_GEN TRANSPORT_COMPLETE requests=%u layers=22 logits=32000 nodes=%llu launches=%llu raw_files=%u raw_bytes=%zu metadata_files=%u metadata_bytes=%zu instances=%u/%u numerical=pending finalization=pending\n",completed,(unsigned long long)state.completed_nodes,(unsigned long long)state.launches,sink.files,sink.bytes,metadata_files,metadata_bytes,state.instances_created,state.instances_freed);}
 else std::printf("M3_FULL TRANSPORT_COMPLETE requests=%u layers=22 logits=32000 nodes=%llu launches=%llu raw_files=%u raw_bytes=%zu instances=%u/%u numerical=pending inference=false finalization=pending\n",requests,static_cast<unsigned long long>(nodes),static_cast<unsigned long long>(launches),sink.files,sink.bytes,state.instances_created,state.instances_freed);require(log_sync());
 return {state,sink.files+metadata_files,sink.bytes+metadata_bytes};
}
}
