#include "ps5-device.hpp"
#include "ps5hip/agc_backend.h"
#include "clr_gpu/watchdog.h"
#include "native_runtime.h"
#include "elevation.hpp"
#include <hip/hip_runtime_api.h>
#include <ps5platform/libc.h>
#include <ps5platform/kernel.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ps5_backend.h"
#include "model_gate.h"
#include "native_host_allocator.h"
#include <sys/stat.h>
#include <cstring>
#include <vector>

extern "C" int __cxa_thread_atexit_impl(void (*destructor)(void*), void* object,
                                          void* dso) {
  return ps5___cxa_thread_atexit_impl(destructor, object, dso);
}

namespace {
constexpr const char* title_root = "/system_ex/app/PPSA99014/";
ps5hip::clr::Observation observed;
bool finalizer_installed = false;
bool completed = false;
std::unique_ptr<ps5hip::SubmissionBackend> backend() {
  return std::make_unique<ps5hip::AgcBackend>();
}
void finalized() {
#if PS5HIP_LLM_AUTHENTICATED_MODEL_GATE
  const auto host = ps5hip::llm::native_host_allocator_stats();
  if (host.poisoned || host.failures || host.live_allocations ||
      host.live_requested_bytes || host.allocations != host.releases) {
    std::puts("M0_APP FAIL finalizer-host-storage");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
#endif
  if (!completed || observed.terminal ||
      observed.allocations != observed.frees ||
      observed.queues != observed.retired_queues ||
      !observed.gate_destroyed) {
    std::puts("M0_APP FAIL finalizer");
    (void)ps5hip_canary::sync_log();
    for (;;) ::sleep(1);
  }
  std::printf("M0_APP FINAL allocations=%u frees=%u queues=%u retired=%u gate_destroyed=true application_fini=pending\n",
              unsigned(observed.allocations), unsigned(observed.frees),
              unsigned(observed.queues), unsigned(observed.retired_queues));
  if (!ps5hip_canary::sync_log()) for (;;) ::sleep(1);
}
}

extern "C" bool ps5hip_clr_admit_device() {
  return ps5hip::clr::register_runtime_device(backend, observed);
}

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) {
  std::fprintf(stderr, "M0_APP QUARANTINED reason=%s no_shell_request=true\n", reason);
  (void)ps5hip_canary::sync_log();
  for (;;) ::sleep(1);
}

extern "C" void ps5hip_install_vector_title_finalizer() {
  finalizer_installed = std::atexit(finalized) == 0;
}

static void vocabulary_check() {
 // Tokenizer flag tests intentionally trigger BOS warnings; retain loader errors.
 llama_log_set([](ggml_log_level level,const char* text,void*) {
  if(level==GGML_LOG_LEVEL_ERROR)std::fprintf(stderr,"M0_APP LLAMA_ERROR %s",text);
 },nullptr);
 llama_backend_init(); auto params=llama_model_default_params(); params.vocab_only=true; params.n_gpu_layers=0; params.use_extra_bufts=false;
 auto* model=llama_model_load_from_file((std::string(title_root)+"vocab.gguf").c_str(),params);
 if(!model)ps5hip_clr_gpu_fail("vocab-load");
 const auto* vocab=llama_model_get_vocab(model);
 if(!vocab || llama_vocab_n_tokens(vocab)!=32000)ps5hip_clr_gpu_fail("vocab-size");
 const char*texts[]={"","Hello, world!","The capital of France is","  spaces\nnew line\t tab","café Ελληνικά 日本語 😀","<s>hello</s>","<unk><s></s>","[INST] Say hello [/INST]"};
 const std::vector<std::vector<llama_token>> expected={{},{},{1},{1},{15043,29892,3186,29991},{15043,29892,3186,29991},{1,15043,29892,3186,29991},{1,15043,29892,3186,29991},{450,7483,310,3444,338},{450,7483,310,3444,338},{1,450,7483,310,3444,338},{1,450,7483,310,3444,338},{259,8162,13,1482,1196,12,4434},{259,8162,13,1482,1196,12,4434},{1,259,8162,13,1482,1196,12,4434},{1,259,8162,13,1482,1196,12,4434},{274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{1,274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{1,274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{529,29879,29958,12199,829,29879,29958},{1,22172,2},{1,529,29879,29958,12199,829,29879,29958},{1,1,22172,2},{529,2960,5299,29879,2565,29879,29958},{0,1,2},{1,529,2960,5299,29879,2565,29879,29958},{1,0,1,2},{518,25580,29962,14891,22172,518,29914,25580,29962},{518,25580,29962,14891,22172,518,29914,25580,29962},{1,518,25580,29962,14891,22172,518,29914,25580,29962},{1,518,25580,29962,14891,22172,518,29914,25580,29962}};
 unsigned case_index=0,total=0;
 for(const char* text:texts) for(int special=0;special<2;++special) for(int parse=0;parse<2;++parse) {
  llama_token tokens[256]; const int n=llama_tokenize(vocab,text,int(std::strlen(text)),tokens,256,bool(special),bool(parse));
  if(n<0 || size_t(n)!=expected[case_index].size()) {
   std::fprintf(stderr,"M0_APP VOCAB_MISMATCH case=%u count=%d expected=%u\n",case_index,n,unsigned(expected[case_index].size()));
   ps5hip_clr_gpu_fail("vocab-token-count");
  }
  for(int i=0;i<n;++i)if(tokens[i]!=expected[case_index][i]) {
   std::fprintf(stderr,"M0_APP VOCAB_MISMATCH case=%u token=%d actual=%d expected=%d\n",case_index,i,int(tokens[i]),int(expected[case_index][i]));
   ps5hip_clr_gpu_fail("vocab-token-value");
  }
  total+=unsigned(n); ++case_index;
 }
 m0::require(case_index==32); llama_model_free(model); llama_backend_free();
 std::printf("M0_APP VOCAB cases=%u tokens=%u vocabulary=32000 tensors=0 exact=true\n",case_index,total);
}
static void storage_call_snapshot() {
 const auto counters=m0::hip_call_statistics();
 const char* names[]={"init","device_count","allocate","free","cleanup_free","h2d","d2h","d2d","fill","clear","synchronize"};
 const uint64_t counts[]={1,1,1,1,0,1,3,1,1,1,1};
 const uint64_t bytes[]={0,0,512,512,0,64,192,64,16,512,0};
 static_assert(sizeof(names)/sizeof(names[0])==static_cast<unsigned>(m0::HipCall::count));
 for(unsigned i=0;i<static_cast<unsigned>(m0::HipCall::count);++i){
  const auto& row=counters.calls[i];
  m0::require(row.attempts==counts[i]&&row.completions==counts[i]&&row.successes==counts[i]&&
              !row.failures&&row.requested_bytes==bytes[i]&&!row.last_result&&!row.first_error);
  std::printf("M0_APP HIP_CALL scope=storage_fixture call=%s attempts=%llu completions=%llu successes=%llu failures=%llu requested_bytes=%llu last_result=%d first_error=%d\n",
              names[i],static_cast<unsigned long long>(row.attempts),static_cast<unsigned long long>(row.completions),
              static_cast<unsigned long long>(row.successes),static_cast<unsigned long long>(row.failures),
              static_cast<unsigned long long>(row.requested_bytes),row.last_result,row.first_error);
 }
}
static void storage_and_gate_check() {
 const auto before=m0::hip_call_statistics();
 for(const auto& row:before.calls)m0::require(!row.attempts&&!row.completions&&!row.successes&&!row.failures&&!row.requested_bytes&&!row.last_result&&!row.first_error);
 ggml_backend_register(m0::registration());auto ps5=ggml_backend_dev_init(ggml_backend_reg_dev_get(m0::registration(),0),nullptr);m0::require(ps5);
 ggml_init_params params{1<<20,nullptr,true};auto ctx=ggml_init(params);m0::require(ctx);
 auto a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);auto b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);
 auto storage=ggml_backend_alloc_ctx_tensors(ctx,ps5);m0::require(storage);
 float input[16],output[16];for(unsigned i=0;i<16;++i)input[i]=float(i+1);
 ggml_backend_tensor_set(a,input,0,sizeof(input));ggml_backend_tensor_copy(a,b);ggml_backend_tensor_get(b,output,0,sizeof(output));m0::require(!std::memcmp(input,output,sizeof(input)));
 ggml_backend_tensor_memset(b,0xa5,8,16);ggml_backend_tensor_get(b,output,0,sizeof(output));
 const auto* actual=reinterpret_cast<const unsigned char*>(output);const auto* original=reinterpret_cast<const unsigned char*>(input);
 for(unsigned i=0;i<sizeof(output);++i)m0::require(actual[i]==((i>=8&&i<24)?0xa5:original[i]));
 ggml_backend_buffer_clear(storage,0);ggml_backend_tensor_get(a,output,0,sizeof(output));for(auto f:output)m0::require(f==0);ggml_backend_synchronize(ps5);
 ggml_backend_buffer_free(storage);ggml_free(ctx);storage_call_snapshot();
 ctx=ggml_init(params);a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);auto sum=ggml_add(ctx,a,b);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,sum);
 auto cpu=ggml_backend_cpu_init();m0::require(cpu);ggml_backend_t backends[]={ps5,cpu};auto scheduler=ggml_backend_sched_new(backends,nullptr,2,128,false,false);m0::require(scheduler);
 m0::require(ggml_backend_sched_alloc_graph(scheduler,graph));ggml_backend_tensor_set(a,input,0,sizeof(input));ggml_backend_tensor_set(b,input,0,sizeof(input));float sentinel[16];for(auto& f:sentinel)f=-999;ggml_backend_tensor_set(sum,sentinel,0,sizeof(sentinel));
 m0::require(ggml_backend_sched_get_tensor_backend(scheduler,sum)==cpu);
 m0::require(m0::release_compute(scheduler,ps5,graph)==GGML_STATUS_FAILED&&m0::statistics().forwarded==0);
 ggml_backend_tensor_get(sum,output,0,sizeof(output));m0::require(!std::memcmp(output,sentinel,sizeof(output)));
 m0::require(ggml_backend_graph_compute(ps5,graph)==GGML_STATUS_FAILED&&m0::statistics().direct_graph_calls==1);
 ggml_backend_sched_free(scheduler);ggml_backend_free(cpu);ggml_free(ctx);ggml_backend_free(ps5);
 m0::require(observed.allocations==observed.frees && !observed.terminal && !observed.uncertain && observed.launches==0);
 std::puts("M0_APP STORAGE copied=16 cleared=16 partial_fill_bytes=16 surrounding_bytes_exact=true synthetic_add_rejected=true cpu_compute_calls=0 hip_launches=0 client_graph_executed=false");
}

extern "C" [[noreturn]] void ps5hip_llm_fail(const char* reason){ps5hip_clr_gpu_fail(reason);}
#ifndef PS5HIP_LLM_AUTHENTICATED_MODEL_GATE
#define PS5HIP_LLM_AUTHENTICATED_MODEL_GATE 0
#endif
__attribute__((noinline)) static void model_memory_snapshot(const char* phase){
  const int64_t pool=sceKernelGetDirectMemorySize();
  int64_t start=0;size_t largest=0;
  const int rc=pool>0?sceKernelAvailableDirectMemorySize(0,pool,0x4000,&start,&largest):-1;
  if(rc || pool<=0 || start<0 || start>pool || largest>static_cast<uint64_t>(pool-start)) {
    std::printf("M0_APP MEMORY_QUERY_FAILED phase=%s pool_bytes=%lld rc=%d\n",phase,static_cast<long long>(pool),rc);
    ps5hip_clr_gpu_fail("memory-query");
  }
  // This API reports one largest free extent, not the sum of all free memory.
  std::printf("M0_APP MEMORY phase=%s pool_bytes=%llu largest_free_bytes=%llu total_free_bytes=unknown alignment=16384\n",
              phase,static_cast<unsigned long long>(pool),static_cast<unsigned long long>(largest));
}
static bool configured_model_phase(){
 std::ifstream file(std::string(title_root)+"client-phase.txt",std::ios::binary);
 if(!file.is_open())ps5hip_clr_gpu_fail("client-phase-open");
 char phase[16]{};file.read(phase,sizeof(phase));
 const auto n=file.gcount();
 if(file.bad())ps5hip_clr_gpu_fail("client-phase-read");
 if(!file.eof())ps5hip_clr_gpu_fail("client-phase-config");
 if(n==6&&std::memcmp(phase,"vocab\n",6)==0)return false;
 if(n==11&&std::memcmp(phase,"model-gate\n",11)==0)return true;
 ps5hip_clr_gpu_fail("client-phase-config");
}
static std::string configured_model_path(const std::string& config_path){
 std::ifstream file(config_path,std::ios::binary);
 char bytes[241]{};file.read(bytes,sizeof(bytes));
 const auto n=file.gcount();
 if(file.bad()||!file.eof()||n<1||n>240)ps5hip_clr_gpu_fail("model-path-read");
 const std::string path(bytes,static_cast<size_t>(n));
 if((path.compare(0,15,"/user/homebrew/")!=0&&path.compare(0,6,"/data/")!=0)||
    path.size()<5||path.compare(path.size()-5,5,".gguf")!=0)
   ps5hip_clr_gpu_fail("model-path-config");
 for(const unsigned char c:path)
   if(!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||
        c=='_'||c=='-'||c=='.'||c=='/'))ps5hip_clr_gpu_fail("model-path-config");
 for(size_t begin=1;begin<path.size();){
   const auto end=path.find('/',begin);
   const auto part=path.substr(begin,end==std::string::npos?end:end-begin);
   if(part.empty()||part=="."||part=="..")ps5hip_clr_gpu_fail("model-path-config");
   if(end==std::string::npos)break;
   begin=end+1;
 }
 return path;
}
int main() {
  try {
    std::ifstream config("/app0/results-dir.txt", std::ios::binary);
    char bytes[242]{};
    config.read(bytes, sizeof(bytes));
    const auto length = config.gcount();
    if (!config.eof() || length < 2 || length > 240) return 2;
    const std::string directory(bytes, static_cast<size_t>(length));
    if (!ps5hip_canary::valid_results_directory(directory)) return 2;
    if (elevation::request(elevation::Capability::filesystem, "/app0/lapy.elf") != elevation::Status::ok) return 2;
    if (!ps5hip_canary::redirect_log((directory + "/ps5-hip-m0-app.log").c_str())) return 2;
    std::puts("PS5HIP START target=gfx1013 kernel=m0_client_boundary");
    if (!finalizer_installed) ps5hip_clr_gpu_fail("finalizer-install");
    ps5hip_canary::check_runtime();
    const bool full_model=configured_model_phase();
    // Conservative bounded experiment budget, not a performance measurement.
    ps5hip_clr_gpu::watchdog_start(full_model ? 120000 : 30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    if(full_model){
      // Builder and freeze controller must authenticate locked model bytes before enabling this phase.
      if(!PS5HIP_LLM_AUTHENTICATED_MODEL_GATE)ps5hip_clr_gpu_fail("model-gate-not-authenticated");
      const std::string model_path=configured_model_path(std::string(title_root)+"model-path.txt");
      struct stat st{};m0::require(::stat(model_path.c_str(),&st)==0&&S_ISREG(st.st_mode)&&st.st_size==619831552);
      // This helper accepts only the two exact errors after each observed gate rejection.
      const auto gate=m0::reject_actual_model(model_path.c_str(),model_memory_snapshot);
      m0::require(gate.attempts==2&&gate.graph_reused&&gate.result==-3);
      m0::require(observed.launches==0&&!observed.uncertain&&observed.allocations==observed.frees);
#if PS5HIP_LLM_AUTHENTICATED_MODEL_GATE
      const auto host=ps5hip::llm::native_host_allocator_stats();
      m0::require(host.allocations>0&&host.allocations==host.releases&&
                  !host.live_allocations&&!host.live_requested_bytes&&!host.failures&&
                  !host.poisoned&&host.peak_requested_bytes>=619094016);
      std::printf("M0_APP HOST_STORAGE allocations=%llu releases=%llu live_allocations=0 live_requested_bytes=0 peak_requested_bytes=%llu failures=0 poisoned=false scope=ggml_aligned\n",
                  static_cast<unsigned long long>(host.allocations),
                  static_cast<unsigned long long>(host.releases),
                  static_cast<unsigned long long>(host.peak_requested_bytes));
#endif
    }else{
      vocabulary_check();storage_and_gate_check();
    }
    ps5hip_clr_gpu::watchdog_finish();
    completed = true;
    std::puts("M0_APP PASS init=0 count_error=0 count=1 watchdog_joined=true finalization=pending");
    if (!ps5hip_canary::sync_log()) ps5hip_clr_gpu_fail("result-log");
    return 0;
  } catch (...) { ps5hip_clr_gpu_fail("exception"); }
}
