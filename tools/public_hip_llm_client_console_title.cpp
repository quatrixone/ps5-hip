#include "ps5-device.hpp"
#include "ps5hip/agc_backend.h"
#include "clr_gpu/watchdog.h"
#include "native_runtime.h"
#include "elevation.hpp"
#include <hip/hip_runtime_api.h>
#include <ps5platform/libc.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ps5_backend.h"
#include "model_gate.h"
#include <sys/stat.h>
#include <cstring>
#include <vector>

extern "C" int __cxa_thread_atexit_impl(void (*destructor)(void*), void* object,
                                          void* dso) {
  return ps5___cxa_thread_atexit_impl(destructor, object, dso);
}

namespace {
ps5hip::clr::Observation observed;
bool finalizer_installed = false;
bool completed = false;
std::unique_ptr<ps5hip::SubmissionBackend> backend() {
  return std::make_unique<ps5hip::AgcBackend>();
}
void finalized() {
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
 auto* model=llama_model_load_from_file("/app0/vocab.gguf",params); m0::require(model);
 const auto* vocab=llama_model_get_vocab(model); m0::require(llama_vocab_n_tokens(vocab)==32000);
 const char*texts[]={"","Hello, world!","The capital of France is","  spaces\nnew line\t tab","café Ελληνικά 日本語 😀","<s>hello</s>","<unk><s></s>","[INST] Say hello [/INST]"};
 const std::vector<std::vector<llama_token>> expected={{},{},{1},{1},{15043,29892,3186,29991},{15043,29892,3186,29991},{1,15043,29892,3186,29991},{1,15043,29892,3186,29991},{450,7483,310,3444,338},{450,7483,310,3444,338},{1,450,7483,310,3444,338},{1,450,7483,310,3444,338},{259,8162,13,1482,1196,12,4434},{259,8162,13,1482,1196,12,4434},{1,259,8162,13,1482,1196,12,4434},{1,259,8162,13,1482,1196,12,4434},{274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{1,274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{1,274,28059,29871,30311,30142,30142,30183,30133,30136,30173,30216,29871,30325,30346,30968,29871,243,162,155,131},{529,29879,29958,12199,829,29879,29958},{1,22172,2},{1,529,29879,29958,12199,829,29879,29958},{1,1,22172,2},{529,2960,5299,29879,2565,29879,29958},{0,1,2},{1,529,2960,5299,29879,2565,29879,29958},{1,0,1,2},{518,25580,29962,14891,22172,518,29914,25580,29962},{518,25580,29962,14891,22172,518,29914,25580,29962},{1,518,25580,29962,14891,22172,518,29914,25580,29962},{1,518,25580,29962,14891,22172,518,29914,25580,29962}};
 unsigned case_index=0,total=0;
 for(const char* text:texts) for(int special=0;special<2;++special) for(int parse=0;parse<2;++parse) {
  llama_token tokens[256]; const int n=llama_tokenize(vocab,text,int(std::strlen(text)),tokens,256,bool(special),bool(parse));
  m0::require(n>=0 && size_t(n)==expected[case_index].size());
  for(int i=0;i<n;++i)m0::require(tokens[i]==expected[case_index][i]); total+=unsigned(n); ++case_index;
 }
 m0::require(case_index==32); llama_model_free(model); llama_backend_free();
 std::printf("M0_APP VOCAB cases=%u tokens=%u vocabulary=32000 tensors=0 exact=true\n",case_index,total);
}
static void storage_and_gate_check() {
 ggml_backend_register(m0::registration());auto ps5=ggml_backend_dev_init(ggml_backend_reg_dev_get(m0::registration(),0),nullptr);m0::require(ps5);
 ggml_init_params params{1<<20,nullptr,true};auto ctx=ggml_init(params);m0::require(ctx);
 auto a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);auto b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);
 auto storage=ggml_backend_alloc_ctx_tensors(ctx,ps5);m0::require(storage);
 float input[16],output[16];for(unsigned i=0;i<16;++i)input[i]=float(i+1);
 ggml_backend_tensor_set(a,input,0,sizeof(input));ggml_backend_tensor_copy(a,b);ggml_backend_tensor_get(b,output,0,sizeof(output));m0::require(!std::memcmp(input,output,sizeof(input)));
 ggml_backend_buffer_clear(storage,0);ggml_backend_tensor_get(a,output,0,sizeof(output));for(auto f:output)m0::require(f==0);ggml_backend_synchronize(ps5);
 ggml_backend_buffer_free(storage);ggml_free(ctx);
 ctx=ggml_init(params);a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);b=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,16);auto sum=ggml_add(ctx,a,b);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,sum);
 auto cpu=ggml_backend_cpu_init();m0::require(cpu);ggml_backend_t backends[]={ps5,cpu};auto scheduler=ggml_backend_sched_new(backends,nullptr,2,128,false,false);m0::require(scheduler);
 m0::require(ggml_backend_sched_alloc_graph(scheduler,graph));ggml_backend_tensor_set(a,input,0,sizeof(input));ggml_backend_tensor_set(b,input,0,sizeof(input));float sentinel[16];for(auto& f:sentinel)f=-999;ggml_backend_tensor_set(sum,sentinel,0,sizeof(sentinel));
 m0::require(ggml_backend_sched_get_tensor_backend(scheduler,sum)==cpu);
 m0::require(m0::release_compute(scheduler,ps5,graph)==GGML_STATUS_FAILED&&m0::statistics().forwarded==0);
 ggml_backend_tensor_get(sum,output,0,sizeof(output));m0::require(!std::memcmp(output,sentinel,sizeof(output)));
 m0::require(ggml_backend_graph_compute(ps5,graph)==GGML_STATUS_FAILED&&m0::statistics().direct_graph_calls==1);
 ggml_backend_sched_free(scheduler);ggml_backend_free(cpu);ggml_free(ctx);ggml_backend_free(ps5);
 m0::require(observed.allocations==observed.frees && !observed.terminal && !observed.uncertain && observed.launches==0);
 std::puts("M0_APP STORAGE copied=16 cleared=16 synthetic_add_rejected=true cpu_compute_calls=0 hip_launches=0 client_graph_executed=false");
}

extern "C" [[noreturn]] void ps5hip_llm_fail(const char* reason){ps5hip_clr_gpu_fail(reason);}
#ifndef PS5HIP_LLM_AUTHENTICATED_MODEL_GATE
#define PS5HIP_LLM_AUTHENTICATED_MODEL_GATE 0
#endif
static bool configured_model_phase(){
 std::ifstream file("/app0/client-phase.txt",std::ios::binary);char phase[16]{};file.read(phase,sizeof(phase));
 const auto n=file.gcount();m0::require(file.eof());
 if(n==6&&std::memcmp(phase,"vocab\n",6)==0)return false;
 if(n==11&&std::memcmp(phase,"model-gate\n",11)==0)return true;
 ps5hip_clr_gpu_fail("client-phase-config");
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
    ps5hip_clr_gpu::watchdog_start(30000);
    const hipError_t initialized = hipInit(0);
    int count = -1;
    const hipError_t discovered = hipGetDeviceCount(&count);
    if (initialized != hipSuccess || discovered != hipSuccess || count != 1)
      ps5hip_clr_gpu_fail("public-discovery");
    const bool full_model=configured_model_phase();
    if(full_model){
      // Builder and freeze controller must authenticate locked model bytes before enabling this phase.
      if(!PS5HIP_LLM_AUTHENTICATED_MODEL_GATE)ps5hip_clr_gpu_fail("model-gate-not-authenticated");
      struct stat st{};m0::require(::stat("/app0/model.gguf",&st)==0&&S_ISREG(st.st_mode)&&st.st_size==619831552);
      m0::reject_actual_model("/app0/model.gguf");
      m0::require(observed.launches==0&&!observed.uncertain&&observed.allocations==observed.frees);
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
