#include "llm_prefix_hip_mock.h"
#undef DEPRECATED
#include "native_prefix.h"
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
extern "C" [[noreturn]] void ps5hip_llm_fail(const char * reason){std::fprintf(stderr,"TEST_TERMINAL %s\n",reason);std::abort();}
extern "C" ggml_status __wrap_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*);
extern "C" ggml_status __real_ggml_backend_sched_graph_compute_async(ggml_backend_sched_t,ggml_cgraph*){assert(false);return GGML_STATUS_FAILED;}
extern "C" ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t s,ggml_cgraph * g){return __wrap_ggml_backend_sched_graph_compute_async(s,g);}
static void write_raw(const char * name,const void * bytes,size_t size,void * directory){
 const std::string path=std::string(static_cast<const char*>(directory))+"/"+name;
 const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);assert(fd>=0);
 size_t done=0;unsigned interruptions=0;
 while(done<size){const auto count=::write(fd,static_cast<const char*>(bytes)+done,size-done);
  if(count<0&&errno==EINTR&&++interruptions<=8)continue;
  assert(count>0&&size_t(count)<=size-done);done+=size_t(count);
 }
 assert(::fsync(fd)==0);assert(::close(fd)==0);
}
static bool sync_log(){return std::fflush(stdout)==0;}
static void memory_snapshot(const char * phase){
 constexpr unsigned long long pool=12884901888ull,before=12882804736ull;
 const unsigned long long free=std::strcmp(phase,"after_load")==0?before-640*1024*1024:std::strcmp(phase,"after_context")==0?before-680*1024*1024:before;
 std::printf("M0_APP MEMORY phase=%s pool_bytes=%llu largest_free_bytes=%llu total_free_bytes=unknown alignment=16384\n",phase,pool,free);
}
int main(int argc,char ** argv){
 assert(argc==3);
 const auto result=m0::run_native_prefix(argv[1],{"getrows","norm","weighted","q8","q4"},write_raw,argv[2],sync_log,memory_snapshot);
 assert(result.prefix.invocations==4&&result.prefix.completed_launches==36&&result.files==24&&result.raw_bytes==139264);
 assert(allocations.empty()&&unloads==5&&frees==5&&uploaded_bytes==159285264);
}
