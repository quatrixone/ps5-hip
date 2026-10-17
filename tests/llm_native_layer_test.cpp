#include "llm_layer_hip_mock.h"
#undef DEPRECATED
#include "native_layer.h"
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
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
static std::vector<std::string> phases;
static void memory_snapshot(const char * phase){
 phases.emplace_back(phase);
 if(std::strcmp(phase,"before_load")==0||std::strcmp(phase,"after_cleanup")==0)assert(allocations.empty());
 else assert(allocations.size()==2);
 constexpr unsigned long long pool=12884901888ull,before=12882804736ull;
 const unsigned long long free=std::strcmp(phase,"after_load")==0?before-640*1024*1024:std::strcmp(phase,"after_context")==0?before-680*1024*1024:before;
 std::printf("M0_APP MEMORY phase=%s pool_bytes=%llu largest_free_bytes=%llu total_free_bytes=unknown alignment=16384\n",phase,pool,free);
}
int main(int argc,char ** argv){
 assert(argc==3);
 const auto result=m0::run_native_layer(argv[1],{"getrows","norm","weighted","q8","q4","layer"},write_raw,argv[2],sync_log,memory_snapshot);
 assert(result.layer.invocations==514&&result.layer.completed_launches==16962&&result.files==151);
 assert((phases==std::vector<std::string>{"before_load","after_load","after_context","after_cleanup"}));
 assert(allocations.empty()&&loads==6&&unloads==6&&uploaded==63502368&&allocation_count==528&&free_count==528);
}
