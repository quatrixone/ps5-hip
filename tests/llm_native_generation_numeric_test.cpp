#include "llm_full_numeric_mock.h"
#include "native_full.h"
#include <fstream>
#include <set>
static std::string directory;static std::set<std::string> names;
static bool sync_log(){return true;}
static void write_output(const char* name,const void* bytes,size_t size,void*){
 assert(name&&size&&std::strchr(name,'/')==nullptr&&std::strstr(name,"..") == nullptr&&names.insert(name).second);
 std::ofstream out(directory+"/"+name,std::ios::binary);out.write(static_cast<const char*>(bytes),size);out.close();assert(out.good());
}
int main(int argc,char** argv){
 assert(argc==3);directory=argv[2];
 const auto result=m0::run_native_full(argv[1],{"getrows","norm","weighted","q8","q4","layer"},write_output,nullptr,sync_log,nullptr,m0::NativeFullProfile::generation);
 const auto& state=result.full;assert(state.requests==16&&state.completed_nodes==12768&&state.launches==11728&&state.instances_created==state.instances_freed&&state.instances_created>0);
 assert(real_scheduler_calls==16&&device_graph_calls>0&&cpu_arithmetic_nodes==0&&allocations.empty()&&allocation_count==free_count&&loads==unloads&&names.size()==386);
 std::ofstream counters(directory+"/backend.json");counters<<"{\"requests\":"<<state.requests<<",\"nodes\":"<<state.completed_nodes<<",\"launches\":"<<state.launches<<",\"CPU_arithmetic\":0,\"scheduler_forwarded\":"<<real_scheduler_calls<<",\"allocations\":"<<allocation_count<<",\"frees\":"<<free_count<<",\"instances_created\":"<<state.instances_created<<",\"instances_freed\":"<<state.instances_freed<<"}\n";counters.close();assert(counters.good());
 std::puts("GENERATION_NUMERIC_PASS actual_scheduler=true CPU_arithmetic=0 live_logit_feedback=true GPU_execution=false");
}
