#include "llm_full_numeric_mock.h"
#include "native_full.h"
#include <fstream>
#include <set>
#include <filesystem>
static std::string directory;static std::set<std::string> names;
static bool sync_log(){return true;}
static void write_output(const char* name,const void* bytes,size_t size,void*){
 assert(name&&size&&std::strstr(name,"..") == nullptr&&names.insert(name).second);
 const auto path=std::filesystem::path(name);assert(!path.is_absolute());
 if(path.has_parent_path()){bool known=false;for(const auto& item:m0::native_generation_cases)known|=path.parent_path()==item.id;assert(known);std::filesystem::create_directories(directory+"/"+path.parent_path().string());}
 std::ofstream out(directory+"/"+name,std::ios::binary);out.write(static_cast<const char*>(bytes),size);out.close();assert(out.good());
}
int main(int argc,char** argv){
 const bool corpus=argc==4&&!std::strcmp(argv[3],"corpus");assert(argc==3||corpus);directory=argv[2];
 const auto result=m0::run_native_full(argv[1],{"getrows","norm","weighted","q8","q4","layer"},write_output,nullptr,sync_log,nullptr,corpus?m0::NativeFullProfile::corpus:m0::NativeFullProfile::generation);
 const auto& state=result.full;assert((corpus?state.requests>=160:state.requests==16)&&state.completed_nodes==state.requests*798&&state.launches==state.requests*733&&state.instances_created==state.instances_freed&&state.instances_created>0);
 assert(real_scheduler_calls==state.requests&&device_graph_calls>0&&cpu_arithmetic_nodes==0&&allocations.empty()&&allocation_count==free_count&&loads==unloads&&names.size()==state.requests*24+(corpus?11:2));
 std::ofstream counters(directory+"/backend.json");counters<<"{\"requests\":"<<state.requests<<",\"nodes\":"<<state.completed_nodes<<",\"launches\":"<<state.launches<<",\"CPU_arithmetic\":0,\"scheduler_forwarded\":"<<real_scheduler_calls<<",\"allocations\":"<<allocation_count<<",\"frees\":"<<free_count<<",\"instances_created\":"<<state.instances_created<<",\"instances_freed\":"<<state.instances_freed<<"}\n";counters.close();assert(counters.good());
 if(corpus)std::puts("GENERATION_NUMERIC_PASS profile=corpus cases=5 runs=2 max_new_tokens=16 actual_scheduler=true CPU_arithmetic=0 live_logit_feedback=true GPU_execution=false");
 else std::puts("GENERATION_NUMERIC_PASS actual_scheduler=true CPU_arithmetic=0 live_logit_feedback=true GPU_execution=false");
}
