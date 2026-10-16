"""Compile the real GGML wrapper and NativeMemory against modeled kernel calls."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include "native_host_allocator.h"
#include "ps5hip/native_memory.h"
#include <ps5platform/kernel.h>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#include <cstdio>
static int alloc_error, map_error, unmap_error, release_error;
static unsigned allocs,maps,unmaps,releases;
static uint64_t last_size,last_alignment;
static int64_t next_physical=0x200000;
static void check(bool b){if(!b)throw std::runtime_error("assertion");}
struct Quarantine{};
extern "C" [[noreturn]] void ps5hip_llm_fail(const char*){throw Quarantine{};}
extern "C" int64_t sceKernelGetDirectMemorySize(){return int64_t(12)<<30;}
extern "C" int32_t sceKernelAllocateDirectMemory(int64_t low,int64_t high,size_t n,size_t a,int type,int64_t* p){
 check(low==0&&high==(int64_t(12)<<30)&&type==12);++allocs;last_size=n;last_alignment=a;
 *p=next_physical;next_physical+=0x400000;return alloc_error;
}
extern "C" int32_t sceKernelMapDirectMemory(void** p,size_t n,int prot,int flags,int64_t phys,size_t a){
 check(n==last_size&&a==last_alignment&&prot==0x33&&flags==0);++maps;
 *p=reinterpret_cast<void*>(0x4000000000ULL+phys);return map_error;
}
extern "C" int32_t sceKernelMunmap(void*,size_t){++unmaps;return unmap_error;}
extern "C" int32_t sceKernelReleaseDirectMemory(int64_t,size_t){++releases;return release_error;}
template<class F> void poisoned(F f){bool caught=false;try{f();}catch(Quarantine&){caught=true;}check(caught&&ps5hip::llm::native_host_allocator_stats().poisoned);}
int main(int argc,char** argv){try{
 check(argc==2);const char* m=argv[1];
 auto alloc=__wrap_ggml_aligned_malloc;auto free=__wrap_ggml_aligned_free;
 using ps5hip::llm::native_host_allocator_stats;
 if(!std::strcmp(m,"large")){
  auto p=alloc(619094016);check(p&&last_size==619102208&&last_alignment==0x200000);
  auto s=native_host_allocator_stats();check(s.allocations==1&&s.live_requested_bytes==619094016);
  free(p,619094016);check(unmaps==1&&releases==1);
 }else if(!std::strcmp(m,"bounds")){
  check(!alloc(0)&&!alloc(SIZE_MAX)&&!alloc(uint64_t(1)<<47)&&!alloc(uint64_t(13)<<30));check(!allocs);
  free(nullptr,123);auto p=alloc(1);check(last_size==0x4000&&last_alignment==0x10000);free(p,1);
 }else if(!std::strcmp(m,"capacity")){
  std::vector<void*> p;for(int i=0;i<256;++i){p.push_back(alloc(64));check(p.back());}
  check(!alloc(64)&&allocs==256);for(auto x:p)free(x,64);
  auto x=alloc(64);check(x);free(x,64);
 }else if(!std::strcmp(m,"allocate-failure")){
  alloc_error=1;check(!alloc(64)&&maps==0&&releases==0);alloc_error=0;auto p=alloc(64);check(p);free(p,64);
 }else if(!std::strcmp(m,"map-failure")){
  map_error=1;check(!alloc(64)&&releases==1&&unmaps==0);map_error=0;auto p=alloc(64);check(p);free(p,64);
 }else if(!std::strcmp(m,"setup-retained")){
  map_error=release_error=1;poisoned([&]{alloc(64);});check(releases==1&&!unmaps);
  check(ps5hip::native_memory_stats().live_bytes==0x4000);
  poisoned([&]{alloc(64);});check(allocs==1&&releases==1);return 0;
 }else if(!std::strcmp(m,"unmap-failure")||!std::strcmp(m,"release-failure")){
  auto p=alloc(64);check(p);unmap_error=!std::strcmp(m,"unmap-failure");release_error=!unmap_error;
  poisoned([&]{free(p,64);});check(unmaps==1&&releases==unsigned(!unmap_error));
  check(native_host_allocator_stats().live_allocations==1&&ps5hip::native_memory_stats().live_bytes==0x4000);
  poisoned([&]{free(p,64);});poisoned([&]{alloc(64);});check(unmaps==1);return 0;
 }else if(!std::strcmp(m,"wrong-size")||!std::strcmp(m,"interior")||!std::strcmp(m,"foreign")||!std::strcmp(m,"double-free")){
  auto p=alloc(64);check(p);
  if(!std::strcmp(m,"double-free"))free(p,64);
  auto prior=unmaps;
  poisoned([&]{free(!std::strcmp(m,"foreign")?reinterpret_cast<void*>(1):!std::strcmp(m,"interior")?static_cast<char*>(p)+1:p,!std::strcmp(m,"wrong-size")?63:64);});
  check(unmaps==prior);return 0;
 }else if(!std::strcmp(m,"parallel")){
  std::atomic<unsigned> ready{0};std::atomic<bool> go{false};std::vector<std::thread> threads;
  for(int i=0;i<8;++i)threads.emplace_back([&]{auto p=alloc(123);check(p);++ready;while(!go.load()){}free(p,123);});
  while(ready.load()!=8){}auto s=native_host_allocator_stats();check(s.live_allocations==8&&s.live_requested_bytes==984);go=true;
  for(auto& t:threads)t.join();check(allocs==8&&unmaps==8&&releases==8);
 }else throw std::runtime_error("unknown mode");
 auto s=native_host_allocator_stats();check(!s.poisoned&&!s.live_allocations&&!s.live_requested_bytes&&s.allocations==s.releases);
 auto n=ps5hip::native_memory_stats();check(n.live_bytes==0&&n.allocations==n.releases);
 return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
'''


class NativeHostAllocatorTest(unittest.TestCase):
    def test_actual_wrapper_kernel_lifecycle(self):
        compiler = shutil.which("clang++-18")
        self.assertIsNotNone(compiler, "clang++-18 is required")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            # Keep the real SDK platform declarations, without replacing host libc headers.
            (folder / "ps5platform").mkdir()
            shutil.copyfile(ROOT / ".deps/sdk/target/include/ps5platform/kernel.h",
                            folder / "ps5platform/kernel.h")
            source = folder / "test.cpp"
            source.write_text(HARNESS)
            binary = folder / "test"
            subprocess.run([compiler, "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                            "-I", str(folder), "-I", str(ROOT / "include"),
                            "-I", str(ROOT / "ports/ggml"), str(source),
                            str(ROOT / "ports/ggml/native_host_allocator.cpp"),
                            str(ROOT / "src/native_memory.cpp"), "-o", str(binary)], check=True)
            for mode in ("large", "bounds", "capacity", "allocate-failure", "map-failure",
                         "setup-retained", "unmap-failure", "release-failure", "wrong-size",
                         "interior", "foreign", "double-free", "parallel"):
                with self.subTest(mode=mode):
                    subprocess.run([str(binary), mode], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
