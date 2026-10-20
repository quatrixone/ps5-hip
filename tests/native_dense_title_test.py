"""Exercise exact dense title parser/sink without SDK, model or console."""
from pathlib import Path
import subprocess,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
STUBS=r'''#include <array>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>
#include <cassert>
#include <fcntl.h>
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
[[noreturn]] void ps5hip_clr_gpu_fail(const char* s){throw std::runtime_error(s);}
static unsigned helper_calls=0,snapshots=0;
struct Observed { bool terminal=false,uncertain=false; unsigned allocations=0,frees=0; } observed;
namespace ps5hip_canary { static bool sync_log(){return true;} }
namespace ps5hip { namespace llm {
struct HostStats { unsigned long long allocations=1,releases=1,peak_requested_bytes=4096; unsigned live_allocations=0,live_requested_bytes=0,failures=0; bool poisoned=false; };
static HostStats native_host_allocator_stats(){return {};}
}}
static void model_memory_snapshot(const char*){++snapshots;}
namespace m0 {
inline void require(bool ok){if(!ok)throw std::runtime_error("dense-require");}
struct LayerStatistics { unsigned invocations=0,cache_resets=0,completed_launches=0; };
struct NativeDenseResult { LayerStatistics layer; unsigned requests=0,records=0; uint64_t bytes=0; };
using DensePackedWrite=bool (*)(uint64_t,const void*,size_t,void*);
using DenseLogSync=bool (*)();
inline NativeDenseResult run_native_dense_layer(const std::array<const char*,6>& modules,
 DensePackedWrite writer,void* state,DenseLogSync log_sync,void (*snapshot)(const char*)=nullptr){
 ++helper_calls;assert(writer&&state&&log_sync&&snapshot);
 for(const auto* name:modules)assert(std::strstr(name,"/dense-layer/"));
 return {};
}
}
'''
DRIVER=r'''#include "exact.h"
int main(int,char** a){
 bool rejected=false;try{ps5hip_dense_phase(nullptr);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&helper_calls==0);
 rejected=false;try{ps5hip_dense_phase((std::string(a[1])+"/missing").c_str());}catch(const std::runtime_error& e){rejected=std::string(e.what())=="dense-raw-open";}
 assert(rejected&&helper_calls==0);
 // The declared helper result is zero, so the entry rejects before finalization.
 rejected=false;try{ps5hip_dense_phase(a[1]);}catch(const std::runtime_error& e){rejected=std::string(e.what())=="dense-require";}
 assert(rejected&&helper_calls==1&&snapshots==0);
 struct stat info{};assert(::stat((std::string(a[1])+"/readback.bin").c_str(),&info)==0&&info.st_size==0);
 return 0;}
'''
class DenseTitle(unittest.TestCase):
 def source(self):return (ROOT/'tools/public_hip_llm_client_console_title.cpp').read_text()
 def compile_run(self,source,args=()):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory);(p/'test.cpp').write_text(source)
   subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
   subprocess.run([str(p/'test'),*args],check=True,capture_output=True)
 def test_exact_phase_reader(self):
  s=self.source();reader=s[s.index('enum class ClientPhase'):s.index('static std::string configured_model_path')]
  pre='#include <fstream>\n#include <string>\n#include <cstring>\n#include <stdexcept>\n#include <cassert>\nstatic const char* title_root;\n[[noreturn]] void ps5hip_clr_gpu_fail(const char* s){throw std::runtime_error(s);}\n'
  with tempfile.TemporaryDirectory() as d:
   body='''int main(int,char** a){std::string root=std::string(a[1])+"/";title_root=root.c_str();
for(const auto& text:{std::string("dense-layer\\n"),std::string("dense-layer"),std::string("dense-layer\\r\\n"),std::string("dense-layer\\nextra")}){
{std::ofstream f(root+"client-phase.txt",std::ios::binary);f<<text;}
bool valid=false;try{valid=configured_model_phase()==ClientPhase::dense_layer;}catch(const std::runtime_error&){}assert(valid==(text=="dense-layer\\n"));}}
'''
   self.compile_run(pre+reader+body,(d,))
 def test_sink_bounds_partial_writes_and_finalization(self):
  s=self.source();writer=s[s.index('struct DensePackedWriter'):s.index('extern "C" __attribute__((noinline)) void ps5hip_dense_transport_complete',s.index('struct DensePackedWriter'))]
  pre='''#include <string>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <fcntl.h>
#include <cerrno>
#include <sys/stat.h>
static int flags=0,writes=0,syncs=0,closes=0,interrupts=0;static bool sync_fail=false,close_fail=false,zero=false;static off_t size=0;
int test_open(const char*,int f,int){flags=f;return 7;}
int test_fstat(int,struct stat* s){s->st_mode=S_IFREG;s->st_size=size;return 0;}
ssize_t test_write(int,const void*,size_t n){++writes;if(interrupts){--interrupts;errno=EINTR;return -1;}if(zero)return 0;auto count=n>2?2:n;size+=count;return count;}
int test_fsync(int){++syncs;return sync_fail?-1:0;}
int test_close(int){++closes;return close_fail?-1:0;}
[[noreturn]] void ps5hip_clr_gpu_fail(const char* s){throw std::runtime_error(s);}
#define open test_open
#define fstat test_fstat
#define write test_write
#define fsync test_fsync
#define close test_close
'''
  # Macros also rename the public callback; its implementation is otherwise exact.
  body='''
int main(){DensePackedWriter w;w.test_open("/results");assert((flags&(O_NOFOLLOW|O_EXCL|O_CREAT))==(O_NOFOLLOW|O_EXCL|O_CREAT));char b[4]{};
assert(!DensePackedWriter::test_write(1,b,4,&w)&&writes==0);interrupts=1;assert(DensePackedWriter::test_write(0,b,4,&w)&&writes==3&&w.bytes==4&&w.records==1);
assert(!DensePackedWriter::test_write(4,nullptr,4,&w));zero=true;assert(!DensePackedWriter::test_write(4,b,4,&w)&&w.bytes==4);zero=false;
interrupts=9;assert(!DensePackedWriter::test_write(4,b,4,&w)&&w.bytes==4);interrupts=0;
w.bytes=w.expected_bytes;assert(!DensePackedWriter::test_write(w.bytes,b,1,&w));w.records=11084;size=w.expected_bytes;sync_fail=true;
bool rejected=false;try{w.finish();}catch(...){rejected=true;}assert(rejected&&syncs==1&&closes==1&&w.fd==-1);
w.fd=7;sync_fail=false;close_fail=true;try{w.finish();assert(false);}catch(...){}assert(syncs==2&&closes==2&&w.fd==-1);
w.fd=7;close_fail=false;size--;try{w.finish();assert(false);}catch(...){}assert(syncs==3&&closes==3&&w.fd==-1);
w.fd=7;size=w.expected_bytes;w.finish();assert(syncs==4&&closes==4&&w.fd==-1);
}
'''
  self.compile_run(pre+writer+body)
 def test_exact_entry_compiles_with_argument_and_rejects_before_finalization(self):
  # Host compilation only: the m0/ggml backend, allocator and canary helpers are
  # stubs, so this proves the exact entry's source, argument and rejection order,
  # never native execution, transport volume or numerical results.
  s=self.source();start=s.index('struct DensePackedWriter')
  region=s[start:s.index('extern "C" __attribute__((noinline)) void ps5hip_dense_app_complete')]
  pre=STUBS
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory);(p/'exact.h').write_text(pre+region);(p/'test.cpp').write_text(DRIVER)
   subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(p),str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
   subprocess.run([str(p/'test'),directory],check=True,capture_output=True)

 def test_compile_gate_rejects_mixed_profiles(self):
  s=self.source();gate=s[s.index('#ifndef PS5HIP_LLM_AUTHENTICATED_DENSE_LAYER'):s.index('#ifndef PS5HIP_LLM_ORIGINAL_MODEL')]
  gate='\n'.join(line for line in gate.splitlines() if not line.startswith('#include'))
  for other in ('', 'PS5HIP_LLM_AUTHENTICATED_MODEL_FULL','PS5HIP_LLM_AUTHENTICATED_MODEL_LAYER','PS5HIP_LLM_AUTHENTICATED_MODEL_PREFIX','PS5HIP_LLM_ORIGINAL_MODEL'):
   command=['c++','-E','-x','c++','-DPS5HIP_LLM_AUTHENTICATED_DENSE_LAYER=1']+(['-D'+other+'=1'] if other else [])+['-']
   result=subprocess.run(command,input=gate,text=True,capture_output=True)
   self.assertEqual(result.returncode==0,not bool(other))
 def test_title_callsite_and_audit_expectations(self):
  s=self.source();branch=s[s.index('void ps5hip_dense_phase('):s.index('void ps5hip_dense_app_complete()')]
  self.assertNotIn('configured_model_path',branch);self.assertNotIn('llama_model_load',branch)
  self.assertEqual(branch.count('/dense-layer/'),6)
  self.assertLess(branch.index('packed.finish();'),branch.index('ps5hip_dense_transport_complete();'))
  self.assertIn('watchdog_start(dense_layer ? 120000',s)
  self.assertIn('PS5HIP_LLM_AUTHENTICATED_MODEL_GATE || PS5HIP_LLM_AUTHENTICATED_DENSE_LAYER',s)
  main=s[s.index('int main()'):]
  self.assertLess(main.index('watchdog_finish();'),main.index('ps5hip_dense_app_complete();'))
  # Actual preprocessor must eliminate model routes in the dense main branch.
  main='\n'.join(line for line in main.splitlines() if not line.startswith('#include'))
  result=subprocess.run(['c++','-E','-P','-x','c++','-DPS5HIP_LLM_AUTHENTICATED_DENSE_LAYER=1','-'],input=main,text=True,capture_output=True,check=True).stdout
  self.assertIn('ps5hip_dense_phase(directory.c_str());',result)
  self.assertNotIn('model-path.txt',result)
  for forbidden in ('configured_model_path','vocabulary_check','reject_actual_model','run_native_full','run_native_layer','run_native_prefix'):
   self.assertNotIn(forbidden,result)
if __name__=='__main__':unittest.main()
