"""Compile actual ihipMemcpy control flow with mocked ROCclr completion.

This tests status/reference handling, not native transport or HIP conformance.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_public_hip_copy import prepare, AFTER
from prepare_public_hip_memory import digest

PREAMBLE = r'''
#include <cassert>
#include <cstddef>
#include <vector>
using hipError_t = int;
constexpr int hipSuccess=0, hipErrorUnknown=999, hipErrorInvalidValue=1;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice, hipMemcpyDefault };
enum hipMemoryType {hipMemoryTypeHost,hipMemoryTypeDevice};
constexpr int CL_COMPLETE=0, CL_MEM_SVM_FINE_GRAIN_BUFFER=1, CL_MEM_USE_HOST_PTR=2;
static int enqueued, waited, released, statusReads, hostCopies, validation, construction;
static int finalStatus; static bool waitResult;
namespace amd {
struct Device { int context() {return 0;} }; static Device device;
struct HostQueue { virtual ~HostQueue() = default; struct Command* getLastQueuedCommand(bool) {return nullptr;} };
struct Memory { int getMemFlags() const {return 0;} Device* GetDeviceById() {return &device;} };
struct Command {
  using EventWaitList=std::vector<Command*>;
  void enqueue() {++enqueued;}
  bool awaitCompletion() {assert(released==0);++waited;return waitResult;}
  int status() const {assert(released==0);++statusReads;return finalStatus;}
  void release() {++released;}
  HostQueue* queue();
};
struct Marker:Command {Marker(HostQueue&,bool,const EventWaitList&) {}};
}
namespace hip {struct Stream:amd::HostQueue {void finish() {}}; static Stream stream;
Stream* getNullStream(int) {return &stream;}}
amd::HostQueue* amd::Command::queue() {return &hip::stream;}
static amd::Memory memory; static int deviceA,deviceB,hostA,hostB;
amd::Memory* getMemoryObject(const void* p,size_t&) {return p==&deviceA||p==&deviceB?&memory:nullptr;}
hipError_t ihipMemcpy_validate(void*,const void*,size_t,hipMemcpyKind) {return validation;}
static amd::Command command;
hipError_t ihipMemcpyCommand(amd::Command*& out,void*,const void*,size_t,hipMemcpyKind,hip::Stream&,bool) {
  out=&command;return construction;
}
void ihipHtoHMemcpy(void*,const void*,size_t,hip::Stream&) {++hostCopies;}
'''

DRIVER = r'''
static void reset(int status=0,bool wait=true) {
  enqueued=waited=released=statusReads=hostCopies=validation=construction=0;
  finalStatus=status;waitResult=wait;
}
int main() {
  struct Route {void* dst;const void* src;hipMemcpyKind kind;};
  Route routes[]={{&deviceA,&hostA,hipMemcpyHostToDevice},
                  {&hostA,&deviceA,hipMemcpyDeviceToHost},
                  {&deviceB,&deviceA,hipMemcpyDeviceToDevice}};
  for(auto route:routes) {
    for(int status:{0,-1,1,2}) for(bool wait:{false,true}) {
      reset(status,wait);
      const auto result=ihipMemcpy(route.dst,route.src,4,route.kind,hip::stream,false,false);
      assert(result==((status==0&&wait)?hipSuccess:hipErrorUnknown));
      assert(enqueued==1&&waited==1&&released==1);
      assert(statusReads==(wait?1:0));
    }
    reset();validation=17;
    assert(ihipMemcpy(route.dst,route.src,4,route.kind,hip::stream,false,false)==17);
    assert(!enqueued&&!waited&&!released);
    reset();construction=19;
    assert(ihipMemcpy(route.dst,route.src,4,route.kind,hip::stream,false,false)==19);
    assert(!enqueued&&!waited&&!released);
  }
  reset();
  assert(ihipMemcpy(nullptr,nullptr,0,hipMemcpyDefault,hip::stream,false,false)==0);
  assert(!enqueued&&!waited&&!released);
  reset();
  assert(ihipMemcpy(&hostB,&hostA,4,hipMemcpyDefault,hip::stream,false,false)==0);
  assert(hostCopies==1&&!enqueued&&!waited&&!released);
  reset();
  assert(ihipMemcpy(&deviceA,&deviceA,4,hipMemcpyDefault,hip::stream,false,false)==0);
  assert(!enqueued&&!waited&&!released);
  // Explicitly asynchronous D2D retains upstream nonblocking behavior.
  reset(-1,false);
  assert(ihipMemcpy(&deviceB,&deviceA,4,hipMemcpyDeviceToDevice,hip::stream,true,true)==0);
  assert(enqueued==1&&!waited&&!statusReads&&released==1);
  // Upstream forced host wait remains unchanged for explicit async H2D/D2H.
  for(int i=0;i<2;++i) {
    reset(-1,false);auto route=routes[i];
    assert(ihipMemcpy(route.dst,route.src,4,route.kind,hip::stream,true,true)==0);
    assert(enqueued==1&&waited==1&&!statusReads&&released==1);
  }
}
'''


class CompletionTest(unittest.TestCase):
    def test_actual_function_completion_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in ('rocclr/platform/memory.cpp', 'hipamd/src/hip_memory.cpp'):
                target = source / name
                target.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / '.deps/sources/clr' / name, target)
            svm = (ROOT / 'ports/clr/ps5-public-memory.patch').read_text().split('--- a/hipamd/')[0]
            subprocess.run(['patch','--batch','--fuzz=0','-p1'],cwd=source,input=svm,
                           text=True,check=True,capture_output=True)
            prepare(source)
            target=source/'hipamd/src/hip_memory.cpp'
            self.assertEqual(digest(target),AFTER)
            prepare(source)
            body=target.read_text().split('hipError_t ihipMemcpy(void* dst,')[1].split('\n}\n',1)[0]
            function='hipError_t ihipMemcpy(void* dst,'+body+'\n}\n'
            test=source/'completion.cpp';test.write_text(PREAMBLE+function+DRIVER)
            for enabled in (True,False):
                binary=source/('patched' if enabled else 'upstream')
                args=['clang++-18','-std=c++20','-O1','-g','-fsanitize=address,undefined',str(test),'-o',str(binary)]
                if enabled:args.insert(1,'-DWITH_PS5_DEVICE')
                subprocess.run(args,check=True,capture_output=True)
                result=subprocess.run([str(binary)],capture_output=True)
                if enabled:self.assertEqual(result.returncode,0,result.stderr.decode())
                else:self.assertNotEqual(result.returncode,0,'negative control did not detect original success-on-failure')
            target.write_text(target.read_text()+'\n// changed\n')
            with self.assertRaises(ValueError):prepare(source)


if __name__ == '__main__':
    unittest.main()
