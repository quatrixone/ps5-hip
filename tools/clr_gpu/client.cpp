// Prospective native client. No synthetic memory provider or transport.
#include "ps5-device.hpp"
#include "ps5hip/agc_backend.h"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/context.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "device/device.hpp"
#include "utils/flags.hpp"
#include <cstdio>

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char*);
extern const unsigned char ps5hip_clr_code[];
extern const size_t ps5hip_clr_code_size;
namespace ps5hip_clr_gpu {
namespace {
ps5hip::clr::Observation observation;
std::atomic<unsigned> backend_attempts{0};
unsigned joins=0;
unsigned input[81],output[81];
bool complete=false;
void require(bool condition,const char* reason){if(!condition)ps5hip_clr_gpu_fail(reason);}
std::unique_ptr<ps5hip::SubmissionBackend> backend(){++backend_attempts;return std::make_unique<ps5hip::AgcBackend>();}
class Queue final:public amd::HostQueue {
 bool terminate()override{
  if(!amd::HostQueue::terminate())ps5hip_clr_gpu_fail("queue-join-retain-all");
  ++joins;return true;
 }
public:using amd::HostQueue::HostQueue;
};
}
bool passed(){return complete;}
void run(){
 // Every manual owner survives an exception: the title catches and quarantines
 // without returning through normal application teardown. Data is persistent.
 amd::IS_HIP=true;AMD_DIRECT_DISPATCH=false;
 auto* device=ps5hip::clr::create_device(backend,observation);require(device,"device");
 amd::Context::Info info{};require(amd::Context::checkProperties(nullptr,&info)==CL_SUCCESS,"context-properties");
 auto* context=new amd::Context({device},info);require(context->create(nullptr)==CL_SUCCESS,"context");
 auto* program=new amd::Program(*context);
 require(program->addDeviceProgram(*device,ps5hip_clr_code,ps5hip_clr_code_size,true)==CL_SUCCESS,"aot-add");
 require(program->load(),"aot-load");auto* symbol=program->findSymbol("thread_words");require(symbol,"aot-symbol");
 auto* kernel=new amd::Kernel(*program,*symbol,"thread_words");program->release();
 auto* a=new Queue(*context,*device,0);auto* b=new Queue(*context,*device,0);require(a->create()&&b->create(),"queues");
 for(unsigned i=0;i<81;++i){input[i]=0xa5000000u+i;output[i]=0xdeadbeef;}
 auto* x=new(*context)amd::Buffer(*context,CL_MEM_READ_WRITE,sizeof(input));
 auto* y=new(*context)amd::Buffer(*context,CL_MEM_READ_WRITE,sizeof(input));require(x->create()&&y->create(),"buffers");
 cl_mem handle=as_cl(static_cast<amd::Memory*>(y));unsigned count=73,salt=19;
 kernel->parameters().set(0,sizeof(handle),&handle);kernel->parameters().set(1,4,&count);kernel->parameters().set(2,4,&salt);
 auto* write=new amd::WriteMemoryCommand(*a,CL_COMMAND_WRITE_BUFFER,{},*x,amd::Coord3D(0),amd::Coord3D(sizeof(input)),input);
 auto* copy=new amd::CopyMemoryCommand(*b,CL_COMMAND_COPY_BUFFER,{write},*x,*y,amd::Coord3D(0),amd::Coord3D(0),amd::Coord3D(sizeof(input)));
 size_t offset=0,global=128,local=64;amd::NDRangeContainer shape(1,&offset,&global,&local);
 auto* launch=new amd::NDRangeKernelCommand(*a,{copy},*kernel,shape);require(launch->captureAndValidate()==CL_SUCCESS,"capture");
 auto* read=new amd::ReadMemoryCommand(*b,CL_COMMAND_READ_BUFFER,{launch},*y,amd::Coord3D(0),amd::Coord3D(sizeof(output)),output);
 require(write->validateMemory()&&copy->validateMemory()&&read->validateMemory(),"memory-validation");
 salt=999;kernel->parameters().set(2,4,&salt);kernel->release();x->release();y->release();
 require(observation.allocations==2&&observation.frees==0&&backend_attempts==0,"capture-retention");
 std::puts("CLR_GPU AOT functions=1 captured=true buffers_retained=2 backend_attempts=0");
 write->enqueue();copy->enqueue();launch->enqueue();read->enqueue();
 require(read->awaitCompletion(),"read-completion-retain-all");a->finish(true);b->finish(true);
 for(unsigned i=0;i<81;++i)require(output[i]==(i<73?i*17+19:input[i]),"output-or-guard");
 require(observation.launches==1&&!observation.terminal&&backend_attempts==1,"native-completion");
 write->release();copy->release();launch->release();read->release();a->release();b->release();
 require(joins==2&&observation.retired_queues==2,"joined-ownership");context->release();delete device;
 require(observation.allocations==2&&observation.frees==2&&observation.gate_destroyed&&!observation.terminal,"normal-device-teardown");
 complete=true;
 std::puts("CLR_GPU PASS launches=1 exact_values=73 readback_words=81 guards=8 captured_salt=19 backend_attempts=1 allocations=2 frees=2 queues_joined=2 gate_destroyed=true");
}
}
