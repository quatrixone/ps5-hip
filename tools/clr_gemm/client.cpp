#include "clr_gemm/client.h"
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
#include <cstring>

extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char*);
extern const unsigned char ps5hip_clr_code[];
extern const size_t ps5hip_clr_code_size;
namespace ps5hip_clr_gemm {
namespace {
constexpr unsigned M=9,N=11,K=13,LDA=16,LDB=16,LDC=16;
constexpr float alpha=1.25f,beta=-0.5f;
float a[M*LDA],b[K*LDB],c[M*LDC],out[M*LDC];
ps5hip::clr::Observation observed;
unsigned joins=0;
bool complete=false;
void require(bool ok,const char* reason){if(!ok)ps5hip_clr_gpu_fail(reason);}
class Queue final:public amd::HostQueue {
 bool terminate()override{if(!amd::HostQueue::terminate())ps5hip_clr_gpu_fail("gemm-queue-join");++joins;return true;}
public:using amd::HostQueue::HostQueue;
};
void initialize(){
 for(unsigned i=0;i<M;++i)for(unsigned j=0;j<LDA;++j)a[i*LDA+j]=float(int((i*LDA+j)*7%17)-8);
 for(unsigned i=0;i<K;++i)for(unsigned j=0;j<LDB;++j)b[i*LDB+j]=float(int((i*LDB+j)*11%19)-9);
 for(unsigned i=0;i<M;++i)for(unsigned j=0;j<LDC;++j)c[i*LDC+j]=float(int((i*LDC+j)*3%23)-11);
 for(float& value:out)value=9999;
}
void oracle(){
 unsigned checked=0,guards=0;
 for(unsigned row=0;row<M;++row)for(unsigned col=0;col<LDC;++col){
  float expected=c[row*LDC+col];
  if(col<N){float sum=0;for(unsigned i=0;i<K;++i)sum+=a[row*LDA+i]*b[i*LDB+col];expected=alpha*sum+beta*expected;++checked;}
  else ++guards;
  require(out[row*LDC+col]==expected,"gemm-numerical-or-padding");
 }
 require(checked==99&&guards==45,"gemm-oracle-count");
}
}
bool passed(){return complete;}
void run(){
 amd::IS_HIP=true;AMD_DIRECT_DISPATCH=false;initialize();
 auto* device=ps5hip::clr::create_device([]()->std::unique_ptr<ps5hip::SubmissionBackend>{return std::make_unique<ps5hip::AgcBackend>();},observed);require(device,"gemm-device");
 amd::Context::Info info{};require(amd::Context::checkProperties(nullptr,&info)==CL_SUCCESS,"gemm-context-properties");
 auto* context=new amd::Context({device},info);require(context->create(nullptr)==CL_SUCCESS,"gemm-context");
 auto* program=new amd::Program(*context);
 require(program->addDeviceProgram(*device,ps5hip_clr_code,ps5hip_clr_code_size,true)==CL_SUCCESS,"gemm-aot-add");
 require(program->load(),"gemm-aot-load");auto* symbol=program->findSymbol("ps5hip_sgemm_nn");require(symbol,"gemm-aot-symbol");
 auto* kernel=new amd::Kernel(*program,*symbol,"ps5hip_sgemm_nn");program->release();
 auto* queue=new Queue(*context,*device,0);require(queue->create(),"gemm-queue");
 auto* x=new(*context)amd::Buffer(*context,CL_MEM_READ_WRITE,sizeof(a));
 auto* y=new(*context)amd::Buffer(*context,CL_MEM_READ_WRITE,sizeof(b));
 auto* z=new(*context)amd::Buffer(*context,CL_MEM_READ_WRITE,sizeof(c));
 require(x->create()&&y->create()&&z->create(),"gemm-buffers");
 cl_mem ax=as_cl(static_cast<amd::Memory*>(x)),by=as_cl(static_cast<amd::Memory*>(y)),cz=as_cl(static_cast<amd::Memory*>(z));
 uint32_t m=M,n=N,k=K;size_t lda=LDA,ldb=LDB,ldc=LDC;float scale=alpha,prior=beta;
 auto& params=kernel->parameters();
 params.set(0,sizeof(ax),&ax);params.set(1,sizeof(by),&by);params.set(2,sizeof(cz),&cz);
 params.set(3,sizeof(m),&m);params.set(4,sizeof(n),&n);params.set(5,sizeof(k),&k);
 params.set(6,sizeof(lda),&lda);params.set(7,sizeof(ldb),&ldb);params.set(8,sizeof(ldc),&ldc);
 params.set(9,sizeof(scale),&scale);params.set(10,sizeof(prior),&prior);
 auto* write_a=new amd::WriteMemoryCommand(*queue,CL_COMMAND_WRITE_BUFFER,{},*x,amd::Coord3D(0),amd::Coord3D(sizeof(a)),a);
 auto* write_b=new amd::WriteMemoryCommand(*queue,CL_COMMAND_WRITE_BUFFER,{write_a},*y,amd::Coord3D(0),amd::Coord3D(sizeof(b)),b);
 auto* write_c=new amd::WriteMemoryCommand(*queue,CL_COMMAND_WRITE_BUFFER,{write_b},*z,amd::Coord3D(0),amd::Coord3D(sizeof(c)),c);
 size_t offsets[2]{0,0},globals[2]{16,16},locals[2]{8,8};amd::NDRangeContainer shape(2,offsets,globals,locals);
 auto* launch=new amd::NDRangeKernelCommand(*queue,{write_c},*kernel,shape);require(launch->captureAndValidate()==CL_SUCCESS,"gemm-capture");
 auto* read=new amd::ReadMemoryCommand(*queue,CL_COMMAND_READ_BUFFER,{launch},*z,amd::Coord3D(0),amd::Coord3D(sizeof(out)),out);
 require(write_a->validateMemory()&&write_b->validateMemory()&&write_c->validateMemory()&&read->validateMemory(),"gemm-memory-validation");
 kernel->release();x->release();y->release();z->release();
 write_a->enqueue();write_b->enqueue();write_c->enqueue();launch->enqueue();read->enqueue();
 require(read->awaitCompletion(),"gemm-read-completion");queue->finish(true);oracle();
 require(observed.launches==1&&!observed.terminal,"gemm-native-completion");
 write_a->release();write_b->release();write_c->release();launch->release();read->release();queue->release();
 require(joins==1&&observed.retired_queues==1,"gemm-queue-retirement");context->release();delete device;
 require(observed.allocations==3&&observed.frees==3&&observed.gate_destroyed&&!observed.terminal,"gemm-teardown");
 complete=true;
 std::puts("CLR_GEMM PASS launches=1 exact_values=99 padding_guards=45 allocations=3 frees=3 queues_joined=1 gate_destroyed=true");
}
}
