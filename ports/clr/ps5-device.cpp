#include "ps5-device.hpp"
#include "ps5hip/native_data.h"
#include "ps5hip/fill_layout.h"
#include "ps5hip/native_launch.h"
#include "ps5hip/host_launch.h"
#include "ps5hip/hip_launch_profile.h"
#include "platform/command.hpp"
#include "platform/commandqueue.hpp"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "platform/context.hpp"
#include "device/device.hpp"
#include "device/devprogram.hpp"
#include "device/devkernel.hpp"
#include <pthread.h>
#include <limits>
#include <cstring>
#ifdef PS5HIP_CLR_MODEL
extern int ps5hip_clr_model_destroy(pthread_mutex_t*) noexcept;
#endif

namespace ps5hip::clr {
namespace {
using namespace amd;
struct State : std::enable_shared_from_this<State> {
    pthread_mutex_t gate=PTHREAD_MUTEX_INITIALIZER;
    NativeDataRegistry data;
    BackendFactory factory;
    Observation& observed;
    std::unique_ptr<SubmissionBackend> backend;
    std::unique_ptr<SubmissionQueue> queue;
    std::shared_ptr<State> retained;
    State(BackendFactory f,Observation& o):factory(f),observed(o){}
    ~State(){
        // Only constructor failure reaches here without Device's checked close;
        // such a State has never escaped or acquired native resources.
        if(!observed.gate_destroyed&&pthread_mutex_destroy(&gate))observed.terminal=true;
    }
    void retain_failure() noexcept {
        observed.terminal=true;
        if(queue&&queue->poisoned())observed.uncertain=true;
        // Device installs a self-owner before native operations. Only normal
        // Device teardown breaks it, including lock-failure paths with no gate.
    }
    template<class F> int operation(F action) noexcept {
        if(observed.terminal)return CL_OUT_OF_RESOURCES;
        if(pthread_mutex_lock(&gate)){observed.terminal=true;return CL_OUT_OF_RESOURCES;}
        int status=CL_SUCCESS;
        try {
            if(observed.terminal)status=CL_OUT_OF_RESOURCES;
            else action();
        } catch(const std::bad_alloc&){status=CL_OUT_OF_HOST_MEMORY;}
          catch(const NativeMemorySetupError& e){status=CL_MEM_OBJECT_ALLOCATION_FAILURE;if(e.retained_backing)retain_failure();}
          catch(const NativeMemoryCleanupError&){status=CL_OUT_OF_RESOURCES;retain_failure();}
          catch(...){status=CL_INVALID_OPERATION;}
        if((queue&&queue->poisoned())||data.poisoned()){retain_failure();status=CL_OUT_OF_RESOURCES;}
        if(pthread_mutex_unlock(&gate)){observed.terminal=true;status=CL_OUT_OF_RESOURCES;}
        return status;
    }
    void initialize(){
        if(backend)return;
        // No retry after a backend factory may have touched the native driver.
        try {backend=factory();if(!backend)throw ObjectError("backend factory failed");queue=std::make_unique<SubmissionQueue>(*backend);}
        catch(...){retain_failure();throw;}
    }
};
class Memory final:public amd::device::Memory {
    std::shared_ptr<State> state_;
    void* base_=nullptr;
public:
    Memory(amd::Memory& owner,std::shared_ptr<State> state):amd::device::Memory(owner),state_(std::move(state)){
        if(!owner.asBuffer()||owner.parent()||!owner.getSize()||owner.getHostMem()||
           (owner.getMemFlags()&~(CL_MEM_READ_WRITE|CL_MEM_READ_ONLY|CL_MEM_WRITE_ONLY)))
            throw ObjectError("unsupported buffer ownership");
        if(state_->operation([&]{base_=state_->data.allocate(owner.getSize());})!=CL_SUCCESS)
            throw ObjectError("buffer allocation failed");
        ++state_->observed.allocations;
        if(owner.getSvmPtr()==reinterpret_cast<void*>(amd::Memory::MemoryType::kSvmMemoryPtr))
            owner.setSvmPtr(base_);
    }
    ~Memory(){
        if(base_)retire();
        // Terminal State is retained. Never throw or pretend a failed free worked.
    }
    bool retire() noexcept {
        if(!base_)return false;
        if(state_->operation([&]{state_->data.free(base_);base_=nullptr;})!=CL_SUCCESS)return false;
        ++state_->observed.frees;return true;
    }
    uint64_t virtualAddress()const override{return reinterpret_cast<uintptr_t>(base_);}
};
struct PublicProfile {
    NativeLaunchRoute route;
    std::array<uint32_t,3> threads;
    uint32_t dynamic_lds_limit;
};
static bool public_profile(const ps5hip::Kernel& k,PublicProfile& profile){
    // Explicit admitted union: a non-initial kernel is not necessarily a reduction.
    if(k.rsrc2==0x38c && geometry_launch_rejection(k).empty())
        profile={NativeLaunchRoute::Geometry,{256,1,1},0};
    else if(k.rsrc2==0xb8c && geometry_lds_launch_rejection(k).empty())
        profile={NativeLaunchRoute::GeometryLds,{32,4,1},0};
    else if(initial_launch_rejection(k).empty())
        profile={NativeLaunchRoute::Initial,{64,1,1},0};
    else if(reduction_lds_launch_rejection(k).empty())
        profile={NativeLaunchRoute::ReductionLds,{1024,1,1},65536};
    else return false;
    const uint32_t flat=profile.threads[0]*profile.threads[1]*profile.threads[2];
    return k.max_workgroup_size>=flat && (k.required_workgroup==std::array<uint32_t,3>{} ||
                                        k.required_workgroup==profile.threads);
}
class Program;
class Kernel final:public amd::device::Kernel {
public:
    const Program& owner;
    const ps5hip::Kernel& native;
    Kernel(const amd::Device&,const Program&,const ps5hip::Kernel&);
};
class Program final:public amd::device::Program {
public:
    BundledCodeObject object;
    Program(amd::Device& d,amd::Program& p):amd::device::Program(d,p){}
    ~Program(){releaseClBinary();}
    bool createBinary(amd::option::Options*)override{return false;}
    bool saveBinaryAndSetType(type_t)override{return false;}
    bool createKernels(void*,size_t,bool,bool)override{return false;}
    bool setKernels(void* input,size_t size,amd::Os::FileDesc,size_t,std::string)override{
        try {
            if(!kernels().empty()||!input||!size)return false;
            BundledCodeObject candidate;
            candidate.bytes.assign(static_cast<uint8_t*>(input),static_cast<uint8_t*>(input)+size);
            candidate.object=inspect_code_object(candidate.bytes.data(),candidate.bytes.size());
            if(candidate.object.target!="gfx1013"||candidate.object.kernels.empty())return false;
            candidate.bundle_target="hip-amdgcn-amd-amdhsa--gfx1013";
            std::vector<size_t> supported;
            for(size_t i=0;i<candidate.object.kernels.size();++i){
                const auto& k=candidate.object.kernels[i];
                PublicProfile profile{};
                if(!public_profile(k,profile))continue;
                validate_capture_layout(k,profile.route);
                supported.push_back(i);
            }
            if(supported.empty())return false;
            object=std::move(candidate);
            std::vector<std::unique_ptr<Kernel>> pending;
            for(auto i:supported)pending.push_back(std::make_unique<Kernel>(device(),*this,object.object.kernels[i]));
            for(auto& k:pending){if(!kernels().emplace(k->native.name,k.get()).second)throw ObjectError("duplicate kernel");k.release();}
            return true;
        }catch(...){clear();return false;}
    }
};
Kernel::Kernel(const amd::Device& d,const Program& p,const ps5hip::Kernel& k)
    :amd::device::Kernel(d,k.name,p),owner(p),native(k){
    std::vector<amd::KernelParameterDescriptor> args(k.arguments.size());
    for(size_t i=0;i<args.size();++i){
        const auto& src=k.arguments[i];auto& dst=args[i];
        // Opaque aggregates keep compiler offsets; their byte count is not an alignment.
        dst.offset_=src.offset;dst.size_=src.size;dst.alignment_=src.size>8?1:src.size;dst.typeQualifier_=0;
        if(src.kind=="global_buffer"&&src.size==8){dst.type_=T_POINTER;dst.addressQualifier_=CL_KERNEL_ARG_ADDRESS_GLOBAL;dst.info_.oclObject_=amd::KernelParameterDescriptor::MemoryObject;}
        else if(src.kind=="by_value"&&(src.size==4||src.size==8||(src.size>8&&src.size<=ps5hip::detail::max_by_value_bytes))){dst.type_=src.size==4?T_INT:src.size==8?T_LONG:T_VOID;dst.addressQualifier_=CL_KERNEL_ARG_ADDRESS_PRIVATE;dst.info_.oclObject_=amd::KernelParameterDescriptor::ValueObject;}
        else throw ObjectError("unsupported argument kind");
    }
    if(!createSignature(args,args.size(),amd::KernelSignature::ABIVersion_2))throw ObjectError("CLR signature creation");
    PublicProfile profile{};
    if(!public_profile(k,profile))throw ObjectError("unqualified public kernel profile");
    workGroupInfo_.size_=profile.threads[0]*profile.threads[1]*profile.threads[2];workGroupInfo_.wavefrontSize_=k.wave_size;
    workGroupInfo_.availableLDSSize_=k.group_bytes+profile.dynamic_lds_limit;
    workGroupInfo_.localMemSize_=k.group_bytes;workGroupInfo_.privateMemSize_=k.private_bytes;
}
class Virtual final:public amd::device::VirtualDevice {
    std::shared_ptr<State> state_;
    static void region(amd::Memory& m,const amd::Coord3D& origin,const amd::Coord3D& size){
        if(!m.asBuffer()||!m.asBuffer()->validateRegion(origin,size)||origin[1]||origin[2])
            throw ObjectError("invalid buffer region");
    }
    void* pointer(amd::Memory& m,size_t offset){
        auto* mem=dynamic_cast<Memory*>(m.getDeviceMemory(device(),false));
        if(!mem)throw ObjectError("wrong memory owner");
        if(offset>m.getSize()||offset>UINTPTR_MAX-mem->virtualAddress())throw ObjectError("invalid buffer address");
        return reinterpret_cast<void*>(mem->virtualAddress()+offset);
    }
    template<class F> void execute(amd::Command& c,F f)noexcept {
        c.setStatus(CL_RUNNING);
        const int status=state_->operation(f);
        // Callbacks and reference destruction may reenter the device: no gate held.
        c.setStatus(status==CL_SUCCESS?CL_COMPLETE:status);
    }
    static void unsupported(amd::Command& c)noexcept{c.setStatus(CL_INVALID_OPERATION);}
public:
    Virtual(amd::Device& d,std::shared_ptr<State> s):amd::device::VirtualDevice(d),state_(std::move(s)){++state_->observed.queues;}
    ~Virtual(){++state_->observed.retired_queues;}
    // Completed commands still own batch references until an internal marker flush.
    // Force finish() to wait on a fresh retirement barrier even for completed work.
    bool isHandlerPending()const override{return true;}
    bool isFenceDirty()const override{return false;}
    void submitMarker(amd::Marker& c)override{
        if(c.type()==0){c.setStatus(CL_RUNNING);return;}
        execute(c,[]{});
    }
    void submitWriteMemory(amd::WriteMemoryCommand& c)override{execute(c,[&]{
        if(c.type()!=CL_COMMAND_WRITE_BUFFER)throw ObjectError("unsupported write");
        region(c.destination(),c.origin(),c.size());state_->data.copy_to_device(pointer(c.destination(),c.origin()[0]),c.source(),c.size()[0]);
    });}
    void submitReadMemory(amd::ReadMemoryCommand& c)override{execute(c,[&]{
        if(c.type()!=CL_COMMAND_READ_BUFFER)throw ObjectError("unsupported read");
        region(c.source(),c.origin(),c.size());state_->data.copy_from_device(c.destination(),pointer(c.source(),c.origin()[0]),c.size()[0]);
    });}
    void submitCopyMemory(amd::CopyMemoryCommand& c)override{execute(c,[&]{
        if(c.type()==CL_COMMAND_COPY_BUFFER_RECT){
            auto validate=[&](amd::Memory& memory,const amd::BufferRect& rect){
                FillLayout layout{};
                if(!memory.asBuffer()||rect.start_>memory.getSize()||!rect.rowPitch_||
                   rect.slicePitch_%rect.rowPitch_||
                   !checked_fill_layout(c.size()[0],c.size()[1],c.size()[2],rect.rowPitch_,
                                        rect.slicePitch_/rect.rowPitch_,memory.getSize()-rect.start_,layout))
                    throw ObjectError("invalid rectangular copy span");
            };
            validate(c.source(),c.srcRect());validate(c.destination(),c.dstRect());
            state_->data.copy_device_rect(pointer(c.destination(),c.dstRect().start_),
                c.dstRect().rowPitch_,c.dstRect().slicePitch_,pointer(c.source(),c.srcRect().start_),
                c.srcRect().rowPitch_,c.srcRect().slicePitch_,c.size()[0],c.size()[1],c.size()[2]);
            return;
        }
        if(c.type()!=CL_COMMAND_COPY_BUFFER)throw ObjectError("unsupported copy");
        region(c.source(),c.srcOrigin(),c.size());region(c.destination(),c.dstOrigin(),c.size());
        state_->data.copy_device(pointer(c.destination(),c.dstOrigin()[0]),pointer(c.source(),c.srcOrigin()[0]),c.size()[0]);
    });}
    void submitFillMemory(amd::FillMemoryCommand& c)override{
        if(c.type()!=CL_COMMAND_FILL_BUFFER){unsupported(c);return;}
        execute(c,[&]{
            const size_t pattern=c.patternSize(),width=c.size()[0],pitch=c.surface()[0];
            const size_t origin=c.origin()[0];
            FillLayout layout{};
            if(!c.memory().asBuffer()||c.origin()[1]||c.origin()[2]||origin>c.memory().getSize()||
               !pattern||pattern>amd::FillMemoryCommand::MaxFillPatterSize||
               (pattern&(pattern-1))||width%pattern||origin%pattern||pitch%pattern||
               c.surface()[1]<width||
               !checked_fill_layout(width,c.size()[1],c.size()[2],pitch,c.surface()[2],
                                    c.memory().getSize()-origin,layout))
                throw ObjectError("invalid buffer fill layout");
            auto hold=state_->data.resolve(pointer(c.memory(),origin),layout.span);
            auto* destination=static_cast<unsigned char*>(hold.memory->cpu())+hold.offset;
            // Resolve the entire padded span before any write, under the device gate.
            for(size_t z=0;z<c.size()[2];++z)
                for(size_t y=0;y<c.size()[1];++y)
                    for(size_t x=0;x<width;x+=pattern)
                        std::memcpy(destination+z*layout.slice+y*pitch+x,c.pattern(),pattern);
        });
    }
    void submitKernel(amd::NDRangeKernelCommand& c)override{execute(c,[&]{
        const auto* k=dynamic_cast<const Kernel*>(c.kernel().getDeviceKernel(device()));
        if(!k||!c.parameters()||c.cooperativeGroups()||c.sizes().dimensions()>3)throw ObjectError("invalid selected kernel");
        std::array<uint32_t,3> groups{1,1,1},threads{1,1,1};
        for(size_t i=0;i<c.sizes().dimensions();++i){auto n=c.sizes().global()[i],t=c.sizes().local()[i];
            if(c.sizes().offset()[i]||!n||!t||n%t||t>UINT32_MAX||n/t>UINT32_MAX)throw ObjectError("unsupported NDRange");
            groups[i]=n/t;threads[i]=t;
        }
        auto request=select_hip_launch_profile(k->native,groups,threads,c.sharedMemBytes());
        PublicProfile profile{};
        if(!public_profile(k->native,profile)||request.route!=profile.route)throw ObjectError("unqualified CLR launch route");
        const auto& signature=k->signature();std::vector<const void*> addresses;std::vector<ReadableSpan> spans;
        for(size_t i=0;i<k->native.arguments.size();++i){const auto& a=k->native.arguments[i];const auto& d=signature.at(i);
            if(d.offset_!=a.offset||d.size_!=a.size||d.offset_>signature.paramsSize()||d.size_>signature.paramsSize()-d.offset_)throw ObjectError("captured signature mismatch");
            const auto* p=reinterpret_cast<const uint8_t*>(c.parameters())+d.offset_;addresses.push_back(p);spans.push_back({p,d.size_});
        }
        auto values=capture_launch_arguments(k->native,request.route,{reinterpret_cast<const uint8_t*>(addresses.data()),addresses.size()*sizeof(void*)},spans,state_->data.ranges());
        auto launch=prepare_native_launch(k->owner.object,k->native,values,state_->data,request);
        try {
            state_->initialize();auto result=state_->queue->submit_and_wait(launch.commands,launch.resources,2000);
            if(result.status!=SubmissionStatus::Complete)throw ObjectError("native submission incomplete");
        } catch(...){if(!state_->queue||!state_->queue->poisoned())close_native_launch_temporaries(launch,state_->data);throw;}
        close_native_launch_temporaries(launch,state_->data);++state_->observed.launches;
    });}
    void flush(amd::Command* list=nullptr,bool=false)override{
        while(list){
            auto* next=list->getNext();
            // FIFO retirement releases prior commands and their memory references
            // before the internal barrier publishes completion to finish() callers.
            if(list->type()==0&&list->status()>CL_COMPLETE)execute(*list,[]{});
            else if(list->status()>CL_COMPLETE)list->setStatus(CL_INVALID_OPERATION);
            list->release();list=next;
        }
    }
#include "ps5-virtual-unsupported.inc"
};
class Device final:public amd::Device {
    std::shared_ptr<State> state_;
public:
    Device(BackendFactory factory,Observation& observed):state_(std::make_shared<State>(factory,observed)){
        settings_=new amd::device::Settings();settings_->waitCommand_=false;
        info_.queueProperties_=0;info_.type_=CL_DEVICE_TYPE_GPU;info_.available_=true;
        info_.localMemSize_=65536;info_.svmCapabilities_=CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;info_.minDataTypeAlignSize_=8;
        // Bounded device-buffer policy, not a claim about total physical memory.
        info_.maxMemAllocSize_=64u*1024u*1024u;info_.memBaseAddrAlign_=0x10000;
        info_.maxPhysicalMemAllocSize_=0;
        // Bounded qualified Wave32 profiles, including the 32x4 Q4 block.
        // These are admission limits, not the console's full hardware limits.
        info_.maxWorkGroupSize_=1024;info_.maxWorkItemDimensions_=3;
        info_.maxWorkItemSizes_[0]=1024;info_.maxWorkItemSizes_[1]=4;info_.maxWorkItemSizes_[2]=1;
        info_.wavefrontWidth_=32;info_.localMemSizePerCU_=65536;

        const auto* isa=amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx1013");
        if(!factory||!isa||!amd::Device::create(*isa))throw ObjectError("PS5 CLR device creation");
        state_->retained=state_;
    }
    bool initialize_runtime_context(){
        context_=new amd::Context({this},amd::Context::Info());
        if(context_->create(nullptr)!=CL_SUCCESS){context_->release();context_=nullptr;return false;}
        registerDevice();return true;
    }
    ~Device(){
        if(context_){context_->release();context_=nullptr;}

        if(!state_->observed.terminal&&state_->observed.allocations==state_->observed.frees&&
           state_->observed.queues==state_->observed.retired_queues&&state_.use_count()==2){
#ifdef PS5HIP_CLR_MODEL
            const int result=ps5hip_clr_model_destroy(&state_->gate);
#else
            const int result=pthread_mutex_destroy(&state_->gate);
#endif
            if(result)state_->observed.terminal=true;
            else {state_->observed.gate_destroyed=true;state_->retained.reset();}
        }
    }
    amd::device::Program* createProgram(amd::Program& p,amd::option::Options*)override{try{return new Program(*this,p);}catch(...){return nullptr;}}
    amd::device::Memory* createMemory(amd::Memory& m)const override{try{return new Memory(m,state_);}catch(...){return nullptr;}}
    amd::device::VirtualDevice* createVirtualDevice(amd::CommandQueue*)override{try{return new Virtual(*this,state_);}catch(...){return nullptr;}}
    void* svmAlloc(amd::Context& context,size_t size,size_t alignment,cl_svm_mem_flags flags,void* existing)const override {
        if(existing||!size||size>info_.maxMemAllocSize_||flags||
           (alignment&&(alignment>info_.memBaseAddrAlign_||(alignment&(alignment-1))))||
           context.devices().size()!=1||context.devices()[0]!=this||state_->observed.terminal)
            return nullptr;
        amd::Memory* owner=nullptr;
        try {
            owner=new(context) amd::Buffer(context,0,size,
                reinterpret_cast<void*>(amd::Memory::MemoryType::kSvmMemoryPtr));
            if(!owner||!owner->create(nullptr)){if(owner)owner->release();return nullptr;}
            auto* memory=dynamic_cast<Memory*>(owner->getDeviceMemory(*this,false));
            if(!memory||!owner->getSvmPtr()||owner->getSvmPtr()!=reinterpret_cast<void*>(memory->virtualAddress()))
                throw ObjectError("SVM buffer ownership mismatch");
            amd::MemObjMap::AddMemObj(owner->getSvmPtr(),owner);
            return owner->getSvmPtr();
        }catch(...){if(owner)owner->release();return nullptr;}
    }
    void svmFree(void* pointer)const override {
        auto* owner=amd::MemObjMap::FindMemObj(pointer);
        if(!owner||owner->getSvmPtr()!=pointer||owner->getContext().devices().size()!=1||
           owner->getContext().devices()[0]!=this)return;
        // Public hipFree synchronizes streams first. Direct SVM callers may not
        // retire storage still retained by an upstream command.
        if(owner->referenceCount()!=1)return;
        auto* memory=dynamic_cast<Memory*>(owner->getDeviceMemory(*this,false));
        if(!memory||!memory->retire())return;
        amd::MemObjMap::RemoveMemObj(pointer);owner->release();
    }
    bool validateKernel(const amd::Kernel& k,const amd::device::VirtualDevice* v,bool cooperative)override{
        const auto* selected=dynamic_cast<const Kernel*>(k.getDeviceKernel(*this));
        if(state_->observed.terminal||(v&&&v->device()!=this)||cooperative||!selected||!k.parameters().check())return false;
        // The single PS5 device registers only its allocations in MemObjMap. Its lookup
        // lock is independent of the native gate, which can be held by an in-flight GPU.
        const auto& signature=selected->signature();
        const auto* values=reinterpret_cast<const uint8_t*>(k.parameters().values());
        if(!values||signature.numParameters()!=selected->native.arguments.size())return false;
        for(size_t i=0;i<selected->native.arguments.size();++i){
            const auto& a=selected->native.arguments[i];const auto& d=signature.at(i);
            if(d.offset_!=a.offset||d.size_!=a.size||d.offset_>signature.paramsSize()||d.size_>signature.paramsSize()-d.offset_)return false;
            if(a.kind=="global_buffer"){
                if(d.type_!=T_POINTER||d.size_!=sizeof(void*))return false;
                const void* pointer=nullptr;std::memcpy(&pointer,values+d.offset_,sizeof(pointer));
                // Membership only: never dereference the unretained lookup result.
                // Capture retains storage later; callers must not concurrently free it.
                if(pointer&&!amd::MemObjMap::FindMemObj(pointer))return false;
            }
        }
        return true;
    }
#include "ps5-device-unsupported.inc"
};
}
amd::Device* create_device(BackendFactory factory,Observation& observed)noexcept{
    try{return new Device(factory,observed);}catch(...){return nullptr;}
}
bool register_runtime_device(BackendFactory factory,Observation& observation) noexcept {
    try {
        std::unique_ptr<Device> device(new Device(factory,observation));
        if(!device->initialize_runtime_context())return false;
        device.release();return true; // Owned by upstream Device::tearDown.
    }catch(...){return false;}
}

}
