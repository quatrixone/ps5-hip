// Reuse the existing real-memory/queue doubles; no target ISA is simulated.
#define PS5HIP_NATIVE_ROUTES_MODEL 1
#define main vector_compatibility_model
#include "native_launch_test.cpp"
#undef main

using namespace ps5hip;
BundledCodeObject read_object(const char *path) {
    std::ifstream stream(path,std::ios::binary);check(bool(stream));
    BundledCodeObject result;
    result.bytes={std::istreambuf_iterator<char>(stream),{}};
    result.object=inspect_code_object(result.bytes.data(),result.bytes.size());
    result.bundle_target="hipv4-amdgcn-amd-amdhsa--"+result.object.target;
    return result;
}
const Kernel &named(const BundledCodeObject &bundle,const char *name) {
    const auto i=std::find_if(bundle.object.kernels.begin(),bundle.object.kernels.end(),
        [&](const Kernel &k){return k.name==name;});check(i!=bundle.object.kernels.end());return *i;
}
struct PersistentData {
    NativeDataRegistry data;
    std::array<void *,3> bases;
    PersistentData(){for(auto &p:bases)p=data.allocate(262144);}
    std::vector<LaunchArgument> args(const Kernel &k,const NativeLaunchRequest &request) {
        std::vector<LaunchArgument> out;
        unsigned pointer=0,scalar=0;
        const auto &s=request.shape;
        const std::array<uint32_t,4> geometry{{s.groups[0]*s.threads[0],s.groups[1]*s.threads[1],
            s.groups[2]*s.threads[2],s.groups[0]*s.groups[1]*s.groups[2]*64*16}};
        for(const auto &a:k.arguments) {
            if(a.kind=="global_buffer") {check(pointer<3);out.push_back({uint64_t(bases[pointer++]),262144});}
            else {check(a.kind=="by_value");out.push_back({request.route==NativeLaunchRoute::Geometry?geometry.at(scalar++):5,0});}
        }
        return out;
    }
};
struct RouteBackend:TransportModel {
    int32_t submit(uint32_t *words,uint32_t count) override {
        if(failure==5){++submits;throw 17;}
        return TransportModel::submit(words,count);
    }
    int32_t kick() override {return failure==6?-7:0;}
    uint64_t now_ns() override {if(failure==7)throw 19;return clock;}
};
template<class F> void child_check(F test) {
    const auto child=fork();check(child>=0);
    if(!child){try{test();_exit(0);}catch(const std::exception &e){std::cerr<<e.what()<<'\n';_exit(1);}catch(...){_exit(2);}}
    int status=0;check(waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0);
}
int main(int argc,char **argv) {
    try {
        check(argc==5);
        auto resources=read_object(argv[1]),lds=read_object(argv[2]),geometry=read_object(argv[3]),private_call=read_object(argv[4]);
        struct Case {const BundledCodeObject *bundle;const char *name;NativeLaunchRequest request;};
        const NativeLaunchRequest spill{NativeLaunchRoute::BufferScratch,{{1,1,1},{64,1,1},1},0};
        const std::vector<Case> cases{
            {&resources,"reduce_serial",{NativeLaunchRoute::Initial,{{1,1,1},{64,1,1},1},0}},
            {&resources,"register_spill",spill},
            {&lds,"lds_static",{NativeLaunchRoute::Lds,{{257,1,1},{64,1,1},1},0}},
            {&lds,"lds_dynamic",{NativeLaunchRoute::Lds,{{257,1,1},{64,1,1},1},1280}},
            {&lds,"lds_mixed",{NativeLaunchRoute::Lds,{{257,1,1},{64,1,1},1},1280}},
            {&geometry,"geometry_2d",{NativeLaunchRoute::Geometry,{{3,2,1},{8,8,1},2},0}},
            {&geometry,"geometry_3d",{NativeLaunchRoute::Geometry,{{2,3,4},{8,4,2},3},0}}};
        {
            PersistentData buffers;RouteBackend backend;backend.data=&buffers.data;backend.base=buffers.bases[0];
            SubmissionQueue queue(backend);
            for(unsigned round=0;round<2;++round)for(const auto &c:cases) {
                const auto &k=named(*c.bundle,c.name);auto args=buffers.args(k,c.request);
                auto launch=prepare_native_launch(*c.bundle,k,args,buffers.data,c.request);
                const bool scratch=c.request.route==NativeLaunchRoute::BufferScratch;
                check(launch.temporaries.size()==(scratch?3:2));
                for(size_t i=0;i<launch.temporaries.size();++i)check(launch.resources.at(i)==launch.temporaries[i]);
                const auto *packed=static_cast<const uint8_t*>(launch.temporaries[1]->cpu());
                for(size_t i=0;i<args.size();++i)check(get(packed+k.arguments[i].offset,k.arguments[i].size)==args[i].value);
                if(scratch)check(launch.temporaries[2]->range().bytes>=buffer_scratch_ring_bytes(k.private_bytes));
                check(queue.submit_and_wait(launch.commands,launch.resources,1).status==SubmissionStatus::Complete);
                for(const auto &m:launch.resources)check(!m->gpu_in_use());
                close_native_launch_temporaries(launch,buffers.data);
                check(allocations.size()==3 && !buffers.data.poisoned());
                for(auto p:buffers.bases){auto h=buffers.data.resolve(p,262144);const auto *b=static_cast<const uint8_t*>(p);
                    check(h.remaining==262144 && std::all_of(b,b+262144,[](uint8_t x){return x==0xa5;}));}
                reject([&]{queue.submit_and_wait(launch.commands,launch.resources,1);});
            }
            check(backend.submits==14);
            const auto before=allocate_calls;
            for(const auto &c:cases) {
                const auto &k=named(*c.bundle,c.name);auto args=buffers.args(k,c.request);
                auto bad=c.request;bad.route=NativeLaunchRoute(99);reject([&]{prepare_native_launch(*c.bundle,k,args,buffers.data,bad);});
                bad=c.request;bad.shape.groups[0]=0;reject([&]{prepare_native_launch(*c.bundle,k,args,buffers.data,bad);});
                bad=c.request;bad.shape.threads={64,2,1};reject([&]{prepare_native_launch(*c.bundle,k,args,buffers.data,bad);});
                if(c.request.route!=NativeLaunchRoute::Lds){bad=c.request;bad.dynamic_lds_bytes=256;reject([&]{prepare_native_launch(*c.bundle,k,args,buffers.data,bad);});}
                auto wrong=args;--wrong[0].buffer_bytes;reject([&]{prepare_native_launch(*c.bundle,k,wrong,buffers.data,c.request);});
                wrong=args;wrong[0].value=UINT64_MAX;reject([&]{prepare_native_launch(*c.bundle,k,wrong,buffers.data,c.request);});
                auto copy=k;reject([&]{prepare_native_launch(*c.bundle,copy,args,buffers.data,c.request);});
                auto mutated=*c.bundle;auto &selected=mutated.object.kernels[&k-c.bundle->object.kernels.data()];
                selected.arguments[0].size=16;reject([&]{prepare_native_launch(mutated,selected,args,buffers.data,c.request);});
            }
            for(const auto &k:private_call.object.kernels)
                for(auto route:{NativeLaunchRoute::Initial,NativeLaunchRoute::Lds,NativeLaunchRoute::BufferScratch,NativeLaunchRoute::Geometry}) {
                    auto r=spill;r.route=route;reject([&]{prepare_native_launch(private_call,k,{},buffers.data,r);});
                }
            check(allocate_calls==before);
        }
        check(allocations.empty());
        const auto &k=named(resources,"register_spill");
        for(unsigned stage:{1u,2u,3u})for(bool map:{false,true}) {
            PersistentData buffers;auto args=buffers.args(k,spill);
            (map?fail_map_at:fail_allocate_at)=allocate_calls+stage;
            reject([&]{prepare_native_launch(resources,k,args,buffers.data,spill);});
            fail_map_at=fail_allocate_at=0;check(!buffers.data.poisoned() && allocations.size()==3);
        }
        check(allocations.empty());
        for(int mode=1;mode<=7;++mode)child_check([&] {
            PersistentData buffers;auto args=buffers.args(k,spill);
            auto launch=prepare_native_launch(resources,k,args,buffers.data,spill);
            RouteBackend backend;backend.failure=mode;backend.data=&buffers.data;backend.base=buffers.bases[0];SubmissionQueue queue(backend);
            bool threw=false;
            try {auto result=queue.submit_and_wait(launch.commands,launch.resources,1);check(result.status!=SubmissionStatus::Complete);}
            catch(...){threw=true;}
            if(mode==1){check(threw && !queue.poisoned());close_native_launch_temporaries(launch,buffers.data);check(allocations.size()==3);}
            else {
                check(queue.poisoned());buffers.data.poison();
                for(const auto &m:launch.resources)check(m->gpu_in_use());
                const auto released=releases;reject([&]{close_native_launch_temporaries(launch,buffers.data);});
                check(releases==released && allocations.size()==8); // 3 persistent data +3temps +2queue
            }
        });
        for(unsigned stage:{1u,2u,3u})child_check([&] {
            PersistentData buffers;auto args=buffers.args(k,spill);
            alias_map_at=allocate_calls+stage;
            reject([&]{prepare_native_launch(resources,k,args,buffers.data,spill);});
            check(buffers.data.poisoned());
        });
        for(bool release:{false,true})for(bool partial:{false,true})child_check([&] {
            PersistentData buffers;auto args=buffers.args(k,spill);
            if(partial) {
                fail_map_at=allocate_calls+3;fail_release=release;fail_unmap=!release;
                try{prepare_native_launch(resources,k,args,buffers.data,spill);check(false);}
                catch(const NativeMemoryCleanupError &){check(buffers.data.poisoned());}
                catch(const NativeMemorySetupError &e){check(e.retained_backing && buffers.data.poisoned());}
            } else {
                auto launch=prepare_native_launch(resources,k,args,buffers.data,spill);
                RouteBackend backend;backend.data=&buffers.data;backend.base=buffers.bases[0];SubmissionQueue queue(backend);
                check(queue.submit_and_wait(launch.commands,launch.resources,1).status==SubmissionStatus::Complete);
                fail_release=release;fail_unmap=!release;
                try{close_native_launch_temporaries(launch,buffers.data);check(false);}
                catch(const NativeMemoryCleanupError &){check(buffers.data.poisoned() && !queue.poisoned());}
                for(const auto &m:launch.resources)check(!m->gpu_in_use());
            }
        });
        check(allocations.empty());
        std::cout<<"owned native routes:14 persistent-data submissions, exact temporaries, rejection/setup/queue/cleanup models passed; no ISA execution\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
