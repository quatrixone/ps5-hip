// Existing real host mappings and queue doubles; no shader execution model.
#define PS5HIP_NATIVE_ROUTES_MODEL 1
#define main vector_compatibility_model
#include "native_launch_test.cpp"
#undef main
#include "ps5hip/hip_launch_profile.h"
using namespace ps5hip;
template<class F> void in_child(F f) {
    const auto pid=fork();check(pid>=0);
    if(!pid){try{f();_exit(0);}catch(...){_exit(1);}}
    int status=0;check(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0);
}
int main(int argc,char **argv) {
    try {
        check(argc==2);std::ifstream in(argv[1],std::ios::binary);check(bool(in));
        BundledCodeObject bundle;bundle.bytes={std::istreambuf_iterator<char>(in),{}};
        bundle.object=inspect_code_object(bundle.bytes.data(),bundle.bytes.size());check(bundle.object.kernels.size()==1);
        const auto &k=bundle.object.kernels[0];
        const auto request=select_hip_launch_profile(k,{3,2,1},{8,8,1},0);check(request.route==NativeLaunchRoute::GeometryLds);
        auto arguments=[&](Buffers &b) {
            std::array<uint32_t,3> sizes{3,5,7};std::array<void *,6> pointers{&b.pointers[0],&b.pointers[1],&b.pointers[2],&sizes[0],&sizes[1],&sizes[2]};
            std::vector<ReadableSpan> spans;
            for(unsigned i=0;i<6;++i)spans.push_back({static_cast<const uint8_t*>(pointers[i]),i<3?8u:4u});
            return capture_launch_arguments(k,request.route,{reinterpret_cast<const uint8_t*>(pointers.data()),sizeof(pointers)},spans,b.data.ranges());
        };
        {
            Buffers b;auto args=arguments(b);TransportModel backend;backend.data=&b.data;backend.base=b.bases[0];SubmissionQueue queue(backend);
            for(unsigned round=0;round<2;++round) {
                auto launch=prepare_native_launch(bundle,k,args,b.data,request);
                check(launch.temporaries.size()==2 && launch.resources.size()==5 && allocations.size()==5);
                const auto *packed=static_cast<const uint8_t*>(launch.temporaries[1]->cpu());
                for(size_t i=0;i<args.size();++i)check(get(packed+k.arguments[i].offset,k.arguments[i].size)==args[i].value);
                check(get(packed+40,4)==3 && get(packed+44,4)==2 && get(packed+52,2)==8 && get(packed+54,2)==8);
                check(queue.submit_and_wait(launch.commands,launch.resources,1).status==SubmissionStatus::Complete);
                close_native_launch_temporaries(launch,b.data);check(!b.data.poisoned());
                for(const auto &m:launch.resources)check(!m->gpu_in_use());
                reject([&]{queue.submit_and_wait(launch.commands,launch.resources,1);});
            }
            check(backend.submits==2);const auto before=allocate_calls;
            for(unsigned mutation=0;mutation<6;++mutation) {auto bad=request;
                if(mutation==0)bad.dynamic_lds_bytes=1;else if(mutation==1)bad.shape.dimensions=3;
                else if(mutation==2)bad.shape.groups[1]=0;else if(mutation==3)bad.shape.groups[0]=UINT32_MAX;
                else if(mutation==4)bad.shape.threads={8,4,2};else bad.route=NativeLaunchRoute::Geometry;
                reject([&]{prepare_native_launch(bundle,k,args,b.data,bad);});}
            auto wrong=args;--wrong[0].buffer_bytes;reject([&]{prepare_native_launch(bundle,k,wrong,b.data,request);});
            auto changed=bundle;changed.object.kernels[0].group_bytes=1024;
            reject([&]{prepare_native_launch(changed,changed.object.kernels[0],args,b.data,request);});
            check(before==allocate_calls);
        }
        check(allocations.empty());
        for(unsigned stage:{1u,2u})for(bool mapping:{false,true}) {
            Buffers b;auto args=arguments(b);(mapping?fail_map_at:fail_allocate_at)=allocate_calls+stage;
            reject([&]{prepare_native_launch(bundle,k,args,b.data,request);});
            fail_map_at=fail_allocate_at=0;check(allocations.size()==3 && !b.data.poisoned());
        }
        for(int mode=1;mode<=4;++mode)in_child([&] {
            Buffers b;auto args=arguments(b);auto launch=prepare_native_launch(bundle,k,args,b.data,request);
            TransportModel backend;backend.data=&b.data;backend.base=b.bases[0];backend.failure=mode;SubmissionQueue queue(backend);
            bool threw=false;try {check(queue.submit_and_wait(launch.commands,launch.resources,1).status!=SubmissionStatus::Complete);}catch(const ObjectError &){threw=true;}
            if(mode==1){check(threw && !queue.poisoned());close_native_launch_temporaries(launch,b.data);}
            else {check(queue.poisoned());b.data.poison();const auto before=releases;
                reject([&]{close_native_launch_temporaries(launch,b.data);});check(releases==before && allocations.size()==7);}
        });
        for(bool release:{false,true})in_child([&] {
            Buffers b;auto args=arguments(b);auto launch=prepare_native_launch(bundle,k,args,b.data,request);
            fail_release=release;fail_unmap=!release;
            try {close_native_launch_temporaries(launch,b.data);check(false);}catch(const NativeMemoryCleanupError &){check(b.data.poisoned());}
        });
        check(allocations.empty());std::cout<<"native geometry LDS ownership/setup/queue/cleanup checks passed; no ISA execution\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
