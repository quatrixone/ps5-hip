// Included after the unchanged generic kernels: genuine HIP syntax, one device body.
int main(int argc,char **argv) {
    (void)argc;(void)argv;
#ifdef PS5HIP_RUNTIME_MODEL
    ps5hip_generic_model_configure(argc==2?argv[1]:"normal");
#endif
    void *a=nullptr,*b=nullptr;
    try {
        int count=-1,device=-1,wave=-1;
        require(hipGetDeviceCount(&count)==hipSuccess && count==1);
        require(hipInit(0)==hipSuccess && hipSetDevice(0)==hipSuccess);
        require(hipGetDevice(&device)==hipSuccess && device==0);
        require(hipDeviceGetAttribute(&wave,hipDeviceAttributeWarpSize,0)==hipSuccess && wave==32);
        require(ps5hip_runtime_registration_ready());
        const auto before=ps5hip_runtime_observation();
        require(before.modules==1 && before.functions==2 && !before.completed && !before.backend_attempted);
        std::puts("FILL_DISCOVERY devices=1 selected=0 wave=32 backend_attempted=false");
        unsigned launches=0,values=0,words_checked=0,fills=0;
        for(unsigned n:{1u,73u,257u}) {
            constexpr unsigned guard=7;
            const size_t words=n+2*guard,bytes=words*sizeof(unsigned);
            std::vector<unsigned> input(words,poison),expected(words,poison),observed(words),after(words);
            require(hipMalloc(&a,bytes)==hipSuccess && hipMalloc(&b,bytes)==hipSuccess);
            for(unsigned round=0;round<2;++round) {
                require(hipMemset(a,0xa5,bytes)==hipSuccess);++fills;
                require(hipMemset(static_cast<unsigned*>(a)+guard,round?0x5c:0x123,n*4)==hipSuccess);++fills;
                require(hipMemset(reinterpret_cast<unsigned char*>(a)+guard*4+1,round?0x106:-1,3)==hipSuccess);++fills;
                require(hipMemset(b,0xa5,bytes)==hipSuccess);++fills;
                for(unsigned i=0;i<n;++i) input[guard+i]=round?0x5c5c5c5cu:0x23232323u;
                input[guard]=round?0x0606065cu:0xffffff23u;
                const uint64_t salt=0x13579bdf00000011ull+round;
                for(unsigned i=0;i<n;++i)expected[guard+i]=(input[guard+i]+unsigned(salt))^unsigned(salt>>32);
                affine_words<<<(n+63)/64,64>>>(static_cast<unsigned*>(a)+guard,static_cast<unsigned*>(b)+guard,n,salt);
                require(hipGetLastError()==hipSuccess && hipDeviceSynchronize()==hipSuccess);
                require(hipMemcpy(observed.data(),b,bytes,hipMemcpyDeviceToHost)==hipSuccess);
                require(hipMemcpy(after.data(),a,bytes,hipMemcpyDeviceToHost)==hipSuccess);
                require(observed==expected && after==input);
                ++launches;values+=n;words_checked+=unsigned(words);
                std::printf("FILL_ROUND index=%u n=%u round=%u fills=4 exact=true input_unchanged=true guards=true\n",launches-1,n,round);
            }
            require(hipFree(a)==hipSuccess);a=nullptr;require(hipFree(b)==hipSuccess);b=nullptr;
        }
        require(launches==6 && values==662 && words_checked==746 && fills==24);
        ps5hip_runtime_finish();require(ps5hip_runtime_ready());
        std::puts("FILL_CLIENT_PASS dispatches=6 exact_values=662 output_words=746 input_words=746 fills=24");return 0;
    }catch(...) {
        if(ps5hip_runtime_terminal() || ps5hip_runtime_gpu_unresolved()) {
            std::puts("FILL_CLIENT_FAILURE terminal=true retained=true");return 30;
        }
        bool clean=true;
        if(a){clean=hipFree(a)==hipSuccess&&clean;a=nullptr;}
        if(b){clean=hipFree(b)==hipSuccess&&clean;b=nullptr;}
        std::printf("FILL_CLIENT_FAILURE terminal=false clean=%u\n",unsigned(clean));
        ps5hip_runtime_finish();return 20;
    }
}
