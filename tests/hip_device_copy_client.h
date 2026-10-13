// Host-only experiment; both genuine device bodies remain in hip_generic.hip.
int main(int argc,char **argv) {
    (void)argc;(void)argv;
#ifdef PS5HIP_RUNTIME_MODEL
    ps5hip_generic_model_configure(argc==2?argv[1]:"normal");
#endif
    void *owned[4]{};
    try {
        unsigned dispatches=0,copies=0,copied=0,values=0,comparisons=0,guards=0,allocated=0,freed=0;
        for(unsigned n:{1u,73u,257u})for(unsigned topology=0;topology<2;++topology) {
            constexpr unsigned guard=7;
            const unsigned lengths[4]{n+14,topology?2*n+21:n+14,n+14,n+14};
            const unsigned owners=topology?3:4;
            std::vector<unsigned> expected[4],observed[4];
            for(unsigned j=0;j<owners;++j) {
                expected[j].assign(lengths[j],poison);observed[j].resize(lengths[j]);
                require(hipMalloc(&owned[j],lengths[j]*4)==hipSuccess);++allocated;
                require(hipMemset(owned[j],0xa5,lengths[j]*4)==hipSuccess);
            }
            auto *seed=static_cast<unsigned*>(owned[0])+guard;
            auto *source=static_cast<unsigned*>(owned[1])+guard;
            auto *destination=topology?static_cast<unsigned*>(owned[1])+n+14:static_cast<unsigned*>(owned[2])+guard;
            auto *result=static_cast<unsigned*>(owned[topology?2:3])+guard;
            std::vector<unsigned> previous(n);
            for(unsigned round=0;round<2;++round) {
                const uint64_t producer=0x13579bdf00000011ull+round,consumer=0x2468ace000000071ull+round;
                for(unsigned i=0;i<n;++i) {
                    const unsigned value=(i*37u+n*101u+topology*1009u+round*65521u)^0x951dac37u;
                    const unsigned produced=(value+unsigned(producer))^unsigned(producer>>32);
                    if(round)require(produced!=previous[i]);previous[i]=produced;
                    expected[0][guard+i]=value;expected[1][guard+i]=produced;
                    expected[topology?1:2][(topology?n+14:guard)+i]=produced;
                    expected[topology?2:3][guard+i]=(produced+unsigned(consumer))^unsigned(consumer>>32);
                }
                require(hipMemcpy(seed,expected[0].data()+guard,n*4,hipMemcpyHostToDevice)==hipSuccess);
                affine_words<<<(n+63)/64,64>>>(seed,source,n,producer);
                require(hipGetLastError()==hipSuccess && hipDeviceSynchronize()==hipSuccess);++dispatches;
                require(hipMemcpy(destination,source,n*4,hipMemcpyDeviceToDevice)==hipSuccess);++copies;copied+=n;
                affine_words<<<(n+63)/64,64>>>(destination,result,n,consumer);
                require(hipGetLastError()==hipSuccess && hipDeviceSynchronize()==hipSuccess);++dispatches;
                for(unsigned j=0;j<owners;++j) {
                    require(hipMemcpy(observed[j].data(),owned[j],lengths[j]*4,hipMemcpyDeviceToHost)==hipSuccess);
                    require(observed[j]==expected[j]);comparisons+=lengths[j];
                }
                values+=2*n;guards+=topology?49:56;
                std::printf("COPY_ROUND index=%u topology=%s n=%u round=%u exact=true source_unchanged=true guards=true\n",copies-1,topology?"combined":"distinct",n,round);
            }
            for(unsigned j=0;j<owners;++j){require(hipFree(owned[j])==hipSuccess);owned[j]=nullptr;++freed;}
        }
        require(dispatches==24&&copies==12&&copied==1324&&values==2648&&comparisons==5926&&guards==630&&allocated==21&&freed==21);
        ps5hip_runtime_finish();require(ps5hip_runtime_ready());
        std::puts("COPY_CLIENT_PASS dispatches=24 copies=12 copied_words=1324 copied_bytes=5296 exact_values=2648 readback_words=5926 guard_words=630 allocations=21 frees=21");return 0;
    }catch(...) {
        if(ps5hip_runtime_terminal()||ps5hip_runtime_gpu_unresolved()) {
            std::puts("COPY_CLIENT_FAILURE terminal=true retained=true");return 30;
        }
        bool clean=true;
        for(auto &p:owned)if(p){clean=hipFree(p)==hipSuccess&&clean;p=nullptr;}
        std::printf("COPY_CLIENT_FAILURE terminal=false clean=%u\n",unsigned(clean));
        ps5hip_runtime_finish();return 20;
    }
}
