// Included in the native canary namespace. This is a measured device canary,
// not a general multidimensional hipLaunchKernel implementation.
namespace geometry {
#ifdef PS5HIP_CANARY_GEOMETRY_3D
constexpr bool d3=true;
#else
constexpr bool d3=false;
#endif
constexpr const char *kernel=d3?"geometry_3d":"geometry_2d";
constexpr uint32_t gx=d3?2:3,gy=d3?3:2,gz=d3?4:1;
constexpr uint32_t bx=8,by=d3?4:8,bz=d3?2:1;
constexpr uint32_t sx=gx*bx,sy=gy*by,sz=gz*bz,invocations=sx*sy*sz;
constexpr uint32_t lx=d3?13:21,ly=d3?9:13,lz=d3?7:1,valid=lx*ly*lz;
constexpr uint32_t record_words=16*invocations;
uint32_t input_value(uint32_t i) { return 0x71530000u^(i*131u+17u); }
void validate(const Memory &input,const Memory &records,const Memory &tail) {
    check_guards(input,invocations);check_guards(records,record_words);check_guards(tail,invocations);
    for(uint32_t i=0;i<invocations;++i) {
        const uint32_t x=i%sx,y=(i/sx)%sy,z=i/(sx*sy);
        const uint32_t expected[16]={x/bx,y/by,z/bz,x%bx,y%by,z%bz,bx,by,bz,gx,gy,gz,x,y,z,input_value(i)};
        if(data(input)[i]!=input_value(i))throw ps5hip::ObjectError("geometry input modified");
        for(unsigned j=0;j<16;++j)
            if(data(records)[16*i+j]!=expected[j])throw ps5hip::ObjectError("geometry coordinate record mismatch");
        const uint32_t want=(x<lx && y<ly && z<lz)?input_value(i)^(x*17u+y*257u+z*65537u+0x53a971bdu):poison;
        if(data(tail)[i]!=want)throw ps5hip::ObjectError("geometry logical tail mismatch");
    }
}
void run(ps5hip::SubmissionQueue &queue,const ps5hip::CodeObject &object,const std::vector<uint8_t> &image) {
    auto module=std::make_shared<ps5hip::NativeMemory>(image.size(),65536,true);
    auto kernarg=std::make_shared<ps5hip::NativeMemory>(296,256,false);
    auto input=buffer(invocations),records=buffer(record_words),tail=buffer(invocations);
    const std::vector<Memory> resources{module,kernarg,input,records,tail};
    std::memcpy(module->cpu(),image.data(),image.size());
    for(uint32_t i=0;i<invocations;++i)data(input)[i]=input_value(i);
    const ps5hip::GeometryShape shape{{gx,gy,gz},{bx,by,bz},d3?3u:2u};
    try {
        const auto launch=ps5hip::prepare_geometry_launch(object,kernel,module->range(),kernarg->range(),shape,
            {argument(input,invocations),argument(records,record_words),argument(tail,invocations),{sx},{sy},{sz},{record_words}});
        if(launch.state.kernargs.size()!=296)throw ps5hip::ObjectError("geometry kernarg contract changed");
        std::memcpy(kernarg->cpu(),launch.state.kernargs.data(),296);
        std::printf("PS5HIP SUBMIT target=%s kernel=%s groups=%ux%ux%u threads=%ux%ux%u invocations=%u record_words=%u logical_points=%u kernarg_bytes=296 timeout_ms=2000\n",
            PS5HIP_CANARY_TARGET,kernel,gx,gy,gz,bx,by,bz,invocations,record_words,valid);
        std::fflush(nullptr);
        const auto result=queue.submit_and_wait(ps5hip::encode_geometry_dispatch(launch),resources,2000);
        if(result.status!=ps5hip::SubmissionStatus::Complete)quarantine("geometry submission failed");
        validate(input,records,tail);
    }catch(const std::exception &e){if(queue.poisoned())quarantine(e.what());throw;}
    catch(...){if(queue.poisoned())quarantine("unknown geometry exception");throw;}
    for(const auto &memory:resources)
        if(!memory->close())throw ps5hip::ObjectError("geometry allocation teardown failed");
    std::printf("PS5HIP PASS kernel=%s exact_records=%u record_words=%u logical_points=%u inactive_points=%u inputs=unchanged guards=unchanged allocations=closed\n",
        kernel,invocations,record_words,valid,invocations-valid);
}
} // namespace geometry
