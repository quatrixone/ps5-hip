// Shared host/native copy-capture recovery checks. Caller provides checked and HIP_OK.
#pragma once
namespace copy_recovery {
inline void capture_errors(hipStream_t stream,void* a,void* b,size_t allocation_bytes=128){
 unsigned char input[128]{};
 unsigned cases=0;
 for(auto capture_mode:{hipStreamCaptureModeGlobal,hipStreamCaptureModeThreadLocal,hipStreamCaptureModeRelaxed}){
  for(unsigned api=0;api<4;++api){
   for(unsigned fault=0;fault<(api==0?3u:2u);++fault){
    HIP_OK(hipStreamBeginCapture(stream,capture_mode));
    HIP_OK(hipMemcpyAsync(a,input,128,hipMemcpyHostToDevice,stream));
    const size_t bytes=fault==1?allocation_bytes+1:128;
    hipError_t status;
    switch(api){
     case 0:status=hipMemcpyAsync(fault==0?nullptr:a,input,bytes,fault==2?static_cast<hipMemcpyKind>(99):hipMemcpyHostToDevice,stream);break;
     case 1:status=hipMemcpyHtoDAsync(fault==0?nullptr:a,input,bytes,stream);break;
     case 2:status=hipMemcpyDtoDAsync(fault==0?nullptr:b,a,bytes,stream);break;
     default:status=hipMemcpyDtoHAsync(fault==0?nullptr:input,a,bytes,stream);break;
    }
    checked(status==(fault==2?hipErrorInvalidMemcpyDirection:hipErrorInvalidValue),__LINE__);
    hipStreamCaptureStatus capture_status=hipStreamCaptureStatusNone;
    HIP_OK(hipStreamIsCapturing(stream,&capture_status));
    checked(capture_status==hipStreamCaptureStatusInvalidated,__LINE__);
    checked(hipMemcpyAsync(a,input,128,hipMemcpyHostToDevice,stream)==hipErrorStreamCaptureInvalidated,__LINE__);
    checked(hipStreamBeginCapture(stream,capture_mode)==hipErrorIllegalState,__LINE__);
    hipGraph_t failed=reinterpret_cast<hipGraph_t>(uintptr_t(1));
    checked(hipStreamEndCapture(stream,&failed)==hipErrorStreamCaptureInvalidated&&failed==nullptr,__LINE__);
    HIP_OK(hipStreamIsCapturing(stream,&capture_status));checked(capture_status==hipStreamCaptureStatusNone,__LINE__);
    // A clean subsequent capture must contain only its new copy.
    HIP_OK(hipStreamBeginCapture(stream,capture_mode));
    HIP_OK(hipMemcpyAsync(a,input,128,hipMemcpyHostToDevice,stream));
    hipGraph_t recovered=nullptr;HIP_OK(hipStreamEndCapture(stream,&recovered));
    size_t count=0;HIP_OK(hipGraphGetNodes(recovered,nullptr,&count));checked(count==1,__LINE__);
    HIP_OK(hipGraphDestroy(recovered));++cases;
   }
  }
 }
 checked(cases==27,__LINE__);
}

}
