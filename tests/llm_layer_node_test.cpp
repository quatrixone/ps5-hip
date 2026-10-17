#include "layer_dispatch.h"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cstdio>
static unsigned launches=0;
static void* expected_input=nullptr;
extern "C" hipError_t hipModuleLoad(hipModule_t* m,const char*){*m=(hipModule_t)1;return hipSuccess;}
extern "C" hipError_t hipModuleGetFunction(hipFunction_t* f,hipModule_t,const char* name){
 for(unsigned i=0;i<12;++i)if(!std::strcmp(name,ps5hip_layer::Dispatch::symbols[i])){*f=(hipFunction_t)(uintptr_t(i+10));return hipSuccess;}
 *f=(hipFunction_t)1;return hipSuccess;
}
extern "C" hipError_t hipModuleUnload(hipModule_t){return hipSuccess;}
extern "C" hipError_t hipMalloc(void** p,size_t n){*p=std::malloc(n);return *p?hipSuccess:hipErrorOutOfMemory;}
extern "C" hipError_t hipFree(void* p){std::free(p);return hipSuccess;}
extern "C" hipError_t hipMemcpy(void* d,const void* s,size_t n,hipMemcpyKind){std::memmove(d,s,n);return hipSuccess;}
extern "C" hipError_t hipDeviceSynchronize(){return hipSuccess;}
extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned,hipStream_t,void** a,void**){
 if(uintptr_t(f)==1){assert(gx==1&&gy==1&&gz==1&&bx==1024);++launches;auto* x=*(float**)a[0];auto* out=*(float**)a[1];double sum=0;for(int i=0;i<2048;++i)sum+=double(x[i])*x[i];float scale=1/std::sqrt(float(sum/2048)+*(float*)a[8]);for(int i=0;i<2048;++i)out[i]=scale*x[i];return hipSuccess;}
 if(uintptr_t(f)==16){++launches;auto* x=*(float**)a[0];auto* out=*(float**)a[2];assert(x!=out);unsigned n=*(unsigned*)a[3];for(unsigned i=0;i<n*32;++i)out[i]=x[i]+1;return hipSuccess;}
 if(uintptr_t(f)==18&&*(unsigned*)a[2]!=2048){++launches;std::memcpy(*(void**)a[1],*(void**)a[0],*(unsigned*)a[2]*4);return hipSuccess;}
 assert(gx==32&&gy==1&&gz==1&&bx==64&&by==1&&bz==1);++launches;
 auto* x=*(float**)a[0];assert(x==expected_input);
 if(uintptr_t(f)==11){auto* y=*(float**)a[1];auto* out=*(float**)a[2];assert(*(unsigned*)a[3]==2048);for(unsigned i=0;i<2048;++i)out[i]=x[i]*y[i];}
 else {assert(uintptr_t(f)==18&&*(unsigned*)a[2]==2048);std::memcpy(*(void**)a[1],x,8192);}
 return hipSuccess;
}
static bool owned(const ggml_tensor* t){return t&&t->data;}
static void failure(const char*,hipError_t){std::abort();}
int main(){
 auto* ctx=ggml_init({512*1024,nullptr,false});assert(ctx);
 auto* grandparent=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2048);
 auto* normalized=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2048);
 auto* weights=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2048);
 auto* output=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2048);
 auto* ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,1);
 for(int i=0;i<2048;++i){((float*)grandparent->data)[i]=99;((float*)normalized->data)[i]=3;((float*)weights->data)[i]=7;}
 normalized->op=GGML_OP_RMS_NORM;normalized->src[0]=grandparent;
 output->op=GGML_OP_MUL;output->src[0]=normalized;output->src[1]=weights;
 ps5hip_layer::Dispatch dispatch(owned,failure);dispatch.initialize({"a","b","c","d","e","f"});expected_input=normalized->data;
 assert(dispatch.execute_node(output,true)&&launches==1&&dispatch.expected_position==0);
 for(int i=0;i<2048;++i)assert(((float*)output->data)[i]==21);
 output->op=GGML_OP_GET_ROWS;output->src[1]=ids;*(int32_t*)ids->data=0;
 assert(dispatch.execute_node(output,true)&&launches==2&&dispatch.expected_position==0);
 assert(!std::memcmp(output->data,normalized->data,8192));
 *(int32_t*)ids->data=1;assert(!dispatch.execute_node(output,true)&&launches==2);
 *(int32_t*)ids->data=-1;assert(!dispatch.execute_node(output,true)&&launches==2);
 *(int32_t*)ids->data=0;assert(!dispatch.execute_node(output,false)&&launches==2);
 ids->type=GGML_TYPE_F32;assert(!dispatch.execute_node(output,true)&&launches==2);ids->type=GGML_TYPE_I32;
 void* saved=output->data;output->data=(char*)normalized->data+4;assert(!dispatch.execute_node(output,true)&&launches==2);output->data=saved;
 output->op=GGML_OP_MUL;output->src[1]=weights;weights->ne[0]=1024;assert(!dispatch.execute_node(output,true)&&launches==2);weights->ne[0]=2048;
 // Exact in-place elementwise operations and RMS are admitted; shifted aliases are not.
 output->data=normalized->data;assert(dispatch.execute_node(output,true)&&launches==3);
 for(int i=0;i<2048;++i)assert(((float*)normalized->data)[i]==21);output->data=saved;
 float epsilon=1e-5f;std::memcpy(normalized->op_params,&epsilon,4);void* norm_saved=normalized->data;
 normalized->data=grandparent->data;assert(dispatch.execute_node(normalized,true)&&launches==4);
 for(int i=0;i<2048;++i)assert(std::fabs(((float*)grandparent->data)[i]-1.f)<1e-6f);
 normalized->data=(char*)grandparent->data+4;assert(!dispatch.execute_node(normalized,true)&&launches==4);normalized->data=norm_saved;
 auto* scores=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,32);
 auto* probabilities=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,256,32);
 auto* mask=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,256);
 probabilities->op=GGML_OP_SOFT_MAX;probabilities->src[0]=scores;probabilities->src[1]=mask;
 probabilities->data=scores->data;
 for(unsigned i=0;i<8192;++i)((float*)scores->data)[i]=float(i);
 assert(!dispatch.execute_node(probabilities,true)&&launches==4);
 dispatch.full_scratch=std::malloc(65536);assert(dispatch.full_scratch);
 assert(dispatch.execute_node(probabilities,true)&&launches==6);
 for(unsigned i=0;i<8192;++i)assert(((float*)scores->data)[i]==float(i)+1);
 std::free(dispatch.full_scratch);dispatch.full_scratch=nullptr;
 dispatch.shutdown();ggml_free(ctx);std::puts("PASS direct MUL input, F32 gather and six rejection cases; no GPU proof");
}
