#include "ps5hip/gemm.h"
#include "ps5hip/hip_runtime_control.h"
#include <hip/hip_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <limits>
extern "C" void ps5hip_matmul_model_configure(const char*);
extern "C" unsigned ps5hip_matmul_model_live(),ps5hip_matmul_model_submits();
void require(bool x){if(!x)throw std::runtime_error("GEMM oracle");}
void check(hipError_t e){require(e==hipSuccess);}
int main(int argc,char**argv){
 const char*mode=argc==2?argv[1]:"normal";ps5hip_matmul_model_configure(mode);
 void *da=nullptr,*db=nullptr,*dc=nullptr;unsigned cases=0,values=0,rejections=0;
 try{
  require(ps5hip_runtime_registration_ready());
  struct Shape{unsigned m,n,k;float alpha,beta;};
  const Shape shapes[]={{8,8,8,1,0},{13,21,19,0.7f,-0.3f},{1,17,65,-1.25f,1},{33,9,1,0.5f,0.25f},{7,11,0,1,0},{9,3,17,0,-0.5f},{65,73,129,0.9f,-0.2f}};
  for(auto shape:shapes){
   const size_t lda=shape.k+3,ldb=shape.n+5,ldc=shape.n+7,g=8;
   std::vector<float>a(g+size_t(shape.m)*lda+g,12345),b(g+size_t(shape.k)*ldb+g,23456),c(g+size_t(shape.m)*ldc+g,34567),got(c.size()),aa(a.size()),bb(b.size());
   check(hipMalloc(&da,a.size()*4));check(hipMalloc(&db,b.size()*4));check(hipMalloc(&dc,c.size()*4));
   for(unsigned round=0;round<2;++round){
    for(unsigned i=0;i<shape.m;++i)for(unsigned j=0;j<shape.k;++j)a[g+i*lda+j]=float(int((i*13+j*7+round*5)%31)-15)/11;
    for(unsigned i=0;i<shape.k;++i)for(unsigned j=0;j<shape.n;++j)b[g+i*ldb+j]=float(int((i*5+j*3+round*11)%29)-14)/13;
    for(unsigned i=0;i<shape.m;++i)for(unsigned j=0;j<shape.n;++j)c[g+i*ldc+j]=shape.beta==0?std::numeric_limits<float>::quiet_NaN():float(int((i+j+round)%9)-4)/7;
    check(hipMemcpy(da,a.data(),a.size()*4,hipMemcpyHostToDevice));check(hipMemcpy(db,b.data(),b.size()*4,hipMemcpyHostToDevice));check(hipMemcpy(dc,c.data(),c.size()*4,hipMemcpyHostToDevice));
    auto ap=static_cast<float*>(da)+g,bp=static_cast<float*>(db)+g,cp=static_cast<float*>(dc)+g;
    if(!cases){
     auto reject=[&](hipError_t e){require(e!=hipSuccess);++rejections;(void)hipGetLastError();};
     reject(ps5hipSgemmNN(8,8,8,1,ap,7,bp,ldb,0,cp,ldc));
     reject(ps5hipSgemmNN(8,8,8,1,ap,SIZE_MAX,bp,ldb,0,cp,ldc));
     reject(ps5hipSgemmNN(8,8,8,1,ap,lda,bp,ldb,0,ap,lda));
     reject(ps5hipSgemmNN(8,8,8,1,ap,lda,bp,ldb,0,cp,7));
     reject(ps5hipSgemmNN(8,8,8,1,ap,lda,bp,ldb,0,cp,10000));
     reject(ps5hipSgemmNN(8,8,8,1,reinterpret_cast<float*>(uintptr_t(ap)+1),lda,bp,ldb,0,cp,ldc));
     reject(ps5hipSgemmNN(UINT32_MAX,UINT32_MAX,8,1,ap,lda,bp,ldb,0,cp,ldc));
     reject(ps5hipSgemmNN(8,8,8,std::numeric_limits<float>::infinity(),ap,lda,bp,ldb,0,cp,ldc));
     reject(ps5hipSgemmNN(8,8,8,1,ap,lda,bp,7,0,cp,ldc));
     reject(ps5hipSgemmNN(8,8,8,1,ap,10000,bp,ldb,0,cp,ldc));
     reject(ps5hipSgemmNN(8,8,8,1,ap,lda,bp,ldb,0,nullptr,ldc));
     reject(ps5hipSgemmNN(8,8,8,1,ap,lda,bp,ldb,0,reinterpret_cast<float*>(uintptr_t(cp)+1),ldc));
     float host=0;reject(ps5hipSgemmNN(1,1,1,1,&host,1,bp,ldb,0,cp,ldc));
     check(ps5hipSgemmNN(0,9,8,1,nullptr,0,nullptr,0,0,nullptr,0));
     check(ps5hipSgemmNN(9,0,8,1,nullptr,0,nullptr,0,0,nullptr,0));
     require(!ps5hip_matmul_model_submits()&&!ps5hip_runtime_observation().backend_attempted);
    }
    auto e=ps5hipSgemmNN(shape.m,shape.n,shape.k,shape.alpha,shape.alpha==0||!shape.k?nullptr:ap,lda,shape.alpha==0||!shape.k?nullptr:bp,ldb,shape.beta,cp,ldc);
    if(ps5hip_runtime_terminal()){
     require(e!=hipSuccess&&hipFree(da)!=hipSuccess&&hipDeviceSynchronize()!=hipSuccess);
     unsigned live=ps5hip_matmul_model_live(),unresolved=ps5hip_runtime_gpu_unresolved();
     require(live==(!std::strcmp(mode,"cleanup")?4u:7u)&&unresolved==unsigned(std::strcmp(mode,"cleanup")!=0));
     std::printf("GEMM_TERMINAL live=%u unresolved=%u submits=%u\n",live,unresolved,ps5hip_matmul_model_submits());std::fflush(nullptr);std::_Exit(30);
    }
    check(e);std::puts("GEMM_ORACLE_BEGIN");check(hipMemcpy(got.data(),dc,got.size()*4,hipMemcpyDeviceToHost));check(hipMemcpy(aa.data(),da,aa.size()*4,hipMemcpyDeviceToHost));check(hipMemcpy(bb.data(),db,bb.size()*4,hipMemcpyDeviceToHost));
    require(aa==a&&bb==b);
    auto expected=c;
    for(unsigned i=0;i<shape.m;++i)for(unsigned j=0;j<shape.n;++j){
     double sum=0,magnitude=0;
     if(shape.alpha!=0)for(unsigned t=0;t<shape.k;++t){double term=double(a[g+i*lda+t])*b[g+t*ldb+j];sum+=term;magnitude+=std::abs(term);}
     double ref=double(shape.alpha)*sum;if(shape.beta!=0)ref+=double(shape.beta)*c[g+i*ldc+j];
     double tolerance=2e-6+8*std::numeric_limits<float>::epsilon()*(shape.k+2)*(std::abs(double(shape.alpha))*magnitude+std::abs(ref));
     require(std::isfinite(got[g+i*ldc+j])&&std::abs(double(got[g+i*ldc+j])-ref)<=tolerance);
     expected[g+i*ldc+j]=got[g+i*ldc+j];++values;
    }
    require(got==expected);++cases;
   }
   check(hipFree(dc));dc=nullptr;check(hipFree(db));db=nullptr;check(hipFree(da));da=nullptr;
  }
  ps5hip_runtime_finish();require(ps5hip_runtime_ready());
  std::printf("GEMM_PASS cases=%u values=%u rejections=%u synthetic_transport=true\n",cases,values,rejections);return 0;
 }catch(...){
  bool clean=true;for(void*p:{dc,db,da})if(p)clean=(hipFree(p)==hipSuccess)&&clean;
  require(clean);ps5hip_runtime_finish();std::printf("GEMM_REJECT clean=1 submits=%u\n",ps5hip_matmul_model_submits());return 20;
 }
}
