#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <cstring>
#include <limits>
#include <algorithm>
static unsigned lane=0;
#define __attribute__(x)
#define __builtin_amdgcn_workgroup_id_x() (lane/64)
#define __builtin_amdgcn_workitem_id_x() (lane%64)
extern "C" float __ocml_exp_f32(float x){return std::exp(x);}
extern "C" float __ocml_sin_f32(float x){return std::sin(x);}
extern "C" float __ocml_cos_f32(float x){return std::cos(x);}
extern "C" float __ocml_pow_f32(float x,float y){return std::pow(x,y);}
#include "../kernels/llm_layer.hip"

static void conversion_extremes() {
 const float tiny=std::ldexp(1.f,-24),halfway=std::ldexp(1.f,-25);
 const float input[]={0.f,-0.f,tiny,-tiny,halfway,-halfway,3*halfway,-3*halfway,65504.f,-65504.f};
 _Float16 output[11]{};output[10]=17;
 for(lane=0;lane<64;++lane)layer_to_f16(input,output,10);
 assert(float(output[0])==0&&!std::signbit(float(output[0])));
 assert(float(output[1])==0&&std::signbit(float(output[1])));
 assert(float(output[2])==tiny&&float(output[3])==-tiny);
 assert(float(output[4])==0&&float(output[5])==0);
 assert(float(output[6])==2*tiny&&float(output[7])==-2*tiny);
 assert(float(output[8])==65504.f&&float(output[9])==-65504.f&&float(output[10])==17);
}
static void silu_extremes() {
 const float input[]={-1000.f,-80.f,-20.f,-1.f,-0.f,0.f,1.f,20.f,80.f,1000.f};
 float output[11]{};output[10]=17;
 for(lane=0;lane<64;++lane)layer_silu(input,output,10);
 for(unsigned i=0;i<10;++i){
  const double x=input[i];const double target=x<0?x*std::exp(x)/(1+std::exp(x)):x/(1+std::exp(-x));
  assert(std::isfinite(output[i]));
  assert(std::abs(double(output[i])-target)<=4*std::numeric_limits<float>::epsilon()*std::abs(target)+std::numeric_limits<float>::denorm_min());
 }
 assert(std::signbit(output[0])&&std::signbit(output[4])&&!std::signbit(output[5])&&output[10]==17);
}
static void softmax_extremes() {
 for(unsigned n:{256u,512u}){
  std::vector<float>x(32*n,std::numeric_limits<float>::max()),mask(n,-INFINITY),out(32*n+1,17);
  for(unsigned t=0;t<5;++t)mask[t]=0;
  for(unsigned h=0;h<32;++h)for(unsigned t=0;t<5;++t)
   x[h*n+t]=(h%2?1.e6f:-1.e6f)+(h%4?float(int((t*3+h)%5)-2)*8.f:0.f);
  for(lane=0;lane<64;++lane)layer_softmax(x.data(),mask.data(),out.data(),n);
  for(unsigned h=0;h<32;++h){
   double maximum=-INFINITY;for(unsigned t=0;t<5;++t)maximum=std::max(maximum,double(x[h*n+t])*.125);
   double sum=0;for(unsigned t=0;t<5;++t)sum+=std::exp(double(x[h*n+t])*.125-maximum);
   double actual_sum=0;
   for(unsigned t=0;t<n;++t){double expected=t<5?std::exp(double(x[h*n+t])*.125-maximum)/sum:0;
    assert(std::isfinite(out[h*n+t])&&out[h*n+t]>=0);
    assert(std::abs(out[h*n+t]-expected)<=2e-6);if(t>=5)assert(out[h*n+t]==0);actual_sum+=out[h*n+t];
   }
   assert(std::abs(actual_sum-1)<=2e-6);
  }
  assert(out.back()==17);
 }
}
static void dense_attention_extremes() {
 for(unsigned n:{256u,512u}){
  std::vector<_Float16>k(131072),v(131072),q(2048),p(32*n);
  for(unsigned h=0;h<4;++h)for(unsigned t=0;t<512;++t)for(unsigned d=0;d<64;++d){
   k[t*256+h*64+d]=(_Float16)((d%2?-1.f:1.f)*(d<2?65504.f:float(h+1)*(t%7+1)/32.f));
   v[(h*64+d)*512+t]=(_Float16)((t%2?-1.f:1.f)*(d==0?65504.f:float(h+1)*(d+1)/64.f));
  }
  for(unsigned h=0;h<32;++h){for(unsigned d=0;d<64;++d)q[h*64+d]=(_Float16)((h==0?65504.f:(h%5+1)/8.f));
   p[h*n]=(_Float16).25f;p[h*n+127]=(_Float16).5f;p[h*n+n-1]=(_Float16).25f;}
  std::vector<float>scores(32*n+1,17),values(2049,17);
  for(lane=0;lane<32*n;++lane)layer_scores(k.data(),q.data(),scores.data(),n);
  for(lane=0;lane<2048;++lane)layer_values(v.data(),p.data(),values.data(),n);
  const double u=std::ldexp(1.,-24),score_gamma=64*u/(1-64*u),value_gamma=n*u/(1-n*u);
  for(unsigned h=0;h<32;++h){for(unsigned t=0;t<n;++t){double expected=0,magnitude=0;
    for(unsigned d=0;d<64;++d){double product=double(k[t*256+(h/8)*64+d])*double(q[h*64+d]);expected+=product;magnitude+=std::abs(product);}
    assert(std::isfinite(scores[h*n+t])&&std::abs(scores[h*n+t]-expected)<=score_gamma*magnitude);
   }
   for(unsigned d=0;d<64;++d){double expected=0,magnitude=0;for(unsigned t=0;t<n;++t){double product=double(v[((h/8)*64+d)*512+t])*double(p[h*n+t]);expected+=product;magnitude+=std::abs(product);}
    assert(std::isfinite(values[h*64+d])&&std::abs(values[h*64+d]-expected)<=value_gamma*magnitude);
   }
  }
  assert(scores.back()==17&&values.back()==17);
 }
}
static void rope_extremes() {
 for(unsigned heads:{4u,32u})for(unsigned position:{0u,1u,255u,256u,511u}){
  std::vector<float>x(heads*64),out(heads*64+1,17);
  for(unsigned j=0;j<x.size();++j)x[j]=(j%2?-1.f:1.f)*(j%4<2?1.e10f:1.e-30f);
  for(lane=0;lane<heads*32;++lane)layer_rope(x.data(),out.data(),heads,position);
  float theta=float(position),scale=std::pow(10000.f,-1.f/32.f);
  for(unsigned pair=0;pair<32;++pair){
   for(unsigned h=0;h<heads;++h){unsigned j=h*64+2*pair;double c=std::cos(double(theta)),si=std::sin(double(theta));double a=x[j],b=x[j+1];
    assert(std::isfinite(out[j])&&std::isfinite(out[j+1]));
    assert(std::abs(out[j]-(a*c-b*si))<=4e-6*(std::abs(a)+std::abs(b)));
    assert(std::abs(out[j+1]-(a*si+b*c))<=4e-6*(std::abs(a)+std::abs(b)));
    double before=a*a+b*b,after=double(out[j])*out[j]+double(out[j+1])*out[j+1];assert(std::abs(after-before)<=4e-6*before);
   }
   theta*=scale;
  }
  assert(out.back()==17);
 }
}
static void quantizer_extremes() {
 const float threshold=std::ldexp(127.f,-25);
 for(float maximum:{0.f,std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::min(),
                    std::nextafter(threshold,0.f),threshold,std::nextafter(threshold,INFINITY),
                    std::ldexp(127.f,-24),127.f*65504.f}){
  std::vector<float>x(2048);std::vector<unsigned char>q(64*36+1,0xa5);
  for(unsigned i=0;i<x.size();++i)x[i]=(i%3==0?0.f:(i%3==1?maximum:-maximum));
  for(lane=0;lane<64;++lane)layer_quantize_q8_cpu(x.data(),q.data(),2048);
  for(unsigned b=0;b<64;++b){_Float16 scale,sum;std::memcpy(&scale,q.data()+36*b,2);std::memcpy(&sum,q.data()+36*b+2,2);
   assert(float(scale)==float((_Float16)(maximum/127.f))&&float(sum)==0);
   if(float(scale)==0){for(unsigned j=0;j<36;++j)assert(q[36*b+j]==0);}
   else for(unsigned j=0;j<32;++j){unsigned index=b*32+j;assert((signed char)q[36*b+4+j]==(index%3==0?0:(index%3==1?127:-127)));}
  }
  assert(q.back()==0xa5);
 }
}
int main(){
 conversion_extremes();silu_extremes();softmax_extremes();dense_attention_extremes();rope_extremes();quantizer_extremes();

 float rounded_input[]={1.00048828125f,1.00146484375f,-1.00048828125f,0.000000059604644775390625f};_Float16 rounded[4];float copied[4];for(lane=0;lane<64;lane++){layer_to_f16(rounded_input,rounded,4);layer_copy(rounded_input,copied,4);}assert(float(rounded[0])==1.f&&float(rounded[1])==1.001953125f&&float(rounded[2])==-1.f&&float(rounded[3])==rounded_input[3]);for(unsigned i=0;i<4;i++)assert(copied[i]==rounded_input[i]);
 for(unsigned width:{2048u,5632u}) {
  std::vector<float> x(width,0);std::vector<unsigned char> packed(width/32*36,0xff),weights(width/32*18,0x99);float y[2]={0,-999};
  for(unsigned b=0;b<width/32;b++){x[b*32]=127;x[b*32+1]=.5f;x[b*32+2]=1.5f;x[b*32+3]=-1.5f;_Float16 d=1;std::memcpy(weights.data()+18*b,&d,2);}
  for(lane=0;lane<(width/32+63)/64*64;lane++)layer_quantize_q8_cpu(x.data(),packed.data(),width);
  for(unsigned b=0;b<width/32;b++){assert((signed char)packed[b*36+4]==127&&(signed char)packed[b*36+5]==0&&(signed char)packed[b*36+6]==2&&(signed char)packed[b*36+7]==-2);assert(packed[b*36+2]==0&&packed[b*36+3]==0);}
  for(lane=0;lane<64;lane++)layer_q4_cpu(weights.data(),packed.data(),y,width,1);
  assert(y[0]==127.f*(width/32)&&y[1]==-999);
 }
 std::vector<float>a(5632,2),b(5632,3),o(5633,-999);
 for(lane=0;lane<5696;lane++)layer_add(a.data(),b.data(),o.data(),5632);
 assert(o[0]==5&&o[5631]==5&&o[5632]==-999);
 for(lane=0;lane<5696;lane++)layer_mul(a.data(),b.data(),o.data(),5632);
 assert(o[5631]==6&&o[5632]==-999);
 for(lane=0;lane<5696;lane++)layer_silu(a.data(),o.data(),5632);
 assert(std::abs(o[0]-2/(1+std::exp(-2.f)))<1e-6&&o[5632]==-999);
 for(unsigned pos:{0,1,255,256,511}){
  std::vector<float>x(2048),r(2049,-999);for(unsigned i=0;i<x.size();i++)x[i]=float(int(i%17)-8)/9;
  for(lane=0;lane<1024;lane++)layer_rope(x.data(),r.data(),32,pos);
  for(unsigned j=0;j<2048;j+=2){float t=float(pos);for(unsigned step=0;step<(j/2)%32;step++)t*=std::pow(10000.f,-1.f/32.f);assert(std::abs(r[j]-(x[j]*std::cos(t)-x[j+1]*std::sin(t)))<2e-5);}
  assert(r[2048]==-999);
  std::vector<_Float16> k(131073,0),v(131073,0);std::vector<float>src(256);std::vector<long long>idx(256);for(unsigned d=0;d<256;d++){src[d]=float(d)/256;idx[d]=d*512+pos;}long long slot=pos;
  for(lane=0;lane<256;lane++){layer_set_rows(src.data(),&slot,k.data(),256,1);layer_set_rows(src.data(),idx.data(),v.data(),1,256);}
  for(unsigned d=0;d<256;d++){assert(float(k[pos*256+d])==src[d]);assert(float(v[d*512+pos])==src[d]);}assert(k.back()==0&&v.back()==0);
  unsigned n=pos<256?256:512;std::vector<_Float16> q(2048,1),ph(32*n);std::vector<float> s(32*n),p(32*n),mask(n,-INFINITY),out(2048);mask[pos]=0;
  for(lane=0;lane<32*n;lane++)layer_scores(k.data(),q.data(),s.data(),n);
  for(lane=0;lane<64;lane++)layer_softmax(s.data(),mask.data(),p.data(),n);
  for(unsigned h=0;h<32;h++)for(unsigned t=0;t<n;t++)assert(p[h*n+t]==float(t==pos));
  for(lane=0;lane<32*n;lane++)layer_to_f16(p.data(),ph.data(),32*n);
  for(lane=0;lane<2048;lane++)layer_values(v.data(),ph.data(),out.data(),n);
  for(unsigned i=0;i<2048;i++)assert(out[i]==src[(i/64/8)*64+i%64]);
 }
 std::cout<<"PASS exact kernel-source host execution: conversion/SILU/softmax/dense-attention/RoPE/quantizer extremes, tails5632, K/V layouts, N256/512 GQA\n";
 return 0;
}


