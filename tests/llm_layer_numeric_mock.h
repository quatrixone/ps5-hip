#pragma once
// Numerical test double only: never linked into native runtime or release client.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>
namespace layer_numeric {
inline unsigned lane=0;
#define __attribute__(x)
#define __builtin_amdgcn_workgroup_id_x() (lane/64)
#define __builtin_amdgcn_workitem_id_x() (lane%64)
extern "C" inline float __ocml_exp_f32(float x){return std::exp(x);}
extern "C" inline float __ocml_sin_f32(float x){return std::sin(x);}
extern "C" inline float __ocml_cos_f32(float x){return std::cos(x);}
extern "C" inline float __ocml_pow_f32(float x,float y){return std::pow(x,y);}
#include "../kernels/llm_layer.hip"
#undef __attribute__
#undef __builtin_amdgcn_workgroup_id_x
#undef __builtin_amdgcn_workitem_id_x
#undef DEVICE
#undef KERNEL
inline float half(const unsigned char* p){_Float16 h;std::memcpy(&h,p,2);return float(h);}
inline void store_half(unsigned char* p,float value){_Float16 h=(_Float16)value;std::memcpy(p,&h,2);}
template<class T> T arg(void** a,unsigned i){T v;std::memcpy(&v,a[i],sizeof(v));return v;}
inline float butterfly(float* x){for(unsigned mask:{16u,8u,4u,2u,1u}){float y[32];for(unsigned i=0;i<32;i++)y[i]=x[i]+x[i^mask];std::memcpy(x,y,sizeof(y));}return x[0];}
inline void quantize(const float* x,unsigned char* q,unsigned width){
 for(unsigned b=0;b<width/32;b++){float maximum=0,sums[32];for(unsigned j=0;j<32;j++){sums[j]=x[b*32+j];maximum=std::max(maximum,std::abs(sums[j]));}
  float d=maximum/127.f;store_half(q+36*b,d);store_half(q+36*b+2,butterfly(sums));
  for(unsigned j=0;j<32;j++){int v=d?int(std::round(x[b*32+j]/d)):0;q[36*b+4+j]=static_cast<unsigned char>(static_cast<int8_t>(v));}
 }
}
inline float q4_row(const unsigned char* w,const unsigned char* q,unsigned blocks){
 float waves[4][32]{};
 for(unsigned wave=0;wave<4;wave++)for(unsigned laneid=0;laneid<32;laneid++)for(unsigned b=(32*wave+laneid)/2;b<blocks;b+=64){
  const auto* wb=w+18*b;const auto* qb=q+36*b;int dot=0;
  for(unsigned j=8*(laneid%2);j<8*(laneid%2)+8;j++)dot+=(wb[2+j]&15)*int(static_cast<int8_t>(qb[4+j]))+(wb[2+j]>>4)*int(static_cast<int8_t>(qb[4+j+16]));
  waves[wave][laneid]=std::fma(std::fma(float(dot),half(qb),-4.f*half(qb+2)),half(wb),waves[wave][laneid]);
 }
 float sums[32];for(unsigned i=0;i<32;i++)sums[i]=((waves[0][i]+waves[1][i])+waves[2][i])+waves[3][i];float result=butterfly(sums);
 // Independent full-block FP64 oracle and the existing a-priori gamma bound.
 double reference=0,magnitude=0;
 for(unsigned b=0;b<blocks;b++){
  const auto* wb=w+18*b;const auto* qb=q+36*b;double d4=half(wb),d8=half(qb),s8=half(qb+2);int full=0;
  for(unsigned first:{0u,8u}){int dot=0;for(unsigned j=first;j<first+8;j++)dot+=(wb[2+j]&15)*int(static_cast<int8_t>(qb[4+j]))+(wb[2+j]>>4)*int(static_cast<int8_t>(qb[4+j+16]));full+=dot;magnitude+=std::abs(d4)*(std::abs(d8*dot)+4*std::abs(s8));}
  reference+=d4*(d8*full-8*s8);
 }
 const double nu=double((blocks+63)/64+12)*std::ldexp(1.0,-24),bound=nu/(1-nu)*magnitude;
 if(!std::isfinite(result)||std::abs(double(result)-reference)>bound)throw std::runtime_error("Q4 violates predeclared gamma envelope");
 return result;
}
inline void execute(unsigned id,unsigned work_items,void** a){
 if(id==1){const auto* w=arg<const unsigned char*>(a,0);int token=*arg<const int*>(a,1);float* out=arg<float*>(a,2);w+=1152*token;for(unsigned b=0;b<64;b++)for(unsigned j=0;j<32;j++)out[b*32+j]=half(w+18*b)*(int((w[18*b+2+j%16]>>(j<16?0:4))&15)-8);return;}
 if(id==2||id==3){const float* x=arg<const float*>(a,0);float* out=arg<float*>(a,1);unsigned n=arg<int>(a,2);double sum=0;for(unsigned i=0;i<n;i++)sum+=double(x[i])*x[i];float scale=float(1/std::sqrt(sum/n+arg<float>(a,8)));const float* w=arg<const float*>(a,9);for(unsigned i=0;i<n;i++){float v=x[i]*scale;out[i]=id==3?v*w[i]:v;}return;}
 if(id==4){quantize(arg<const float*>(a,0),arg<unsigned char*>(a,1),unsigned(arg<int64_t>(a,2)));return;}
 if(id==5){auto w=arg<const unsigned char*>(a,0);auto q=arg<const unsigned char*>(a,1);auto out=arg<float*>(a,4);unsigned width=arg<unsigned>(a,5),rows=arg<unsigned>(a,9);for(unsigned r=0;r<rows;r++)out[r]=q4_row(w+r*(width/32)*18,q,width/32);return;}
 for(lane=0;lane<work_items;lane++)switch(id){
 case 6:layer_add(arg<const float*>(a,0),arg<const float*>(a,1),arg<float*>(a,2),arg<unsigned>(a,3));break;
 case 7:layer_mul(arg<const float*>(a,0),arg<const float*>(a,1),arg<float*>(a,2),arg<unsigned>(a,3));break;
 case 8:layer_silu(arg<const float*>(a,0),arg<float*>(a,1),arg<unsigned>(a,2));break;
 case 9:layer_rope(arg<const float*>(a,0),arg<float*>(a,1),arg<unsigned>(a,2),arg<unsigned>(a,3));break;
 case 10:layer_set_rows(arg<const float*>(a,0),arg<const long long*>(a,1),arg<_Float16*>(a,2),arg<unsigned>(a,3),arg<unsigned>(a,4));break;
 case 11:layer_scores(arg<const _Float16*>(a,0),arg<const _Float16*>(a,1),arg<float*>(a,2),arg<unsigned>(a,3));break;
 case 12:layer_softmax(arg<const float*>(a,0),arg<const float*>(a,1),arg<float*>(a,2),arg<unsigned>(a,3));break;
 case 13:layer_values(arg<const _Float16*>(a,0),arg<const _Float16*>(a,1),arg<float*>(a,2),arg<unsigned>(a,3));break;
 case 14:layer_copy(arg<const float*>(a,0),arg<float*>(a,1),arg<unsigned>(a,2));break;
 case 15:layer_to_f16(arg<const float*>(a,0),arg<_Float16*>(a,1),arg<unsigned>(a,2));break;
 case 16:layer_quantize_q8_cpu(arg<const float*>(a,0),arg<unsigned char*>(a,1),arg<unsigned>(a,2));break;
 case 17:layer_q4_cpu(arg<const unsigned char*>(a,0),arg<const unsigned char*>(a,1),arg<float*>(a,2),arg<unsigned>(a,3),arg<unsigned>(a,4));break;
 default:throw std::runtime_error("unknown numerical HIP symbol");
 }
}
}
