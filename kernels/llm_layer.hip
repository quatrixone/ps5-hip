// Bounded one-token layer operations. Host admission owns shapes, aliases and indices.
// Q and probabilities explicitly round to F16 before matmul; sums remain F32.
// This follows CPU dot operands, not upstream HIP default half PV accumulation.
// Fixed 64-thread X workgroups deliberately reuse the qualified Initial route.
using u32 = unsigned int;
using i64 = long long;
typedef unsigned char global_byte __attribute__((address_space(1)));
extern "C" __attribute__((device)) float __ocml_exp_f32(float);
extern "C" __attribute__((device)) float __ocml_sin_f32(float);
extern "C" __attribute__((device)) float __ocml_cos_f32(float);
extern "C" __attribute__((device)) float __ocml_pow_f32(float, float);
#define DEVICE __attribute__((device,always_inline))
#define KERNEL extern "C" __attribute__((global))
#define PS5HIP_CPU_MATH_DEVICE __attribute__((device,always_inline))
#include "llm_cpu_math.h"
#undef PS5HIP_CPU_MATH_DEVICE
static DEVICE u32 index() { return __builtin_amdgcn_workgroup_id_x()*64u+__builtin_amdgcn_workitem_id_x(); }
KERNEL void layer_copy(const float* x,float* out,u32 n) {u32 i=index();if(i<n)out[i]=x[i];}
KERNEL void layer_to_f16(const float* x,_Float16* out,u32 n) {u32 i=index();if(i<n)out[i]=(_Float16)x[i];}
KERNEL void layer_add(const float* a,const float* b,float* out,u32 n) {u32 i=index();if(i<n)out[i]=a[i]+b[i];}
KERNEL void layer_mul(const float* a,const float* b,float* out,u32 n) {u32 i=index();if(i<n)out[i]=a[i]*b[i];}
KERNEL void layer_silu(const float* x,float* out,u32 n) {u32 i=index();if(i<n)out[i]=x[i]/(1.0f+__ocml_exp_f32(-x[i]));}
KERNEL void layer_rope(const float* x,float* out,u32 heads,u32 position) {
 u32 i=index();if(i>=heads*32u)return;u32 d=i%32u;u32 j=i*2u;
 float theta=float(position);
 const float theta_scale=__ocml_pow_f32(10000.0f,-1.0f/32.0f);
 for(u32 step=0;step<d;step++)theta*=theta_scale;
 float c=__ocml_cos_f32(theta),s=__ocml_sin_f32(theta);
 out[j]=x[j]*c-x[j+1]*s;out[j+1]=x[j]*s+x[j+1]*c;
}
// Exact SET_ROWS input: K is one width256 row; V is256 width1 rows.
KERNEL void layer_set_rows(const float* x,const i64* indices,_Float16* out,u32 width,u32 rows) {
 u32 i=index();if(i<width*rows)out[static_cast<unsigned long long>(indices[i/width])*width+i%width]=(_Float16)x[i];
}
// Fixed reduction order bounds rounding drift before subsequent quantization.
static DEVICE float half_dot(const _Float16* x,const _Float16* y,u32 n) {
 float acc[4][8]{};
 for(u32 offset=0;offset<n;offset+=32) {
#pragma unroll
  for(u32 part=0;part<4;part++) {
#pragma unroll
   for(u32 lane=0;lane<8;lane++)acc[part][lane]=__builtin_fmaf(float(x[offset+part*8+lane]),float(y[offset+part*8+lane]),acc[part][lane]);
  }
 }
 float sum[8];
#pragma unroll
 for(u32 lane=0;lane<8;lane++)sum[lane]=(acc[0][lane]+acc[2][lane])+(acc[1][lane]+acc[3][lane]);
 return ((sum[0]+sum[4])+(sum[1]+sum[5]))+((sum[2]+sum[6])+(sum[3]+sum[7]));
}
// One lane computes the scale once; all input reads finish before in-place writes.
KERNEL void layer_rms_cpu(const float* x,float* out) {
 if(index()!=0)return;
 double sum=0;
 for(u32 i=0;i<2048;i++){float square=x[i]*x[i];sum+=double(square);}
 const float mean=float(sum/2048.0);
 const float scale=1.0f/__builtin_sqrtf(mean+1.e-5f);
 for(u32 i=0;i<2048;i++)out[i]=x[i]*scale;
}
KERNEL void layer_scores(const _Float16* k,const _Float16* q,float* out,u32 n) {
 u32 i=index();if(i>=32u*n)return;u32 h=i/n,t=i%n;
 out[i]=half_dot(k+t*256u+(h/8u)*64u,q+h*64u,64);
}
KERNEL void layer_softmax(const float* x,const float* mask,float* out,u32 n) {
 u32 h=index();if(h>=32)return;float m=-__builtin_inff();
 for(u32 t=0;t<n;t++){float v=x[h*n+t]*0.125f+mask[t];m=v>m?v:m;}
 float sum=0;for(u32 t=0;t<n;t++){float v=__ocml_exp_f32(x[h*n+t]*0.125f+mask[t]-m);out[h*n+t]=v;sum+=v;}
 for(u32 t=0;t<n;t++)out[h*n+t]/=sum;
}
KERNEL void layer_values(const _Float16* v,const _Float16* p,float* out,u32 n) {
 u32 i=index();if(i>=2048)return;u32 h=i/64u,d=i%64u;
 out[i]=half_dot(v+((h/8u)*64u+d)*512u,p+h*n,n);
}

// Layer-only CPU AVX-compatible Q8_0 operands in36-byte scratch blocks.
// The unused sum field is zero; quantized bytes remain at offset4.
KERNEL void layer_quantize_q8_cpu(const float* x,unsigned char* out,u32 width) {
 u32 b=index();if(b>=width/32u)return;float maximum=0;
 for(u32 j=0;j<32;j++){float v=__builtin_fabsf(x[b*32+j]);maximum=v>maximum?v:maximum;}
 float d=maximum/127.0f;
 const _Float16 stored_scale=(_Float16)d;
 // A zero stored scale contributes zero; avoid overflowing the reciprocal first.
 if(stored_scale==(_Float16)0){for(u32 j=0;j<36;j++)out[b*36+j]=0;return;}
 float inverse=maximum?127.0f/maximum:0.0f;
 *reinterpret_cast<_Float16*>(out+b*36)=stored_scale;
 *reinterpret_cast<_Float16*>(out+b*36+2)=(_Float16)0;
 for(u32 j=0;j<32;j++)out[b*36+4+j]=(unsigned char)(signed char)__builtin_rintf(x[b*32+j]*inverse);
}
KERNEL void layer_q4_cpu(const unsigned char* weights,const unsigned char* q8,float* out,u32 width,u32 rows) {
 u32 row=index();if(row>=rows)return;u32 blocks=width/32;float acc[8]{};
 for(u32 b=0;b<blocks;b++) {
  const unsigned char* w=weights+(row*blocks+b)*18;const unsigned char* q=q8+b*36;
  float scale=float(*reinterpret_cast<const _Float16*>(w))*float(*reinterpret_cast<const _Float16*>(q));
  unsigned char packed[16];
#pragma unroll
  for(u32 j=0;j<16;j++)packed[j]=*((const volatile global_byte*)(w+2+j));
#pragma unroll
  for(u32 lane=0;lane<8;lane++) {
   int dot=0;
#pragma unroll
   for(u32 k=0;k<4;k++){u32 j=lane*4+k;int nibble=int((packed[j%16]>>(j<16?0:4))&15)-8;dot+=nibble*int((signed char)q[4+j]);}
   acc[lane]=__builtin_fmaf(scale,float(dot),acc[lane]);
  }
 }
 out[row]=((acc[4]+acc[0])+(acc[6]+acc[2]))+((acc[5]+acc[1])+(acc[7]+acc[3]));
}

// Full-decoder arithmetic follows the pinned CPU oracle's explicit rounding.
// The selected-layer symbols above retain their qualified legacy behavior.
KERNEL void layer_silu_cpu(const float* x,float* out,u32 n) {
 u32 i=index();if(i<n)out[i]=ps5hip_cpu_math::silu_f32(x[i]);
}
KERNEL void layer_softmax_cpu(const float* x,const float* mask,float* out,u32 n) {
 u32 h=index();if(h>=32)return;
 for(u32 t=0;t<n;t++)out[h*n+t]=x[h*n+t]*0.125f+mask[t];
 ps5hip_cpu_math::softmax_f32(out+h*n,out+h*n,n);
}
KERNEL void layer_rope_cpu(const float* x,float* out,u32 heads,u32 position) {
 u32 i=index();if(i>=heads*32u)return;u32 j=i*2u;
 float theta=float(position);const float scale=__ocml_pow_f32(10000.0f,-1.0f/32.0f);
 for(u32 step=0;step<i%32u;step++)theta*=scale;
 const float c=__ocml_cos_f32(theta),s=__ocml_sin_f32(theta),a=x[j],b=x[j+1];
 out[j]=__builtin_fmaf(a,c,-b*s);out[j+1]=__builtin_fmaf(a,s,b*c);
}

#ifdef PS5HIP_ENABLE_DENSE_F16
// Host primitive opt-in only; not admitted by the native module builder.
// Only the seven learned-projection shapes of the fixed TinyLlama layer.
// Caller converts the F32 activation with layer_to_f16 before this operation.
KERNEL void layer_dense_f16(const _Float16* weights,const _Float16* x,float* out,u32 width,u32 rows) {
 const bool valid=(width==2048&&(rows==256||rows==2048||rows==5632))||(width==5632&&rows==2048);
 const u32 row=index();
 if(!valid||row>=rows||!weights||!x||!out)return;
 out[row]=half_dot(weights+static_cast<unsigned long long>(row)*width,x,width);
}
#endif
