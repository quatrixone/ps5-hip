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
KERNEL void layer_scores(const _Float16* k,const _Float16* q,float* out,u32 n) {
 u32 i=index();if(i>=32u*n)return;u32 h=i/n,t=i%n;float sum=0;
 for(u32 d=0;d<64;d++)sum+=float(k[t*256u+(h/8u)*64u+d])*q[h*64u+d];out[i]=sum;
}
KERNEL void layer_softmax(const float* x,const float* mask,float* out,u32 n) {
 u32 h=index();if(h>=32)return;float m=-__builtin_inff();
 for(u32 t=0;t<n;t++){float v=x[h*n+t]*0.125f+mask[t];m=v>m?v:m;}
 float sum=0;for(u32 t=0;t<n;t++){float v=__ocml_exp_f32(x[h*n+t]*0.125f+mask[t]-m);out[h*n+t]=v;sum+=v;}
 for(u32 t=0;t<n;t++)out[h*n+t]/=sum;
}
KERNEL void layer_values(const _Float16* v,const _Float16* p,float* out,u32 n) {
 u32 i=index();if(i>=2048)return;u32 h=i/64u,d=i%64u;float sum=0;
 for(u32 t=0;t<n;t++)sum+=float(v[((h/8u)*64u+d)*512u+t])*p[h*n+t];out[i]=sum;
}

// Layer-only CPU AVX-compatible Q8_0 operands in36-byte scratch blocks.
// The unused sum field is zero; quantized bytes remain at offset4.
KERNEL void layer_quantize_q8_cpu(const float* x,unsigned char* out,u32 width) {
 u32 b=index();if(b>=width/32u)return;float maximum=0;
 for(u32 j=0;j<32;j++){float v=__builtin_fabsf(x[b*32+j]);maximum=v>maximum?v:maximum;}
 float d=maximum/127.0f,inverse=maximum?127.0f/maximum:0.0f;
 *reinterpret_cast<_Float16*>(out+b*36)=(_Float16)d;
 *reinterpret_cast<_Float16*>(out+b*36+2)=(_Float16)0;
 for(u32 j=0;j<32;j++)out[b*36+4+j]=(unsigned char)(signed char)__builtin_rintf(x[b*32+j]*inverse);
}
KERNEL void layer_q4_cpu(const unsigned char* weights,const unsigned char* q8,float* out,u32 width,u32 rows) {
 u32 row=index();if(row>=rows)return;u32 blocks=width/32;float sum=0;
 for(u32 b=0;b<blocks;b++) {
  const unsigned char* w=weights+(row*blocks+b)*18;const unsigned char* q=q8+b*36;
  float d4=float(*reinterpret_cast<const _Float16*>(w)),d8=float(*reinterpret_cast<const _Float16*>(q));int dot=0;
  for(u32 j=0;j<16;j++){unsigned v=*((const volatile global_byte*)(w+2+j));dot+=(int(v&15)-8)*int((signed char)q[4+j])+(int(v>>4)-8)*int((signed char)q[20+j]);}
  sum+=(d4*d8)*float(dot);
 }
 out[row]=sum;
}



