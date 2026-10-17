#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <cstring>
static unsigned lane=0;
#define __attribute__(x)
#define __builtin_amdgcn_workgroup_id_x() (lane/64)
#define __builtin_amdgcn_workitem_id_x() (lane%64)
extern "C" float __ocml_exp_f32(float x){return std::exp(x);}
extern "C" float __ocml_sin_f32(float x){return std::sin(x);}
extern "C" float __ocml_cos_f32(float x){return std::cos(x);}
extern "C" float __ocml_pow_f32(float x,float y){return std::pow(x,y);}
#include "../kernels/llm_layer.hip"
int main(){
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
 std::cout<<"PASS exact kernel-source host execution: tails5632, RoPE5positions, K/V layouts, N256/512 masked GQA\n";
}


