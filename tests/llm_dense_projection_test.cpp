#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
static unsigned lane;
#define __attribute__(x)
#define __builtin_amdgcn_workgroup_id_x() (lane/64)
#define __builtin_amdgcn_workitem_id_x() (lane%64)
extern "C" float __ocml_exp_f32(float x){return std::exp(x);}
extern "C" float __ocml_sin_f32(float x){return std::sin(x);}
extern "C" float __ocml_cos_f32(float x){return std::cos(x);}
extern "C" float __ocml_pow_f32(float x,float y){return std::pow(x,y);}
#define PS5HIP_ENABLE_DENSE_F16 1
#include "../kernels/llm_layer.hip"
extern "C" void ggml_vec_dot_f16(int,float*,size_t,uint16_t*,size_t,uint16_t*,size_t,int);
extern "C" void ggml_fp32_to_fp16_row(const float*,uint16_t*,int64_t);
static void equal(float actual,float expected,unsigned width,unsigned row){
 if(std::memcmp(&actual,&expected,4)){std::fprintf(stderr,"dense mismatch width=%u row=%u actual=%a expected=%a\n",width,row,double(actual),double(expected));std::abort();}
}
static float reference(const _Float16* weights,const _Float16* input,unsigned width){
 std::vector<uint16_t> w(width),x(width);std::memcpy(w.data(),weights,width*2);std::memcpy(x.data(),input,width*2);
 float result;ggml_vec_dot_f16(int(width),&result,0,w.data(),0,x.data(),0,1);return result;
}
static void shape(unsigned width,unsigned rows){
 std::vector<float> input(width);std::vector<_Float16> x(width),w(size_t(width)*rows);std::vector<uint16_t> expected_x(width);std::vector<float> out(rows+2,12345);
 for(unsigned i=0;i<width;i++)input[i]=float(int((i*19)%137)-68)*.0137f;
 // Include sign, half subnormal/tie and finite boundary conversion cases.
 const float special[]={0.f,-0.f,0x1p-24f,-0x1p-24f,0x1p-25f,0x1.8p-24f,65504.f,-65504.f};
 std::copy(std::begin(special),std::end(special),input.begin());
 ggml_fp32_to_fp16_row(input.data(),expected_x.data(),width);
 for(lane=0;lane<width;lane++)layer_to_f16(input.data(),x.data(),width);
 assert(std::memcmp(x.data(),expected_x.data(),width*2)==0);
 for(size_t i=0;i<w.size();i++)w[i]=_Float16(float(int((i*29+i/width*7)%251)-125)*.0023f);
 for(lane=0;lane<rows+64;lane++)layer_dense_f16(w.data(),x.data(),out.data()+1,width,rows);
 assert(out.front()==12345&&out.back()==12345);
 for(unsigned row=0;row<rows;row++){
  const float expected=reference(w.data()+size_t(row)*width,x.data(),width);
  equal(out[row+1],expected,width,row);assert(std::isfinite(expected));
 }
 // Exact cancellation, maximum half products, zero and tiny inputs at both widths.
 for(unsigned mode=0;mode<4;mode++){
  std::fill(x.begin(),x.end(),_Float16(0));std::fill(w.begin(),w.begin()+width,_Float16(0));
  if(mode==0){w[0]=65504;w[32]=-65504;x[0]=x[32]=65504;w[1]=x[1]=1;}
  if(mode==1){std::fill(x.begin(),x.end(),_Float16(0x1p-24f));std::fill(w.begin(),w.begin()+width,_Float16(0x1p-24f));}
  if(mode==2){std::fill(x.begin(),x.end(),_Float16(65504));std::fill(w.begin(),w.begin()+width,_Float16(65504));}
  const float expected=reference(w.data(),x.data(),width);
  lane=0;layer_dense_f16(w.data(),x.data(),out.data()+1,width,rows);equal(out[1],expected,width,0);assert(std::isfinite(expected));
  if(mode==0)assert(expected==1.f);
 }
}
int main(){
 for(auto geometry:{std::pair<unsigned,unsigned>{2048,256},{2048,2048},{2048,5632},{5632,2048}})shape(geometry.first,geometry.second);
 float sentinel=17;_Float16 single=1;
 for(auto geometry:{std::pair<unsigned,unsigned>{0,0},{64,256},{2047,256},{2048,0},{2048,257},{2048,5633},{5632,256},{5633,2048},{UINT32_MAX,UINT32_MAX}}){lane=0;layer_dense_f16(&single,&single,&sentinel,geometry.first,geometry.second);assert(sentinel==17);}
 lane=256;layer_dense_f16(&single,&single,&sentinel,2048,256);assert(sentinel==17);
 lane=0;layer_dense_f16(nullptr,nullptr,&sentinel,2048,256);assert(sentinel==17);
 std::puts("DENSE_F16_PRIMITIVE_PASS shapes=4 widths=2048,5632 all_rows=9984 actual_cpu_dot=true GPU_execution=false");
}
