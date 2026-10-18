#include "llm_dense_layer_checks.h"
#include <cassert>
#include <limits>
int main(){
 auto reject=[](auto test){bool failed=false;try{test();}catch(const std::runtime_error&){failed=true;}assert(failed);};
 float x=2e-30f,y=1e-30f;reject([&]{dense_check_values(&x,&y,1);});y=0;reject([&]{dense_check_values(&x,&y,1);});x=0;assert(dense_check_values(&x,&y,1)==0);
 x=std::numeric_limits<float>::quiet_NaN();reject([&]{dense_check_values(&x,&y,1);});x=-0.f;reject([&]{dense_check_values(&x,&y,1,false,true);});
 float a[]={1.f,1e-7f},b[]={1.f,0.f};reject([&]{dense_check_values(a,b,2,false,false,true);});
 _Float16 ah[]={(_Float16)1,(_Float16)0x1p-24f},bh[]={(_Float16)1,(_Float16)0};reject([&]{dense_check_values(ah,bh,2,true);});
 DenseSnapshot saved(21,std::vector<unsigned char>(8,0));dense_check_reset(saved,saved);for(unsigned i=0;i<21;i++){auto changed=saved;changed[i][0]=1;reject([&]{dense_check_reset(saved,changed);});}
}
