/* Independent directed MPFR boundary certificate; no target execution. */
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void require(int yes,const char*why){if(!yes){fprintf(stderr,"%s\n",why);exit(1);}}
static float value(uint32_t word){float f;memcpy(&f,&word,4);return f;}
static uint32_t bits(float f){uint32_t word;memcpy(&word,&f,4);return word;}
int main(void){
 require(sizeof(float)==4 && strcmp(mpfr_get_version(),"4.2.2")==0,"pinned MPFR/binary32 required");
 mpfr_t half,lo,hi,x,y,z;mpfr_inits2(512,half,lo,hi,x,y,z,(mpfr_ptr)0);
 mpfr_set_ui_2exp(half,1,-150,MPFR_RNDN);mpfr_log(lo,half,MPFR_RNDD);mpfr_log(hi,half,MPFR_RNDU);
 mpfr_set_flt(x,value(0xc2cff1b5),MPFR_RNDN);require(mpfr_cmp(x,lo)<0,"lower bracket failed");
 mpfr_set_flt(x,value(0xc2cff1b4),MPFR_RNDN);require(mpfr_cmp(x,hi)>0,"upper bracket failed");
 mpfr_printf("{\"kind\":\"half_min_boundary\",\"lower\":\"%Ra\",\"upper\":\"%Ra\",\"below_bits\":\"c2cff1b5\",\"first_above_bits\":\"c2cff1b4\"}\n",lo,hi);
 const uint32_t controls[]={0xc2ce8ecf,0xc2ce8ed0,0xc2ce8ed1,0xc2aeac4f,0xc2aeac50,0xc2aeac51,0,0x80000000,0x3f800000,0xbf800000,0x42b17217,0x42b17218,0x42b17219};
 for(unsigned i=0;i<sizeof(controls)/sizeof(*controls)+17;++i){
  uint32_t word=i<17?0xc2cff1ac+i:controls[i-17];mpfr_set_flt(x,value(word),MPFR_RNDN);
  mpfr_exp(y,x,MPFR_RNDD);mpfr_exp(z,x,MPFR_RNDU);
  uint32_t a=bits(mpfr_get_flt(y,MPFR_RNDN)),b=bits(mpfr_get_flt(z,MPFR_RNDN));require(a==b,"directed rounding unresolved");
  if(word==0xc2cff1b4)require(a==1,"first-above must round to minimum subnormal");
  if(word==0xc2cff1b5)require(a==0,"first-below must round to positive zero");
  printf("{\"kind\":\"exp_rounding\",\"input_bits\":\"%08x\",\"reference_bits\":\"%08x\"}\n",word,a);
 }
 /* Ordered underflow predicate does not select zero for NaN. Existing
    overflow and numerical path must preserve special-value behavior. */
 puts("{\"kind\":\"analytic_controls\",\"exp_negative_inf\":\"00000000\",\"exp_positive_inf\":\"7f800000\",\"exp_signed_zeros\":\"3f800000\",\"exp_nan\":\"nan\",\"hardware_execution\":false}");
 mpfr_clears(half,lo,hi,x,y,z,(mpfr_ptr)0);return 0;
}
