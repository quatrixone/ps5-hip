#pragma once
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <algorithm>
inline double dense_check_values(const void* actual,const void* reference,size_t count,bool half=false,bool exact=false,bool probabilities=false){
 if(exact&&std::memcmp(actual,reference,count*(half?2:4)))throw std::runtime_error("exact embedding bits");
 double error_energy=0,reference_energy=0,maximum=0;const double relative=half?2e-3:2e-4;
 for(size_t i=0;i<count;i++){
  const double x=half?float(static_cast<const _Float16*>(actual)[i]):static_cast<const float*>(actual)[i];
  const double y=half?float(static_cast<const _Float16*>(reference)[i]):static_cast<const float*>(reference)[i];
  if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("nonfinite boundary");
  const double e=std::abs(x-y);if(e>2e-5+relative*std::abs(y))throw std::runtime_error("element budget");
  if((half||probabilities)&&y==0&&x!=0)throw std::runtime_error("exact cache/masked probability zero");
  maximum=std::max(maximum,e);error_energy+=e*e;reference_energy+=y*y;
 }
 const double rms=reference_energy?std::sqrt(error_energy/reference_energy):(error_energy?INFINITY:0);
 if(rms>relative)throw std::runtime_error("relative RMS budget");return maximum;
}
using DenseSnapshot=std::vector<std::vector<unsigned char>>;
inline void dense_check_reset(const DenseSnapshot& a,const DenseSnapshot& b){if(a.size()!=21||b.size()!=21||a!=b)throw std::runtime_error("reset all21 boundary bits");}
