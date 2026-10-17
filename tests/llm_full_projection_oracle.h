#pragma once
// Independent same-input diagnostic only; never substitutes activations or relaxes the CPU gate.
#include <vector>
extern "C" void quantize_row_q8_0(const float*,void*,int64_t);
inline unsigned q4_calls=0,q8_calls=0;inline size_t q4_rows=0;inline double max_bound_fraction=0;
inline void same_input(unsigned id,void** a){
 if(id==16){const float* input=layer_numeric::arg<const float*>(a,0);const unsigned char* actual=layer_numeric::arg<const unsigned char*>(a,1);unsigned width=layer_numeric::arg<unsigned>(a,2);std::vector<unsigned char> q(width/32*34);quantize_row_q8_0(input,q.data(),width);for(unsigned b=0;b<width/32;++b){assert(!std::memcmp(q.data()+b*34,actual+b*36,2));if(layer_numeric::half(actual+b*36)!=0)assert(!std::memcmp(q.data()+b*34+2,actual+b*36+4,32));}++q8_calls;}
 if(id==17){const auto* weight=layer_numeric::arg<const unsigned char*>(a,0);const auto* q=layer_numeric::arg<const unsigned char*>(a,1);const float* actual=layer_numeric::arg<const float*>(a,2);unsigned width=layer_numeric::arg<unsigned>(a,3),rows=layer_numeric::arg<unsigned>(a,4),blocks=width/32;double nu=(2*blocks+3)*std::ldexp(1.,-24),gamma=nu/(1-nu);
 for(unsigned row=0;row<rows;++row){double sum=0,magnitude=0;for(unsigned b=0;b<blocks;++b){const auto* w=weight+(row*blocks+b)*18;const auto* qb=q+b*36;int dot=0;for(unsigned j=0;j<16;++j)dot+=(int(w[2+j]&15)-8)*int((signed char)qb[4+j])+(int(w[2+j]>>4)-8)*int((signed char)qb[20+j]);double term=double(layer_numeric::half(w))*layer_numeric::half(qb)*dot;sum+=term;magnitude+=std::abs(term);}double error=std::abs(double(actual[row])-sum),bound=gamma*magnitude;assert(std::isfinite(actual[row])&&error<=bound);if(bound)max_bound_fraction=std::max(max_bound_fraction,error/bound);}
 ++q4_calls;q4_rows+=rows;}
}
