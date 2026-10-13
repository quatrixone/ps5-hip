#include "ps5hip/host_registration.h"
#include <array>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <new>
#include <cstdlib>
static long fail_after=-1;
void *operator new(size_t n){if(fail_after==0){fail_after=-1;throw std::bad_alloc();}if(fail_after>0)--fail_after;auto p=std::malloc(n?n:1);if(!p)throw std::bad_alloc();return p;}
void operator delete(void *p) noexcept {std::free(p);}
void operator delete(void *p,size_t) noexcept {std::free(p);}
void check(bool b){if(!b)throw std::runtime_error("multi registration assertion");}
template<class F>void reject(F f){try{f();}catch(const ps5hip::ObjectError&){return;}throw std::runtime_error("invalid multi registration accepted");}
void put(uint8_t *p,uint64_t n){for(unsigned i=0;i<8;++i)p[i]=uint8_t(n>>(i*8));}
int main(int argc,char **argv){try{
 check(argc==2);std::ifstream f(argv[1],std::ios::binary);std::vector<uint8_t> one{std::istreambuf_iterator<char>(f),{}};check(!one.empty());
 std::vector<uint8_t> bundles=one;bundles.insert(bundles.end(),one.begin(),one.end());
 std::array<uint8_t,48> wrappers{};std::array<uint8_t,64> table{};
 for(unsigned i=0;i<2;++i){auto *w=wrappers.data()+24*i,*b=bundles.data()+one.size()*i;put(w,0x0000000148495046);put(w+8,reinterpret_cast<uintptr_t>(b));
 put(table.data()+32*i,24*i);put(table.data()+32*i+8,24*(i+1));put(table.data()+32*i+16,one.size()*i);put(table.data()+32*i+24,one.size()*(i+1));}
 ps5hip::ReadableSpan ws{wrappers.data(),wrappers.size()},bs{bundles.data(),bundles.size()},ts{table.data(),table.size()};
 auto select=[&](unsigned i){return ps5hip::select_host_module(wrappers.data()+i*24,ts,ws,bs);};
 auto first=select(0),second=select(1);check(first.bundle.size==one.size()&&second.wrapper.data==wrappers.data()+24);
 auto original=table;
 for(unsigned field=0;field<8;++field){table=original;put(table.data()+field*8,UINT64_MAX);reject([&]{select(0);});}
 table=original;std::memcpy(table.data()+32,table.data(),32);reject([&]{select(0);});table=original;
 reject([&]{ps5hip::select_host_module(wrappers.data()+1,ts,ws,bs);});
 reject([&]{ps5hip::select_host_module(wrappers.data(),{table.data(),63},ws,bs);});
 for(bool reverse:{false,true}){
  ps5hip::HostRegistration r(2);void *h[2];int tokens[2]{};
  for(unsigned j=0;j<2;++j){unsigned i=reverse?1-j:j;auto p=select(i);h[i]=r.register_module(p.wrapper.data,p.wrapper,p.bundle,"gfx1013");}
  check(!r.modules_ready());const auto name=r.module(h[0]).object.kernels.at(0).name;
  r.register_function(h[0],&tokens[0],name);check(!r.modules_ready());
  reject([&]{r.register_function(h[1],&tokens[0],name);});check(r.active_functions()==1);
  r.register_function(h[1],&tokens[1],name);check(r.modules_ready());
  check(r.lookup_owned(&tokens[0]).owner==h[0]&&r.lookup_owned(&tokens[1]).owner==h[1]);
  reject([&]{r.register_module(first.wrapper.data,first.wrapper,first.bundle,"gfx1013");});
  for(unsigned j=0;j<2;++j){unsigned i=reverse?j:1-j;r.unregister_module(h[i]);reject([&]{r.lookup(&tokens[i]);});reject([&]{r.unregister_module(h[i]);});check(r.active_modules()==1-j);}
 }
 ps5hip::HostRegistration life;void *retired=nullptr;
 for(unsigned i=0;i<32;++i){auto h=life.register_module(first.wrapper.data,first.wrapper,first.bundle,"gfx1013");check(h!=retired);if(retired)reject([&]{life.module(retired);});life.unregister_module(h);retired=h;}
 reject([&]{life.register_module(first.wrapper.data,first.wrapper,first.bundle,"gfx1013");});
 unsigned failures=0;bool reached=false;
 for(long point=0;point<10000&&!reached;++point){
  ps5hip::HostRegistration r(2);auto a=r.register_module(first.wrapper.data,first.wrapper,first.bundle,"gfx1013");
  int token=0;const auto name=r.module(a).object.kernels.at(0).name;r.register_function(a,&token,name);
  fail_after=point;
  try{r.register_module(second.wrapper.data,second.wrapper,second.bundle,"gfx1013");reached=true;}
  catch(const std::bad_alloc&){++failures;}
  fail_after=-1;check(r.active_modules()==(reached?2u:1u)&&r.active_functions()==1&&r.lookup_owned(&token).owner==a);
 }
 check(reached&&failures>2);
 {ps5hip::HostRegistration r(2);auto a=r.register_module(first.wrapper.data,first.wrapper,first.bundle,"gfx1013");int token=0;const auto name=r.module(a).object.kernels.at(0).name;
 fail_after=0;bool failed=false;try{r.register_function(a,&token,name);}catch(const std::bad_alloc&){failed=true;}fail_after=-1;check(failed&&r.active_functions()==0);r.register_function(a,&token,name);check(r.lookup_owned(&token).owner==a);}
 std::printf("ALLOCATION_FAILURES transactional=%u function_rollback=true\n",failures);
 std::puts("PASS exact table bounds, duplicate/overlap rejection, scoped names/global tokens, owner selection, reverse retirement and lifetime cap");
 }catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
