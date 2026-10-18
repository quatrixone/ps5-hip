#pragma once
#include "extended_generation.h"
#include <cstring>
#include <cstdio>
namespace ps5hip_generation {
// Returns exact float size, or the bounded ceiling for index/case metadata.
inline size_t extended_output_limit(const char* filename){
 if(!filename)return 0;
 if(!std::strcmp(filename,"extended.json"))return 64*1024;
 const ExtendedCase* selected=nullptr;const char* base=nullptr;
 for(const auto& item:extended_cases){const size_t n=std::strlen(item.id);if(!std::strncmp(filename,item.id,n)&&filename[n]=='/'){selected=&item;base=filename+n+1;}}
 if(!selected||!base||std::strchr(base,'/'))return 0;
 if(!std::strcmp(base,"generation.json")||!std::strcmp(base,"steps.jsonl"))return 1024*1024;
 if(std::strlen(base)<12||std::strncmp(base,"run-",4)||(base[4]!='0'&&base[4]!='1')||std::strncmp(base+5,"-pos-",5))return 0;
 const unsigned run=unsigned(base[4]-'0');const char* digit=base+10;unsigned pos=0,count=0;
 while(*digit>='0'&&*digit<='9'&&count<3){pos=pos*10+unsigned(*digit++-'0');++count;}
 if(!count||*digit!='-'||pos>=selected->prompt_count+selected->generated-1)return 0;
 for(unsigned stage=0;stage<24;++stage){
  if(stage<23&&!extended_checkpoint(*selected,pos))continue;
  char label[24],expected[96];if(stage<22)std::snprintf(label,sizeof(label),"layer-%u",stage);else std::snprintf(label,sizeof(label),"%s",stage==22?"norm":"logits");
  std::snprintf(expected,sizeof(expected),"run-%u-pos-%u-%s.f32le",run,pos,label);
  if(!std::strcmp(base,expected))return stage==23?128000:8192;
 } return 0;
}
inline bool extended_output_size(const char* name,size_t bytes){
 const auto limit=extended_output_limit(name);return limit&&((limit==64*1024||limit==1024*1024)?(bytes>0&&bytes<=limit):bytes==limit);
}
}
