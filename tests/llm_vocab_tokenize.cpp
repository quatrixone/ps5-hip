#include "llama.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
int main(int argc,char**argv){
 if(argc!=2)return 2;llama_backend_init();auto p=llama_model_default_params();p.vocab_only=true;p.n_gpu_layers=0;p.use_extra_bufts=false;
 auto*m=llama_model_load_from_file(argv[1],p);if(!m)return 3;auto*v=llama_model_get_vocab(m);if(llama_vocab_n_tokens(v)!=32000)return 4;
 const char*texts[]={"","Hello, world!","The capital of France is","  spaces\nnew line\t tab",u8"café Ελληνικά 日本語 😀","<s>hello</s>","<unk><s></s>","[INST] Say hello [/INST]"};
 for(unsigned c=0;c<sizeof(texts)/sizeof(*texts);++c){ for(int a=0;a<2;++a){ for(int s=0;s<2;++s){
  std::vector<llama_token> t(256);int n=llama_tokenize(v,texts[c],int(strlen(texts[c])),t.data(),int(t.size()),bool(a),bool(s));if(n<0)return 5;
  printf("{\"case\":%u,\"add_special\":%s,\"parse_special\":%s,\"tokens\":[",c,a?"true":"false",s?"true":"false");for(int i=0;i<n;++i)printf("%s%d",i?",":"",t[i]);puts("]}");
 } } }
 llama_model_free(m);llama_backend_free();return 0;
}