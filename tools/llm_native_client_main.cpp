#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
int main(int argc,char**argv){if(argc!=2)return 2;auto p=llama_model_default_params();p.vocab_only=true;auto*m=llama_model_load_from_file(argv[1],p);if(!m)return 3;const auto*v=llama_model_get_vocab(m);llama_token ids[32];int n=llama_tokenize(v,"Hello",5,ids,32,true,true);std::printf("tokens=%d backends=%zu\n",n,ggml_backend_reg_count());llama_model_free(m);return n>0?0:4;}
