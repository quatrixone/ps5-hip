#include "tinyllama_source_metadata.h"
#include "llama.h"
static direct_tiny::Metadata metadata(nullptr,gguf_free);
static llama_model* load_original(const char* path,llama_model_params params){metadata=direct_tiny::construct(path);return llama_model_init_from_user(metadata.get(),nullptr,nullptr,params);}
#define llama_model_load_from_file load_original
#include "llm_vocab_tokenize.cpp"
