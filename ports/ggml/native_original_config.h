#pragma once
#include "tinyllama_source_metadata.h"
#include <sys/stat.h>
namespace ps5hip {
// Inventory hashes are authenticated by the producer/controller; this is a
// native path/type/size and semantic configuration check, not SHA verification.
inline std::string original_model_directory(const std::string& config_path){
 using direct_tiny::need;
 const auto config=direct_tiny::json(direct_tiny::read(config_path,65536));
 direct_tiny::keys(config,{"schema","format","directory","files"});
 need(config.at("schema").is_number_integer()&&config.at("schema")==1&&config.at("format")=="original_safetensors","original format config");
 const std::string directory=config.at("directory");
 need(directory.size()>17&&directory.size()<=240&&directory.compare(0,17,"/data/hip/models/")==0,"original directory prefix");
 for(unsigned char c:directory)need((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.'||c=='/',"original directory character");
 for(size_t at=1;at<directory.size();){const auto end=directory.find('/',at);const auto part=directory.substr(at,end==std::string::npos?end:end-at);need(!part.empty()&&part!="."&&part!="..","original directory component");struct stat st{};const auto prefix=directory.substr(0,end);need(::lstat(prefix.c_str(),&st)==0&&S_ISDIR(st.st_mode),"original directory kind");if(end==std::string::npos)break;at=end+1;}
 need(directory.back()!='/',"original directory ending");
 const char* names[]={"config.json","tokenizer.model","tokenizer_config.json","tokenizer.json","special_tokens_map.json","model.safetensors"};
 const auto& files=config.at("files");need(files.is_object()&&files.size()==6,"original inventory");
 for(const auto* name:names){const auto& item=files.at(name);direct_tiny::keys(item,{"bytes","sha256"});need(item.size()==2&&item.at("bytes").is_number_unsigned(),"original file identity");const auto bytes=item.at("bytes").get<uint64_t>();const std::string hash=item.at("sha256");need(bytes>0&&bytes<=2200119864ULL&&hash.size()==64,"original file bound");for(char c:hash)need((c>='0'&&c<='9')||(c>='a'&&c<='f'),"original hash syntax");struct stat st{};need(::lstat((directory+"/"+name).c_str(),&st)==0&&S_ISREG(st.st_mode)&&st.st_size>=0&&uint64_t(st.st_size)==bytes,"original file size/kind");}
 return directory;
}
}
