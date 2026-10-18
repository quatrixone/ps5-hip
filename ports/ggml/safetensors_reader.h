#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps5hip {
namespace safetensors_detail {
inline void check(bool ok,const char* reason){if(!ok)throw std::runtime_error(std::string("safetensors: ")+reason);}
inline uint64_t integer(const nlohmann::json& value){check(value.is_number_unsigned(),"expected unsigned integer");return value.get<uint64_t>();}
inline nlohmann::json parse(const std::string& text) {
 using Json=nlohmann::json;std::vector<std::set<std::string>> keys;
 auto callback=[&](int depth,Json::parse_event_t event,Json& value) {
  check(depth<=8,"JSON nesting limit");
  if(event==Json::parse_event_t::object_start)keys.emplace_back();
  if(event==Json::parse_event_t::key)check(!keys.empty()&&keys.back().insert(value.get<std::string>()).second,"duplicate JSON key");
  if(event==Json::parse_event_t::object_end)keys.pop_back();
  return true;
 };
 return Json::parse(text,callback);
}
}
// Reader prerequisite only: no model mapping or dtype conversion.
class SafetensorsReader {
public:
 struct Tensor { std::string dtype; std::vector<uint64_t> shape; uint64_t begin,end; };
 static constexpr size_t max_header=8*1024*1024,max_chunk=1024*1024;
 explicit SafetensorsReader(const std::string& path):file(path,std::ios::binary) {
  check(bool(file),"open failed");file.seekg(0,std::ios::end);
  auto length=file.tellg();check(length>=8,"truncated prefix");size=uint64_t(length);file.seekg(0);
  unsigned char prefix[8];exact(prefix,8);uint64_t header=0;
  for(unsigned i=0;i<8;++i)header|=uint64_t(prefix[i])<<(i*8);
  check(header>0&&header<=max_header&&header<=size-8,"invalid header length");base=8+header;
  std::string text(size_t(header),'\0');exact(&text[0],text.size());check(text[0]=='{',"header must start with object");
  auto document=safetensors_detail::parse(text);check(document.is_object()&&document.size()<=100000+document.count("__metadata__"),"tensor count limit");
  static const std::map<std::string,uint64_t> widths={{"BOOL",1},{"U8",1},{"I8",1},{"U16",2},{"I16",2},{"F16",2},{"BF16",2},{"U32",4},{"I32",4},{"F32",4},{"U64",8},{"I64",8},{"F64",8}};
  for(auto it=document.begin();it!=document.end();++it) {
   const auto& item=it.value();
   if(it.key()=="__metadata__") {check(item.is_object(),"metadata must be object");for(const auto& value:item)check(value.is_string(),"metadata must contain strings");continue;}
   check(!it.key().empty()&&it.key().find('\0')==std::string::npos,"invalid tensor name");
   check(item.is_object()&&item.size()==3&&item.contains("dtype")&&item.contains("shape")&&item.contains("data_offsets"),"tensor fields differ");
   check(item["dtype"].is_string(),"dtype must be string");Tensor tensor;tensor.dtype=item["dtype"].get<std::string>();
   auto width=widths.find(tensor.dtype);check(width!=widths.end(),"unsupported dtype");
   const auto& shape=item["shape"];check(shape.is_array()&&shape.size()<=8,"shape rank limit");uint64_t bytes=width->second;
   for(const auto& dimension:shape) {auto n=integer(dimension);tensor.shape.push_back(n);check(!n||bytes<=std::numeric_limits<uint64_t>::max()/n,"shape overflow");bytes*=n;}
   const auto& offsets=item["data_offsets"];check(offsets.is_array()&&offsets.size()==2,"invalid offsets");
   tensor.begin=integer(offsets[0]);tensor.end=integer(offsets[1]);
   check(tensor.begin<=tensor.end&&tensor.end<=size-base&&tensor.end-tensor.begin==bytes,"tensor byte range differs");
   tensors.emplace(it.key(),std::move(tensor));
  }
  std::vector<const Tensor*> ordered;for(const auto& entry:tensors)ordered.push_back(&entry.second);
  std::sort(ordered.begin(),ordered.end(),[](const Tensor* a,const Tensor* b){return a->begin<b->begin||(a->begin==b->begin&&a->end<b->end);});
  uint64_t end=0;for(const auto* tensor:ordered){check(tensor->begin==end,"overlap or gap");end=tensor->end;}
  check(end==size-base,"unclaimed data bytes");
 }
 const std::map<std::string,Tensor>& inventory() const {return tensors;}
 void read(const std::string& name,uint64_t offset,void* destination,size_t bytes) {
  auto it=tensors.find(name);check(it!=tensors.end(),"unknown tensor");const auto& tensor=it->second;
  check(bytes<=max_chunk&&offset<=tensor.end-tensor.begin&&bytes<=tensor.end-tensor.begin-offset,"read range differs");
  check(destination||!bytes,"null destination");file.clear();file.seekg(0,std::ios::end);
  check(file.tellg()>=0&&uint64_t(file.tellg())==size,"file size changed");
  file.seekg(std::streamoff(base+tensor.begin+offset));check(bool(file),"seek failed");
  if(bytes)exact(destination,bytes);
 }
private:
 std::ifstream file;uint64_t size=0,base=0;std::map<std::string,Tensor> tensors;
 static void check(bool ok,const char* reason){safetensors_detail::check(ok,reason);}
 static uint64_t integer(const nlohmann::json& value){return safetensors_detail::integer(value);}
 void exact(void* target,size_t bytes){file.read(static_cast<char*>(target),std::streamsize(bytes));check(bool(file)&&size_t(file.gcount())==bytes,"truncated read");}
};
}
