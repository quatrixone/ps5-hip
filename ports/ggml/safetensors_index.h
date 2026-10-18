#pragma once
#include "safetensors_reader.h"
#include <filesystem>
#include <memory>

namespace ps5hip {
class SafetensorsIndex {
public:
 explicit SafetensorsIndex(const std::filesystem::path& index) {
  using namespace safetensors_detail;
  auto path=std::filesystem::absolute(index).lexically_normal();auto directory=path.parent_path();
  check(std::filesystem::canonical(directory)==directory,"symlink index directory");
  regular(path);std::ifstream file(path,std::ios::binary);check(bool(file),"index open failed");
  file.seekg(0,std::ios::end);auto length=file.tellg();check(length>0&&uint64_t(length)<=SafetensorsReader::max_header,"index size limit");
  file.seekg(0);std::string text(size_t(length),'\0');file.read(&text[0],length);check(bool(file),"index truncated");
  auto document=parse(text);check(document.is_object()&&document.contains("weight_map")&&document.size()==1+document.count("metadata"),"index fields differ");
  const auto& map=document["weight_map"];check(map.is_object()&&!map.empty()&&map.size()<=100000,"index tensor count limit");
  for(auto it=map.begin();it!=map.end();++it) {
   check(!it.key().empty()&&it.key().find('\0')==std::string::npos&&it.value().is_string(),"invalid weight map");
   std::string name=it.value().get<std::string>();
   check(name.size()>12&&name.substr(name.size()-12)==".safetensors"&&name.find_first_of("/\\:")==std::string::npos&&
         std::none_of(name.begin(),name.end(),[](unsigned char c){return c<32||c==127;}),"unsafe shard name");
   ownership.emplace(it.key(),name);
  }
  std::set<std::string> names;for(const auto& entry:ownership)names.insert(entry.second);
  check(names.size()<=64,"shard count limit");uint64_t total=0;size_t count=0;
  for(const auto& name:names) {
   auto shard=directory/name;regular(shard);auto reader=std::make_unique<SafetensorsReader>(shard.string());
   check(!reader->inventory().empty(),"empty mapped shard");
   for(const auto& entry:reader->inventory()) {
    auto owner=ownership.find(entry.first);check(owner!=ownership.end()&&owner->second==name,"tensor shard ownership differs");
    auto bytes=entry.second.end-entry.second.begin;check(total<=UINT64_MAX-bytes,"index total size overflow");total+=bytes;++count;
   }
   shards.emplace(name,std::move(reader));
  }
  check(count==ownership.size(),"mapped tensor missing");
  if(document.contains("metadata")) {
   const auto& metadata=document["metadata"];check(metadata.is_object()&&metadata.size()==metadata.count("total_size"),"unsupported index metadata");
   if(metadata.contains("total_size"))check(integer(metadata["total_size"])==total,"index total size differs");
  }
 }
 const std::map<std::string,std::string>& weight_map() const {return ownership;}
 const SafetensorsReader::Tensor& tensor(const std::string& name) const {
  return shards.at(ownership.at(name))->inventory().at(name);
 }
 void read(const std::string& name,uint64_t offset,void* destination,size_t bytes) {
  shards.at(ownership.at(name))->read(name,offset,destination,bytes);
 }
private:
 std::map<std::string,std::string> ownership;
 std::map<std::string,std::unique_ptr<SafetensorsReader>> shards;
 static void regular(const std::filesystem::path& path) {
  safetensors_detail::check(std::filesystem::is_regular_file(std::filesystem::symlink_status(path)),"index/shard must be regular non-symlink file");
 }
};
}
