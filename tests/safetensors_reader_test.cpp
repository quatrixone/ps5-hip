#include "safetensors_reader.h"
#include "safetensors_index.h"
#include <cassert>
#include <functional>
#include <iostream>
#include <unistd.h>
using Reader=ps5hip::SafetensorsReader;
using Json=nlohmann::json;
static unsigned checks=0;
static void rejects(const std::function<void()>& call) {++checks;bool caught=false;try{call();}catch(const std::exception&){caught=true;}assert(caught);}
int main(int argc,char** argv) {
 assert(argc==2);std::string path=argv[1];
 rejects([&]{Reader reader(path+".missing");});
 auto write=[&](const std::string& header,const std::string& data,uint64_t size=0) {
  std::ofstream out(path,std::ios::binary|std::ios::trunc);if(!size)size=header.size();
  for(unsigned i=0;i<8;++i){out.put(char(size>>(i*8)));}
  out<<header;out.write(data.data(),data.size());assert(bool(out));
 };
 auto tensor=[](std::string dtype,Json shape,Json offsets){return Json{{"dtype",dtype},{"shape",shape},{"data_offsets",offsets}};};
 auto valid=tensor("F16",Json::array({2}),Json::array({0,4}));
 auto bad=[&](Json value,const std::string& data="1234"){write(value.dump(),data);rejects([&]{Reader reader(path);});};
 write(Json{{"weight",valid},{"__metadata__",{{"format","pt"}}}}.dump(),"1234");
 {Reader reader(path);assert(reader.inventory().at("weight").shape==std::vector<uint64_t>{2});char out[4]{};
  reader.read("weight",0,out,2);reader.read("weight",2,out+2,2);assert(std::string(out,4)=="1234");++checks;
  rejects([&]{reader.read("missing",0,out,1);});rejects([&]{reader.read("weight",4,out,1);});
  rejects([&]{reader.read("weight",5,out,0);});rejects([&]{reader.read("weight",0,nullptr,1);});
  rejects([&]{reader.read("weight",0,out,Reader::max_chunk+1);});reader.read("weight",4,nullptr,0);
  assert(truncate(path.c_str(),8)==0);rejects([&]{reader.read("weight",0,out,4);});}
 for(const auto& type:{"F16","BF16","F32","F64","BOOL","I8","U8","I16","U16","I32","U32","I64","U64"}) {
  write(Json{{"empty",tensor(type,Json::array({0}),Json::array({0,0}))}}.dump(),"");Reader reader(path);++checks;
 }
 write(Json{{"scalar",tensor("U8",Json::array(),Json::array({0,1}))}}.dump(),"x");{Reader reader(path);++checks;}
 write("{}","");{Reader reader(path);++checks;}
 for(const auto& header:{std::string("{\"w\":")+valid.dump()+",\"w\":"+valid.dump()+"}",
       std::string("{\"w\":{\"dtype\":\"F16\",\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]}}"),
       std::string("{\"__metadata__\":{\"a\":\"b\",\"a\":\"c\"},\"w\":")+valid.dump()+"}",
       std::string("[]"),std::string("{} garbage"),std::string("{bad"),std::string(" {}")}) {
  write(header,"1234");rejects([&]{Reader reader(path);});
 }
 for(const auto& shape:{Json::array({-1}),Json::array({1.5}),Json::array({true}),Json::array({UINT64_MAX}),Json::array({UINT64_MAX,2}),Json::array({1,1,1,1,1,1,1,1,1})}) {
  auto value=valid;value["shape"]=shape;bad(Json{{"w",value}});
 }
 for(const auto& offsets:{Json::array({0,3}),Json::array({1,5}),Json::array({4,0}),Json::array({0,5}),Json::array({-1,3}),Json::array({0,4.0}),Json::array({0}),Json::array({UINT64_MAX-3,UINT64_MAX}),Json::array({0,UINT64_MAX})}) {
  auto value=valid;value["data_offsets"]=offsets;bad(Json{{"w",value}});
 }
 auto value=valid;value["dtype"]="F8_E4M3";bad(Json{{"w",value}});
 value=valid;value["extra"]=0;bad(Json{{"w",value}});value=valid;value.erase("dtype");bad(Json{{"w",value}});
 bad(Json{{"w",valid},{"v",valid}});bad(Json{{"w",valid}},"12345");bad(Json{{"w",valid}},"123");
 bad(Json{{"__metadata__",{{"a",1}}},{"w",valid}});bad(Json{{"",valid}});
 write("{}","",Reader::max_header+1);rejects([&]{Reader reader(path);});
 write("{","",2);rejects([&]{Reader reader(path);});
 {std::ofstream out(path,std::ios::trunc);out<<"123";}rejects([&]{Reader reader(path);});
 {std::ofstream out(path,std::ios::binary|std::ios::trunc);out.write("\0\0\0\0\0\0\0\0",8);}rejects([&]{Reader reader(path);});
 write("{\"w\":[[[[[[[[[[[0]]]]]]]]]]]}","");rejects([&]{Reader reader(path);});
 // Empty ranges at a shared boundary are valid and cannot overlap nonempty storage.
 write(Json{{"a",tensor("U8",Json::array({0}),Json::array({0,0}))},{"b",tensor("U8",Json::array({1}),Json::array({0,1}))}}.dump(),"x");{Reader reader(path);++checks;}
 // Header keys sort unlike storage order; empty tensors may share either boundary.
 write(Json{{"z-first",tensor("U8",Json::array({2}),Json::array({0,2}))},
            {"a-second",tensor("U8",Json::array({2}),Json::array({2,4}))},
            {"middle-empty",tensor("F16",Json::array({0}),Json::array({2,2}))},
            {"end-empty",tensor("BF16",Json::array({0}),Json::array({4,4}))}}.dump(),"abcd");
 {Reader reader(path);char first[2],second[2];reader.read("z-first",0,first,2);reader.read("a-second",0,second,2);
  reader.read("middle-empty",0,nullptr,0);reader.read("end-empty",0,nullptr,0);
  assert(std::string(first,2)=="ab"&&std::string(second,2)=="cd");++checks;}
 std::string payload(Reader::max_chunk+1,'z');write(Json{{"large",tensor("U8",Json::array({payload.size()}),Json::array({0,payload.size()}))}}.dump(),payload);
 {Reader reader(path);std::vector<char> buffer(Reader::max_chunk);reader.read("large",0,buffer.data(),buffer.size());char last;reader.read("large",buffer.size(),&last,1);assert(last=='z'&&buffer.front()=='z'&&buffer.back()=='z');++checks;}
 // Two tiny shards exercise the HF index without reading a model's weights.
 using Index=ps5hip::SafetensorsIndex;namespace fs=std::filesystem;
 auto folder=fs::path(path).parent_path();auto first=folder/"a.safetensors",second=folder/"b.safetensors";
 auto shard=[&](const fs::path& target,const std::string& name) {
  write(Json{{name,tensor("U8",Json::array({2}),Json::array({0,2}))}}.dump(),name=="a"?"ab":"cd");fs::rename(path,target);
 };
 shard(first,"a");shard(second,"b");
 Json index={{"metadata",{{"total_size",4}}},{"weight_map",{{"a","a.safetensors"},{"b","b.safetensors"}}}};
 auto index_write=[&](const std::string& text){std::ofstream out(path,std::ios::trunc);out<<text;assert(bool(out));};
 auto bad_index=[&](const Json& value){index_write(value.dump());rejects([&]{Index reader(path);});};
 index_write(index.dump());{Index reader(path);char a[2],b[2];reader.read("a",0,a,2);reader.read("b",0,b,2);
  assert(std::string(a,2)=="ab"&&std::string(b,2)=="cd"&&reader.weight_map().size()==2&&reader.tensor("a").dtype=="U8");++checks;
  rejects([&]{reader.read("missing",0,a,1);});rejects([&]{reader.read("a",0,a,Reader::max_chunk+1);});}
 for(const auto& name:{"../a.safetensors","/a.safetensors","x/a.safetensors","x\\a.safetensors","C:a.safetensors","a.bin",".safetensors","bad\n.safetensors"}) {
  auto value=index;value["weight_map"]["a"]=name;bad_index(value);
 }
 auto changed=index;changed["weight_map"]["a"]="missing.safetensors";bad_index(changed);
 {auto value=index;value.erase("metadata");index_write(value.dump());Index reader(path);++checks;}
 changed=index;changed["weight_map"]=Json::object();for(unsigned i=0;i<65;++i)changed["weight_map"][std::to_string(i)]=std::to_string(i)+".safetensors";bad_index(changed);
 changed=index;changed["weight_map"]["a"]="b.safetensors";bad_index(changed);
 changed=index;changed["weight_map"]["extra"]="a.safetensors";bad_index(changed);
 changed=index;changed["weight_map"].erase("b");changed["weight_map"]["a"]="b.safetensors";bad_index(changed);
 changed=index;changed["metadata"]["total_size"]=5;bad_index(changed);
 changed=index;changed["metadata"]["total_size"]=-1;bad_index(changed);
 changed=index;changed["weight_map"]=Json::object();bad_index(changed);
 changed=index;changed["weight_map"]["a"]=1;bad_index(changed);
 changed=index;changed["extra"]=0;bad_index(changed);
 for(const auto& text:{"{\"weight_map\":{\"a\":\"a.safetensors\",\"a\":\"b.safetensors\"}}",
                       "{\"weight_map\":{},\"weight_map\":{}}","{bad"}) {
  index_write(text);rejects([&]{Index reader(path);});
 }
 index_write(std::string(Reader::max_header+1,' '));rejects([&]{Index reader(path);});
 // An extra tensor duplicated in another shard cannot be hidden by the index.
 shard(second,"a");bad_index(index);shard(second,"b");
 fs::remove(first);fs::create_symlink(second,first);bad_index(index);fs::remove(first);shard(first,"a");
 index_write(index.dump());auto link=folder/"index-link.json";fs::create_symlink(path,link);rejects([&]{Index reader(link);});fs::remove(link);
 auto dirlink=folder/"directory-link";fs::create_directory_symlink(folder,dirlink);rejects([&]{Index reader(dirlink/fs::path(path).filename());});fs::remove(dirlink);
 fs::remove(first);fs::remove(second);std::remove(path.c_str());std::cout<<checks<<" checks PASS\n";
}
