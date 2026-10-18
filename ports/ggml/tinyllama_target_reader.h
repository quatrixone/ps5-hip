#pragma once
#include "tinyllama_source_tensors.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
namespace ps5hip {
// Caller authenticates immutable source bytes and supplies a validated TinyLlama mapping.
// Sequential use only: the underlying stream and these row buffers are mutable.
class TinyLlamaTargetReader {
public:
 TinyLlamaTargetReader(SafetensorsReader& source,SourceTensorMapping mapping,ggml_type type):source_(source),mapping_(std::move(mapping)),type_(type) {
  using safetensors_detail::check;
  const uint16_t endian=1;check(*reinterpret_cast<const unsigned char*>(&endian)==1,"little-endian target required");
  auto found=source.inventory().find(mapping_.source);check(found!=source.inventory().end(),"missing source tensor");
  const auto& tensor=found->second;check(tensor.shape==mapping_.shape,"source mapping shape differs");
  check(tensor.dtype=="F16"||tensor.dtype=="BF16","source dtype unsupported");bf16_=tensor.dtype=="BF16";
  check(mapping_.shape.size()==1||mapping_.shape.size()==2,"target rank unsupported");
  norm_=mapping_.shape.size()==1;columns_=mapping_.shape.back();rows_=norm_?1:mapping_.shape[0];
  check(columns_>0&&columns_<=5632&&rows_>0,"bounded row shape required");
  check(type_==(norm_?GGML_TYPE_F32:GGML_TYPE_Q4_0),"target dtype differs");
  check(!norm_||!mapping_.permutation_heads,"norm permutation forbidden");
  if(mapping_.permutation_heads)check(rows_%mapping_.permutation_heads==0&&(rows_/mapping_.permutation_heads)%2==0,"permutation heads differ");
  check(norm_||columns_%32==0,"Q4_0 row alignment");
  row_bytes_=uint64_t(ggml_row_size(type_,int64_t(columns_)));
  check(rows_<=UINT64_MAX/row_bytes_,"target size overflow");size_=rows_*row_bytes_;
  raw_.resize(size_t(columns_*2));values_.resize(size_t(columns_));prepared_.resize(size_t(row_bytes_));
 }
 uint64_t size() const {return size_;}
 size_t scratch_bytes() const {return raw_.size()+values_.size()*sizeof(float)+prepared_.size();}
 void read(uint64_t offset,void* destination,size_t bytes) {
  using safetensors_detail::check;
  check(offset<=size_&&bytes<=size_-offset,"target read range differs");check(destination||!bytes,"null target destination");
  auto* output=static_cast<unsigned char*>(destination);
  while(bytes) {
   const auto row=offset/row_bytes_,within=offset%row_bytes_;prepare(row);
   const auto count=size_t(std::min<uint64_t>(bytes,row_bytes_-within));
   std::memcpy(output,prepared_.data()+within,count);output+=count;offset+=count;bytes-=count;
  }
 }
private:
 SafetensorsReader& source_;SourceTensorMapping mapping_;ggml_type type_;bool norm_=false,bf16_=false;
 uint64_t columns_=0,rows_=0,row_bytes_=0,size_=0;
 std::vector<unsigned char> raw_,prepared_;std::vector<float> values_;
 void prepare(uint64_t row) {
  using safetensors_detail::check;
  const auto offset=row*columns_*2;
  if(mapping_.permutation_heads)read_llama_permuted16(source_,mapping_.source,mapping_.permutation_heads,offset,raw_.data(),raw_.size());
  else source_.read(mapping_.source,offset,raw_.data(),raw_.size());
  for(size_t i=0;i<values_.size();++i) {
   const uint16_t bits=uint16_t(raw_[2*i])|(uint16_t(raw_[2*i+1])<<8);
   float value=bf16_?ggml_bf16_to_fp32(ggml_bf16_t{bits}):ggml_fp16_to_fp32(bits);
   check(std::isfinite(value),"nonfinite source value");
   if(!norm_)value=ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
   check(std::isfinite(value),"nonfinite F16 intermediate");values_[i]=value;
  }
  if(norm_)std::memcpy(prepared_.data(),values_.data(),prepared_.size());
  else check(ggml_quantize_chunk(GGML_TYPE_Q4_0,values_.data(),prepared_.data(),0,1,int64_t(columns_),nullptr)==prepared_.size(),"quantizer size differs");
 }
};
}
