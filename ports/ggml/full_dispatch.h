#pragma once
#include "full_graph.h"
#include "layer_dispatch.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <exception>

namespace ps5hip_full {
// One admitted original request at a time. No copied activation graph or implicit
// successful decode; the owner must finish with the real scheduler result.
struct Dispatch {
 using Span=ps5hip_prefix::Dispatch::Span;
 using Fail=ps5hip_prefix::Dispatch::Fail;
 ps5hip_layer::Dispatch layer;
 unsigned expected_position=0;
 bool active=false;
 bool controls_checked=false;
 Selection selection{};
 size_t cursor=0;
 struct Binding {ggml_tensor* tensor;void* data;ggml_backend_buffer* buffer;size_t last_use=0;};
 std::vector<Binding> resident;
 std::vector<Binding> request_inputs;
 std::vector<ggml_tensor*> sequence;
 std::vector<std::pair<ggml_tensor*,ggml_tensor>> captured;
 std::array<ggml_tensor*,3> inactive{};
 std::array<bool,3> inactive_seen{};
 Dispatch(Span owned,Fail fail):layer(owned,fail){}
 static constexpr size_t scratch_bytes=65536;
 void initialize(const std::array<const char*,6>& paths){
  layer.initialize(paths,true);
  layer.base.call(ps5hip_prefix::Call::scratch_allocate,scratch_bytes,"full_scratch_allocate",[&]{return hipMalloc(&layer.full_scratch,scratch_bytes);});
  layer.base.ok(layer.full_scratch?hipSuccess:hipErrorInvalidValue,"null_full_scratch");
 }
 void shutdown(){
  ps5hip_prefix::check(!active&&!layer.base.terminal,"active or uncertain full request release");
  layer.base.call(ps5hip_prefix::Call::scratch_free,scratch_bytes,"full_scratch_free",[&]{return hipFree(layer.full_scratch);});layer.full_scratch=nullptr;
  layer.shutdown();
 }
 static bool metadata(ggml_op op){return op==GGML_OP_VIEW||op==GGML_OP_RESHAPE||op==GGML_OP_PERMUTE;}
 static bool disjoint(const ggml_tensor* a,const ggml_tensor* b){return ps5hip_prefix::Dispatch::disjoint(a,b);}
 void span(const ggml_tensor* t) const {
  check(t&&t->data&&layer.base.owned(t),"full GPU span");
  for(unsigned depth=0;t->view_src;++depth){
   check(depth<16,"full view cycle");auto* parent=t->view_src;
   check(parent->data&&layer.base.owned(parent),"full view owner span");
   const auto p=reinterpret_cast<uintptr_t>(parent->data),v=reinterpret_cast<uintptr_t>(t->data);
   check(t->view_offs<=UINTPTR_MAX-p&&v==p+t->view_offs,"full view address");
   check(t->view_offs<=ggml_nbytes(parent)&&ggml_nbytes(t)<=ggml_nbytes(parent)-t->view_offs,"full view bounds");t=parent;
  }
 }
 void capture(ggml_tensor* t){
  if(std::any_of(captured.begin(),captured.end(),[&](const auto& p){return p.first==t;}))return;
  span(t);captured.emplace_back(t,*t);
  const auto scratch=reinterpret_cast<uintptr_t>(layer.full_scratch),data=reinterpret_cast<uintptr_t>(t->data);
  check(scratch&&scratch<=UINTPTR_MAX-scratch_bytes&&data<=UINTPTR_MAX-ggml_nbytes(t)&&
        (data+ggml_nbytes(t)<=scratch||scratch+scratch_bytes<=data),"full scratch tensor alias");
  for(auto* s:t->src)if(s)capture(s);
  if(t->view_src)capture(t->view_src);
 }
 bool unchanged() const {
  for(const auto& p:captured){const auto& a=*p.first;const auto& b=p.second;
   if(a.type!=b.type||a.op!=b.op||a.flags!=b.flags||a.data!=b.data||a.buffer!=b.buffer||a.view_src!=b.view_src||a.view_offs!=b.view_offs||
      std::memcmp(a.ne,b.ne,sizeof(a.ne))||std::memcmp(a.nb,b.nb,sizeof(a.nb))||std::memcmp(a.src,b.src,sizeof(a.src))||std::memcmp(a.op_params,b.op_params,sizeof(a.op_params)))return false;
  }return true;
 }
 void aliases(ggml_tensor* t,const std::vector<Binding>& owners,size_t ordinal) const {
  if(metadata(t->op))return;
  for(const auto& p:request_inputs)if(ordinal<=p.last_use)check(disjoint(t,p.tensor),"full output overlaps live request control");
  for(const auto& p:owners){
   if(t->op==GGML_OP_SET_ROWS&&t->view_src==p.tensor)continue;
   check(disjoint(t,p.tensor),"full output overlaps resident tensor");
  }
  for(unsigned i=0;i<GGML_MAX_SRC;++i)if(auto* s=t->src[i]){
   if(t->op==GGML_OP_SET_ROWS&&i==2)continue;
   bool exact=t->data==s->data&&ggml_nbytes(t)==ggml_nbytes(s);
   bool inplace=t->op==GGML_OP_RMS_NORM||(t->op==GGML_OP_GET_ROWS&&i==0&&s->type==GGML_TYPE_F32)||t->op==GGML_OP_ADD||t->op==GGML_OP_MUL||t->op==GGML_OP_ROPE||(t->op==GGML_OP_GLU&&i==0)||(t->op==GGML_OP_SOFT_MAX&&i==0&&layer.full_scratch);
   if(!(disjoint(t,s)||(exact&&inplace)))std::fprintf(stderr,"full alias rejected: %s/%s %p+%zu source%u %s %p+%zu\n",t->name,ggml_op_name(t->op),t->data,ggml_nbytes(t),i,s->name,s->data,ggml_nbytes(s));
   check(disjoint(t,s)||(exact&&inplace),"full unsafe live input alias");
  }
 }
 bool begin(ggml_cgraph* graph){
  if(active||layer.base.terminal||!layer.base.ready||expected_position>=512)return false;
  try{
   auto selected=select(graph,true);check(selected.layers[0].extent==(expected_position<256?256:512),"full request extent");
   std::vector<Binding> owners;
   auto add=[&](ggml_tensor* t){span(t);if(std::none_of(owners.begin(),owners.end(),[&](const auto& x){return x.tensor==t;}))owners.push_back({t,t->data,t->buffer});};
   for(auto& l:selected.layers){for(auto* t:l.weights)add(t);for(auto* t:l.caches)add(t);}
   add(selected.logits->src[0]);add(selected.logits->src[1]->src[1]);
   for(size_t i=0;i<owners.size();++i)for(size_t j=0;j<i;++j)check(disjoint(owners[i].tensor,owners[j].tensor),"full resident alias");
   request_inputs.clear();
   auto protect=[&](ggml_tensor* t){span(t);for(const auto& p:owners)check(disjoint(t,p.tensor),"full control overlaps resident");for(const auto& p:request_inputs)check(disjoint(t,p.tensor),"full control overlap");request_inputs.push_back({t,t->data,t->buffer});};
   for(auto* t:selected.layers[0].requests)protect(t);protect(selected.output_ids);
   if(!resident.empty()){
    check(resident.size()==owners.size(),"full resident inventory changed");
    for(size_t i=0;i<owners.size();++i)check(resident[i].tensor==owners[i].tensor&&resident[i].data==owners[i].data&&resident[i].buffer==owners[i].buffer,"full resident identity changed");
   }
   sequence.clear();captured.clear();inactive_seen={};
   for(int i=0;i<801;++i){auto* t=ggml_graph_node(graph,i);if(i>=1&&i<=3){inactive[i-1]=t;continue;}capture(t);sequence.push_back(t);}
   // Scheduler storage can be reused only after a control's final consumer.
   for(auto& p:request_inputs)for(size_t i=0;i<sequence.size();++i)for(auto* s:sequence[i]->src)if(s==p.tensor)p.last_use=i;
   bool bad_alias=false;
   for(size_t i=0;i<sequence.size();++i)try{aliases(sequence[i],owners,i);}catch(const std::exception& e){std::fprintf(stderr,"full alias preflight %s: %s\n",sequence[i]->name,e.what());bad_alias=true;}
   check(!bad_alias,"full unsafe alias inventory");
   selection=selected;controls_checked=false;resident=std::move(owners);layer.expected_position=expected_position;cursor=0;active=true;return true;
  }catch(const std::exception& e){std::fprintf(stderr,"ps5hip full begin rejected: %s\n",e.what());sequence.clear();captured.clear();return false;}
  catch(...){sequence.clear();captured.clear();return false;}
 }
 void validate_controls(){
   auto& controls=selection.layers[0].requests;
   int32_t token=-1,position=-1,out=-1;int64_t k=-1;std::array<int64_t,256> v{};std::array<float,512> mask{};
   layer.read(controls[0],&token,4);layer.read(controls[1],&position,4);layer.read(controls[2],&k,8);layer.read(controls[3],v.data(),sizeof(v));
   layer.read(controls[4],mask.data(),size_t(selection.layers[0].extent)*4);layer.read(selection.output_ids,&out,4);
   check(token>=0&&token<32000&&position==int32_t(expected_position)&&k==position&&out==0,"full live input values");
   for(size_t i=0;i<v.size();++i)check(v[i]==int64_t(i*512+expected_position),"full V slot");
   for(int i=0;i<selection.layers[0].extent;++i)check(i<=position?mask[i]==0:(std::isinf(mask[i])&&mask[i]<0),"full causal mask");
  controls_checked=true;
 }
 bool fail(){if(active&&cursor)layer.base.terminal=true;active=false;return false;}
 bool execute_graph(ggml_cgraph* split){
  if(!active||!split||ggml_graph_n_nodes(split)<=0||!unchanged())return fail();
  try{
   // Validate the whole subset before its first submission.
   size_t next=cursor;auto seen=inactive_seen;
   for(int i=0;i<ggml_graph_n_nodes(split);++i){auto* t=ggml_graph_node(split,i);auto it=std::find(inactive.begin(),inactive.end(),t);
    if(it!=inactive.end()){size_t index=size_t(it-inactive.begin());check(next==1&&!seen[index],"full inactive node order");seen[index]=true;continue;}
    check(next<sequence.size()&&sequence[next]==t,"full split order");span(t);for(auto* s:t->src)if(s)span(s);aliases(t,resident,next);++next;
   }
   if(!controls_checked)validate_controls();
   for(int i=0;i<ggml_graph_n_nodes(split);++i){auto* t=ggml_graph_node(split,i);if(std::find(inactive.begin(),inactive.end(),t)!=inactive.end())continue;
    // An exception may occur after a submission: quarantine conservatively.
    try{check(layer.execute_node(t,true),"full node execution");}catch(...){layer.base.terminal=true;throw;}
    ++cursor;
   }
   inactive_seen=seen;return true;
  }catch(const std::exception& e){std::fprintf(stderr,"ps5hip full execute rejected at %zu: %s\n",cursor,e.what());return fail();}
  catch(...){return fail();}
 }
 bool finish(bool scheduler_success){
  if(!active||!scheduler_success||cursor!=sequence.size()||!unchanged())return fail();
  active=false;++expected_position;layer.expected_position=expected_position;return true;
 }
 void reset(){check(!active&&!layer.base.terminal,"full reset state");layer.reset();expected_position=0;cursor=0;sequence.clear();captured.clear();}
};
}
