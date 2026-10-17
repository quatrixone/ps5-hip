#pragma once
#include "ggml.h"
#include <array>
#include <cstring>
#include <set>
#include <vector>
#include <stdexcept>

namespace ps5hip_layer {
inline void check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
struct Upload {ggml_tensor *original;ggml_tensor *clone;};
struct Selection {std::array<ggml_tensor*,37> nodes{};std::array<ggml_tensor*,10> weights{};std::array<ggml_tensor*,5> requests{};std::array<ggml_tensor*,2> caches{};int64_t extent=0;};
struct Weights {std::array<Upload,10> uploads{};};
struct Cache {ggml_tensor *k;ggml_tensor *v;};
struct Graph {ggml_cgraph* graph=nullptr;std::array<ggml_tensor*,37> nodes{};std::array<Upload,5> uploads{};std::array<ggml_tensor*,21> outputs{};int64_t extent=0;};
namespace detail {
struct Spec {const char* name;ggml_type type;ggml_op op;int64_t ne[4];size_t nb[4];uint32_t params[16];int refs[4];int owner;};
static const std::array<Spec,37> node_specs={{
{"embd",GGML_TYPE_F32,GGML_OP_GET_ROWS,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-1,-11,-100,-100},-100},
{"norm-0",GGML_TYPE_F32,GGML_OP_RMS_NORM,{2048,1,1,1},{4,8192,8192,8192},{925353388u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{0,-100,-100,-100},-100},
{"attn_norm-0",GGML_TYPE_F32,GGML_OP_MUL,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{1,-2,-100,-100},-100},
{"Qcur-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-3,2,-100,-100},-100},
{"Qcur-0",GGML_TYPE_F32,GGML_OP_RESHAPE,{64,32,1,1},{4,256,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{3,-100,-100,-100},3},
{"Qcur-0",GGML_TYPE_F32,GGML_OP_ROPE,{64,32,1,1},{4,256,8192,8192},{0u,64u,0u,0u,2048u,1176256512u,1065353216u,0u,1065353216u,1107296256u,1065353216u,0u,0u,0u,0u,0u},{4,-12,-100,-100},-100},
{"Vcur-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{256,1,1,1},{4,1024,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-5,2,-100,-100},-100},
{"Vcur-0",GGML_TYPE_F32,GGML_OP_RESHAPE,{64,4,1,1},{4,256,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{6,-100,-100,-100},6},
{"Kcur-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{256,1,1,1},{4,1024,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-4,2,-100,-100},-100},
{"Kcur-0",GGML_TYPE_F32,GGML_OP_RESHAPE,{64,4,1,1},{4,256,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{8,-100,-100,-100},8},
{"Kcur-0",GGML_TYPE_F32,GGML_OP_ROPE,{64,4,1,1},{4,256,1024,1024},{0u,64u,0u,0u,2048u,1176256512u,1065353216u,0u,1065353216u,1107296256u,1065353216u,0u,0u,0u,0u,0u},{9,-12,-100,-100},-100},
{"Kcur-0 (view)",GGML_TYPE_F32,GGML_OP_VIEW,{256,1,1,1},{4,1024,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{10,-100,-100,-100},10},
{"cache_k_l0 (view)",GGML_TYPE_F16,GGML_OP_SET_ROWS,{256,512,1,1},{2,512,262144,262144},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{11,-13,-16,-100},-16},
{"Vcur-0 (reshaped)",GGML_TYPE_F32,GGML_OP_RESHAPE,{256,1,1,1},{4,1024,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{7,-100,-100,-100},6},
{"Vcur-0 (reshaped) (reshaped)",GGML_TYPE_F32,GGML_OP_RESHAPE,{1,256,1,1},{4,4,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{13,-100,-100,-100},6},
{"cache_v_l0 (reshaped)",GGML_TYPE_F16,GGML_OP_RESHAPE,{1,131072,1,1},{2,2,262144,262144},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-17,-100,-100,-100},-17},
{"cache_v_l0 (reshaped) (view)",GGML_TYPE_F16,GGML_OP_SET_ROWS,{1,131072,1,1},{2,2,262144,262144},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{14,-14,15,-100},-17},
{"cache_v_l0 (view)",GGML_TYPE_F16,GGML_OP_VIEW,{256,4,64,1},{2,65536,1024,262144},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-17,-100,-100,-100},-17},
{"cache_v_l0 (view) (permuted)",GGML_TYPE_F16,GGML_OP_PERMUTE,{256,64,4,1},{2,1024,65536,262144},{0u,2u,1u,3u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{17,-100,-100,-100},-17},
{"cache_k_l0 (view)",GGML_TYPE_F16,GGML_OP_VIEW,{64,4,256,1},{2,128,512,262144},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-16,-100,-100,-100},-16},
{"cache_k_l0 (view) (permuted)",GGML_TYPE_F16,GGML_OP_PERMUTE,{64,256,4,1},{2,512,128,262144},{0u,2u,1u,3u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{19,-100,-100,-100},-16},
{"Qcur-0 (view)",GGML_TYPE_F32,GGML_OP_VIEW,{64,32,1,1},{4,256,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{5,-100,-100,-100},5},
{"Qcur-0 (view) (permuted)",GGML_TYPE_F32,GGML_OP_PERMUTE,{64,1,32,1},{4,8192,256,8192},{0u,2u,1u,3u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{21,-100,-100,-100},5},
{"kq-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{256,1,32,1},{4,1024,1024,32768},{10u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{20,22,-100,-100},-100},
{"kq_soft_max-0",GGML_TYPE_F32,GGML_OP_SOFT_MAX,{256,1,32,1},{4,1024,1024,32768},{1040187392u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{23,-15,-100,-100},-100},
{"kqv-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{64,1,32,1},{4,256,256,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{18,24,-100,-100},-100},
{"kqv-0 (permuted)",GGML_TYPE_F32,GGML_OP_PERMUTE,{64,32,1,1},{4,256,256,8192},{0u,2u,1u,3u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{25,-100,-100,-100},25},
{"kqv_out-0",GGML_TYPE_F32,GGML_OP_CONT,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{26,-100,-100,-100},-100},
{"attn_out-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-6,27,-100,-100},-100},
{"ffn_inp-0",GGML_TYPE_F32,GGML_OP_ADD,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{28,0,-100,-100},-100},
{"norm-0",GGML_TYPE_F32,GGML_OP_RMS_NORM,{2048,1,1,1},{4,8192,8192,8192},{925353388u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{29,-100,-100,-100},-100},
{"ffn_norm-0",GGML_TYPE_F32,GGML_OP_MUL,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{30,-7,-100,-100},-100},
{"ffn_gate-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{5632,1,1,1},{4,22528,22528,22528},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-8,31,-100,-100},-100},
{"ffn_up-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{5632,1,1,1},{4,22528,22528,22528},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-9,31,-100,-100},-100},
{"ffn_swiglu-0",GGML_TYPE_F32,GGML_OP_GLU,{5632,1,1,1},{4,22528,22528,22528},{2u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{32,33,-100,-100},-100},
{"ffn_out-0",GGML_TYPE_F32,GGML_OP_MUL_MAT,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-10,34,-100,-100},-100},
{"l_out-0",GGML_TYPE_F32,GGML_OP_ADD,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{35,29,-100,-100},-100}
}};
static const std::array<Spec,17> leaf_specs={{
{"token_embd.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{2048,32000,1,1},{18,1152,36864000,36864000},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.attn_norm.weight",GGML_TYPE_F32,GGML_OP_NONE,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.attn_q.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{2048,2048,1,1},{18,1152,2359296,2359296},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.attn_k.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{2048,256,1,1},{18,1152,294912,294912},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.attn_v.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{2048,256,1,1},{18,1152,294912,294912},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.attn_output.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{2048,2048,1,1},{18,1152,2359296,2359296},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.ffn_norm.weight",GGML_TYPE_F32,GGML_OP_NONE,{2048,1,1,1},{4,8192,8192,8192},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.ffn_gate.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{2048,5632,1,1},{18,1152,6488064,6488064},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.ffn_up.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{2048,5632,1,1},{18,1152,6488064,6488064},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"blk.0.ffn_down.weight",GGML_TYPE_Q4_0,GGML_OP_NONE,{5632,2048,1,1},{18,3168,6488064,6488064},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"inp_tokens",GGML_TYPE_I32,GGML_OP_NONE,{1,1,1,1},{4,4,4,4},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"inp_pos",GGML_TYPE_I32,GGML_OP_NONE,{1,1,1,1},{4,4,4,4},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"attn_inp_k_idxs",GGML_TYPE_I64,GGML_OP_NONE,{1,1,1,1},{8,8,8,8},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"attn_inp_v_idxs",GGML_TYPE_I64,GGML_OP_NONE,{256,1,1,1},{8,2048,2048,2048},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"attn_inp_kq_mask",GGML_TYPE_F32,GGML_OP_NONE,{256,1,1,1},{4,1024,1024,1024},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"cache_k_l0",GGML_TYPE_F16,GGML_OP_NONE,{256,512,1,1},{2,512,262144,262144},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100},
{"cache_v_l0",GGML_TYPE_F16,GGML_OP_NONE,{256,512,1,1},{2,512,262144,262144},{0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u},{-100,-100,-100,-100},-100}
}};
inline Spec spec(size_t index,int64_t n,bool leaf=false){
 check(n==256||n==512,"layer cache extent");auto s=leaf?leaf_specs.at(index):node_specs.at(index);
 if(leaf&&index==14){s.ne[0]=n;s.nb[1]=s.nb[2]=s.nb[3]=size_t(4*n);}
 if(!leaf){
  if(index==17||index==18)s.ne[0]=n;
  if(index==19)s.ne[2]=n;
  if(index==20)s.ne[1]=n;
  if(index==23||index==24){s.ne[0]=n;s.nb[1]=s.nb[2]=size_t(4*n);s.nb[3]=size_t(128*n);}
 }
 return s;
}
inline void match(const ggml_tensor* t,const Spec& s){
 check(t&&t->type==s.type&&t->op==s.op&&!std::strcmp(t->name,s.name),"layer tensor identity/type/op");
 check(sizeof(t->op_params)==sizeof(s.params),"layer op-parameter ABI");
 check(!std::memcmp(t->ne,s.ne,sizeof(s.ne))&&!std::memcmp(t->nb,s.nb,sizeof(s.nb))&&!std::memcmp(t->op_params,s.params,sizeof(s.params)),"layer shape/stride/parameters");
 check(t->view_offs==0,"layer nonzero view offset");
}
inline ggml_tensor* leaf(const Selection& s,size_t i){return i<10?s.weights[i]:i<15?s.requests[i-10]:s.caches[i-15];}
inline ggml_tensor* ref(const Selection& s,int index){return index>=0?s.nodes.at(size_t(index)):leaf(s,size_t(-1-index));}
}
inline Selection select(ggml_cgraph* graph){
 check(graph,"layer missing graph");Selection s{};ggml_tensor *end=nullptr,*ks=nullptr,*vs=nullptr;std::set<ggml_tensor*> available;
 for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);check(t&&available.insert(t).second,"layer null/duplicate graph node");
  if(!std::strcmp(t->name,"l_out-0")&&t->op==GGML_OP_ADD){check(!end,"layer duplicate endpoint");end=t;}
  if(!std::strcmp(t->name,"cache_k_l0 (view)")&&t->op==GGML_OP_SET_ROWS){check(!ks,"layer duplicate K store");ks=t;}
  if(!std::strcmp(t->name,"cache_v_l0 (reshaped) (view)")&&t->op==GGML_OP_SET_ROWS){check(!vs,"layer duplicate V store");vs=t;}
 }
 check(end&&ks&&vs,"layer endpoint/cache stores missing");std::set<ggml_tensor*> needed,active;
 const auto visit=[&](auto&& self,ggml_tensor* t)->void{
  check(t,"layer null source");if(t->op==GGML_OP_NONE||needed.count(t))return;
  check(available.count(t)&&active.insert(t).second&&active.size()<=37,"layer missing/cyclic dependency");
  for(auto* source:t->src)if(source)self(self,source);
  active.erase(t);needed.insert(t);check(needed.size()<=37,"layer extra dependency");
 };
 visit(visit,ks);visit(visit,vs);visit(visit,end);check(needed.size()==37,"layer exact node inventory");
 size_t count=0;for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto* t=ggml_graph_node(graph,i);if(needed.count(t))s.nodes.at(count++)=t;}
 s.extent=s.nodes[23]->ne[0];check(s.extent==256||s.extent==512,"layer attention extent");
 std::array<ggml_tensor*,17> leaves{};
 for(size_t i=0;i<s.nodes.size();++i){auto* t=s.nodes[i];auto expected=detail::spec(i,s.extent);detail::match(t,expected);
  for(int j=0;j<GGML_MAX_SRC;++j){const int index=j<4?expected.refs[j]:-100;auto* source=t->src[j];
   if(index==-100){check(!source,"layer unexpected source slot");continue;}
   check(source,"layer missing source slot");
   if(index>=0){check(size_t(index)<i&&source==s.nodes.at(size_t(index)),"layer source topology");}
   else {const size_t k=size_t(-1-index);check(!leaves.at(k)||leaves[k]==source,"layer inconsistent leaf");leaves[k]=source;}
  }
 }
 for(size_t i=0;i<leaves.size();++i){auto* t=leaves[i];detail::match(t,detail::spec(i,s.extent,true));check(!t->view_src,"layer leaf aliases");for(auto* source:t->src)check(!source,"layer computed leaf");
  if(i<10)s.weights[i]=t;else if(i<15)s.requests[i-10]=t;else s.caches[i-15]=t;
 }
 for(size_t i=0;i<s.nodes.size();++i){int owner=detail::node_specs[i].owner;check(s.nodes[i]->view_src==(owner==-100?nullptr:detail::ref(s,owner)),"layer view owner");}
 return s;
}
inline Weights make_weights(ggml_context* ctx,const Selection& selected){
 check(ctx,"layer weight context");Weights result{};
 for(size_t i=0;i<result.uploads.size();++i){auto* t=ggml_dup_tensor(ctx,selected.weights[i]);check(t,"layer weight allocation");ggml_set_name(t,selected.weights[i]->name);ggml_set_input(t);result.uploads[i]={selected.weights[i],t};}
 return result;
}
inline Cache make_cache(ggml_context* ctx){
 check(ctx,"layer cache context");Cache result{ggml_new_tensor_2d(ctx,GGML_TYPE_F16,256,512),ggml_new_tensor_2d(ctx,GGML_TYPE_F16,256,512)};
 check(result.k&&result.v,"layer cache allocation");ggml_set_name(result.k,"cache_k_l0");ggml_set_name(result.v,"cache_v_l0");ggml_set_input(result.k);ggml_set_input(result.v);return result;
}
// Inputs are CPU scheduling controls only. Caller synthesizes each declared step;
// it must never copy original CPU cache bytes or stale original request controls.
inline Graph materialize(ggml_context* ctx,const Selection& source,const Weights& weights,const Cache& cache,int64_t extent=0){
 check(ctx,"layer request context");Graph result{};result.extent=extent?extent:source.extent;check(result.extent==256||result.extent==512,"layer materialized extent");std::array<ggml_tensor*,17> leaves{};
 for(size_t i=0;i<10;++i){check(weights.uploads[i].original==source.weights[i],"layer changed resident weight identity");auto* t=weights.uploads[i].clone;detail::match(t,detail::spec(i,result.extent,true));check(!t->view_src,"layer resident weight alias");leaves[i]=t;}
 for(size_t i=10;i<15;++i){auto expected=detail::spec(i,result.extent,true);auto* t=ggml_new_tensor(ctx,expected.type,4,expected.ne);check(t,"layer request allocation");ggml_set_name(t,expected.name);ggml_set_input(t);leaves[i]=t;result.uploads[i-10]={source.requests[i-10],t};}
 leaves[15]=cache.k;leaves[16]=cache.v;
 for(size_t i=15;i<17;++i){detail::match(leaves[i],detail::spec(i,result.extent,true));check(!leaves[i]->view_src,"layer cache owner alias");}
 auto ref=[&](int index){return index>=0?result.nodes.at(size_t(index)):leaves.at(size_t(-1-index));};
 result.graph=ggml_new_graph_custom(ctx,64,false);check(result.graph,"layer graph allocation");size_t outputs=0;
 for(size_t i=0;i<result.nodes.size();++i){auto expected=detail::spec(i,result.extent);ggml_tensor* t;
  if(expected.owner==-100)t=ggml_new_tensor(ctx,expected.type,4,expected.ne);
  else t=ggml_view_4d(ctx,ref(expected.owner),expected.ne[0],expected.ne[1],expected.ne[2],expected.ne[3],expected.nb[1],expected.nb[2],expected.nb[3],0);
  check(t,"layer node allocation");t->op=expected.op;std::memcpy(t->nb,expected.nb,sizeof(expected.nb));std::memcpy(t->op_params,expected.params,sizeof(expected.params));ggml_set_name(t,expected.name);
  for(int j=0;j<GGML_MAX_SRC;++j)t->src[j]=(j<4&&expected.refs[j]!=-100)?ref(expected.refs[j]):nullptr;
  result.nodes[i]=t;ggml_build_forward_expand(result.graph,t);
  if(expected.owner==-100&&t->type==GGML_TYPE_F32)result.outputs.at(outputs++)=t;
 }
 check(outputs==result.outputs.size()&&ggml_graph_n_nodes(result.graph)==37,"layer materialized graph inventory");
 for(size_t i=0;i<result.nodes.size();++i)check(ggml_graph_node(result.graph,int(i))==result.nodes[i],"layer materialized cache write order");
 select(result.graph);return result;
}
} // namespace ps5hip_layer
