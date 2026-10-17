#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "model_layer.h"
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}


struct Trace { std::ofstream out; int position = -1; const char * phase = "init"; size_t nodes = 0; size_t completed = 0; bool layer_done=true; std::string prefix; std::vector<ggml_tensor*> original;
 ggml_context* resident=nullptr;ggml_backend_t backend=nullptr;ggml_backend_buffer_t storage=nullptr;
 ps5hip_layer::Weights weights{};ps5hip_layer::Cache cache{};
 ~Trace(){if(storage)ggml_backend_buffer_free(storage);if(backend)ggml_backend_free(backend);if(resident)ggml_free(resident);}
 };
static void quoted(std::ostream & o, const char * s) {
    o << '"';
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(s); *p; ++p) {
        if (*p == '"' || *p == '\\') o << '\\' << *p;
        else if (*p < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(*p) << std::dec;
        else o << *p;
    }
    o << '"';
}
static void tensor(std::ostream & o, const ggml_tensor * t) {
    o << "{\"name\":"; quoted(o, t->name);
    o << ",\"dtype\":"; quoted(o, ggml_type_name(t->type));
    o << ",\"op\":"; quoted(o, ggml_op_name(t->op));
    o << ",\"description\":"; quoted(o, ggml_op_desc(t));
    o << ",\"ne\":["; for (int i=0;i<4;++i) { if(i)o<<',';o<<t->ne[i]; }
    o << "],\"nb\":["; for (int i=0;i<4;++i) { if(i)o<<',';o<<t->nb[i]; }
    o << "],\"view_offset\":" << t->view_offs << ",\"view_source\":";
    if (t->view_src) quoted(o,t->view_src->name); else o << "null";
    o << ",\"op_params_hex\":\"";
    const auto * bytes = reinterpret_cast<const unsigned char *>(t->op_params);
    for (size_t i=0;i<sizeof(t->op_params);++i) o << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
    o << std::dec << "\"}";
}
static bool trace(ggml_tensor * t, bool ask, void * user) {
    if (ask) return true;
    auto & state = *static_cast<Trace *>(user);
    ++state.completed;
    if(state.layer_done) return true;
    if (state.position > 3 && state.position != 31 && state.position != 32 && state.position != 255 && state.position != 256 && state.position != 510 && state.position != 511) return true;
    state.original.push_back(t);
    auto & o = state.out;
    o << "{\"position\":" << state.position << ",\"phase\":"; quoted(o,state.phase);
    o << ",\"node\":" << state.nodes++ << ",\"tensor\":";tensor(o,t);
    o << ",\"sources\":[";
    bool first = true;
    for (unsigned i=0;i<GGML_MAX_SRC;++i) if(t->src[i]) { if(!first)o<<',';first=false; o << "{\"index\":" << i << ",\"tensor\":";tensor(o,t->src[i]);o<<'}'; }
    o << "]}\n";
    const auto save=[&](const ggml_tensor* value,const std::string& suffix){
        std::vector<char> raw(ggml_nbytes(value));ggml_backend_tensor_get(value,raw.data(),0,raw.size());
        std::ofstream file(state.prefix+".p"+std::to_string(state.position)+".n"+std::to_string(state.nodes-1)+suffix+".bin",std::ios::binary);
        file.write(raw.data(),raw.size());file.close();require(file.good(),"tensor capture failed");
    };
    if(t->type==GGML_TYPE_F32 && !t->view_src && std::strcmp(t->name,"node_1") && std::strcmp(t->name,"node_2"))save(t,".out");
    for(int i=0;i<GGML_MAX_SRC;++i)if(t->src[i] && (std::strcmp(t->src[i]->name,"inp_pos")==0 || std::strncmp(t->src[i]->name,"attn_inp_",9)==0))save(t->src[i],".src"+std::to_string(i));
    if(std::strcmp(t->name,"l_out-0")==0){
      state.layer_done=true;
    }
    return o.good();
}

static void check_layer(Trace& state){
    auto* context=ggml_init({2*1024*1024,nullptr,true});require(context,"clone context");
    auto* original=ggml_new_graph_custom(context,128,false);for(auto* node:state.original)ggml_graph_add_node(original,node);
    auto selected=ps5hip_layer::select(original);
    if(!state.resident){
     state.resident=ggml_init({2*1024*1024,nullptr,true});state.weights=ps5hip_layer::make_weights(state.resident,selected);state.cache=ps5hip_layer::make_cache(state.resident);
     state.backend=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(state.backend,4);state.storage=ggml_backend_alloc_ctx_tensors(state.resident,state.backend);require(state.storage,"resident storage");
     for(auto pair:state.weights.uploads){std::vector<char> data(ggml_nbytes(pair.original));ggml_backend_tensor_get(pair.original,data.data(),0,data.size());ggml_backend_tensor_set(pair.clone,data.data(),0,data.size());}
     std::vector<char> zeros(262144);ggml_backend_tensor_set(state.cache.k,zeros.data(),0,zeros.size());ggml_backend_tensor_set(state.cache.v,zeros.data(),0,zeros.size());
    }
    auto cloned=ps5hip_layer::materialize(context,selected,state.weights,state.cache);
    auto buffer=ggml_backend_alloc_ctx_tensors(context,state.backend);require(buffer,"request storage");
    for(auto pair:cloned.uploads){std::vector<char> data(ggml_nbytes(pair.original));ggml_backend_tensor_get(pair.original,data.data(),0,data.size());ggml_backend_tensor_set(pair.clone,data.data(),0,data.size());}
    require(ggml_backend_graph_compute(state.backend,cloned.graph)==GGML_STATUS_SUCCESS,"cloned actual layer compute");
    for(size_t index=0;index<cloned.nodes.size();++index){
      auto* output=cloned.nodes[index];if(output->view_src||output->type!=GGML_TYPE_F32)continue;
      auto found=std::find(state.original.begin(),state.original.end(),selected.nodes[index]);require(found!=state.original.end(),"actual node match");
      const auto raw_path=state.prefix+".p"+std::to_string(state.position)+".n"+std::to_string(found-state.original.begin())+".out.bin";
      std::ifstream file(raw_path,std::ios::binary);std::vector<char> expected((std::istreambuf_iterator<char>(file)),std::istreambuf_iterator<char>()),actual(ggml_nbytes(output));
      ggml_backend_tensor_get(output,actual.data(),0,actual.size());
      if(actual!=expected){std::cerr<<"mismatch index="<<index<<" name="<<output->name<<" actual="<<actual.size()<<" expected="<<expected.size()<<"\n";}
      require(actual==expected,"cloned/original captured intermediate differs");
      std::ofstream captured(state.prefix+".clone-p"+std::to_string(state.position)+"-n"+std::to_string(index)+".bin",std::ios::binary);captured.write(actual.data(),actual.size());captured.close();require(captured.good(),"clone capture");
    }
    if(state.position==0){
     const auto rejects=[&](auto action){bool rejected=false;try{action();}catch(const std::runtime_error&){rejected=true;}require(rejected,"malformed layer accepted");};
     auto* tail=cloned.nodes[36];auto* saved=tail->src[1];tail->src[1]=cloned.nodes[0];rejects([&]{ps5hip_layer::select(cloned.graph);});tail->src[1]=saved;
     auto* glu=cloned.nodes[34];int32_t param=glu->op_params[0];glu->op_params[0]=0;rejects([&]{ps5hip_layer::select(cloned.graph);});glu->op_params[0]=param;
     auto* view=cloned.nodes[17];auto* owner=view->view_src;view->view_src=state.cache.k;rejects([&]{ps5hip_layer::select(cloned.graph);});view->view_src=owner;
     auto stride=cloned.nodes[23]->nb[1];cloned.nodes[23]->nb[1]+=4;rejects([&]{ps5hip_layer::select(cloned.graph);});cloned.nodes[23]->nb[1]=stride;
     auto* missing=ggml_new_graph_custom(context,64,false);for(size_t i=0;i<37;++i)if(i!=12)ggml_graph_add_node(missing,cloned.nodes[i]);rejects([&]{ps5hip_layer::select(missing);});
     auto* reordered=ggml_new_graph_custom(context,64,false);for(size_t i=0;i<37;++i)ggml_graph_add_node(reordered,cloned.nodes[i==12?23:i==23?12:i]);rejects([&]{ps5hip_layer::select(reordered);});
     rejects([&]{ps5hip_layer::materialize(context,selected,state.weights,state.cache,513);});
     auto invalid_cache=state.cache;invalid_cache.v=invalid_cache.k;rejects([&]{ps5hip_layer::materialize(context,selected,state.weights,invalid_cache);});
     auto changed=selected;changed.weights[0]=selected.weights[1];rejects([&]{ps5hip_layer::materialize(context,changed,state.weights,state.cache);});
     std::cout<<"LAYER_REJECTIONS_PASS cases=9 GPU_execution=false\n";
    }
    auto* large_ctx=ggml_init({2*1024*1024,nullptr,true});auto large=ps5hip_layer::materialize(large_ctx,selected,state.weights,state.cache,512);require(large.nodes[23]->ne[0]==512&&large.uploads[4].clone->ne[0]==512,"large extent");ggml_free(large_ctx);
    std::cout<<"ACTUAL_LAYER_CLONE_PASS position="<<state.position<<" nodes=37 outputs=21 persistent_KV=true GPU_execution=false\n";
    ggml_backend_buffer_free(buffer);ggml_free(context);
}

static int run(int argc, char ** argv) {
    require(argc == 3, "usage: graph model output-prefix");
    Trace state;state.prefix=argv[2];
    state.out.open(std::string(argv[2]) + ".trace.jsonl");
    std::ofstream steps(std::string(argv[2]) + ".steps.jsonl");
    std::ofstream logits(std::string(argv[2]) + ".f32le", std::ios::binary);
    require(state.out.good() && steps.good() && logits.good(), "output open failed");
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.check_tensors = true;
    mp.use_extra_bufts = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(model != nullptr, "model load failed");
    require(llama_vocab_n_tokens(llama_model_get_vocab(model.get())) == 32000, "vocabulary mismatch");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = false;
    cp.no_perf = true;
    cp.cb_eval = trace; cp.cb_eval_user_data = &state;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    require(ctx && llama_n_ctx(ctx.get()) == 512 && llama_n_batch(ctx.get()) == 1 && llama_n_ubatch(ctx.get()) == 1,
            "context/batch profile mismatch");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    require(batch != nullptr, "batch allocation failed");
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE float32");
    const uint16_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "requires little endian host");
    const llama_token prompt[] = {0, 1, 2, 31999};
    int next = 0;
    for (int position = 0; position < 4; ++position) {
        state.position = position; state.layer_done=false; state.nodes=0; state.original.clear(); state.phase = position < 4 ? "prefill" : "decode";
        const int token = position < 4 ? prompt[position] : next;
        const auto before = state.completed;
        llama_batch_ext_clear(batch.get());
        const int index = llama_batch_ext_add_token(batch.get(), 0, token);
        const llama_pos pos = position;
        require(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &pos)
                && llama_batch_ext_set_output_logits(batch.get(), index, true), "single-token batch failed");
        require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0 && state.out.good(), "decode failed");
        check_layer(state);
        const float * row = llama_get_logits_ith(ctx.get(), -1);
        require(row != nullptr && state.completed > before, "missing logits or executed graph");
        next = 0;
        for (int i = 0; i < 32000; ++i) {
            require(std::isfinite(row[i]), "non-finite logit");
            if (row[i] > row[next]) next = i;
        }
        logits.write(reinterpret_cast<const char *>(row), 32000*sizeof(float));
        steps << "{\"position\":" << position << ",\"phase\":\"" << state.phase
              << "\",\"input_token\":" << token << ",\"next_token\":" << next
              << ",\"completed_nodes\":" << state.completed-before << "}\n";
        require(logits.good() && steps.good(), "step output failed");
    }
    state.out.close(); steps.close(); logits.close();
    require(state.out.good() && steps.good() && logits.good(), "output close failed");
    std::cout << "EXECUTED_CPU_GRAPH positions=4 first_layer_capture=true batch=1 gpu_execution=false\n";
    return 0;
}
int main(int argc, char ** argv) {
    llama_backend_init();
    int result;
    try { result = run(argc, argv); }
    catch (const std::exception & error) { std::cerr << error.what() << '\n'; result = 1; }
    llama_backend_free();
    return result;
}
