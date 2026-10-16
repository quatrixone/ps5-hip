#include "llama.h"
#include "ggml.h"
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


struct Trace { std::ofstream out; int position = -1; const char * phase = "init"; size_t nodes = 0; size_t completed = 0; };
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
    if (state.position > 3 && state.position != 31 && state.position != 32 && state.position != 255 && state.position != 256 && state.position != 510 && state.position != 511) return true;
    auto & o = state.out;
    o << "{\"position\":" << state.position << ",\"phase\":"; quoted(o,state.phase);
    o << ",\"node\":" << state.nodes++ << ",\"tensor\":";tensor(o,t);
    o << ",\"sources\":[";
    bool first = true;
    for (unsigned i=0;i<GGML_MAX_SRC;++i) if(t->src[i]) { if(!first)o<<',';first=false; o << "{\"index\":" << i << ",\"tensor\":";tensor(o,t->src[i]);o<<'}'; }
    o << "]}\n";
    return o.good();
}

static int run(int argc, char ** argv) {
    require(argc == 3, "usage: graph model output-prefix");
    Trace state;
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
    const llama_token prompt[] = {1, 9038, 2501, 263};
    int next = 0;
    for (int position = 0; position < 512; ++position) {
        state.position = position; state.phase = position < 4 ? "prefill" : "decode";
        const int token = position < 4 ? prompt[position] : next;
        const auto before = state.completed;
        llama_batch_ext_clear(batch.get());
        const int index = llama_batch_ext_add_token(batch.get(), 0, token);
        const llama_pos pos = position;
        require(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &pos)
                && llama_batch_ext_set_output_logits(batch.get(), index, true), "single-token batch failed");
        require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0 && state.out.good(), "decode failed");
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
    std::cout << "EXECUTED_CPU_GRAPH positions=512 batch=1 gpu_execution=false\n";
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
