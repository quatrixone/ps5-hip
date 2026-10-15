#include "llama.h"
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

static int run(int argc, char ** argv) {
    require(argc == 6, "usage: fixture tokenize|decode model input output steps");
    const bool tokenize = std::string(argv[1]) == "tokenize";
    require(tokenize || std::string(argv[1]) == "decode", "unknown mode");
    size_t consumed;
    int steps = std::stoi(argv[5], &consumed);
    require(consumed == std::string(argv[5]).size() && steps >= 0 && steps <= 64, "invalid decode step count");
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.vocab_only = tokenize;
    mp.check_tensors = !tokenize;
    mp.use_extra_bufts = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[2], mp), llama_model_free);
    require(model != nullptr, "model load failed");
    std::cout << llama_print_system_info() << '\n';
    const auto * vocab = llama_model_get_vocab(model.get());
    require(llama_vocab_n_tokens(vocab) == 32000, "unexpected vocabulary size");
    std::ifstream input(argv[3]);
    require(input.good(), "cannot open input");
    std::vector<llama_token> tokens;
    if (tokenize) {
        std::string text((std::istreambuf_iterator<char>(input)), {});
        require(text.size() < 16384, "prompt too large");
        const int count = -llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), nullptr, 0, true, true);
        require(count > 0 && count <= 512, "invalid prompt token count");
        tokens.resize(count);
        require(llama_tokenize(vocab, text.data(), static_cast<int>(text.size()), tokens.data(), count, true, true) == count,
                "tokenization changed size");
        std::ofstream output(argv[4]);
        for (auto token : tokens) output << token << '\n';
        output.close();
        require(output.good(), "token output failed");
        return 0;
    }
    std::string token_text;
    while (input >> token_text) {
        const auto token = std::stoll(token_text, &consumed);
        require(consumed == token_text.size(), "malformed input token");
        require(token >= 0 && token < 32000 && tokens.size() < 512, "invalid input token");
        tokens.push_back(static_cast<llama_token>(token));
    }
    require(input.eof() && !tokens.empty() && tokens.size() + steps <= 512, "invalid input/context size");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.no_perf = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    require(ctx != nullptr && llama_n_ctx(ctx.get()) == 512, "unexpected context allocation");
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
    require(batch != nullptr, "batch allocation failed");
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int index = llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
        const llama_pos pos = static_cast<llama_pos>(i);
        require(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &pos), "prefill batch failed");
    }
    require(llama_batch_ext_set_output_logits(batch.get(), static_cast<int>(tokens.size()) - 1, true), "logits selection failed");
    require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "prefill failed");
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "requires IEEE float32");
    const uint16_t endian = 1;
    require(*reinterpret_cast<const uint8_t *>(&endian) == 1, "requires little endian host");
    std::ofstream logits(std::string(argv[4]) + ".f32le", std::ios::binary);
    std::ofstream generated(std::string(argv[4]) + ".tokens");
    require(logits.good() && generated.good(), "cannot open output");
    for (int step = 0; step <= steps; ++step) {
        const float * row = llama_get_logits_ith(ctx.get(), -1);
        require(row != nullptr, "missing logits");
        int best = 0;
        for (int i = 0; i < 32000; ++i) {
            require(std::isfinite(row[i]), "non-finite logit");
            if (row[i] > row[best]) best = i;
        }
        logits.write(reinterpret_cast<const char *>(row), 32000 * sizeof(float));
        require(logits.good(), "logits write failed");
        if (step == steps) break;
        // Fixed-step numerical fixtures deliberately continue through end-of-generation tokens.
        generated << best << '\n';
        llama_batch_ext_clear(batch.get());
        const int index = llama_batch_ext_add_token(batch.get(), 0, best);
        const llama_pos pos = static_cast<llama_pos>(tokens.size()) + step;
        require(index >= 0 && llama_batch_ext_set_pos(batch.get(), index, &pos)
                && llama_batch_ext_set_output_logits(batch.get(), index, true), "decode batch failed");
        require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decode failed");
    }
    logits.close();
    generated.close();
    require(logits.good() && generated.good(), "output close failed");
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
