// Offline routing model: actual native harness, parser, eight-SGPR planner and queue.
// Host memory + synthetic GPU ranges model byte addressing, not GPU ISA/apertures.
#include "../tools/native_runtime.h"
extern "C" unsigned mock_quarantine_sleep(unsigned);
#define sleep mock_quarantine_sleep
#define main native_title_main_not_called
#include "../tools/native_canary.cpp"
#undef main
#undef sleep
#include <algorithm>
#include <cstdlib>
#include <map>
#include <set>
#include <sys/wait.h>

namespace {
#ifdef PS5HIP_CANARY_PRIVATE_CALLS_BUFFER
constexpr uint32_t model_private = 304, model_stride = 10240, model_ring = 13107200,
                   model_tmpring = 0xa500, model_outer = 16, model_inner = 160;
constexpr const char *model_profile = "private_calls_buffer";
#else
constexpr uint32_t model_private = 288, model_stride = 9216, model_ring = 11796480,
                   model_tmpring = 0x9500, model_outer = 0, model_inner = 144;
constexpr const char *model_profile = "nested_private";
#endif
struct Allocation { void *cpu; uint64_t bytes, gpu; unsigned holds = 0; };
std::map<int64_t, Allocation> allocations;
int64_t next_id = 1;
unsigned owned_close = 0, explicit_closed = 0, destructor_closed = 0;
bool in_destructor = false;
void require(bool ok) { if (!ok) throw ps5hip::ObjectError("private-call model assertion"); }
void *translate(uint64_t gpu, uint64_t bytes) {
    for (const auto &[id, a] : allocations) {
        (void)id;
        if (gpu >= a.gpu && gpu - a.gpu <= a.bytes && bytes <= a.bytes - (gpu - a.gpu))
            return static_cast<char *>(a.cpu) + gpu - a.gpu;
    }
    throw ps5hip::ObjectError("model address outside live allocation");
}
Allocation &containing(uint64_t gpu) {
    for (auto &[id, a] : allocations) { (void)id; if (gpu >= a.gpu && gpu - a.gpu < a.bytes) return a; }
    throw ps5hip::ObjectError("model allocation not found");
}
// Separate source-array evaluation; does not use the harness's closed-form oracle.
uint32_t source_result(uint32_t x, bool lose_store = false) {
    uint32_t outer[32], inner[32];
    for (unsigned j = 0; j < 32; ++j) outer[j] = x + j;
    for (unsigned j = 0; j < 32; ++j) inner[j] = outer[j] + j;
    const uint32_t updated = inner[x & 31u] + 3u;
    if (!lose_store) inner[x & 31u] = updated;
    const uint32_t leaf = updated + inner[(x + 7u) & 31u];
    return leaf + inner[x & 31u] + outer[x & 31u];
}
struct Model : ps5hip::SubmissionBackend {
    const ps5hip::CodeObject &object;
    const std::vector<uint8_t> &image;
    unsigned corruption, failure, submits = 0, polls = 0, calls = 0;
    uint64_t clock = 0;
    Model(const ps5hip::CodeObject &o, const std::vector<uint8_t> &i, unsigned c, unsigned f = 0)
        : object(o), image(i), corruption(c), failure(f) {}
    void flush(void *cpu, size_t bytes) override {
        require(std::any_of(allocations.begin(), allocations.end(), [&](const auto &pair) {
            return pair.second.cpu == cpu && bytes <= pair.second.bytes;
        }));
    }
    int32_t submit(uint32_t *words, uint32_t length) override {
        require(++submits == 1 && allocations.size() == 7);
        for (const auto &[id, a] : allocations) { (void)id; require(a.holds == 1); }
        std::map<uint32_t, uint32_t> regs;
        size_t pos = 10; unsigned user_count = 0;
        while (pos < length && ((words[pos] >> 8) & 255) == 0x76) {
            const unsigned n = (words[pos] >> 16) & 0x3fff;
            require(n && pos + n + 2 <= length);
            const uint32_t address = 0xb000 + words[pos + 1] * 4;
            if (address == 0xb900) user_count = n;
            for (unsigned j = 0; j < n; ++j) require(regs.emplace(address + j * 4, words[pos + j + 2]).second);
            pos += n + 2;
        }
        require(user_count == 8 && !regs.count(0xb920) && !regs.count(0xb924));
        require(pos + 15 == length && words[pos] == 0xc0031502 && words[pos+1] == 257 &&
                words[pos+2] == 1 && words[pos+3] == 1 && words[pos+4] == 0x8005);
        require(regs.at(0xb81c) == 64 && regs.at(0xb820) == 1 && regs.at(0xb824) == 1);
        const auto kernel = std::find_if(object.kernels.begin(), object.kernels.end(),
                                        [](const auto &k) { return k.name == "nested_private"; });
        require(kernel != object.kernels.end() && kernel->private_bytes == model_private && kernel->properties == 0x429);
        const uint64_t entry = uint64_t(regs.at(0xb830)) << 8 | uint64_t(regs.at(0xb834)) << 40;
        require(entry >= kernel->entry_address);
        const uint64_t module = entry - kernel->entry_address;
        require(std::memcmp(translate(module, image.size()), image.data(), image.size()) == 0);
        require(regs.at(0xb848) == kernel->rsrc1 && regs.at(0xb84c) == 0x91 && regs.at(0xb860) == model_tmpring);
        const uint64_t arg_address = regs.at(0xb910) | uint64_t(regs.at(0xb914)) << 32;
        const auto *args = static_cast<const uint8_t *>(translate(arg_address, 16));
        uint64_t input_address = 0, output_address = 0;
        std::memcpy(&input_address, args, 8); std::memcpy(&output_address, args + 8, 8);
        auto *input = static_cast<uint32_t *>(translate(input_address, count * 4));
        auto *output = static_cast<uint32_t *>(translate(output_address, count * 4));
        require(input != output);
        const uint64_t scratch_base = regs.at(0xb918) | uint64_t(regs.at(0xb91c)) << 32;
        require(regs.at(0xb900) == uint32_t(scratch_base) &&
                regs.at(0xb904) == (uint32_t(scratch_base >> 32) | 0x80000000u) &&
                regs.at(0xb908) == model_ring && regs.at(0xb90c) == 0x21c14fac);
        require(scratch_base == containing(scratch_base).gpu + 65536);
        auto *scratch = static_cast<uint32_t *>(translate(scratch_base, model_ring));
        if (failure == 1) return -10;
        if (failure == 4) throw ps5hip::ObjectError("injected submit exception");
        if (failure == 5) throw 7;
        for (uint32_t i = 0; i < count; ++i) {
            require(input[i] == uint32_t(uint64_t(i) * 4099 + 17) && output[i] == poison);
            const uint32_t x = input[i], lane = i % 32, wave = i / 32;
            const uint32_t cumulative = model_private; // Zero-byte leaf frame.
            require(cumulative * 32 <= model_stride && (wave + 1) * model_stride <= model_ring);
            auto private_word = [&](uint32_t byte) -> uint32_t & {
                require(byte % 4 == 0 && byte < cumulative);
                const uint32_t offset = wave * model_stride + (byte / 4) * 128 + lane * 4;
                require(offset + 4 <= model_ring);
                return scratch[offset / 4];
            };
            if (corruption == 14) { output[i] = source_result(x); calls += 2; continue; }
#ifdef PS5HIP_CANARY_PRIVATE_CALLS_BUFFER
            // ABI/WWM sentinels verify source-array writes do not alias measured save slots.
            private_word(144) = x ^ 0x12345678; private_word(148) = x ^ 0x87654321;
            private_word(288) = x ^ 0x31415926;
#endif
            for (unsigned j = 0; j < 32; ++j) private_word(model_outer + j * 4) = x + j;
            ++calls; // Retained middle call's source behavior, not ISA emulation.
            for (unsigned j = 0; j < 32; ++j) private_word(model_inner + j * 4) = private_word(model_outer + j * 4) + j;
            ++calls;
            const uint32_t updated = private_word(model_inner + (x & 31u) * 4) + 3u;
            if (corruption != 15) private_word(model_inner + (x & 31u) * 4) = updated;
            const uint32_t leaf = updated + private_word(model_inner + ((x + 7u) & 31u) * 4);
            output[i] = leaf + private_word(model_inner + (x & 31u) * 4) + private_word(model_outer + (x & 31u) * 4);
            require(output[i] == source_result(x, corruption == 15));
#ifdef PS5HIP_CANARY_PRIVATE_CALLS_BUFFER
            require(private_word(144) == (x ^ 0x12345678) && private_word(148) == (x ^ 0x87654321) &&
                    private_word(288) == (x ^ 0x31415926));
#endif
        }
        require(calls == 2 * count);
        auto &out_alloc = containing(output_address); auto &in_alloc = containing(input_address);
        auto &scratch_alloc = containing(scratch_base);
        if (corruption == 1) output[0] ^= 1;
        if (corruption == 2) output[count - 1] ^= 1;
        if (corruption == 3) input[count / 2] ^= 1;
        if (corruption == 4) output[-1] ^= 1;
        if (corruption == 5) output[count] ^= 1;
        if (corruption == 6) static_cast<uint32_t *>(out_alloc.cpu)[out_alloc.bytes / 4 - 1] ^= 1;
        if (corruption == 7) input[-1] ^= 1;
        if (corruption == 8) input[count] ^= 1;
        if (corruption == 9) static_cast<uint32_t *>(in_alloc.cpu)[in_alloc.bytes / 4 - 1] ^= 1;
        if (corruption == 10) static_cast<uint32_t *>(scratch_alloc.cpu)[0] ^= 1;
        if (corruption == 11) scratch[-1] ^= 1;
        if (corruption == 12) scratch[model_ring / 4] ^= 1;
        if (corruption == 13) static_cast<uint32_t *>(scratch_alloc.cpu)[scratch_alloc.bytes / 4 - 1] ^= 1;
        if (corruption == 16) std::swap(output[0], output[count - 1]);
        const auto *end = words + length - 8;
        require(end[-2] == 0xc0004600 && end[-1] == 0x407 && end[0] == 0xc0064900 && end[5] == 1);
        auto *marker = static_cast<uint32_t *>(translate(uint64_t(end[3]) | uint64_t(end[4]) << 32, 4));
        require(*marker == 0); if (!failure) *marker = 1;
        return 0;
    }
    int32_t kick() override { return failure == 2 ? -20 : 0; }
    uint64_t now_ns() override { return clock; }
    void sleep_us(unsigned us) override { require(failure == 3 && us == 100); ++polls; clock += uint64_t(us) * 1000; }
};
Model *quarantined_model = nullptr;
ps5hip::SubmissionQueue *quarantined_queue = nullptr;
}
namespace ps5hip {
NativeMemory::NativeMemory(uint64_t bytes, uint64_t alignment, bool code) {
    bytes_ = (bytes + 0x3fff) & ~uint64_t(0x3fff); cpu_ = std::calloc(1, bytes_);
    if (!cpu_) throw std::bad_alloc(); physical_ = next_id++;
    const uint64_t gpu = code ? 0x200000000ull + uint64_t(physical_) * 65536 : 0x4000000000ull + uint64_t(physical_) * 0x2000000;
    require(gpu % alignment == 0);
    allocations.emplace(physical_, Allocation{cpu_, bytes_, gpu});
}
NativeMemory::~NativeMemory() { in_destructor = true; if (!close()) std::abort(); in_destructor = false; }
GpuRange NativeMemory::range() const { require(cpu_); return {allocations.at(physical_).gpu, bytes_}; }
void NativeMemory::acquire_gpu_use() { ++gpu_uses_; ++allocations.at(physical_).holds; }
void NativeMemory::complete_gpu_use() { require(gpu_uses_); --gpu_uses_; --allocations.at(physical_).holds; }
bool NativeMemory::close() {
    if (gpu_uses_) { ++owned_close; return false; }
    if (cpu_) { if (in_destructor) ++destructor_closed; else ++explicit_closed;
        std::free(cpu_); allocations.erase(physical_); cpu_ = nullptr; }
    return true;
}
}
extern "C" int32_t sceAgcInit(uint32_t) { std::abort(); }
extern "C" int32_t sceAgcDriverSubmitDcb(void *) { std::abort(); }
extern "C" int32_t sceAgcSuspendPoint() { std::abort(); }
extern "C" unsigned mock_quarantine_sleep(unsigned seconds) {
    bool ok = seconds == 1 && quarantined_model && quarantined_queue && quarantined_queue->poisoned() &&
              quarantined_model->submits == 1 && allocations.size() == 7 && owned_close == 0 &&
              explicit_closed == 0 && destructor_closed == 0 &&
              quarantined_model->polls == (quarantined_model->failure == 3 ? 20000u : 0u);
    for (const auto &[id, a] : allocations) { (void)id; ok = ok && a.holds == 1; }
    ::_exit(ok ? 0 : 7);
}
int main(int argc, char **argv) {
    try {
        require(argc == 2); std::ifstream file(argv[1], std::ios::binary); require(bool(file));
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
        const auto object = ps5hip::inspect_code_object(bytes.data(), bytes.size());
        const auto image = ps5hip::materialize_code_object(bytes.data(), bytes.size());
        require(object.target == PS5HIP_CANARY_TARGET && count == 16448 && scratch_private_bytes == model_private);
#ifdef PS5HIP_CANARY_PRIVATE_CALLS_BUFFER
        // The historical 288-byte metadata cannot silently select this fixed profile.
        auto historical = object;
        for (auto &kernel : historical.kernels) if (kernel.name == "nested_private") kernel.private_bytes = 288;
        Model old_model(historical, image, 0); ps5hip::SubmissionQueue old_queue(old_model);
        bool old_rejected = false;
        try { run_guarded_experiment(old_queue, historical, image); }
        catch (const ps5hip::ObjectError &) { old_rejected = true; }
        require(old_rejected && !old_queue.poisoned() && old_model.submits == 0 && allocations.empty() && owned_close == 0);
#endif
        for (uint32_t base : {0u, UINT32_MAX - 63u, 0x7fffffe0u})
            for (unsigned j = 0; j < 32; ++j) {
                const uint32_t x = base + j;
                require(source_result(x) == private_expected(x) && source_result(x) - source_result(x, true) == 3);
            }
        std::set<uint32_t> inputs, outputs;
        for (uint32_t i = 0; i < count; ++i) { inputs.insert(private_input(i)); outputs.insert(source_result(private_input(i))); }
        require(inputs.size() == count && outputs.size() == count);
        for (unsigned corruption = 0; corruption <= 16; ++corruption) {
            explicit_closed = destructor_closed = 0;
            Model model(object, image, corruption); ps5hip::SubmissionQueue queue(model);
            FILE *log = std::tmpfile(); require(log); std::fflush(nullptr);
            const int saved_out = ::dup(STDOUT_FILENO), saved_err = ::dup(STDERR_FILENO);
            require(saved_out >= 0 && saved_err >= 0 && ::dup2(::fileno(log), STDOUT_FILENO) >= 0 && ::dup2(::fileno(log), STDERR_FILENO) >= 0);
            bool rejected = false;
            try { run_guarded_experiment(queue, object, image); } catch (const ps5hip::ObjectError &) { rejected = true; }
            std::fflush(nullptr); require(::dup2(saved_out, STDOUT_FILENO) >= 0 && ::dup2(saved_err, STDERR_FILENO) >= 0);
            ::close(saved_out); ::close(saved_err); std::rewind(log);
            char text[2048]{}; const auto n = std::fread(text, 1, sizeof(text), log); std::fclose(log);
            const std::string logged(text, n);
            require(logged.find(std::string("SUBMIT target=") + PS5HIP_CANARY_TARGET + " kernel=" + model_profile +
                    " invocations=16448 groups=257 user_sgprs=8 private_bytes=" + std::to_string(model_private) +
                    " scratch_bytes=" + std::to_string(model_ring) + " timeout_ms=2000") != std::string::npos);
            require(rejected == (corruption != 0) && !queue.poisoned() && allocations.empty() && owned_close == 0);
            require(model.submits == 1 && model.calls == 2 * count && model.polls == 0);
            require(explicit_closed == (rejected ? 2u : 7u) && destructor_closed == (rejected ? 5u : 0u));
            require((logged.find(std::string("PASS kernel=") + model_profile + " exact_values=16448 guards=unchanged allocations=closed") != std::string::npos) == !rejected);
            if (!rejected) require(logged.find("SCRATCH payload=modified guards=unchanged") < logged.find(std::string("PASS kernel=") + model_profile));
        }
        for (unsigned failure = 1; failure <= 5; ++failure) {
            FILE *log = std::tmpfile(); require(log); std::fflush(nullptr);
            const auto child = ::fork();
            if (!child) {
                explicit_closed = destructor_closed = 0;
                if (::dup2(::fileno(log), STDOUT_FILENO) < 0 || ::dup2(::fileno(log), STDERR_FILENO) < 0) ::_exit(2);
                Model model(object, image, 0, failure); ps5hip::SubmissionQueue queue(model);
                quarantined_model = &model; quarantined_queue = &queue; run_guarded_experiment(queue, object, image); ::_exit(3);
            }
            int status = 0; require(child > 0 && ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
            std::rewind(log); char text[2048]{}; const auto n = std::fread(text, 1, sizeof(text), log); std::fclose(log);
            const std::string logged(text, n);
            require((failure <= 3 ? logged.find("PS5HIP submission status=" + std::to_string(failure)) != std::string::npos :
                     logged.find(failure == 4 ? "injected submit exception" : "unknown exception") != std::string::npos) &&
                    logged.find("PS5HIP QUARANTINED: GPU ownership unresolved") != std::string::npos &&
                    logged.find("PS5HIP PASS") == std::string::npos && allocations.empty());
        }
        std::puts("private-call host model: 8-slot routing, two-call source oracle, swizzled cumulative frames, full guards, closure and quarantine passed; no GPU execution");
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
