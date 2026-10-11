// Host-only routing model: actual harness, parser, launch encoder and queue.
// Synthetic GPU addresses translate to ASan-covered host storage. No GPU or
// native allocator behavior is emulated as hardware evidence.
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
struct Allocation { void *cpu; uint64_t bytes; bool code; unsigned gpu_uses = 0; };
std::map<int64_t, Allocation> allocations;
int64_t next_id = 1;
unsigned attempted_owned_close = 0;
void require(bool value) { if (!value) throw ps5hip::ObjectError("suite model assertion"); }
uint64_t address(int64_t id) { return 0x200000000ull + uint64_t(id) * 0x10000; }
void *translate(uint64_t gpu, size_t bytes) {
    for (const auto &[id, allocation] : allocations)
        if (gpu >= address(id) && gpu - address(id) <= allocation.bytes &&
            bytes <= allocation.bytes - (gpu - address(id)))
            return static_cast<char *>(allocation.cpu) + gpu - address(id);
    throw ps5hip::ObjectError("model address outside live allocation");
}
struct Model : ps5hip::SubmissionBackend {
    const ps5hip::CodeObject &object;
    const std::vector<uint8_t> &image;
    unsigned submits = 0, corrupt_dispatch, corruption, failure, polls = 0;
    uint64_t clock = 0;
    bool first_released = false;
    std::set<int64_t> previous;
    Model(const ps5hip::CodeObject &o, const std::vector<uint8_t> &i, unsigned d, unsigned c, unsigned f = 0)
        : object(o), image(i), corrupt_dispatch(d), corruption(c), failure(f) {}
    void flush(void *, size_t) override {}
    int32_t submit(uint32_t *words, uint32_t length) override {
        ++submits;
        require(submits <= 2 && allocations.size() == (submits == 1 ? 5 : 7));
        for (const auto &[id, allocation] : allocations) {
            require(!previous.count(id));
            (void)allocation;
        }
        if (submits == 2) first_released = true;
        previous.clear();
        for (const auto &[id, allocation] : allocations) { previous.insert(id); (void)allocation; }
        std::map<uint32_t, uint32_t> registers;
        size_t pos = 10;
        while (pos < length && ((words[pos] >> 8) & 255) == 0x76) {
            const unsigned count = (words[pos] >> 16) & 0x3fff;
            require(count && pos + count + 2 <= length);
            for (unsigned i = 0; i < count; ++i)
                registers[0xb000 + words[pos + 1] * 4 + i * 4] = words[pos + 2 + i];
            pos += count + 2;
        }
        require(pos + 15 == length && words[pos] == 0xc0031502 && words[pos+1] == 2 &&
                words[pos+2] == 1 && words[pos+3] == 1 && words[pos+4] == 0x8005);
        require(registers.at(0xb81c) == 64 && registers.at(0xb820) == 1 && registers.at(0xb824) == 1);
        const char *name = submits == 1 ? "indexed_store" : "vector_add";
        const auto kernel = std::find_if(object.kernels.begin(), object.kernels.end(),
                                        [&](const auto &k) { return k.name == name; });
        require(kernel != object.kernels.end());
        const uint64_t entry = uint64_t(registers.at(0xb830)) << 8 | uint64_t(registers.at(0xb834)) << 40;
        require(entry >= kernel->entry_address);
        const uint64_t module = entry - kernel->entry_address;
        require(std::memcmp(translate(module, image.size()), image.data(), image.size()) == 0);
        require(registers.at(0xb848) == kernel->rsrc1 && registers.at(0xb84c) == kernel->rsrc2);
        const uint32_t base = (kernel->properties & 1) ? 0xb910 : 0xb900;
        const uint64_t arg_address = registers.at(base) | uint64_t(registers.at(base + 4)) << 32;
        const auto *args = static_cast<const uint8_t *>(translate(arg_address, kernel->kernarg_size));
        auto argument_value = [&](size_t index) {
            uint64_t value = 0;
            const auto &layout = kernel->arguments.at(index);
            std::memcpy(&value, args + layout.offset, layout.size);
            return value;
        };
        require(argument_value(submits == 1 ? 1 : 3) == 73);
        const auto out_address = argument_value(submits == 1 ? 0 : 2);
        auto *output = static_cast<uint32_t *>(translate(out_address, 73 * 4));
        uint32_t *a = nullptr, *b = nullptr;
        if (submits == 2) {
            a = static_cast<uint32_t *>(translate(argument_value(0), 73 * 4));
            b = static_cast<uint32_t *>(translate(argument_value(1), 73 * 4));
            require(a != b && a != output && b != output);
        }
        if (submits == 2 && failure == 1) return -10;
        for (uint32_t i = 0; i < 73; ++i) {
            require(output[i] == poison);
            if (submits == 1) output[i] = i ^ 0x5a17c9e3;
            else {
                require(a[i] == bits(float(i) * 0.5f) && b[i] == bits(2.0f - float(i) * 0.25f));
                float av, bv;
                std::memcpy(&av, &a[i], 4); std::memcpy(&bv, &b[i], 4);
                output[i] = bits(av + bv);
            }
        }
        if (submits == corrupt_dispatch) {
            if (corruption == 0) output[72] ^= 1;
            if (corruption == 1) output[-1] ^= 1;
            if (corruption == 2) output[73] ^= 1;
            if (corruption == 3) a[37] ^= 1;
            if (corruption == 4) b[72] ^= 1;
            if (corruption == 5) a[-1] ^= 1;
            if (corruption == 6) b[73] ^= 1;
        }
        const auto *end = words + length - 8;
        require(end[-2] == 0xc0004600 && end[-1] == 0x407 && end[0] == 0xc0064900 && end[5] == 1);
        auto *marker = static_cast<uint32_t *>(translate(uint64_t(end[3]) | uint64_t(end[4]) << 32, 4));
        require(*marker == 0);
        if (submits != 2 || !failure) *marker = 1;
        return 0;
    }
    int32_t kick() override { return submits == 2 && failure == 2 ? -20 : 0; }
    uint64_t now_ns() override { return clock; }
    void sleep_us(unsigned us) override {
        require(submits == 2 && failure == 3 && us == 100);
        ++polls; clock += uint64_t(us) * 1000;
    }
};
Model *quarantined_model = nullptr;
ps5hip::SubmissionQueue *quarantined_queue = nullptr;
}
namespace ps5hip {
NativeMemory::NativeMemory(uint64_t bytes, uint64_t, bool code) {
    bytes_ = (bytes + 0x3fff) & ~uint64_t(0x3fff);
    cpu_ = std::calloc(1, bytes_);
    if (!cpu_) throw std::bad_alloc();
    physical_ = next_id++;
    allocations.emplace(physical_, Allocation{cpu_, bytes_, code});
}
NativeMemory::~NativeMemory() { if (!close()) std::abort(); }
GpuRange NativeMemory::range() const { require(cpu_ != nullptr); return {address(physical_), bytes_}; }
void NativeMemory::acquire_gpu_use() { ++gpu_uses_; ++allocations.at(physical_).gpu_uses; }
void NativeMemory::complete_gpu_use() { require(gpu_uses_ != 0); --gpu_uses_; --allocations.at(physical_).gpu_uses; }
bool NativeMemory::close() {
    if (gpu_uses_) { ++attempted_owned_close; return false; }
    if (cpu_) { std::free(cpu_); allocations.erase(physical_); cpu_ = nullptr; }
    return true;
}
}
extern "C" int32_t sceAgcInit(uint32_t) { std::abort(); }
extern "C" int32_t sceAgcDriverSubmitDcb(void *) { std::abort(); }
extern "C" int32_t sceAgcSuspendPoint() { std::abort(); }

// Stop only the forked test process at quarantine's first sleep. Verify retained
// ownership before _exit; normal destruction would incorrectly free or abort it.
extern "C" unsigned mock_quarantine_sleep(unsigned seconds) {
    bool ok = seconds == 1 && quarantined_model && quarantined_queue && quarantined_queue->poisoned() &&
              quarantined_model->submits == 2 && quarantined_model->first_released &&
              allocations.size() == 7 && attempted_owned_close == 0 &&
              quarantined_model->polls == (quarantined_model->failure == 3 ? 20000u : 0u);
    for (const auto &[id, allocation] : allocations)
        ok = ok && allocation.gpu_uses == 1 && quarantined_model->previous.count(id);
    ::_exit(ok ? 0 : 7);
}

int main(int argc, char **argv) {
    try {
        require(argc == 2);
        std::ifstream file(argv[1], std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
        const auto object = ps5hip::inspect_code_object(bytes.data(), bytes.size());
        const auto image = ps5hip::materialize_code_object(bytes.data(), bytes.size());
        for (unsigned bad_dispatch = 0; bad_dispatch <= 2; ++bad_dispatch) {
            for (unsigned fault = 0; fault < (bad_dispatch == 0 ? 1u : bad_dispatch == 1 ? 3u : 7u); ++fault) {
                Model model(object, image, bad_dispatch, fault);
                ps5hip::SubmissionQueue queue(model);
                FILE *log = std::tmpfile(); require(log != nullptr);
                std::fflush(stdout);
                const int saved = ::dup(STDOUT_FILENO); require(saved >= 0);
                require(::dup2(::fileno(log), STDOUT_FILENO) >= 0);
                bool rejected = false;
                try { run_experiment(queue, object, image); }
                catch (const ps5hip::ObjectError &) { rejected = true; }
                std::fflush(stdout); require(::dup2(saved, STDOUT_FILENO) >= 0); ::close(saved);
                std::rewind(log);
                char text[2048]{}; const size_t length = std::fread(text, 1, sizeof(text), log); std::fclose(log);
                const std::string output(text, length);
                require(rejected == (bad_dispatch != 0) && !queue.poisoned() && allocations.empty());
                require(model.submits == (bad_dispatch == 1 ? 1 : 2));
                require((output.find("PASS kernel=arithmetic_suite dispatches=2 exact_values=146") != std::string::npos) == !rejected);
                require((output.find("DISPATCH index=2/2 kernel=vector_add") != std::string::npos) == (bad_dispatch != 1));
            }
        }
        for (unsigned failure = 1; failure <= 3; ++failure) {
            FILE *log = std::tmpfile(); require(log != nullptr);
            std::fflush(nullptr);
            const pid_t child = ::fork();
            if (child == 0) {
                if (::dup2(::fileno(log), STDOUT_FILENO) < 0 || ::dup2(::fileno(log), STDERR_FILENO) < 0)
                    ::_exit(2);
                Model model(object, image, 0, 0, failure);
                ps5hip::SubmissionQueue queue(model);
                quarantined_model = &model; quarantined_queue = &queue;
                run_experiment(queue, object, image);
                ::_exit(3); // A failed second submission must never return or pass.
            }
            int status = 0;
            require(child > 0 && ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
            std::rewind(log);
            char text[2048]{}; const size_t length = std::fread(text, 1, sizeof(text), log); std::fclose(log);
            const std::string output(text, length);
            const auto first = output.find("PASS kernel=indexed_store exact_values=73 guards=unchanged allocations=closed");
            const auto second = output.find("DISPATCH index=2/2 kernel=vector_add");
            require(first != std::string::npos && second != std::string::npos && first < second);
            require(output.find("PS5HIP submission status=" + std::to_string(failure)) != std::string::npos);
            require(output.find("PS5HIP QUARANTINED: GPU ownership unresolved; no retry or local teardown") != std::string::npos);
            require(output.find("PASS kernel=vector_add") == std::string::npos && output.find("PASS kernel=arithmetic_suite") == std::string::npos);
            require(allocations.empty());
        }
        std::puts("arithmetic suite host model: real fixture routing, same queue, fresh allocations, exact oracles and rejection passed; no GPU evidence");
    } catch (const std::exception &e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
