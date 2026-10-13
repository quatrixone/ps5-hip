// Ordinary CUDA source: HIPIFY generates the HIP input without hand edits.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

extern "C" __global__ __launch_bounds__(64)
void translated_add(const unsigned *a, const unsigned *b, unsigned *c, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool value) {
    if (!value) throw std::runtime_error("translated vector result mismatch");
}
}

int main() {
    void *da = nullptr, *db = nullptr, *dc = nullptr;
    try {
        for (unsigned n : {1u, 73u, 257u}) {
            constexpr unsigned guard = 7, poison = 0xdeadbeef;
            const size_t words = n + 2 * guard, bytes = words * sizeof(unsigned);
            std::vector<unsigned> a(words, poison), b(words, poison), expected(words, poison);
            std::vector<unsigned> observed(words), after_a(words), after_b(words);
            check(cudaMalloc(&da, bytes)); check(cudaMalloc(&db, bytes)); check(cudaMalloc(&dc, bytes));
            for (unsigned round = 0; round < 2; ++round) {
                for (unsigned i = 0; i < n; ++i) {
                    a[guard+i] = UINT32_MAX - i * 17u - round * 101u;
                    b[guard+i] = i * 31u + round * 701u + 9u;
                    expected[guard+i] = a[guard+i] + b[guard+i];
                }
                std::vector<unsigned> initial(words, poison);
                check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
                check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
                check(cudaMemcpy(dc, initial.data(), bytes, cudaMemcpyHostToDevice));
                translated_add<<<(n+63)/64, 64>>>(static_cast<unsigned*>(da)+guard,
                    static_cast<unsigned*>(db)+guard, static_cast<unsigned*>(dc)+guard, n);
                check(cudaGetLastError()); check(cudaDeviceSynchronize());
                check(cudaMemcpy(observed.data(), dc, bytes, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(after_a.data(), da, bytes, cudaMemcpyDeviceToHost));
                check(cudaMemcpy(after_b.data(), db, bytes, cudaMemcpyDeviceToHost));
                require(observed == expected && after_a == a && after_b == b);
            }
            check(cudaFree(dc)); dc = nullptr;
            check(cudaFree(db)); db = nullptr;
            check(cudaFree(da)); da = nullptr;
        }
        std::puts("TRANSLATED_CLIENT_PASS launches=6 exact_u32=true inputs_unchanged=true full_guards=true");
        return 0;
    } catch (const std::exception &error) {
        bool clean = true;
        for (void *pointer : {dc, db, da}) if (pointer) clean = (cudaFree(pointer) == cudaSuccess) && clean;
        std::fprintf(stderr, "TRANSLATED_CLIENT_FAIL clean=%u error=%s\n", unsigned(clean), error.what());
        return 1;
    }
}
