#include "ps5hip/native_memory.h"
#include <cstdio>
#include <cstring>

// Build/link check offline; run only in an authorized native title experiment.
int main(int argc, char **argv) {
    if (argc != 2 || std::strcmp(argv[1], "--memory-only") != 0) {
        std::fprintf(stderr, "usage: native-memory-smoke --memory-only (CPU mapping check, no GPU submission)\n");
        return 2;
    }
    try {
        ps5hip::NativeMemory memory(0x4000, 256, true);
        auto *bytes = static_cast<volatile unsigned char *>(memory.cpu());
        for (unsigned i = 0; i < 0x4000; ++i) bytes[i] = static_cast<unsigned char>(i ^ 0x5a);
        for (unsigned i = 0; i < 0x4000; ++i)
            if (bytes[i] != static_cast<unsigned char>(i ^ 0x5a)) throw ps5hip::ObjectError("CPU mapping mismatch");
        if (!memory.close()) throw ps5hip::ObjectError("native memory cleanup failed");
        std::puts("CPU direct-memory mapping passed; GPU execution was not tested");
    } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
