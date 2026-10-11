#pragma once
#include "ps5hip/code_object.h"
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <time.h>
#include <unistd.h>

namespace ps5hip_canary {
// O_EXCL prevents an old PASS record from being overwritten or reused silently.
inline bool redirect_log(const char *path) {
    if (std::fflush(nullptr) != 0) return false;
    const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return false;
    const bool ready = ::dup2(fd, STDOUT_FILENO) >= 0 && ::dup2(fd, STDERR_FILENO) >= 0;
    if (fd > STDERR_FILENO) ::close(fd);
    if (!ready) return false;
    return std::setvbuf(stdout, nullptr, _IONBF, 0) == 0 &&
           std::setvbuf(stderr, nullptr, _IONBF, 0) == 0;
}

inline bool sync_log() {
    return std::fflush(nullptr) == 0 && ::fsync(STDOUT_FILENO) == 0;
}

inline std::string constructor_probe = std::string(80, 'c');
inline thread_local unsigned tls_probe = 17;

struct UnwindProbe {
    unsigned &destroyed;
    ~UnwindProbe() { ++destroyed; }
};
[[gnu::noinline]] inline void throw_probe(unsigned &destroyed) {
    UnwindProbe probe{destroyed};
    throw ps5hip::ObjectError(std::string(80, 'e'));
}

inline void check_runtime() {
    if (constructor_probe != std::string(80, 'c') || tls_probe != 17)
        throw ps5hip::ObjectError("static constructor or TLS initialization failed");
    tls_probe = 23;
    unsigned destroyed = 0;
    bool caught = false;
    try { throw_probe(destroyed); }
    catch (const ps5hip::ObjectError &error) {
        caught = std::strcmp(error.what(), std::string(80, 'e').c_str()) == 0;
    }
    if (!caught || destroyed != 1 || tls_probe != 23)
        throw ps5hip::ObjectError("native exception cleanup or TLS failed");
    timespec before{}, after{};
    if (::clock_gettime(CLOCK_MONOTONIC, &before) || ::clock_gettime(CLOCK_MONOTONIC, &after) ||
        before.tv_sec < 0 || after.tv_sec < before.tv_sec ||
        (after.tv_sec == before.tv_sec && after.tv_nsec < before.tv_nsec))
        throw ps5hip::ObjectError("native monotonic clock failed");
    std::printf("PS5HIP RUNTIME constructors=pass heap=pass unwind=pass tls=pass clock=pass\n");
}
} // namespace ps5hip_canary
