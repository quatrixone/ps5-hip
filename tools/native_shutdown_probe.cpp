// CPU-only diagnostic: observe teardown without replacing the platform exit path.
#include <cstdlib>
#include <unistd.h>

namespace {
template <unsigned N> void checkpoint(const char (&text)[N]) {
    // Avoid stdio and heap use while the runtime is destroying global objects.
    if (::write(STDOUT_FILENO, text, N - 1) != N - 1 || ::fsync(STDOUT_FILENO)) {
        constexpr char failure[] = "PS5HIP FAIL: shutdown checkpoint persistence\n";
        (void)::write(STDERR_FILENO, failure, sizeof(failure) - 1);
    }
}
struct DestructorProbe {
    ~DestructorProbe() { checkpoint("PS5HIP SHUTDOWN static-dtor=observed\n"); }
};
void exit_callback() { checkpoint("PS5HIP SHUTDOWN atexit=observed\n"); }
}

extern "C" void catchReturnFromMain(int status) {
    if (status != 0) {
        checkpoint("PS5HIP FAIL: main returned nonzero before shutdown\n");
        return;
    }
    static DestructorProbe probe;
    if (std::atexit(exit_callback)) {
        checkpoint("PS5HIP FAIL: shutdown callback registration\n");
        return;
    }
    checkpoint("PS5HIP SHUTDOWN main-return=0\n");
}
