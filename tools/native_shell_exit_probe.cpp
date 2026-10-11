// Finite canary lifecycle: finish application callbacks while platform libraries live.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

extern "C" void _fini();
extern "C" int sceShellCoreUtilExitApp();

namespace {
bool installed = false, failed = false;
unsigned phase = 0;

template <unsigned N> void checkpoint(const char (&text)[N]) {
    if (::write(STDOUT_FILENO, text, N - 1) != N - 1 || ::fsync(STDOUT_FILENO)) {
        failed = true;
        constexpr char error[] = "PS5HIP FAIL: shell-exit checkpoint persistence\n";
        (void)::write(STDERR_FILENO, error, sizeof(error) - 1);
    }
}
[[noreturn]] void wait_for_controller() {
    // The bounded host case closes this exact title if the service does not.
    for (;;) ::sleep(1);
}
struct DestructorProbe {
    ~DestructorProbe() {
        if (phase != 2) failed = true;
        phase = 3;
        checkpoint("PS5HIP SHUTDOWN static-dtor=observed\n");
    }
};
void exit_callback() {
    if (phase != 1) failed = true;
    phase = 2;
    checkpoint("PS5HIP SHUTDOWN atexit=observed\n");
}
[[noreturn]] void application_finished() {
    if (failed || phase != 3) {
        checkpoint("PS5HIP FAIL: shell-exit callback order\n");
        wait_for_controller();
    }
    // This callback precedes startup's _fini and loader teardown in reverse order.
    // Run application fini now; never return and execute it a second time.
    _fini();
    if (std::fflush(nullptr)) {
        checkpoint("PS5HIP FAIL: shell-exit stdio flush\n");
        wait_for_controller();
    }
    checkpoint("PS5HIP SHELL_EXIT application-fini=completed\n");
    if (failed) wait_for_controller();
    checkpoint("PS5HIP SHELL_EXIT request=begin\n");
    if (failed) wait_for_controller();
    const int result = sceShellCoreUtilExitApp();
    if (result == 0)
        checkpoint("PS5HIP SHELL_EXIT request-return=0\n");
    else {
        char message[96];
        const int length = std::snprintf(message, sizeof(message),
            "PS5HIP FAIL: shell-exit request-return=0x%08x\n", static_cast<unsigned>(result));
        if (length > 0 && static_cast<unsigned>(length) < sizeof(message)) {
            (void)::write(STDERR_FILENO, message, static_cast<unsigned>(length));
            (void)::fsync(STDERR_FILENO);
        }
    }
    wait_for_controller();
}
}

extern "C" void ps5hip_install_shell_exit_probe() {
    // Before application initializers: their destructors/callbacks run first.
    installed = std::atexit(application_finished) == 0;
}

extern "C" void catchReturnFromMain(int status) {
    if (status != 0 || !installed) {
        checkpoint("PS5HIP FAIL: shell-exit startup or main\n");
        wait_for_controller();
    }
    static DestructorProbe probe;
    if (std::atexit(exit_callback)) {
        checkpoint("PS5HIP FAIL: shell-exit callback registration\n");
        wait_for_controller();
    }
    phase = 1;
    checkpoint("PS5HIP SHUTDOWN main-return=0\n");
}
