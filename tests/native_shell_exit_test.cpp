#include "../tools/native_runtime.h"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/wait.h>

extern "C" void mock_application_fini();
extern "C" int mock_shell_exit();
extern "C" unsigned mock_sleep(unsigned);
#define _fini mock_application_fini
#define sceShellCoreUtilExitApp mock_shell_exit
#define sleep mock_sleep
#include "../tools/native_shell_exit_probe.cpp"
#undef sleep
#undef sceShellCoreUtilExitApp
#undef _fini

namespace {
int scenario = 0, requests = 0, fini_calls = 0;
bool application_object_destroyed = false;
struct ApplicationObject {
    ~ApplicationObject() { application_object_destroyed = true; }
};
void initialize_application() { static ApplicationObject object; }
}
extern "C" void mock_application_fini() {
    if (!application_object_destroyed || ++fini_calls != 1) ::_exit(10);
}
extern "C" int mock_shell_exit() {
    if (!application_object_destroyed || fini_calls != 1 || ++requests != 1) ::_exit(11);
    return scenario == 1 ? -7 : 0;
}
extern "C" unsigned mock_sleep(unsigned) {
    const bool expected_request = scenario < 2;
    ::_exit(requests == (expected_request ? 1 : 0) ? 0 : 12);
}

int main() {
    char directory[] = "/tmp/ps5hip-shell-exit-XXXXXX";
    if (!::mkdtemp(directory)) return 1;
    bool passed = true;
    for (scenario = 0; scenario != 4; ++scenario) {
        const std::string path = std::string(directory) + "/" + std::to_string(scenario);
        const pid_t child = ::fork();
        if (child == 0) {
            if (!ps5hip_canary::redirect_log(path.c_str())) ::_exit(2);
            if (scenario != 3) ps5hip_install_shell_exit_probe();
            initialize_application();
            catchReturnFromMain(scenario == 2 ? 1 : 0);
            std::exit(0);
        }
        int status = 0;
        if (child < 0 || ::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
            WEXITSTATUS(status) != 0) passed = false;
        std::ifstream input(path);
        const std::string text{std::istreambuf_iterator<char>(input), {}};
        std::string expected = "PS5HIP FAIL: shell-exit startup or main\n";
        if (scenario < 2) {
            expected = "PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\n"
                       "PS5HIP SHUTDOWN static-dtor=observed\n"
                       "PS5HIP SHELL_EXIT application-fini=completed\n"
                       "PS5HIP SHELL_EXIT request=begin\n";
            expected += scenario == 0 ? "PS5HIP SHELL_EXIT request-return=0\n" :
                "PS5HIP FAIL: shell-exit request-return=0xfffffff9\n";
        }
        if (text != expected) {
            std::fprintf(stderr, "scenario %d: unexpected log: %s\n", scenario, text.c_str());
            passed = false;
        }
        ::unlink(path.c_str());
    }
    ::rmdir(directory);
    return passed ? 0 : 1;
}
