#include "../tools/native_runtime.h"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/wait.h>

extern "C" void catchReturnFromMain(int);

int main() {
    char directory[] = "/tmp/ps5hip-shutdown-XXXXXX";
    if (!::mkdtemp(directory)) return 1;
    bool passed = true;
    for (const int code : {0, 1}) {
        const std::string path = std::string(directory) + "/" + std::to_string(code);
        const pid_t child = ::fork();
        if (child == 0) {
            if (!ps5hip_canary::redirect_log(path.c_str())) ::_exit(2);
            catchReturnFromMain(code);
            std::exit(code);
        }
        int status = 0;
        if (child < 0 || ::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
            WEXITSTATUS(status) != code) passed = false;
        std::ifstream input(path);
        const std::string actual{std::istreambuf_iterator<char>(input), {}};
        const std::string expected = code == 0 ?
            "PS5HIP SHUTDOWN main-return=0\nPS5HIP SHUTDOWN atexit=observed\n"
            "PS5HIP SHUTDOWN static-dtor=observed\n" :
            "PS5HIP FAIL: main returned nonzero before shutdown\n";
        if (actual != expected) passed = false;
        ::unlink(path.c_str());
    }
    ::rmdir(directory);
    return passed ? 0 : 1;
}
