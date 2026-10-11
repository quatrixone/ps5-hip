#include "../tools/native_runtime.h"
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/wait.h>

int main() {
    char directory[] = "/tmp/ps5hip-log-XXXXXX";
    if (!::mkdtemp(directory)) return 1;
    const std::string path = std::string(directory) + "/run.log";
    int result = 1;
    try {
        if (!ps5hip_canary::valid_results_directory("/chosen/results")) return 1;
        for (const char *invalid : {"/", "relative", "/a/", "/a//b", "/a/../b", "/a/./b", "/a\n", "/a\\b"})
            if (ps5hip_canary::valid_results_directory(invalid)) return 1;
        if (ps5hip_canary::valid_results_directory("/" + std::string(240, 'x'))) return 1;
        ps5hip_canary::check_runtime();
        const pid_t child = ::fork();
        if (child == 0) {
            if (!ps5hip_canary::redirect_log(path.c_str())) ::_exit(2);
            std::printf("stdout checkpoint\n");
            std::fprintf(stderr, "stderr checkpoint\n");
            if (!ps5hip_canary::sync_log()) ::_exit(3);
            if (ps5hip_canary::redirect_log(path.c_str()) || errno != EEXIST) ::_exit(4);
            ::_exit(0);
        }
        int status = 0;
        if (child > 0 && ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            std::ifstream input(path);
            const std::string text{std::istreambuf_iterator<char>(input), {}};
            if (text == "stdout checkpoint\nstderr checkpoint\n") result = 0;
            else std::fprintf(stderr, "unexpected captured log: %s\n", text.c_str());
        }
        else std::fprintf(stderr, "log child status=%d\n", status);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
    }
    ::unlink(path.c_str());
    ::rmdir(directory);
    return result;
}
