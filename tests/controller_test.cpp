#include <cstring>
#include <stdexcept>
#include <cstdio>
#define main controller_main
#include PS5HIP_CONTROLLER_SOURCE
#undef main

namespace {
void check(bool value) { if (!value) throw std::runtime_error("controller assertion failed"); }
int calls = 0, result = 0;
}

#ifdef PS5HIP_TEST_LAUNCH
namespace { int init_result = 0, user_result = 0, terminated = 0; }
int sceUserServiceInitialize(void *arg) { check(arg == nullptr); return init_result; }
int sceUserServiceGetForegroundUser(uint32_t *user) { *user = 27; return user_result; }
void sceUserServiceTerminate() { ++terminated; }
int sceSystemServiceLaunchApp(const char *title, char **args, launch_context_t *context) {
    ++calls;
    check(std::strcmp(title, BOOTSTRAP_TITLE_ID) == 0 && args && !args[0]);
    check(context->size == sizeof(*context) && context->user_id == 27 &&
          !context->options && !context->crash_report && !context->flags);
    return result;
}
#else
namespace { const char *running = BOOTSTRAP_TITLE_ID; int running_id = 42, lookup_result = 0, fallback_id = 0, fallback_calls = 0; }
int sceLncUtilGetAppIdOfRunningBigApp() { return running_id; }
int sceLncUtilGetAppTitleId(uint32_t id, char *title) {
    check(id == 42); std::strcpy(title, running); return lookup_result;
}
int sceLncUtilKillApp(uint32_t id) {
    ++calls; check(id == 42 && std::strcmp(running, BOOTSTRAP_TITLE_ID) == 0); return result;
}
int sceSystemServiceGetAppId(const char *title) { check(std::strcmp(title, BOOTSTRAP_TITLE_ID) == 0); return fallback_id; }
int sceSystemServiceKillApp(int id, int a, int b, int c) {
    ++fallback_calls; check(id == 99 && a == -1 && b == 0 && c == 0); return result;
}
#endif

int main() {
    try {
#ifdef PS5HIP_TEST_LAUNCH
        static_assert(sizeof(launch_context_t) == 32);
        init_result = -1;
        check(controller_main() == 1 && calls == 0 && terminated == 0);
        init_result = 0; user_result = -2;
        check(controller_main() == 1 && calls == 0 && terminated == 1);
        user_result = 0; result = -3;
        check(controller_main() == 1 && calls == 1 && terminated == 2);
        result = 0;
        check(controller_main() == 0 && calls == 2 && terminated == 3);
#else
        check(controller_main() == 0 && calls == 1 && fallback_calls == 0);
        result = -1;
        check(controller_main() == 1 && calls == 2 && fallback_calls == 0);
        result = 0; running = "PPSA99999";
        check(controller_main() == 1 && calls == 2 && fallback_calls == 0);
        fallback_id = 99;
        check(controller_main() == 0 && calls == 2 && fallback_calls == 1);
        lookup_result = -1;
        check(controller_main() == 0 && calls == 2 && fallback_calls == 2);
        running_id = 0; fallback_id = 0;
        check(controller_main() == 1 && calls == 2 && fallback_calls == 2);
#endif
    } catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}

