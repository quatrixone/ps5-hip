#define main cleanup_main
#include "../evidence/first-failure-close.c"
#undef main
#include <assert.h>

static int query_result, calls;
static const char *title;
int sceKernelGetAppInfo(int pid, struct AppInfo *info) {
    assert(pid == 204);
    strcpy(info->title_id, title);
    return query_result;
}
int kill(int pid, int sig) {
    assert(pid == 204 && sig == SIGKILL);
    ++calls;
    return 0;
}
int main(void) {
    title = "PPSA99014"; query_result = -1;
    assert(cleanup_main() == 1 && calls == 0);
    query_result = 0; title = "PPSA99008";
    assert(cleanup_main() == 1 && calls == 0);
    title = "PPSA99014";
    assert(cleanup_main() == 0 && calls == 1);
    return 0;
}
