/* One-shot cleanup of the observed CPU-only loader failure, not a generic killer. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>

struct AppInfo {
    uint32_t app_id;
    uint64_t unknown1;
    char title_id[14];
    char unknown2[0x3c];
};
int sceKernelGetAppInfo(int pid, struct AppInfo *info);

int main(void)
{
    struct AppInfo info = {0};
    const int query = sceKernelGetAppInfo(204, &info);
    if (query != 0 || memcmp(info.title_id, "PPSA99014", sizeof("PPSA99014")) != 0) {
        printf("PS5HIP CLEANUP refused pid=204 query=0x%08x\n", (unsigned)query);
        return 1;
    }
    const int result = kill(204, SIGKILL);
    printf("PS5HIP CLEANUP title=PPSA99014 pid=204 app=0x%08x kill_rc=%d\n", info.app_id, result);
    return result != 0;
}
