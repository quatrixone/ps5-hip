/* Read-only diagnostic for first-native-cpu.json; no launch, close or kernel writes. */
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>

int sceLncUtilGetAppIdOfRunningBigApp(void);
int sceSystemServiceGetAppId(const char *title);

int main(void)
{
    errno = 0;
    const int exists = kill(204, 0); /* Signal zero queries existence; sends no signal. */
    const int query_errno = errno;
    const int running = sceLncUtilGetAppIdOfRunningBigApp();
    const int title = sceSystemServiceGetAppId("PPSA99014");
    printf("PS5HIP STATUS observed_pid=204 existence_rc=%d errno=%d running_app=0x%08x title=PPSA99014 title_app=0x%08x\n",
           exists, query_errno, (unsigned)running, (unsigned)title);
    return 0;
}
