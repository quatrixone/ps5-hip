// Read-only diagnostic, never a release certificate. No launch/close or mount writes.
#include <sys/param.h>
#include <sys/mount.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include "teardown_snapshot.h"

extern "C" int sceLncUtilGetAppIdOfRunningBigApp(void);
extern "C" int sceSystemServiceGetAppId(const char* title);

static teardown::Status status() {
    errno = 0;
    const int exists = kill(134, 0); // Signal zero only: PID reuse is still possible.
    const int error = errno;
    return {exists, error, sceLncUtilGetAppIdOfRunningBigApp(),
            sceSystemServiceGetAppId(teardown::title)};
}
static void print_status(const char* phase, const teardown::Status& value) {
    printf("PS5HIP TEARDOWN STATUS phase=%s pid=134 rc=%d errno=%d running=0x%08x title_app=0x%08x\n",
                phase, value.exists, value.error, static_cast<unsigned>(value.running),
                static_cast<unsigned>(value.title_app));
}
static void hex(const void* pointer, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(pointer);
    for (size_t i = 0; i != size; ++i) printf("%02x", bytes[i]);
}
static void print_snapshot(unsigned index, const teardown::Snapshot<struct statfs>& snapshot) {
    printf("PS5HIP TEARDOWN SNAPSHOT index=%u result=%s count=%zu matching=%zu errno=%d\n",
                index, teardown::name(snapshot.result), snapshot.count, snapshot.matching, snapshot.query_errno);
    const auto& initial = snapshot.initial;
    printf("PS5HIP TEARDOWN INITIAL snapshot=%u attempted=%u return_signed=%d return_hex=0x%08x errno=%d\n",
                index, static_cast<unsigned>(initial.attempted), initial.returned,
                static_cast<unsigned>(initial.returned), initial.error);
    for (unsigned attempt = 0; attempt != 3; ++attempt) {
        const auto& trace = snapshot.fills[attempt];
        printf("PS5HIP TEARDOWN FILL snapshot=%u attempt=%u attempted=%u capacity_records=%zu capacity_bytes=%ld return_signed=%d return_hex=0x%08x errno=%d\n",
                    index, attempt + 1, static_cast<unsigned>(trace.attempted),
                    trace.capacity_records, trace.capacity_bytes, trace.returned,
                    static_cast<unsigned>(trace.returned), trace.error);
    }
    if (snapshot.result != teardown::Result::complete) return;
    for (size_t i = 0; i != snapshot.count; ++i) {
        const auto& row = snapshot.rows[i];
        const auto kind = teardown::classify(row.f_mntonname, row.f_mntfromname);
        if (kind == teardown::Kind::unrelated) continue;
        printf("PS5HIP TEARDOWN MOUNT snapshot=%u kind=%s type=%u flags=0x%llx fsid_hex=",
                    index, teardown::name(kind), row.f_type, static_cast<unsigned long long>(row.f_flags));
        hex(&row.f_fsid, sizeof(row.f_fsid));
        printf(" fstype_hex="); hex(row.f_fstypename, strlen(row.f_fstypename));
        printf(" to_hex="); hex(row.f_mntonname, strlen(row.f_mntonname));
        printf(" from_hex="); hex(row.f_mntfromname, strlen(row.f_mntfromname));
        printf("\n");
    }
}
int main() {
    static_assert(sizeof(int) == 4 && sizeof(unsigned) == 4);
    // Two *128 bounded relevant records; SDK path lengths keep output below256KiB.
    static_assert(sizeof(((struct statfs*)nullptr)->f_mntonname) <= 88);
    static_assert(sizeof(((struct statfs*)nullptr)->f_mntfromname) <= 88);
    static_assert(sizeof(((struct statfs*)nullptr)->f_fstypename) <= 16);
    printf("PS5HIP TEARDOWN BEGIN title=PPSA99014 observed_pid=134 version=3 uid=%u euid=%u statfs_size=%zu statfs_version=0x%x namespace=unqualified\n",
                static_cast<unsigned>(getuid()), static_cast<unsigned>(geteuid()), sizeof(struct statfs), STATFS_VERSION);
    const auto before = status();
    teardown::Snapshot<struct statfs> first, second;
    auto query = [](struct statfs* rows, long bytes) { return getfsstat(rows, bytes, MNT_NOWAIT); };
    teardown::collect(first, STATFS_VERSION, query);
    teardown::collect(second, STATFS_VERSION, query);
    const auto after = status();
    print_status("before", before);
    print_snapshot(1, first); print_snapshot(2, second);
    print_status("after", after);
    printf("PS5HIP TEARDOWN END snapshots_stable=%u status_stable=%u pid134_esrch_and_app_queries_empty=%u namespace=unqualified release=unknown accepted=false\n",
                teardown::same(first, second), teardown::same(before, after),
                teardown::queried_absence(before) && teardown::queried_absence(after));
    // Unknown visibility/release is intentionally a non-success diagnostic outcome.
    return 2;
}
