#pragma once
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Record is the SDK's struct statfs in the payload. Host fixtures model fields,
// not the firmware ABI. No successful snapshot certifies namespace completeness.
namespace teardown {
constexpr size_t max_records = 1024;
constexpr size_t max_matching = 128;
constexpr const char* title = "PPSA99014";
constexpr const char* runtime = "/system_ex/app/PPSA99014";
constexpr const char* image = "/mnt/shadowmnt/PPSA99014_b6d9b2e4";

enum class Result { complete, query_error, count_bound, allocation_error,
                    unstable_capacity, malformed, output_bound };
inline const char* name(Result result) {
    switch (result) {
    case Result::complete: return "complete";
    case Result::query_error: return "query_error";
    case Result::count_bound: return "count_bound";
    case Result::allocation_error: return "allocation_error";
    case Result::unstable_capacity: return "unstable_capacity";
    case Result::malformed: return "malformed";
    case Result::output_bound: return "output_bound";
    }
    return "unknown";
}
inline bool component(const char* path) {
    // Match exact title components and numeric sandbox instances, never prefixes.
    for (const char* p = path; *p; ++p) {
        if ((p == path || p[-1] == '/') && strncmp(p, title, 9) == 0) {
            const char* end = p + 9;
            if (*end == '_') {
                ++end;
                if (*end < '0' || *end > '9') continue;
                while (*end >= '0' && *end <= '9') ++end;
            }
            if (*end == '\0' || *end == '/') return true;
        }
    }
    return false;
}
inline bool within(const char* path, const char* root) {
    const size_t length = strlen(root);
    return strncmp(path, root, length) == 0 &&
           (path[length] == '\0' || path[length] == '/');
}
enum class Kind { registered_image, runtime_layer, sandbox, other_title, unrelated };
inline Kind classify(const char* to, const char* from) {
    if (within(to, image)) return Kind::registered_image;
    if (within(to, runtime)) return Kind::runtime_layer;
    if ((within(to, "/mnt/sandbox") || within(to, "/sandbox")) && component(to))
        return Kind::sandbox;
    if (component(to) || component(from) || within(from, image)) return Kind::other_title;
    return Kind::unrelated;
}
inline const char* name(Kind kind) {
    switch (kind) {
    case Kind::registered_image: return "registered_image";
    case Kind::runtime_layer: return "runtime_layer";
    case Kind::sandbox: return "sandbox";
    case Kind::other_title: return "other_title";
    case Kind::unrelated: return "unrelated";
    }
    return "unknown";
}
template<class Record> bool valid(const Record& record, uint32_t version) {
    return record.f_version == version &&
        memchr(record.f_fstypename, 0, sizeof(record.f_fstypename)) &&
        memchr(record.f_mntonname, 0, sizeof(record.f_mntonname)) &&
        memchr(record.f_mntfromname, 0, sizeof(record.f_mntfromname));
}
template<class Record> int compare(const void* left, const void* right) {
    const auto& a = *static_cast<const Record*>(left);
    const auto& b = *static_cast<const Record*>(right);
    int order = strcmp(a.f_mntonname, b.f_mntonname);
    if (!order) order = strcmp(a.f_mntfromname, b.f_mntfromname);
    if (!order) order = strcmp(a.f_fstypename, b.f_fstypename);
    if (!order) order = memcmp(&a.f_fsid, &b.f_fsid, sizeof(a.f_fsid));
    if (!order && a.f_flags != b.f_flags) order = a.f_flags < b.f_flags ? -1 : 1;
    if (!order && a.f_type != b.f_type) order = a.f_type < b.f_type ? -1 : 1;
    return order;
}
struct QueryTrace {
    bool attempted = false;
    size_t capacity_records = 0;
    long capacity_bytes = 0;
    int returned = 0;
    int error = 0;
};
template<class Record> struct Snapshot {
    Record* rows = nullptr;
    size_t count = 0;
    size_t matching = 0;
    int query_errno = 0;
    Result result = Result::query_error;
    QueryTrace initial;
    QueryTrace fills[3];
    ~Snapshot() { free(rows); }
    Snapshot() = default;
    Snapshot(const Snapshot&) = delete;
    Snapshot& operator=(const Snapshot&) = delete;
};
// Query takes (Record*, long bytes); the native wrapper fixes MNT_NOWAIT.
template<class Record, class Query>
void collect(Snapshot<Record>& out, uint32_t version, Query query,
             void* (*allocate)(size_t, size_t) = calloc) {
    if (out.rows) { out.result = Result::malformed; return; }
    out.initial.attempted = true;
    errno = 0;
    const int size = query(nullptr, 0);
    out.query_errno = errno;
    out.initial.returned = size;
    out.initial.error = out.query_errno;
    if (size < 0) return;
    if (static_cast<size_t>(size) > max_records - 16 ||
        sizeof(Record) > static_cast<size_t>(LONG_MAX) / max_records) {
        out.result = Result::count_bound; return;
    }
    size_t capacity = static_cast<size_t>(size) + 16;
    for (int attempt = 0; attempt != 3; ++attempt) {
        auto* rows = static_cast<Record*>(allocate(capacity, sizeof(Record)));
        if (!rows) { out.result = Result::allocation_error; return; }
        auto& trace = out.fills[attempt];
        trace.attempted = true;
        trace.capacity_records = capacity;
        trace.capacity_bytes = static_cast<long>(capacity * sizeof(Record));
        errno = 0;
        const int count = query(rows, static_cast<long>(capacity * sizeof(Record)));
        out.query_errno = errno;
        trace.returned = count;
        trace.error = out.query_errno;
        if (count < 0) { free(rows); return; }
        if (static_cast<size_t>(count) >= capacity) {
            free(rows);
            if (capacity == max_records) break;
            capacity = capacity > max_records / 2 ? max_records : capacity * 2;
            continue;
        }
        out.rows = rows;
        out.count = static_cast<size_t>(count);
        for (size_t i = 0; i != out.count; ++i) {
            if (!valid(rows[i], version)) { out.result = Result::malformed; return; }
            if (classify(rows[i].f_mntonname, rows[i].f_mntfromname) != Kind::unrelated)
                ++out.matching;
        }
        if (out.matching > max_matching) { out.result = Result::output_bound; return; }
        qsort(rows, out.count, sizeof(Record), compare<Record>);
        out.result = Result::complete;
        return;
    }
    out.result = Result::unstable_capacity;
}
template<class Record> bool same(const Snapshot<Record>& a, const Snapshot<Record>& b) {
    if (a.result != Result::complete || b.result != Result::complete || a.count != b.count)
        return false;
    for (size_t i = 0; i != a.count; ++i)
        if (compare<Record>(a.rows + i, b.rows + i)) return false;
    return true;
}
struct Status { int exists; int error; int running; int title_app; };
inline bool same(const Status& a, const Status& b) {
    return a.exists == b.exists && a.error == b.error && a.running == b.running &&
           a.title_app == b.title_app;
}
inline bool queried_absence(const Status& status) {
    return status.exists == -1 && status.error == ESRCH &&
           status.running == -1 && status.title_app == -1;
}
} // namespace teardown
