#include "../tools/teardown_snapshot.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Deliberately a host model, never a claim about firmware struct layout.
struct Row {
    std::uint32_t f_version = 7, f_type = 1;
    std::uint64_t f_flags = 0;
    int f_fsid[2] = {1, 2};
    char f_fstypename[16] = "nullfs";
    char f_mntonname[88] = "/unrelated";
    char f_mntfromname[88] = "/unrelated-source";
    unsigned volatile_free_blocks = 0;
};
void check(bool value) { if (!value) throw std::runtime_error("teardown snapshot assertion"); }
Row row(const char* to, const char* from = "/unrelated-source") {
    Row value;
    check(std::strlen(to) < sizeof(value.f_mntonname));
    check(std::strlen(from) < sizeof(value.f_mntfromname));
    std::strcpy(value.f_mntonname, to); std::strcpy(value.f_mntfromname, from);
    return value;
}
struct Query {
    std::vector<Row> rows;
    int calls = 0, size_override = -1, error_on = -1, full_calls = 0;
    int operator()(Row* output, long bytes) {
        ++calls;
        if (calls == error_on) { errno = EACCES; return -1; }
        if (!output) { check(bytes == 0); return size_override < 0 ? static_cast<int>(rows.size()) : size_override; }
        check(bytes > 0 && bytes % sizeof(Row) == 0);
        const auto capacity = static_cast<std::size_t>(bytes) / sizeof(Row);
        check(capacity <= teardown::max_records);
        if (full_calls-- > 0) return static_cast<int>(capacity);
        check(rows.size() < capacity);
        std::copy(rows.begin(), rows.end(), output);
        return static_cast<int>(rows.size());
    }
};
void collect(teardown::Snapshot<Row>& result, Query& query) {
    teardown::collect(result, 7, [&query](Row* output, long bytes) { return query(output, bytes); });
}
int main() {
    using teardown::Kind;
    using teardown::Result;
    check(teardown::classify(teardown::image, "/dev/lvd2") == Kind::registered_image);
    check(teardown::classify(teardown::runtime, teardown::image) == Kind::runtime_layer);
    check(teardown::classify("/mnt/sandbox/PPSA99014_000/app0", teardown::runtime) == Kind::sandbox);
    check(teardown::classify("/sandbox/PPSA99014_12", "x") == Kind::sandbox);
    check(teardown::classify("/unknown/PPSA99014", "x") == Kind::other_title);
    check(teardown::classify("/other", "/mnt/shadowmnt/PPSA99014_b6d9b2e4/sub") == Kind::other_title);
    for (const char* path : {"", "/PPSA990140", "/PPSA99014x", "/PPSA99014_", "/PPSA99014_a",
                             "/PPSA99014_000x", "/foreignPPSA99014", "/system_ex/app/PPSA990140",
                             "/mnt/shadowmnt/PPSA99014_b6d9b2e4-foreign"})
        check(teardown::classify(path, "") == Kind::unrelated);
    for (const char* path : {"PPSA99014", "/PPSA99014/child", "/PPSA99014_000/child"})
        check(teardown::component(path));

    Query q;
    teardown::Snapshot<Row> allocation_failure;
    teardown::collect(allocation_failure, 7,
        [&q](Row* rows, long bytes) { return q(rows, bytes); },
        +[](std::size_t, std::size_t) -> void* { return nullptr; });
    check(allocation_failure.result == Result::allocation_error && !allocation_failure.rows);
    teardown::Snapshot<Row> empty1, empty2; collect(empty1, q); collect(empty2, q);
    check(empty1.result == Result::complete && empty1.count == 0 && teardown::same(empty1, empty2));
    q.rows = {row(teardown::runtime, teardown::image), row(teardown::image, "/dev/lvd2"),
              row("/mnt/sandbox/PPSA99014_000/app0"), row("/other")};
    teardown::Snapshot<Row> first; collect(first, q);
    check(first.result == Result::complete && first.count == 4 && first.matching == 3);
    std::reverse(q.rows.begin(), q.rows.end());
    q.rows[0].volatile_free_blocks = 123;
    teardown::Snapshot<Row> reverse; collect(reverse, q); check(teardown::same(first, reverse));
    // Duplicate mount layers are evidence and must not disappear through sorting.
    q.rows.push_back(q.rows[3]);
    teardown::Snapshot<Row> duplicate; collect(duplicate, q);
    check(duplicate.matching == 4 && !teardown::same(first, duplicate));
    for (unsigned mutation = 0; mutation != 6; ++mutation) {
        Query changed; changed.rows = {row(teardown::runtime, teardown::image)};
        teardown::Snapshot<Row> base; collect(base, changed);
        switch (mutation) {
        case 0: changed.rows[0].f_flags = 1; break;
        case 1: changed.rows[0].f_type = 2; break;
        case 2: changed.rows[0].f_fsid[1] = 4; break;
        case 3: std::strcpy(changed.rows[0].f_fstypename, "unionfs"); break;
        case 4: std::strcpy(changed.rows[0].f_mntonname, "/other"); break;
        case 5: std::strcpy(changed.rows[0].f_mntfromname, "/other"); break;
        }
        teardown::Snapshot<Row> different; collect(different, changed); check(!teardown::same(base, different));
    }
    for (unsigned mutation = 0; mutation != 4; ++mutation) {
        Query broken; broken.rows = {Row{}};
        switch (mutation) {
        case 0: broken.rows[0].f_version = 0; break;
        case 1: std::memset(broken.rows[0].f_fstypename, 'x', 16); break;
        case 2: std::memset(broken.rows[0].f_mntonname, 'x', 88); break;
        case 3: std::memset(broken.rows[0].f_mntfromname, 'x', 88); break;
        }
        teardown::Snapshot<Row> result; collect(result, broken);
        check(result.result == Result::malformed && !teardown::same(result, result));
    }
    for (int phase : {1, 2}) {
        Query error; error.error_on = phase;
        teardown::Snapshot<Row> result; collect(result, error);
        check(result.result == Result::query_error && result.query_errno == EACCES && !result.rows);
    }
    for (int count : {1009, 1024, INT_MAX}) {
        Query large; large.size_override = count;
        teardown::Snapshot<Row> result; collect(result, large);
        check(result.result == Result::count_bound && large.calls == 1);
    }
    Query growth; growth.full_calls = 2;
    teardown::Snapshot<Row> grown; collect(grown, growth);
    check(grown.result == Result::complete && growth.calls == 4);
    Query endless; endless.full_calls = 3;
    teardown::Snapshot<Row> unstable; collect(unstable, endless);
    check(unstable.result == Result::unstable_capacity && endless.calls == 4);
    Query maximum; maximum.size_override = 1008; maximum.full_calls = 1;
    teardown::Snapshot<Row> capped; collect(capped, maximum);
    check(capped.result == Result::unstable_capacity && maximum.calls == 2);
    Query output; output.rows.assign(129, row(teardown::runtime));
    teardown::Snapshot<Row> too_many; collect(too_many, output);
    check(too_many.result == Result::output_bound && !teardown::same(too_many, too_many));
    output.rows.pop_back();
    teardown::Snapshot<Row> allowed; collect(allowed, output);
    check(allowed.result == Result::complete && allowed.matching == 128);

    const teardown::Status absent{-1, ESRCH, -1, -1};
    check(teardown::queried_absence(absent));
    for (const auto& value : {teardown::Status{0, 0, -1, -1}, {-1, EPERM, -1, -1},
                             {-1, ESRCH, 4, -1}, {-1, ESRCH, -1, 4}, {-1, 0, -1, -1}}) {
        check(!teardown::queried_absence(value)); check(!teardown::same(absent, value));
    }
    std::puts("teardown host model passed: bounded snapshots, field/string validation, exact title classification, duplicate layers, status uncertainty; no firmware ABI or release certification");
}
