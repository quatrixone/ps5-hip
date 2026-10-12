#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <array>

// Interpose only the production header's free calls, without changing its API.
void tracked_free(void* pointer);
#define free(pointer) tracked_free(pointer)
#include "../tools/teardown_snapshot.h"
#undef free

// Deliberately a host model, never a claim about firmware struct layout.
struct Row {
    std::uint32_t f_version = 7, f_type = 1;
    std::uint64_t f_flags = 0;
    int f_fsid[2] = {1, 2};
    char f_fstypename[16] = "nullfs";
    char f_mntonname[88] = "/unrelated";
    char f_mntfromname[88] = "/unrelated-source";
    unsigned volatile_free_blocks = 0;
    unsigned char stride_padding[252] = {};
};
static_assert(sizeof(Row) == 472, "Exercise SDK-sized strides, not firmware ABI compatibility");
void check(bool value) { if (!value) throw std::runtime_error("teardown snapshot assertion"); }
struct Allocation { void* pointer = nullptr; size_t bytes = 0; };
std::array<Allocation, 4> allocations;
size_t live_bytes = 0, peak_bytes = 0, allocate_calls = 0, fail_allocate_call = 0;
void* tracked_allocate(size_t count, size_t size) {
    ++allocate_calls;
    if (allocate_calls == fail_allocate_call) return nullptr;
    check(count <= teardown::max_records && size == sizeof(Row));
    void* pointer = std::calloc(count, size);
    check(pointer != nullptr);
    for (auto& allocation : allocations) if (!allocation.pointer) {
        allocation = {pointer, count * size};
        live_bytes += allocation.bytes;
        peak_bytes = std::max(peak_bytes, live_bytes);
        check(live_bytes <= 2 * teardown::max_records * sizeof(Row));
        return pointer;
    }
    throw std::runtime_error("unexpected live allocation count");
}
void tracked_free(void* pointer) {
    for (auto& allocation : allocations) if (pointer && allocation.pointer == pointer) {
        live_bytes -= allocation.bytes;
        allocation = {};
        break;
    }
    std::free(pointer);
}
void reset_allocations() {
    check(live_bytes == 0);
    for (const auto& allocation : allocations) check(!allocation.pointer);
    peak_bytes = allocate_calls = fail_allocate_call = 0;
}
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
    std::vector<teardown::QueryTrace> expected;
    teardown::collect(result, 7, [&](Row* output, long bytes) {
        const int returned = query(output, bytes);
        const int error = errno;
        expected.push_back({true, output ? static_cast<size_t>(bytes) / sizeof(Row) : 0,
                            bytes, returned, error});
        errno = error;
        return returned;
    });
    check(!expected.empty() && expected.size() <= 4);
    const auto compare_trace = [](const teardown::QueryTrace& actual, const teardown::QueryTrace& wanted) {
        check(actual.attempted == wanted.attempted && actual.capacity_records == wanted.capacity_records &&
              actual.capacity_bytes == wanted.capacity_bytes && actual.returned == wanted.returned &&
              actual.error == wanted.error);
    };
    compare_trace(result.initial, expected.front());
    for (size_t i = 0; i != 3; ++i)
        compare_trace(result.fills[i], i + 1 < expected.size() ? expected[i + 1] : teardown::QueryTrace{});
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
    check(allocation_failure.initial.attempted && allocation_failure.initial.returned == 0);
    for (const auto& fill : allocation_failure.fills) check(!fill.attempted);
    teardown::Snapshot<Row> empty1, empty2; collect(empty1, q); collect(empty2, q);
    check(empty1.result == Result::complete && empty1.count == 0 && teardown::same(empty1, empty2));
    check(empty1.initial.attempted && empty1.initial.returned == 0 && empty1.fills[0].attempted &&
          empty1.fills[0].returned == 0 && empty1.fills[0].capacity_records == 16);
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
        const auto& failed = phase == 1 ? result.initial : result.fills[0];
        check(failed.attempted && failed.returned == -1 && static_cast<unsigned>(failed.returned) == UINT32_MAX &&
              failed.error == EACCES && error.calls == phase);
        if (phase == 1) for (const auto& fill : result.fills) check(!fill.attempted);
    }
    for (int count : {2033, 2048, INT_MAX}) {
        Query large; large.size_override = count;
        teardown::Snapshot<Row> result; collect(result, large);
        check(result.result == Result::count_bound && large.calls == 1);
        check(result.initial.attempted && result.initial.returned == count && result.initial.error == 0 &&
              !result.rows && result.count == 0 && !teardown::same(result, empty1));
        for (const auto& fill : result.fills) check(!fill.attempted && fill.capacity_bytes == 0);
    }
    Query growth; growth.full_calls = 2;
    teardown::Snapshot<Row> grown; collect(grown, growth);
    check(grown.result == Result::complete && growth.calls == 4);
    for (size_t i = 0; i != 3; ++i) {
        check(grown.fills[i].attempted && grown.fills[i].capacity_records == (size_t(16) << i) &&
              grown.fills[i].capacity_bytes == static_cast<long>((size_t(16) << i) * sizeof(Row)));
        check(grown.fills[i].returned == (i == 2 ? 0 : int(16 << i)));
    }
    Query endless; endless.full_calls = 3;
    teardown::Snapshot<Row> unstable; collect(unstable, endless);
    check(unstable.result == Result::unstable_capacity && endless.calls == 4);
    Query maximum; maximum.size_override = 2032; maximum.full_calls = 1;
    teardown::Snapshot<Row> capped; collect(capped, maximum);
    check(capped.result == Result::unstable_capacity && maximum.calls == 2);
    check(capped.initial.returned == 2032 && capped.fills[0].capacity_records == 2048 &&
          capped.fills[0].returned == 2048 && !capped.fills[1].attempted);
    Query output; output.rows.assign(129, row(teardown::runtime));
    teardown::Snapshot<Row> too_many; collect(too_many, output);
    check(too_many.result == Result::output_bound && !teardown::same(too_many, too_many));
    output.rows.pop_back();
    teardown::Snapshot<Row> allowed; collect(allowed, output);
    check(allowed.result == Result::complete && allowed.matching == 128);

    // The observed initial return now reaches a real, SDK-stride-sized fill.
    reset_allocations();
    {
        Query observed; observed.rows.assign(1110, Row{});
        observed.rows.back() = row(teardown::image, "/dev/lvd2");
        teardown::Snapshot<Row> result;
        teardown::collect(result, 7, [&](Row* rows, long bytes) { return observed(rows, bytes); }, tracked_allocate);
        check(result.result == Result::complete && result.count == 1110 && result.matching == 1);
        check(result.initial.returned == 1110 && result.fills[0].capacity_records == 1126 &&
              result.fills[0].capacity_bytes == 531472 && observed.calls == 2);
        check(peak_bytes == 531472);
    }
    reset_allocations();
    for (int full_calls : {1, 2}) {
        Query observed; observed.size_override = 1110; observed.full_calls = full_calls;
        observed.rows.assign(1110, Row{});
        teardown::Snapshot<Row> result;
        teardown::collect(result, 7, [&](Row* rows, long bytes) { return observed(rows, bytes); }, tracked_allocate);
        check(result.fills[0].capacity_records == 1126 && result.fills[1].capacity_records == 2048 &&
              result.fills[1].capacity_bytes == 966656 && !result.fills[2].attempted && observed.calls == 3);
        check(result.result == (full_calls == 1 ? Result::complete : Result::unstable_capacity));
        check(peak_bytes == 966656); // The discarded first allocation was freed before growth.
    }
    reset_allocations();
    {
        Query maximum_rows; maximum_rows.size_override = 2032; maximum_rows.rows.assign(2047, Row{});
        teardown::Snapshot<Row> a, b;
        const auto query = [&](Row* rows, long bytes) { return maximum_rows(rows, bytes); };
        teardown::collect(a, 7, query, tracked_allocate);
        teardown::collect(b, 7, query, tracked_allocate);
        check(teardown::same(a, b) && a.count == 2047 && a.matching == 0);
        check(live_bytes == 1933312 && peak_bytes == 1933312);
    }
    reset_allocations();
    for (size_t fail_at : {size_t(1), size_t(2)}) {
        fail_allocate_call = allocate_calls + fail_at;
        Query observed; observed.size_override = 1110; observed.full_calls = 1;
        teardown::Snapshot<Row> result;
        teardown::collect(result, 7, [&](Row* rows, long bytes) { return observed(rows, bytes); }, tracked_allocate);
        check(result.result == Result::allocation_error && !result.rows && live_bytes == 0);
        check(observed.calls == static_cast<int>(fail_at) && !result.fills[fail_at - 1].attempted);
    }
    reset_allocations();
    for (int returned : {2049, INT_MAX}) {
        int calls = 0;
        teardown::Snapshot<Row> result;
        teardown::collect(result, 7, [&](Row* rows, long) {
            ++calls; return rows ? returned : 1110;
        }, tracked_allocate);
        check(result.result == Result::unstable_capacity && calls == 3 && !result.rows && live_bytes == 0);
        check(result.fills[0].returned == returned && result.fills[1].returned == returned);
    }
    reset_allocations();
    for (unsigned mutation = 0; mutation != 6; ++mutation) {
        teardown::Snapshot<Row> result;
        teardown::collect(result, 7, [&](Row* rows, long bytes) {
            if (!rows) return 1110;
            check(bytes == 531472);
            std::fill_n(rows, 1110, Row{});
            // Every row must be validated even when unrelated to our title.
            auto& last = rows[1109];
            switch (mutation) {
            case 0: last.f_version = 0; break;
            case 1: std::memset(last.f_fstypename, 'x', sizeof(last.f_fstypename)); break;
            case 2: std::memset(last.f_mntonname, 'x', sizeof(last.f_mntonname)); break;
            case 3: std::memset(last.f_mntfromname, 'x', sizeof(last.f_mntfromname)); break;
            case 4: std::memset(static_cast<void*>(&last), 0, sizeof(last)); break;
            case 5: // Wrong record stride leaves the expected final version zero.
                std::memset(rows, 0, static_cast<size_t>(bytes));
                for (size_t i = 0; i != 1110; ++i) {
                    Row value;
                    std::memcpy(reinterpret_cast<unsigned char*>(rows) + i * 464, &value, sizeof(value));
                }
                break;
            }
            return 1110;
        }, tracked_allocate);
        check(result.result == Result::malformed && !teardown::same(result, result));
    }
    reset_allocations();

    const teardown::Status absent{-1, ESRCH, -1, -1};
    check(teardown::queried_absence(absent));
    for (const auto& value : {teardown::Status{0, 0, -1, -1}, {-1, EPERM, -1, -1},
                             {-1, ESRCH, 4, -1}, {-1, ESRCH, -1, 4}, {-1, 0, -1, -1}}) {
        check(!teardown::queried_absence(value)); check(!teardown::same(absent, value));
    }
    std::puts("teardown host model passed: bounded snapshots, field/string validation, exact title classification, duplicate layers, status uncertainty; no firmware ABI or release certification");
}
