#include "ps5hip/fat_binary.h"

#include <cstring>
#include <utility>

namespace ps5hip {
namespace {
constexpr char magic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr size_t max_bundle = 64u * 1024u * 1024u;
struct Entry { uint64_t offset, size; std::string target; };

void require(bool condition, const char *message) {
    if (!condition) throw ObjectError(message);
}
} // namespace

BundledCodeObject read_fat_binary(const uint8_t *data, size_t size,
                                 const std::string &target) {
    require(target == "gfx1013" || target == "gfx1030", "unsupported fat-binary target");
    require(data && size >= 32 && size <= max_bundle, "invalid fat-binary size");
    require(std::memcmp(data, magic, sizeof(magic) - 1) == 0,
            "expected uncompressed Clang offload bundle");
    size_t cursor = sizeof(magic) - 1;
    auto number = [&]() {
        require(cursor <= size && size - cursor >= 8, "truncated fat-binary directory");
        uint64_t value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= uint64_t(data[cursor++]) << (i * 8);
        return value;
    };
    const uint64_t count = number();
    require(count && count <= 64, "fat-binary entry limit");
    std::vector<Entry> entries;
    for (uint64_t i = 0; i < count; ++i) {
        Entry entry{};
        entry.offset = number(); entry.size = number();
        const auto length = number();
        require(length && length <= 256 && length <= size - cursor,
                "invalid fat-binary target length");
        entry.target.assign(reinterpret_cast<const char *>(data + cursor), size_t(length));
        cursor += size_t(length);
        for (unsigned char c : entry.target)
            require(c >= 0x21 && c <= 0x7e, "invalid fat-binary target identifier");
        for (const auto &prior : entries)
            require(prior.target != entry.target, "duplicate fat-binary target");
        require(entry.offset <= size && entry.size <= size - entry.offset,
                "fat-binary payload outside input");
        entries.push_back(std::move(entry));
    }
    const Entry *selected = nullptr;
    const std::string wanted = "hipv4-amdgcn-amd-amdhsa--" + target;
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto &entry = entries[i];
        require(entry.offset >= cursor, "fat-binary payload overlaps directory");
        if (entry.size) for (size_t j = 0; j < i; ++j) {
            const auto &prior = entries[j];
            require(!prior.size || entry.offset >= prior.offset + prior.size ||
                    prior.offset >= entry.offset + entry.size, "overlapping fat-binary payloads");
        }
        if (entry.target == wanted) selected = &entry;
    }
    require(selected && selected->size, "no exact device target in fat binary");
    BundledCodeObject result;
    result.bundle_target = selected->target;
    result.bytes.assign(data + selected->offset, data + selected->offset + selected->size);
    result.object = inspect_code_object(result.bytes.data(), result.bytes.size());
    require(result.object.target == target, "fat-binary target disagrees with device ELF");
    return result;
}

} // namespace ps5hip
