#include "ps5hip/code_object.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace ps5hip {
namespace {
constexpr uint64_t max_image = 64u * 1024u * 1024u;

void require(bool condition, const char *message) {
    if (!condition) throw ObjectError(message);
}
bool power2(uint64_t n) { return n != 0 && (n & (n - 1)) == 0; }
bool contains(uint64_t begin, uint64_t size, uint64_t at, uint64_t count) {
    return at >= begin && at - begin <= size && count <= size - (at - begin);
}

struct Bytes {
    const uint8_t *data;
    size_t size;
    void check(uint64_t offset, uint64_t count) const {
        require(offset <= size && count <= size - offset, "file range out of bounds");
    }
    uint64_t u(uint64_t offset, unsigned width) const {
        check(offset, width);
        uint64_t value = 0;
        for (unsigned i = 0; i < width; ++i) value |= uint64_t(data[offset + i]) << (i * 8);
        return value;
    }
    std::string text(uint64_t offset, uint64_t count) const {
        check(offset, count);
        return std::string(reinterpret_cast<const char *>(data + offset), size_t(count));
    }
    std::string cstring(uint64_t offset, uint64_t count) const {
        check(offset, count);
        require(count <= max_image, "string table too large");
        const void *end = std::memchr(data + offset, 0, size_t(count));
        require(end != nullptr, "unterminated ELF string");
        auto length = static_cast<const uint8_t *>(end) - (data + offset);
        require(length <= 4096, "ELF symbol name too long");
        return text(offset, uint64_t(length));
    }
};

// Small bounded MessagePack reader for the AMDGPU metadata grammar. Unsupported
// encodings fail explicitly; maps reject duplicate keys before schema validation.
struct Value {
    Value();
    ~Value();
    Value(const Value&);
    Value(Value&&) noexcept;
    Value& operator=(const Value&);
    Value& operator=(Value&&) noexcept;
    enum Type { Integer, String, Array, Map, Boolean, Nil } type = Nil;
    uint64_t number = 0;
    std::string string;
    std::vector<Value> items;
    std::vector<std::pair<std::string, Value>> fields;
    const Value &field(const char *key) const;
    uint32_t integer() const {
        require(type == Integer && number <= UINT32_MAX, "expected uint32 metadata value");
        return uint32_t(number);
    }
    const std::string &text() const {
        require(type == String, "expected metadata string");
        return string;
    }
};

// Instantiate the recursive pair only after Value is complete (Clang/libstdc++).
Value::Value() = default;
Value::~Value() = default;
Value::Value(const Value&) = default;
Value::Value(Value&&) noexcept = default;
Value& Value::operator=(const Value&) = default;
Value& Value::operator=(Value&&) noexcept = default;
const Value &Value::field(const char *key) const {
    require(type == Map, "expected metadata map");
    for (const auto &kv : fields) if (kv.first == key) return kv.second;
    throw ObjectError(std::string("missing metadata field: ") + key);
}

struct Msgpack {
    Bytes bytes;
    uint64_t cursor = 0, nodes = 0;
    uint64_t big(unsigned count) {
        bytes.check(cursor, count);
        uint64_t n = 0;
        while (count--) n = (n << 8) | bytes.data[cursor++];
        return n;
    }
    Value read(unsigned depth = 0) {
        require(depth <= 16 && ++nodes <= 16384, "metadata complexity limit");
        const auto tag = big(1);
        Value v;
        uint64_t count = 0;
        if (tag < 0x80) { v.type = Value::Integer; v.number = tag; return v; }
        if (tag >= 0xa0 && tag <= 0xbf) { v.type = Value::String; count = tag & 31; }
        else if (tag >= 0x90 && tag <= 0x9f) { v.type = Value::Array; count = tag & 15; }
        else if (tag >= 0x80 && tag <= 0x8f) { v.type = Value::Map; count = tag & 15; }
        else if (tag >= 0xcc && tag <= 0xcf) {
            v.type = Value::Integer; v.number = big(1u << (tag - 0xcc)); return v;
        } else if (tag == 0xc2 || tag == 0xc3) {
            v.type = Value::Boolean; v.number = tag == 0xc3; return v;
        } else if (tag == 0xc0) { return v; }
        else if (tag >= 0xd9 && tag <= 0xdb) { v.type = Value::String; count = big(1u << (tag - 0xd9)); }
        else if (tag == 0xdc || tag == 0xdd) { v.type = Value::Array; count = big(tag == 0xdc ? 2 : 4); }
        else if (tag == 0xde || tag == 0xdf) { v.type = Value::Map; count = big(tag == 0xde ? 2 : 4); }
        else throw ObjectError("unsupported MessagePack encoding");
        require(count <= 4096, "metadata container limit");
        if (v.type == Value::String) {
            v.string = bytes.text(cursor, count); cursor += count;
        } else if (v.type == Value::Array) {
            for (uint64_t i = 0; i < count; ++i) v.items.push_back(read(depth + 1));
        } else {
            for (uint64_t i = 0; i < count; ++i) {
                auto key = read(depth + 1).text();
                for (const auto &kv : v.fields) require(kv.first != key, "duplicate metadata key");
                auto value = read(depth + 1);
                v.fields.emplace_back(std::move(key), std::move(value));
            }
        }
        return v;
    }
};

struct Section { uint32_t type, link; uint64_t flags, address, offset, size, entry; };
struct Symbol { std::string name; uint64_t address, size; uint8_t type; uint16_t section; };

uint64_t file_address(const CodeObject &object, uint64_t address, uint64_t size, uint32_t flags) {
    for (const auto &segment : object.segments)
        if (contains(segment.address, segment.file_size, address, size) &&
            (segment.flags & flags) == flags)
            return segment.file_offset + address - segment.address;
    throw ObjectError("address does not belong to a matching file-backed load segment");
}

uint64_t add_signed(uint64_t base, uint64_t encoded) {
    if (encoded >> 63) {
        const uint64_t magnitude = ~encoded + 1;
        require(base >= magnitude, "negative entry offset outside image");
        return base - magnitude;
    }
    require(encoded <= UINT64_MAX - base, "entry address overflow");
    return base + encoded;
}
bool identifier(const std::string &name) {
    if (name.empty() || name.size() > 4096) return false;
    for (unsigned char c : name)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '$')) return false;
    return true;
}
} // namespace

CodeObject inspect_code_object(const uint8_t *data, size_t size) {
    require(data != nullptr && size >= 64 && size <= max_image, "invalid object size");
    Bytes b{data, size};
    require(b.u(0, 4) == 0x464c457f && b.u(4, 1) == 2 && b.u(5, 1) == 1 &&
            b.u(6, 1) == 1 && b.u(7, 1) == 64 && b.u(8, 1) == 3,
            "expected little-endian ELF64 AMDHSA code object V5");
    require(b.u(16, 2) == 3 && b.u(18, 2) == 224 && b.u(20, 4) == 1,
            "expected linked AMDGPU object");
    require(b.u(24, 8) == 0 && b.u(52, 2) == 64, "unsupported ELF header");
    const uint32_t flags = uint32_t(b.u(48, 4));
    const uint32_t machine = flags & 0xff;
    require(machine == 0x42 || machine == 0x36, "unsupported AMDGPU target");
    // XNACK unsupported/ANY/OFF are inspectable; ON requires unavailable fault handling.
    require((flags & ~0x3ffu) == 0 && (flags & 0x300) != 0x300, "unsupported target feature flags");
    CodeObject result;
    result.target = machine == 0x42 ? "gfx1013" : "gfx1030";
    const uint64_t phoff = b.u(32, 8), shoff = b.u(40, 8);
    const uint64_t phnum = b.u(56, 2), shnum = b.u(60, 2);
    require(b.u(54, 2) == 56 && b.u(58, 2) == 64 && phnum > 0 && phnum <= 64 &&
            shnum > 0 && shnum <= 256 && b.u(62, 2) < shnum, "invalid ELF table dimensions");
    b.check(phoff, phnum * 56); b.check(shoff, shnum * 64);
    for (uint64_t i = 0; i < phnum; ++i) {
        const uint64_t p = phoff + i * 56;
        const uint32_t type = uint32_t(b.u(p, 4));
        const uint32_t access = uint32_t(b.u(p + 4, 4));
        const uint64_t offset = b.u(p + 8, 8), address = b.u(p + 16, 8);
        const uint64_t file_size = b.u(p + 32, 8), mem_size = b.u(p + 40, 8), align = b.u(p + 48, 8);
        b.check(offset, file_size);
        require(type == 0 || type == 1 || type == 2 || type == 4 || type == 6 ||
                type == 0x6474e551 || type == 0x6474e552, "unsupported program header");
        require((access & ~7u) == 0, "invalid segment permissions");
        if (type == 2) {
            require(file_size % 16 == 0 && file_size <= 65536, "invalid dynamic table");
            bool terminated = false;
            for (uint64_t pos = offset; pos < offset + file_size; pos += 16) {
                const uint64_t tag = b.u(pos, 8);
                if (tag == 0) { terminated = true; break; }
                require(tag == 4 || tag == 5 || tag == 6 || tag == 10 || tag == 11 || tag == 0x6ffffef5,
                        "dynamic linking or relocation unsupported");
            }
            require(terminated, "unterminated dynamic table");
        }
        if (type != 1) continue;
        require(file_size <= mem_size && mem_size > 0 && address < max_image &&
                mem_size <= max_image - address, "invalid load memory range");
        require(power2(align) && align <= max_image && (address % align) == (offset % align),
                "invalid load alignment");
        require((access & 4) != 0 && (access & 3) != 3, "unreadable or writable executable segment");
        for (const auto &prior : result.segments)
            require(address >= prior.address + prior.memory_size || prior.address >= address + mem_size,
                    "overlapping load segments");
        result.segments.push_back({offset, address, file_size, mem_size, align, access});
        result.image_size = std::max(result.image_size, address + mem_size);
    }
    require(!result.segments.empty(), "no load segments");
    std::vector<Section> sections;
    Value metadata;
    bool found_metadata = false;
    for (uint64_t i = 0; i < shnum; ++i) {
        const uint64_t p = shoff + i * 64;
        Section s{uint32_t(b.u(p + 4, 4)), uint32_t(b.u(p + 40, 4)), b.u(p + 8, 8),
                  b.u(p + 16, 8), b.u(p + 24, 8), b.u(p + 32, 8), b.u(p + 56, 8)};
        require(s.size <= max_image && (s.flags & 0x400) == 0, "oversized or TLS section unsupported");
        if (s.type != 8) b.check(s.offset, s.size);
        require(!((s.type == 4 || s.type == 9 || s.type == 19) && s.size), "relocations unsupported");
        if ((s.flags & 2) && s.size) {
            const uint32_t access = 4u | ((s.flags & 1) ? 2u : 0u) | ((s.flags & 4) ? 1u : 0u);
            if (s.type != 8)
                require(file_address(result, s.address, s.size, access) == s.offset, "section/load mapping mismatch");
            else {
                bool mapped = false;
                for (const auto &segment : result.segments)
                    if (contains(segment.address, segment.memory_size, s.address, s.size) &&
                        (segment.flags & access) == access && s.address >= segment.address + segment.file_size)
                        mapped = true;
                require(mapped, "NOBITS section outside zero-filled load storage");
            }
        }
        sections.push_back(s);
        if (s.type != 7) continue;
        uint64_t pos = s.offset;
        while (pos < s.offset + s.size) {
            require(contains(s.offset, s.size, pos, 12), "truncated ELF note header");
            const uint64_t names = b.u(pos, 4), desc = b.u(pos + 4, 4), type = b.u(pos + 8, 4);
            pos += 12;
            const uint64_t name_padded = (names + 3) & ~uint64_t(3), desc_padded = (desc + 3) & ~uint64_t(3);
            require(contains(s.offset, s.size, pos, name_padded + desc_padded), "truncated ELF note body");
            if (names == 7 && b.text(pos, 7) == std::string("AMDGPU\0", 7) && type == 32) {
                require(!found_metadata && desc <= 1024 * 1024, "duplicate or excessive metadata");
                Msgpack parser{{data + pos + name_padded, size_t(desc)}};
                metadata = parser.read();
                require(parser.cursor == desc, "trailing MessagePack bytes");
                found_metadata = true;
            }
            pos += name_padded + desc_padded;
        }
    }
    require(found_metadata, "AMDGPU metadata missing");
    const auto &version = metadata.field("amdhsa.version");
    require(version.type == Value::Array && version.items.size() == 2 &&
            version.items[0].integer() == 1 && version.items[1].integer() == 2, "unsupported metadata version");
    require(metadata.field("amdhsa.target").text() == "amdgcn-amd-amdhsa--" + result.target,
            "metadata target mismatch");
    std::vector<Symbol> symbols;
    bool dynsym_found = false;
    for (const auto &s : sections) {
        if (s.type != 11) continue;
        require(!dynsym_found && s.entry == 24 && s.size % 24 == 0 && s.link < sections.size(),
                "invalid dynamic symbol table");
        dynsym_found = true;
        const auto &strings = sections[s.link];
        require(strings.type == 3, "symbol names not a string table");
        for (uint64_t p = s.offset; p < s.offset + s.size; p += 24) {
            const uint64_t name = b.u(p, 4);
            require(name < strings.size, "symbol name offset out of bounds");
            auto text = b.cstring(strings.offset + name, strings.size - name);
            uint16_t section = uint16_t(b.u(p + 6, 2));
            require(text.empty() || (section > 0 && section < sections.size()), "unresolved or special symbol unsupported");
            if (!text.empty()) {
                const auto &owner = sections[section];
                require((owner.flags & 2) && contains(owner.address, owner.size, b.u(p + 8, 8), b.u(p + 16, 8)),
                        "dynamic symbol outside allocated owner section");
                for (const auto &prior : symbols) require(prior.name != text, "duplicate dynamic symbol");
                symbols.push_back({std::move(text), b.u(p + 8, 8), b.u(p + 16, 8), uint8_t(b.u(p + 4, 1) & 15), section});
            }
        }
    }
    require(dynsym_found, "dynamic symbol table missing");
    const auto &kernels = metadata.field("amdhsa.kernels");
    require(kernels.type == Value::Array && !kernels.items.empty() && kernels.items.size() <= 128,
            "invalid kernel inventory");
    for (const auto &item : kernels.items) {
        Kernel k{};
        k.name = item.field(".name").text();
        require(identifier(k.name), "unsupported kernel name");
        for (const auto &prior : result.kernels) require(prior.name != k.name, "duplicate kernel");
        const auto &symbol = item.field(".symbol").text();
        const Symbol *descriptor = nullptr;
        for (const auto &s : symbols) if (s.name == symbol) descriptor = &s;
        require(descriptor && descriptor->type == 1 && descriptor->size == 64, "invalid kernel descriptor symbol");
        k.descriptor_address = descriptor->address;
        require(k.descriptor_address % 64 == 0, "misaligned kernel descriptor");
        const uint64_t d = file_address(result, k.descriptor_address, 64, 4);
        for (unsigned i = 0; i < 64; ++i)
            if ((i >= 12 && i < 16) || (i >= 24 && i < 44) || i >= 60)
                require(b.u(d + i, 1) == 0, "reserved kernel descriptor field");
        k.group_bytes = uint32_t(b.u(d, 4)); k.private_bytes = uint32_t(b.u(d + 4, 4));
        k.kernarg_size = uint32_t(b.u(d + 8, 4));
        k.entry_address = add_signed(k.descriptor_address, b.u(d + 16, 8));
        require(k.entry_address % 256 == 0, "misaligned kernel entry");
        file_address(result, k.entry_address, 4, 5);
        const Symbol *function = nullptr;
        for (const auto &s : symbols) if (s.name == k.name) function = &s;
        require(function && function->type == 2 && function->address == k.entry_address && function->size >= 4,
                "kernel entry/function mismatch");
        file_address(result, function->address, function->size, 5);
        k.rsrc3 = uint32_t(b.u(d + 44, 4)); k.rsrc1 = uint32_t(b.u(d + 48, 4));
        k.rsrc2 = uint32_t(b.u(d + 52, 4)); k.properties = uint16_t(b.u(d + 56, 2));
        k.preload = uint16_t(b.u(d + 58, 2));
        k.kernarg_alignment = item.field(".kernarg_segment_align").integer();
        k.max_workgroup_size = item.field(".max_flat_workgroup_size").integer();
        k.wave_size = item.field(".wavefront_size").integer();
        k.sgpr_spills = item.field(".sgpr_spill_count").integer();
        k.vgpr_spills = item.field(".vgpr_spill_count").integer();
        for (const auto &field : item.fields) if (field.first == ".reqd_workgroup_size") {
            require(field.second.type == Value::Array && field.second.items.size() == 3,
                    "invalid required workgroup shape");
            uint64_t threads = 1;
            for (unsigned axis = 0; axis < 3; ++axis) {
                k.required_workgroup[axis] = field.second.items[axis].integer();
                require(k.required_workgroup[axis] > 0 && k.required_workgroup[axis] <= 1024,
                        "invalid required workgroup dimension");
                threads *= k.required_workgroup[axis];
            }
            require(threads <= k.max_workgroup_size, "required workgroup exceeds kernel limit");
        }
        require(k.kernarg_size == item.field(".kernarg_segment_size").integer() &&
                k.private_bytes == item.field(".private_segment_fixed_size").integer() &&
                k.group_bytes == item.field(".group_segment_fixed_size").integer(), "descriptor/metadata size mismatch");
        require(k.kernarg_size <= 65536 && power2(k.kernarg_alignment) && k.kernarg_alignment <= 4096 &&
                k.group_bytes <= 65536 && k.private_bytes <= max_image &&
                k.max_workgroup_size > 0 && k.max_workgroup_size <= 1024, "invalid kernel resource limits");
        require((k.wave_size == 32 || k.wave_size == 64) &&
                bool(k.properties & 0x400) == (k.wave_size == 32), "descriptor/metadata wave mismatch");
        const auto &args = item.field(".args");
        require(args.type == Value::Array && args.items.size() <= 256, "invalid argument inventory");
        for (const auto &arg : args.items) {
            KernelArgument a{arg.field(".offset").integer(), arg.field(".size").integer(),
                             arg.field(".value_kind").text()};
            require(a.size > 0 && contains(0, k.kernarg_size, a.offset, a.size), "argument outside kernarg storage");
            for (const auto *inventory : {&k.arguments, &k.hidden_arguments})
                for (const auto &prior : *inventory)
                    require(uint64_t(a.offset) >= uint64_t(prior.offset) + prior.size ||
                            uint64_t(prior.offset) >= uint64_t(a.offset) + a.size, "overlapping kernel arguments");
            auto &inventory = a.kind.compare(0, 7, "hidden_") == 0 ? k.hidden_arguments : k.arguments;
            inventory.push_back(std::move(a));
        }
        result.kernels.push_back(std::move(k));
    }
    return result;
}

std::vector<uint8_t> materialize_code_object(const uint8_t *data, size_t size) {
    const auto object = inspect_code_object(data, size);
    std::vector<uint8_t> image(size_t(object.image_size), 0);
    for (const auto &segment : object.segments)
        std::memcpy(image.data() + segment.address, data + segment.file_offset, size_t(segment.file_size));
    return image;
}

static std::string launch_rejection(const Kernel &k, bool scratch, bool lds = false, bool flat_call = false, bool geometry = false, bool reduction = false) {
    // Preserve the original envelope; weighted RMSNorm adds one measured tuple.
    const bool weighted_norm = k.rsrc1 == 0x60af0184u && k.rsrc2 == 0x38cu &&
        k.kernarg_size == 488 && k.kernarg_alignment == 8 && k.max_workgroup_size == 1024;
    const bool original_reduction = (k.rsrc1 & 0xfffffc00u) == 0x60af0000u &&
        (k.rsrc1 & 0x3fu) <= 3 && ((k.rsrc1 >> 6) & 0xfu) <= 3;
    if (reduction && ((!original_reduction && !weighted_norm) || k.group_bytes || k.max_workgroup_size < 1024 ||
        k.sgpr_spills || k.vgpr_spills)) return "unsupported reduction resources";
    if (k.wave_size != 32) return "initial launch supports wave32 only";
    if (flat_call) {
        if (!k.private_bytes || k.properties != 0x429 || k.rsrc2 != 0x91 || !k.hidden_arguments.empty())
            return "unsupported experimental flat-call entry contract";
    } else if (scratch) {
        if (!k.private_bytes || k.properties != 0x409 || k.rsrc2 != 0x8d || !k.hidden_arguments.empty())
            return "unsupported buffer-scratch entry contract";
    } else if (k.private_bytes || (k.rsrc2 & 1) || (k.properties & (0x20 | 0x40 | 0x800)))
        return "private storage requires a qualified scratch implementation";
    const bool getrows = geometry && !lds && k.rsrc2 == 0x38cu && k.rsrc1 == 0x60af01c1u;
    if (getrows) {
        if (k.kernarg_size != 384 || k.kernarg_alignment != 8 || k.max_workgroup_size != 1024 ||
            k.arguments.size() != 15 || k.hidden_arguments.size() != 13)
            return "unsupported embedding geometry ABI";
        constexpr std::array<uint32_t,15> offsets{0,8,16,24,32,40,56,64,72,80,88,96,104,112,120};
        for (size_t i=0;i<offsets.size();++i)
            if (k.arguments[i].offset != offsets[i] || k.arguments[i].size != (i==5 ? 12u : 8u) ||
                k.arguments[i].kind != (i<3 ? "global_buffer" : "by_value"))
                return "unsupported embedding argument layout";
    }
    const bool q8 = geometry && !lds && k.rsrc2 == 0x38cu && !getrows;
    const bool q4 = geometry && lds && k.rsrc2 == 0xb8cu;
    if ((q8 || q4) && ((q8 ? k.rsrc1 != 0x60af00c1u : (k.rsrc1 != 0x60af00c3u && k.rsrc1 != 0x60af00c4u)) ||
        k.kernarg_size != (q8 ? 328u : 192u) || k.kernarg_alignment != 8 ||
        k.max_workgroup_size != (q8 ? 256u : 128u) ||
        k.hidden_arguments.size() != (q8 ? 13u : 0u))) return "unsupported measured quantized geometry ABI";
    if (lds) {
        if (k.group_bytes > 65536 || k.properties != 0x409 || (reduction ? (k.rsrc2 != 0x8cu && k.rsrc2 != 0x38cu) : k.rsrc2 != (geometry ? (q4 ? 0xb8cu : 0x98cu) : 0x8cu)) ||
            (geometry ? (k.group_bytes != (q4 ? 384u : 512u) || (!q4 && k.hidden_arguments.empty())) : (!reduction && !k.hidden_arguments.empty())))
            return "unsupported LDS entry contract";
    } else if (k.group_bytes || (k.rsrc2 & 0x00ff8000)) return "LDS requires the G3 launch path";
    if (k.preload || k.rsrc3) return "preload/resource3 requires additional launch support";
    if ((k.properties & ~uint16_t(flat_call ? 0x429 : 0x409)) || !(k.properties & 8))
        return "unsupported initial user-SGPR contract";
    const uint32_t user_sgprs = (k.rsrc2 >> 1) & 31;
    if (user_sgprs != (flat_call ? 8u : ((k.properties & 1) ? 6u : 2u))) return "user-SGPR count mismatch";
    if (geometry) {
        if (k.rsrc2 != (lds ? (q4 ? 0xb8cu : 0x98cu) : ((q8 || getrows) ? 0x38cu : 0x138cu)) || k.properties != 0x409 || (!q4 && k.hidden_arguments.empty()) || k.sgpr_spills || k.vgpr_spills)
            return "unsupported measured XYZ entry contract";
    } else if ((k.rsrc2 & ~(reduction ? 0x3beu : scratch ? 0xbfu : 0xbeu)) || !(k.rsrc2 & 0x80))
        return "unsupported system-SGPR/work-item contract";
    if (k.rsrc1 & ((1u << 20) | (1u << 22) | (1u << 25))) return "privileged/debug execution unsupported";
    uint64_t explicit_end = 0;
    for (size_t i = 0; i < k.arguments.size(); ++i) {
        const auto &arg = k.arguments[i];
        if (!((arg.kind == "global_buffer" && arg.size == 8 && arg.offset % 8 == 0) ||
              (arg.kind == "by_value" &&
               (((arg.size == 4 || arg.size == 8) && arg.offset % arg.size == 0) ||
                (arg.size > 8 && arg.size <= detail::max_by_value_bytes)))))
            return "unsupported initial argument kind or layout";
        if (!contains(0, k.kernarg_size, arg.offset, arg.size)) return "argument outside kernarg storage";
        for (size_t j = 0; j < i; ++j)
            if (uint64_t(arg.offset) < uint64_t(k.arguments[j].offset) + k.arguments[j].size &&
                uint64_t(k.arguments[j].offset) < uint64_t(arg.offset) + arg.size)
                return "overlapping kernel arguments";
        explicit_end = std::max(explicit_end, uint64_t(arg.offset) + arg.size);
    }
    const uint64_t base = (explicit_end + 7) & ~uint64_t(7);
    // LLVM rounds explicit-only storage to four bytes, including aggregate tails.
    if (k.hidden_arguments.empty() && k.kernarg_size != ((explicit_end + 3) & ~uint64_t(3)))
        return "unaccounted initial kernarg storage";
    if (!k.hidden_arguments.empty()) {
        if (k.hidden_arguments.size() != detail::cov5_geometry.size() + (reduction ? 1u : 0u) ||
            k.kernarg_size != base + 256 || k.kernarg_size > 65536 ||
            !power2(k.kernarg_alignment) || k.kernarg_alignment < 8 || k.kernarg_alignment > 4096)
            return "unsupported COv5 implicit geometry storage";
        std::array<bool, 13> seen{};
        bool dynamic_seen = false;
        for (const auto &arg : k.hidden_arguments) {
            if (reduction && arg.kind == "hidden_dynamic_lds_size") {
                if (dynamic_seen || arg.offset != base + 120 || arg.size != 4) return "invalid dynamic LDS argument";
                dynamic_seen = true;
                continue;
            }
            const auto field = std::find_if(detail::cov5_geometry.begin(), detail::cov5_geometry.end(),
                                           [&](const auto &item) { return arg.kind == item.kind; });
            if (field == detail::cov5_geometry.end()) return "unsupported hidden argument kind";
            const size_t index = size_t(field - detail::cov5_geometry.begin());
            if (seen[index] || arg.offset != base + field->offset || arg.size != field->size)
                return "invalid COv5 hidden geometry layout";
            seen[index] = true;
        }
    }
    return {};
}

std::string initial_launch_rejection(const Kernel &k) { return launch_rejection(k, false); }
std::string buffer_scratch_launch_rejection(const Kernel &k) { return launch_rejection(k, true); }

std::string lds_launch_rejection(const Kernel &k) { return launch_rejection(k, false, true); }
std::string reduction_lds_launch_rejection(const Kernel &k) {
    if (k.hidden_arguments.empty()) return "unsupported reduction ABI";
    return launch_rejection(k, false, true, false, false, true);
}

std::string flat_call_launch_rejection(const Kernel &k) { return launch_rejection(k, true, false, true); }

std::string geometry_launch_rejection(const Kernel &k) { return launch_rejection(k, false, false, false, true); }
std::string geometry_lds_launch_rejection(const Kernel &k) { return launch_rejection(k, false, true, false, true); }

} // namespace ps5hip
