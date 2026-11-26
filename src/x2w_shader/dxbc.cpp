// The DXBC container.
#include "x2w_shader/dxbc.h"

#include <cstring>

namespace x2w::vs::dxbc {

namespace {
constexpr std::size_t HEADER_SIZE = 0x20;
constexpr std::size_t MAX_PARTS = 256;
}  // namespace

std::string FourCC::to_string() const {
    std::string s;
    for (std::uint8_t c : b) {
        // is_ascii_graphic: '!'..='~'
        s += (c >= 0x21 && c <= 0x7E) ? char(c) : '.';
    }
    return s;
}

Container Container::parse(Bytes data) {
    if (data.size() < HEADER_SIZE || std::memcmp(data.data(), "DXBC", 4) != 0) {
        throw ShaderError::not_dxbc();
    }
    Container c;
    std::memcpy(c.digest.data(), data.data() + 4, 16);
    std::uint16_t major = u16_at(data, 0x14);
    std::uint16_t minor = u16_at(data, 0x16);
    c.version = {major, minor};
    std::size_t total = u32_at(data, 0x18);
    if (total > data.size() || total < HEADER_SIZE) {
        throw ShaderError::malformed("container size disagrees with data");
    }
    data = data.first(total);
    std::size_t count = u32_at(data, 0x1C);
    if (count > MAX_PARTS) {
        throw ShaderError::malformed("implausible part count");
    }
    c.parts.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t offset = u32_at(data, HEADER_SIZE + 4 * i);
        // data.get(offset..offset + 4): offset + 4 cannot overflow a usize from a u32 here.
        if (offset + 4 > data.size()) {
            throw ShaderError::malformed("part header out of bounds");
        }
        FourCC fourcc;
        std::memcpy(fourcc.b.data(), data.data() + offset, 4);
        std::size_t size = u32_at(data, offset + 4);
        if (offset + 8 + size > data.size()) {
            throw ShaderError::malformed("part data out of bounds");
        }
        c.parts.push_back(Part{fourcc, offset, data.subspan(offset + 8, size)});
    }
    return c;
}

const Part* Container::part(FourCC fourcc) const {
    for (const Part& p : parts) {
        if (p.fourcc == fourcc) return &p;
    }
    return nullptr;
}

std::uint16_t u16_at(Bytes data, std::size_t at) {
    if (at > data.size() || data.size() - at < 2) throw ShaderError::malformed("truncated");
    return std::uint16_t(data[at] | data[at + 1] << 8);
}

std::uint32_t u32_at(Bytes data, std::size_t at) {
    if (at > data.size() || data.size() - at < 4) throw ShaderError::malformed("truncated");
    return std::uint32_t(data[at]) | std::uint32_t(data[at + 1]) << 8 | std::uint32_t(data[at + 2]) << 16 |
           std::uint32_t(data[at + 3]) << 24;
}

std::uint64_t u64_at(Bytes data, std::size_t at) {
    if (at > data.size() || data.size() - at < 8) throw ShaderError::malformed("truncated");
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; --i) v = v << 8 | data[at + i];
    return v;
}

std::vector<std::uint8_t> build(const std::vector<std::pair<FourCC, Bytes>>& parts) {
    std::size_t table = HEADER_SIZE + 4 * parts.size();
    std::size_t total = table;
    for (const auto& p : parts) total += 8 + p.second.size();
    std::vector<std::uint8_t> out;
    out.reserve(total);
    auto put32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
    };
    out.insert(out.end(), {'D', 'X', 'B', 'C'});
    out.insert(out.end(), 16, 0);
    out.insert(out.end(), {1, 0, 0, 0});  // version 1.0
    put32(std::uint32_t(total));
    put32(std::uint32_t(parts.size()));
    std::size_t offset = table;
    for (const auto& p : parts) {
        put32(std::uint32_t(offset));
        offset += 8 + p.second.size();
    }
    for (const auto& p : parts) {
        out.insert(out.end(), p.first.b.begin(), p.first.b.end());
        put32(std::uint32_t(p.second.size()));
        out.insert(out.end(), p.second.begin(), p.second.end());
    }
    return out;
}

}  // namespace x2w::vs::dxbc
