// The DXIL program inside a container's `DXIL` (or `STAT`) part.
#include "x2w_shader/dxil.h"

#include <cstring>

namespace x2w::vs::dxil {

namespace {
constexpr std::uint8_t LLVM_BITCODE_MAGIC[4] = {0x42, 0x43, 0xC0, 0xDE};
constexpr std::uint32_t STANDARD_BITCODE_OFFSET = 16;
}  // namespace

ShaderKind shader_kind_from_raw(std::uint16_t v) { return static_cast<ShaderKind>(v); }

std::string debug(ShaderKind k) {
    static const char* const NAMES[16] = {
        "Pixel", "Vertex", "Geometry", "Hull", "Domain", "Compute", "Library", "RayGeneration",
        "Intersection", "AnyHit", "ClosestHit", "Miss", "Callable", "Mesh", "Amplification", "Node",
    };
    auto v = static_cast<std::uint16_t>(k);
    if (v < 16) return NAMES[v];
    return "Unknown(" + std::to_string(v) + ")";
}

DxilProgram DxilProgram::parse(Bytes part) {
    std::uint32_t version = dxbc::u32_at(part, 0);
    std::size_t size = std::size_t(dxbc::u32_at(part, 4)) * 4;
    if (!(size <= part.size() && size >= 24)) {
        throw ShaderError::malformed("program size disagrees with part size");
    }
    part = part.first(size);
    if (std::memcmp(part.data() + 8, "DXIL", 4) != 0) {
        throw ShaderError::malformed("missing DXIL magic");
    }
    std::uint32_t dxil_version = dxbc::u32_at(part, 12);
    std::uint32_t offset = dxbc::u32_at(part, 16);
    std::size_t len = dxbc::u32_at(part, 20);
    std::size_t start = 8 + std::size_t(offset);
    if (start + len > part.size()) {
        throw ShaderError::malformed("bitcode out of bounds");
    }
    Bytes bitcode = part.subspan(start, len);
    if (bitcode.size() < 4 || std::memcmp(bitcode.data(), LLVM_BITCODE_MAGIC, 4) != 0) {
        throw ShaderError::malformed("bitcode lacks LLVM magic");
    }
    if (start < 24) {
        throw ShaderError::malformed("bitcode offset too small");
    }
    DxilProgram p;
    p.kind = shader_kind_from_raw(std::uint16_t(version >> 16));
    p.shader_model = {std::uint8_t((version >> 4) & 0xF), std::uint8_t(version & 0xF)};
    p.dxil_version = dxil_version;
    p.bitcode = bitcode;
    p.prefix = part.subspan(24, start - 24);
    p.trailer = part.subspan(start + len);
    return p;
}

bool DxilProgram::is_standard_layout() const { return prefix.empty() && trailer.empty(); }

XboxExtras DxilProgram::xbox_extras() const {
    XboxExtras x;
    x.xdat = prefix.size() >= 4 && std::memcmp(prefix.data(), "XDAT", 4) == 0;
    for (std::size_t i = 0; i + 4 <= trailer.size(); ++i) {
        if (std::memcmp(trailer.data() + i, "xbop", 4) == 0) {
            x.xbop = true;
            break;
        }
    }
    return x;
}

std::vector<std::uint8_t> build_standard(std::uint16_t kind, std::pair<std::uint8_t, std::uint8_t> sm,
                                         std::uint32_t dxil_version, Bytes bitcode) {
    std::vector<std::uint8_t> out;
    auto put32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
    };
    std::uint32_t version = std::uint32_t(kind) << 16 | std::uint32_t(sm.first) << 4 | std::uint32_t(sm.second);
    std::size_t padded = (bitcode.size() + 3) / 4 * 4;
    std::size_t size_dwords = (24 + padded) / 4;
    put32(version);
    put32(std::uint32_t(size_dwords));
    out.insert(out.end(), {'D', 'X', 'I', 'L'});
    put32(dxil_version);
    put32(STANDARD_BITCODE_OFFSET);
    put32(std::uint32_t(bitcode.size()));
    out.insert(out.end(), bitcode.begin(), bitcode.end());
    out.resize(24 + padded, 0);
    return out;
}

}  // namespace x2w::vs::dxil
