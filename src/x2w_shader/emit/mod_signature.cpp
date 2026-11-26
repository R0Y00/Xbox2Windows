// Read shader signatures and vertex output interpolation.

#include <bit>
#include <string>

#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/signature.h"
#include "x2w_shader/error.h"

namespace x2w::vs {

void vertex_output_interpolation(std::span<const std::uint8_t> data, std::span<container::SigElement> outputs) {
    const ShaderError TRUNCATED = ShaderError::malformed("truncated vertex pipeline validation data");
    auto take = [&](std::size_t len) {
        if (len > data.size()) throw TRUNCATED;
        auto head = data.first(len);
        data = data.subspan(len);
        return head;
    };
    auto size = [&]() {
        auto b = take(4);
        return std::size_t(std::uint32_t(b[0]) | std::uint32_t(b[1]) << 8 | std::uint32_t(b[2]) << 16 |
                           std::uint32_t(b[3]) << 24);
    };
    auto mul = [&](std::size_t a, std::size_t b) {
        if (b != 0 && a > SIZE_MAX / b) throw TRUNCATED;
        return a * b;
    };
    const std::size_t info_size = size();
    const auto info = take(info_size);
    // PSVRuntimeInfo0 has no signature. PSVRuntimeInfo1 and later share these stage and signature
    // count fields.
    if (info.size() < 36 || info[24] != 1 || info[29] == 0) return;
    const std::size_t resources = size();
    if (resources != 0) {
        const std::size_t stride = size();
        take(mul(resources, stride));
    }
    const std::size_t strings = size();
    take(strings);
    const std::size_t indices = size();
    take(mul(indices, 4));
    const std::size_t stride = size();
    if (stride < 16) throw TRUNCATED;
    const std::size_t count = std::size_t(info[28]) + std::size_t(info[29]);
    const auto elements = take(mul(count, stride));
    for (std::size_t r = std::size_t(info[28]); r < count; ++r) {
        const auto record = elements.subspan(r * stride, stride);
        const std::uint8_t packed = record[10];
        if ((packed & 0x40) == 0) continue;
        if (record[13] > 7) throw ShaderError::malformed("invalid vertex output interpolation");
        const std::uint32_t first = record[9];
        const std::uint32_t last = first + record[8];
        for (container::SigElement& output : outputs) {
            if (output.row && *output.row >= first && *output.row < last && output.col == ((packed >> 4) & 3) &&
                output.cols == (packed & 15) && output.kind == record[11])
                output.interp = record[13];
        }
    }
}

std::vector<container::SigElement> signature(std::span<const std::uint8_t> part) {
    using dxbc::u32_at;
    const std::size_t count = u32_at(part, 0);
    const std::size_t base = u32_at(part, 4);
    std::vector<container::SigElement> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t at = base + 32 * i;
        const std::size_t name_at = u32_at(part, at + 4);
        if (name_at > part.size()) throw ShaderError::malformed("signature name out of bounds");
        std::size_t end = name_at;
        while (end < part.size() && part[end] != 0) ++end;
        const std::string name(part.begin() + std::ptrdiff_t(name_at), part.begin() + std::ptrdiff_t(end));
        if (at + 24 >= part.size()) throw ShaderError::malformed("truncated");
        const std::uint8_t mask = part[at + 24];
        if (u32_at(part, at + 28) != 0) throw ShaderError::unsupported(0, "a minimum-precision signature element");
        if (u32_at(part, at) != 0) throw ShaderError::unsupported(0, "a signature element of another stream");
        std::uint8_t kind;
        switch (u32_at(part, at + 12)) {
        case 0: kind = 0; break;
        case 1: kind = 3; break;
        case 2: kind = 6; break;
        case 3: kind = 7; break;
        case 4: kind = 4; break;
        case 5: kind = 5; break;
        case 6: kind = 1; break;
        case 7: kind = 10; break;
        case 8: kind = 2; break;
        case 9: kind = 13; break;
        case 10: kind = 12; break;
        case 64: kind = 16; break;
        case 65: kind = 17; break;
        case 66: kind = 14; break;
        case 67: kind = 19; break;
        case 68: kind = 18; break;
        case 69: kind = 20; break;
        case 70: kind = 15; break;
        case 23: kind = 28; break;  // barycentrics
        case 25: kind = 30; break;  // cull primitive
        default: throw ShaderError::unsupported(0, "a system value in the signature");
        }
        // 16-bit elements are declared 32-bit; reading or writing one converts (16-bit values
        // travel packed on the console).
        const std::uint32_t raw = u32_at(part, at + 16);
        std::uint8_t comp;
        if (kind == 13) {
            // DXIL declares the front face a uint (the validator insists).
            comp = 5;
        } else if (raw == 1 || raw == 4) {
            comp = 5;
        } else if (raw == 2 || raw == 5) {
            comp = 4;
        } else if (raw == 3 || raw == 6) {
            comp = 9;
        } else {
            throw ShaderError::unsupported(0, raw == 0 ? "a signature element of unknown type" : "a 64-bit signature element");
        }
        const std::uint32_t reg = u32_at(part, at + 20);
        container::SigElement e;
        e.name = name;
        e.index = u32_at(part, at + 8);
        e.kind = kind;
        e.comp = comp;
        e.interp = 0;
        if (reg != UINT32_MAX) e.row = reg;
        e.col = std::uint8_t(std::min(std::countr_zero(std::uint8_t(mask)), 3));
        e.cols = std::uint8_t(std::max(std::popcount(std::uint8_t(mask)), 1));
        e.usage = 0;
        e.half = raw >= 4 && raw <= 6;
        out.push_back(std::move(e));
    }
    return out;
}

}  // namespace x2w::vs
