#pragma once

#include <d3d12.h>
#include <cstdint>
#include <cstring>

namespace x2w_late_texture {
// A copied title descriptor must retain every field needed to rebuild its SRV.
// Array markers use the high bit of word 5, unused by D3D12 component mapping.
constexpr std::uint32_t magic = 0x54573258; // X2WT
constexpr std::uint32_t lazy = 0x4c573258; // X2WL
inline bool marker(std::uint32_t word) { return word == magic || word == lazy; }
inline bool texture_format(DXGI_FORMAT f)
{
    // Standard DXGI values plus console formats 116–120 translated by desktop_format().
    const UINT n = static_cast<UINT>(f);
    return (n >= 1 && n <= 120) || (n >= 130 && n <= 132) || (n >= 189 && n <= 191);
}
inline bool valid_shape(const D3D12_RESOURCE_DESC& r, bool allow_full_chain = false)
{
    const bool volume = r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    if (!volume && r.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return false;
    const UINT limit = volume ? D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION : D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION;
    const UINT slices = volume ? limit : D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
    if (!r.Width || r.Width > limit || !r.Height || r.Height > limit || !r.DepthOrArraySize ||
        r.DepthOrArraySize > slices || (!r.MipLevels && !allow_full_chain) || !texture_format(r.Format)) return false;
    UINT largest = r.Width > r.Height ? static_cast<UINT>(r.Width) : r.Height;
    if (volume && r.DepthOrArraySize > largest) largest = r.DepthOrArraySize;
    UINT levels = 1;
    while (largest >>= 1) ++levels;
    return r.MipLevels <= levels;
}
inline bool encode(const D3D12_RESOURCE_DESC& r, const D3D12_SHADER_RESOURCE_VIEW_DESC& v,
                   std::uint64_t address, std::uint32_t (&w)[8])
{
    const bool volume = r.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D && v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE3D;
    const bool cube = v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURECUBE;
    if (!address || !valid_shape(r) || (!volume && r.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) || r.Alignment ||
        !r.Width || r.Width > 65535 || !r.Height || r.Height > 65535 || !r.DepthOrArraySize ||
        (v.Format != DXGI_FORMAT_UNKNOWN && !texture_format(v.Format)) ||
        unsigned(r.Layout) < 0x100 || unsigned(r.Layout) > 0x1ff || unsigned(r.Flags) != 0x300000 ||
        r.SampleDesc.Count != 1 || r.SampleDesc.Quality || v.Shader4ComponentMapping > 0x1fff)
        return false;
    const bool array = v.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
    if (array) {
        const auto& a = v.Texture2DArray;
        if (r.DepthOrArraySize > 2048 || a.PlaneSlice || a.FirstArraySlice >= r.DepthOrArraySize ||
            !a.ArraySize || a.ArraySize > r.DepthOrArraySize - a.FirstArraySlice ||
            a.MostDetailedMip > 255 || (!a.MipLevels || (a.MipLevels >= 511 && a.MipLevels != UINT32_MAX))) return false;
    } else if (volume || cube) {
        const UINT most = volume ? v.Texture3D.MostDetailedMip : v.TextureCube.MostDetailedMip;
        const UINT levels = volume ? v.Texture3D.MipLevels : v.TextureCube.MipLevels;
        if ((cube && (r.DepthOrArraySize != 6 || r.Width != r.Height)) ||
            most > 65535 || !levels || (levels >= 65535 && levels != UINT32_MAX)) return false;
    } else if (r.DepthOrArraySize != 1 || v.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D ||
               v.Texture2D.PlaneSlice || v.Texture2D.MostDetailedMip > 65535 ||
               (v.Texture2D.MipLevels > 65535 && v.Texture2D.MipLevels != UINT32_MAX)) return false;
    w[0] = magic;
    w[1] = std::uint32_t(address);
    w[2] = std::uint32_t(address >> 32);
    w[3] = std::uint32_t(r.Width) | (r.Height << 16);
    w[4] = unsigned(r.Format) | (unsigned(v.Format) << 8) | ((unsigned(r.Layout) & 255) << 16) |
           (unsigned(r.MipLevels) << 24);
    if (array) {
        const auto& a = v.Texture2DArray;
        const std::uint64_t packed = (1ull << 63) | v.Shader4ComponentMapping |
            (std::uint64_t(r.DepthOrArraySize - 1) << 13) | (std::uint64_t(a.FirstArraySlice) << 24) |
            (std::uint64_t(a.ArraySize - 1) << 35) | (std::uint64_t(a.MostDetailedMip) << 46) |
            (std::uint64_t(a.MipLevels == UINT32_MAX ? 511 : a.MipLevels) << 54);
        w[5] = std::uint32_t(packed >> 32);
        w[6] = std::uint32_t(packed);
        std::memcpy(w + 7, &a.ResourceMinLODClamp, 4);
    } else if (volume || cube) {
        const UINT most = volume ? v.Texture3D.MostDetailedMip : v.TextureCube.MostDetailedMip;
        const UINT levels = volume ? v.Texture3D.MipLevels : v.TextureCube.MipLevels;
        w[5] = (1u << 30) | (unsigned(volume) << 29) | (unsigned(r.DepthOrArraySize) << 13) | v.Shader4ComponentMapping;
        w[6] = most | ((levels & 65535) << 16);
        const float clamp = volume ? v.Texture3D.ResourceMinLODClamp : v.TextureCube.ResourceMinLODClamp;
        std::memcpy(w + 7, &clamp, 4);
    } else {
        w[5] = v.Shader4ComponentMapping | (v.Texture2D.MipLevels == UINT32_MAX ? 1u << 28 : 0u);
        w[6] = v.Texture2D.MostDetailedMip | (v.Texture2D.MipLevels << 16);
        std::memcpy(w + 7, &v.Texture2D.ResourceMinLODClamp, 4);
    }
    return true;
}
enum class Resolution { hosted, unavailable, refused, publish_failed, published };

// A GPU resource with the exact address and description may outlive the CPU
// mapping of its texels. Only a new detiled copy requires that mapping.
template<class Lookup, class Check, class Publish>
auto resolve(Lookup lookup, Check check, Publish publish, Resolution& why)
{
    auto texture = lookup();
    if (texture) {
        why = Resolution::hosted;
        return texture;
    }
    const unsigned decline = check();
    if (decline) {
        why = decline == 8 ? Resolution::unavailable : Resolution::refused;
        return texture;
    }
    texture = publish();
    why = texture ? Resolution::published : Resolution::publish_failed;
    return texture;
}

inline bool decode(const std::uint32_t* w, D3D12_RESOURCE_DESC& r, D3D12_SHADER_RESOURCE_VIEW_DESC& v)
{
    r = {};
    v = {};
    if (!w || !marker(w[0]) || (!w[1] && !w[2])) return false;
    r.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    r.Width = w[3] & 65535;
    r.Height = w[3] >> 16;
    r.DepthOrArraySize = 1;
    r.Format = DXGI_FORMAT(w[4] & 255);
    r.MipLevels = UINT16(w[4] >> 24);
    r.Layout = D3D12_TEXTURE_LAYOUT(0x100 | ((w[4] >> 16) & 255));
    r.SampleDesc.Count = 1;
    r.Flags = D3D12_RESOURCE_FLAGS(0x300000);
    v.Format = DXGI_FORMAT((w[4] >> 8) & 255);
    if (v.Format != DXGI_FORMAT_UNKNOWN && !texture_format(v.Format)) return false;
    if (w[5] >> 31) {
        const std::uint64_t packed = (std::uint64_t(w[5]) << 32) | w[6];
        r.DepthOrArraySize = UINT16(((packed >> 13) & 2047) + 1);
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        v.Shader4ComponentMapping = UINT(packed & 0x1fff);
        auto& a = v.Texture2DArray;
        a.FirstArraySlice = UINT((packed >> 24) & 2047);
        a.ArraySize = UINT(((packed >> 35) & 2047) + 1);
        a.MostDetailedMip = UINT((packed >> 46) & 255);
        a.MipLevels = UINT((packed >> 54) & 511);
        if (a.MipLevels == 511) a.MipLevels = UINT32_MAX;
        std::memcpy(&a.ResourceMinLODClamp, w + 7, 4);
        return valid_shape(r) && a.FirstArraySlice < r.DepthOrArraySize &&
            a.ArraySize <= r.DepthOrArraySize - a.FirstArraySlice && a.MipLevels;
    }
    if (w[5] & (1u << 30)) {
        const bool volume = (w[5] & (1u << 29)) != 0;
        if (volume) r.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        r.DepthOrArraySize = UINT16(w[5] >> 13);
        v.ViewDimension = volume ? D3D12_SRV_DIMENSION_TEXTURE3D : D3D12_SRV_DIMENSION_TEXTURECUBE;
        v.Shader4ComponentMapping = w[5] & 0x1fff;
        const UINT most = w[6] & 65535;
        const UINT levels = (w[6] >> 16) == 65535 ? UINT32_MAX : w[6] >> 16;
        float clamp;
        std::memcpy(&clamp, w + 7, 4);
        if (volume) v.Texture3D = {most, levels, clamp};
        else v.TextureCube = {most, levels, clamp};
        return valid_shape(r) && (volume || (r.Width == r.Height && r.DepthOrArraySize == 6)) && levels;
    }
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = w[5] & 0x1fff;
    v.Texture2D.MostDetailedMip = w[6] & 65535;
    v.Texture2D.MipLevels = (w[5] & (1u << 28)) ? UINT32_MAX : w[6] >> 16;
    std::memcpy(&v.Texture2D.ResourceMinLODClamp, w + 7, 4);
    return valid_shape(r);
}
}
