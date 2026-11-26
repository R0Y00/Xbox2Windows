#pragma once

#include <dxgiformat.h>
#include <windows.h>
#include <DirectXPackedVector.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>

namespace x2w_draw_bmp {

inline bool supported(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        || format == DXGI_FORMAT_B8G8R8A8_UNORM || format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
}

inline bool hdr(DXGI_FORMAT format)
{
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT || format == DXGI_FORMAT_R11G11B10_FLOAT || format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
}

inline bool trace_supported(DXGI_FORMAT format)
{
    return supported(format) || hdr(format) || format == DXGI_FORMAT_R16G16_FLOAT;
}

// A fixed exposure for diagnostic images across frames. Preserve the original
// float values in the trace's raw sample grid. Negative values and NaNs go black.
inline unsigned char display_float(float value)
{
    if (!(value > 0)) return 0;
    return static_cast<unsigned char>(255.f * (1.f - 1.f / (1.f + value)) + 0.5f);
}

// Signed two-channel data: zero is mid-gray, +/-1 use half the available
// contrast, larger magnitudes approach the corresponding endpoint.
inline unsigned char display_signed(float value)
{
    if (value > 0) return static_cast<unsigned char>(128.f + 127.f * (1.f - 1.f / (1.f + value)) + 0.5f);
    if (value < 0) return static_cast<unsigned char>(128.f / (1.f - value) + 0.5f);
    return 128;
}

// D3D12 copy footprints include row padding. BMP input here starts at the footprint offset.
inline bool write(FILE* file, const unsigned char* pixels, UINT width, UINT height,
                  UINT row_pitch, DXGI_FORMAT format)
{
    const std::uint64_t stride = std::uint64_t{width} * 4;
    constexpr DWORD header_bytes = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    const UINT pixel_bytes = format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u : 4u;
    if (!file || !pixels || !trace_supported(format) || !width || !height || std::uint64_t{width} * pixel_bytes > row_pitch ||
        stride * height > MAXDWORD - header_bytes) return false;

    BITMAPFILEHEADER header{};
    header.bfType = 0x4d42;
    header.bfSize = static_cast<DWORD>(header_bytes + stride * height);
    header.bfOffBits = header_bytes;
    BITMAPINFOHEADER info{};
    info.biSize = sizeof(info);
    info.biWidth = static_cast<LONG>(width);
    info.biHeight = static_cast<LONG>(height);
    info.biPlanes = 1;
    info.biBitCount = 32;
    info.biSizeImage = header.bfSize - header_bytes;
    if (std::fwrite(&header, sizeof(header), 1, file) != 1 ||
        std::fwrite(&info, sizeof(info), 1, file) != 1) return false;

    const bool rgba = format == DXGI_FORMAT_R8G8B8A8_UNORM || format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    std::vector<unsigned char> row(static_cast<size_t>(stride));
    for (UINT y = height; y > 0; --y) {
        const auto* source = pixels + size_t{y - 1} * row_pitch;
        for (size_t x = 0; x < row.size(); x += 4) {
            if (format == DXGI_FORMAT_R16G16_FLOAT) {
                using namespace DirectX::PackedVector;
                HALF channels[2];
                std::memcpy(channels, source + x, sizeof(channels));
                row[x] = 128;
                row[x + 1] = display_signed(XMConvertHalfToFloat(channels[1]));
                row[x + 2] = display_signed(XMConvertHalfToFloat(channels[0]));
                row[x + 3] = 255;
                continue;
            }
            if (hdr(format)) {
                DirectX::XMFLOAT3 rgb{};
                using namespace DirectX::PackedVector;
                if (pixel_bytes == 8) {
                    HALF channels[4];
                    std::memcpy(channels, source + 2 * x, sizeof(channels));
                    rgb = {XMConvertHalfToFloat(channels[0]), XMConvertHalfToFloat(channels[1]), XMConvertHalfToFloat(channels[2])};
                } else if (format == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) {
                    XMFLOAT3SE packed;
                    std::memcpy(&packed, source + x, sizeof(packed));
                    DirectX::XMStoreFloat3(&rgb, XMLoadFloat3SE(&packed));
                } else {
                    XMFLOAT3PK packed;
                    std::memcpy(&packed, source + x, sizeof(packed));
                    DirectX::XMStoreFloat3(&rgb, XMLoadFloat3PK(&packed));
                }
                row[x] = display_float(rgb.z);
                row[x + 1] = display_float(rgb.y);
                row[x + 2] = display_float(rgb.x);
                row[x + 3] = 255;
                continue;
            }
            row[x] = source[x + (rgba ? 2 : 0)];
            row[x + 1] = source[x + 1];
            row[x + 2] = source[x + (rgba ? 0 : 2)];
            row[x + 3] = 255;
        }
        if (std::fwrite(row.data(), 1, row.size(), file) != row.size()) return false;
    }
    return true;
}
}
