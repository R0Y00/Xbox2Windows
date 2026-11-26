#include "startup_stubs/frame_rgb_sample.h"

#include <windows.h>
#include <d3d12.h>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

static void log(const char*, ...) {}
#include "startup_stubs/frame_rgb_report.inl"

static bool check_image_file()
{
    namespace fs = std::filesystem;
    auto dir = fs::temp_directory_path() / (L"x2w-frame-rgb-" + std::to_wstring(GetCurrentProcessId()));
    if (!fs::create_directory(dir)) return false;
    auto trap = dir / L"trap.log";
    auto file = dir / L"frame-123-present.pfm";
    if (!SetEnvironmentVariableW(L"X2W_TRAP_LOG", trap.c_str())) return false;
    std::array<unsigned char, 768> texture{};
    texture[256] = 255;      // top row red, bottom row blue; nonzero footprint offset
    texture[512 + 2] = 255;
    D3D12_RESOURCE_DESC desc{};
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Width = 1;
    desc.Height = 2;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    fp.Offset = 256;
    fp.Footprint.RowPitch = 256;
    report_frame_rgb("present", 123, nullptr, texture.data(), desc, fp);
    std::ifstream in(file, std::ios::binary);
    std::string magic, dimensions, scale;
    std::getline(in, magic);
    std::getline(in, dimensions);
    std::getline(in, scale);
    std::vector<float> rgb(96 * 54 * 3);
    in.read(reinterpret_cast<char*>(rgb.data()), rgb.size() * sizeof(float));
    const bool ok = in && magic == "PF" && dimensions == "96 54" && scale == "-1.0" &&
                    rgb[2] == 1 && rgb[0] == 0 && rgb[rgb.size()-3] == 1 && rgb.back() == 0 &&
                    in.peek() == std::ifstream::traits_type::eof();
    in.close();
    fs::remove(file);
    fs::remove(dir);
    return ok;
}

int main()
{
    using namespace x2w_frame_sample;
    const std::array<std::uint16_t, 4> hdr{0xbc00, 0x0001, 0x4000, 0x3c00};
    auto pixel = decode(reinterpret_cast<const unsigned char*>(hdr.data()), Format::rgba16f);
    if (pixel.r != -1 || pixel.g != std::ldexp(1.f, -24) || pixel.b != 2 || pixel.a != 1) return 1;
    Counts c;
    c.add(pixel);
    c.add({0, -0.f, 0, 1});
    c.add({0, 0, 0, 0});
    c.add({NAN, 1, 1, 1});
    if (c.rgb != 1 || c.alpha_only != 1 || c.nonfinite != 1 || c.peak != 2) return 2;
    const unsigned char bytes[]{0, 51, 255, 128};
    pixel = decode(bytes, Format::bgra8);
    if (pixel.r != 1 || pixel.b != 0 || pixel.g != 51 / 255.f || pixel.a != 128 / 255.f) return 3;
    pixel = decode(bytes, Format::rgba8);
    if (pixel.r != 0 || pixel.b != 1) return 4;
    if (coordinate(3840, 48, 96) != 1940 || coordinate(1, 95, 96) != 0) return 5;
    return check_image_file() ? 0 : 6;
}
