// Both stages sample the same normalized coordinates. Values stay in their native
// encoding (linear half float or normalized display bytes); alpha is counted separately.
void report_frame_rgb(const char* stage, LONG frame, const void* surface, const void* mapped,
                      const D3D12_RESOURCE_DESC& desc, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp)
{
    using namespace x2w_frame_sample;
    Format format;
    switch (desc.Format) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT: format = Format::rgba16f; break;
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: format = Format::rgba8; break;
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: format = Format::bgra8; break;
    default:
        log("x2w-frame-pair %s submit_frame=%ld unsupported format=%u", stage, frame, desc.Format);
        return;
    }
    const auto* base = static_cast<const unsigned char*>(mapped) + fp.Offset;
    constexpr unsigned width = 96, height = 54;
    std::vector<float> image(width * height * 3);
    for (unsigned band = 0; band < 6; ++band) {
        Counts c;
        for (unsigned iy = band * 9; iy < (band + 1) * 9; ++iy) {
            auto* row = base + UINT64{coordinate(desc.Height, iy, 54)} * fp.Footprint.RowPitch;
            for (unsigned ix = 0; ix < 96; ++ix) {
                const auto x = coordinate(static_cast<unsigned>(desc.Width), ix, 96);
                const auto pixel = decode(row + x * (format == Format::rgba16f ? 8 : 4), format);
                c.add(pixel);
                const unsigned at = ((height - 1 - iy) * width + ix) * 3;  // PFM stores the bottom row first.
                image[at] = pixel.r;
                image[at + 1] = pixel.g;
                image[at + 2] = pixel.b;
            }
        }
        log("x2w-frame-pair %s submit_frame=%ld target=%p fmt=%u band=%u/6 samples=864 rgb=%u alpha-only=%u nonfinite=%u peak-abs-rgb=%.7g",
            stage, frame, surface, desc.Format, band + 1, c.rgb, c.alpha_only, c.nonfinite, c.peak);
    }
    wchar_t trap_path[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableW(L"X2W_TRAP_LOG", trap_path, MAX_PATH);
    if (!length || length >= MAX_PATH) return;
    std::wstring path(trap_path, length);
    path.resize(path.find_last_of(L"/\\") + 1);
    path += L"frame-" + std::to_wstring(frame) + L"-" + std::wstring(stage, stage + std::strlen(stage)) + L".pfm";
    FILE* output = nullptr;
    if (_wfopen_s(&output, path.c_str(), L"wb") || !output) {
        log("x2w-frame-pair %s submit_frame=%ld image open failed", stage, frame);
        return;
    }
    const bool written = std::fprintf(output, "PF\n%u %u\n-1.0\n", width, height) > 0 &&
                         std::fwrite(image.data(), sizeof(float), image.size(), output) == image.size();
    if (std::fclose(output) || !written) log("x2w-frame-pair %s submit_frame=%ld image write failed", stage, frame);
}
