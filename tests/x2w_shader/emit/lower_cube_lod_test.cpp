// Port of the upstream Rust module emit/cube_lod_tests.rs.
//
// Cube samples with gradients select their level by the footprint (after upstream commit 921d8d3b).

#include "lower_card_util.h"

using namespace card;

namespace {

// ---- rs:11 --------------------------------------------------------------------------------------
// The level is the red channel on all six faces so every sample returns its selected mip directly
// (including the fraction with trilinear filtering).
void guest_cube_gradient_selects_mip_by_footprint() {
    x2w::vs::lower::test::set_dump_name("emit__cube_lod_tests__guest_cube_gradient_selects_mip_by_footprint");
    constexpr std::size_t SIZE = 64;
    constexpr std::uint16_t MIPS = 7;
    std::vector<std::uint32_t> words = {
        0xBE81'0380,
        0xBE83'0380,  // s1, s3 = 0 (upper halves of the tables)
        0xF40C'0400,
        0xFA00'0000,  // cube SRV to s[16:23] from s[0:1]
        0xF408'0300,
        0xFA00'0020,  // raw output UAV to s[12:15] from s[0:1]+32
        0xF408'0601,
        0xFA00'0000,  // sampler to s[24:27] from s[2:3]
        0x7E1E'0280,  // v15 = 0 for output byte offsets
    };
    struct Case {
        std::uint32_t op;
        float footprint, lod, mip;
    };
    const Case cases[] = {
        // image_sample_d: successive XY footprints (in texels).
        {0x22, 1.0f, 0.0f, 0.0f},
        {0x22, 2.0f, 0.0f, 1.0f},
        {0x22, 16.0f, 0.0f, 4.0f},
        {0x22, 0.000001f, 0.0f, 0.0f},
        {0x22, 0.0f, 0.0f, 0.0f},
        // image_sample_d_cl: an instruction-specified minimum level.
        {0x23, 1.0f, 3.0f, 3.0f},
        {0x23, 16.0f, 2.0f, 4.0f},
        {0x23, 0.0f, 2.0f, 2.0f},
        // image_sample_l and image_sample_lz: the level is given directly.
        {0x24, 0.0f, 2.0f, 2.0f},
        {0x27, 0.0f, 0.0f, 0.0f},
    };
    constexpr std::uint32_t CASES = sizeof cases / sizeof cases[0];
    for (std::uint32_t i = 0; i < CASES; ++i) {
        const Case& c = cases[i];
        float delta = c.footprint / static_cast<float>(SIZE);
        std::vector<float> addr;
        if (c.op == 0x22 || c.op == 0x23)
            for (float f : {delta, 0.0f, 0.0f, delta}) addr.push_back(f);  // derivatives along the face
        for (float f : {1.5f, 1.5f, 0.0f}) addr.push_back(f);  // positive X face, center
        if (c.op == 0x23 || c.op == 0x24) addr.push_back(c.lod);
        for (std::size_t reg = 0; reg < addr.size(); ++reg) {
            words.push_back(0x7E00'02FFu | (static_cast<std::uint32_t>(reg) << 17));
            words.push_back(f32_bits(addr[reg]));
        }
        // MIMG, CUBE, dmask:R, v0... inputs, SRV s16, sampler s24, result v10.
        words.push_back(0xF000'0118u | (c.op << 18));
        words.push_back(0x00C4'0A00u);
        // buffer_store_dword v10, v15, s[12:15], 0 offen offset:i*4
        words.push_back(0xE070'1000u | (i * 4));
        words.push_back(0x8003'0A0Fu);
    }
    words.push_back(0xBF81'0000u);  // s_endpgm
    auto table = [](std::uint8_t slot, std::uint8_t user_data) {
        return root_param(RootType::Table(), 0, 1, 0, user_data, slot);
    };
    auto range = [](ResourceType kind, std::uint32_t at, std::uint8_t dwords, std::uint8_t slot, std::uint8_t user_data) {
        return used_resource(kind, RootType::Table(), at, 0, 0, dwords, user_data, slot);
    };
    NativeProgram native;
    native.stage = HwStage::Cs;
    native.wave64 = false;
    native.vgprs = 16;
    native.layout = {table(0, 0), table(1, 2)};
    native.used = {range(ResourceType::Srv(), 0, 8, 0, 0), range(ResourceType::Uav(), 8, 4, 0, 0),
                   range(ResourceType::Sampler(), 0, 4, 1, 2)};
    native.io = compute_io(3 << 1, {1, 1, 1});
    native.code = le_bytes(words);
    HostShader shader = build_compute_shader(native).first;
    std::vector<Texel> texels;
    for (int face = 0; face < 6; ++face)
        for (std::uint16_t mip = 0; mip < MIPS; ++mip) {
            std::size_t n = (SIZE >> mip) * (SIZE >> mip);
            float level = static_cast<float>(mip);
            texels.insert(texels.end(), n, Texel{level, level, level, level});
        }
    D3D12_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxLOD = std::numeric_limits<float>::max();
    sampler.MaxAnisotropy = 1;
    const D3D12_SAMPLER_DESC samplers[] = {sampler};
    const std::pair<const char*, bool> devices[] = {{"WARP", true}, {"hardware", false}};
    for (auto [name, warp] : devices) {
        auto host = new_host(warp);
        if (!host) {
            std::fprintf(stderr, "no SM 6.6 %s device; skipped\n", name);
            continue;
        }
        std::string why;
        auto pipe = compute_pipeline(*host, shader, &why);
        if (!pipe) fail(std::string(name) + ": compute_pipeline: " + why);
        const View views[] = {cube(static_cast<std::uint32_t>(SIZE), MIPS, texels), raw_uav(CASES)};
        auto out = dispatch(*host, *pipe, std::vector<std::uint32_t>(shader.user_data_dwords(), 0), views, samplers,
                            {1, 1, 1}, name);
        std::vector<float> levels, expected;
        for (std::uint32_t w : out[0]) levels.push_back(f32_from_bits(w));
        for (const Case& c : cases) expected.push_back(c.mip);
        require(levels == expected, name);
    }
}

}  // namespace

int main() {
    guarded("guest_cube_gradient_selects_mip_by_footprint", guest_cube_gradient_selects_mip_by_footprint);
    return test_util::finish("lower_cube_lod_test");
}
