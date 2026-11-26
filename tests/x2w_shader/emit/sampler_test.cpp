// Port of the tests in the upstream Rust module emit/sampler.rs.

#include "x2w_shader/emit/sampler.h"
#include "test_util.h"

using x2w::vs::sampler::StaticSampler;
using A3 = std::array<std::uint32_t, 3>;

namespace {

void console_compiled_samplers_decode() {
    // Words the console compiler wrote for static samplers of known descriptions.
    auto point = StaticSampler::from_words({0x3088, 0x00540100, 0x14000080, 0x80000000});
    StaticSampler want;
    want.filter = 0;
    want.address = {1, 2, 3};
    want.mip_lod_bias = 0.5f;
    want.max_anisotropy = 1;
    want.comparison = 4;
    want.border = 2;
    want.min_lod = 1.0f;
    want.max_lod = 5.25f;
    CHECK(point == want);
    auto aniso = StaticSampler::from_words({0x369E, 0x00FFF000, 0x18F03E00, 0x80000000});
    CHECK(aniso.filter == 0x55 && (aniso.address == A3{4, 5, 3}) && aniso.max_anisotropy == 8);
    CHECK(aniso.mip_lod_bias == -2.0f && aniso.max_lod == 4095.0f / 256.0f);
    auto compare = StaticSampler::from_words({0x3006, 0x00FFF000, 0x14500000, 0x40000000});
    CHECK(compare.filter == 0x14 && (compare.address == A3{4, 1, 1}) && compare.border == 1);
    auto minimum = StaticSampler::from_words({0x20003003, 0x00FFF000, 0x18500000, 0x80000000});
    CHECK(minimum.filter == 0x115 && minimum.address[0] == 5);
    auto mag_mip = StaticSampler::from_words({0x3002, 0, 0x18100000, 0x80000000});
    CHECK(mag_mip.filter == 0x05 && mag_mip.max_lod == 0.0f);
    auto max_aniso = StaticSampler::from_words({0x40003800, 0x00FFF000, 0x18F00F00, 0x80000000});
    CHECK(max_aniso.filter == 0x1D5 && max_aniso.max_anisotropy == 16 && max_aniso.mip_lod_bias == 15.0f);
}

void anisotropic_samplers_preserve_mip_point_filtering() {
    const std::uint32_t cases[][2] = {{0, 0x54}, {1, 0x154}, {2, 0x1D4}};
    for (auto& c : cases) {
        std::uint32_t w0 = c[0] << 29;
        for (std::uint32_t mip : {0u, 1u}) {
            std::uint32_t w2 = (mip << 26) | 0x00F00000;
            CHECK(StaticSampler::from_words({w0, 0x00FFF000, w2, 0}).filter == c[1]);
        }
        std::uint32_t w2 = (2u << 26) | 0x00F00000;
        CHECK(StaticSampler::from_words({w0, 0x00FFF000, w2, 0}).filter == (c[1] | 1));
    }
    auto mixed = StaticSampler::from_words({0, 0, (1u << 26) | (1u << 22) | (2u << 20), 0});
    CHECK(mixed.filter == 0x54);
}

void host_filter_falls_back_without_changing_reduction_or_lod() {
    const std::uint32_t cases[][2] = {{0x54, 0x55}, {0xD4, 0xD5}, {0x154, 0x155}, {0x1D4, 0x1D5}};
    for (auto& c : cases) {
        for (std::uint32_t mip : {0u, 1u}) {
            StaticSampler s = StaticSampler::from_words({0, 0x00540100, (mip << 26) | 0x00F00000, 0});
            s.filter = c[0];
            CHECK(s.filter_for_host(false) == c[1]);
            CHECK(s.filter_for_host(true) == c[0]);
            if (mip == 0)
                CHECK(s.min_lod == 0.0f && s.max_lod == 0.0f);
            else
                CHECK(s.min_lod == 1.0f && s.max_lod == 5.25f);
        }
    }
}

void host_filter_keeps_other_filters_unchanged() {
    for (std::uint32_t reduction : {0u, 0x80u, 0x100u, 0x180u}) {
        for (std::uint32_t base : {0u, 1u, 4u, 5u, 0x10u, 0x11u, 0x14u, 0x15u, 0x55u}) {
            StaticSampler s = StaticSampler::from_words({0, 0, 0, 0});
            s.filter = base | reduction;
            CHECK(s.filter_for_host(false) == s.filter);
            CHECK(s.filter_for_host(true) == s.filter);
        }
    }
}

}  // namespace

int main() {
    console_compiled_samplers_decode();
    anisotropic_samplers_preserve_mip_point_filtering();
    host_filter_falls_back_without_changing_reduction_or_lod();
    host_filter_keeps_other_filters_unchanged();
    return test_util::finish("sampler_test");
}
