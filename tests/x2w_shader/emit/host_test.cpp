// host.cpp (port of emit/host.rs) smoke test. host.rs has no tests of its own (its #[cfg(test)]
// parts are the draw/dispatch API the emit tests use); this runs that API on shaders dxc.exe
// compiles at test time: a compute dispatch with root constants, a root SRV, heap views (raw UAV,
// raw buffer SRV, texture UAV), a refused pipeline, a draw through Host::pipeline and one
// through Host::mesh_pipeline, and the stub shaders. Skips (and says so) without a hardware SM 6.6 device or dxc.exe.

#include "x2w_shader/emit/host.h"

#include <windows.h>

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "test_util.h"
#include "x2w_shader/emit/validator.h"

namespace fs = std::filesystem;
using namespace x2w::vs::host;

namespace {

std::optional<std::vector<std::uint8_t>> dxc(const fs::path& dxc_exe, const char* hlsl, const char* profile) {
    const fs::path dir = fs::temp_directory_path() / ("x2w_vs_host_test_" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    const fs::path src = dir / "shader.hlsl";
    const fs::path out = dir / "shader.dxil";
    std::ofstream(src, std::ios::binary) << hlsl;
    // The whole command quoted once more: cmd.exe strips the outer pair.
    const std::string cmd = "\"\"" + dxc_exe.string() + "\" -nologo -T " + profile + " -E main -Fo \"" +
                            out.string() + "\" \"" + src.string() + "\"\"";
    const int rc = std::system(cmd.c_str());
    std::ifstream in(out, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (rc != 0 || bytes.empty()) return std::nullopt;
    return bytes;
}

// What a HostShader offers the host (Stage converts from it implicitly).
struct FakeShader {
    std::vector<std::uint8_t> dxil;
    std::uint32_t dwords = 0;
    std::vector<x2w::vs::sampler::StaticSampler> static_samplers;
    std::vector<std::uint8_t> root_cbvs, root_srvs, root_uavs;
    std::uint32_t user_data_dwords() const { return dwords; }
};

// User data at b0 space 0 (root parameter 0), a root SRV t2 in ROOT_CBV_SPACE (space 1), the
// view heap directly indexed: [0] raw UAV, [1] raw SRV, [2] RGBA32F texture UAV.
const char* CS = R"(
cbuffer U : register(b0, space0) { uint4 k; };
ByteAddressBuffer root_srv : register(t2, space1);
[numthreads(4, 1, 1)]
void main(uint i : SV_DispatchThreadID) {
    RWByteAddressBuffer o = ResourceDescriptorHeap[0];
    ByteAddressBuffer b = ResourceDescriptorHeap[1];
    RWTexture2D<float4> t = ResourceDescriptorHeap[2];
    o.Store(i * 4, i * k.x + k.y + root_srv.Load(i * 4) + b.Load(i * 4));
    t[uint2(i & 1, i >> 1)] = float4(i, k.x, 0.5, 1);
}
)";

const char* VS = R"(
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
)";

const char* PS = R"(
cbuffer U : register(b0, space0) { float4 c; };
float4 main(float4 pos : SV_Position) : SV_Target { return c + float4(0, 0, 0, pos.x > 2 ? 1 : 0); }
)";

const char* MS = R"(
struct V { float4 pos : SV_Position; };
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(out vertices V v[3], out indices uint3 t[1]) {
    SetMeshOutputCounts(3, 1);
    v[0].pos = float4(-1, 1, 0, 1);
    v[1].pos = float4(3, 1, 0, 1);
    v[2].pos = float4(-1, -3, 0, 1);
    t[0] = uint3(0, 1, 2);
}
)";

}  // namespace

int main() {
    std::optional<fs::path> dxil = x2w::vs::validator::Validator::default_path();
    if (!dxil) {
        std::printf("host_test: skipped (no Windows SDK dxc)\n");
        return 0;
    }
    fs::path dxc_exe = *dxil;
    dxc_exe.replace_filename("dxc.exe");
    std::optional<Host> host;
    try {
        host.emplace(false);
    } catch (const Error& e) {
        std::printf("host_test: skipped (no device: %s)\n", e.what());
        return 0;
    }
    {
        DXGI_ADAPTER_DESC1 desc{};
        Microsoft::WRL::ComPtr<IDXGIFactory4> f;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> a;
        if (SUCCEEDED(CreateDXGIFactory2(0, IID_PPV_ARGS(&f))) &&
            SUCCEEDED(f->EnumAdapterByLuid(host->device->GetAdapterLuid(), IID_PPV_ARGS(&a))))
            a->GetDesc1(&desc);
        std::printf("host_test: device %ls\n", desc.Description);
    }

    // Compute: root constants, a root SRV, heap views.
    auto cs_bytes = dxc(dxc_exe, CS, "cs_6_6");
    CHECK(cs_bytes.has_value());
    if (cs_bytes) {
        FakeShader cs{*cs_bytes, 4, {}, {}, {2}, {}};
        try {
            auto [root, pso] = host->compute_pipeline(cs);
            const std::uint32_t srv_words[4] = {1000, 2000, 3000, 4000};
            const View views[] = {
                View::raw_uav(4),
                View::buffer(srv_words, Stride::raw()),
                View::texture_uav(2, 2),
            };
            Dispatch d;
            d.roots.emplace_back(0, Root::constants({10, 7, 0, 0}));
            // Root parameter 1: the root SRV t2 (after the user data).
            d.roots.emplace_back(1, Root::view(true, {100, 200, 300, 400}));
            d.views = views;
            d.groups = {1, 1, 1};
            auto out = host->dispatch(root.Get(), pso.Get(), d);
            CHECK(out.size() == 2);
            if (out.size() == 2) {
                const std::vector<std::uint32_t> want = {1107, 2217, 3327, 4437};
                CHECK(out[0] == want);
                CHECK(out[1].size() == 16);
                for (std::uint32_t i = 0; i < 4 && out[1].size() == 16; ++i) {
                    const std::size_t at = 4 * ((i >> 1) * 2 + (i & 1));
                    CHECK(std::bit_cast<float>(out[1][at]) == float(i));
                    CHECK(std::bit_cast<float>(out[1][at + 1]) == 10.0f);
                    CHECK(std::bit_cast<float>(out[1][at + 2]) == 0.5f);
                    CHECK(std::bit_cast<float>(out[1][at + 3]) == 1.0f);
                }
                std::printf("host_test: dispatch raw UAV = %u %u %u %u\n", out[0][0], out[0][1], out[0][2], out[0][3]);
            }
            // A container the runtime refuses: the pipeline call reports it.
            std::vector<std::uint8_t> broken = *cs_bytes;
            broken[0] = 'X';
            bool refused = false;
            try {
                host->compute_pipeline_with(root.Get(), broken);
            } catch (const Error& e) {
                refused = true;
                std::printf("host_test: broken container refused: %s\n", e.what());
            }
            CHECK(refused);
        } catch (const Error& e) {
            std::fprintf(stderr, "compute: %s\n", e.what());
            CHECK(false);
        }
    }

    // Draw: Host::pipeline (the root signature translated shaders expect, user data of VS at 0,
    // PS at 1; targets from the pixel shader's OSG1), the full-screen triangle.
    auto vs_bytes = dxc(dxc_exe, VS, "vs_6_6");
    auto ps_bytes = dxc(dxc_exe, PS, "ps_6_6");
    CHECK(vs_bytes && ps_bytes);
    const Root colour = Root::constants({std::bit_cast<std::uint32_t>(0.25f), std::bit_cast<std::uint32_t>(0.5f),
                                         std::bit_cast<std::uint32_t>(0.75f), 0});
    auto check_texels = [](const char* what, const std::vector<std::array<float, 4>>& texels) {
        CHECK(texels.size() == 8);
        for (std::size_t i = 0; i < texels.size(); ++i) {
            const float a = (i % 4) >= 2 ? 1.0f : 0.0f;  // pixel centres x = 2.5, 3.5
            CHECK((texels[i] == std::array<float, 4>{0.25f, 0.5f, 0.75f, a}));
        }
        if (texels.size() > 3)
            std::printf("host_test: %s texel 3 = %g %g %g %g\n", what, texels[3][0], texels[3][1], texels[3][2],
                        texels[3][3]);
    };
    if (vs_bytes && ps_bytes) {
        FakeShader vs{*vs_bytes, 0, {}, {}, {}, {}};
        FakeShader ps{*ps_bytes, 4, {}, {}, {}, {}};
        try {
            auto [root, pso] = host->pipeline(vs, ps);
            Draw d;
            d.roots.emplace_back(1, colour);
            d.size = {4, 2};
            check_texels("draw", host->draw(root.Get(), pso.Get(), d));
        } catch (const Error& e) {
            std::fprintf(stderr, "draw: %s\n", e.what());
            CHECK(false);
        }

        // The stub shaders (built with shader::Shader): the validator accepts them, and each
        // makes a pipeline with the other side's real shader.
        try {
            std::vector<std::uint8_t> svs = stub_vs_dxil(ps.dxil);
            std::vector<std::uint8_t> sps = stub_ps_dxil();
            std::string error;
            auto v = x2w::vs::validator::Validator::load(x2w::vs::validator::Validator::default_path(), &error);
            CHECK(v.has_value());
            if (v) {
                auto r1 = v->validate(svs);
                auto r2 = v->validate(sps);
                CHECK(!r1);
                CHECK(!r2);
                if (r1) std::fprintf(stderr, "stub_vs rejected: %s\n", r1->c_str());
                if (r2) std::fprintf(stderr, "stub_ps rejected: %s\n", r2->c_str());
            }
            FakeShader stub_vs_s{svs, 0, {}, {}, {}, {}};
            FakeShader stub_ps_s{sps, 0, {}, {}, {}, {}};
            auto [root1, pso1] = host->pipeline(stub_vs_s, ps);
            CHECK(pso1 != nullptr);
            auto [root2, pso2] = host->pipeline(vs, stub_ps_s);
            CHECK(pso2 != nullptr);
            // The stub pixel shader writes zeros to target 0.
            Draw d;
            d.size = {4, 2};
            auto texels = host->draw(root2.Get(), pso2.Get(), d);
            CHECK(texels.size() == 8);
            for (const auto& t : texels) CHECK((t == std::array<float, 4>{0, 0, 0, 0}));
            std::printf("host_test: stub_vs %zu bytes, stub_ps %zu bytes: validated, pipelines made\n", svs.size(),
                        sps.size());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "stubs: %s\n", e.what());
            CHECK(false);
        }

        // Mesh: Host::mesh_pipeline and a draw with mesh_groups, where the device has mesh
        // shaders.
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7{};
        const bool mesh = SUCCEEDED(host->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof o7)) &&
                          o7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
        std::optional<std::vector<std::uint8_t>> ms_bytes;
        if (!mesh) {
            std::printf("host_test: mesh skipped (no mesh shaders)\n");
        } else {
            ms_bytes = dxc(dxc_exe, MS, "ms_6_6");
            CHECK(ms_bytes.has_value());
        }
        if (ms_bytes) {
            FakeShader ms{*ms_bytes, 0, {}, {}, {}, {}};
            try {
                auto [root, pso] = host->mesh_pipeline(ms, ps);
                Draw d;
                d.roots.emplace_back(1, colour);
                d.size = {4, 2};
                d.mesh_groups = 1;
                check_texels("mesh", host->draw(root.Get(), pso.Get(), d));
            } catch (const Error& e) {
                std::fprintf(stderr, "mesh: %s\n", e.what());
                CHECK(false);
            }
        }
    }

    std::printf("host_test: ran on the GPU\n");
    return test_util::finish("host_test");
}
