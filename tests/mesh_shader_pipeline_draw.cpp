#include "desktop_gpu.h"
#include "mesh_shader_pipeline.h"
#include "meshlet_builder.h"

#include "meshlet_pipeline_as.h"
#include "meshlet_pipeline_ms.h"
#include "meshlet_pipeline_ps.h"
#include "meshlet_pipeline_vs.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using Microsoft::WRL::ComPtr;
using x2w::meshlet::MeshShaderBytecode;
using x2w::meshlet::MeshShaderCapabilities;
using x2w::meshlet::MeshShaderPipeline;
using x2w::meshlet::MeshShaderPipelineMode;
using x2w::meshlet::MeshletBufferData;
using x2w::meshlet::MeshletGpuResources;
using x2w::meshlet::SceneConstantsGPU;

constexpr UINT kSide = 64;

struct RenderResult {
    std::vector<std::uint8_t> rgba;
    std::array<std::uint32_t, 4> cull_stats{};
    MeshShaderCapabilities caps{};
    bool used_mesh_shaders = false;
};

SceneConstantsGPU make_scene_constants(std::uint32_t meshlet_count) {
    SceneConstantsGPU cb{};
    // Identity view-projection matrix (row-major).
    cb.view_proj[0] = 1.0f;
    cb.view_proj[5] = 1.0f;
    cb.view_proj[10] = 1.0f;
    cb.view_proj[15] = 1.0f;

    // 6 NDC frustum planes for D3D12 depth [0, 1].
    const float planes[6][4]{
        { 1.0f,  0.0f,  0.0f, 1.0f}, // Left
        {-1.0f,  0.0f,  0.0f, 1.0f}, // Right
        { 0.0f,  1.0f,  0.0f, 1.0f}, // Bottom
        { 0.0f, -1.0f,  0.0f, 1.0f}, // Top
        { 0.0f,  0.0f,  1.0f, 0.0f}, // Near
        { 0.0f,  0.0f, -1.0f, 1.0f}, // Far
    };
    std::memcpy(cb.frustum_planes, planes, sizeof(planes));

    cb.camera_pos[0] = 0.0f;
    cb.camera_pos[1] = 0.0f;
    cb.camera_pos[2] = -2.0f;
    cb.camera_pos[3] = 1.0f; // Enable AS culling
    cb.meshlet_count = meshlet_count;
    return cb;
}

RenderResult render_scene(ID3D12Device* device, const MeshletBufferData& scene, MeshShaderPipelineMode mode) {
    MeshShaderBytecode shaders{};
    shaders.as = {meshlet_pipeline_as, sizeof(meshlet_pipeline_as)};
    shaders.ms = {meshlet_pipeline_ms, sizeof(meshlet_pipeline_ms)};
    shaders.vs = {meshlet_pipeline_vs, sizeof(meshlet_pipeline_vs)};
    shaders.ps = {meshlet_pipeline_ps, sizeof(meshlet_pipeline_ps)};

    MeshShaderPipeline pipeline;
    if (!pipeline.initialize(device, shaders, mode, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_D32_FLOAT)) {
        throw std::runtime_error("MeshShaderPipeline::initialize unexpectedly returned false");
    }

    if (pipeline.using_mesh_shaders()) {
        using namespace x2w::meshlet;
        const auto expect_refusal = [&](const std::vector<MeshletVertex>& vertices, const std::vector<std::uint32_t>& indices) {
            const auto oversized = build_meshlets(vertices, indices, kHardwareMaxMeshletVertices, kHardwareMaxMeshletPrimitives);
            try {
                pipeline.create_gpu_resources(device, oversized);
            } catch (const std::invalid_argument&) {
                return;
            }
            throw std::runtime_error("Meshlet exceeding the shader output capacity was accepted");
        };

        std::vector<MeshletVertex> vertices(kMaxMeshletVertices + 1, scene.vertices.front());
        std::vector<std::uint32_t> indices;
        for (std::uint32_t i = 0; i < vertices.size(); ++i) indices.push_back(i);
        indices.push_back(0);
        expect_refusal(vertices, indices);
        indices.clear();
        for (std::uint32_t i = 0; i <= kMaxMeshletPrimitives; ++i)
            indices.insert(indices.end(), {0, 1, 2});
        expect_refusal(scene.vertices, indices);
    }
    const MeshletGpuResources res = pipeline.create_gpu_resources(device, scene);
    const SceneConstantsGPU cb_data = make_scene_constants(res.meshlet_count);

    auto cb_upload = make_buffer(device, sizeof(SceneConstantsGPU), D3D12_HEAP_TYPE_UPLOAD,
                                 D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    const D3D12_RANGE empty{0, 0};
    check(cb_upload->Map(0, &empty, &mapped), "Map scene constants");
    std::memcpy(mapped, &cb_data, sizeof(cb_data));
    cb_upload->Unmap(0, nullptr);

    constexpr UINT64 kStatsBytes = 16;
    auto stats_zero = make_buffer(device, kStatsBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    check(stats_zero->Map(0, &empty, &mapped), "Map stats zero upload");
    std::memset(mapped, 0, kStatsBytes);
    stats_zero->Unmap(0, nullptr);

    auto stats_uav = make_buffer(device, kStatsBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto stats_readback = make_buffer(device, kStatsBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    D3D12_RESOURCE_DESC rtv_desc{};
    rtv_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rtv_desc.Width = kSide;
    rtv_desc.Height = kSide;
    rtv_desc.DepthOrArraySize = 1;
    rtv_desc.MipLevels = 1;
    rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtv_desc.SampleDesc.Count = 1;
    rtv_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const float clear_color[4]{0.0f, 0.0f, 0.0f, 1.0f};
    D3D12_CLEAR_VALUE rtv_clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.0f, 0.0f, 0.0f, 1.0f}};
    auto target = resource(device, rtv_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET, &rtv_clear);

    D3D12_RESOURCE_DESC dsv_desc = rtv_desc;
    dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
    dsv_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE dsv_clear{};
    dsv_clear.Format = DXGI_FORMAT_D32_FLOAT;
    dsv_clear.DepthStencil.Depth = 1.0f;
    auto depth = resource(device, dsv_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_DEPTH_WRITE, &dsv_clear);

    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    check(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap)), "Create RTV heap");
    const auto rtv_handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv_handle);

    ComPtr<ID3D12DescriptorHeap> dsv_heap;
    D3D12_DESCRIPTOR_HEAP_DESC dsv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    check(device->CreateDescriptorHeap(&dsv_heap_desc, IID_PPV_ARGS(&dsv_heap)), "Create DSV heap");
    const auto dsv_handle = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(depth.Get(), nullptr, dsv_handle);

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 readback_bytes = 0;
    device->GetCopyableFootprints(&rtv_desc, 0, 1, 0, &footprint, nullptr, nullptr, &readback_bytes);
    auto pixel_readback = make_buffer(device, readback_bytes, D3D12_HEAP_TYPE_READBACK,
                                      D3D12_RESOURCE_STATE_COPY_DEST);

    ComputeCommands commands(device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
    auto* list = commands.list.Get();

    list->CopyBufferRegion(stats_uav.Get(), 0, stats_zero.Get(), 0, kStatsBytes);
    transition(list, stats_uav.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    const D3D12_VIEWPORT viewport{0.0f, 0.0f, static_cast<float>(kSide), static_cast<float>(kSide), 0.0f, 1.0f};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(kSide), static_cast<LONG>(kSide)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    list->OMSetRenderTargets(1, &rtv_handle, FALSE, &dsv_handle);
    list->ClearRenderTargetView(rtv_handle, clear_color, 0, nullptr);
    list->ClearDepthStencilView(dsv_handle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    pipeline.record_draw(list, cb_upload->GetGPUVirtualAddress(), stats_uav->GetGPUVirtualAddress(), res);

    transition(list, stats_uav.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(stats_readback.Get(), 0, stats_uav.Get(), 0, kStatsBytes);

    transition(list, target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = pixel_readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    commands.submit_and_wait(device);

    RenderResult result;
    result.caps = pipeline.capabilities();
    result.used_mesh_shaders = pipeline.using_mesh_shaders();
    result.rgba.resize(static_cast<std::size_t>(kSide) * kSide * 4);

    const D3D12_RANGE stats_range{0, kStatsBytes};
    check(stats_readback->Map(0, &stats_range, &mapped), "Map cull stats readback");
    std::memcpy(result.cull_stats.data(), mapped, kStatsBytes);
    stats_readback->Unmap(0, &empty);

    const D3D12_RANGE pixel_range{0, static_cast<SIZE_T>(readback_bytes)};
    check(pixel_readback->Map(0, &pixel_range, &mapped), "Map rendered pixels");
    const auto* base = static_cast<const std::uint8_t*>(mapped);
    for (UINT y = 0; y < kSide; ++y) {
        std::memcpy(&result.rgba[static_cast<std::size_t>(y) * kSide * 4],
                    base + footprint.Offset + static_cast<std::size_t>(y) * footprint.Footprint.RowPitch,
                    kSide * 4);
    }
    pixel_readback->Unmap(0, &empty);
    return result;
}

void verify_rendered_image(const std::vector<std::uint8_t>& rgba, const char* label) {
    auto pixel_at = [&](UINT x, UINT y) -> std::array<std::uint8_t, 4> {
        const std::size_t idx = (static_cast<std::size_t>(y) * kSide + x) * 4;
        return {rgba[idx + 0], rgba[idx + 1], rgba[idx + 2], rgba[idx + 3]};
    };
    const std::array<std::uint8_t, 4> green{0, 255, 0, 255};
    const std::array<std::uint8_t, 4> blue{0, 0, 255, 255};
    const std::array<std::uint8_t, 4> black{0, 0, 0, 255};

    if (pixel_at(16, 32) != green) {
        std::fprintf(stderr, "%s: expected green at (16,32)\n", label);
        throw std::runtime_error("Left visible meshlet pixel mismatch");
    }
    if (pixel_at(48, 32) != blue) {
        std::fprintf(stderr, "%s: expected blue at (48,32)\n", label);
        throw std::runtime_error("Right visible meshlet pixel mismatch");
    }
    if (pixel_at(32, 32) != black) {
        std::fprintf(stderr, "%s: expected black at (32,32) (backface-culled meshlet leaked)\n", label);
        throw std::runtime_error("Backface-culled meshlet was not culled");
    }

    for (UINT y = 0; y < kSide; ++y) {
        for (UINT x = 0; x < kSide; ++x) {
            const auto p = pixel_at(x, y);
            if (p[0] > 0) {
                std::fprintf(stderr, "%s: unexpected red channel %u at (%u,%u)\n", label, p[0], x, y);
                throw std::runtime_error("Culled red/yellow meshlet rendered a pixel");
            }
        }
    }
}

} // namespace

bool check_mesh_shader_pipeline(ID3D12Device* device, bool force_fallback) {
    const MeshletBufferData scene = x2w::meshlet::build_culling_test_scene();

    if (force_fallback) {
        const RenderResult fallback = render_scene(device, scene, MeshShaderPipelineMode::ForceVertexFallback);
        std::printf("mesh_shader_tier=%u mode=vertex_fallback\n",
                    static_cast<unsigned>(fallback.caps.mesh_shader_tier));
        verify_rendered_image(fallback.rgba, "vertex_fallback");
        return true;
    }

    const RenderResult auto_result = render_scene(device, scene, MeshShaderPipelineMode::Auto);
    std::printf("mesh_shader_tier=%u shader_model=0x%x mode=%s\n",
                static_cast<unsigned>(auto_result.caps.mesh_shader_tier),
                static_cast<unsigned>(auto_result.caps.highest_shader_model),
                auto_result.used_mesh_shaders ? "mesh" : "vertex_fallback");

    verify_rendered_image(auto_result.rgba, auto_result.used_mesh_shaders ? "mesh_shader" : "vertex_fallback");

    if (auto_result.used_mesh_shaders) {
        std::printf("cull_stats tested=%u frustum_culled=%u backface_culled=%u dispatched=%u\n",
                    auto_result.cull_stats[0], auto_result.cull_stats[1],
                    auto_result.cull_stats[2], auto_result.cull_stats[3]);
        if (auto_result.cull_stats != std::array<std::uint32_t, 4>{4, 1, 1, 2}) {
            throw std::runtime_error("Unexpected amplification shader cull statistics");
        }

        const RenderResult fallback = render_scene(device, scene, MeshShaderPipelineMode::ForceVertexFallback);
        if (auto_result.rgba != fallback.rgba) {
            throw std::runtime_error("Mesh shader output differs from classic vertex fallback output");
        }
        std::puts("fallback_parity=exact");
    }

    return true;
}
