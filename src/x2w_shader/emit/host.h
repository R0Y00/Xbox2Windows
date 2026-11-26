#pragma once

// Check emitted shaders by creating host D3D12 pipelines. Host::draw renders a full-screen triangle and
// Host::dispatch executes a compute shader. Both support readback in tests.
//
// Host failures throw host::Error with the diagnostic in what().
//
// Stage borrows the DXIL, user data size, static samplers and root CBV/SRV/UAV bindings of a HostShader
// (emit.h). It converts implicitly from a compatible shader so the host can build a pipeline without
// depending on native.h. Keep the shader alive through the call.
//
// stub_vs and stub_ps need HostShader complete: they are compiled once x2w_shader/native.h exists.
// The DXIL they carry is stub_vs_dxil/stub_ps_dxil (additions, built now).

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "x2w_shader/emit/sampler.h"

namespace x2w::vs {
struct HostShader;
}

namespace x2w::vs::host {

using Microsoft::WRL::ComPtr;

// A host pipeline or device failure.
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// What the host reads of a HostShader. Borrows.
struct Stage {
    std::span<const std::uint8_t> dxil;
    std::uint32_t user_data_dwords = 0;
    std::span<const sampler::StaticSampler> static_samplers;
    std::span<const std::uint8_t> root_cbvs;
    std::span<const std::uint8_t> root_srvs;
    std::span<const std::uint8_t> root_uavs;

    Stage() = default;
    template <class S>
        requires requires(const S& s) {
            s.dxil;
            s.user_data_dwords();
            s.static_samplers;
            s.root_cbvs;
            s.root_srvs;
            s.root_uavs;
        }
    Stage(const S& s)  // NOLINT: implicit shader-to-stage conversion
        : dxil(s.dxil),
          user_data_dwords(s.user_data_dwords()),
          static_samplers(s.static_samplers),
          root_cbvs(s.root_cbvs),
          root_srvs(s.root_srvs),
          root_uavs(s.root_uavs) {}
};

using Pipeline = std::pair<ComPtr<ID3D12RootSignature>, ComPtr<ID3D12PipelineState>>;

// A root argument for Host::draw / Host::dispatch.
struct Root {
    enum Kind : std::uint8_t { Constants, Table, View };
    Kind kind = Constants;
    // Constants: the dwords. View: the buffer's dwords.
    std::vector<std::uint32_t> data;
    // Table: a table starting at this heap index (of the sampler heap with sampler).
    bool sampler = false;
    std::size_t index = 0;
    // View: a root CBV (or with srv, a root SRV) over a buffer of data (Host::dispatch only).
    bool srv = false;

    static Root constants(std::vector<std::uint32_t> data) { return Root{Constants, std::move(data), false, 0, false}; }
    static Root table(bool sampler, std::size_t index) { return Root{Table, {}, sampler, index, false}; }
    static Root view(bool srv, std::vector<std::uint32_t> data) { return Root{View, std::move(data), false, 0, srv}; }
};

// An RGBA32F texture for Host::draw: (width, height, texels).
struct Texture {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::span<const std::array<float, 4>> texels;
};

// Bindings for Host::draw: root arguments and the heap contents.
struct Draw {
    std::vector<std::pair<std::uint32_t, Root>> roots;
    // RGBA32F textures at view-heap indices 0..
    std::span<const Texture> textures;
    // Samplers at sampler-heap indices 0..
    std::span<const D3D12_SAMPLER_DESC> samplers;
    // Three vertices of stride bytes (none for position-free layouts).
    std::span<const float> vertices;
    std::uint32_t stride = 0;
    std::pair<std::uint32_t, std::uint32_t> size{};
    // Mesh thread groups to dispatch instead of drawing (0: draw).
    std::uint32_t mesh_groups = 0;
};

// How View::Buffer is viewed.
struct Stride {
    enum Kind : std::uint8_t {
        Raw,
        // RGBA32F elements.
        Typed,
        // Structured, elements of bytes bytes.
        Structured,
        // Typed, elements in format of bytes bytes.
        Format,
    };
    Kind kind = Raw;
    std::uint32_t bytes = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;

    static Stride raw() { return Stride{Raw, 0, DXGI_FORMAT_UNKNOWN}; }
    static Stride typed() { return Stride{Typed, 0, DXGI_FORMAT_UNKNOWN}; }
    static Stride structured(std::uint32_t bytes) { return Stride{Structured, bytes, DXGI_FORMAT_UNKNOWN}; }
    static Stride format_of(DXGI_FORMAT f, std::uint32_t bytes) { return Stride{Format, bytes, f}; }
};

// A view in the heap for Host::dispatch.
struct View {
    enum Kind : std::uint8_t {
        // An RGBA32F texture SRV (width, height, texels).
        Texture,
        // An RGBA32F cube SRV (face size, mip count, texels in D3D subresource order).
        Cube,
        // A raw buffer UAV of n dwords, zeroed.
        RawUav,
        // An RGBA32F texture UAV (width, height), zeroed.
        TextureUav,
        // An R32_UINT texture UAV (width, height), zeroed (atomics).
        UintTextureUav,
        // A typed RGBA16F buffer UAV of n elements, zeroed.
        HalfUav,
        // A constant buffer of data.
        Cbv,
        // A buffer SRV over data.
        Buffer,
    };
    Kind kind = RawUav;
    // Texture/TextureUav/UintTextureUav: width, height. Cube: face size in w (and h).
    std::uint32_t w = 0;
    std::uint32_t h = 0;
    // Cube: mip count.
    std::uint16_t mips = 1;
    // RawUav: dwords. HalfUav: elements.
    std::uint32_t n = 0;
    std::span<const std::array<float, 4>> texels;
    std::span<const std::uint32_t> data;
    Stride stride;

    static View texture(std::uint32_t w, std::uint32_t h, std::span<const std::array<float, 4>> t) {
        View v; v.kind = Texture; v.w = w; v.h = h; v.texels = t; return v;
    }
    static View cube(std::uint32_t size, std::uint16_t mips, std::span<const std::array<float, 4>> t) {
        View v; v.kind = Cube; v.w = size; v.h = size; v.mips = mips; v.texels = t; return v;
    }
    static View raw_uav(std::uint32_t dwords) { View v; v.kind = RawUav; v.n = dwords; return v; }
    static View texture_uav(std::uint32_t w, std::uint32_t h) { View v; v.kind = TextureUav; v.w = w; v.h = h; return v; }
    static View uint_texture_uav(std::uint32_t w, std::uint32_t h) {
        View v; v.kind = UintTextureUav; v.w = w; v.h = h; return v;
    }
    static View half_uav(std::uint32_t elements) { View v; v.kind = HalfUav; v.n = elements; return v; }
    static View cbv(std::span<const std::uint32_t> d) { View v; v.kind = Cbv; v.data = d; return v; }
    static View buffer(std::span<const std::uint32_t> d, Stride s) {
        View v; v.kind = Buffer; v.data = d; v.stride = s; return v;
    }
};

// Bindings for Host::dispatch: root arguments, the view heap in order, the sampler heap, and the
// group counts.
struct Dispatch {
    std::vector<std::pair<std::uint32_t, Root>> roots;
    std::span<const View> views;
    std::span<const D3D12_SAMPLER_DESC> samplers;
    std::array<std::uint32_t, 3> groups{1, 1, 1};
};

class Host {
public:
    ComPtr<ID3D12Device> device;

    // The default adapter, or WARP with warp (or X2W_SHADERCFG_WARP=1). Throws Error (also when the
    // default device is the software adapter or lacks SM 6.6: tests then skip).
    explicit Host(bool warp);
    static Host create(bool warp) { return Host(warp); }

    // The root signature translated shaders expect: user data per stage at b0, immediate
    // samplers per stage at s<n>, heaps directly indexed.
    ComPtr<ID3D12RootSignature> root_signature(const Stage& vs, const Stage& ps) const;
    // The root signature for these stages: user data of stage k is root parameter k, root views
    // follow.
    ComPtr<ID3D12RootSignature> root_signature_for(std::span<const std::pair<Stage, D3D12_SHADER_VISIBILITY>> stages) const;
    ComPtr<ID3D12RootSignature> root_signature_for(std::initializer_list<std::pair<Stage, D3D12_SHADER_VISIBILITY>> stages) const {
        return root_signature_for(std::span(stages.begin(), stages.size()));
    }

    // A compute pipeline; user data is root parameter 0.
    Pipeline compute_pipeline(const Stage& cs) const;
    ComPtr<ID3D12PipelineState> compute_pipeline_with(ID3D12RootSignature* root, std::span<const std::uint8_t> cs) const;

    // A graphics pipeline from a vertex and a pixel shader, with float or integer 32-bit targets
    // as the pixel shader writes them.
    Pipeline pipeline(const Stage& vs, const Stage& ps) const;
    // A graphics pipeline with the given root signature; vertex inputs packed in register order.
    ComPtr<ID3D12PipelineState> pipeline_with(ID3D12RootSignature* root, std::span<const std::uint8_t> vs,
                                              std::span<const std::uint8_t> ps) const;
    // As pipeline_with, with a geometry shader between the two (the rectangle stage,
    // rect_list_gs); gs empty: none.
    ComPtr<ID3D12PipelineState> pipeline_with_gs(ID3D12RootSignature* root, std::span<const std::uint8_t> vs,
                                                 std::span<const std::uint8_t> gs,
                                                 std::span<const std::uint8_t> ps) const;
    // A mesh pipeline from a mesh and a pixel shader (user data of the mesh stage is root
    // parameter 0, the pixel stage's 1).
    Pipeline mesh_pipeline(const Stage& ms, const Stage& ps) const;
    // A mesh pipeline with the given root signature: the pipeline-state stream form, with the
    // targets pipeline_with would use.
    ComPtr<ID3D12PipelineState> mesh_pipeline_with(ID3D12RootSignature* root, std::span<const std::uint8_t> ms,
                                                   std::span<const std::uint8_t> ps) const;

    // Draws one triangle into an RGBA32F target and returns its texels.
    std::vector<std::array<float, 4>> draw(ID3D12RootSignature* root, ID3D12PipelineState* pso, const Draw& d) const;
    // Runs a compute pipeline and returns each UAV's contents in heap order (texels as float
    // bits, row by row).
    std::vector<std::vector<std::uint32_t>> dispatch(ID3D12RootSignature* root, ID3D12PipelineState* pso,
                                                     const Dispatch& d) const;

    // The error text with the debug layer's messages since the last call, if the layer is on
    // (X2W_SHADERCFG_D3D12_DEBUG).
    std::string explained(HRESULT hr) const;
};

// A vertex shader that feeds ps's inputs (zeros) and a position. (Needs native.h.)
HostShader stub_vs(const HostShader& ps);
// A pixel shader that reads nothing and writes target 0. (Needs native.h.)
HostShader stub_ps();
// The containers of stub_vs (from the pixel shader's container) and stub_ps.
std::vector<std::uint8_t> stub_vs_dxil(std::span<const std::uint8_t> ps);
std::vector<std::uint8_t> stub_ps_dxil();

// Format the system message for an HRESULT.
std::string message(HRESULT hr);

}  // namespace x2w::vs::host
