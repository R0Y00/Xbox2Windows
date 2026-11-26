// Decode a console pipeline-state stream for host pipeline creation. Subobjects begin with a 32-bit type and are
// padded to eight bytes. Data starts at +4, or +8 for pointer-bearing subobjects. Console amplification and mesh
// shader types are 23 and 24 (desktop 24 and 25). Stop at an unknown size or truncated subobject.
//
// The caller maps console formats through VpFormatMap so pipelines use the same desktop formats and packed-target
// stand-ins as their textures. Parsing requires no device.
#pragma once

#include <d3d12.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace x2w_gpu {

// Console stage numbers in a stream (subobject types).
enum VpStage : std::uint32_t {
    vp_stage_none = 0,
    vp_stage_vertex = 1,
    vp_stage_pixel = 2,
    vp_stage_domain = 3,
    vp_stage_hull = 4,
    vp_stage_geometry = 5,
    vp_stage_compute = 6,
    vp_stage_amplification = 23,
    vp_stage_mesh = 24,
};

inline const char* vp_stage_name(std::uint32_t stage)
{
    switch (stage) {
    case vp_stage_vertex: return "vertex";
    case vp_stage_pixel: return "pixel";
    case vp_stage_domain: return "domain";
    case vp_stage_hull: return "hull";
    case vp_stage_geometry: return "geometry";
    case vp_stage_compute: return "compute";
    case vp_stage_amplification: return "amplification";
    case vp_stage_mesh: return "mesh";
    default: return "none";
    }
}

// Subobject size including the type, and stage for shader subobjects. Zero means unknown size.
struct VpSubobject {
    std::size_t size;
    std::uint32_t stage;
};

inline VpSubobject vp_subobject(std::uint32_t kind)
{
    switch (kind) {
    case 1: case 2: case 3: case 4: case 5: case 6: case 23: case 24: return {24, kind};
    case 0: return {16, 0};
    case 12: case 19: return {24, 0};
    case 7: case 15: return {40, 0};
    case 8: return {336, 0};
    case 10: return {48, 0};
    case 11: return {56, 0};
    case 17: return {16, 0};
    case 21: return {64, 0};
    case 22: return {32, 0};
    case 9: case 13: case 14: case 16: case 18: case 20: return {8, 0};
    default: return {0, 0};
    }
}

// The console's own primitive topology types; both draw as triangles on the host.
constexpr int vp_topology_rectangles = 19;
constexpr int vp_topology_quads = 32;

inline D3D12_PRIMITIVE_TOPOLOGY_TYPE vp_host_topology_type(D3D12_PRIMITIVE_TOPOLOGY_TYPE t)
{
    const int v = static_cast<int>(t);
    return v == vp_topology_rectangles || v == vp_topology_quads ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE : t;
}

// Map console depth formats 118–120 to desktop formats.
inline DXGI_FORMAT vp_pc_format(std::uint32_t format)
{
    switch (format) {
    case 118: return DXGI_FORMAT_D24_UNORM_S8_UINT;
    case 119: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case 120: return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
    default: return static_cast<DXGI_FORMAT>(format);
    }
}

// Console format -> host format, per use. Null members fall back on vp_pc_format.
struct VpFormatMap {
    DXGI_FORMAT (*target)(std::uint32_t) = nullptr;  // render target formats (subobject 15)
    DXGI_FORMAT (*depth)(std::uint32_t) = nullptr;   // depth-stencil format (16)
    DXGI_FORMAT (*input)(std::uint32_t) = nullptr;   // input elements (12)
};

inline DXGI_FORMAT vp_map(DXGI_FORMAT (*f)(std::uint32_t), std::uint32_t format)
{
    return f ? f(format) : vp_pc_format(format);
}

// Bytes of one element of `format` and texels per element. {0, 0}: not known.
struct VpElement {
    std::uint32_t bytes;
    std::uint32_t texels;
};

inline VpElement vp_element(std::uint32_t f)
{
    if (f >= 1 && f <= 4) return {16, 1};
    if (f >= 5 && f <= 8) return {12, 1};
    if (f >= 9 && f <= 22) return {8, 1};
    if ((f >= 23 && f <= 47) || f == 67 || (f >= 87 && f <= 93) || f == 116 || f == 117 || f == 189 || f == 191) return {4, 1};
    if (f >= 118 && f <= 120) return {3, 1};
    if ((f >= 48 && f <= 59) || f == 68 || f == 69 || f == 85 || f == 86 || f == 115) return {2, 1};
    if ((f >= 60 && f <= 65) || f == 190) return {1, 1};
    if ((f >= 70 && f <= 72) || (f >= 79 && f <= 81)) return {8, 4};
    if ((f >= 73 && f <= 78) || (f >= 82 && f <= 84) || (f >= 94 && f <= 99)) return {16, 4};
    return {0, 0};
}

// Integer render target formats: D3D12 cannot blend into them.
inline bool vp_integer_format(DXGI_FORMAT f)
{
    switch (f) {
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32_UINT: case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT:
    case DXGI_FORMAT_R8_UINT: case DXGI_FORMAT_R8_SINT:
        return true;
    default:
        return false;
    }
}

// The title's blend state as the host takes it: blending off on integer render targets (the console ignores it
// there, D3D12 refuses the pipeline). With one description for every target, each target gets its own copy so the
// others keep theirs.
inline D3D12_BLEND_DESC vp_host_blend(const D3D12_BLEND_DESC& blend, const D3D12_RT_FORMAT_ARRAY& targets)
{
    const UINT count = std::min<UINT>(targets.NumRenderTargets, 8);
    const bool per_target = blend.IndependentBlendEnable != FALSE;
    const auto blends = [&](UINT i) { return blend.RenderTarget[per_target ? i : 0].BlendEnable != FALSE; };
    bool any = false;
    for (UINT i = 0; i < count; ++i) any = any || (vp_integer_format(targets.RTFormats[i]) && blends(i));
    if (!any) return blend;
    D3D12_BLEND_DESC host = blend;
    if (!per_target) {
        host.IndependentBlendEnable = TRUE;
        for (auto& t : host.RenderTarget) t = blend.RenderTarget[0];
    }
    for (UINT i = 0; i < count; ++i)
        if (vp_integer_format(targets.RTFormats[i])) host.RenderTarget[i].BlendEnable = FALSE;
    return host;
}

// One input element as the draw path reads it: vertex buffer slot, byte offset, format.
struct VpInput {
    std::uint32_t slot;
    std::uint32_t offset;
    std::uint32_t format;
    bool per_instance;
};

// A stream's graphics state, owned (the title's stream lives only for the creation call).
// Subobjects the stream leaves out keep D3D12's defaults.
struct VpGraphicsState {
    D3D12_BLEND_DESC blend{};
    UINT sample_mask = ~0u;
    D3D12_RASTERIZER_DESC rasterizer{};
    D3D12_DEPTH_STENCIL_DESC depth{};
    bool depth_bounds = false;
    // Input elements with SemanticName left null; the names are kept in `names` (input_layout()
    // puts the pointers in).
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputs;
    std::vector<std::string> names;
    // The layout had more than 32 elements, or names or elements were not readable: the pipeline
    // is refused.
    bool bad_inputs = false;
    D3D12_INDEX_BUFFER_STRIP_CUT_VALUE strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED;
    D3D12_RT_FORMAT_ARRAY targets{};
    DXGI_FORMAT depth_format = DXGI_FORMAT_UNKNOWN;
    DXGI_SAMPLE_DESC samples{1, 0};

    VpGraphicsState()
    {
        D3D12_RENDER_TARGET_BLEND_DESC target{};
        target.SrcBlend = D3D12_BLEND_ONE;
        target.DestBlend = D3D12_BLEND_ZERO;
        target.BlendOp = D3D12_BLEND_OP_ADD;
        target.SrcBlendAlpha = D3D12_BLEND_ONE;
        target.DestBlendAlpha = D3D12_BLEND_ZERO;
        target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        target.LogicOp = D3D12_LOGIC_OP_NOOP;
        target.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        for (auto& t : blend.RenderTarget) t = target;
        rasterizer.FillMode = D3D12_FILL_MODE_SOLID;
        rasterizer.CullMode = D3D12_CULL_MODE_BACK;
        rasterizer.DepthClipEnable = TRUE;
        const D3D12_DEPTH_STENCILOP_DESC keep{D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP,
                                              D3D12_COMPARISON_FUNC_ALWAYS};
        depth.DepthEnable = TRUE;
        depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        depth.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        depth.StencilReadMask = 0xff;
        depth.StencilWriteMask = 0xff;
        depth.FrontFace = keep;
        depth.BackFace = keep;
    }

    // A pass over the screen: no vertex input, no depth or stencil, a target, triangles.
    bool screen_pass() const
    {
        return inputs.empty() &&
            (depth_format == DXGI_FORMAT_UNKNOWN || (!depth.DepthEnable && !depth.StencilEnable)) &&
            targets.NumRenderTargets > 0 && topology == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }

    D3D12_DEPTH_STENCIL_DESC1 depth1() const
    {
        D3D12_DEPTH_STENCIL_DESC1 d{};
        d.DepthEnable = depth.DepthEnable;
        d.DepthWriteMask = depth.DepthWriteMask;
        d.DepthFunc = depth.DepthFunc;
        d.StencilEnable = depth.StencilEnable;
        d.StencilReadMask = depth.StencilReadMask;
        d.StencilWriteMask = depth.StencilWriteMask;
        d.FrontFace = depth.FrontFace;
        d.BackFace = depth.BackFace;
        d.DepthBoundsTestEnable = depth_bounds ? TRUE : FALSE;
        return d;
    }

    // The input elements with their names in place (valid while this state lives unchanged).
    std::vector<D3D12_INPUT_ELEMENT_DESC> input_layout() const
    {
        std::vector<D3D12_INPUT_ELEMENT_DESC> out = inputs;
        for (std::size_t i = 0; i < out.size() && i < names.size(); ++i) out[i].SemanticName = names[i].c_str();
        return out;
    }

    // The vertex buffer slots the input layout reads (bit per slot).
    std::uint32_t slots() const
    {
        std::uint32_t m = 0;
        for (const auto& e : inputs) m |= 1u << std::min<UINT>(e.InputSlot, 31);
        return m;
    }

    // The input elements with their byte offsets worked out. An appended element starts at the
    // next multiple of min(4, its size) after the previous one, where D3D12 places it.
    std::vector<VpInput> elements() const
    {
        std::uint32_t next[32] = {};
        std::vector<VpInput> out;
        out.reserve(inputs.size());
        for (const auto& e : inputs) {
            const std::uint32_t slot = std::min<UINT>(e.InputSlot, 31);
            const std::uint32_t format = static_cast<std::uint32_t>(e.Format);
            const std::uint32_t bytes = vp_element(format).bytes;
            std::uint32_t offset = e.AlignedByteOffset;
            if (e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT) {
                const std::uint64_t align = std::clamp<std::uint32_t>(bytes, 1, 4);
                const std::uint64_t rounded = (static_cast<std::uint64_t>(next[slot]) + align - 1) / align * align;
                offset = rounded > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(rounded);
            }
            const std::uint64_t end = static_cast<std::uint64_t>(offset) + bytes;
            next[slot] = end > 0xffffffffull ? 0xffffffffu : static_cast<std::uint32_t>(end);
            out.push_back({slot, offset, format, e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA});
        }
        return out;
    }

    // Everything the state holds, as bytes for a key (names included).
    void append_key(std::string& key) const
    {
        const auto put = [&key](const void* p, std::size_t n) { key.append(static_cast<const char*>(p), n); };
        put(&blend, sizeof(blend));
        put(&sample_mask, sizeof(sample_mask));
        put(&rasterizer, sizeof(rasterizer));
        put(&depth, sizeof(depth));
        const char bounds = depth_bounds ? 1 : 0;
        put(&bounds, 1);
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            D3D12_INPUT_ELEMENT_DESC e = inputs[i];
            e.SemanticName = nullptr;
            put(&e, sizeof(e));
            if (i < names.size()) key += names[i];
            key += '\0';
        }
        put(&strip_cut, sizeof(strip_cut));
        put(&topology, sizeof(topology));
        put(&targets, sizeof(targets));
        put(&depth_format, sizeof(depth_format));
        put(&samples, sizeof(samples));
    }
};

// Whether `bytes` at `p` may be read; the default trusts the caller.
using VpReadable = bool (*)(const void* p, std::size_t bytes);

// Takes in one subobject of a stream: its type, then its data at +4 (+8 for those holding pointers). `at` points at
// a whole subobject of type `kind`.
inline void vp_read_subobject(VpGraphicsState& g, std::uint32_t kind, const unsigned char* at, const VpFormatMap& map,
                              VpReadable readable = nullptr)
{
    const unsigned char* data = at + 4;
    switch (kind) {
    case 8: std::memcpy(&g.blend, data, sizeof(g.blend)); break;
    case 9: std::memcpy(&g.sample_mask, data, sizeof(g.sample_mask)); break;
    case 10: std::memcpy(&g.rasterizer, data, sizeof(g.rasterizer)); break;
    case 11:
        std::memcpy(&g.depth, data, sizeof(g.depth));
        g.depth_bounds = false;
        break;
    case 21: {
        D3D12_DEPTH_STENCIL_DESC1 d{};
        std::memcpy(&d, data, sizeof(d));
        g.depth_bounds = d.DepthBoundsTestEnable != FALSE;
        g.depth.DepthEnable = d.DepthEnable;
        g.depth.DepthWriteMask = d.DepthWriteMask;
        g.depth.DepthFunc = d.DepthFunc;
        g.depth.StencilEnable = d.StencilEnable;
        g.depth.StencilReadMask = d.StencilReadMask;
        g.depth.StencilWriteMask = d.StencilWriteMask;
        g.depth.FrontFace = d.FrontFace;
        g.depth.BackFace = d.BackFace;
        break;
    }
    case 12: {
        D3D12_INPUT_LAYOUT_DESC layout{};
        std::memcpy(&layout, at + 8, sizeof(layout));
        if (!layout.pInputElementDescs) return;
        g.inputs.clear();
        g.names.clear();
        UINT count = layout.NumElements;
        if (count > D3D12_IA_VERTEX_INPUT_STRUCTURE_ELEMENT_COUNT) {
            g.bad_inputs = true;
            count = 0;
        }
        if (count && readable && !readable(layout.pInputElementDescs, count * sizeof(D3D12_INPUT_ELEMENT_DESC))) {
            g.bad_inputs = true;
            count = 0;
        }
        for (UINT i = 0; i < count; ++i) {
            D3D12_INPUT_ELEMENT_DESC e{};
            std::memcpy(&e, layout.pInputElementDescs + i, sizeof(e));
            std::string name;
            if (e.SemanticName) {
                // Semantic names are short; one past 255 characters, or not readable, is refused.
                const char* s = e.SemanticName;
                bool ended = false;
                for (std::size_t n = 0; n < 256; ++n) {
                    if (readable && !readable(s + n, 1)) break;
                    if (!s[n]) {
                        ended = true;
                        break;
                    }
                    name += s[n];
                }
                if (!ended) g.bad_inputs = true;
            }
            e.SemanticName = nullptr;
            e.Format = vp_map(map.input, static_cast<std::uint32_t>(e.Format));
            g.inputs.push_back(e);
            g.names.push_back(std::move(name));
        }
        break;
    }
    case 13: {
        std::int32_t v = 0;
        std::memcpy(&v, data, sizeof(v));
        g.strip_cut = static_cast<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>(v);
        break;
    }
    case 14: {
        std::int32_t v = 0;
        std::memcpy(&v, data, sizeof(v));
        g.topology = static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(v);
        break;
    }
    case 15:
        std::memcpy(&g.targets, data, sizeof(g.targets));
        for (auto& f : g.targets.RTFormats) f = vp_map(map.target, static_cast<std::uint32_t>(f));
        break;
    case 16: {
        std::uint32_t f = 0;
        std::memcpy(&f, data, sizeof(f));
        g.depth_format = vp_map(map.depth, f);
        break;
    }
    case 17: std::memcpy(&g.samples, data, sizeof(g.samples)); break;
    default: break;
    }
}

// A shader stage in a stream: its console stage number and the title's bytecode.
struct VpStreamShader {
    std::uint32_t stage;
    const void* code;
    std::size_t size;
};

struct VpStream {
    std::vector<VpStreamShader> shaders;
    VpGraphicsState graphics;
    const void* root = nullptr;     // the title's root signature (subobject 0), as passed
    std::uint32_t stopped_kind = 0; // a type of unknown size ended the walk (stopped is true)
    std::size_t stopped_at = 0;
    bool stopped = false;
    bool truncated = false;          // a subobject ran past the stream's end
};

// Walks a stream. `stream` is `size` readable bytes (the caller checks that and should pass a copy). Shader stages
// with null code or no bytes are left out.
inline VpStream vp_parse_stream(const void* stream, std::size_t size, const VpFormatMap& map, VpReadable readable = nullptr)
{
    VpStream out;
    const auto* bytes = static_cast<const unsigned char*>(stream);
    std::size_t at = 0;
    while (bytes && at + 8 <= size) {
        std::uint32_t kind = 0;
        std::memcpy(&kind, bytes + at, sizeof(kind));
        const VpSubobject sub = vp_subobject(kind);
        if (!sub.size) {
            out.stopped = true;
            out.stopped_kind = kind;
            out.stopped_at = at;
            break;
        }
        if (sub.size > size - at) {
            out.truncated = true;
            out.stopped_kind = kind;
            out.stopped_at = at;
            break;
        }
        if (kind == 0) std::memcpy(&out.root, bytes + at + 8, sizeof(out.root));
        if (sub.stage) {
            D3D12_SHADER_BYTECODE code{};
            std::memcpy(&code, bytes + at + 8, sizeof(code));
            if (code.pShaderBytecode && code.BytecodeLength) out.shaders.push_back({sub.stage, code.pShaderBytecode, code.BytecodeLength});
        }
        vp_read_subobject(out.graphics, kind, bytes + at, map, readable);
        at += sub.size;
    }
    return out;
}

}  // namespace x2w_gpu

// d3d12_xs_device.cpp includes only this WP2 header at its top (outside every namespace); the root signature and
// pipeline builders and the lazy cache that gpu_pipeline.inl uses come with it. Define
// X2W_GPU_PIPELINE_STATE_ONLY for the header-only check, which links no d3d12.
#ifndef X2W_GPU_PIPELINE_STATE_ONLY
#include "startup_stubs/gpu_pipeline_pending.h"
#endif
