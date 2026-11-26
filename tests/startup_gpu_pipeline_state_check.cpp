// Checks src/startup_stubs/gpu_pipeline_state.h without a device: the stream walk with the
// console's subobject numbering (MS = 24, AS = 23), the graphics state defaults, DEPTH_STENCIL1
// folding, the stop on an unknown or truncated subobject, host_blend on integer targets and the
// offsets of appended input elements. The cases are the upstream Rust tests (vg/pipeline.rs
// streams_yield_their_shaders, vg/pipeline/host.rs integer_targets_do_not_blend,
// blend_states_without_integer_blending_are_kept, appended_elements_are_aligned,
// a_huge_explicit_offset_does_not_overflow) plus the stops.
//
// Negative controls: --mutate desktop-numbering writes the mesh stage with the desktop number 25,
// which the console walk does not know, so the shaders are not found; --mutate blend-kept expects
// the title's blend state back unchanged on an integer target.
#define X2W_GPU_PIPELINE_STATE_ONLY
#include "startup_stubs/gpu_pipeline_state.h"

#include <cstdio>
#include <cstring>
#include <vector>

#ifdef _MSC_VER
#pragma warning(disable : 4324)  // the subobjects are padded to 8 on purpose
#endif

using namespace x2w_gpu;

namespace {

enum class Mutation { none, desktop_numbering, blend_kept };
Mutation mutation = Mutation::none;
int failures = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL line %d: %s\n", __LINE__, #cond);            \
            ++failures;                                                    \
        }                                                                  \
    } while (0)

void put64(std::vector<unsigned char>& s, std::size_t at, std::uint64_t v) { std::memcpy(&s[at], &v, 8); }

// The stream layout the title builds (upstream test, title+0x86f090): root signature, five shader
// slots (the first a mesh shader), blend, sample mask, rasterizer, depth-stencil 1, sample
// description, input layout, topology, render target and depth formats: 0x2b0 bytes.
void title_stream()
{
    const unsigned char ms[7] = {9, 9, 9, 9, 9, 9, 9};
    const unsigned char ps[5] = {4, 4, 4, 4, 4};
    std::vector<unsigned char> s(0x2b0, 0);
    const std::uint64_t mesh_kind = mutation == Mutation::desktop_numbering ? 25 : 24;
    const std::pair<std::size_t, std::uint64_t> kinds[] = {
        {0x10, mesh_kind}, {0x28, 2}, {0x40, 3}, {0x58, 4}, {0x70, 5}, {0x88, 8}, {0x1d8, 9}, {0x1e0, 10},
        {0x210, 21}, {0x250, 17}, {0x260, 12}, {0x278, 14}, {0x280, 15}, {0x2a8, 16}};
    for (const auto& k : kinds) put64(s, k.first, k.second);
    put64(s, 0x18, reinterpret_cast<std::uint64_t>(ms));
    put64(s, 0x20, sizeof(ms));
    put64(s, 0x30, reinterpret_cast<std::uint64_t>(ps));
    put64(s, 0x38, sizeof(ps));
    // Topology type 3 (triangles), one render target of format 28.
    put64(s, 0x278, 14 | 3ull << 32);
    put64(s, 0x280, 15 | 28ull << 32);
    put64(s, 0x2a0, 1ull << 32);
    // DEPTH_STENCIL1 at 0x210 with the depth bounds test on (data at +4: DepthBoundsTestEnable at +52).
    const std::uint32_t one = 1;
    std::memcpy(&s[0x210 + 4 + offsetof(D3D12_DEPTH_STENCIL_DESC1, DepthBoundsTestEnable)], &one, 4);
    // Depth format 118 (console D24S8).
    put64(s, 0x2a8, 16 | 118ull << 32);
    const VpStream out = vp_parse_stream(s.data(), s.size(), VpFormatMap{});
    EXPECT(!out.stopped && !out.truncated);
    EXPECT(out.shaders.size() == 2);
    if (out.shaders.size() == 2) {
        EXPECT(out.shaders[0].stage == vp_stage_mesh && out.shaders[0].code == ms && out.shaders[0].size == sizeof(ms));
        EXPECT(out.shaders[1].stage == vp_stage_pixel && out.shaders[1].code == ps && out.shaders[1].size == sizeof(ps));
    }
    const VpGraphicsState& g = out.graphics;
    EXPECT(g.topology == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    EXPECT(g.targets.NumRenderTargets == 1 && g.targets.RTFormats[0] == DXGI_FORMAT_R8G8B8A8_UNORM);
    EXPECT(g.depth_bounds);
    EXPECT(g.depth_format == DXGI_FORMAT_D24_UNORM_S8_UINT);
    // The zeroed subobjects replaced the defaults (blend, mask, rasterizer, depth read as zero).
    EXPECT(g.sample_mask == 0 && g.rasterizer.CullMode == 0 && !g.depth.DepthEnable);
    // The input layout with a null pointer keeps none.
    EXPECT(g.inputs.empty() && !g.bad_inputs);
}

void defaults()
{
    const VpGraphicsState g;
    EXPECT(g.sample_mask == 0xffffffffu);
    EXPECT(g.rasterizer.FillMode == D3D12_FILL_MODE_SOLID && g.rasterizer.CullMode == D3D12_CULL_MODE_BACK && g.rasterizer.DepthClipEnable);
    EXPECT(g.depth.DepthEnable && g.depth.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL && g.depth.DepthFunc == D3D12_COMPARISON_FUNC_LESS);
    EXPECT(g.depth.StencilReadMask == 0xff && g.depth.StencilWriteMask == 0xff && !g.depth.StencilEnable);
    EXPECT(g.depth.FrontFace.StencilFunc == D3D12_COMPARISON_FUNC_ALWAYS && g.depth.BackFace.StencilPassOp == D3D12_STENCIL_OP_KEEP);
    for (const auto& t : g.blend.RenderTarget) {
        EXPECT(!t.BlendEnable && t.SrcBlend == D3D12_BLEND_ONE && t.DestBlend == D3D12_BLEND_ZERO);
        EXPECT(t.RenderTargetWriteMask == D3D12_COLOR_WRITE_ENABLE_ALL && t.LogicOp == D3D12_LOGIC_OP_NOOP);
    }
    EXPECT(g.samples.Count == 1 && g.samples.Quality == 0);
    EXPECT(g.topology == D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED && g.depth_format == DXGI_FORMAT_UNKNOWN);
    EXPECT(g.strip_cut == D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED && !g.depth_bounds);
}

// Subobject 21 sets depth and the bounds flag; a later 11 clears the flag.
void depth_folding()
{
    struct alignas(8) Sub21 {
        std::uint32_t type;
        D3D12_DEPTH_STENCIL_DESC1 d;
    } a{};
    a.type = 21;
    a.d.DepthEnable = FALSE;
    a.d.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    a.d.StencilEnable = TRUE;
    a.d.StencilReadMask = 0x0f;
    a.d.DepthBoundsTestEnable = TRUE;
    VpGraphicsState g;
    vp_read_subobject(g, 21, reinterpret_cast<const unsigned char*>(&a), VpFormatMap{});
    EXPECT(g.depth_bounds && !g.depth.DepthEnable && g.depth.StencilEnable && g.depth.StencilReadMask == 0x0f);
    EXPECT(g.depth1().DepthBoundsTestEnable && g.depth1().DepthFunc == D3D12_COMPARISON_FUNC_ALWAYS);
    struct alignas(8) Sub11 {
        std::uint32_t type;
        D3D12_DEPTH_STENCIL_DESC d;
    } b{};
    b.type = 11;
    b.d.DepthEnable = TRUE;
    vp_read_subobject(g, 11, reinterpret_cast<const unsigned char*>(&b), VpFormatMap{});
    EXPECT(!g.depth_bounds && g.depth.DepthEnable);
}

// A type of unknown size stops the walk (the rest ignored); a subobject past the end does too.
void stops()
{
    std::vector<unsigned char> s(0x40, 0);
    put64(s, 0x00, 14 | 4ull << 32);  // topology type 4 (patch): kept as given
    put64(s, 0x08, 99);               // unknown
    put64(s, 0x10, 9 | 7ull << 32);   // sample mask after it: not read
    VpStream out = vp_parse_stream(s.data(), s.size(), VpFormatMap{});
    EXPECT(out.stopped && out.stopped_kind == 99 && out.stopped_at == 8 && !out.truncated);
    EXPECT(out.graphics.topology == 4 && out.graphics.sample_mask == 0xffffffffu);
    // A blend subobject (336 bytes) in a 0x40-byte stream.
    std::vector<unsigned char> t(0x40, 0);
    put64(t, 0x00, 8);
    out = vp_parse_stream(t.data(), t.size(), VpFormatMap{});
    EXPECT(out.truncated && out.stopped_kind == 8 && out.stopped_at == 0);
    EXPECT(out.graphics.blend.RenderTarget[0].SrcBlend == D3D12_BLEND_ONE);  // untouched
    // AS (23) is a stage; a null shader is left out.
    std::vector<unsigned char> u(0x30, 0);
    put64(u, 0x00, 23);
    put64(u, 0x18, 6);
    out = vp_parse_stream(u.data(), u.size(), VpFormatMap{});
    EXPECT(out.shaders.empty() && !out.stopped && !out.truncated);
}

void topologies()
{
    EXPECT(vp_host_topology_type(static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(19)) == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    EXPECT(vp_host_topology_type(static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(32)) == D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    EXPECT(vp_host_topology_type(D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE) == D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
}

D3D12_RT_FORMAT_ARRAY targets(std::initializer_list<DXGI_FORMAT> formats)
{
    D3D12_RT_FORMAT_ARRAY t{};
    for (DXGI_FORMAT f : formats) t.RTFormats[t.NumRenderTargets++] = f;
    return t;
}

D3D12_BLEND_DESC blending(bool independent)
{
    D3D12_BLEND_DESC b = VpGraphicsState().blend;
    b.IndependentBlendEnable = independent;
    for (auto& t : b.RenderTarget) {
        t.BlendEnable = TRUE;
        t.SrcBlend = D3D12_BLEND_SRC_ALPHA;
        t.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    }
    return b;
}

bool same(const D3D12_BLEND_DESC& a, const D3D12_BLEND_DESC& b) { return std::memcmp(&a, &b, sizeof(a)) == 0; }

void blends()
{
    const auto t = targets({DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8_UINT});
    for (bool independent : {false, true}) {
        const D3D12_BLEND_DESC host = mutation == Mutation::blend_kept ? blending(independent) : vp_host_blend(blending(independent), t);
        EXPECT(host.IndependentBlendEnable);
        EXPECT(host.RenderTarget[0].BlendEnable && host.RenderTarget[1].BlendEnable && !host.RenderTarget[2].BlendEnable);
        EXPECT(host.RenderTarget[2].SrcBlend == D3D12_BLEND_SRC_ALPHA && host.RenderTarget[0].DestBlend == D3D12_BLEND_INV_SRC_ALPHA);
    }
    // No integer target; an integer target that does not blend; integer formats past the targets in use.
    D3D12_BLEND_DESC b = blending(false);
    EXPECT(same(vp_host_blend(b, targets({DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R11G11B10_FLOAT})), b));
    b = blending(true);
    b.RenderTarget[1].BlendEnable = FALSE;
    EXPECT(same(vp_host_blend(b, targets({DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R32_UINT})), b));
    auto past = targets({DXGI_FORMAT_R8G8B8A8_UNORM});
    past.RTFormats[1] = DXGI_FORMAT_R8_UINT;
    EXPECT(same(vp_host_blend(blending(false), past), blending(false)));
}

std::vector<std::uint32_t> offsets(std::initializer_list<std::pair<DXGI_FORMAT, UINT>> elements)
{
    VpGraphicsState g;
    for (const auto& e : elements) {
        D3D12_INPUT_ELEMENT_DESC d{};
        d.Format = e.first;
        d.AlignedByteOffset = e.second;
        g.inputs.push_back(d);
        g.names.push_back("A");
    }
    std::vector<std::uint32_t> out;
    for (const auto& i : g.elements()) out.push_back(i.offset);
    return out;
}

void inputs()
{
    const UINT append = D3D12_APPEND_ALIGNED_ELEMENT;
    EXPECT((offsets({{DXGI_FORMAT_R8G8_UNORM, 0}, {DXGI_FORMAT_R32G32_FLOAT, append}}) == std::vector<std::uint32_t>{0, 4}));
    EXPECT((offsets({{DXGI_FORMAT_R8_UNORM, 0}, {DXGI_FORMAT_R16_FLOAT, append}, {DXGI_FORMAT_R8_UNORM, append}}) ==
            std::vector<std::uint32_t>{0, 2, 4}));
    EXPECT((offsets({{DXGI_FORMAT_R8G8B8A8_UNORM, 0}, {DXGI_FORMAT_R32G32_FLOAT, append}, {DXGI_FORMAT_R32G32B32_FLOAT, append}}) ==
            std::vector<std::uint32_t>{0, 4, 12}));
    const auto huge = offsets({{DXGI_FORMAT_R32G32B32A32_FLOAT, 0xffffffffu - 8}, {DXGI_FORMAT_R32_FLOAT, append}});
    EXPECT(huge.size() == 2 && huge[1] == 0xffffffffu);

    // An input layout subobject: names copied, formats mapped, slots and pointers.
    D3D12_INPUT_ELEMENT_DESC e[2] = {
        {"POSITION", 0, static_cast<DXGI_FORMAT>(116), 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 3, DXGI_FORMAT_R16G16_FLOAT, 2, D3D12_APPEND_ALIGNED_ELEMENT, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1}};
    struct alignas(8) Sub12 {
        std::uint32_t type, pad;
        D3D12_INPUT_LAYOUT_DESC layout;
    } sub{12, 0, {e, 2}};
    VpFormatMap map;
    map.input = [](std::uint32_t f) { return f == 116 ? DXGI_FORMAT_R10G10B10A2_UNORM : static_cast<DXGI_FORMAT>(f); };
    VpGraphicsState g;
    vp_read_subobject(g, 12, reinterpret_cast<const unsigned char*>(&sub), map);
    EXPECT(g.inputs.size() == 2 && g.names.size() == 2 && !g.bad_inputs);
    if (g.inputs.size() == 2) {
        EXPECT(g.names[0] == "POSITION" && g.names[1] == "TEXCOORD");
        EXPECT(g.inputs[0].SemanticName == nullptr && g.inputs[0].Format == DXGI_FORMAT_R10G10B10A2_UNORM);
        EXPECT(g.slots() == 0x5);
        const auto layout = g.input_layout();
        EXPECT(std::strcmp(layout[1].SemanticName, "TEXCOORD") == 0 && layout[1].SemanticIndex == 3);
        const auto el = g.elements();
        EXPECT(el[1].slot == 2 && el[1].offset == 0 && el[1].per_instance && !el[0].per_instance);
    }
    // More than 32 elements: refused.
    Sub12 many{12, 0, {e, 33}};
    VpGraphicsState h;
    vp_read_subobject(h, 12, reinterpret_cast<const unsigned char*>(&many), map);
    EXPECT(h.bad_inputs && h.inputs.empty());
}

void formats()
{
    // Render targets and depth go through the caller's map; the default mapping uses pc_format.
    EXPECT(vp_pc_format(118) == DXGI_FORMAT_D24_UNORM_S8_UINT && vp_pc_format(119) == DXGI_FORMAT_R24_UNORM_X8_TYPELESS &&
           vp_pc_format(120) == DXGI_FORMAT_X24_TYPELESS_G8_UINT && vp_pc_format(28) == DXGI_FORMAT_R8G8B8A8_UNORM);
    struct alignas(8) Sub15 {
        std::uint32_t type;
        D3D12_RT_FORMAT_ARRAY t;
    } sub{15, {}};
    sub.t.NumRenderTargets = 2;
    sub.t.RTFormats[0] = static_cast<DXGI_FORMAT>(67);
    sub.t.RTFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
    VpFormatMap map;
    map.target = [](std::uint32_t f) { return f == 67 ? DXGI_FORMAT_R11G11B10_FLOAT : static_cast<DXGI_FORMAT>(f); };
    VpGraphicsState g;
    vp_read_subobject(g, 15, reinterpret_cast<const unsigned char*>(&sub), map);
    EXPECT(g.targets.NumRenderTargets == 2 && g.targets.RTFormats[0] == DXGI_FORMAT_R11G11B10_FLOAT &&
           g.targets.RTFormats[1] == DXGI_FORMAT_R8G8B8A8_UNORM);
}

void sizes()
{
    // The subobject sizes hold the desktop structures at +4 (+8 for pointers).
    EXPECT(vp_subobject(8).size >= 4 + sizeof(D3D12_BLEND_DESC));
    EXPECT(vp_subobject(10).size >= 4 + sizeof(D3D12_RASTERIZER_DESC));
    EXPECT(vp_subobject(11).size >= 4 + sizeof(D3D12_DEPTH_STENCIL_DESC));
    EXPECT(vp_subobject(21).size >= 4 + sizeof(D3D12_DEPTH_STENCIL_DESC1));
    EXPECT(vp_subobject(12).size >= 8 + sizeof(D3D12_INPUT_LAYOUT_DESC));
    EXPECT(vp_subobject(15).size >= 4 + sizeof(D3D12_RT_FORMAT_ARRAY));
    EXPECT(vp_subobject(17).size >= 4 + sizeof(DXGI_SAMPLE_DESC));
    EXPECT(vp_subobject(24).stage == vp_stage_mesh && vp_subobject(23).stage == vp_stage_amplification);
    EXPECT(vp_subobject(25).size == 0 && vp_subobject(26).size == 0);
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::strcmp(argv[1], "--mutate") == 0) {
        if (std::strcmp(argv[2], "desktop-numbering") == 0) mutation = Mutation::desktop_numbering;
        else if (std::strcmp(argv[2], "blend-kept") == 0) mutation = Mutation::blend_kept;
        else return 2;
    } else if (argc != 1) {
        return 2;
    }
    title_stream();
    defaults();
    depth_folding();
    stops();
    topologies();
    blends();
    inputs();
    formats();
    sizes();
    std::printf("startup_gpu_pipeline_state_check: %d failed\n", failures);
    return failures ? 1 : 0;
}
