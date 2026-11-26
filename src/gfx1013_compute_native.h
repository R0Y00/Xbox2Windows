#pragma once
#include "shader_compute_request.h"
#include <array>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

struct D3D12_ROOT_SIGNATURE_DESC1;
struct D3D12_ROOT_DESCRIPTOR_TABLE1;
namespace x2w {
enum class GpuStage;
struct GuestShaderStage;
class UnsupportedComputeShader : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};
enum class ComputeViewKind {
    buffer_srv,
    typed_buffer_srv,
    texture_srv,
    texture_uav,
    raw_vector_srv,
    half_scalar_uav,
    heap_stride_srv,
    raw_buffer_uav,
    typed_buffer_uav,
    half_buffer_uav,
    root_buffer_srv,
    comparison_texture_srv,
    half2_texture_uav,
    half4_texture_uav,
    cube_sample_srv,
    cube_comparison_srv,
    indexed_atomic_uav,
    d16_sample_srv,
    byte_window_srv
};
inline constexpr unsigned compute_resource_heap = ~0u;
inline bool compute_buffer_uav(ComputeViewKind kind) {
    return kind == ComputeViewKind::raw_buffer_uav || kind == ComputeViewKind::typed_buffer_uav ||
           kind == ComputeViewKind::half_buffer_uav || kind == ComputeViewKind::indexed_atomic_uav;
}
struct IndexedAtomicView {
    unsigned stride = 0, count = 0; // Bytes per guest index and number of valid indices.
};
inline bool compute_uav(ComputeViewKind kind) {
    return kind == ComputeViewKind::texture_uav || kind == ComputeViewKind::half_scalar_uav ||
           kind == ComputeViewKind::half2_texture_uav || kind == ComputeViewKind::half4_texture_uav || compute_buffer_uav(kind);
}
struct ComputeViewBinding {
    ComputeViewKind kind;
    unsigned root_parameter;
    unsigned byte_offset;
    unsigned shader_register;
    unsigned required_bytes;
    unsigned array_count = 1, array_position = 0;
    unsigned image_dimension = 1; // ISA dimension field; 1 denotes Texture2D.
    enum class ScalarTable { unknown, reflected_cbv, declared_cbv, srv };
    ScalarTable scalar_table = ScalarTable::unknown;
};
bool apply_reflected_table_use(ComputeViewBinding &binding, const GuestShaderStage &stage,
                               std::optional<unsigned> packed_sgpr = {});
void apply_scalar_table_layout(ComputeViewBinding &binding, const D3D12_ROOT_DESCRIPTOR_TABLE1 &table);
struct ComputeSampler {
    std::array<unsigned, 4> words;
    bool explicit_lod;
    bool comparison = false;
};
struct ComputeConstants {
    unsigned root_parameter;
    unsigned first_sgpr;
    unsigned count;
};
// Both graphics stages read a single desktop b0. Pack each guest root once,
// in parameter order, irrespective of the SGPR order in either stage.
struct GraphicsConstantLayout {
    std::array<unsigned, 16> counts{};
    unsigned offset(unsigned parameter) const {
        unsigned position = 0;
        for (unsigned r = 0; r < parameter; ++r)
            position += counts.at(r);
        return position;
    }
    void include(unsigned parameter, unsigned count) {
        if (parameter >= counts.size() || !count || count > 32)
            throw UnsupportedComputeShader("invalid graphics root constant range");
        if (counts[parameter] < count)
            counts[parameter] = count;
        if (offset(static_cast<unsigned>(counts.size())) > 32)
            throw UnsupportedComputeShader("linked graphics root exceeds 32 words");
    }
};
struct GraphicsLinkage {
    std::string name;
    unsigned index, row, first, width, component, used, kind, interpolation;
};
struct NativeComputeShader {
    std::vector<GraphicsLinkage> inputs, outputs;
    std::vector<ComputeConstants> constants;
    std::string hlsl, dxil_ir;
    std::vector<ComputeViewBinding> bindings;
    std::vector<ComputeSampler> samplers;
    unsigned instruction_count{};
    std::array<unsigned, 3> group{};
    bool vertex_fetch = false;
    std::array<unsigned, 8> pixel_writes{};
};
NativeComputeShader translate_compute(const ComputeRequest &request, unsigned entry_byte_offset,
                                      bool direct_dxil = false);
NativeComputeShader translate_compute_container(ComputeContainer container);
struct NativeVertexElement {
    std::string semantic;
    unsigned index, format, slot, offset, classification, step_rate;
    bool used = false;
};
struct DirectGraphicsPair {
    NativeComputeShader vertex, pixel;
    std::vector<NativeVertexElement> input_layout;
    GraphicsConstantLayout constant_layout() const {
        GraphicsConstantLayout layout;
        for (auto *stage : {&vertex, &pixel})
            for (const auto &block : stage->constants)
                layout.include(block.root_parameter, block.count);
        return layout;
    }
};
DirectGraphicsPair translate_graphics_pair(std::string_view a, std::string_view b,
                                          const D3D12_ROOT_SIGNATURE_DESC1 &layout, std::string_view pixel_entry = {},
                                          std::optional<GpuStage> first_stage = {});
DirectGraphicsPair translate_graphics_pair(std::string_view a, std::string_view b);
DirectGraphicsPair translate_reflected_graphics(GuestShaderStage &vs, GuestShaderStage &ps,
                                                const D3D12_ROOT_SIGNATURE_DESC1 &layout);
std::string native_rectangle_geometry(const std::vector<GraphicsLinkage>& vertex_outputs,
                                      const std::vector<GraphicsLinkage>& pixel_inputs);
NativeComputeShader translate_graphics_container(const std::string &bytes, bool pixel);
} // namespace x2w
