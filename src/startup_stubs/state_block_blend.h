// Experimental (docs/startup-runtime.md): the blend state of a pipeline created through device
// slot 57, read from its state block A (#1216).
//
// State block A holds the eight CB_BLENDn_CONTROL words at +0xD0, one per render target (measured
// on #1216). They are decoded with cb_blend.h. Every target keeps all four channels here; the
// write masks (CB_TARGET_MASK & CB_SHADER_MASK) are pipeline_state_decode.h's, behind a switch.
#pragma once

#include "pipeline_state_decode.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace x2w_startup {
constexpr std::size_t state_block_blend_offset = 0xD0;

// Dual-source blending reads the pixel shader's second output, which exists for render target 0
// only; a SRC1 factor on any other target, or blending another target alongside a dual-source
// target 0, is not a valid D3D12 blend state.
inline bool uses_dual_source(const D3D12_RENDER_TARGET_BLEND_DESC& t)
{
    for (const D3D12_BLEND f : {t.SrcBlend, t.DestBlend, t.SrcBlendAlpha, t.DestBlendAlpha})
        if (f == D3D12_BLEND_SRC1_COLOR || f == D3D12_BLEND_INV_SRC1_COLOR || f == D3D12_BLEND_SRC1_ALPHA || f == D3D12_BLEND_INV_SRC1_ALPHA)
            return true;
    return false;
}

// Fills out from state block A. Returns false, leaving out alone, when the block is too short or
// the decoder refuses the word of any of the eight targets; reason then names the first cause and
// refused_targets has a bit per refused target. Nothing is substituted for a refused word: drawing
// that target unblended would change its output, so the caller keeps the path it had before.
inline bool state_block_blend(const unsigned char* block, std::size_t bytes, bool alpha_factor_supported, D3D12_BLEND_DESC* out,
                              const char** reason = nullptr, unsigned* refused_targets = nullptr)
{
    const char* ignored = nullptr;
    const char*& why = reason ? *reason : ignored;
    why = nullptr;
    if (refused_targets) *refused_targets = 0;
    if (!block || bytes < state_block_blend_offset + 8 * sizeof(std::uint32_t)) {
        why = "state block A shorter than the blend words";
        return false;
    }
    // One decoder for the eight words (pipeline_state_decode.h), with every target writing RGBA.
    std::uint32_t words[8];
    std::memcpy(words, block + state_block_blend_offset, sizeof(words));
    const x2w_pipeline_state::Blend decoded = x2w_pipeline_state::decode_blend(words, 0, 0, false, alpha_factor_supported);
    // Upstream rule kept at the merge: D3D12 dual-source blending is target 0 only, and no other target may blend
    // beside it. A dual-source factor on another target, or a blending target beside a dual-source target 0, is refused.
    unsigned dual_refused = 0;
    const char* dual_why = nullptr;
    const auto& t0 = decoded.desc.RenderTarget[0];
    const bool target0_dual_source = t0.BlendEnable && uses_dual_source(t0);
    for (unsigned target = 1; target < 8; ++target) {
        const auto& t = decoded.desc.RenderTarget[target];
        if (!t.BlendEnable) continue;
        const bool dual = uses_dual_source(t);
        if (!dual && !target0_dual_source) continue;
        if (!dual_refused && !(decoded.refused_targets & ((1u << target) - 1)))
            dual_why = dual ? "dual-source factor on a target other than 0" : "dual-source blend on target 0 with another blending target";
        dual_refused |= 1u << target;
    }
    if (refused_targets) *refused_targets = decoded.refused_targets | dual_refused;
    if (!decoded.ok || dual_refused) {
        why = dual_why && (!decoded.refused_targets || (dual_refused & -dual_refused) < (decoded.refused_targets & -decoded.refused_targets))
                  ? dual_why : decoded.reason ? decoded.reason : dual_why;
        return false;
    }
    const D3D12_BLEND_DESC& blend = decoded.desc;
    *out = blend;
    return true;
}
}  // namespace x2w_startup
