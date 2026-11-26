#pragma once

// Emit Shader Model 6.6 DXIL for console vertex, pixel, compute and mesh shaders. The host binds user data
// at b0 space 0, indexes ResourceDescriptorHeap and SamplerDescriptorHeap with console indices, binds
// immediate samplers as static samplers at s<n> space 0, and binds root CBVs/SRVs/UAVs in ROOT_CBV_SPACE.
// Signatures follow the console container.
//
// Translation errors throw ShaderError.

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "x2w_shader/emit/rect.h"
#include "x2w_shader/emit/sampler.h"
#include "x2w_shader/emit/settings.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/emit/signature.h"
#include "x2w_shader/error.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"

namespace x2w::vs {

namespace dxbc {
struct Container;
}
namespace ir {
struct Program;
}
namespace structure {
struct Structured;
}
namespace io {
struct Io;
}

// Public shader, sampler, and rectangle-list interfaces.
using sampler::StaticSampler;
using shader::MEMORY_FAULTS;
using shader::MEMORY_PAGES;
using shader::MEMORY_READ_ONLY;
using shader::raw_views;
using shader::ROOT_CBV_SPACE;
using shader::SAMPLER_SPACE;
using shader::set_raw_views;
using shader::set_view_heap_limit;
using shader::SLOT_FAULTS;
using shader::SLOT_MEMORY;
using shader::SLOT_STRIDES;
using shader::SLOT_STRIDES_COUNT;
using shader::USER_DATA_SPACE;
using shader::USER_DISPATCH_THREADS_X;
using shader::USER_MEMORY_BASE;
using shader::USER_ROOT_ADDRESS;
using shader::USER_VIEW_ADDRESS;
using shader::USER_VIEW_BIAS;
using shader::VIEW_INTEGER;
using shader::VIEW_SIGNED;
using shader::VIEW_STRIDE_BITS;
using shader::view_heap_limit;

// X2W_VS_SOURCE_HASH and X2W_VS_SWITCHES identify the translator build and its switches for the cache
// fingerprint.
std::vector<std::uint8_t> translation_fingerprint();

// One dword of user data and the root parameter that fills it.
struct UserData {
    // Dword in the user-data buffer (the console's user-data index).
    std::uint32_t index = 0;
    // Root parameter index in the title's root signature.
    std::uint8_t slot = 0;
    // Constants or Table (see the module notes for the value).
    native::RootType root{};
    // Dword within the parameter.
    std::uint32_t dword = 0;
    bool operator==(const UserData&) const = default;
};

// A translated shader and what the host binds for it.
struct HostShader {
    native::HwStage stage{};
    // The DXIL container, hashed as D3D12 expects.
    std::vector<std::uint8_t> dxil;
    // The user-data dwords the shader reads.
    std::vector<UserData> user_data;
    // Immediate samplers: the four sampler words of s<n>.
    std::vector<std::array<std::uint32_t, 4>> samplers;
    // The same samplers decoded, s<n> at position n: the static sampler descriptions the host's
    // root signature declares.
    std::vector<StaticSampler> static_samplers;
    // Root CBVs the shader reads: root parameter indices, each bound as b<index> in ROOT_CBV_SPACE.
    std::vector<std::uint8_t> root_cbvs;
    // Root SRVs (raw buffers) the shader reads: t<index> in ROOT_CBV_SPACE.
    std::vector<std::uint8_t> root_srvs;
    // Root UAVs (raw buffers) the shader reads or writes: u<index> in ROOT_CBV_SPACE.
    std::vector<std::uint8_t> root_uavs;
    // Heap descriptors that can be selected from root state at bind time.
    std::vector<resource::RootIndexedView> root_indexed_views;

    // Size of the user-data buffer the shader declares, in dwords (a multiple of 4; 0 when it
    // reads none). Root constants must cover it.
    std::uint32_t user_data_dwords() const;
    bool operator==(const HostShader&) const = default;
};

// Translates the vertex, pixel, compute or mesh program of a console shader container (bytes as
// the title passes them at pipeline creation). Throws ShaderError.
HostShader translate(std::span<const std::uint8_t> container);

// Emits a program whose earlier stages already ran. Throws ShaderError.
HostShader emit(const dxbc::Container& c, const native::NativeProgram& native, const ir::Program& p,
                const structure::Structured& st, const io::Io& io, const resource::Resources& res);

}  // namespace x2w::vs
