// Native GFX10 programs that the console compiler stores in a `DXIL` part.
//
// An `XDAT` index after the program header lists blobs stored around the bitcode; one or more of
// them hold native programs, zlib-packed (a vertex shader can carry two: compiled for use with and
// without a geometry shader). Layout as observed (index offsets are relative to the part data):
//
//   +0x18 "XDAT"  u32 entry count, u32 (0x24 seen), then per entry: u32 kind, u32 offset, u32 size
//   kind 0x0d     "XBCH" u32 version (3), u64 packed size, u64 unpacked size, zlib stream
//   unpacked      "FSSM" u32 version (0x101), u64 payload size, payload
//   payload +0x00 u64 graphics slot count (6)
//           +0x08 u64 record offsets: LS HS ES GS VS PS (0x08..0x30), mesh 0x48, compute 0x60;
//                 unused ones are 0xDEADBEEF_DEADBEEF
//   record  +0x00 u64 code offset, +0x08 u64 code size (payload-relative),
//           +0x14 u32 flags: bit 5 wave32 (else wave64)
//           +0x28 u64 root layout table, +0x30 u64 used-resource table (payload-relative),
//           +0x38 u32 and +0x3c u32 their counts
//           +0x50 u32 bits 0-9 VGPRs used, bits 20-27 SGPRs used
//   layout  20 B: u32 root type, u32 table offset (dwords), u32 user SGPRs, u32 base register,
//                 u8 first user-data dword, u8 root slot, u16 0
//   used    20 B: u8 resource type, u8 root type, u16 0, u32 table offset (dwords), u32 lower
//                 bound, u32 stride, u8 descriptor dwords, u8 first user-data dword, u8 root slot,
//                 u8 0
//
// Stage data follows the record; offsets from its end (E = record + 0x54):
//
//   PS  E+0x24  32 inputs of 8 B: u8 usage, u8 signature row, u8 channel mask, u8 attribute
//               (v_interp `attr`), u8 default, u8 second row (of a packed pair), u8 parameter slot,
//               u8 flags (bit 0 flat, 1 linear, 3 fp16, 4 fp16 packed); E+0x124 u32 count
//       E+0x134 u32 SPI_PS_INPUT_ADDR, E+0x138 u32 DB_SHADER_CONTROL
//       E+0x148 u32 export patch size (dwords), E+0x14c 8 patches, one per MRT: u32 code offset
//               (dwords, -1 none), then the export code for each color format in 10 slots of 10
//               dwords (ZERO, 32_R, 32_GR, 32_AR, FP16, UNORM16, SNORM16, UINT16, SINT16, 32_ABGR)
//   VS  E+0x24  32 inputs of 5 B: u8 usage, u8 row, u8 channel mask, u8 VGPR of channel 0 (channel
//               k in VGPR + k), u8 highest channel; E+0xc4 u32 count
//       E+0xc8  32 outputs of 6 B: u8 usage, u8 row, u8 parameter (`exp param<n>`), u8 row again,
//               u8 slot, u8 flags (bit 0 fp16, bit 1 two fp16 values per dword); E+0x188 u32 count
//   CS  E+0x24  u32 COMPUTE_PGM_RSRC2, E+0x28 u32 x3 thread-group size
//   MS  E+0xc8  outputs as VS, E+0x188 count; the param byte's bit 7 marks a per-primitive output
//               (`PSG1` row)
//       E+0x18c u32 SPI_VS_OUT_CONFIG, SPI_SHADER_POS_FORMAT, PA_CL_VS_OUT_CNTL, ...
//       E+0x448 u16 thread-group size in lanes, u16 maximum vertices, u16 maximum primitives,
//               u8 user SGPRs, u8 output topology (2 triangles)
//   record +0x4c u32 LDS bytes (mesh programs read it here)
//
// Rows are registers of the container's signature parts (`ISG1`, `OSG1`). Merged stages keep their
// stage data in another layout (not read). Unknown versions are rejected so a format change shows
// up as an error.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

#include "x2w_shader/error.h"

namespace x2w::vs::native {

// Hardware shader stage a native program runs as.
enum class HwStage : std::uint8_t {
    Ls,
    Hs,
    Es,
    Gs,
    Vs,
    Ps,
    // Mesh shader.
    Ms,
    Cs,
};

// The SGPR holding user-data dword `index`: merged stages (and mesh shaders, which run as one) have
// 8 system SGPRs first.
std::uint8_t user_sgpr(HwStage stage, std::uint8_t index);

// Semantic usage from stage data. Only Other carries a numeric code.
struct Usage {
    enum class Kind : std::uint8_t { Generic, ClipDistance, Other };
    Kind kind = Kind::Generic;
    std::uint8_t other = 0;

    static constexpr Usage Generic() { return {Kind::Generic, 0}; }
    static constexpr Usage ClipDistance() { return {Kind::ClipDistance, 0}; }
    static constexpr Usage Other(std::uint8_t n) { return {Kind::Other, n}; }
    static Usage from_raw(std::uint8_t raw);
    bool operator==(const Usage&) const = default;
};

// A pixel shader input: what `v_interp` of attribute `attr` reads.
struct PsInput {
    Usage usage;
    // Register in the input signature.
    std::uint8_t row = 0;
    std::uint8_t mask = 0;
    std::uint8_t attr = 0;
    bool flat = false;
    bool fp16 = false;
    // Two 16-bit values per dword (VsOutput::pack2); `row2` is the second element's register.
    bool pack2 = false;
    std::uint8_t row2 = 0;
    bool operator==(const PsInput&) const = default;
};

// A vertex attribute the fetch shader loads: channel k into VGPR `vgpr + k`.
struct VsInput {
    Usage usage;
    std::uint8_t row = 0;
    std::uint8_t mask = 0;
    std::uint8_t vgpr = 0;
    bool operator==(const VsInput&) const = default;
};

// What `exp param<param>` of a vertex shader writes.
struct VsOutput {
    Usage usage;
    // Register in the output signature.
    std::uint8_t row = 0;
    std::uint8_t param = 0;
    // 16-bit values, one per dword (low half) unless `pack2`: then two per dword, channel c of this
    // row in dword c / 2 (odd channels in the high half), then a second 16-bit row from the next
    // whole dword (the next 16-bit row of the signature no entry names).
    bool fp16 = false;
    bool pack2 = false;
    bool operator==(const VsOutput&) const = default;
};

// The export code of one render target for its full 32-bit format.
struct ExportPatch {
    std::uint8_t mrt = 0;
    // Where it goes in the code, in dwords.
    std::uint32_t offset = 0;
    std::vector<std::uint32_t> code;
    bool operator==(const ExportPatch&) const = default;
};

// StageIo::Vs.
struct VsIo {
    std::vector<VsInput> inputs;
    std::vector<VsOutput> outputs;
    bool operator==(const VsIo&) const = default;
};

// StageIo::Ps.
struct PsIo {
    std::vector<PsInput> inputs;
    // `SPI_PS_INPUT_ADDR`: which system values arrive in VGPRs.
    std::uint32_t input_addr = 0;
    // `DB_SHADER_CONTROL`: depth export, kill.
    std::uint32_t db_shader_control = 0;
    std::vector<ExportPatch> patches;
    bool operator==(const PsIo&) const = default;
};

// A mesh program's outputs and limits (StageIo::Ms). It runs as one hardware NGG stage: lane n of
// the group (`threads` lanes) exports vertex n (`exp pos`, `exp param`) and primitive n
// (`exp prim`).
struct MeshIo {
    // `exp param<n>` targets; `param` bit 7 set: a per-primitive output, `row` then a row of the
    // primitive signature (`PSG1`).
    std::vector<VsOutput> outputs;
    // `PA_CL_VS_OUT_CNTL`: which clip and cull distances `pos1+` carry.
    std::uint32_t clip_cntl = 0;
    // Thread-group size in lanes (the HLSL group rounded up to waves).
    std::uint32_t threads = 0;
    std::uint32_t max_vertices = 0;
    std::uint32_t max_primitives = 0;
    // 1 lines, 2 triangles.
    std::uint8_t topology = 0;
    std::uint32_t lds_bytes = 0;
    bool operator==(const MeshIo&) const = default;
};

// A compute program's dispatch settings (StageIo::Cs).
struct ComputeIo {
    // `COMPUTE_PGM_RSRC2`: user SGPRs, which system SGPRs and VGPRs follow them, LDS size.
    std::uint32_t rsrc2 = 0;
    // Thread-group size (`numthreads`).
    std::array<std::uint32_t, 3> threads{};

    // User SGPRs before the system ones.
    std::uint8_t user_sgprs() const;
    // The SGPR of each enabled thread-group id component, in order after the user SGPRs, then the
    // thread-group info SGPR if enabled.
    std::pair<std::array<std::optional<std::uint8_t>, 3>, std::optional<std::uint8_t>> system_sgprs() const;
    // SGPRs the hardware initializes: the user SGPRs, the enabled system ones, then the scratch wave
    // offset when scratch is enabled (bit 0). The ones above hold whatever the previous wave left.
    std::uint8_t initialized_sgprs() const;
    // Thread id components in v0, v1, v2 (1 to 3).
    std::uint8_t thread_id_vgprs() const;
    // LDS bytes per thread group (512-byte granules).
    std::uint32_t lds_bytes() const;
    bool operator==(const ComputeIo&) const = default;
};

// Input and output stage data. std::monostate denotes missing stage information.
using StageIo = std::variant<std::monostate, VsIo, PsIo, ComputeIo, MeshIo>;

// Only Other carries an unrecognized numeric root type.
struct RootType {
    enum class Kind : std::uint8_t {
        Constants,
        // A 64-bit buffer address.
        Descriptor,
        // The 32-bit heap address of a descriptor table range.
        Table,
        // A sampler the root signature fixes.
        ImmediateSampler,
        // The 32-bit address of the CBV/SRV/UAV heap (SM 6.6 dynamic resources).
        ResourceHeap,
        // The 32-bit address of the sampler heap.
        SamplerHeap,
        Other,
    };
    Kind kind = Kind::Constants;
    std::uint32_t other = 0;

    static constexpr RootType Constants() { return {Kind::Constants, 0}; }
    static constexpr RootType Descriptor() { return {Kind::Descriptor, 0}; }
    static constexpr RootType Table() { return {Kind::Table, 0}; }
    static constexpr RootType ImmediateSampler() { return {Kind::ImmediateSampler, 0}; }
    static constexpr RootType ResourceHeap() { return {Kind::ResourceHeap, 0}; }
    static constexpr RootType SamplerHeap() { return {Kind::SamplerHeap, 0}; }
    static constexpr RootType Other(std::uint32_t n) { return {Kind::Other, n}; }
    static RootType from_raw(std::uint32_t raw);
    bool operator==(const RootType&) const = default;
};

// One root parameter (or table range) and the user SGPRs holding it.
struct RootParam {
    RootType root;
    // Of the range within its table.
    std::uint32_t table_offset_dwords = 0;
    // User SGPRs from `user_data` on.
    std::uint32_t sgprs = 0;
    // Shader register of the first descriptor (`b`/`t`/`u`/`s` number).
    std::uint32_t base_register = 0;
    // First user-data dword (user_sgpr).
    std::uint8_t user_data = 0;
    // Index of the parameter in the root signature.
    std::uint8_t slot = 0;
    bool operator==(const RootParam&) const = default;
};

struct ResourceType {
    enum class Kind : std::uint8_t { Srv, Sampler, Cbv, Uav, Other };
    Kind kind = Kind::Srv;
    std::uint8_t other = 0;

    static constexpr ResourceType Srv() { return {Kind::Srv, 0}; }
    static constexpr ResourceType Sampler() { return {Kind::Sampler, 0}; }
    static constexpr ResourceType Cbv() { return {Kind::Cbv, 0}; }
    static constexpr ResourceType Uav() { return {Kind::Uav, 0}; }
    static constexpr ResourceType Other(std::uint8_t n) { return {Kind::Other, n}; }
    static ResourceType from_raw(std::uint8_t raw);
    bool operator==(const ResourceType&) const = default;
};

// A descriptor range the program reads.
struct UsedResource {
    ResourceType kind;
    RootType root;
    std::uint32_t table_offset_dwords = 0;
    // Shader register of the first descriptor read.
    std::uint32_t lower_bound = 0;
    std::uint32_t stride = 0;
    // Size of the descriptor (4 or 8).
    std::uint8_t dwords = 0;
    std::uint8_t user_data = 0;
    std::uint8_t slot = 0;
    bool operator==(const UsedResource&) const = default;
};

struct NativeProgram {
    HwStage stage = HwStage::Ls;
    // Runs as 64 lanes per wave (else 32); sizes lane masks.
    bool wave64 = false;
    // VGPRs the program uses (v0 up to this), the bound of indexed access.
    std::uint16_t vgprs = 0;
    // Where the root signature's parameters arrive (user SGPRs).
    std::vector<RootParam> layout;
    // The descriptors the program reads, by root parameter.
    std::vector<UsedResource> used;
    StageIo io;
    // GFX10 machine code, as stored.
    std::vector<std::uint8_t> code;

    // The code with each render target's export patched for its full 32-bit format, so exports carry
    // the shader's values unconverted (as the console does at pipeline creation for that format).
    std::vector<std::uint8_t> patched_code() const;
    bool operator==(const NativeProgram&) const = default;
};

// The native programs in a `DXIL` part's data, or in an `ACCC` part's (fspx_program); empty for the
// PC layout. Throws ShaderError::malformed.
std::vector<NativeProgram> native_programs(std::span<const std::uint8_t> part);

// The compute program of an `ACCC` part, which a container carries in place of `DXIL` when another
// compiler made the native code (an upscaler's shaders). Layout as observed (offsets from "FSPX"):
//
//   +0x00 "FSPX" u32 version (1), u32 entry count, u32 size; entries of 16 B from +0x10: u32 kind,
//         u32 offset, u32 size, u32 0
//   kind 0  pipeline registers, 0x1c B: +0x0c u16 x3 thread-group size, +0x12 u16 flags (bit 4
//           wave32), +0x14 COMPUTE_PGM_RSRC1, +0x18 COMPUTE_PGM_RSRC2
//   kind 1  pipeline metadata (a hash); kind 6 the GFX10 code
//   kind 7  program metadata: u32 0x403, u32 size - 0x14, u32 code dwords, u64 0, u8 0x40, u8 first
//           user-data dword, u8 count of 0x41 records, u8 0; then 32-byte records from +0x18, each
//           tagged by its first byte:
//     0x41  a resource the code reads: u8 resource type (as FSSM's used entries), u8 root type, u16 0,
//           u32 table offset (dwords), u32 register, u32 view dimension, u8 descriptor dwords, u8
//           user-data dword, u8 root slot, u8 0, u32 -1, u32 0
//     0x42  a view declaration
//     0x43  the shader hash (16 B); the last record
NativeProgram fspx_program(std::span<const std::uint8_t> part);

namespace detail {
// Unpack the XBCH blob for tests.
std::vector<std::uint8_t> unpack(std::span<const std::uint8_t> blob);
}  // namespace detail

}  // namespace x2w::vs::native
