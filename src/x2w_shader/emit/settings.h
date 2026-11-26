#pragma once

// Translation switches shared by shader emission and lowering, kept separate from emit.h to avoid a header
// cycle.

#include <cstdint>
#include <optional>
#include <string_view>

namespace x2w::vs {

// The loop limit without an override: 2^20 (lower LOOP_LIMIT).
inline constexpr std::uint32_t DEFAULT_LOOP_LIMIT = 1u << 20;

// DEFAULT_LOOP_LIMIT bounds iterations per lane and per loop entry. Builds with X2W_VS_DIAGNOSTICS read
// X2W_SHADERCFG_SHADER_LOOP_LIMIT.
std::uint32_t loop_limit();
// Whether a pixel shader's helper lane reads its own value where the console reads the first
// lane's; on unless X2W_SHADERCFG_HELPER_OWN=0.
bool helper_own();
// Which group barriers wait for the whole group (X2W_SHADERCFG_GROUP_SYNC: unset/2 every one, 1 only
// in blocks every lane runs, 0 none).
std::uint8_t group_sync();
// Whether buffer atomics of the ray-tracing structure builder stand between two fences
// (X2W_SHADERCFG_BVH_FENCES=0: they do not).
bool builder_fences();
// Whether the structure builder's accesses through its root views stay near them
// (X2W_SHADERCFG_BVH_BOUNDS=0: not).
bool builder_bounds();
// The clean-up passes (bitcode::opt bits) that run before a module is written
// (X2W_SHADERCFG_SHADER_OPT; unset: opt::DEFAULT).
std::uint32_t shader_opt();
// On an RDNA host (set_host_rdna): which float instructions are written as the one host
// operation (X2W_SHADERCFG_RDNA_FLOAT, rdna bits; unset: rdna::ALL).
std::uint32_t rdna_float();

// The bits of rdna_float().
namespace rdna {
// v_max_f32 and the like: the host's FMax and FMin, without the code that orders -0 below +0.
inline constexpr std::uint32_t MAX_MIN = 1;
// v_mul_legacy_f32 and the like (0 times anything is 0).
inline constexpr std::uint32_t LEGACY = 2;
// v_med3_f32 of a value between two ordered constants: a minimum of a maximum.
inline constexpr std::uint32_t MED3 = 4;
// v_fract_f32: the host's Frc as it is.
inline constexpr std::uint32_t FRACT = 8;
inline constexpr std::uint32_t ALL = 15;
}  // namespace rdna

// Pure parsers of the switches above (for testing).
std::uint32_t parse_set(std::optional<std::string_view> value, std::uint32_t unset, std::uint32_t all);
std::uint32_t parse_shader_opt(std::optional<std::string_view> value);
std::uint32_t parse_loop_limit(std::optional<std::string_view> value);

}  // namespace x2w::vs
