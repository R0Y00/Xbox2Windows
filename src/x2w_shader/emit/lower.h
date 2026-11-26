// IR to DXIL, one machine instruction at a time.
//
// Every register dword is an `i32` (SCC an `i1`); float operations bitcast their operands. Lane
// masks exist only as the per-lane booleans of structurisation; branches and guards are built from
// them. A value the emitter cannot express is `Missing` with a reason, and only fails the program
// when something with an effect (an export, a branch) needs it.
//
// This header is what the rest of emit/ (mod.cpp) uses of the lowering. The emitter's state and
// its methods are private: lower_impl.h.
//
// Lowering reports translation failures by throwing ShaderError (lib.h).
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "x2w_shader/emit/container.h"  // SigElement
#include "x2w_shader/emit/emit.h"       // UserData
#include "x2w_shader/error.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/structure.h"

namespace x2w::vs::lower {

using container::SigElement;

// static text, compared by content.
using Why = const char*;

struct Lowered {
    std::vector<std::uint8_t> dxil;
    // Bit n: immediate sampler n is a comparison sampler.
    std::uint64_t cmp_samplers = 0;
    std::vector<std::uint8_t> root_cbvs;
    std::vector<std::uint8_t> root_srvs;
    std::vector<std::uint8_t> root_uavs;
    std::vector<UserData> user_data;
    std::vector<std::array<std::uint32_t, 4>> samplers;
    // The wave size the shader requires of the host, if any.
    std::optional<std::uint32_t> wave_size;
    // Test counters record non-uniform heap indexing and wave operations.
    std::uint32_t nonuniform_handles = 0;
    bool wave_ops = false;
};

struct Parts {
    const native::NativeProgram* native = nullptr;
    const ir::Program* p = nullptr;
    const structure::Structured* st = nullptr;
    const io::Io* io = nullptr;
    const resource::Resources* res = nullptr;
    std::vector<SigElement> inputs;
    std::vector<SigElement> outputs;
    std::vector<SigElement> prims;
    // Mesh: what the group id arrives multiplied by.
    std::uint32_t mesh_stride = 0;
};

// Throws ShaderError.
Lowered lower(const Parts& parts);

// `lower`; `counted`: the lanes past a dispatch's count of threads leave at once (shader.h
// USER_DISPATCH_THREADS_X). Throws ShaderError.
Lowered lower_counted(const Parts& parts, bool counted);

// Iterations after which a loop is left (per entry into it), unless x2w::vs::loop_limit() says
// otherwise. A lane that reaches the limit leaves the program as killed (a pixel discards).
inline constexpr std::uint32_t LOOP_LIMIT = 1u << 20;

// What a value holding a lane mask read as a number stops with.
inline constexpr Why LANE_MASK = "a lane mask read as data";
// A mask with lanes outside EXEC where it is written, read as a number where the lanes that run
// that block are not all of the wave.
inline constexpr Why MASK_ELSEWHERE = "a lane mask holding lanes elsewhere read as data";
// The refusal of a first half of an interpolation whose result is read as data.
inline constexpr Why FIRST_HALF = "the first half of an interpolation";

// The per-lane branches (by block) whose regions hold a block of `needs`: the ones to take
// wave-wide (structure::structure_wide). None if one of them has no known wave-wide meaning.
std::optional<std::vector<std::size_t>> branches_around(const structure::Structured& st,
                                                        const std::vector<std::size_t>& needs);
// Whether a refusal is of code that needs every lane of the wave where it stands.
bool needs_every_lane(std::string_view what);
// The blocks where the last lowering on this thread wanted every lane and did not have it, which
// it forgets.
std::vector<std::size_t> take_needs();

// Whether a console wave64 program may run with each host wave as a wave of its own. Error:
// (offset, what).
std::optional<std::pair<std::size_t, Why>> wave_local(const ir::Program& p,
                                                      const structure::Structured& st);

bool reads_mask_as_data(const structure::Structured& st, const ir::Inst& i);

// The host GPU's WaveLaneCountMax (process-wide).
void set_host_max_wave(std::uint32_t lanes);
std::uint32_t host_max_wave();
// Tells the translator that the host's shaders run on AMD RDNA hardware (emit::rdna_float()).
void set_host_rdna(bool on);
// The bits of emit::rdna that count on this host.
std::uint32_t host_rdna();
// What the translator was told of the host, for a cache's key.
std::array<std::uint32_t, 4> host_settings();

namespace detail {
// Thread-local switches for the with_* scopes below.
bool replace_local_waves(bool on);
bool replace_local_any(bool on);
bool replace_first_half_pairs(bool on);
}  // namespace detail

// Run f with each 32-lane host wave treated as an independent wave for console Wave64 programs. Restore the
// switch even if f throws.
template <class F>
auto with_local_waves(F&& f) -> decltype(f()) {
    struct Restore {
        bool was;
        ~Restore() { detail::replace_local_waves(was); }
    } r{detail::replace_local_waves(true)};
    return f();
}

// As with_local_waves, with every other use of a lane mask as a number and every lane read and
// write taken per host wave too.
template <class F>
auto with_local_waves_any(F&& f) -> decltype(f()) {
    struct Restore {
        bool was;
        ~Restore() { detail::replace_local_any(was); }
    } r{detail::replace_local_any(true)};
    return with_local_waves(std::forward<F>(f));
}

// Runs `translate` with first halves of interpolations evaluated as the pair they start.
template <class F>
auto with_first_half_pairs(F&& translate) -> decltype(translate()) {
    struct Restore {
        bool was;
        ~Restore() { detail::replace_first_half_pairs(was); }
    } r{detail::replace_first_half_pairs(true)};
    return translate();
}

// Per-thread test hooks.
namespace test {
// VS_DUMP=<dir> names the modules lowered from now on <name>__<n>.dxil.
void set_dump_name(std::string_view name);
void set_f16_nearest(std::optional<bool> on);
void set_rdna(std::optional<std::uint32_t> bits);
void set_host_wave(std::optional<std::uint32_t> lanes);
// TEST_MEMORY identifies a heap slot for a raw buffer and the title address of its first byte for device
// tests.
void set_memory(std::optional<std::pair<std::uint32_t, std::uint64_t>> slot_and_address);
}  // namespace test

}  // namespace x2w::vs::lower
