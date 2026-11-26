// The spilled user-data table of a compute program.
//
// A compute program has as many user SGPRs as `COMPUTE_PGM_RSRC2` says. A root signature that needs
// more dwords than that keeps none of its parameters in registers: user-data dwords 0 and 1 then hold
// the 40-bit address of a table in memory (`s0`, and `s1 & 0xff` as bits 32-39), and the parameter the
// layout declares at user-data index N (16 or above) is at byte `4 * (N - 16)` of that table. The code
// reads it with `s_load_dword*` through that pointer. (Every compute program of the console's
// ray-tracing structure builder is of this kind; measured there: 118 of 118 such loads land on a
// declared parameter.)
//
// user_data() rewrites those loads away: each dword they read becomes the entry value of the SGPR its
// user-data index names, as if the parameter had arrived in a register. Resource tracking and
// lowering then treat a spilled parameter like any other (the host binds user data by root slot and
// dword, not by register).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"

namespace x2w::vs::spill {

// ir::build of `native`'s code as the hardware runs it, with the loads of a spilled user-data table
// rewritten (user_data). Throws ShaderError.
ir::Program build(const native::NativeProgram& native);

// Whether `native` reads its root parameters from a spilled user-data table. (In the title exactly
// the compute programs of the console's ray-tracing structure builder do.)
bool is_spilled(const native::NativeProgram& native);

// Rewrites the loads of `native`'s spilled user-data table in `p`; returns how many loads it
// replaced. Nothing changes unless the program is a compute program whose layout has entries past
// its user SGPRs, the first at index 16. A load is left alone (and resource tracking then refuses it
// as before) when its offset is not a constant on a dword, or when a dword it reads and the program
// uses is in no layout entry.
std::size_t user_data(ir::Program& p, const native::NativeProgram& native);

namespace detail {
// The entries no register holds, as (first user-data index, dwords), if the table starts at index 16 and
// the program has the two registers of its address.
std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>> spilled(const native::NativeProgram& native);
}  // namespace detail

}  // namespace x2w::vs::spill
