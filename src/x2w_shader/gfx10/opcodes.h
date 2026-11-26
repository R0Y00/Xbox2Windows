// GFX10 opcode names generated from Mesa's aco_opcodes.py (mesa-22.3.0, GFX10 column). See opcodes.cpp for the MIT notice.
//
// Each table is sorted by opcode. VOP3 holds only VOP3-only opcodes; VOP3 encodings of
// VOPC/VOP2/VOP1 ops (0x000-0x13f, 0x180-0x1ff) use those tables.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace x2w::vs::gfx10::opcodes {

struct Entry {
    std::uint16_t op;
    std::string_view name;
};
using Table = std::span<const Entry>;

extern const Table SOP2;
extern const Table SOPK;
extern const Table SOP1;
extern const Table SOPC;
extern const Table SOPP;
extern const Table SMEM;
extern const Table VOP2;
extern const Table VOP1;
extern const Table VOPC;
extern const Table VOP3;
extern const Table VOP3P;
extern const Table VINTRP;
extern const Table DS;
extern const Table MUBUF;
extern const Table MTBUF;
extern const Table MIMG;
extern const Table FLAT;
extern const Table GLOBAL;
extern const Table SCRATCH;

}  // namespace x2w::vs::gfx10::opcodes
