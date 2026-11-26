// Operand fields of GFX10 instructions, by encoding.
//
// Layouts follow AMD's RDNA ISA reference. Register ranges carry their size in dwords, which depends
// on the opcode (read from its name) and, for lane masks (VCC, EXEC, compare results), on the wave
// size.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace x2w::vs::gfx10 {

// A source or destination of an instruction. Leave unused fields at zero so equality compares only relevant
// state.
struct Operand {
    enum class Kind : std::uint8_t { Scalar, Vector, Const, Literal, Attr };

    Kind kind = Kind::Const;
    // Scalar: registers by operand code: 0-105 `s0`-`s105`, 106/107 `vcc_lo`/`vcc_hi`, 108-123
    // `ttmp0`-`ttmp15`, 124 `m0`, 125 `null`, 126/127 `exec_lo`/`exec_hi`. The decoder gives no code
    // above 127; the IR names scalars parked in VGPR lanes by the codes from 128 up (ir::PARKED).
    // Const: inline constant or read-only special source, by operand code (128-192: 0-64,
    // 193-208: -1 to -16, 240-248: floats, 251-254: vccz, execz, scc, lds_direct).
    std::uint8_t code = 0;
    // Vector: the first VGPR.
    std::uint8_t reg = 0;
    // Scalar, Vector.
    std::uint8_t dwords = 0;
    // Literal: the 32-bit literal that follows the instruction.
    std::uint32_t literal = 0;
    // Attr: interpolation attribute and channel (0-3 = x-w); `high` selects the upper f16 half.
    std::uint8_t index = 0;
    std::uint8_t chan = 0;
    bool high = false;

    static constexpr Operand Scalar(std::uint8_t code, std::uint8_t dwords) {
        Operand o;
        o.kind = Kind::Scalar;
        o.code = code;
        o.dwords = dwords;
        return o;
    }
    static constexpr Operand Vector(std::uint8_t reg, std::uint8_t dwords) {
        Operand o;
        o.kind = Kind::Vector;
        o.reg = reg;
        o.dwords = dwords;
        return o;
    }
    static constexpr Operand Const(std::uint8_t code) {
        Operand o;
        o.kind = Kind::Const;
        o.code = code;
        return o;
    }
    static constexpr Operand Literal(std::uint32_t value) {
        Operand o;
        o.kind = Kind::Literal;
        o.literal = value;
        return o;
    }
    static constexpr Operand Attr(std::uint8_t index, std::uint8_t chan, bool high) {
        Operand o;
        o.kind = Kind::Attr;
        o.index = index;
        o.chan = chan;
        o.high = high;
        return o;
    }

    constexpr bool is_scalar() const { return kind == Kind::Scalar; }
    constexpr bool is_vector() const { return kind == Kind::Vector; }
    constexpr bool is_const() const { return kind == Kind::Const; }
    constexpr bool is_literal() const { return kind == Kind::Literal; }
    constexpr bool is_attr() const { return kind == Kind::Attr; }

    bool operator==(const Operand&) const = default;

    // Display: SP3 syntax (`s5`, `v[2:3]`, `vcc`, `lit(0x3f000000)`, `attr1.x`...), in sp3.cpp.
    std::string to_string() const;
    // Diagnostic spelling: `Scalar { code: 5, dwords: 1 }`, `Const(128)`, `Attr { index: 1, chan: 0, high:
    // false }`.
    std::string debug() const;
};

// Operand code of the null register, which always reads as zero.
inline constexpr std::uint8_t NULL_CODE = 125;

struct Cache {
    bool glc = false;
    bool slc = false;
    bool dlc = false;
    bool operator==(const Cache&) const = default;
};

struct Sdwa {
    // Selects: 0-3 byte, 4-5 word, 6 dword. Compares have no `dst_sel`.
    std::uint8_t dst_sel = 0;
    // 0 pad with zeros, 1 sign-extend, 2 preserve.
    std::uint8_t dst_unused = 0;
    std::array<std::uint8_t, 2> src_sel{};
    // Bit i sign-extends `src[i]`.
    std::uint8_t sext = 0;
    bool operator==(const Sdwa&) const = default;
};

// DPP16: `ctrl` is the 9-bit permutation (quad_perm, row shifts...).
struct DppRow {
    std::uint16_t ctrl = 0;
    std::uint8_t row_mask = 0;
    std::uint8_t bank_mask = 0;
    bool bound_ctrl = false;
    bool fi = false;
    bool operator==(const DppRow&) const = default;
};

// DPP8: eight 3-bit lane selects.
struct DppLanes {
    std::uint32_t sel = 0;
    bool fi = false;
    bool operator==(const DppLanes&) const = default;
};

using Dpp = std::variant<DppRow, DppLanes>;

// VOP1, VOP2, VOPC, VOP3 and VOP3P, with their SDWA and DPP forms.
struct Valu {
    // A VGPR range, an SGPR (`v_readlane_b32`...) or a lane mask (compares).
    std::optional<Operand> dst;
    // Carry-out or the second result of VOP3B (`v_add_co_u32`, `v_div_scale_f32`...).
    std::optional<Operand> carry;
    // In operand order, including the implicit VCC of VOP2 `v_cndmask_b32` and carry-in, and the
    // literal of `v_madmk`/`v_madak` forms.
    std::vector<Operand> src;
    // Bit i applies to `src[i]`. For VOP3P, `neg` is the low half's.
    std::uint8_t neg = 0;
    std::uint8_t abs = 0;
    // VOP3P: negate the high half (`v_fma_mix*`: absolute value).
    std::uint8_t neg_hi = 0;
    // VOP3: bit 3 is the destination. VOP3P with `opsel_hi`: the halves used.
    std::uint8_t opsel = 0;
    std::uint8_t opsel_hi = 0;
    bool clamp = false;
    // Output modifier: 1 x2, 2 x4, 3 /2.
    std::uint8_t omod = 0;
    std::optional<Sdwa> sdwa;
    std::optional<Dpp> dpp;
    bool operator==(const Valu&) const = default;
};

// Decoded operand field variants.

// SOP1, SOP2, SOPC, SOPK; `imm` is SOPK's 16-bit immediate.
struct Sop {
    std::optional<Operand> dst;
    std::vector<Operand> src;
    std::optional<std::uint16_t> imm;
    bool operator==(const Sop&) const = default;
};

// The 16-bit immediate: branch offset in dwords, wait counts, message...
struct Sopp {
    std::uint16_t imm = 0;
    bool operator==(const Sopp&) const = default;
};

struct Smem {
    std::optional<Operand> data;
    Operand base;
    // Signed byte offset, added to `soffset`.
    std::int32_t offset = 0;
    Operand soffset;
    Cache cache;
    bool operator==(const Smem&) const = default;
};

struct Vintrp {
    Operand dst;
    // The i/j VGPR, or for `v_interp_mov_f32` the parameter (0 `p10`, 1 `p20`, 2 `p0`) as Const.
    Operand src;
    Operand attr;
    bool operator==(const Vintrp&) const = default;
};

struct Ds {
    std::optional<Operand> dst;
    std::optional<Operand> addr;
    std::vector<Operand> data;
    std::uint8_t offset0 = 0;
    std::uint8_t offset1 = 0;
    bool gds = false;
    bool operator==(const Ds&) const = default;
};

// MUBUF, and MTBUF with `format` set.
struct Buffer {
    // Loaded, stored, or the atomic operand (and result with `glc`).
    std::optional<Operand> data;
    Operand addr;
    Operand rsrc;
    Operand soffset;
    std::uint16_t offset = 0;
    bool offen = false;
    bool idxen = false;
    std::optional<std::uint8_t> format;
    bool lds = false;
    bool tfe = false;
    Cache cache;
    bool operator==(const Buffer&) const = default;
};

struct Image {
    Operand data;
    // One range, or one VGPR per address with the NSA form.
    std::vector<Operand> addr;
    Operand rsrc;
    std::optional<Operand> sampler;
    std::uint8_t dmask = 0;
    std::uint8_t dim = 0;
    bool unorm = false;
    bool r128 = false;
    bool a16 = false;
    bool d16 = false;
    bool tfe = false;
    bool lwe = false;
    Cache cache;
    bool operator==(const Image&) const = default;
};

// FLAT, GLOBAL and SCRATCH.
struct Flat {
    std::optional<Operand> dst;
    std::optional<Operand> addr;
    std::optional<Operand> data;
    std::optional<Operand> saddr;
    std::int16_t offset = 0;
    bool lds = false;
    Cache cache;
    bool operator==(const Flat&) const = default;
};

struct Exp {
    std::uint8_t target = 0;
    // Per channel; nullopt when disabled.
    std::array<std::optional<Operand>, 4> src{};
    bool compr = false;
    bool done = false;
    bool vm = false;
    bool operator==(const Exp&) const = default;
};

// Use std::get_if<Sop>(&fields) and the corresponding variant types to inspect decoded operands.
using Fields = std::variant<Sop, Sopp, Smem, Valu, Vintrp, Ds, Buffer, Image, Flat, Exp>;

// LDS opcodes on two addresses (`ds_read2_b32`...), with `offset0`/`offset1` (pub(super)).
bool ds_pair(std::string_view name);

}  // namespace x2w::vs::gfx10
