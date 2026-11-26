// Integer VALU operations, compares, lane bits and interpolation.
#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "x2w_shader/emit/lower_impl.h"

namespace x2w::vs::lower {

namespace {

bool one_of(std::string_view s, std::initializer_list<std::string_view> names) {
    for (auto n : names) {
        if (s == n) return true;
    }
    return false;
}

}  // namespace

V Lower::integer(const Inst& inst, Why name_) {
    const std::string_view name(name_);
    const std::size_t c = this->cur;
    auto s = [&](Lower& l, std::size_t k) { return l.src(inst, k); };
    auto k = [](Lower& l, std::uint32_t x) { return l.s.m.i32(x); };
    auto m31 = [c](Lower& l, V x) {
        auto k31 = l.s.m.i32(31);
        return l.s.m.bin(c, Bin::And, x, k31);
    };
    if (one_of(name, {"v_and_b32", "v_or_b32", "v_xor_b32", "v_add_nc_u32", "v_sub_nc_u32",
                      "v_subrev_nc_u32", "v_mul_lo_u32"})) {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        Bin op;
        if (name == "v_and_b32") {
            op = Bin::And;
        } else if (name == "v_or_b32") {
            op = Bin::Or;
        } else if (name == "v_xor_b32") {
            op = Bin::Xor;
        } else if (name == "v_add_nc_u32") {
            op = Bin::Add;
        } else if (name == "v_sub_nc_u32") {
            op = Bin::Sub;
        } else if (name == "v_subrev_nc_u32") {
            op = Bin::Sub;
            std::swap(a, b);
        } else {
            op = Bin::Mul;
        }
        return this->s.m.bin(c, op, a, b);
    }
    if (name == "v_xnor_b32") {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto x = this->s.m.bin(c, Bin::Xor, a, b);
        auto ones = k(*this, 0xFFFF'FFFFu);
        return this->s.m.bin(c, Bin::Xor, x, ones);
    }
    if (name == "v_not_b32") {
        auto a = s(*this, 0);
        auto ones = k(*this, 0xFFFF'FFFFu);
        return this->s.m.bin(c, Bin::Xor, a, ones);
    }
    if (one_of(name, {"v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32"})) {
        auto sh = s(*this, 0);
        auto a = s(*this, 1);
        sh = m31(*this, sh);
        Bin op = name == "v_lshlrev_b32" ? Bin::Shl : name == "v_lshrrev_b32" ? Bin::LShr : Bin::AShr;
        return this->s.m.bin(c, op, a, sh);
    }
    if (one_of(name, {"v_lshl_add_u32", "v_lshl_or_b32", "v_add_lshl_u32", "v_and_or_b32", "v_or3_b32",
                      "v_xad_u32", "v_add3_u32", "v_xor3_b32"})) {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto d = s(*this, 2);
        if (name == "v_lshl_add_u32" || name == "v_lshl_or_b32") {
            b = m31(*this, b);
            auto t = this->s.m.bin(c, Bin::Shl, a, b);
            Bin op = name == "v_lshl_add_u32" ? Bin::Add : Bin::Or;
            return this->s.m.bin(c, op, t, d);
        }
        if (name == "v_add_lshl_u32") {
            auto t = this->s.m.bin(c, Bin::Add, a, b);
            d = m31(*this, d);
            return this->s.m.bin(c, Bin::Shl, t, d);
        }
        Bin o1, o2;
        if (name == "v_and_or_b32") {
            o1 = Bin::And, o2 = Bin::Or;
        } else if (name == "v_or3_b32") {
            o1 = Bin::Or, o2 = Bin::Or;
        } else if (name == "v_xad_u32") {
            o1 = Bin::Xor, o2 = Bin::Add;
        } else if (name == "v_add3_u32") {
            o1 = Bin::Add, o2 = Bin::Add;
        } else {
            o1 = Bin::Xor, o2 = Bin::Xor;
        }
        auto t = this->s.m.bin(c, o1, a, b);
        return this->s.m.bin(c, o2, t, d);
    }
    if (name == "v_mul_u32_u24" || name == "v_mul_i32_i24") {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto sh = k(*this, 8);
        Bin op = name == "v_mul_i32_i24" ? Bin::AShr : Bin::LShr;
        a = this->s.m.bin(c, Bin::Shl, a, sh);
        a = this->s.m.bin(c, op, a, sh);
        b = this->s.m.bin(c, Bin::Shl, b, sh);
        b = this->s.m.bin(c, op, b, sh);
        return this->s.m.bin(c, Bin::Mul, a, b);
    }
    if (name == "v_mul_hi_u32_u24" || name == "v_mul_hi_i32_i24") {
        // The high dword of the same 48-bit product: the low 24 bits of each operand, zero- or
        // sign-extended, multiplied.
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        bool signed_ = name == "v_mul_hi_i32_i24";
        auto sh = k(*this, 8);
        Bin op = signed_ ? Bin::AShr : Bin::LShr;
        a = this->s.m.bin(c, Bin::Shl, a, sh);
        a = this->s.m.bin(c, op, a, sh);
        b = this->s.m.bin(c, Bin::Shl, b, sh);
        b = this->s.m.bin(c, op, b, sh);
        return this->mul_hi(a, b, signed_);
    }
    if (name == "v_bfe_u32" || name == "v_bfe_i32") {
        auto a = s(*this, 0);
        auto off = s(*this, 1);
        auto w = s(*this, 2);
        w = m31(*this, w);
        return this->bfe(a, off, w, name == "v_bfe_i32");
    }
    if (name == "v_bfi_b32") {
        auto m = s(*this, 0);
        auto a = s(*this, 1);
        auto b = s(*this, 2);
        auto ones = k(*this, 0xFFFF'FFFFu);
        auto nm = this->s.m.bin(c, Bin::Xor, m, ones);
        auto x = this->s.m.bin(c, Bin::And, m, a);
        auto y = this->s.m.bin(c, Bin::And, nm, b);
        return this->s.m.bin(c, Bin::Or, x, y);
    }
    if (one_of(name, {"v_max_i32", "v_min_i32", "v_max_u32", "v_min_u32"})) {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        std::uint8_t p = name == "v_max_i32"   ? pred::SGT
                         : name == "v_min_i32" ? pred::SLT
                         : name == "v_max_u32" ? pred::IUGT
                                               : pred::IULT;
        auto t = this->s.m.cmp(c, p, a, b);
        return this->s.m.select(c, t, a, b);
    }
    if (name == "v_mul_hi_u32" || name == "v_mul_hi_i32") {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        return this->mul_hi(a, b, name == "v_mul_hi_i32");
    }
    if (name == "v_alignbit_b32" || name == "v_alignbyte_b32") {
        // The low dword of {a, b} >> shift.
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto sh = s(*this, 2);
        if (name == "v_alignbyte_b32") {
            auto three = k(*this, 3);
            auto t = this->s.m.bin(c, Bin::And, sh, three);
            auto k3 = k(*this, 3);
            sh = this->s.m.bin(c, Bin::Shl, t, k3);
        } else {
            sh = m31(*this, sh);
        }
        auto k32 = k(*this, 32);
        auto lo = this->s.m.bin(c, Bin::LShr, b, sh);
        auto left = this->s.m.bin(c, Bin::Sub, k32, sh);
        left = m31(*this, left);
        auto hi = this->s.m.bin(c, Bin::Shl, a, left);
        auto zero = k(*this, 0);
        auto none = this->s.m.cmp(c, pred::EQ, sh, zero);
        hi = this->s.m.select(c, none, zero, hi);
        return this->s.m.bin(c, Bin::Or, lo, hi);
    }
    if (name == "v_bfrev_b32") {
        auto a = s(*this, 0);
        return this->s.op(c, "unary.i32", Attr::ReadNone, this->s.t.i32, 30, std::vector<V>{a});
    }
    if (name == "v_bcnt_u32_b32") {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto n = this->s.op(c, "unaryBits.i32", Attr::ReadNone, this->s.t.i32, 31, std::vector<V>{a});
        return this->s.m.bin(c, Bin::Add, n, b);
    }
    if (name == "v_mad_u32_u24" || name == "v_mad_i32_i24") {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto d = s(*this, 2);
        auto sh = k(*this, 8);
        Bin op = name == "v_mad_i32_i24" ? Bin::AShr : Bin::LShr;
        a = this->s.m.bin(c, Bin::Shl, a, sh);
        a = this->s.m.bin(c, op, a, sh);
        b = this->s.m.bin(c, Bin::Shl, b, sh);
        b = this->s.m.bin(c, op, b, sh);
        auto m = this->s.m.bin(c, Bin::Mul, a, b);
        return this->s.m.bin(c, Bin::Add, m, d);
    }
    if (one_of(name, {"v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_msad_u8"})) {
        // src2 plus the sum of the unsigned differences |a - b| of the operands' bytes, halves or
        // whole dwords, wrapping at 32 bits. v_sad_hi_u8 adds the sum shifted to the high half;
        // v_msad_u8 leaves out a byte whose mask byte (src1) is 0.
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto d = s(*this, 2);
        std::uint32_t width, parts;
        if (name == "v_sad_u16") {
            width = 16, parts = 2;
        } else if (name == "v_sad_u32") {
            width = 32, parts = 1;
        } else {
            width = 8, parts = 4;
        }
        auto zero = k(*this, 0);
        auto sum = zero;
        for (std::uint32_t i = 0; i < parts; ++i) {
            auto x = a;
            auto y = b;
            if (width < 32) {
                auto sh = k(*this, width * i);
                auto mask = k(*this, (1u << width) - 1);
                x = this->s.m.bin(c, Bin::LShr, x, sh);
                x = this->s.m.bin(c, Bin::And, x, mask);
                y = this->s.m.bin(c, Bin::LShr, y, sh);
                y = this->s.m.bin(c, Bin::And, y, mask);
            }
            auto above = this->s.m.cmp(c, pred::IUGT, x, y);
            auto hi = this->s.m.select(c, above, x, y);
            auto lo = this->s.m.select(c, above, y, x);
            auto diff = this->s.m.bin(c, Bin::Sub, hi, lo);
            if (name == "v_msad_u8") {
                auto masked = this->s.m.cmp(c, pred::NE, y, zero);
                diff = this->s.m.select(c, masked, diff, zero);
            }
            sum = this->s.m.bin(c, Bin::Add, sum, diff);
        }
        if (name == "v_sad_hi_u8") {
            auto k16 = k(*this, 16);
            sum = this->s.m.bin(c, Bin::Shl, sum, k16);
        }
        return this->s.m.bin(c, Bin::Add, sum, d);
    }
    if (one_of(name, {"v_min3_i32", "v_max3_i32", "v_min3_u32", "v_max3_u32", "v_med3_i32", "v_med3_u32"})) {
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto d = s(*this, 2);
        bool signed_ = name.ends_with("i32");
        std::uint8_t lt = signed_ ? pred::SLT : pred::IULT;
        std::uint8_t gt = signed_ ? pred::SGT : pred::IUGT;
        auto pick = [c](Lower& l, std::uint8_t p, V x, V y) {
            auto t = l.s.m.cmp(c, p, x, y);
            return l.s.m.select(c, t, x, y);
        };
        auto kind = name.substr(2, 4);
        if (kind == "min3") {
            auto t = pick(*this, lt, a, b);
            return pick(*this, lt, t, d);
        }
        if (kind == "max3") {
            auto t = pick(*this, gt, a, b);
            return pick(*this, gt, t, d);
        }
        auto lo = pick(*this, lt, a, b);
        auto hi = pick(*this, gt, a, b);
        auto t = pick(*this, lt, hi, d);
        return pick(*this, gt, lo, t);
    }
    if (name == "v_perm_b32") {
        // Byte i of the result is byte S of {src0, src1} (src1 the low dword) for selector byte S
        // below 8; 8-11 repeat the sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
        // higher 0xFF.
        const auto& v = std::get<gfx10::Valu>(inst.fields);
        std::uint32_t sel;
        const Operand* s2 = v.src.size() > 2 ? &v.src[2] : nullptr;
        if (s2 && s2->kind == Operand::Kind::Literal) {
            sel = s2->literal;
        } else if (s2 && s2->kind == Operand::Kind::Const) {
            sel = ok_or(resource::inline_constant(s2->code), name_);
        } else {
            // A computed selector: each byte picked at run time.
            auto a = s(*this, 0);
            auto b = s(*this, 1);
            auto sv = s(*this, 2);
            auto r = k(*this, 0);
            for (std::uint32_t i = 0; i < 4; ++i) {
                auto sh = k(*this, 8 * i);
                auto ff = k(*this, 0xFF);
                auto selv = this->s.m.bin(c, Bin::LShr, sv, sh);
                selv = this->s.m.bin(c, Bin::And, selv, ff);
                // Select a dword and a byte within it. Later selects handle codes 8 and above.
                auto four = k(*this, 4);
                auto zero = k(*this, 0);
                auto high = this->s.m.bin(c, Bin::And, selv, four);
                high = this->s.m.cmp(c, pred::NE, high, zero);
                auto word = this->s.m.select(c, high, a, b);
                auto three = k(*this, 3);
                auto index = this->s.m.bin(c, Bin::And, selv, three);
                auto bits = this->s.m.bin(c, Bin::Shl, index, three);
                auto x = this->s.m.bin(c, Bin::LShr, word, bits);
                auto byte = this->s.m.bin(c, Bin::And, x, ff);
                // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                const std::array<std::tuple<std::uint32_t, V, std::uint32_t>, 4> signs{
                    {{8, b, 15}, {9, b, 31}, {10, a, 15}, {11, a, 31}}};
                for (const auto& [n, src, bit] : signs) {
                    auto left = k(*this, 31 - bit);
                    auto t = this->s.m.bin(c, Bin::Shl, src, left);
                    auto k31 = k(*this, 31);
                    t = this->s.m.bin(c, Bin::AShr, t, k31);
                    t = this->s.m.bin(c, Bin::And, t, ff);
                    auto kn = k(*this, n);
                    auto is = this->s.m.cmp(c, pred::EQ, selv, kn);
                    byte = this->s.m.select(c, is, t, byte);
                }
                auto k12 = k(*this, 12);
                auto z = k(*this, 0);
                auto is12 = this->s.m.cmp(c, pred::EQ, selv, k12);
                byte = this->s.m.select(c, is12, z, byte);
                auto above = this->s.m.cmp(c, pred::IUGT, selv, k12);
                byte = this->s.m.select(c, above, ff, byte);
                auto at = k(*this, 8 * i);
                byte = this->s.m.bin(c, Bin::Shl, byte, at);
                r = this->s.m.bin(c, Bin::Or, r, byte);
            }
            return r;
        }
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto r = k(*this, 0);
        for (std::uint32_t i = 0; i < 4; ++i) {
            std::uint32_t pick = (sel >> (8 * i)) & 0xFF;
            V byte;
            if (pick <= 7) {
                V src = pick < 4 ? b : a;
                std::uint32_t n = pick < 4 ? pick : pick - 4;
                auto sh = k(*this, 8 * n);
                auto t = this->s.m.bin(c, Bin::LShr, src, sh);
                auto ff = k(*this, 0xFF);
                byte = this->s.m.bin(c, Bin::And, t, ff);
            } else if (pick <= 11) {
                V src = pick < 10 ? b : a;
                std::uint32_t bit = pick % 2 == 0 ? 15 : 31;
                auto left = k(*this, 31 - bit);
                auto t = this->s.m.bin(c, Bin::Shl, src, left);
                auto k31 = k(*this, 31);
                t = this->s.m.bin(c, Bin::AShr, t, k31);
                auto ff = k(*this, 0xFF);
                byte = this->s.m.bin(c, Bin::And, t, ff);
            } else if (pick == 12) {
                continue;
            } else {
                byte = k(*this, 0xFF);
            }
            auto at = k(*this, 8 * i);
            byte = this->s.m.bin(c, Bin::Shl, byte, at);
            r = this->s.m.bin(c, Bin::Or, r, byte);
        }
        return r;
    }
    if (name == "v_bfm_b32") {
        // ((1 << src0[4:0]) - 1) << src1[4:0]
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        auto one = k(*this, 1);
        a = m31(*this, a);
        b = m31(*this, b);
        auto m = this->s.m.bin(c, Bin::Shl, one, a);
        m = this->s.m.bin(c, Bin::Sub, m, one);
        return this->s.m.bin(c, Bin::Shl, m, b);
    }
    if (name == "v_cvt_pk_u8_f32") {
        // src0 rounded to nearest even and saturated to a byte, into byte src1[1:0] of src2.
        auto f = this->fsrc(inst, 0);
        auto b = s(*this, 1);
        auto d = s(*this, 2);
        f = this->unary(26, f);
        auto zero = this->s.m.f32(0);
        auto max = this->s.m.f32(f32_bits(255.0f));
        f = this->binary(35, f, zero);
        f = this->binary(36, f, max);
        auto x = this->s.m.cast(c, Cast::FpToUi, f, this->s.t.i32);
        auto three = k(*this, 3);
        auto k3 = k(*this, 3);
        auto sh = this->s.m.bin(c, Bin::And, b, three);
        sh = this->s.m.bin(c, Bin::Shl, sh, k3);
        x = this->s.m.bin(c, Bin::Shl, x, sh);
        auto ff = k(*this, 0xFF);
        auto hole = this->s.m.bin(c, Bin::Shl, ff, sh);
        auto ones = k(*this, 0xFFFF'FFFFu);
        hole = this->s.m.bin(c, Bin::Xor, hole, ones);
        d = this->s.m.bin(c, Bin::And, d, hole);
        return this->s.m.bin(c, Bin::Or, d, x);
    }
    if (name == "v_ffbl_b32") {
        // -1 for zero, as DXIL's first-bit-low.
        auto a = s(*this, 0);
        return this->s.op(c, "unaryBits.i32", Attr::ReadNone, this->s.t.i32, 32, std::vector<V>{a});
    }
    if (name == "v_mbcnt_lo_u32_b32" || name == "v_mbcnt_hi_u32_b32") {
        // Bits of the mask's low (high) half below this lane, plus src1: with an all-ones mask,
        // the lane index.
        auto a = s(*this, 0);
        auto b = s(*this, 1);
        // Mesh lanes are the console's by construction; pixel and vertex lanes are the host's, as
        // their masks are.
        if (this->mask_data == MaskData::Host) {
            // Host lanes: the mask must be one (a ballot) or all lanes.
            const auto& v = std::get<gfx10::Valu>(inst.fields);
            bool host = false;
            if (!v.src.empty()) {
                const Operand& first = v.src.front();
                if (first.kind == Operand::Kind::Const && first.code == 193) {
                    host = true;
                } else if (first.kind == Operand::Kind::Scalar) {
                    auto m = inst.read(Reg::S(first.code));
                    host = m && (this->x.st->masks.contains(*m) || this->x.st->highs.contains(*m));
                }
            }
            if (!host) {
                fail("v_mbcnt of a mask that is no ballot (console lane layout)");
            }
            this->s.flags |= flags::WAVE_OPS;
        } else if (!this->s.mesh) {
            this->cross_lane(false);
        }
        auto lane = this->console_lane();
        V below;
        if (name == "v_mbcnt_lo_u32_b32") {
            below = lane;
        } else {
            auto k32 = k(*this, 32);
            auto z = k(*this, 0);
            auto t = this->s.m.bin(c, Bin::Sub, lane, k32);
            auto low = this->s.m.cmp(c, pred::IULT, lane, k32);
            below = this->s.m.select(c, low, z, t);
        }
        // (1 << below) - 1, all ones from 32 on.
        auto one = k(*this, 1);
        auto k32 = k(*this, 32);
        auto ones = k(*this, 0xFFFF'FFFFu);
        auto sh = m31(*this, below);
        auto m = this->s.m.bin(c, Bin::Shl, one, sh);
        m = this->s.m.bin(c, Bin::Sub, m, one);
        auto full = this->s.m.cmp(c, pred::IUGE, below, k32);
        m = this->s.m.select(c, full, ones, m);
        auto x = this->s.m.bin(c, Bin::And, a, m);
        auto n = this->s.op(c, "unaryBits.i32", Attr::ReadNone, this->s.t.i32, 31, std::vector<V>{x});
        return this->s.m.bin(c, Bin::Add, n, b);
    }
    fail(name_);
}

void Lower::compare(const Inst& inst, std::string_view rest) {
    auto us = rest.rfind('_');
    if (us == std::string_view::npos) fail("a compare");
    const std::string_view op = rest.substr(0, us);
    const std::string_view ty = rest.substr(us + 1);
    const std::size_t c = this->cur;
    V r;
    if ((ty == "f32" || ty == "f16") && op == "class") {
        // Bit k of src1 tests class k: signalling NaN, quiet NaN, -inf, -normal, -denormal, -0,
        // +0, +denormal, +normal, +inf.
        bool f16 = ty == "f16";
        V x;
        if (f16) {
            const auto& v = std::get<gfx10::Valu>(inst.fields);
            auto h = this->half_src(inst, 0);
            if ((v.abs & 1) != 0) {
                auto m = this->s.m.i32(0x7FFF);
                h = this->s.m.bin(c, Bin::And, h, m);
            }
            if ((v.neg & 1) != 0) {
                auto sg = this->s.m.i32(0x8000);
                h = this->s.m.bin(c, Bin::Xor, h, sg);
            }
            x = h;
        } else {
            auto f = this->fsrc(inst, 0);
            x = this->as_int(f);
        }
        auto mask = this->src(inst, 1);
        auto k = [](Lower& l, std::uint32_t v) { return l.s.m.i32(v); };
        std::uint32_t exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit;
        if (f16) {
            exp_sh = 10, sign_sh = 15, exp_mask = 0x1F, mant_mask = 0x03FF, quiet_bit = 0x0200;
        } else {
            exp_sh = 23, sign_sh = 31, exp_mask = 0xFF, mant_mask = 0x7F'FFFF, quiet_bit = 0x40'0000;
        }
        auto k_exp = k(*this, exp_sh);
        auto k_sign = k(*this, sign_sh);
        auto k_emax = k(*this, exp_mask);
        auto mant = k(*this, mant_mask);
        auto quiet = k(*this, quiet_bit);
        auto e = this->s.m.bin(c, Bin::LShr, x, k_exp);
        e = this->s.m.bin(c, Bin::And, e, k_emax);
        auto m = this->s.m.bin(c, Bin::And, x, mant);
        auto neg = this->s.m.bin(c, Bin::LShr, x, k_sign);
        auto zero = k(*this, 0);
        auto e0 = this->s.m.cmp(c, pred::EQ, e, zero);
        auto emax = this->s.m.cmp(c, pred::EQ, e, k_emax);
        auto m0 = this->s.m.cmp(c, pred::EQ, m, zero);
        auto q = this->s.m.bin(c, Bin::And, x, quiet);
        q = this->s.m.cmp(c, pred::NE, q, zero);
        // Positive classes 6-9; negative ones are 11 minus those.
        auto normal = k(*this, 8);
        auto inf = k(*this, 9);
        auto zero_c = k(*this, 6);
        auto denorm = k(*this, 7);
        auto t = this->s.m.select(c, m0, zero_c, denorm);
        t = this->s.m.select(c, e0, t, normal);
        t = this->s.m.select(c, emax, inf, t);
        auto eleven = k(*this, 11);
        auto mirrored = this->s.m.bin(c, Bin::Sub, eleven, t);
        auto is_neg = this->s.m.cmp(c, pred::NE, neg, zero);
        t = this->s.m.select(c, is_neg, mirrored, t);
        auto one = k(*this, 1);
        auto snan = k(*this, 0);
        auto nan_class = this->s.m.select(c, q, one, snan);
        auto yes = this->s.m.bool_(true);
        auto nan = this->s.m.bin(c, Bin::Xor, m0, yes);
        nan = this->s.m.bin(c, Bin::And, nan, emax);
        t = this->s.m.select(c, nan, nan_class, t);
        auto bit = this->s.m.bin(c, Bin::LShr, mask, t);
        bit = this->s.m.bin(c, Bin::And, bit, one);
        r = this->s.m.cmp(c, pred::NE, bit, zero);
    } else if (ty == "f32" || ty == "f16") {
        V a, b;
        if (ty == "f16") {
            a = this->half_fsrc(inst, 0);
            b = this->half_fsrc(inst, 1);
        } else {
            a = this->fsrc(inst, 0);
            b = this->fsrc(inst, 1);
        }
        if (op == "lg") {
            // Ordered and not equal, false for a NaN, as `olt | ogt`. `fcmp one` means the same,
            // but the AMD D3D12 driver runs it as unordered (true for a NaN) where NVIDIA and
            // WARP do not; olt and ogt agree on all three.
            auto lt = this->s.m.cmp(c, pred::OLT, a, b);
            auto gt = this->s.m.cmp(c, pred::OGT, a, b);
            r = this->s.m.bin(c, Bin::Or, lt, gt);
        } else {
            std::uint8_t p;
            if (op == "f") {
                p = pred::FALSE_;
            } else if (op == "lt") {
                p = pred::OLT;
            } else if (op == "eq") {
                p = pred::OEQ;
            } else if (op == "le") {
                p = pred::OLE;
            } else if (op == "gt") {
                p = pred::OGT;
            } else if (op == "ge") {
                p = pred::OGE;
            } else if (op == "o") {
                p = pred::ORD;
            } else if (op == "u") {
                p = pred::UNO;
            } else if (op == "nge") {
                p = pred::ULT;
            } else if (op == "nlg") {
                p = pred::UEQ;
            } else if (op == "ngt") {
                p = pred::ULE;
            } else if (op == "nle") {
                p = pred::UGT;
            } else if (op == "neq") {
                p = pred::UNE;
            } else if (op == "nlt") {
                p = pred::UGE;
            } else if (op == "tru") {
                p = pred::TRUE_;
            } else {
                fail(inst.name().data());
            }
            r = this->s.m.cmp(c, p, a, b);
        }
    } else if (ty == "u64" || ty == "i64") {
        auto a = this->src64(inst, 0);
        auto b = this->src64(inst, 1);
        bool signed_ = ty == "i64";
        std::uint8_t p;
        if (op == "eq") {
            p = pred::EQ;
        } else if (op == "ne" || op == "lg") {
            p = pred::NE;
        } else if (op == "lt") {
            p = signed_ ? pred::SLT : pred::IULT;
        } else if (op == "le") {
            p = signed_ ? pred::SLE : pred::IULE;
        } else if (op == "gt") {
            p = signed_ ? pred::SGT : pred::IUGT;
        } else if (op == "ge") {
            p = signed_ ? pred::SGE : pred::IUGE;
        } else {
            fail("a 64-bit compare");
        }
        r = this->s.m.cmp(c, p, a, b);
    } else if (ty == "u32" || ty == "i32" || ty == "u16" || ty == "i16") {
        V a, b;
        if (ty.ends_with("16")) {
            a = this->half_src(inst, 0);
            b = this->half_src(inst, 1);
            if (ty == "i16") {
                auto k16 = this->s.m.i32(16);
                for (V* x : {&a, &b}) {
                    auto t = this->s.m.bin(c, Bin::Shl, *x, k16);
                    *x = this->s.m.bin(c, Bin::AShr, t, k16);
                }
            }
        } else {
            a = this->src(inst, 0);
            b = this->src(inst, 1);
        }
        bool signed_ = ty == "i32" || ty == "i16";
        std::uint8_t p;
        if (op == "eq") {
            p = pred::EQ;
        } else if (op == "ne" || op == "lg") {
            p = pred::NE;
        } else if (op == "lt") {
            p = signed_ ? pred::SLT : pred::IULT;
        } else if (op == "le") {
            p = signed_ ? pred::SLE : pred::IULE;
        } else if (op == "gt") {
            p = signed_ ? pred::SGT : pred::IUGT;
        } else if (op == "ge") {
            p = signed_ ? pred::SGE : pred::IUGE;
        } else if (op == "f" || op == "t") {
            this->lane(inst, op == "t");
            return;
        } else {
            fail("an integer compare");
        }
        r = this->s.m.cmp(c, p, a, b);
    } else {
        fail(inst.name().data());
    }
    this->lane_bits_for(inst, r);
}

void Lower::lane(const Inst& inst, std::optional<bool> value) {
    auto b = this->s.m.bool_(value == std::optional<bool>(true));
    this->lane_bits_for(inst, b);
}

void Lower::lane_bits_for(const Inst& inst, V r) {
    const auto* v = std::get_if<gfx10::Valu>(&inst.fields);
    if (!v) return;
    std::optional<std::uint8_t> code;
    if (inst.name().starts_with("v_cmpx")) {
        code = ir::EXEC;
    } else if (v->dst && v->dst->kind == Operand::Kind::Scalar) {
        code = v->dst->code;
    }
    if (code) {
        if (auto lo = inst.write(Reg::S(*code))) {
            this->lane_bits.insert_or_assign(*lo, r);
        }
    }
    for (const auto& [_, w] : inst.writes) {
        this->vals.try_emplace(w, Val::Missing("a lane mask read as data"));
    }
}

void Lower::interp(const Inst& inst, Site site) {
    auto found = this->interps.find(site);
    if (found == this->interps.end()) {
        // The first half of a pair: its second half interpolates the attribute itself, so nothing
        // needs this half's partial sum. But an f16 second half that writes the high half of the
        // register this one wrote keeps the low half as it was (the console compiler reuses the
        // register; the low half is not read), so the register needs a value: 0.
        auto name = inst.name();
        if (name == "v_interp_p1_f32" || name == "v_interp_p1ll_f16") {
            std::optional<Operand> dst;
            if (const auto* vi = std::get_if<gfx10::Vintrp>(&inst.fields)) {
                dst = vi->dst;
            } else if (const auto* va = std::get_if<gfx10::Valu>(&inst.fields)) {
                dst = va->dst;
            }
            if (dst && dst->kind == Operand::Kind::Vector) {
                auto zero = this->s.m.i32(0);
                this->write(inst, Reg::V(dst->reg), zero, std::nullopt);
                return;
            }
        }
        fail(FIRST_HALF);
    }
    const Interp& interp = *found->second;
    const auto name = inst.name();
    Operand dst;
    std::optional<Operand> src;
    if (const auto* vi = std::get_if<gfx10::Vintrp>(&inst.fields)) {
        dst = vi->dst;
        src = vi->src;
    } else if (const auto* va = std::get_if<gfx10::Valu>(&inst.fields)) {
        if (va->abs != 0 || va->neg != 0) {
            fail("interpolation operand modifiers");
        }
        dst = ok_or(va->dst, "an interpolation");
        if (va->src.size() > 1) src = va->src[1];
    } else {
        throw std::logic_error("unreachable: interp fields");
    }
    if (dst.kind != Operand::Kind::Vector) {
        fail("an interpolation");
    }
    const std::uint8_t reg = dst.reg;
    // A float operand of the instruction.
    auto float_ = [&inst](Lower& l, std::optional<Operand> op) -> V {
        auto o = ok_or(op, "an interpolation operand");
        auto x = l.operand(inst, o);
        return l.as_float(x);
    };
    // What the result adds to the evaluated channel (a second half's first half, when the halves
    // evaluate on their own).
    std::optional<V> add;
    io::Attribute attribute;
    Eval eval;
    if (const auto* in = interp.kind == io::Interp::Kind::Interpolated ? &interp : nullptr) {
        attribute = in->attribute;
        auto mode = ok_or(interp_mode(in->barycentric), "pull-model interpolation");
        if (this->per_vertex(attribute.row)) {
            // The element is read per vertex elsewhere, so it is constant: weigh its vertices
            // with SV_Barycentrics.
            std::array<std::optional<V>, 2> ij{};
            for (std::size_t index = 0; index < ij.size(); ++index) {
                Val bv = this->barycentric(in->barycentric, static_cast<std::uint8_t>(index));
                V xv;
                if (bv.is_i32()) {
                    xv = bv.v;
                } else if (bv.is_missing()) {
                    fail(bv.why);
                } else {
                    throw std::logic_error("unreachable: a barycentric bit");
                }
                ij[index] = this->as_float(xv);
            }
            eval = Eval::Weights(true, ij[0], ij[1]);
        } else {
            eval = Eval::Mode(mode);
        }
    } else if (const auto* vx = interp.kind == io::Interp::Kind::Vertex ? &interp : nullptr) {
        attribute = vx->attribute;
        if (vx->vertex == 2) {
            // P0, the provoking vertex: a flat (nointerpolation) input.
            eval = Eval::Mode(1);
        } else if (this->interpolated(attribute.row)) {
            // P10 and P20: what the first and second halves weigh, P1 - P0 and P2 - P0, which is
            // what the hardware holds for an attribute the program also interpolates.
            std::optional<V> one = this->s.m.f32(f32_bits(1.0f));
            std::optional<V> i, j;
            if (vx->vertex == 0) {
                i = one;
            } else {
                j = one;
            }
            eval = Eval::Weights(false, i, j);
        } else {
            fail("a non-provoking vertex value");
        }
    } else {
        const auto& wt = interp;  // Weighted
        attribute = wt.attribute;
        if (name.starts_with("v_interp_p1")) {
            // A half of an interpolation with computed i or j: P0 + i P10, or the old dst (f16:
            // the third operand) + j P20.
            std::optional<V> i = float_(*this, src);
            eval = Eval::Weights(true, i, std::nullopt);
        } else {
            std::optional<Operand> old;
            if (const auto* va = std::get_if<gfx10::Valu>(&inst.fields)) {
                if (va->src.size() > 2) old = va->src[2];
            } else {
                old = dst;
            }
            add = float_(*this, old);
            std::optional<V> j = float_(*this, src);
            eval = Eval::Weights(false, std::nullopt, j);
        }
    }
    // (pack2, packs) of a 16-bit input row.
    std::optional<std::pair<bool, std::vector<std::pair<std::uint8_t, std::optional<std::uint8_t>>>>> packing;
    if (const auto* ps = std::get_if<native::PsIo>(&this->x.native->io)) {
        for (const auto& i : ps->inputs) {
            if (i.row == attribute.row && i.fp16) {
                std::vector<std::pair<std::uint8_t, std::optional<std::uint8_t>>> packs;
                if (i.pack2) {
                    packs = input_packs(this->x.native->io, this->s.inputs);
                }
                packing.emplace(i.pack2, std::move(packs));
                break;
            }
        }
    }
    const bool f16 = name.ends_with("_f16");
    // The evaluated channel plus `add`, as a float.
    auto total = [eval, add](Lower& l, std::uint8_t row, std::uint8_t ch) -> V {
        auto xv = l.eval_channel(row, ch, eval);
        xv = l.as_float(xv);
        if (add) return l.s.m.bin(l.cur, Bin::Add, *add, xv);
        return xv;
    };
    V x;
    if (!packing) {
        if (f16 || attribute.high) fail("a 16-bit read of a 32-bit input");
        auto t = total(*this, attribute.row, attribute.channel);
        x = this->as_int(t);
    } else {
        const bool pack2 = packing->first;
        const auto& packs = packing->second;
        // 16-bit values: one in the low half of each dword, or two per dword (`packed_half`); the
        // host's are 32-bit floats.
        auto half = [&](Lower& l, std::uint8_t h) -> std::optional<V> {
            std::optional<std::pair<std::uint8_t, std::uint8_t>> srcp;
            if (pack2) {
                srcp = packed_half(l.s.inputs, packs, attribute.row, h);
            } else if (h % 2 == 0) {
                srcp = std::pair<std::uint8_t, std::uint8_t>{attribute.row, static_cast<std::uint8_t>(h / 2)};
            }
            if (srcp) return total(l, srcp->first, srcp->second);
            return std::nullopt;
        };
        auto to_f16 = [](Lower& l, V f) {
            auto i32t = l.s.t.i32;
            return l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, std::vector<V>{f});
        };
        const std::uint8_t h = static_cast<std::uint8_t>(2 * attribute.channel + (attribute.high ? 1 : 0));
        const std::size_t c = this->cur;
        if (name == "v_interp_p1ll_f16") {
            // An f32 result from 16-bit vertex values.
            auto f = ok_or(half(*this, h), "a packed input half no element holds");
            x = this->as_int(f);
        } else if (f16) {
            auto f = ok_or(half(*this, h), "a packed input half no element holds");
            if (const auto* v = std::get_if<gfx10::Valu>(&inst.fields)) {
                // Apply omod and clamp before the single rounding to 16 bits. op_sel[3] selects the destination half.
                const std::uint8_t omod = v->omod;
                const bool clamp = v->clamp;
                if (omod != 0) {
                    const float factor = std::array<float, 4>{1.0f, 2.0f, 4.0f, 0.5f}.at(omod);
                    auto kf = this->s.m.f32(f32_bits(factor));
                    f = this->s.m.bin(c, Bin::Mul, f, kf);
                }
                if (clamp) {
                    f = this->unary(7, f);
                }
                auto lo = to_f16(*this, f);
                this->write_half(inst, lo, std::nullopt);
                return;
            }
            auto lo = to_f16(*this, f);
            x = this->low_half(inst, reg, lo);
        } else if (eval.kind == Eval::Kind::Mode && eval.mode == 1) {
            // The whole dword of a flat input: both halves.
            const std::uint8_t h0 = static_cast<std::uint8_t>(2 * attribute.channel);
            auto z = this->s.m.i32(0);
            V lo, hi;
            if (auto f = half(*this, h0)) {
                lo = to_f16(*this, *f);
            } else {
                lo = z;
            }
            if (auto f = half(*this, static_cast<std::uint8_t>(h0 + 1))) {
                hi = to_f16(*this, *f);
            } else {
                hi = z;
            }
            auto k16 = this->s.m.i32(16);
            hi = this->s.m.bin(c, Bin::Shl, hi, k16);
            x = this->s.m.bin(c, Bin::Or, lo, hi);
        } else {
            fail("32-bit interpolation of a 16-bit input");
        }
    }
    // A pixel interpolates in every lane; guards only hide inactive ones.
    this->write(inst, Reg::V(reg), x, std::nullopt);
}

// Whether the program reads attribute row `row` per vertex (an interpolation with computed
// barycentrics, or P10 or P20): then its element is declared constant and read with
// `attributeAtVertex`.
bool Lower::per_vertex(std::uint8_t row) const {
    for (const auto& [_, i] : this->x.io->interps) {
        if (i.kind == io::Interp::Kind::Weighted) {
            const auto* wt = &i;
            if (wt->attribute.row == row) return true;
        } else if (i.kind == io::Interp::Kind::Vertex) {
            const auto* vx = &i;
            if (vx->attribute.row == row && vx->vertex != 2 && this->interpolated(row)) return true;
        }
    }
    return false;
}

// Whether the program interpolates attribute row `row` (so the hardware holds P1 - P0 and P2 - P0
// for it).
bool Lower::interpolated(std::uint8_t row) const {
    for (const auto& [_, i] : this->x.io->interps) {
        if (i.kind == io::Interp::Kind::Interpolated || i.kind == io::Interp::Kind::Weighted) {
            if (i.attribute.row == row) return true;
        }
    }
    return false;
}

// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
V Lower::eval_channel(std::uint8_t row, std::uint8_t ch, Eval eval) {
    auto [id, col] = ok_or(this->find_input(row, ch), "an input the signature lacks");
    const std::uint8_t mode = eval.kind == Eval::Kind::Mode ? eval.mode : 1;
    {
        auto& e = this->s.inputs.at(id);
        if (e.interp != mode && e.usage != 0) {
            fail("one input interpolated two ways");
        }
        e.interp = mode;
    }
    if (eval.kind != Eval::Kind::Weights) {
        Val v = this->load_any_input(id, col);
        if (v.is_i32()) return v.v;
        fail("an input");
    }
    const bool p0 = eval.p0;
    auto v0 = this->vertex_input(id, col, 0);
    const std::size_t c = this->cur;
    std::optional<V> acc;
    if (p0) acc = v0;
    const std::array<std::pair<std::uint8_t, std::optional<V>>, 2> ws{{{1, eval.i}, {2, eval.j}}};
    for (const auto& [k, w] : ws) {
        if (!w) continue;
        auto vk = this->vertex_input(id, col, k);
        auto d = this->s.m.bin(c, Bin::Sub, vk, v0);
        if (acc) {
            acc = this->fmad(*w, d, *acc);
        } else {
            acc = this->s.m.bin(c, Bin::Mul, *w, d);
        }
    }
    auto a = ok_or(acc, "an interpolation");
    return this->as_int(a);
}

// Vertex `k` (0 the provoking vertex, as `barycentric` orders them) of a constant input element's
// channel, as a float.
V Lower::vertex_input(std::size_t id, std::uint8_t col, std::uint8_t k) {
    auto& e = this->s.inputs.at(id);
    e.usage |= static_cast<std::uint8_t>(1u << col);
    const bool int_ = e.comp != 9;
    Ty ty = int_ ? this->s.t.i32 : this->s.t.f32;
    std::string_view suffix = int_ ? ".i32" : ".f32";
    this->s.flags |= flags::BARYCENTRICS;
    const std::size_t entry = this->s.entry;
    auto a0 = this->s.m.i32(static_cast<std::uint32_t>(id));
    auto a1 = this->s.m.i32(0);
    auto a2 = this->s.m.i8(col);
    auto a3 = this->s.m.i8(k);
    std::string opname = "attributeAtVertex";
    opname += suffix;
    auto v = this->s.op(entry, opname, Attr::ReadNone, ty, 137, std::vector<V>{a0, a1, a2, a3});
    if (int_) {
        return this->s.m.cast(entry, Cast::Bitcast, v, this->s.t.f32);
    }
    return v;
}

// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit results keep the register's
// high half (GFX10).
V Lower::low_half(const Inst& inst, std::uint8_t reg, V x) {
    const std::size_t c = this->cur;
    std::optional<V> got;
    if (auto o = inst.read(Reg::V(reg))) {
        got = ok([&] { return this->i32_of(*o); });
    }
    V old = got ? *got : this->s.m.i32(0);
    auto high = this->s.m.i32(0xFFFF'0000u);
    auto low = this->s.m.i32(0xFFFF);
    old = this->s.m.bin(c, Bin::And, old, high);
    x = this->s.m.bin(c, Bin::And, x, low);
    return this->s.m.bin(c, Bin::Or, old, x);
}

}  // namespace x2w::vs::lower
