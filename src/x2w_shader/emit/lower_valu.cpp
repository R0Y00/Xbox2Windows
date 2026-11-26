// VALU instructions. See GMR/vs_port/lower/PORTING.md for the conventions.
#include "x2w_shader/emit/lower_impl.h"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace x2w::vs::lower {

namespace {

// `Some(Operand::Vector { reg, .. })` of an optional operand.
std::optional<std::uint8_t> vector_reg(const std::optional<Operand>& o) {
    if (o && o->kind == Operand::Kind::Vector) return o->reg;
    return std::nullopt;
}
std::optional<std::uint8_t> vector_reg(const Operand* o) {
    if (o && o->kind == Operand::Kind::Vector) return o->reg;
    return std::nullopt;
}
// `Some(Operand::Scalar { code, .. })` of an optional operand.
std::optional<std::uint8_t> scalar_code(const std::optional<Operand>& o) {
    if (o && o->kind == Operand::Kind::Scalar) return o->code;
    return std::nullopt;
}
std::optional<std::uint8_t> scalar_code(const Operand* o) {
    if (o && o->kind == Operand::Kind::Scalar) return o->code;
    return std::nullopt;
}
// `v.src.get(k)`.
const Operand* src_at(const gfx10::Valu& v, std::size_t k) {
    return k < v.src.size() ? &v.src[k] : nullptr;
}

}  // namespace

void Lower::valu(const Inst& inst, std::size_t b, std::size_t i) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    std::string_view name = inst.name();
  
    {
        const auto& compares = this->x.res->waterfall.compares;
        if (std::find(compares.begin(), compares.end(), Site{b, i}) != compares.end()) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            this->lane(inst, true);
            return;
        }
    }
    // One lane of a VGPR a whole-wave section left.
    if (name == "v_readlane_b32" && !inst.writes.empty()) {
        Value w = inst.writes.front().second;
        std::optional<wave::Shape> sh;
        if (auto s = this->sections.find(b); s != this->sections.end()) {
            // Take the first completed section that read w. Sections do not overlap, and each value is normally read by
            // at most one qualifying section. The table has no stable iteration order.
            for (const auto& [start, sec] : s->second) {
                if (!(sec.end < i)) continue;
                if (auto r = sec.reads.find(w); r != sec.reads.end()) {
                    sh = r->second;
                    break;
                }
            }
        }
        if (sh) {
            V x = this->shape(*sh, false);
            this->vals.insert_or_assign(w, Val::I32(x));
            return;
        }
    }
    {
        std::optional<std::string_view> rest;
        if (name.starts_with("v_cmp_")) {
            rest = name.substr(6);
        } else if (name.starts_with("v_cmpx_")) {
            rest = name.substr(7);
        }
        if (rest) {
            std::optional<Why> why;
            if (v.dpp) {
                why = "a compare with DPP";
            } else {
                try {
                    this->compare(inst, *rest);
                } catch (const WhyFail& e) {
                    why = e.why;
                }
            }
            if (why) {
                for (const auto& [_, w] : inst.writes) {
                    this->lane_fail.insert_or_assign(w, *why);
                }
                fail(*why);
            }
            return;
        }
    }
    std::optional<V> guard = this->guard(b, i);
    if (v.dpp) {
        auto [x, enable] = this->dpp(inst, *v.dpp);
        if (guard && enable) {
            guard = this->s.m.bin(this->cur, Bin::And, *guard, *enable);
        } else if (!guard) {
            guard = enable;
        }
        this->src0 = x;
    }
    // Clear src0 after evaluating the instruction.
    try {
        this->valu_op(inst, guard);
    } catch (...) {
        this->src0 = std::nullopt;
        throw;
    }
    this->src0 = std::nullopt;
}

// DPP: src0 as read from another lane, and which lanes write their result (none: all).
std::pair<V, std::optional<V>> Lower::dpp(const Inst& inst, const Dpp& dpp) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    auto reg0 = vector_reg(v.src.empty() ? nullptr : &v.src.front());
    if (!reg0) fail("a DPP operand");
    std::uint8_t reg = *reg0;
    V x = this->i32_of(ok_or(inst.read(Reg::V(reg)), "a DPP operand"));
    // Quad permutations and DPP8 stay in groups the pixel layout shares.
    const auto* row_dpp = std::get_if<gfx10::DppRow>(&dpp);
    bool quad = row_dpp && row_dpp->ctrl <= 0xFF;
    bool mesh_quad = this->s.mesh.has_value() && quad;
    if (!mesh_quad) {
        this->cross_lane(quad);
    }
    std::size_t c = this->cur;
    V lane = this->lane_index();
    auto k = [&](std::uint32_t n) { return this->s.m.i32(n); };
    V sixteen_mask = k(~15u);
    V low = k(15);
    V row = this->s.m.bin(c, Bin::And, lane, sixteen_mask);
    V pos = this->s.m.bin(c, Bin::And, lane, low);
    // Source lane, and whether it lies in the row.
    V src;
    std::optional<V> inside;
    std::optional<std::pair<std::uint8_t, std::uint8_t>> masks;
    bool zero = false;
    if (const auto* lanes = std::get_if<gfx10::DppLanes>(&dpp)) {
        V seven = k(7);
        V not7 = k(~7u);
        V three = k(3);
        V p = this->s.m.bin(c, Bin::And, lane, seven);
        V sh = this->s.m.bin(c, Bin::Mul, p, three);
        V sel = k(lanes->sel);
        V pick = this->s.m.bin(c, Bin::LShr, sel, sh);
        pick = this->s.m.bin(c, Bin::And, pick, seven);
        V base = this->s.m.bin(c, Bin::And, lane, not7);
        src = this->s.m.bin(c, Bin::Or, base, pick);
    } else {
        const auto& d = *row_dpp;
        std::uint16_t ctrl = d.ctrl;
        std::uint32_t n = std::uint32_t(ctrl & 15);
        if (ctrl <= 0xFF) {
            V three = k(3);
            V one = k(1);
            V not3 = k(~3u);
            V p = this->s.m.bin(c, Bin::And, lane, three);
            V sh = this->s.m.bin(c, Bin::Shl, p, one);
            V perm = k(std::uint32_t(ctrl));
            V pick = this->s.m.bin(c, Bin::LShr, perm, sh);
            pick = this->s.m.bin(c, Bin::And, pick, three);
            V base = this->s.m.bin(c, Bin::And, lane, not3);
            src = this->s.m.bin(c, Bin::Or, base, pick);
        } else if ((ctrl >= 0x101 && ctrl <= 0x10F) || (ctrl >= 0x111 && ctrl <= 0x11F)) {
            V nn = k(n);
            bool left = ctrl < 0x110;
            Bin op = left ? Bin::Add : Bin::Sub;
            std::uint8_t p = left ? pred::IULT : pred::IUGE;
            V s = this->s.m.bin(c, op, lane, nn);
            // Left: pos + n < 16; right: pos >= n.
            V bound = left ? k(16 - n) : nn;
            src = s;
            inside = this->s.m.cmp(c, p, pos, bound);
        } else if (ctrl >= 0x121 && ctrl <= 0x12F) {
            V nn = k(n);
            V t = this->s.m.bin(c, Bin::Sub, pos, nn);
            t = this->s.m.bin(c, Bin::And, t, low);
            src = this->s.m.bin(c, Bin::Or, row, t);
        } else if (ctrl == 0x140 || ctrl == 0x141) {
            V m = k(ctrl == 0x140 ? 15 : 7);
            src = this->s.m.bin(c, Bin::Xor, lane, m);
        } else if (ctrl >= 0x150 && ctrl <= 0x15F) {
            V nn = k(n);
            src = this->s.m.bin(c, Bin::Or, row, nn);
        } else if (ctrl >= 0x160 && ctrl <= 0x16F) {
            V nn = k(n);
            src = this->s.m.bin(c, Bin::Xor, lane, nn);
        } else {
            fail("a DPP lane pattern");
        }
        if (d.row_mask != 0xF || d.bank_mask != 0xF) masks = std::pair{d.row_mask, d.bank_mask};
        zero = d.bound_ctrl;
    }
    V r;
    if (mesh_quad) {
        r = this->quad_pick(x, src);
    } else if (row_dpp && this->helpers_read(quad)) {
        std::uint16_t ctrl = row_dpp->ctrl;
        std::array<std::uint8_t, 4> pick{};
        for (int q = 0; q < 4; ++q) pick[q] = std::uint8_t((ctrl >> (2 * q)) & 3);
        r = this->quad_read(x, pick);
    } else {
        r = this->read_lane(x, src);
    }
    std::optional<V> enable;
    if (inside) {
        if (zero) {
            V z = k(0);
            r = this->s.m.select(c, *inside, r, z);
        } else {
            enable = *inside;
        }
    }
    if (masks) {
        auto [rows, banks] = *masks;
        // Row (16 lanes) and bank (4 lanes of a row) write masks.
        V four = k(4);
        V two = k(2);
        V three = k(3);
        V ri = this->s.m.bin(c, Bin::LShr, lane, four);
        ri = this->s.m.bin(c, Bin::And, ri, three);
        V bi = this->s.m.bin(c, Bin::LShr, lane, two);
        bi = this->s.m.bin(c, Bin::And, bi, three);
        V rm = k(std::uint32_t(rows));
        V bm = k(std::uint32_t(banks));
        V rb = this->s.m.bin(c, Bin::LShr, rm, ri);
        V bb = this->s.m.bin(c, Bin::LShr, bm, bi);
        V both = this->s.m.bin(c, Bin::And, rb, bb);
        V one = k(1);
        V z = k(0);
        both = this->s.m.bin(c, Bin::And, both, one);
        V on = this->s.m.cmp(c, pred::NE, both, z);
        if (enable) {
            enable = this->s.m.bin(c, Bin::And, *enable, on);
        } else {
            enable = on;
        }
    }
    return {r, enable};
}

void Lower::valu_op(const Inst& inst, std::optional<V> guard) {
    const auto& v = std::get<gfx10::Valu>(inst.fields);
    std::string_view name = inst.name();
    std::size_t c = this->cur;
    auto f = [&](std::size_t k) { return this->fsrc(inst, k); };
    auto s = [&](std::size_t k) { return this->src(inst, k); };
    auto float_w = [&](V r) { this->write_f(inst, r, guard); };
    auto int_w = [&](V r) { this->write_i(inst, r, guard); };
    if (this->valu16(inst, name, guard)) {
        return;
    }
    if (name.starts_with("v_pk_")) {
        this->packed16(inst, name.substr(5), guard);
        return;
    }
    auto is = [&](std::initializer_list<std::string_view> names) {
        for (auto n : names) {
            if (name == n) return true;
        }
        return false;
    };
    if (name == "v_mad_u64_u32") {
        // a * b + c, c and the result 64 bits wide; the carry out
        // of the sum is this lane's bit of the mask it names.
        V a = s(0);
        V b = s(1);
        V x = this->src64(inst, 2);
        Ty i64t = this->s.m.int_(64);
        this->s.flags |= flags::INT64;
        a = this->s.m.cast(c, Cast::ZExt, a, i64t);
        b = this->s.m.cast(c, Cast::ZExt, b, i64t);
        V product = this->s.m.bin(c, Bin::Mul, a, b);
        V r = this->s.m.bin(c, Bin::Add, product, x);
        V carry = this->s.m.cmp(c, pred::IULT, r, x);
        auto reg = vector_reg(v.dst);
        if (!reg) fail("a 64-bit result");
        auto [lo, hi] = this->split64(r);
        this->write(inst, Reg::V(*reg), lo, guard);
        this->write(inst, Reg::V(std::uint8_t(*reg + 1)), hi, guard);
        if (auto code = scalar_code(v.carry)) {
            if (auto w = inst.write(Reg::S(*code))) {
                this->lane_bits.insert_or_assign(*w, carry);
            }
        }
        for (const auto& [_, w] : inst.writes) {
            this->vals.try_emplace(w, Val::Missing("a lane mask read as data"));
        }
        return;
    } else if (is({"v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64"})) {
        V n = s(0);
        V x = this->src64(inst, 1);
        Ty i64t = this->s.m.int_(64);
        V k63 = this->s.m.i32(63);
        n = this->s.m.bin(c, Bin::And, n, k63);
        n = this->s.m.cast(c, Cast::ZExt, n, i64t);
        Bin op = name == "v_lshlrev_b64" ? Bin::Shl : name == "v_lshrrev_b64" ? Bin::LShr : Bin::AShr;
        V r = this->s.m.bin(c, op, x, n);
        auto reg = vector_reg(v.dst);
        if (!reg) fail("a 64-bit result");
        auto [lo, hi] = this->split64(r);
        this->write(inst, Reg::V(*reg), lo, guard);
        this->write(inst, Reg::V(std::uint8_t(*reg + 1)), hi, guard);
        return;
    } else if (is({"v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32", "v_cubema_f32"})) {
        // Cube-map face coordinates (AMD's ISA): the major axis is z
        // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
        // else x.
        V x = f(0);
        V y = f(1);
        V z = f(2);
        V ax = this->unary(6, x);
        V ay = this->unary(6, y);
        V az = this->unary(6, z);
        V zx = this->s.m.cmp(c, pred::OGE, az, ax);
        V zy = this->s.m.cmp(c, pred::OGE, az, ay);
        V zmaj = this->s.m.bin(c, Bin::And, zx, zy);
        V ymaj = this->s.m.cmp(c, pred::OGE, ay, ax);
        V zero = this->s.m.f32(0);
        auto neg = [&](V w) { return this->s.m.cmp(c, pred::OLT, w, zero); };
        auto minus = [&](V w) {
            V zz = this->s.m.f32(0x8000'0000u);
            return this->s.m.bin(c, Bin::Sub, zz, w);
        };
        auto pick = [&](V zv, V yv, V xv) {
            V t = this->s.m.select(c, ymaj, yv, xv);
            return this->s.m.select(c, zmaj, zv, t);
        };
        V r;
        if (name == "v_cubeid_f32") {
            auto k = [&](float w) { return this->s.m.f32(f32_bits(w)); };
            V nz = neg(z);
            V ny = neg(y);
            V nx = neg(x);
            V f5 = k(5.0f);
            V f4 = k(4.0f);
            V f3 = k(3.0f);
            V f2 = k(2.0f);
            V f1 = k(1.0f);
            V f0 = k(0.0f);
            V zf = this->s.m.select(c, nz, f5, f4);
            V yf = this->s.m.select(c, ny, f3, f2);
            V xf = this->s.m.select(c, nx, f1, f0);
            r = pick(zf, yf, xf);
        } else if (name == "v_cubesc_f32") {
            V nz = neg(z);
            V nx = neg(x);
            V mx = minus(x);
            V mz = minus(z);
            V zf = this->s.m.select(c, nz, mx, x);
            V xf = this->s.m.select(c, nx, z, mz);
            r = pick(zf, x, xf);
        } else if (name == "v_cubetc_f32") {
            V ny = neg(y);
            V my = minus(y);
            V mz = minus(z);
            V yf = this->s.m.select(c, ny, mz, z);
            r = pick(my, yf, my);
        } else {
            V t = pick(z, y, x);
            V two = this->s.m.f32(f32_bits(2.0f));
            r = this->s.m.bin(c, Bin::Mul, t, two);
        }
        float_w(r);
        return;
    } else if (is({"v_movrels_b32", "v_movreld_b32", "v_movrelsd_b32"})) {
        // M0 indexes the source (movrels), the destination (movreld)
        // or both, from the encoded register up to the program's
        // VGPR count (the registers the instruction reads or writes).
        V m0 = this->i32_of(ok_or(inst.read(Reg::S(ir::M0)), "M0"));
        auto sreg_o = vector_reg(v.src.empty() ? nullptr : &v.src.front());
        auto dreg_o = vector_reg(v.dst);
        if (!sreg_o || !dreg_o) fail("an M0-relative move");
        std::uint8_t sreg = *sreg_o;
        std::uint8_t dreg = *dreg_o;
        bool indexed_src = name != "v_movreld_b32";
        V value;
        if (indexed_src) {
            V r = this->s.m.i32(0);
            for (unsigned k = 0; k <= 255u - sreg; ++k) {
                auto xv = inst.read(Reg::V(std::uint8_t(sreg + k)));
                if (!xv) break;
                V xx = this->i32_of(*xv);
                V kk = this->s.m.i32(std::uint32_t(k));
                V eq = this->s.m.cmp(c, pred::EQ, m0, kk);
                r = this->s.m.select(c, eq, xx, r);
            }
            value = r;
        } else {
            value = s(0);
        }
        if (name == "v_movrels_b32") {
            int_w(value);
            return;
        }
        for (unsigned k = 0; k <= 255u - dreg; ++k) {
            Reg rk = Reg::V(std::uint8_t(dreg + k));
            if (!inst.write(rk)) break;
            V old;
            if (auto o = inst.read(rk)) {
                old = this->i32_of(*o);
            } else {
                old = this->s.m.i32(0);
            }
            V kk = this->s.m.i32(std::uint32_t(k));
            V eq = this->s.m.cmp(c, pred::EQ, m0, kk);
            V xx = this->s.m.select(c, eq, value, old);
            this->write_placed(inst, rk, xx, guard);
        }
        return;
    } else if (name == "v_swap_b32") {
        auto sreg_o = vector_reg(v.src.empty() ? nullptr : &v.src.front());
        auto dreg_o = vector_reg(v.dst);
        if (!sreg_o || !dreg_o) fail("v_swap_b32");
        std::uint8_t sreg = *sreg_o;
        std::uint8_t dreg = *dreg_o;
        V a = this->i32_of(ok_or(inst.read(Reg::V(dreg)), "v_swap_b32"));
        V b = this->i32_of(ok_or(inst.read(Reg::V(sreg)), "v_swap_b32"));
        this->write_placed(inst, Reg::V(dreg), b, guard);
        if (dreg != sreg) {
            this->write_placed(inst, Reg::V(sreg), a, guard);
        }
        return;
    } else if (name == "v_ffbh_u32" || name == "v_ffbh_i32") {
        // Counted from the top bit, -1 when none (DXIL's too).
        V a = s(0);
        std::uint32_t op = name == "v_ffbh_u32" ? 33 : 34;
        V r = this->s.op(c, "unaryBits.i32", Attr::ReadNone, this->s.t.i32, op, std::vector<V>{a});
        int_w(r);
        return;
    } else if (name == "v_dot4c_i32_i8") {
        // D += the dot product of four signed bytes, wrapping: DXIL's
        // Dot4AddI8Packed.
        V a = s(0);
        V bb = s(1);
        auto reg = vector_reg(v.dst);
        if (!reg) fail("an accumulator");
        V acc = this->i32_of(ok_or(inst.read(Reg::V(*reg)), "an accumulator"));
        Ty i32_ty = this->s.t.i32;
        V r = this->s.op(c, "dot4AddPacked.i32", Attr::ReadNone, i32_ty, 163, std::vector<V>{acc, a, bb});
        int_w(r);
        return;
    } else if (name == "v_dot4_i32_i8") {
        // VOP3P: op_sel picks the half each multiplicand's low half
        // reads, op_sel_hi its high half's; the accumulator is read
        // whole (as AMD's compilers fold constants into it).
        if (v.clamp || v.neg_hi != 0) {
            fail("a saturating or negated dot product");
        }
        std::vector<V> xs;
        for (std::size_t k = 0; k < 2; ++k) {
            V xx = s(k);
            bool lo = ((v.opsel >> k) & 1) != 0;
            bool hi = ((v.opsel_hi >> k) & 1) != 0;
            if (!lo && hi) {
                xs.push_back(xx);
                continue;
            }
            V k16 = this->s.m.i32(16);
            V lowm = this->s.m.i32(0xFFFF);
            V top = this->s.m.bin(c, Bin::LShr, xx, k16);
            V bottom = this->s.m.bin(c, Bin::And, xx, lowm);
            V hv = this->s.m.bin(c, Bin::Shl, hi ? top : bottom, k16);
            xs.push_back(this->s.m.bin(c, Bin::Or, lo ? top : bottom, hv));
        }
        V acc = s(2);
        Ty i32_ty = this->s.t.i32;
        V r = this->s.op(c, "dot4AddPacked.i32", Attr::ReadNone, i32_ty, 163,
                         std::vector<V>{acc, xs[0], xs[1]});
        int_w(r);
        return;
    } else if (name == "v_mov_b32") {
        if (v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0) {
            V a = f(0);
            float_w(a);
            return;
        }
        V a = s(0);
        int_w(a);
        return;
    } else if (is({"v_add_f32", "v_sub_f32", "v_subrev_f32", "v_mul_f32", "v_mul_legacy_f32", "v_max_f32",
                   "v_min_f32"})) {
        V a = f(0);
        V bb = f(1);
        V r;
        if (name == "v_add_f32") {
            r = this->s.m.bin(c, Bin::Add, a, bb);
        } else if (name == "v_sub_f32") {
            r = this->s.m.bin(c, Bin::Sub, a, bb);
        } else if (name == "v_subrev_f32") {
            r = this->s.m.bin(c, Bin::Sub, bb, a);
        } else if (name == "v_mul_f32") {
            r = this->s.m.bin(c, Bin::Mul, a, bb);
        } else if (name == "v_mul_legacy_f32") {
            r = this->legacy_mul(a, bb);
        } else if (name == "v_max_f32") {
            r = this->max_min(true, a, bb);
        } else {
            r = this->max_min(false, a, bb);
        }
        float_w(r);
        return;
    } else if (is({"v_fma_f32", "v_fmaak_f32", "v_fmamk_f32"})) {
        // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
        V a = f(0);
        V bb = f(1);
        V cc = f(2);
        V r = this->fmad(a, bb, cc);
        float_w(r);
        return;
    } else if (name == "v_div_scale_f32") {
        // RDNA2 ISA: src0 = value to scale, src1 = denominator, src2 = numerator.
        V a = f(0);
        V den = f(1);
        V num = f(2);
        V ai = this->as_int(a);
        V ni = this->as_int(num);
        V di = this->as_int(den);
        V k23 = this->s.m.i32(23);
        V k255 = this->s.m.i32(0xFF);
        V kabs = this->s.m.i32(0x7FFF'FFFF);
        V zero = this->s.m.i32(0);
        V k23_exp = this->s.m.i32(23);
        V k232_exp = this->s.m.i32(232);
        V k96 = this->s.m.i32(96);
        auto exp_of = [&](V xx) {
            V sh = this->s.m.bin(c, Bin::LShr, xx, k23);
            return this->s.m.bin(c, Bin::And, sh, k255);
        };
        V e1 = exp_of(ni);
        V e2 = exp_of(di);
        V n_abs = this->s.m.bin(c, Bin::And, ni, kabs);
        V d_abs = this->s.m.bin(c, Bin::And, di, kabs);
        V z1 = this->s.m.cmp(c, pred::EQ, n_abs, zero);
        V z2 = this->s.m.cmp(c, pred::EQ, d_abs, zero);
        V any_zero = this->s.m.bin(c, Bin::Or, z1, z2);
        V special;
        {
            V s1 = this->s.m.cmp(c, pred::EQ, e1, k255);
            V s2 = this->s.m.cmp(c, pred::EQ, e2, k255);
            V a12 = this->s.m.bin(c, Bin::Or, s1, s2);
            special = this->s.m.bin(c, Bin::Or, a12, any_zero);
        }
        V is_den = this->s.m.cmp(c, pred::EQ, ai, di);
        V diff = this->s.m.bin(c, Bin::Sub, e2, e1);
        V den_tiny = this->s.m.cmp(c, pred::IULE, e2, k23_exp);
        V den_huge = this->s.m.cmp(c, pred::IUGE, e2, k232_exp);
        V num_tiny_exp = this->s.m.cmp(c, pred::IULE, e1, k23_exp);
        V num_huge = this->s.m.cmp(c, pred::IUGE, e1, k232_exp);
        V num_gap = this->s.m.cmp(c, pred::SGE, diff, k96);
        V not_den_huge;
        {
            V t = this->s.m.bool_(true);
            not_den_huge = this->s.m.bin(c, Bin::Xor, den_huge, t);
        }
        num_gap = this->s.m.bin(c, Bin::And, num_gap, not_den_huge);
        V num_tiny = this->s.m.bin(c, Bin::Or, num_tiny_exp, num_gap);
        V up_raw = this->s.m.select(c, is_den, den_tiny, num_tiny);
        V dn_raw = this->s.m.select(c, is_den, den_huge, num_huge);
        V not_special;
        {
            V t = this->s.m.bool_(true);
            not_special = this->s.m.bin(c, Bin::Xor, special, t);
        }
        V up = this->s.m.bin(c, Bin::And, up_raw, not_special);
        V dn = this->s.m.bin(c, Bin::And, dn_raw, not_special);
        V uncancelled;
        {
            V x_tiny = this->s.m.bin(c, Bin::Xor, den_tiny, num_tiny);
            V x_huge = this->s.m.bin(c, Bin::Xor, den_huge, num_huge);
            uncancelled = this->s.m.bin(c, Bin::Or, x_tiny, x_huge);
        }
        V flag_raw = this->s.m.select(c, is_den, uncancelled, up_raw);
        V flag = this->s.m.bin(c, Bin::And, flag_raw, not_special);
        V k_up = this->s.m.f32(0x5F80'0000u);
        V k_dn = this->s.m.f32(0x1F80'0000u);
        V a_up = this->s.m.bin(c, Bin::Mul, a, k_up);
        V a_dn = this->s.m.bin(c, Bin::Mul, a, k_dn);
        V r = this->s.m.select(c, dn, a_dn, a);
        r = this->s.m.select(c, up, a_up, r);
        V qnan = this->s.m.f32(0x7FC0'0000u);
        r = this->s.m.select(c, any_zero, qnan, r);
        float_w(r);
        if (auto code = scalar_code(v.carry)) {
            if (auto w = inst.write(Reg::S(*code))) {
                this->lane_bits.insert_or_assign(*w, flag);
            }
        }
        for (const auto& [_, w] : inst.writes) {
            this->vals.try_emplace(w, Val::Missing("a lane mask read as data"));
        }
        return;
    } else if (name == "v_div_fmas_f32") {
        V a = f(0);
        V bb = f(1);
        V cc = f(2);
        std::optional<Bool> m;
        if (auto vcc = inst.read(Reg::S(ir::VCC))) {
          
            if (auto it = this->x.st->masks.find(*vcc); it != this->x.st->masks.end()) m = it->second;
        }
        if (!m) fail("v_div_fmas_f32 without VCC");
        V vcc = this->boolean(*m, c);
        V tmp = this->fmad(a, bb, cc);
        V ci = this->as_int(cc);
        V k23 = this->s.m.i32(23);
        V k255 = this->s.m.i32(0xFF);
        V k128 = this->s.m.i32(128);
        V sh = this->s.m.bin(c, Bin::LShr, ci, k23);
        V exp = this->s.m.bin(c, Bin::And, sh, k255);
        V exp_ge = this->s.m.cmp(c, pred::IUGE, exp, k128);
        V k_up = this->s.m.f32(0x5F80'0000u);
        V k_dn = this->s.m.f32(0x1F80'0000u);
        V scale = this->s.m.select(c, exp_ge, k_up, k_dn);
        V scaled = this->s.m.bin(c, Bin::Mul, tmp, scale);
        V r = this->s.m.select(c, vcc, scaled, tmp);
        float_w(r);
        return;
    } else if (name == "v_div_fixup_f32") {
        V q = f(0);
        V den = f(1);
        V num = f(2);
        V qi = this->as_int(q);
        V di = this->as_int(den);
        V ni = this->as_int(num);
        V ksign = this->s.m.i32(0x8000'0000u);
        V kabs = this->s.m.i32(0x7FFF'FFFF);
        V kinf = this->s.m.i32(0x7F80'0000);
        V kqnan_bit = this->s.m.i32(0x0040'0000);
        V kneg_qnan = this->s.m.i32(0xFFC0'0000u);
        V zero = this->s.m.i32(0);
        V sign;
        {
            V xx = this->s.m.bin(c, Bin::Xor, ni, di);
            sign = this->s.m.bin(c, Bin::And, xx, ksign);
        }
        V q_abs = this->s.m.bin(c, Bin::And, qi, kabs);
        V d_abs = this->s.m.bin(c, Bin::And, di, kabs);
        V n_abs = this->s.m.bin(c, Bin::And, ni, kabs);
        V q_nan = this->s.m.cmp(c, pred::IUGT, q_abs, kinf);
        V d_nan = this->s.m.cmp(c, pred::IUGT, d_abs, kinf);
        V n_nan = this->s.m.cmp(c, pred::IUGT, n_abs, kinf);
        V d_inf = this->s.m.cmp(c, pred::EQ, d_abs, kinf);
        V n_inf = this->s.m.cmp(c, pred::EQ, n_abs, kinf);
        V d_zero = this->s.m.cmp(c, pred::EQ, d_abs, zero);
        V n_zero = this->s.m.cmp(c, pred::EQ, n_abs, zero);
        V both_inf = this->s.m.bin(c, Bin::And, n_inf, d_inf);
        V both_zero = this->s.m.bin(c, Bin::And, n_zero, d_zero);
        V invalid = this->s.m.bin(c, Bin::Or, both_inf, both_zero);
        V make_inf;
        {
            V t = this->s.m.bin(c, Bin::Or, n_inf, d_zero);
            make_inf = this->s.m.bin(c, Bin::Or, t, q_nan);
        }
        V make_zero = this->s.m.bin(c, Bin::Or, d_inf, n_zero);
        V signed_q = this->s.m.bin(c, Bin::Or, sign, q_abs);
        V signed_inf = this->s.m.bin(c, Bin::Or, sign, kinf);
        V n_qnan = this->s.m.bin(c, Bin::Or, ni, kqnan_bit);
        V d_qnan = this->s.m.bin(c, Bin::Or, di, kqnan_bit);
        V r = this->s.m.select(c, make_zero, sign, signed_q);
        r = this->s.m.select(c, make_inf, signed_inf, r);
        r = this->s.m.select(c, invalid, kneg_qnan, r);
        r = this->s.m.select(c, d_nan, d_qnan, r);
        r = this->s.m.select(c, n_nan, n_qnan, r);
        V rf = this->as_float(r);
        float_w(rf);
        return;
    } else if (is({"v_fmac_f32", "v_mac_f32", "v_mac_legacy_f32"})) {
        V a = f(0);
        V bb = f(1);
        auto reg = vector_reg(v.dst);
        if (!reg) fail("an accumulator");
        Value acc_v = ok_or(inst.read(Reg::V(*reg)), "an accumulator");
        V acc;
        if (auto it = this->floats.find(acc_v); it != this->floats.end()) {
            acc = it->second;
        } else {
            V xx = this->i32_of(acc_v);
            acc = this->as_float(xx);
        }
        V r;
        if (name == "v_fmac_f32") {
            r = this->fmad(a, bb, acc);
        } else if (name == "v_mac_f32") {
            V m = this->s.m.bin(c, Bin::Mul, a, bb);
            r = this->s.m.bin(c, Bin::Add, m, acc);
        } else {
            V m = this->legacy_mul(a, bb);
            r = this->s.m.bin(c, Bin::Add, m, acc);
        }
        float_w(r);
        return;
    } else if (is({"v_mad_f32", "v_madak_f32", "v_madmk_f32", "v_mad_legacy_f32"})) {
        V a = f(0);
        V bb = f(1);
        V cc = f(2);
        V m = name == "v_mad_legacy_f32" ? this->legacy_mul(a, bb) : this->s.m.bin(c, Bin::Mul, a, bb);
        V r = this->s.m.bin(c, Bin::Add, m, cc);
        float_w(r);
        return;
    } else if (is({"v_max3_f32", "v_min3_f32", "v_med3_f32"})) {
        V a = f(0);
        V bb = f(1);
        V cc = f(2);
        V r;
        if (name == "v_max3_f32") {
            V t = this->max_min(true, a, bb);
            r = this->max_min(true, t, cc);
        } else if (name == "v_min3_f32") {
            V t = this->max_min(false, a, bb);
            r = this->max_min(false, t, cc);
        } else {
            r = this->med3(a, bb, cc);
        }
        float_w(r);
        return;
    } else if (name == "v_rcp_f32" || name == "v_rcp_iflag_f32") {
        // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
        V a = f(0);
        V one = this->s.m.f32(f32_bits(1.0f));
        V r = this->s.m.bin(c, Bin::SDiv, one, a);
        float_w(r);
        return;
    } else if (is({"v_rsq_f32", "v_sqrt_f32", "v_log_f32", "v_exp_f32", "v_fract_f32", "v_trunc_f32",
                   "v_floor_f32", "v_ceil_f32", "v_rndne_f32", "v_sin_f32", "v_cos_f32"})) {
        V a = f(0);
        std::uint32_t op = name == "v_rsq_f32"     ? 25
                           : name == "v_sqrt_f32"  ? 24
                           : name == "v_log_f32"   ? 23
                           : name == "v_exp_f32"   ? 21
                           : name == "v_fract_f32" ? 22
                           : name == "v_trunc_f32" ? 29
                           : name == "v_floor_f32" ? 27
                           : name == "v_ceil_f32"  ? 28
                           : name == "v_rndne_f32" ? 26
                           : name == "v_sin_f32"   ? 13
                                                   : 12;
        V r;
        if (op == 12 || op == 13) {
            r = this->trig_turns(a, op == 12);
        } else if (op == 22) {
            r = this->fract(a, 1.0f, f32_from_bits(0x3F7F'FFFF));
        } else {
            r = this->unary(op, a);
        }
        float_w(r);
        return;
    } else if (is({"v_cvt_i32_f32", "v_cvt_u32_f32", "v_cvt_flr_i32_f32", "v_cvt_rpi_i32_f32"})) {
        V a = f(0);
        if (name == "v_cvt_rpi_i32_f32") {
            V h = this->s.m.f32(f32_bits(0.5f));
            a = this->s.m.bin(c, Bin::Add, a, h);
        }
        if (name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32") {
            a = this->unary(27, a);
        }
        // D3D's conversions saturate and give 0 for NaN, as the
        // hardware does (checked on the RX 6700 XT and WARP).
        Cast op = name == "v_cvt_u32_f32" ? Cast::FpToUi : Cast::FpToSi;
        V r = this->s.m.cast(c, op, a, this->s.t.i32);
        int_w(r);
        return;
    } else if (name == "v_cvt_f32_f64") {
        auto [lo, hi] = this->f64_pair(inst, 0);
        V bits = this->f64_to_f32_bits(lo, hi);
        V r = this->as_float(bits);
        float_w(r);
        return;
    } else if (is({"v_cvt_f32_i32", "v_cvt_f32_u32", "v_cvt_f32_ubyte0", "v_cvt_f32_ubyte1", "v_cvt_f32_ubyte2",
                   "v_cvt_f32_ubyte3"})) {
        V a = s(0);
        if (name.starts_with("v_cvt_f32_ubyte")) {
            std::string_view n = name.substr(std::string_view("v_cvt_f32_ubyte").size());
            V sh = this->s.m.i32(8 * std::uint32_t(std::uint8_t(n.at(0) - '0')));
            V ff = this->s.m.i32(0xff);
            a = this->s.m.bin(c, Bin::LShr, a, sh);
            a = this->s.m.bin(c, Bin::And, a, ff);
        }
        Cast op = name == "v_cvt_f32_i32" ? Cast::SiToFp : Cast::UiToFp;
        V r = this->s.m.cast(c, op, a, this->s.t.f32);
        float_w(r);
        return;
    } else if (name == "v_cvt_off_f32_i4") {
        // The low nibble as a signed 4-bit integer, over 16.
        V a = s(0);
        V k28 = this->s.m.i32(28);
        a = this->s.m.bin(c, Bin::Shl, a, k28);
        a = this->s.m.bin(c, Bin::AShr, a, k28);
        V fl = this->s.m.cast(c, Cast::SiToFp, a, this->s.t.f32);
        V k = this->s.m.f32(f32_bits(0.0625f));
        V r = this->s.m.bin(c, Bin::Mul, fl, k);
        float_w(r);
        return;
    } else if (is({"v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16"})) {
        // Each source is f32, or with op_sel_hi the low (op_sel: high)
        // f16 half; neg_hi is the absolute value here.
        std::vector<V> xs;
        for (std::size_t i = 0; i < 3; ++i) {
            const Operand* op = ok_or(src_at(v, i), "a missing operand");
            V xx;
            if (((v.opsel_hi >> i) & 1) != 0) {
                V h0 = this->f16_operand(inst, *op);
                V h = h0;
                if (((v.opsel >> i) & 1) != 0) {
                    V k16 = this->s.m.i32(16);
                    h = this->s.m.bin(c, Bin::LShr, h0, k16);
                }
                xx = this->s.op(c, "legacyF16ToF32", Attr::ReadNone, this->s.t.f32, 131, std::vector<V>{h});
            } else {
                V o = this->operand(inst, *op);
                xx = this->as_float(o);
            }
            if (((v.neg_hi >> i) & 1) != 0) {
                xx = this->unary(6, xx);
            }
            if (((v.neg >> i) & 1) != 0) {
                V z = this->s.m.f32(0x8000'0000u);
                xx = this->s.m.bin(c, Bin::Sub, z, xx);
            }
            xs.push_back(xx);
        }
        V r = this->fmad(xs[0], xs[1], xs[2]);
        if (name == "v_fma_mix_f32") {
            float_w(r);
            return;
        }
        if (v.clamp) {
            r = this->unary(7, r);
        }
        // The f16 result goes to one half; the other half stays.
        auto reg = vector_reg(v.dst);
        if (!reg) fail("a mixed-precision result");
        V old;
        if (auto o = inst.read(Reg::V(*reg))) {
            old = this->i32_of(*o);
        } else {
            old = this->s.m.i32(0);
        }
        V h = this->s.op(c, "legacyF32ToF16", Attr::ReadNone, this->s.t.i32, 130, std::vector<V>{r});
        std::uint32_t keep_bits;
        if (name == "v_fma_mixhi_f16") {
            V k16 = this->s.m.i32(16);
            keep_bits = 0xFFFF;
            h = this->s.m.bin(c, Bin::Shl, h, k16);
        } else {
            keep_bits = 0xFFFF'0000u;
        }
        V keep = this->s.m.i32(keep_bits);
        old = this->s.m.bin(c, Bin::And, old, keep);
        V xx = this->s.m.bin(c, Bin::Or, old, h);
        this->write(inst, Reg::V(*reg), xx, guard);
        return;
    } else if (name == "v_cvt_pkrtz_f16_f32") {
        V a = f(0);
        V bb = f(1);
        V lo = this->f16_toward_zero(a);
        V hi = this->f16_toward_zero(bb);
        V sh = this->s.m.i32(16);
        hi = this->s.m.bin(c, Bin::Shl, hi, sh);
        V r = this->s.m.bin(c, Bin::Or, lo, hi);
        int_w(r);
        return;
    } else if (name == "v_cvt_f16_f32" && v.opsel == 0) {
        // Rounded to nearest even (the MODE default) into the low half.
        V a = f(0);
        V r = this->s.op(c, "legacyF32ToF16", Attr::ReadNone, this->s.t.i32, 130, std::vector<V>{a});
        auto reg = vector_reg(v.dst);
        if (!reg) fail("a 16-bit result outside a VGPR");
        r = this->low_half(inst, *reg, r);
        int_w(r);
        return;
    } else if (name == "v_pack_b32_f16") {
        // Two f16 sources (op_sel: their high halves; abs and neg on
        // the sign bit) into the low and high half.
        std::vector<V> halves;
        for (std::size_t k = 0; k < 2; ++k) {
            const Operand* op = ok_or(src_at(v, k), "a missing operand");
            V xx = this->f16_operand(inst, *op);
            if (((v.opsel >> k) & 1) != 0) {
                V k16 = this->s.m.i32(16);
                xx = this->s.m.bin(c, Bin::LShr, xx, k16);
            }
            std::uint32_t mask_bits = ((v.abs >> k) & 1) != 0 ? 0x7FFF : 0xFFFF;
            V mask = this->s.m.i32(mask_bits);
            xx = this->s.m.bin(c, Bin::And, xx, mask);
            if (((v.neg >> k) & 1) != 0) {
                V sign = this->s.m.i32(0x8000);
                xx = this->s.m.bin(c, Bin::Xor, xx, sign);
            }
            halves.push_back(xx);
        }
        V k16 = this->s.m.i32(16);
        V hi = this->s.m.bin(c, Bin::Shl, halves[1], k16);
        V r = this->s.m.bin(c, Bin::Or, halves[0], hi);
        int_w(r);
        return;
    } else if (name == "v_cvt_f32_f16" && v.opsel == 0) {
        V a = s(0);
        V r = this->s.op(c, "legacyF16ToF32", Attr::ReadNone, this->s.t.f32, 131, std::vector<V>{a});
        float_w(r);
        return;
    } else if (name == "v_cndmask_b32") {
        std::optional<Value> mask;
        if (auto code = scalar_code(src_at(v, 2))) mask = inst.read(Reg::S(*code));
        static constexpr Why UNTRACKED = "a select on an untracked mask";
        std::optional<Bool> tracked;
        if (mask) {
          
            if (auto it = this->x.st->masks.find(*mask); it != this->x.st->masks.end()) tracked = it->second;
        }
        V cond;
        if (tracked) {
            cond = this->boolean(*tracked, c);
        } else if (this->mask_data == MaskData::Console) {
            // A mask the code holds as a number (a constant bit
            // pattern: every other lane, every other pair...):
            // this lane's bit of it. Exact where host lanes are
            // the console's.
            // Each host wave taken as a wave of its own has no
            // console lane numbers to pick the bits by.
            if (this->local) {
                fail("a constant lane mask in a half-wave");
            }
            Value m = ok_or(mask, UNTRACKED);
            V lo;
            try {
                lo = this->i32_of(m);
            } catch (const WhyFail&) {
                fail(UNTRACKED);
            }
            // wave64: the register after it holds lanes 32-63.
            std::optional<Value> high;
            if (auto code = scalar_code(src_at(v, 2)); code && this->x.native->wave64) {
                high = ok_or(inst.read(Reg::S(std::uint8_t(*code + 1))), UNTRACKED);
            }
            std::optional<V> hi;
            if (high) {
                try {
                    hi = this->i32_of(*high);
                } catch (const WhyFail&) {
                    fail(UNTRACKED);
                }
            }
            this->cross_lane(false);
            cond = this->lane_bit(c, lo, hi);
        } else {
            fail(UNTRACKED);
        }
        if (v.abs != 0 || v.neg != 0) {
            V a = f(0);
            V bb = f(1);
            V r = this->s.m.select(c, cond, bb, a);
            float_w(r);
            return;
        }
        V a = s(0);
        V bb = s(1);
        V r = this->s.m.select(c, cond, bb, a);
        int_w(r);
        return;
    } else if (is({"v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32", "v_add_co_ci_u32", "v_sub_co_ci_u32",
                   "v_subrev_co_ci_u32"})) {
        // The carry (borrow) out is a lane mask: this lane's bit.
        V a = s(0);
        V bb = s(1);
        if (name.find("subrev") != std::string_view::npos) {
            std::swap(a, bb);
        }
        std::optional<V> cin;
        if (name.ends_with("_ci_u32")) {
            std::optional<Value> m;
            if (auto code = scalar_code(src_at(v, 2))) m = inst.read(Reg::S(*code));
            std::optional<Bool> mb;
            if (m) {
              
                if (auto it = this->x.st->masks.find(*m); it != this->x.st->masks.end()) mb = it->second;
            }
            Bool bm = ok_or(mb, "a carry-in from an untracked mask");
            cin = this->boolean(bm, c);
        }
        auto [r, carry] = this->add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
        int_w(r);
        if (auto code = scalar_code(v.carry)) {
            if (auto w = inst.write(Reg::S(*code))) {
                this->lane_bits.insert_or_assign(*w, carry);
            }
        }
        for (const auto& [_, w] : inst.writes) {
            this->vals.try_emplace(w, Val::Missing("a lane mask read as data"));
        }
        return;
    } else if (name == "v_permlane16_b32" || name == "v_permlanex16_b32") {
        // RDNA2 ISA 12.12: each nibble of {src2, src1} selects a lane
        // in this row of 16, or the other row of the same 32-lane half.
        // OP_SEL[0:1] are fetch-inactive and bound-control, not half selects.
        if (v.abs != 0 || v.neg != 0 || (v.opsel & ~3u) != 0) {
            fail("permlane input modifiers");
        }
        this->cross_lane(false);
        bool fetch_inactive = (v.opsel & 1) != 0;
        if (fetch_inactive) {
            auto all = all_lane_blocks(*this->x.st);
            if (!(all && (*all).at(this->cur_ir))) {
                // DXIL cannot read the last VGPR value of a host invocation
                // that has left through a lane branch. A guard keeps it here.
                fail("a permlane fetch from lanes outside the host branch");
            }
        }
        V xx = s(0);
        V lo = s(1);
        V hi = s(2);
        V lane = this->lane_index();
        V k7 = this->s.m.i32(7);
        V k8 = this->s.m.i32(8);
        V k15 = this->s.m.i32(15);
        V k16 = this->s.m.i32(16);
        V k31 = this->s.m.i32(31);
        V k32 = this->s.m.i32(32);
        V k4 = this->s.m.i32(4);
        V one = this->s.m.i32(1);
        V zero = this->s.m.i32(0);
        V pos = this->s.m.bin(c, Bin::And, lane, k15);
        V lower = this->s.m.cmp(c, pred::IULT, pos, k8);
        V packed = this->s.m.select(c, lower, lo, hi);
        V shift = this->s.m.bin(c, Bin::And, pos, k7);
        shift = this->s.m.bin(c, Bin::Mul, shift, k4);
        V pick = this->s.m.bin(c, Bin::LShr, packed, shift);
        pick = this->s.m.bin(c, Bin::And, pick, k15);
        V not15 = this->s.m.i32(~15u);
        V row = this->s.m.bin(c, Bin::And, lane, not15);
        if (name == "v_permlanex16_b32") {
            row = this->s.m.bin(c, Bin::Xor, row, k16);
        }
        V src = this->s.m.bin(c, Bin::Or, row, pick);
        V r;
        if (fetch_inactive) {
            r = this->read_lane(xx, src);
        } else {
            // Ballot the console EXEC guard as well as the host's active
            // invocations. Never shuffle from an inactive host lane.
            V enabled = guard ? *guard : this->s.m.bool_(true);
            Ty i32t = this->s.t.i32;
            Ty four = this->s.m.named_struct("dx.types.fouri32", std::vector<Ty>(4, i32t));
            V mask = this->s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, std::vector<V>{enabled});
            V mlo = this->s.m.extract(c, mask, 0);
            V mhi = this->s.m.extract(c, mask, 1);
            V lower2 = this->s.m.cmp(c, pred::IULT, src, k32);
            V word = this->s.m.select(c, lower2, mlo, mhi);
            V bit = this->s.m.bin(c, Bin::And, src, k31);
            bit = this->s.m.bin(c, Bin::LShr, word, bit);
            bit = this->s.m.bin(c, Bin::And, bit, one);
            V active = this->s.m.cmp(c, pred::NE, bit, zero);
            V safe = this->s.m.select(c, active, src, lane);
            V read = this->read_lane(xx, safe);
            V fallback;
            if ((v.opsel & 2) != 0) {
                fallback = zero;
            } else {
                auto reg = vector_reg(v.dst);
                if (!reg) fail("a permlane result outside a VGPR");
                fallback = this->i32_of(ok_or(inst.read(Reg::V(*reg)), "a permlane old value"));
            }
            r = this->s.m.select(c, active, read, fallback);
        }
        int_w(r);
        return;
    } else if (name == "v_readlane_b32") {
        V a = s(0);
        V lane = s(1);
        // A computed lane in a pixel or vertex program is a host lane
        // (from a mask or `v_mbcnt`); a constant one is the console's.
        if (this->mask_data == MaskData::Host && !(lane.kind == V::Kind::Const)) {
            this->s.flags |= flags::WAVE_OPS;
        } else {
            this->cross_lane(false);
        }
        if (this->exchange) {
            std::uint32_t slot = std::get<0>(*this->exchange);
            if (this->local && this->local_any) {
                // this host wave's lane of that number (lanes 32..63 are 0..31 here)
                V k31 = this->s.m.i32(31);
                V here = this->s.m.bin(this->cur, Bin::And, lane, k31);
                V r = this->read_lane(a, here);
                int_w(r);
                return;
            }
            if (this->local) {
                fail("a lane read in a half-wave");
            }
            if (std::get<1>(*this->exchange)) {
                // a group of 32 or fewer: the lane is in this host wave
                V r = this->read_lane(a, lane);
                int_w(r);
                return;
            }
            // the lane that holds it may be in the other host wave: it gives it through LDS
            if (!this->all_lanes(this->cur_ir)) {
                need(this->cur_ir);
                fail("a lane read where not every lane runs");
            }
            V cl = this->console_lane();
            V eq = this->s.m.cmp(this->cur, pred::EQ, cl, lane);
            V k = this->exchange_pair(slot);
            auto at = this->begin_guarded(eq);
            V lds = this->s.lds();
            V ptr = this->s.m.element(this->cur, lds, k);
            this->s.m.store(this->cur, ptr, a);
            this->end_guarded(at, {});
            std::size_t c2 = this->cur;
            this->group_sync(c2);
            V ptr2 = this->s.m.element(c2, lds, k);
            V r = this->s.m.load(c2, ptr2);
            this->group_sync(c2);
            int_w(r);
            return;
        }
        V r = this->read_lane(a, lane);
        int_w(r);
        return;
    } else if (name == "v_writelane_b32") {
        // The lane given takes the scalar; the others keep their value.
        V a = s(0);
        V lane = s(1);
        auto reg = vector_reg(v.dst);
        if (!reg) fail("a lane write");
        V old;
        if (auto o = inst.read(Reg::V(*reg))) {
            old = this->i32_of(*o);
        } else {
            old = this->s.m.i32(0);
        }
        this->cross_lane(false);
        if (this->local && !this->local_any) {
            fail("a lane write in a half-wave");
        }
        V me = this->console_lane();
        V eq = this->s.m.cmp(c, pred::EQ, me, lane);
        V r = this->s.m.select(c, eq, a, old);
        int_w(r);
        return;
    } else if (name == "v_readfirstlane_b32" && !inst.writes.empty() && [&] {
                 
                   const auto& reads = this->x.res->waterfall.reads;
                   return std::find(reads.begin(), reads.end(), inst.writes.front().second) != reads.end();
               }()) {
        // A waterfall loop's read of a descriptor dword: the lane's own.
        V a = s(0);
        int_w(a);
        return;
    } else if (name == "v_readfirstlane_b32" && [&] {
                   auto reg = vector_reg(v.src.empty() ? nullptr : &v.src.front());
                   if (!reg) return false;
                   auto xr = inst.read(Reg::V(*reg));
                   return xr && this->own_bits.alike.count(*xr) != 0;
               }()) {
        // The bits a loop over a wave OR has still to visit (`own_bits`): every lane of
        // the host wave holds the same, so the first lane's is the lane's own.
        V a = s(0);
        int_w(a);
        return;
    } else if (name == "v_readfirstlane_b32") {
        V a = s(0);
        auto [mine, any] = this->first_lane(a, guard);
        // two host waves that exchange: the console's first lane may be in the other
        // one, unless every lane holds the same value
        bool alike = true;
        if (auto reg = vector_reg(v.src.empty() ? nullptr : &v.src.front())) {
            auto xr = inst.read(Reg::V(*reg));
            alike = xr && this->uniform.count(*xr) != 0;
        }
        bool halves = this->exchange && !std::get<1>(*this->exchange) && !this->local;
        V r;
        if (halves && !alike) {
            if (!this->all_lanes(this->cur_ir)) {
                // shortcut: `X2W_SHADERCFG_FIRSTLANE_OWN=1` takes the host wave's own
                // first lane, exact when the value is the same in every lane of
                // the console wave (the ray-tracing builder's scans read a sum of
                // scalars this way, inside a loop whose exit is per console wave).
                static const bool OWN = [] {
                    const char* e = std::getenv("X2W_SHADERCFG_FIRSTLANE_OWN");
                    return e && std::string_view(e) == "1";
                }();
                if (OWN) {
                    int_w(mine);
                    return;
                }
                need(this->cur_ir);
                fail("a first-lane read where not every lane runs");
            }
            std::size_t cc = this->cur;
            if (!guard && (shader_opt() & opt::FIRST) != 0) {
                // Every lane is in EXEC: the console's first lane is
                // lane 0, which the lower half of every console wave
                // has. No half needs to ask the other whether it has
                // a lane (an exchange of the constant 1: two group
                // barriers), and the lower half's value is the answer.
                auto [lo, _] = this->exchange_halves(cc, mine);
                r = lo;
            } else {
                Ty i32t = this->s.t.i32;
                V has = this->s.m.cast(cc, Cast::ZExt, any, i32t);
                auto [lo_has, hi_has] = this->exchange_halves(cc, has);
                auto [lo, hi] = this->exchange_halves(cc, mine);
                V z = this->s.m.i32(0);
                V in_lo = this->s.m.cmp(cc, pred::NE, lo_has, z);
                V in_hi = this->s.m.cmp(cc, pred::NE, hi_has, z);
                // neither half has a lane in EXEC: the console reads lane 0
                V hi2 = this->s.m.select(cc, in_hi, hi, lo);
                r = this->s.m.select(cc, in_lo, lo, hi2);
            }
        } else {
            r = mine;
        }
        int_w(r);
        return;
    } else {
        // `name` is static (the opcode table), NUL-terminated.
        V r = this->integer(inst, name.data());
        int_w(r);
        return;
    }
}

}  // namespace x2w::vs::lower
