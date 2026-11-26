// LDS, instruction dispatch, and scalar ALU lowering. See lower_impl.h and GMR/vs_port/lower/PORTING.md.
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "x2w_shader/emit/lower_impl.h"
#include "x2w_shader/spill.h"

namespace x2w::vs::lower {

namespace {

// removes every trailing repetition of `pat`.
std::string_view trim_end_matches(std::string_view s, std::string_view pat) {
    while (!pat.empty() && s.ends_with(pat)) s.remove_suffix(pat.size());
    return s;
}

// every non-overlapping occurrence, left to right.
std::string replace_all(std::string_view s, std::string_view from, std::string_view to) {
    std::string out;
    std::size_t at = 0;
    while (true) {
        std::size_t k = s.find(from, at);
        if (k == std::string_view::npos) break;
        out.append(s.substr(at, k - at));
        out.append(to);
        at = k + from.size();
    }
    out.append(s.substr(at));
    return out;
}

bool contains(std::string_view s, std::string_view x) { return s.find(x) != std::string_view::npos; }

// the text after the last '_' (all of it without one).
std::string_view last_part(std::string_view n) {
    std::size_t k = n.rfind('_');
    return k == std::string_view::npos ? n : n.substr(k + 1);
}

}  // namespace

void Lower::ds(const Inst& inst, std::size_t b, std::size_t i) {
    const auto& f = std::get<gfx10::Ds>(inst.fields);
    const auto& dst = f.dst;
    const auto& addr = f.addr;
    const auto& data = f.data;
    const std::uint8_t offset0 = f.offset0;
    const std::uint8_t offset1 = f.offset1;
    if (f.gds) {
        fail("GDS");
    }
    const std::string_view name = inst.name();
    const std::optional<V> guard = this->guard(b, i);
    const std::size_t c = this->cur;
    const std::uint32_t offset = std::uint32_t(offset0) | (std::uint32_t(offset1) << 8);
    std::optional<std::uint8_t> dst_reg;
    if (dst && dst->kind == Operand::Kind::Vector) dst_reg = dst->reg;
    if (name == "ds_swizzle_b32") {
        // Bit 15 set: each lane of a quad picks a lane of it; else lanes
        // of each group of 32 combine their index with and/or/xor masks.
        const bool quad = (offset & 0x8000) != 0 ||
                          ((offset & 0x1C) == 0x1C && (offset & 0x380) == 0 && ((offset >> 10) & 0x1F) < 4);
        // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
        // console's, so quad-local swizzles read through quad operations.
        const bool mesh_quad = this->s.mesh.has_value() && quad;
        if (!mesh_quad) {
            this->cross_lane(quad);
        }
        // The source sits in the address field.
        const Operand src_op = ok_or(addr, "a swizzle source");
        const V x = this->vgprs(inst, src_op, 1).at(0);
        const V lane = this->lane_index();
        auto k = [](Lower& l, std::uint32_t v) { return l.s.m.i32(v); };
        V src{};
        if ((offset & 0x8000) != 0) {
            V three = k(*this, 3);
            V one = k(*this, 1);
            V sel = this->s.m.bin(c, Bin::And, lane, three);
            V sh = this->s.m.bin(c, Bin::Shl, sel, one);
            V pattern = k(*this, offset & 0xFF);
            V pick = this->s.m.bin(c, Bin::LShr, pattern, sh);
            pick = this->s.m.bin(c, Bin::And, pick, three);
            V not3 = k(*this, ~3u);
            V base = this->s.m.bin(c, Bin::And, lane, not3);
            src = this->s.m.bin(c, Bin::Or, base, pick);
        } else {
            V and_ = k(*this, (offset & 0x1F) | ~0x1Fu);
            V or_ = k(*this, (offset >> 5) & 0x1F);
            V xor_ = k(*this, (offset >> 10) & 0x1F);
            V t = this->s.m.bin(c, Bin::And, lane, and_);
            t = this->s.m.bin(c, Bin::Or, t, or_);
            src = this->s.m.bin(c, Bin::Xor, t, xor_);
        }
        V r{};
        if (mesh_quad) {
            r = this->quad_pick(x, src);
        } else if (this->helpers_read(quad)) {
            std::array<std::uint8_t, 4> lanes{};
            for (std::uint32_t q = 0; q < 4; ++q) {
                if ((offset & 0x8000) != 0) {
                    lanes[q] = std::uint8_t((offset >> (2 * q)) & 3);
                } else {
                    lanes[q] = std::uint8_t((((q & offset) | ((offset >> 5) & 3)) ^ ((offset >> 10) & 3))) & 3;
                }
            }
            r = this->quad_read(x, lanes);
        } else {
            r = this->read_lane(x, src);
        }
        const std::uint8_t reg = ok_or(dst_reg, "a swizzle result");
        this->write(inst, Reg::V(reg), r, guard);
        return;
    }
    if (name == "ds_bpermute_b32") {
        // Backward permute: no LDS is touched. Each lane takes the data
        // register of the lane its address names, (address + offset) / 4
        // within the wave; a source lane outside EXEC gives 0. (the title's
        // checkerboard shadow-mask generators read their neighbours' samples
        // with it; without them every shadowed light was multiplied by an
        // empty mask.)
        // shortcut: a source lane that is not at this instruction on the
        // host (it left through a branch, not a guard) is read as the
        // host gives it; the console reads 0.
        this->cross_lane(false);
        if (this->exchange && !std::get<1>(*this->exchange)) {
            // a console wave of 64 run as two host waves: the lane named
            // may be in the other one
            fail("a lane permute across two host waves");
        }
        const Operand at_op = ok_or(addr, "a permute address");
        V at = this->vgprs(inst, at_op, 1).at(0);
        const Operand* x_op = ok_or(data.empty() ? nullptr : &data[0], "a permute source");
        V x = this->vgprs(inst, *x_op, 1).at(0);
        V o = this->s.m.i32(offset);
        at = this->s.m.bin(c, Bin::Add, at, o);
        V two = this->s.m.i32(2);
        V src = this->s.m.bin(c, Bin::LShr, at, two);
        V last = this->s.m.i32(this->wave() - 1);
        src = this->s.m.bin(c, Bin::And, src, last);
        if (guard) {
            V z = this->s.m.i32(0);
            x = this->s.m.select(c, *guard, x, z);
        }
        V r = this->read_lane(x, src);
        const std::uint8_t reg = ok_or(dst_reg, "a permute result");
        this->write(inst, Reg::V(reg), r, guard);
        return;
    }
    if (this->s.lds_dwords == 0) {
        fail("LDS the program does not allocate");
    }
    const Operand a_op = ok_or(addr, "an LDS address");
    const V a = this->vgprs(inst, a_op, 1).at(0);
    // shortcut: LDS addresses are taken as dword-aligned and in bounds
    // (out-of-bounds console accesses read 0 and drop writes).
    auto dword = [a](Lower& l, std::uint32_t bytes) -> V {
        const std::size_t c = l.cur;
        V o = l.s.m.i32(bytes);
        V at = l.s.m.bin(c, Bin::Add, a, o);
        V two = l.s.m.i32(2);
        V index = l.s.m.bin(c, Bin::LShr, at, two);
        V lds = l.s.lds();
        return l.s.m.element(c, lds, index);
    };
    auto width = [](std::string_view n) -> std::uint32_t {
        std::string_view last = last_part(n);
        if (last == "b64") return 2;
        if (last == "b96") return 3;
        if (last == "b128") return 4;
        return 1;
    };
    // (byte offset, dwords) per element; read2/write2 scale their two
    // offsets by the element size (x64 for st64).
    auto pair_of = [&](std::string_view n) -> std::vector<std::pair<std::uint32_t, std::uint32_t>> {
        const std::uint32_t w = width(n);
        const std::uint32_t scale = 4 * w * (contains(n, "st64") ? 64u : 1u);
        return {
            {std::uint32_t(offset0) * scale, w},
            {std::uint32_t(offset1) * scale, w},
        };
    };
    // Bytes and halves: the dword holding them, and their bit position
    // in it (naturally aligned, so they never straddle dwords).
    // `_d16` reads fill a register's low half (`_d16_hi` its high half)
    // and keep the other; `_d16_hi` writes take the data's high half.
    const std::string_view base = trim_end_matches(trim_end_matches(name, "_hi"), "_d16");
    std::optional<std::uint32_t> sub;
    if (base == "ds_read_u8" || base == "ds_read_i8" || base == "ds_write_b8") {
        sub = 8u;
    } else if (base == "ds_read_u16" || base == "ds_read_i16" || base == "ds_write_b16") {
        sub = 16u;
    }
    if (sub && !(base == name || (name.ends_with("_d16") && base.starts_with("ds_read")) ||
                 name.ends_with("_d16_hi"))) {
        sub.reset();
    }
    if (sub) {
        const std::uint32_t bits = *sub;
        V p = dword(*this, offset);
        V o = this->s.m.i32(offset);
        V at = this->s.m.bin(c, Bin::Add, a, o);
        V three = this->s.m.i32(3);
        V byte = this->s.m.bin(c, Bin::And, at, three);
        V sh = this->s.m.bin(c, Bin::Shl, byte, three);
        V mask = this->s.m.i32((1u << bits) - 1);
        if (name.starts_with("ds_read")) {
            const std::uint8_t reg = ok_or(dst_reg, "an LDS read result");
            V w = this->s.m.load(c, p);
            V x = this->s.m.bin(c, Bin::LShr, w, sh);
            if (contains(base, "_i")) {
                V k = this->s.m.i32(32 - bits);
                x = this->s.m.bin(c, Bin::Shl, x, k);
                x = this->s.m.bin(c, Bin::AShr, x, k);
            } else {
                x = this->s.m.bin(c, Bin::And, x, mask);
            }
            if (base != name) {
                V old{};
                if (auto v = inst.read(Reg::V(reg))) {
                    old = this->i32_of(*v);
                } else {
                    old = this->s.m.i32(0);
                }
                V half = this->s.m.i32(0xFFFF);
                x = this->s.m.bin(c, Bin::And, x, half);
                V keep{};
                if (name.ends_with("_hi")) {
                    V k16 = this->s.m.i32(16);
                    x = this->s.m.bin(c, Bin::Shl, x, k16);
                    keep = half;
                } else {
                    keep = this->s.m.i32(0xFFFF'0000u);
                }
                old = this->s.m.bin(c, Bin::And, old, keep);
                x = this->s.m.bin(c, Bin::Or, old, x);
            }
            this->write(inst, Reg::V(reg), x, guard);
        } else {
            // Clear, then set: two atomics, so lanes writing other
            // bytes of the dword keep theirs.
            const Operand* d = ok_or(data.empty() ? nullptr : &data[0], "an LDS write");
            V x = this->vgprs(inst, *d, 1).at(0);
            if (name != base) {
                // The data's high half.
                V k16 = this->s.m.i32(16);
                x = this->s.m.bin(c, Bin::LShr, x, k16);
            }
            x = this->s.m.bin(c, Bin::And, x, mask);
            x = this->s.m.bin(c, Bin::Shl, x, sh);
            V m = this->s.m.bin(c, Bin::Shl, mask, sh);
            V ones = this->s.m.i32(0xFFFF'FFFFu);
            V keep = this->s.m.bin(c, Bin::Xor, m, ones);
            auto at_g = this->begin_guarded(guard);
            this->s.m.atomic(this->cur, Rmw::And, p, keep);
            this->s.m.atomic(this->cur, Rmw::Or, p, x);
            this->end_guarded(at_g, {});
        }
        return;
    }
    if (name.starts_with("ds_read")) {
        const std::string_view rest = name.substr(std::string_view("ds_read").size());
        std::vector<std::pair<std::uint32_t, std::uint32_t>> elements;
        if (rest.starts_with('2')) {
            elements = pair_of(name);
        } else if (rest == "_b32" || rest == "_b64" || rest == "_b96" || rest == "_b128") {
            elements = {{offset, width(name)}};
        } else {
            fail("an LDS read of bytes or halves");
        }
        const std::uint8_t reg = ok_or(dst_reg, "an LDS read result");
        std::uint8_t n = 0;
        for (auto [off, w] : elements) {
            for (std::uint32_t k = 0; k < w; ++k) {
                V p = dword(*this, off + 4 * k);
                V x = this->s.m.load(this->cur, p);
                this->write(inst, Reg::V(std::uint8_t(reg + n)), x, guard);
                n += 1;
            }
        }
        return;
    }
    if (name.starts_with("ds_write")) {
        const std::string_view rest = name.substr(std::string_view("ds_write").size());
        std::vector<std::pair<std::uint32_t, std::uint32_t>> elements;
        if (rest.starts_with('2')) {
            elements = pair_of(name);
        } else if (rest == "_b32" || rest == "_b64" || rest == "_b96" || rest == "_b128") {
            elements = {{offset, width(name)}};
        } else {
            fail("an LDS write of bytes or halves");
        }
        std::vector<std::pair<std::uint32_t, V>> stores;
        for (std::size_t e = 0; e < elements.size(); ++e) {
            auto [off, w] = elements[e];
            const Operand* d = ok_or(e < data.size() ? &data[e] : nullptr, "an LDS write");
            std::vector<V> vals = this->vgprs(inst, *d, std::uint8_t(w));
            for (std::size_t k = 0; k < vals.size(); ++k) {
                stores.emplace_back(off + 4 * std::uint32_t(k), vals[k]);
            }
        }
        auto at_g = this->begin_guarded(guard);
        for (auto [off, x] : stores) {
            V p = dword(*this, off);
            this->s.m.store(this->cur, p, x);
        }
        this->end_guarded(at_g, {});
        return;
    }
    // Atomics: the old value returns with `_rtn`.
    if (!name.starts_with("ds_")) fail("an LDS operation");
    const std::string op = replace_all(name.substr(3), "_rtn", "");
    std::optional<Rmw> rmw;
    if (op == "add_u32") {
        rmw = Rmw::Add;
    } else if (op == "sub_u32") {
        rmw = Rmw::Sub;
    } else if (op == "and_b32") {
        rmw = Rmw::And;
    } else if (op == "or_b32") {
        rmw = Rmw::Or;
    } else if (op == "xor_b32") {
        rmw = Rmw::Xor;
    } else if (op == "min_i32") {
        rmw = Rmw::Min;
    } else if (op == "max_i32") {
        rmw = Rmw::Max;
    } else if (op == "min_u32") {
        rmw = Rmw::UMin;
    } else if (op == "max_u32") {
        rmw = Rmw::UMax;
    } else if (op == "wrxchg_b32") {
        rmw = Rmw::Xchg;
    } else if (op == "cmpst_b32" || op == "max_f32" || op == "min_f32") {
        rmw.reset();
    } else {
        fail("an LDS operation");
    }
    const Operand* d0 = ok_or(data.empty() ? nullptr : &data[0], "an LDS atomic");
    const V v0 = this->vgprs(inst, *d0, 1).at(0);
    std::optional<V> v1;
    if (data.size() > 1) {
        v1 = this->vgprs(inst, data[1], 1).at(0);
    }
    std::optional<std::pair<std::uint8_t, Value>> old_reg;
    if (dst_reg) {
        if (auto v = inst.read(Reg::V(*dst_reg))) old_reg = std::make_pair(*dst_reg, *v);
    }
    V outside{};
    if (old_reg) {
        // both are evaluated, in this order.
        std::optional<V> got = ok([&] { return this->i32_of(old_reg->second); });
        V z = this->s.m.i32(0);
        outside = got ? *got : z;
    } else {
        outside = this->s.m.i32(0);
    }
    const V p = dword(*this, offset);
    std::optional<bool> fminmax;
    if (op == "max_f32") {
        fminmax = true;
    } else if (op == "min_f32") {
        fminmax = false;
    }
    if (fminmax) {
        const bool max = *fminmax;
        auto at_g = this->begin_guarded(guard);
        V r = this->lds_float_min_max(p, v0, max);
        std::vector<std::pair<V, V>> vals{{r, outside}};
        r = this->end_guarded(at_g, vals).at(0);
        if (dst_reg) {
            this->write(inst, Reg::V(*dst_reg), r, std::nullopt);
        }
        return;
    }
    auto at_g = this->begin_guarded(guard);
    V r{};
    if (rmw) {
        r = this->s.m.atomic(this->cur, *rmw, p, v0);
    } else {
        // Compare with data0, store data1.
        const std::size_t cur = this->cur;
        const V nv = ok_or(v1, "an LDS compare");
        r = this->s.m.cmpxchg(cur, p, v0, nv);
    }
    std::vector<std::pair<V, V>> vals{{r, outside}};
    r = this->end_guarded(at_g, vals).at(0);
    if (dst_reg) {
        this->write(inst, Reg::V(*dst_reg), r, std::nullopt);
    }
}

// `ds_max_f32` (`max`) or `ds_min_f32` on the LDS dword at `p` with the float bits `data`; the old
// bits return. DXIL has no float atomics on group-shared memory, so this is a compare-exchange loop
// around lds_float_pick, which gives what RDNA2 writes (measured on gfx1036): an sNaN, data first,
// then memory, is written quieted; otherwise a NaN loses to a number (two quiet NaNs give the
// data); numbers compare as floats with -0 below +0.
// shortcut: f32 denormals compare as kept; with the console's MODE flushing them they would order
// as +-0 first (MODE is not modelled).
V Lower::lds_float_min_max(V p, V data, bool max) {
    const std::size_t pre = this->cur;
    V init = this->s.m.load(pre, p);
    const std::size_t lp = this->s.m.new_block();
    const std::size_t exit = this->s.m.new_block();
    this->s.m.terminate(pre, Term::br(lp));
    const Ty i32t = this->s.t.i32;
    V old = this->s.m.phi(lp, i32t);
    this->cur = lp;
    V nw = this->lds_float_pick(old, data, max);
    V prev = this->s.m.cmpxchg(lp, p, old, nw);
    V done = this->s.m.cmp(lp, pred::EQ, prev, old);
    this->s.m.set_phi(old, std::vector<std::pair<V, std::size_t>>{{init, pre}, {prev, lp}});
    this->s.m.terminate(lp, Term::cond_br(done, exit, lp));
    this->cur = exit;
    return old;
}

// What `ds_max_f32` (`max`) or `ds_min_f32` writes over the dword `mem` with `data` (float bits).
V Lower::lds_float_pick(V mem, V data, bool max) {
    const std::size_t c = this->cur;
    auto k = [](Lower& l, std::uint32_t v) { return l.s.m.i32(v); };
    const V abs_mask = k(*this, 0x7FFF'FFFFu);
    const V inf = k(*this, 0x7F80'0000u);
    const V quiet = k(*this, 0x40'0000u);
    const V zero = k(*this, 0);
    auto nan = [&](Lower& l, V x) {
        V m = l.s.m.bin(c, Bin::And, x, abs_mask);
        return l.s.m.cmp(c, pred::IUGT, m, inf);
    };
    auto snan = [&](Lower& l, V x) {
        V n = nan(l, x);
        V q = l.s.m.bin(c, Bin::And, x, quiet);
        V sig = l.s.m.cmp(c, pred::EQ, q, zero);
        return l.s.m.bin(c, Bin::And, n, sig);
    };
    // A total order of the numbers as signed integers, -0 below +0:
    // negative floats' magnitude bits reversed.
    auto key = [&](Lower& l, V x) {
        V k31 = l.s.m.i32(31);
        V sign = l.s.m.bin(c, Bin::AShr, x, k31);
        V flip = l.s.m.bin(c, Bin::And, sign, abs_mask);
        return l.s.m.bin(c, Bin::Xor, x, flip);
    };
    V dk = key(*this, data);
    V mk = key(*this, mem);
    V take = this->s.m.cmp(c, max ? pred::SGT : pred::SLT, dk, mk);
    V r = this->s.m.select(c, take, data, mem);
    V dn = nan(*this, data);
    V mn = nan(*this, mem);
    // One NaN: the number; two quiet NaNs: the data.
    r = this->s.m.select(c, mn, data, r);
    V both = this->s.m.bin(c, Bin::And, dn, mn);
    V only_data{};
    {
        V t = this->s.m.bool_(true);
        V not_both = this->s.m.bin(c, Bin::Xor, both, t);
        only_data = this->s.m.bin(c, Bin::And, dn, not_both);
    }
    r = this->s.m.select(c, only_data, mem, r);
    // Signalling NaNs win, quieted: the data's first.
    V qm = this->s.m.bin(c, Bin::Or, mem, quiet);
    V sm = snan(*this, mem);
    r = this->s.m.select(c, sm, qm, r);
    V qd = this->s.m.bin(c, Bin::Or, data, quiet);
    V sd = snan(*this, data);
    return this->s.m.select(c, sd, qd, r);
}

// ---- instructions ----

// (Result<(), Stop>)
void Lower::inst(std::size_t b, std::size_t i) {
    const Inst& inst = this->x.p->blocks.at(b).insts.at(i);
    const std::size_t offset = inst.machine.offset;
    const std::string_view name = inst.name();
    this->local_ok = name.starts_with("s_bcnt1_i32_") || name.starts_with("s_ff1_i32_") ||
                     name.starts_with("v_mbcnt_");
    switch (this->mask_data) {
    case MaskData::Console:
        this->mask_ok = true;
        break;
    case MaskData::Host:
        this->mask_ok = name.starts_with("s_bcnt1_i32_") || name.starts_with("s_ff1_i32_") ||
                        name.starts_with("v_mbcnt_") || name == "s_cmp_eq_u32" || name == "s_cmp_lg_u32" ||
                        name == "s_cmp_eq_u64" || name == "s_cmp_lg_u64";
        break;
    case MaskData::None:
        // Mesh: only "do all lanes hold the first lane's value?" (`uniform_test`), which
        // each host wave answers for itself.
        this->mask_ok = this->uniform_test(inst);
        break;
    }
    const bool is_ds = std::holds_alternative<gfx10::Ds>(inst.fields);
    const bool effect = std::holds_alternative<gfx10::Exp>(inst.fields) ||
                        (std::holds_alternative<gfx10::Flat>(inst.fields) && !name.starts_with("global_load_")) ||
                        contains(name, "store") || contains(name, "atomic") || name == "s_barrier" ||
                        (is_ds && !(name.starts_with("ds_read") || name.starts_with("ds_swizzle") ||
                                    contains(name, "permute")));
    if (this->split_lds && is_ds && !(name.starts_with("ds_swizzle") || contains(name, "permute")) &&
        (!this->lds_meet || this->lds_meet->first.at(b))) {
        // the two host waves of a console wave meet before each LDS access: what the other
        // half wrote is visible, and what it still reads is not overwritten
        V mode = this->s.m.i32(meeting_mode());
        const Ty void_t = this->s.t.void_;
        this->s.op(this->cur, "barrier", Attr::NoDuplicate, void_t, 80, std::vector<V>{mode});
    }
    // The structure builder uses atomics to hand work from one thread to the next as it builds a tree from its
    // leaves. The next thread must see writes preceding the previous thread's atomic, including writes ordered
    // by console waits.
    const bool ordered_ = (name.starts_with("buffer_atomic") || name.starts_with("global_atomic")) &&
                          spill::is_spilled(*this->x.native) && emit::builder_fences();
    if (ordered_) {
        this->memory_fence();
    }
    std::optional<Why> err;
    try {
        const Fields& fields = inst.fields;
        if (std::holds_alternative<gfx10::Sop>(fields)) {
            this->sop(inst);
        } else if (std::holds_alternative<gfx10::Sopp>(fields) && name == "s_barrier" &&
                   this->no_barrier.contains(b)) {
        } else if (std::holds_alternative<gfx10::Sopp>(fields) && name == "s_barrier") {
            // Thread-group sync with a shared-memory fence, and a UAV fence when the program
            // writes UAVs (the waits before an s_barrier make those writes visible on the console).
            std::uint32_t mode = this->writes_uavs ? (1 | 2 | 8) : (1 | 8);
            // Without the wait for the group (bit 0) where not every lane arrives, or where the
            // switch says so.
            bool waits = false;
            switch (emit::group_sync()) {
            case 0:
                waits = false;
                break;
            case 1:
                waits = !this->fence_only.contains(b);
                break;
            default:
                waits = true;
                break;
            }
            if (!waits) {
                mode &= ~1u;
            }
            V mode_v = this->s.m.i32(mode);
            const Ty void_t = this->s.t.void_;
            this->s.op(this->cur, "barrier", Attr::NoDuplicate, void_t, 80, std::vector<V>{mode_v});
        } else if (std::holds_alternative<gfx10::Sopp>(fields)) {
        } else if (std::holds_alternative<gfx10::Smem>(fields)) {
            this->smem(inst, Site{b, i});
        } else if (std::holds_alternative<gfx10::Valu>(fields) && name.starts_with("v_interp_")) {
            this->interp(inst, Site{b, i});
        } else if (std::holds_alternative<gfx10::Valu>(fields)) {
            this->valu(inst, b, i);
        } else if (std::holds_alternative<gfx10::Vintrp>(fields)) {
            this->interp(inst, Site{b, i});
        } else if (std::holds_alternative<gfx10::Image>(fields) && name.starts_with("image_bvh")) {
            // The ray and node test. A failure is a refusal at once: an instruction without a
            // side effect otherwise only stops the program where its results are needed, and
            // two trace programs came out without their test that way.
            std::optional<Why> bvh_err;
            try {
                this->bvh(inst, b, i);
            } catch (const WhyFail& e) {
                bvh_err = e.why;
            }
            if (bvh_err) unsupported(offset, *bvh_err);
            return;
        } else if (std::holds_alternative<gfx10::Image>(fields)) {
            this->image(inst, b, i);
        } else if (std::holds_alternative<gfx10::Buffer>(fields)) {
            this->buffer(inst, b, i);
        } else if (std::holds_alternative<gfx10::Exp>(fields)) {
            this->export_(b, i);
        } else if (is_ds) {
            this->ds(inst, b, i);
        } else if (std::holds_alternative<gfx10::Flat>(fields) && name.starts_with("global_")) {
            this->flat(inst, b, i);
        } else {
            fail("flat memory");
        }
    } catch (const WhyFail& e) {
        err = e.why;
    }
    if (ordered_ && !err) {
        this->memory_fence();
    }
    if (!err) return;
    if (effect) unsupported(offset, *err);
    for (const auto& w : inst.writes) {
        this->vals.insert_or_assign(w.second, Val::Missing(*err));
    }
}

// Writes to buffers made before are seen, by every group, before those made after.
void Lower::memory_fence() {
    V mode = this->s.m.i32(2);
    const Ty void_t = this->s.t.void_;
    this->s.op(this->cur, "barrier", Attr::NoDuplicate, void_t, 80, std::vector<V>{mode});
}

// The two host waves of a console wave meet (a group barrier with a shared-memory fence): LDS
// writes before are seen by reads after.
void Lower::lds_meeting() {
    V mode = this->s.m.i32(meeting_mode());
    const Ty void_t = this->s.t.void_;
    this->s.op(this->cur, "barrier", Attr::NoDuplicate, void_t, 80, std::vector<V>{mode});
}

// Whether `v` is a register no program reads as a number: a descriptor dword without a host value
// (stale), a lane mask's dword (outside compute, where a ballot gives the mask as data), or a join
// of only such values. Compiled code joins numbers with numbers: where one of these meets numbers
// at a join, the register is dead on that path.
bool Lower::dead(Value v, std::vector<Value>& seen) const {
    if (this->stale.contains(v)) {
        return true;
    }
    const Structured& st = *this->x.st;
    if (this->mask_data != MaskData::Console && (st.masks.contains(v) || st.highs.contains(v))) {
        return true;
    }
    auto it = this->phi_at.find(v);
    if (it == this->phi_at.end()) {
        return false;
    }
    const auto [b, i] = it->second;
    if (std::find(seen.begin(), seen.end(), v) != seen.end()) {
        return true;
    }
    seen.push_back(v);
    for (const Value& a : this->x.p->blocks.at(b).phis.at(i).args) {
        if (!this->dead(a, seen)) return false;
    }
    return true;
}

// `s_bcnt1_i32_b*` of the lanes that differ from the first lane's value: `v_readfirstlane_b32 s,
// v`, `v_cmp_eq_u32 m, s, v`, `s_xor_b* d, m, exec`. The compiler tests a value for being the same
// in the whole wave this way, to take a scalar path; both paths give each lane the same result, so
// the lanes of a host wave may ask it of their own first lane, whatever the wave's size.
bool Lower::uniform_test(const Inst& inst) const {
    const Program& p = *this->x.p;
    auto writer = [&](std::optional<Value> v) -> const Inst* {
        if (!v) return nullptr;
        auto it = this->def_block.find(*v);
        if (it == this->def_block.end()) return nullptr;
        for (const Inst& i : p.blocks.at(it->second).insts) {
            for (const auto& w : i.writes) {
                if (w.second == *v) return &i;
            }
        }
        return nullptr;
    };
    auto scalar = [](const Inst& i, const Operand& op) -> std::optional<Value> {
        if (op.kind == Operand::Kind::Scalar && op.code != gfx10::NULL_CODE && op.code != ir::EXEC) {
            return i.read(Reg::S(op.code));
        }
        return std::nullopt;
    };
    if (!inst.name().starts_with("s_bcnt1_i32_")) {
        return false;
    }
    const auto* sop = std::get_if<gfx10::Sop>(&inst.fields);
    if (!sop) {
        return false;
    }
    const Inst* xr = sop->src.empty() ? nullptr : writer(scalar(inst, sop->src[0]));
    if (!xr) {
        return false;
    }
    const auto* xsop = std::get_if<gfx10::Sop>(&xr->fields);
    if (!xsop) {
        return false;
    }
    const std::string_view xname = xr->name();
    bool has_exec = false;
    for (const Operand& o : xsop->src) {
        if (o.kind == Operand::Kind::Scalar && o.code == ir::EXEC) {
            has_exec = true;
            break;
        }
    }
    if (!(xname == "s_xor_b64" || xname == "s_xor_b32") || !has_exec) {
        return false;
    }
    std::optional<Value> cmp_v;
    for (const Operand& o : xsop->src) {
        if (auto v = scalar(*xr, o)) {
            cmp_v = v;
            break;
        }
    }
    const Inst* cmp = writer(cmp_v);
    if (!cmp) {
        return false;
    }
    const auto* f = std::get_if<gfx10::Valu>(&cmp->fields);
    if (!f) {
        return false;
    }
    const std::string_view cname = cmp->name();
    if (!(cname == "v_cmp_eq_u32" || cname == "v_cmp_eq_i32") || f->sdwa.has_value() || f->dpp.has_value()) {
        return false;
    }
    std::optional<Value> lanes;
    for (const Operand& o : f->src) {
        if (o.kind == Operand::Kind::Vector) {
            if (auto v = cmp->read(Reg::V(o.reg))) {
                lanes = v;
                break;
            }
        }
    }
    std::optional<Value> first_v;
    for (const Operand& o : f->src) {
        if (auto v = scalar(*cmp, o)) {
            first_v = v;
            break;
        }
    }
    const Inst* fl = writer(first_v);
    if (!lanes.has_value() || !fl) {
        return false;
    }
    if (fl->name() != "v_readfirstlane_b32") {
        return false;
    }
    const auto* g = std::get_if<gfx10::Valu>(&fl->fields);
    if (!g || g->src.empty() || g->src[0].kind != Operand::Kind::Vector) {
        return false;
    }
    return fl->read(Reg::V(g->src[0].reg)) == lanes;
}

bool Lower::is_mask(std::optional<Value> v) const {
    return v.has_value() && this->x.st->masks.contains(*v);
}

void Lower::sop(const Inst& inst) {
    const auto& f = std::get<gfx10::Sop>(inst.fields);
    const auto& dst = f.dst;
    const auto& src = f.src;
    const auto& imm = f.imm;
    const std::string_view name = inst.name();
    const std::optional<Value> scc = inst.write(Reg::Scc);
    std::optional<std::uint8_t> dst_code;
    if (dst && dst->kind == Operand::Kind::Scalar) dst_code = dst->code;
    const std::optional<Value> lo = dst_code ? inst.write(Reg::S(*dst_code)) : std::nullopt;
    if (inst.write(Reg::S(ir::EXEC)).has_value() || this->is_mask(lo)) {
        fail(LANE_MASK);
    }
    if (name == "s_setreg_imm32_b32" || name == "s_setreg_b32" || name == "s_swappc_b64" || name == "s_version" ||
        name.starts_with("s_waitcnt")) {
        // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
        // vertex fetch call, whose results arrive as inputs.
        fail("a register the emitter does not model");
    }
    if (name == "s_getpc_b64") {
        // The address of the next instruction is no number the host has: tables in the code are
        // read by their tracked address. As a number the counter is that instruction's offset in
        // the code, so an address paths join before one load is the byte in the code, whichever
        // table each path took (resource tracking gives such a join as the computed term, with no
        // base).
        V z = this->s.m.i32(0);
        V at = this->s.m.i32(std::uint32_t(inst.machine.offset) + 4);
        const std::optional<Value> hi =
            dst_code ? inst.write(Reg::S(std::uint8_t(*dst_code + 1))) : std::nullopt;
        this->set(lo, Val::I32(at));
        this->set(hi, Val::I32(z));
        return;
    }
    const std::size_t c = this->cur;
    auto src32 = [&](Lower& l, std::size_t k) -> V {
        // SOPK arithmetic reads its destination as the first source.
        if (k == 0 && (name == "s_addk_i32" || name == "s_mulk_i32")) {
            const Operand d = ok_or(dst, "a missing operand");
            return l.operand(inst, d);
        }
        const Operand* op = ok_or(k < src.size() ? &src[k] : nullptr, "a missing operand");
        return l.operand(inst, *op);
    };
    auto sext16 = [&](Lower& l) {
        return l.s.m.i32(std::uint32_t(std::int32_t(std::int16_t(imm.value_or(0)))));
    };
    // Dword `h` of 64-bit source `k` (inline constants sign-extend).
    auto half = [&](Lower& l, std::size_t k, std::uint8_t h) -> V {
        const Operand* op = k < src.size() ? &src[k] : nullptr;
        if (op && op->kind == Operand::Kind::Scalar && op->code != gfx10::NULL_CODE) {
            Value v = ok_or(inst.read(Reg::S(std::uint8_t(op->code + h))), "an unread operand");
            return l.i32_of(v);
        }
        if (op && h == 0) {
            return l.operand(inst, *op);
        }
        if (op && op->kind == Operand::Kind::Const) {
            std::uint32_t x = ok_or(resource::inline_constant(op->code), "a special operand");
            return l.s.m.i32(std::int32_t(x) < 0 ? 0xFFFF'FFFFu : 0u);
        }
        // Of a signed operand, a literal sign-extends too.
        if (op && op->kind == Operand::Kind::Literal && name.ends_with("_i64")) {
            return l.s.m.i32(std::int32_t(op->literal) < 0 ? 0xFFFF'FFFFu : 0u);
        }
        return l.s.m.i32(0);
    };
    if (name.starts_with("s_bitcmp")) {
        const std::string_view op = name.substr(std::string_view("s_bitcmp").size());
        // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
        // of src0 is clear (bitcmp0) or set (bitcmp1).
        V bit = src32(*this, 1);
        V k31 = this->s.m.i32(31);
        V idx = this->s.m.bin(c, Bin::And, bit, k31);
        V word{};
        if (op.ends_with("_b64")) {
            V a = half(*this, 0, 0);
            V b = half(*this, 0, 1);
            V k32 = this->s.m.i32(32);
            V upper = this->s.m.bin(c, Bin::And, bit, k32);
            V z = this->s.m.i32(0);
            upper = this->s.m.cmp(c, pred::NE, upper, z);
            word = this->s.m.select(c, upper, b, a);
        } else {
            word = src32(*this, 0);
        }
        V shifted = this->s.m.bin(c, Bin::LShr, word, idx);
        V one = this->s.m.i32(1);
        V tested = this->s.m.bin(c, Bin::And, shifted, one);
        V want = this->s.m.i32(op.starts_with('1') ? 1u : 0u);
        V r = this->s.m.cmp(c, pred::EQ, tested, want);
        this->set(scc, Val::Bit(r));
        return;
    }
    if (name.starts_with("s_cmp_") && name.substr(6).ends_with("_u64")) {
        const std::string_view op = name.substr(6);
        // eq and lg only: both dwords.
        std::optional<V> r;
        for (std::uint8_t h = 0; h < 2; ++h) {
            V a = half(*this, 0, h);
            V b = half(*this, 1, h);
            V e = this->s.m.cmp(c, pred::EQ, a, b);
            if (r) {
                r = this->s.m.bin(c, Bin::And, *r, e);
            } else {
                r = e;
            }
        }
        V rr = *r;  // two dwords
        if (op.starts_with("lg")) {
            V t = this->s.m.bool_(true);
            rr = this->s.m.bin(c, Bin::Xor, rr, t);
        }
        this->set(scc, Val::Bit(rr));
        return;
    }
    if (name.starts_with("s_cmp_") || name.starts_with("s_cmpk_")) {
        const std::string_view op = name.starts_with("s_cmp_") ? name.substr(6) : name.substr(7);
        V a{};
        V b{};
        if (name.starts_with("s_cmpk_")) {
            if (op.ends_with("_i32")) {
                b = sext16(*this);
            } else {
                b = this->s.m.i32(std::uint32_t(imm.value_or(0)));
            }
            a = src32(*this, 0);
        } else {
            if (op.ends_with("64")) {
                fail("a 64-bit scalar compare");
            }
            a = src32(*this, 0);
            b = src32(*this, 1);
        }
        const bool signed_ = op.ends_with("_i32");
        const std::string_view two = op.substr(0, 2);
        std::uint8_t pr = 0;
        if (two == "eq") {
            pr = pred::EQ;
        } else if (two == "lg") {
            pr = pred::NE;
        } else if (two == "gt" && signed_) {
            pr = pred::SGT;
        } else if (two == "ge" && signed_) {
            pr = pred::SGE;
        } else if (two == "lt" && signed_) {
            pr = pred::SLT;
        } else if (two == "le" && signed_) {
            pr = pred::SLE;
        } else if (two == "gt") {
            pr = pred::IUGT;
        } else if (two == "ge") {
            pr = pred::IUGE;
        } else if (two == "lt") {
            pr = pred::IULT;
        } else if (two == "le") {
            pr = pred::IULE;
        } else {
            fail("a scalar compare");
        }
        V r = this->s.m.cmp(c, pr, a, b);
        this->set(scc, Val::Bit(r));
        return;
    }
    const bool wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
    if (wide) {
        const std::optional<Value> hi =
            dst_code ? inst.write(Reg::S(std::uint8_t(*dst_code + 1))) : std::nullopt;
        auto nonzero = [&](Lower& l, V a, V b) {
            V o = l.s.m.bin(c, Bin::Or, a, b);
            V z = l.s.m.i32(0);
            return l.s.m.cmp(c, pred::NE, o, z);
        };
        if (name == "s_quadmask_b64") {
            V a = half(*this, 0, 0);
            V b = half(*this, 0, 1);
            V low = this->quad_mask(a);
            V high = this->quad_mask(b);
            V k8 = this->s.m.i32(8);
            high = this->s.m.bin(c, Bin::Shl, high, k8);
            V mask = this->s.m.bin(c, Bin::Or, low, high);
            V zero = this->s.m.i32(0);
            this->set(lo, Val::I32(mask));
            this->set(hi, Val::I32(zero));
            V nz = this->s.m.cmp(c, pred::NE, mask, zero);
            this->set(scc, Val::Bit(nz));
            return;
        }
        if (name == "s_bcnt1_i32_b64") {
            V a = half(*this, 0, 0);
            V b = half(*this, 0, 1);
            const Ty i32t = this->s.t.i32;
            V na = this->s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, std::vector<V>{a});
            V nb = this->s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, std::vector<V>{b});
            V r = this->s.m.bin(c, Bin::Add, na, nb);
            this->set(lo, Val::I32(r));
            V z = this->s.m.i32(0);
            V sv = this->s.m.cmp(c, pred::NE, r, z);
            this->set(scc, Val::Bit(sv));
            return;
        }
        if (name == "s_ff1_i32_b64") {
            // The lowest set bit; -1 for none (as DXIL's first bit low).
            V a = half(*this, 0, 0);
            V b = half(*this, 0, 1);
            const Ty i32t = this->s.t.i32;
            V fa = this->s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, std::vector<V>{a});
            V fb = this->s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, std::vector<V>{b});
            V z = this->s.m.i32(0);
            V k32 = this->s.m.i32(32);
            V fb32 = this->s.m.bin(c, Bin::Add, fb, k32);
            V none = this->s.m.cmp(c, pred::EQ, b, z);
            fb = this->s.m.select(c, none, fb, fb32);
            V low = this->s.m.cmp(c, pred::NE, a, z);
            V r = this->s.m.select(c, low, fa, fb);
            this->set(lo, Val::I32(r));
            return;
        }
        if (name == "s_and_b64" || name == "s_or_b64" || name == "s_xor_b64" || name == "s_andn2_b64" ||
            name == "s_orn2_b64" || name == "s_nand_b64" || name == "s_nor_b64" || name == "s_xnor_b64" ||
            name == "s_not_b64") {
            V ones = this->s.m.i32(0xFFFF'FFFFu);
            std::vector<V> r;
            for (std::uint8_t h = 0; h < 2; ++h) {
                V a = half(*this, 0, h);
                V x{};
                if (name == "s_not_b64") {
                    x = this->s.m.bin(c, Bin::Xor, a, ones);
                } else {
                    V b = half(*this, 1, h);
                    if (contains(name, "n2")) {
                        b = this->s.m.bin(c, Bin::Xor, b, ones);
                    }
                    const std::string_view mid = name.substr(2, 3);
                    Bin op = Bin::Xor;
                    if (mid == "and" || mid == "nan") {
                        op = Bin::And;
                    } else if (mid == "or_" || mid == "orn" || mid == "nor") {
                        op = Bin::Or;
                    }
                    x = this->s.m.bin(c, op, a, b);
                    if (name == "s_nand_b64" || name == "s_nor_b64" || name == "s_xnor_b64") {
                        x = this->s.m.bin(c, Bin::Xor, x, ones);
                    }
                }
                r.push_back(x);
            }
            this->set(lo, Val::I32(r[0]));
            this->set(hi, Val::I32(r[1]));
            V sv = nonzero(*this, r[0], r[1]);
            this->set(scc, Val::Bit(sv));
            return;
        }
        if (name == "s_lshl_b64" || name == "s_lshr_b64" || name == "s_ashr_i64") {
            V a = half(*this, 0, 0);
            V b = half(*this, 0, 1);
            V n = src32(*this, 1);
            const Ty i64t = this->s.m.int_(64);
            this->s.flags |= flags::INT64;
            a = this->s.m.cast(c, Cast::ZExt, a, i64t);
            b = this->s.m.cast(c, Cast::ZExt, b, i64t);
            V k32 = this->s.m.int_const(i64t, 32);
            b = this->s.m.bin(c, Bin::Shl, b, k32);
            V x = this->s.m.bin(c, Bin::Or, a, b);
            V k63 = this->s.m.i32(63);
            n = this->s.m.bin(c, Bin::And, n, k63);
            n = this->s.m.cast(c, Cast::ZExt, n, i64t);
            Bin op = Bin::AShr;
            if (name == "s_lshl_b64") {
                op = Bin::Shl;
            } else if (name == "s_lshr_b64") {
                op = Bin::LShr;
            }
            V r = this->s.m.bin(c, op, x, n);
            V rl = this->s.m.cast(c, Cast::Trunc, r, this->s.t.i32);
            V rh = this->s.m.bin(c, Bin::LShr, r, k32);
            rh = this->s.m.cast(c, Cast::Trunc, rh, this->s.t.i32);
            this->set(lo, Val::I32(rl));
            this->set(hi, Val::I32(rh));
            V sv = nonzero(*this, rl, rh);
            this->set(scc, Val::Bit(sv));
            return;
        }
        if (name == "s_mov_b64") {
            // Each half by itself: two numbers that travel as a pair (two scalars parked in
            // neighbouring VGPR lanes, say) need not both have a value.
            const std::pair<std::uint8_t, std::optional<Value>> both[2] = {{0, lo}, {1, hi}};
            for (const auto& [h, v] : both) {
                Val val{};
                try {
                    val = Val::I32(half(*this, 0, h));
                } catch (const WhyFail& e) {
                    val = Val::Missing(e.why);
                }
                this->set(v, val);
            }
            return;
        }
        if (name == "s_cselect_b64") {
            V cond = this->scc_in(inst);
            const std::pair<std::uint8_t, std::optional<Value>> both[2] = {{0, lo}, {1, hi}};
            for (const auto& [h, v] : both) {
                V a = half(*this, 0, h);
                V b = half(*this, 1, h);
                V r = this->s.m.select(c, cond, a, b);
                this->set(v, Val::I32(r));
            }
            return;
        }
        fail(name.data());
    }
    V r{};
    std::optional<V> carry;
    if (name == "s_mov_b32") {
        r = src32(*this, 0);
    } else if (name == "s_movk_i32") {
        r = sext16(*this);
    } else if (name == "s_quadmask_b32") {
        V pixels = src32(*this, 0);
        r = this->quad_mask(pixels);
    } else if (name == "s_cselect_b32") {
        V cond = this->scc_in(inst);
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        r = this->s.m.select(c, cond, a, b);
    } else if (name == "s_brev_b32") {
        V a = src32(*this, 0);
        r = this->s.op(c, "unary.i32", Attr::ReadNone, this->s.t.i32, 30, std::vector<V>{a});
    } else if (name == "s_not_b32") {
        V a = src32(*this, 0);
        V ones = this->s.m.i32(0xFFFF'FFFFu);
        r = this->s.m.bin(c, Bin::Xor, a, ones);
    } else if (name == "s_add_u32" || name == "s_add_i32" || name == "s_addk_i32") {
        V a = src32(*this, 0);
        V b = name == "s_addk_i32" ? sext16(*this) : src32(*this, 1);
        r = this->s.m.bin(c, Bin::Add, a, b);
        if (name == "s_add_u32") {
            carry = this->s.m.cmp(c, pred::IULT, r, a);
        }
    } else if (name == "s_sub_u32" || name == "s_sub_i32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        r = this->s.m.bin(c, Bin::Sub, a, b);
        if (name == "s_sub_u32") {
            carry = this->s.m.cmp(c, pred::IULT, a, b);
        }
    } else if (name == "s_addc_u32" || name == "s_subb_u32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        V cin = this->scc_in(inst);
        auto [rr, cc] = this->add_sub_carry(c, name == "s_addc_u32", a, b, cin);
        r = rr;
        carry = cc;
    } else if (name == "s_mul_i32" || name == "s_mulk_i32") {
        V a = src32(*this, 0);
        V b = name == "s_mulk_i32" ? sext16(*this) : src32(*this, 1);
        r = this->s.m.bin(c, Bin::Mul, a, b);
    } else if (name == "s_mul_hi_u32" || name == "s_mul_hi_i32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        r = this->mul_hi(a, b, name.ends_with("i32"));
    } else if (name == "s_flbit_i32_b32" || name == "s_flbit_i32") {
        // The first bit from the top (differing from the sign for the signed form), -1 for none:
        // DXIL's first-bit-high.
        V a = src32(*this, 0);
        const std::uint32_t op = name == "s_flbit_i32_b32" ? 33 : 34;
        r = this->s.op(c, "unaryBits.i32", Attr::ReadNone, this->s.t.i32, op, std::vector<V>{a});
    } else if (name == "s_ff1_i32_b32" || name == "s_bcnt1_i32_b32") {
        // First set bit (-1 for zero, as DXIL's), or the bit count.
        V a = src32(*this, 0);
        const std::uint32_t op = name == "s_ff1_i32_b32" ? 32 : 31;
        r = this->s.op(c, "unaryBits.i32", Attr::ReadNone, this->s.t.i32, op, std::vector<V>{a});
    } else if (name == "s_bfe_u32" || name == "s_bfe_i32") {
        // src1: offset in bits 0-4, width in bits 16-22.
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        V k16 = this->s.m.i32(16);
        V k127 = this->s.m.i32(0x7F);
        V w = this->s.m.bin(c, Bin::LShr, b, k16);
        w = this->s.m.bin(c, Bin::And, w, k127);
        r = this->bfe(a, b, w, name == "s_bfe_i32");
    } else if (name == "s_and_b32" || name == "s_or_b32" || name == "s_xor_b32" || name == "s_andn2_b32" ||
               name == "s_orn2_b32" || name == "s_nand_b32" || name == "s_nor_b32" || name == "s_xnor_b32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        V ones = this->s.m.i32(0xFFFF'FFFFu);
        if (contains(name, "n2")) {
            b = this->s.m.bin(c, Bin::Xor, b, ones);
        }
        const std::string_view mid = name.substr(2, 3);
        Bin op = Bin::Xor;
        if (mid == "and" || mid == "nan") {
            op = Bin::And;
        } else if (mid == "or_" || mid == "orn" || mid == "nor") {
            op = Bin::Or;
        }
        r = this->s.m.bin(c, op, a, b);
        if (name == "s_nand_b32" || name == "s_nor_b32" || name == "s_xnor_b32") {
            r = this->s.m.bin(c, Bin::Xor, r, ones);
        }
    } else if (name == "s_lshl_b32" || name == "s_lshr_b32" || name == "s_ashr_i32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        V k31 = this->s.m.i32(31);
        b = this->s.m.bin(c, Bin::And, b, k31);
        Bin op = Bin::AShr;
        if (name == "s_lshl_b32") {
            op = Bin::Shl;
        } else if (name == "s_lshr_b32") {
            op = Bin::LShr;
        }
        r = this->s.m.bin(c, op, a, b);
    } else if (name == "s_lshl1_add_u32" || name == "s_lshl2_add_u32" || name == "s_lshl3_add_u32" ||
               name == "s_lshl4_add_u32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        V k = this->s.m.i32(std::uint32_t(std::uint8_t(name[6]) - std::uint8_t('0')));
        V t = this->s.m.bin(c, Bin::Shl, a, k);
        r = this->s.m.bin(c, Bin::Add, t, b);
    } else if (name == "s_min_u32" || name == "s_max_u32" || name == "s_min_i32" || name == "s_max_i32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        std::uint8_t pr = pred::SGT;
        if (name == "s_min_u32") {
            pr = pred::IULT;
        } else if (name == "s_max_u32") {
            pr = pred::IUGT;
        } else if (name == "s_min_i32") {
            pr = pred::SLT;
        }
        V t = this->s.m.cmp(c, pr, a, b);
        r = this->s.m.select(c, t, a, b);
    } else if (name == "s_abs_i32") {
        V a = src32(*this, 0);
        V z = this->s.m.i32(0);
        V neg = this->s.m.cmp(c, pred::SLT, a, z);
        V n = this->s.m.bin(c, Bin::Sub, z, a);
        r = this->s.m.select(c, neg, n, a);
    } else if (name == "s_bitset0_b32" || name == "s_bitset1_b32") {
        // Sets (clears) bit src[4:0] of the destination it reads.
        const Operand d = ok_or(dst, "a missing operand");
        V old = this->operand(inst, d);
        V a = src32(*this, 0);
        V k31 = this->s.m.i32(31);
        V one = this->s.m.i32(1);
        a = this->s.m.bin(c, Bin::And, a, k31);
        V bit = this->s.m.bin(c, Bin::Shl, one, a);
        if (name == "s_bitset1_b32") {
            r = this->s.m.bin(c, Bin::Or, old, bit);
        } else {
            V ones = this->s.m.i32(0xFFFF'FFFFu);
            V nb = this->s.m.bin(c, Bin::Xor, bit, ones);
            r = this->s.m.bin(c, Bin::And, old, nb);
        }
    } else if (name == "s_bfm_b32") {
        V a = src32(*this, 0);
        V b = src32(*this, 1);
        V k31 = this->s.m.i32(31);
        V one = this->s.m.i32(1);
        a = this->s.m.bin(c, Bin::And, a, k31);
        b = this->s.m.bin(c, Bin::And, b, k31);
        V m = this->s.m.bin(c, Bin::Shl, one, a);
        m = this->s.m.bin(c, Bin::Sub, m, one);
        r = this->s.m.bin(c, Bin::Shl, m, b);
    } else {
        fail(name.data());
    }
    this->set(lo, Val::I32(r));
    if (scc) {
        Val val{};
        if (carry) {
            val = Val::Bit(*carry);
        } else if (name == "s_mov_b32" || name == "s_movk_i32" || name == "s_cselect_b32" || name == "s_brev_b32" ||
                   name == "s_mul_i32" || name == "s_mulk_i32") {
            return;
        } else if (name.starts_with("s_add") || name.starts_with("s_sub")) {
            val = Val::Missing("the carry of a scalar operation");
        } else {
            V z = this->s.m.i32(0);
            val = Val::Bit(this->s.m.cmp(c, pred::NE, r, z));
        }
        this->vals.insert_or_assign(*scc, val);
    }
}

// Packs whether each consecutive group of four pixels has a set bit.
V Lower::quad_mask(V pixels) {
    const std::size_t c = this->cur;
    V one = this->s.m.i32(1);
    V shifted = this->s.m.bin(c, Bin::LShr, pixels, one);
    V mask = this->s.m.bin(c, Bin::Or, pixels, shifted);
    // Fold each nibble into its bottom bit, then gather those bits into the low byte in three
    // stages without needing a 64-bit operation.
    const std::pair<std::uint32_t, std::uint32_t> steps[4] = {
        {2, 0x1111'1111u},
        {3, 0x0303'0303u},
        {6, 0x000f'000fu},
        {12, 0x0000'00ffu},
    };
    for (const auto& [dist, keep_bits] : steps) {
        V distance = this->s.m.i32(dist);
        V sh = this->s.m.bin(c, Bin::LShr, mask, distance);
        V combined = this->s.m.bin(c, Bin::Or, mask, sh);
        V keep = this->s.m.i32(keep_bits);
        mask = this->s.m.bin(c, Bin::And, combined, keep);
    }
    return mask;
}

// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the scalar and vector
// add-with-carry instructions set it.
std::pair<V, V> Lower::add_sub_carry(std::size_t c, bool add, V a, V b, std::optional<V> cin) {
    const Bin op = add ? Bin::Add : Bin::Sub;
    V r = this->s.m.bin(c, op, a, b);
    // Without a carry in: a + b wraps below a; a - b borrows when a < b.
    V carry = add ? this->s.m.cmp(c, pred::IULT, r, a) : this->s.m.cmp(c, pred::IULT, a, b);
    if (cin) {
        V one = this->s.m.cast(c, Cast::ZExt, *cin, this->s.t.i32);
        V t = r;
        r = this->s.m.bin(c, op, t, one);
        // The carry in carries on when the partial result is all ones (add) or zero (sub).
        V edge = this->s.m.i32(add ? 0xFFFF'FFFFu : 0u);
        V at_edge = this->s.m.cmp(c, pred::EQ, t, edge);
        V more = this->s.m.bin(c, Bin::And, *cin, at_edge);
        carry = this->s.m.bin(c, Bin::Or, carry, more);
    }
    return {r, carry};
}

V Lower::scc_in(const Inst& inst) {
    const Value v = ok_or(inst.read(Reg::Scc), "an unread SCC");
    std::optional<Why> err;
    Val g{};
    try {
        g = this->get(v);
    } catch (const WhyFail& e) {
        err = e.why;
    }
    if (err) {
        // Of a mask operation whose result lies within EXEC: whether some lane is left.
        if (same(*err, LANE_MASK) && this->within_exec(v)) {
            return this->scc_some(v);
        }
        fail(*err);
    }
    if (g.is_bit()) {
        return g.v;
    }
    fail("SCC without a value");
}

// Whether the mask operation that sets the SCC value `scc` gives no lane outside EXEC a bit.
bool Lower::within_exec(Value scc) const {
    auto it = this->def_block.find(scc);
    if (it == this->def_block.end()) {
        return false;
    }
    const Inst* by = nullptr;
    for (const Inst& i : this->x.p->blocks.at(it->second).insts) {
        if (i.write(Reg::Scc) == std::optional<Value>(scc)) {
            by = &i;
            break;
        }
    }
    if (!by) {
        return false;
    }
    if (by->write(Reg::S(ir::EXEC)).has_value()) {
        return true;
    }
    const auto* sop = std::get_if<gfx10::Sop>(&by->fields);
    if (sop && sop->dst && sop->dst->kind == Operand::Kind::Scalar) {
        if (auto lo = by->write(Reg::S(sop->dst->code))) {
            return this->x.st->active.contains(*lo);
        }
    }
    return false;
}

// Whether the dwords of one console load are read by one call for up to four of them
// (`bitcode::opt::WIDE` of `X2W_SHADERCFG_SHADER_OPT`), not by a call each.
bool Lower::wide_loads() {
    return (emit::shader_opt() & opt::WIDE) != 0;
}

// `count` dwords from byte `addr` on of the raw view `h`, as integers, read in the current block by one
// call for every four of them (wide_loads). They are the dwords of ONE console load, which the console
// reads as one (address plus 4, 8,... without wrapping at 32 bits): the same bytes as a call a dword reads.
// Loads of different console instructions are never joined: the program computes their offsets in 32-bit
// registers, and `x - 4` and `x` at `x = 0` are one dword out of the view and its first one, where one call
// from `x - 4` on would read both out of it.
std::vector<V> Lower::raw_dwords(V h, V addr, std::uint32_t count) {
    const std::size_t c = this->cur;
    V u = this->s.m.undef(this->s.t.i32);
    V al = this->s.m.i32(4);
    std::vector<V> out;
    out.reserve(count);
    for (std::uint32_t first = 0; first < count; first += 4) {
        const std::uint32_t n = std::min<std::uint32_t>(count - first, 4);
        V off = this->s.m.i32(4 * first);
        V a = this->s.m.bin(c, Bin::Add, addr, off);
        V mask = this->s.m.i8(std::uint8_t((1u << n) - 1));
        V r = this->s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                         std::vector<V>{h, a, u, mask, al});
        for (std::uint32_t k = 0; k < n; ++k) {
            V x = this->s.m.extract(c, r, k);
            out.push_back(this->as_int(x));
        }
    }
    return out;
}

}  // namespace x2w::vs::lower
