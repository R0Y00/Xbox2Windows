// Buffer accesses, exports, block exits, loop counters, edges and phis.
#include <algorithm>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>

#include "x2w_shader/emit/lower_impl.h"

namespace x2w::vs::lower {

namespace {

[[noreturn]] void unreachable_here() { throw std::logic_error("unreachable"); }

// Replace every occurrence.
std::string replace_all(std::string_view s, std::string_view from, std::string_view to) {
    std::string out;
    std::size_t at = 0;
    while (true) {
        std::size_t k = s.find(from, at);
        if (k == std::string_view::npos) {
            out.append(s.substr(at));
            return out;
        }
        out.append(s.substr(at, k - at));
        out.append(to);
        at = k + from.size();
    }
}

std::optional<std::string_view> strip_prefix(std::string_view s, std::string_view p) {
    if (s.starts_with(p)) return s.substr(p.size());
    return std::nullopt;
}

// ---- front-end shapes (vs-front headers not published when written) ----
// Usage::Other(0)
bool usage_is_other0(const Usage& u) { return u.kind == Usage::Kind::Other && u.other == 0; }
// LaneExit::Branch { when, .. } on a condition not over uniform atoms only.
bool all_uniform(const Structured& st, Bool when) {
    for (const Atom& a : st.bools.support(when)) {
        if (a.kind != Atom::Kind::Uniform) return false;
    }
    return true;
}

}  // namespace

void Lower::buffer(const Inst& inst, std::size_t b, std::size_t i) {
    this->root_copy = std::nullopt;
    const auto* bf = std::get_if<gfx10::Buffer>(&inst.fields);
    if (!bf) unreachable_here();
    const auto& data = bf->data;
    const auto& addr = bf->addr;
    const auto& soffset = bf->soffset;
    const std::uint16_t offset = bf->offset;
    const bool offen = bf->offen;
    const bool idxen = bf->idxen;
    const auto& format = bf->format;
    std::string_view name = inst.name();
    if (name == "buffer_gl0_inv" || name == "buffer_gl1_inv") {
        // The host has no console buffer cache to invalidate.
        return;
    }
    if (bf->lds || bf->tfe) fail("a buffer access to LDS or with tfe");
    if (format.has_value() && !name.starts_with("tbuffer_load_format_")) fail(name.data());
    const Site site = {b, i};
    const std::optional<V> guard = this->guard(b, i);
    const std::size_t c = this->cur;
    const bool store = name.starts_with("buffer_store_");
    // DXIL atomic operations: add, and, or, xor, signed and unsigned min and max, exchange;
    // compare-exchange apart.
    std::optional<std::uint32_t> atomic;
    if (auto rest = strip_prefix(name, "buffer_atomic_")) {
        if (*rest == "add") atomic = 0;
        else if (*rest == "and") atomic = 1;
        else if (*rest == "or") atomic = 2;
        else if (*rest == "xor") atomic = 3;
        else if (*rest == "smin") atomic = 4;
        else if (*rest == "smax") atomic = 5;
        else if (*rest == "umin") atomic = 6;
        else if (*rest == "umax") atomic = 7;
        else if (*rest == "swap") atomic = 8;
        else if (*rest == "cmpswap") atomic = UINT32_MAX;
        else fail(name.data());
    }
    // Byte and 16-bit accesses: bits, and whether a load sign-extends.
    std::optional<std::pair<std::uint32_t, bool>> narrow;
    if (name == "buffer_load_ubyte" || name == "buffer_store_byte") narrow = std::pair{8u, false};
    else if (name == "buffer_load_sbyte") narrow = std::pair{8u, true};
    else if (name == "buffer_load_ushort" || name == "buffer_store_short") narrow = std::pair{16u, false};
    else if (name == "buffer_load_sshort") narrow = std::pair{16u, true};
    // d16: two 16-bit channels per VGPR, low half first (buffer and tbuffer formats alike).
    const bool d16 = name.find("_format_d16_") != std::string_view::npos;
    const std::string plain = replace_all(name, "_format_d16_", "_format_");
    std::optional<std::string_view> kind = strip_prefix(plain, "buffer_load_");
    if (!kind) kind = strip_prefix(plain, "buffer_store_");
    if (!kind) kind = strip_prefix(plain, "tbuffer_load_");
    bool typed = false;
    std::uint32_t count = 0;
    if (atomic.has_value() || narrow.has_value()) {
        typed = false, count = 1;
    } else if (kind == "dword") {
        typed = false, count = 1;
    } else if (kind == "dwordx2") {
        typed = false, count = 2;
    } else if (kind == "dwordx3") {
        typed = false, count = 3;
    } else if (kind == "dwordx4") {
        typed = false, count = 4;
    } else if (kind == "format_x") {
        typed = true, count = 1;
    } else if (kind == "format_xy") {
        typed = true, count = 2;
    } else if (kind == "format_xyz") {
        typed = true, count = 3;
    } else if (kind == "format_xyzw") {
        typed = true, count = 4;
    } else {
        fail(name.data());
    }
    if (addr.kind != Operand::Kind::Vector) fail("a buffer address");
    const std::uint8_t areg = addr.reg;
    auto vaddr = [&](Lower& l, std::uint8_t k) -> V {
        Value v = ok_or(inst.read(Reg::V(static_cast<std::uint8_t>(areg + k))), "a buffer address");
        return l.i32_of(v);
    };
    std::optional<V> index;
    if (idxen) index = vaddr(*this, 0);
    V off = this->s.m.i32(static_cast<std::uint32_t>(offset));
    if (offen) {
        V o = vaddr(*this, static_cast<std::uint8_t>(idxen ? 1 : 0));
        off = this->s.m.bin(c, Bin::Add, off, o);
    }
    if (!((soffset.kind == Operand::Kind::Scalar && soffset.code == gfx10::NULL_CODE) ||
          (soffset.kind == Operand::Kind::Const && soffset.code == 128))) {
        V so = this->operand(inst, soffset);
        off = this->s.m.bin(c, Bin::Add, off, so);
    }
    const Source* src = nullptr;
    {
        auto it = this->descriptors.find(std::pair{site, false});
        if (it == this->descriptors.end()) fail("an untracked buffer");
        src = it->second;
    }
    const UsedResource* used = this->used_kind(*src);
    auto [base, extra] = this->built_offset(*src);
    if (base != 0) {
        V bk = this->s.m.i32(base);
        off = this->s.m.bin(c, Bin::Add, off, bk);
    }
    if (extra) off = this->s.m.bin(c, Bin::Add, off, *extra);
    // A buffer the program picks from the view heap by an index it computes: no used-resource
    // entry gives its stride, the host's table by heap slot does; the access becomes a raw one at
    // the element's byte. (Also tells the host this shader reads such buffers.)
    {
        std::optional<V> stride = this->slot_stride(*src);
        if (stride && index && !typed && used == nullptr) {
            V at = this->s.m.bin(c, Bin::Mul, *index, *stride);
            off = this->s.m.bin(c, Bin::Add, at, off);
            index = std::nullopt;
        }
    }
    // A structured buffer by its used-resource entry's stride: raw bytes too. The host makes its
    // buffer views raw (a raw read through a structured view gives zeros on NVIDIA), and one kind
    // of view can then serve every shader that reads the buffer.
    if (index && used && !typed && used->kind != ResourceType::Cbv() && used->stride >= 4 &&
        used->stride % 4 == 0) {
        V k = this->s.m.i32(used->stride);
        V at = this->s.m.bin(c, Bin::Mul, *index, k);
        off = this->s.m.bin(c, Bin::Add, at, off);
        index = std::nullopt;
    }
    if (used && used->kind == ResourceType::Cbv() && !typed) {
        // A constant buffer read per lane (a dynamically indexed array).
        // shortcut: the console's CBV descriptor is taken to have a 16-byte stride, as the vector
        // loads of whole rows suggest.
        V h = this->handle(site, *src, false, {13, 0x1'0000});
        V byte = off;
        if (index) {
            V sixteen = this->s.m.i32(16);
            V rows = this->s.m.bin(c, Bin::Mul, *index, sixteen);
            byte = this->s.m.bin(c, Bin::Add, rows, off);
        }
        if (!data || data->kind != Operand::Kind::Vector) fail("a buffer result");
        const std::uint8_t reg = data->reg;
        for (std::uint32_t k = 0; k < count; ++k) {
            V four = this->s.m.i32(4 * k);
            V a = this->s.m.bin(c, Bin::Add, byte, four);
            V x = this->cbuffer_dword(h, a);
            this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), x, guard);
        }
        return;
    }
    if (this->root_address(*src).has_value()) {
        // Title memory by address: the byte in the host copy from here on.
        if (index) {
            if (used == nullptr || used->stride != 1) {
                fail("an indexed access through a root view by address");
            }
            off = this->s.m.bin(c, Bin::Add, *index, off);
            index = std::nullopt;
        }
        if (auto m = this->root_memory(*src, off, guard, store || atomic.has_value())) {
            off = m->second;
        }
    }
    const std::size_t c2 = this->cur;
    const bool uav = store || atomic.has_value() || (used && used->kind == ResourceType::Uav());
    std::uint32_t u = uav ? 0x1000 : 0;
    if (uav && !typed) {
        auto e = this->entry(*src);
        if (e && this->coherent.contains(*e)) u |= 0x4000;
    }
    V iu = this->s.m.undef(this->s.t.i32);
    if (narrow) {
        const auto [bits, signed_] = *narrow;
        // Below a dword: raw buffers are read and written in dwords on 4 bytes, so the dword that
        // holds the value is read and shifted, or changed with two atomic operations that touch
        // only the value's own bits (other lanes may write the rest of the dword at the same
        // time). A 16-bit value at a dword's last byte continues in the next dword.
        V byte = off;
        if (index) {
            if (used && used->stride == 1) {
                byte = this->s.m.bin(c2, Bin::Add, *index, off);
            } else {
                fail("a byte or 16-bit buffer access by index");
            }
        }
        V h = this->handle(site, *src, false, {11 | u, 0});
        V k3 = this->s.m.i32(3);
        V within = this->s.m.bin(c2, Bin::And, byte, k3);
        V aligned = this->s.m.bin(c2, Bin::Sub, byte, within);
        V shift = this->s.m.bin(c2, Bin::Shl, within, k3);
        V four = this->s.m.i32(4);
        V next = this->s.m.bin(c2, Bin::Add, aligned, four);
        V ones = this->s.m.i32(bits == 8 ? 0xff : 0xffff);
        V zero = this->s.m.i32(0);
        V k8 = this->s.m.i32(8);
        V straddles = this->s.m.cmp(c2, pred::EQ, within, k3);
        if (!store) {
            V mask = this->s.m.i8(1);
            V align = this->s.m.i32(4);
            auto dword = [&](Lower& l, V at) -> V {
                V r = l.s.op(c2, "rawBufferLoad.f32", Attr::ReadOnly, l.s.t.res_ret, 139,
                             std::vector<V>{h, at, iu, mask, align});
                V f = l.s.m.extract(c2, r, 0);
                return l.as_int(f);
            };
            V low = dword(*this, aligned);
            V x = this->s.m.bin(c2, Bin::LShr, low, shift);
            if (bits == 16) {
                V high = dword(*this, next);
                high = this->s.m.bin(c2, Bin::Shl, high, k8);
                high = this->s.m.select(c2, straddles, high, zero);
                x = this->s.m.bin(c2, Bin::Or, x, high);
            }
            x = this->s.m.bin(c2, Bin::And, x, ones);
            if (signed_) {
                V up = this->s.m.i32(32 - bits);
                x = this->s.m.bin(c2, Bin::Shl, x, up);
                x = this->s.m.bin(c2, Bin::AShr, x, up);
            }
            if (!data || data->kind != Operand::Kind::Vector) fail("a buffer result");
            this->write(inst, Reg::V(data->reg), x, guard);
            return;
        }
        if (!data) fail("buffer store data");
        const Operand& d = *data;
        V v = this->vgprs(inst, d, 1).at(0);
        v = this->s.m.bin(c2, Bin::And, v, ones);
        V all = this->s.m.i32(UINT32_MAX);
        V clear = this->s.m.bin(c2, Bin::Shl, ones, shift);
        V keep = this->s.m.bin(c2, Bin::Xor, clear, all);
        V set = this->s.m.bin(c2, Bin::Shl, v, shift);
        V keep_next = this->s.m.i32(0xffff'ff00u);
        V set_next = this->s.m.bin(c2, Bin::LShr, v, k8);
        V and_ = this->s.m.i32(1);
        V or_ = this->s.m.i32(2);
        const Ty i32t = this->s.t.i32;
        auto at = this->begin_guarded(guard);
        for (auto [op, x] : {std::pair{and_, keep}, std::pair{or_, set}}) {
            this->s.op(this->cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78,
                       std::vector<V>{h, op, aligned, iu, iu, x});
        }
        if (bits == 16) {
            auto second = this->begin_guarded(straddles);
            for (auto [op, x] : {std::pair{and_, keep_next}, std::pair{or_, set_next}}) {
                this->s.op(this->cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78,
                           std::vector<V>{h, op, next, iu, iu, x});
            }
            this->end_guarded(second, {});
        }
        this->end_guarded(at, {});
        return;
    }
    if (atomic) {
        const std::uint32_t op = *atomic;
        if (!data) fail("buffer atomic data");
        const Operand& d = *data;
        std::vector<V> vals = this->vgprs(inst, d, op == UINT32_MAX ? 2 : 1);
        V h{};
        std::array<V, 3> coords{};
        if (index && used && used->stride == 1) {
            // A raw buffer (stride 1) read by index: the index is the byte.
            V byte = this->s.m.bin(c2, Bin::Add, *index, off);
            h = this->handle(site, *src, false, {11 | u, 0});
            coords = {byte, iu, iu};
        } else if (index) {
            const std::uint32_t stride = used ? used->stride : 0;
            if (stride < 4 || stride % 4 != 0) fail("a structured buffer of unknown stride");
            h = this->handle(site, *src, false, {12 | u, stride});
            coords = {*index, off, iu};
        } else {
            h = this->handle(site, *src, false, {11 | u, 0});
            coords = {off, iu, iu};
        }
        if (d.kind != Operand::Kind::Vector) fail("buffer atomic data");
        const std::uint8_t reg = d.reg;
        const bool returns = bf->cache.glc;
        V old{};
        {
            auto r = inst.read(Reg::V(reg));
            if (r && returns) {
                old = this->i32_of(*r);
            } else {
                old = this->s.m.i32(0);
            }
        }
        auto at = this->begin_guarded(guard);
        const Ty i32t = this->s.t.i32;
        V r{};
        if (op == UINT32_MAX) {
            // cmpswap: data is (new, compare).
            std::vector<V> args{h};
            args.insert(args.end(), coords.begin(), coords.end());
            args.push_back(vals.at(1));
            args.push_back(vals.at(0));
            r = this->s.op(this->cur, "atomicCompareExchange.i32", Attr::NoUnwind, i32t, 79, args);
        } else {
            V opk = this->s.m.i32(op);
            std::vector<V> args{h, opk};
            args.insert(args.end(), coords.begin(), coords.end());
            args.push_back(vals.at(0));
            r = this->s.op(this->cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, args);
        }
        const std::pair<V, V> joined[] = {{r, old}};
        r = this->end_guarded(at, joined).at(0);
        if (returns) this->write(inst, Reg::V(reg), r, std::nullopt);
        return;
    }
    if (store) {
        const std::uint32_t regs = d16 ? div_ceil(count, 2u) : count;
        if (!data) fail("buffer store data");
        std::vector<V> vals = this->vgprs(inst, *data, static_cast<std::uint8_t>(regs));
        if (d16) {
            // Widen d16 data as f16 for a 32-bit float store. For an integer view, zero- or sign-extend the 16-bit
            // integer. The face network stores halves through UINT views. This format information is available for
            // views selected from the heap. The handle remains float-typed and the driver passes its bits through.
            std::optional<std::pair<V, V>> integer;
            if (typed) integer = this->integer_view(*src);
            std::vector<V> wide;
            for (std::uint32_t k = 0; k < count; ++k) {
                V x = vals.at(k / 2);
                if (k % 2 == 1) {
                    V k16 = this->s.m.i32(16);
                    x = this->s.m.bin(c2, Bin::LShr, x, k16);
                }
                const Ty f32t = this->s.t.f32;
                V f = this->s.op(c2, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{x});
                V w = this->as_int(f);
                if (integer) {
                    const auto [int_, signed_] = *integer;
                    V k16 = this->s.m.i32(16);
                    V low = this->s.m.i32(0xFFFF);
                    V zero_ext = this->s.m.bin(c2, Bin::And, x, low);
                    V up = this->s.m.bin(c2, Bin::Shl, x, k16);
                    V sign_ext = this->s.m.bin(c2, Bin::AShr, up, k16);
                    V n = this->s.m.select(c2, signed_, sign_ext, zero_ext);
                    w = this->s.m.select(c2, int_, n, w);
                }
                wide.push_back(w);
            }
            vals = std::move(wide);
        }
        std::vector<V> args;
        std::uint32_t op = 0;
        std::string_view opname;
        if (typed) {
            // shortcut: typed stores are taken to write all four channels of a float format.
            if (count != 4 || offen || offset != 0) {
                fail("a typed buffer store of some channels or at an offset");
            }
            args.push_back(this->handle(site, *src, false, {10 | u, 0x409}));
            // Create the zero constant even when an index is present to preserve constant numbering.
            V zero = this->s.m.i32(0);
            args.push_back(index.value_or(zero));
            args.push_back(iu);
            for (V x : vals) args.push_back(this->as_float(x));
            op = 69, opname = "bufferStore.f32";
        } else {
            if (index && used && used->stride == 1) {
                // A raw buffer (stride 1) written by index: the index is the byte.
                V byte = this->s.m.bin(c2, Bin::Add, *index, off);
                args.push_back(this->handle(site, *src, false, {11 | u, 0}));
                args.push_back(byte);
                args.push_back(iu);
            } else if (index) {
                const std::uint32_t stride = used ? used->stride : 0;
                if (stride < 4 || stride % 4 != 0) fail("a structured buffer of unknown stride");
                args.push_back(this->handle(site, *src, false, {12 | u, stride}));
                args.push_back(*index);
                args.push_back(off);
            } else {
                args.push_back(this->handle(site, *src, false, {11 | u, 0}));
                args.push_back(off);
                args.push_back(iu);
            }
            args.insert(args.end(), vals.begin(), vals.end());
            args.resize(3 + 4, iu);
            op = 140, opname = "rawBufferStore.i32";
        }
        args.push_back(this->s.m.i8(typed ? 0xF : static_cast<std::uint8_t>((1u << count) - 1)));
        if (!typed) args.push_back(this->s.m.i32(4));
        auto at = this->begin_guarded(guard);
        const Ty void_t = this->s.t.void_;
        this->s.op(this->cur, opname, Attr::NoUnwind, void_t, op, args);
        this->end_guarded(at, {});
        return;
    }
    if (format) {
        const std::uint8_t f = *format;
        if (index) fail("a typed-format load by index");
        V h = this->handle(site, *src, false, {11 | u, 0});
        if (!data || data->kind != Operand::Kind::Vector) fail("a buffer result");
        const std::uint8_t reg = data->reg;
        std::vector<V> xs = this->format_load(h, off, f, count);
        if (d16) xs = this->d16_channels(f, std::move(xs));
        for (std::size_t k = 0; k < xs.size(); ++k) {
            this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), xs[k], guard);
        }
        return;
    }
    V r{};
    if (typed) {
        if (offen || off.kind == V::Inst || offset != 0) {
            fail("a typed buffer load at an offset");
        }
        if (uav) this->s.flags |= flags::TYPED_UAV_LOADS;
        V h = this->handle(site, *src, false, {10 | u, 0x409});
        // Create the zero constant even when an index is present to preserve constant numbering.
        V zero = this->s.m.i32(0);
        V ix = index.value_or(zero);
        r = this->s.op(c2, "bufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 68,
                       std::vector<V>{h, ix, iu});
    } else {
        V mask = this->s.m.i8(static_cast<std::uint8_t>((1u << count) - 1));
        V align = this->s.m.i32(4);
        if (index && used && used->stride == 1) {
            // A raw buffer (stride 1) read by index: the index is the byte.
            V byte = this->s.m.bin(c2, Bin::Add, *index, off);
            V h = this->handle(site, *src, false, {11 | u, 0});
            r = this->s.op(c2, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                           std::vector<V>{h, byte, iu, mask, align});
        } else if (index) {
            const std::uint32_t stride = used ? used->stride : 0;
            if (stride < 4 || stride % 4 != 0) fail("a structured buffer of unknown stride");
            V h = this->handle(site, *src, false, {12 | u, stride});
            r = this->s.op(c2, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                           std::vector<V>{h, *index, off, mask, align});
        } else {
            V h = this->handle(site, *src, false, {11 | u, 0});
            r = this->s.op(c2, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                           std::vector<V>{h, off, iu, mask, align});
        }
    }
    if (!data || data->kind != Operand::Kind::Vector) fail("a buffer result");
    const std::uint8_t reg = data->reg;
    if (d16) {
        // Two f16 channels per VGPR (an odd last high half 0); the low 16 bits of an integer
        // view's (see the stores above).
        std::optional<std::pair<V, V>> integer;
        if (typed) integer = this->integer_view(*src);
        V k16 = this->s.m.i32(16);
        const Ty i32t = this->s.t.i32;
        for (std::uint32_t k = 0; k < div_ceil(count, 2u); ++k) {
            std::optional<V> x;
            const std::uint32_t end = std::min(2 * k + 2, count);
            for (std::uint32_t ch = end; ch-- > 2 * k;) {
                V f = this->s.m.extract(c2, r, ch);
                V h = this->s.op(c2, "legacyF32ToF16", Attr::ReadNone, i32t, 130, std::vector<V>{f});
                if (integer) {
                    const V int_ = integer->first;
                    V bits = this->as_int(f);
                    V low = this->s.m.i32(0xFFFF);
                    low = this->s.m.bin(c2, Bin::And, bits, low);
                    h = this->s.m.select(c2, int_, low, h);
                }
                if (x) {
                    V hi = this->s.m.bin(c2, Bin::Shl, *x, k16);
                    x = this->s.m.bin(c2, Bin::Or, h, hi);
                } else {
                    x = h;
                }
            }
            if (!x) unreachable_here();  // expect("a channel")
            this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), *x, guard);
        }
        return;
    }
    for (std::uint32_t k = 0; k < count; ++k) {
        V x = this->s.m.extract(c2, r, k);
        x = this->as_int(x);
        this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + k)), x, guard);
    }
}

void Lower::export_(std::size_t b, std::size_t i) {
    const Export* e = nullptr;
    {
        auto it = this->exports.find(Site{b, i});
        if (it == this->exports.end()) fail("an export I/O raising did not see");
        e = it->second;
    }
    if (e->compressed) fail("a compressed export");
    const std::optional<V> guard = this->guard(b, i);
    const bool mesh = this->s.mesh.has_value();
    const std::size_t c = this->cur;
    if (guard && !mesh) {
        const V g = *guard;
        if (!(e->done && this->s.pixel)) fail("a conditional export");
        V t = this->s.m.bool_(true);
        V kill = this->s.m.bin(c, Bin::Xor, g, t);
        this->s.op(c, "discard", Attr::NoUnwind, this->s.t.void_, 82, std::vector<V>{kill});
    }
    const Target& target = e->target;
    if (mesh && target.kind == Target::Kind::Primitive) {
        this->emit_indices(*e, guard);
        return;
    }
    auto param_row = [&](const Usage& usage, std::uint8_t row) -> std::optional<std::uint8_t> {
        // The 2020 compiler leaves a mesh program's first entry zeroed (usage "position"): its row
        // is the one no other entry names.
        if (usage_is_other0(usage) && mesh) return this->unnamed_row();
        return row;
    };
    // (output element, channel) per exported channel; per-primitive outputs are elements of
    // `prims`.
    std::vector<std::tuple<std::size_t, std::uint8_t, Value, std::optional<bool>>> stores;
    const bool primitive = target.kind == Target::Kind::PrimitiveParam;
    const std::vector<SigElement>& outputs = primitive ? this->s.prims : this->s.outputs;
    auto by_kind = [&](std::uint8_t kind) -> std::optional<std::size_t> {
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            if (outputs[k].kind == kind) return k;
        }
        return std::nullopt;
    };
    // 16-bit vertex outputs: the halves each dword carries, as (output element, channel, high
    // half) per channel.
    std::optional<std::pair<std::uint8_t, std::pair<bool, std::optional<std::uint8_t>>>> halves;
    if (target.kind == Target::Kind::Param) {
        if (auto r = param_row(target.usage, target.row)) {
            auto it = this->half_outputs_.find(*r);
            if (it != this->half_outputs_.end()) halves = std::pair{*r, it->second};
        }
    }
    if (halves) {
        const std::uint8_t row = halves->first;
        const bool pack2 = halves->second.first;
        // order: packed_layout sorts the packs.
        std::vector<std::pair<std::uint8_t, std::optional<std::uint8_t>>> packs;
        for (const auto& [r, h] : this->half_outputs_) {
            if (h.first) packs.emplace_back(r, h.second);
        }
        for (std::uint8_t ch = 0; ch < 4; ++ch) {
            const std::optional<Value> value = e->values[ch];
            if (!value) continue;
            for (bool high : {false, true}) {
                const std::uint8_t h = static_cast<std::uint8_t>(2 * ch + (high ? 1 : 0));
                std::optional<std::pair<std::uint8_t, std::uint8_t>> src;
                if (pack2) {
                    src = packed_half(outputs, packs, row, h);
                } else if (!high) {
                    src = std::pair{row, ch};
                }
                std::optional<std::size_t> found;
                if (src) {
                    const auto [r, cc] = *src;
                    for (std::size_t k = 0; k < outputs.size(); ++k) {
                        const auto& o = outputs[k];
                        if (o.row == std::optional<std::uint32_t>(r) && cc >= o.col &&
                            cc < o.col + o.cols) {
                            found = k;
                            break;
                        }
                    }
                }
                if (found) {
                    const std::uint8_t col = static_cast<std::uint8_t>(src->second - outputs[*found].col);
                    stores.emplace_back(*found, col, *value, std::optional<bool>(high));
                }
            }
        }
    }
    for (std::uint8_t ch = 0; ch < 4; ++ch) {
        const std::optional<Value> value = e->values[ch];
        if (!value) continue;
        if (halves) break;
        std::optional<std::pair<std::size_t, std::uint8_t>> found;
        switch (target.kind) {
        case Target::Kind::Color: {
            for (std::size_t k = 0; k < outputs.size(); ++k) {
                if (outputs[k].kind == 16 &&
                    outputs[k].row == std::optional<std::uint32_t>(target.n)) {
                    found = std::pair{k, ch};
                    break;
                }
            }
            break;
        }
        case Target::Kind::Position:
            if (target.n == 0) {
                if (auto id = by_kind(3)) found = std::pair{*id, ch};
            } else if (target.n >= 1 && target.n <= 2) {
                const std::uint8_t n = target.n;
                // shortcut: without the misc vector (point size, render target or viewport index)
                // pos1 and pos2 carry the clip then cull distances, in semantic order.
                for (const auto& o : outputs) {
                    if (o.kind == 4 || o.kind == 5) fail("render-target or viewport index exports");
                }
                std::vector<std::size_t> dists;
                for (std::size_t k = 0; k < outputs.size(); ++k) {
                    if (outputs[k].kind == 6 || outputs[k].kind == 7) dists.push_back(k);
                }
                std::stable_sort(dists.begin(), dists.end(), [&](std::size_t a, std::size_t bb) {
                    return std::pair{outputs[a].kind, outputs[a].index} <
                           std::pair{outputs[bb].kind, outputs[bb].index};
                });
                std::vector<std::pair<std::size_t, std::uint8_t>> flat;
                for (std::size_t k : dists) {
                    for (std::uint8_t cc = 0; cc < outputs[k].cols; ++cc) flat.emplace_back(k, cc);
                }
                // Channels past the distances the signature declares are ones the clipper does
                // not enable (`PA_CL_VS_OUT_CNTL`).
                const std::size_t k = static_cast<std::size_t>(n - 1) * 4 + ch;
                if (k < flat.size()) found = flat[k];
            } else {
                fail("clip, cull or point-size exports");
            }
            break;
        case Target::Kind::Depth:
            if (ch == 0) {
                auto id = by_kind(17);
                if (!id) id = by_kind(18);
                if (!id) id = by_kind(19);
                if (id) found = std::pair{*id, std::uint8_t{0}};
            } else if (ch == 1) {
                if (auto id = by_kind(20)) found = std::pair{*id, std::uint8_t{0}};
            } else if (ch == 2) {
                if (auto id = by_kind(14)) found = std::pair{*id, std::uint8_t{0}};
            }
            break;
        case Target::Kind::Param:
        case Target::Kind::PrimitiveParam: {
            const auto row = param_row(target.usage, target.row);
            for (std::size_t k = 0; k < outputs.size(); ++k) {
                const auto& o = outputs[k];
                if (row && o.row == std::optional<std::uint32_t>(*row) && ch >= o.col &&
                    ch < o.col + o.cols) {
                    found = std::pair{k, static_cast<std::uint8_t>(ch - o.col)};
                    break;
                }
            }
            break;
        }
        case Target::Kind::Null:
        case Target::Kind::Unread:
            break;
        case Target::Kind::Primitive:
            fail("a primitive export");
        }
        if (found) stores.emplace_back(found->first, found->second, *value, std::nullopt);
    }
    auto elements = [&](Lower& l, bool prim) -> std::vector<SigElement>& {
        return prim ? l.s.prims : l.s.outputs;
    };
    for (const auto& st : stores) {
        if (!std::get<3>(st).has_value() && elements(*this, primitive).at(std::get<0>(st)).half) {
            fail("a 16-bit output the stage data does not mark");
        }
    }
    // A mesh lane writes its own vertex or primitive, where it exports.
    std::optional<std::pair<std::size_t, std::size_t>> at;
    if (mesh) {
        V flat = this->mesh_thread();
        std::uint32_t limit = 0;
        if (this->s.mesh) limit = primitive ? this->s.mesh->max_primitives : this->s.mesh->max_vertices;
        const std::uint32_t threads = this->s.threads ? (*this->s.threads)[0] : UINT32_MAX;
        std::optional<V> mesh_guard;
        if (limit > 0 && threads > limit) {
            V lim = this->s.m.i32(limit);
            V in_bounds = this->s.m.cmp(this->cur, pred::IULT, flat, lim);
            if (guard) {
                mesh_guard = this->s.m.bin(this->cur, Bin::And, *guard, in_bounds);
            } else {
                mesh_guard = in_bounds;
            }
        } else {
            mesh_guard = guard;
        }
        at = this->begin_guarded(mesh_guard);
    }
    std::string_view name;
    std::uint32_t opcode = 0;
    if (!mesh) {
        name = "storeOutput", opcode = 5;
    } else if (!primitive) {
        name = "storeVertexOutput", opcode = 171;
    } else {
        name = "storePrimitiveOutput", opcode = 172;
    }
    for (const auto& [id, col, value, half] : stores) {
        V x = this->i32_of(value);
        if (half) {
            // An f16 half, declared a 32-bit float on the host.
            const std::size_t cc = this->cur;
            if (*half) {
                V k16 = this->s.m.i32(16);
                x = this->s.m.bin(cc, Bin::LShr, x, k16);
            }
            const Ty f32t = this->s.t.f32;
            V f = this->s.op(cc, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{x});
            x = this->as_int(f);
        }
        SigElement& el = elements(*this, primitive).at(id);
        el.usage = static_cast<std::uint8_t>(el.usage | (1u << col));
        const std::uint8_t comp = el.comp;
        std::string_view suffix;
        if (comp == 9 && half.has_value()) {
            suffix = ".f32", x = this->as_float(x);
        } else if (comp == 9) {
            auto it = this->floats.find(value);
            V f = it != this->floats.end() ? it->second : this->as_float(x);
            suffix = ".f32", x = f;
        } else {
            suffix = ".i32";
        }
        V a0 = this->s.m.i32(static_cast<std::uint32_t>(id));
        V a1 = this->s.m.i32(0);
        V a2 = this->s.m.i8(col);
        std::vector<V> args{a0, a1, a2, x};
        if (mesh) args.push_back(this->mesh_thread());
        const Ty void_t = this->s.t.void_;
        std::string full = std::string(name) + std::string(suffix);
        this->s.op(this->cur, full, Attr::NoUnwind, void_t, opcode, args);
    }
    this->end_guarded(at, {});
}

// Mesh: `exp prim` as the lane's primitive. Each corner takes ten bits: a 9-bit vertex index
// and, above it, its edge flag (bits 9, 19 and 29), which only the console's edge-flag
// rasterisation reads and which is no part of the index. Bit 31 culls it (all indices 0: no
// area).
void Lower::emit_indices(const Export& e, std::optional<V> guard) {
    Value v = ok_or(e.values[0], "a primitive export without its indices");
    V x = this->i32_of(v);
    V flat = this->mesh_thread();
    if (this->s.mesh) {
        const Mesh mesh = *this->s.mesh;
        const std::uint32_t threads = this->s.threads ? (*this->s.threads)[0] : UINT32_MAX;
        if (threads > mesh.max_primitives) {
            V p = this->s.m.i32(mesh.max_primitives);
            V in_bounds = this->s.m.cmp(this->cur, pred::IULT, flat, p);
            if (guard) {
                guard = this->s.m.bin(this->cur, Bin::And, *guard, in_bounds);
            } else {
                guard = in_bounds;
            }
        }
    }
    auto at = this->begin_guarded(guard);
    const std::size_t c = this->cur;
    V z = this->s.m.i32(0);
    V culled = this->s.m.cmp(c, pred::SLT, x, z);
    // A primitive that names a vertex past the declared count, or one no lane writes (lane n
    // writes vertex n alone), is culled too: the host would read a vertex nothing set, at any
    // position.
    std::optional<std::uint32_t> written;
    if (this->s.mesh) {
        const std::uint32_t t = this->s.threads ? (*this->s.threads)[0] : UINT32_MAX;
        written = std::min(this->s.mesh->max_vertices, t);
    }
    std::array<V, 3> indices = {z, z, z};
    for (std::size_t k = 0; k < indices.size(); ++k) {
        V shift = this->s.m.i32(10 * static_cast<std::uint32_t>(k));
        V mask = this->s.m.i32(0x1FF);
        V idx = this->s.m.bin(c, Bin::LShr, x, shift);
        idx = this->s.m.bin(c, Bin::And, idx, mask);
        if (written) {
            V n = this->s.m.i32(*written);
            V past = this->s.m.cmp(c, pred::IUGE, idx, n);
            culled = this->s.m.bin(c, Bin::Or, culled, past);
        }
        indices[k] = idx;
    }
    std::vector<V> args{flat};
    for (V idx : indices) args.push_back(this->s.m.select(c, culled, z, idx));
    const Ty void_t = this->s.t.void_;
    this->s.op(c, "emitIndices", Attr::NoUnwind, void_t, 169, args);
    this->end_guarded(at, {});
}

// The per-vertex arbitrary row no mesh output entry names but one (the 2020 compiler's zeroed
// entry).
std::optional<std::uint8_t> Lower::unnamed_row() const {
    const auto* ms = std::get_if<native::MeshIo>(&this->x.native->io);
    if (!ms) return std::nullopt;
    auto named = [&](std::uint32_t r) {
        for (const auto& o : ms->outputs) {
            if ((o.param & 0x80) == 0 && !usage_is_other0(o.usage) &&
                static_cast<std::uint32_t>(o.row) == r) {
                return true;
            }
        }
        return false;
    };
    std::set<std::uint32_t> free;
    for (const auto& o : this->s.outputs) {
        if (o.kind != 0 || !o.row) continue;
        if (!named(*o.row)) free.insert(*o.row);
    }
    if (free.size() == 1) {
        const std::uint32_t r = *free.begin();
        if (r <= 0xFF) return static_cast<std::uint8_t>(r);
        return std::nullopt;
    }
    return std::nullopt;
}

void Lower::exit(std::size_t b) {
    const auto& blk = this->x.p->blocks.at(b);
    const std::size_t offset = blk.insts.empty() ? blk.offset : blk.insts.back().machine.offset;
    const LaneExit& ex = this->x.st->exits.at(b);
    Term term = Term::ret();
    switch (ex.kind) {
    case LaneExit::Kind::Jump: {
        const std::size_t t = ex.target;
        if (this->counters.contains(t) && this->rpo.at(t) <= this->rpo.at(b)) {
            V ok_ = this->count_iteration(t, pred::IULT);
            const std::size_t abort_ = this->abort_block();
            term = Term::cond_br(ok_, this->block.at(t).value(), abort_);
        } else {
            this->enter_loop(b, t, this->cur);
            term = Term::br(this->block.at(t).value());
        }
        break;
    }
    case LaneExit::Kind::Branch: {
        const Bool when = ex.when;
        std::size_t taken = ex.taken;
        std::size_t not_taken = ex.not_taken;
        // A negated condition branches the other way round.
        V cond{};
        const auto node = this->x.st->bools.node(when);
        try {
            if (this->x.st->wide.contains(b)) {
                cond = this->wave_wide(b, when);
            } else if (node && std::get<1>(*node) == Bool::True && std::get<2>(*node) == Bool::False) {
                cond = this->atom(std::get<0>(*node), this->cur);
                std::swap(taken, not_taken);
            } else {
                cond = this->boolean(when, this->cur);
            }
        } catch (const WhyFail& w) {
            unsupported(offset, w.why);
        }
        // A back edge is taken only below the iteration limit.
        auto back = [&](const Lower& l, std::size_t to) {
            return l.counters.contains(to) && l.rpo.at(to) <= l.rpo.at(b);
        };
        // (No `xor` of a compare: a loop exit testing one hung the AMD driver.)
        if (back(*this, taken)) {
            V ok_ = this->count_iteration(taken, pred::IULT);
            cond = this->s.m.bin(this->cur, Bin::And, cond, ok_);
        } else if (back(*this, not_taken)) {
            V stop = this->count_iteration(not_taken, pred::IUGE);
            cond = this->s.m.bin(this->cur, Bin::Or, cond, stop);
        }
        // Critical edges get a block of their own, so loops have dedicated exits and preheaders
        // as DXC emits them.
        const std::size_t t = this->edge(b, taken);
        const std::size_t f = this->edge(b, not_taken);
        for (auto [to, via] : {std::pair{taken, t}, std::pair{not_taken, f}}) {
            if (!back(*this, to)) {
                // From an edge block, or from here.
                const std::size_t from =
                    std::optional<std::size_t>(via) == this->block.at(to) ? this->cur : via;
                this->enter_loop(b, to, from);
            }
        }
        term = Term::cond_br(cond, t, f);
        break;
    }
    case LaneExit::Kind::End:
        if (this->s.pixel && !this->exported.at(b) && !this->has_done_export(b)) {
            // A lane leaving without its final export was killed.
            V t = this->s.m.bool_(true);
            this->s.op(this->cur, "discard", Attr::NoUnwind, this->s.t.void_, 82, std::vector<V>{t});
        }
        term = Term::ret();
        break;
    case LaneExit::Kind::Indirect:
        unsupported(offset, "an indirect jump");
    }
    this->s.m.terminate(this->cur, term);
}

// A forward edge into loop header `to` (from DXIL block `via`) starts its count at 0.
void Lower::enter_loop(std::size_t /*from*/, std::size_t to, std::size_t via) {
    auto it = this->counters.find(to);
    if (it != this->counters.end()) {
        V z = this->s.m.i32(0);
        it->second.second.emplace_back(z, via);
    }
}

// A back edge to header `to` (a back edge keeps its block, no critical-edge block): the count goes
// up; the result compares it with LOOP_LIMIT by `p` (below: go on; at or above: stop).
V Lower::count_iteration(std::size_t to, std::uint8_t p) {
    const std::size_t c = this->cur;
    const V phi = this->counters.at(to).first;
    V one = this->s.m.i32(1);
    V next = this->s.m.bin(c, Bin::Add, phi, one);
    this->s.m.tag(phi, Tag::Counter);
    this->s.m.tag(next, Tag::Counter);
    // Per stage unless the switch names one: a loop that never meets its exit (a count read from
    // a buffer the host has not got yet, say) runs to the limit in every lane. A million
    // iterations a pixel hung the GPU for seconds a draw until Windows reset it
    // shortcut: a pixel loop past 128 iterations, a vertex or mesh loop past 256 or a compute loop
    // past 4,096 is cut short. (They were 1,024, 4,096 and 65,536: the opening cutscene then lost
    // the card 14-25 frames in, in a minimap or shadow mesh draw; with every loop cut at 64 it ran
    // 90 frames. Raise them again once shaders stop reading data that is not there yet.)
    const std::uint32_t named = loop_limit();
    std::uint32_t limit = named;
    if (named == LOOP_LIMIT) {
        if (this->s.pixel) {
            limit = 1u << 7;
        } else if (this->s.mesh.has_value() || !this->s.threads.has_value()) {
            limit = 1u << 8;
        } else {
            limit = 1u << 12;
        }
    }
    V lim = this->s.m.i32(limit);
    this->counters.at(to).second.emplace_back(next, c);
    V over = this->s.m.cmp(c, p, next, lim);
    this->s.m.tag(over, Tag::Counter);
    return over;
}

// The block a lane stopped by LOOP_LIMIT leaves through.
std::size_t Lower::abort_block() {
    if (this->abort) return *this->abort;
    const std::size_t a = this->s.m.new_block();
    if (this->s.pixel) {
        V t = this->s.m.bool_(true);
        this->s.op(a, "discard", Attr::NoUnwind, this->s.t.void_, 82, std::vector<V>{t});
    }
    this->s.m.terminate(a, Term::ret());
    this->abort = a;
    return a;
}

// The DXIL block a branch from IR block `from` to `to` goes to.
std::size_t Lower::edge(std::size_t from, std::size_t to) {
    const std::size_t target = this->block.at(to).value();
    std::size_t preds = 0;
    for (std::size_t q : this->x.p->blocks.at(to).preds) {
        if (this->block.at(q).has_value()) ++preds;
    }
    // Back edges stay as they are: the exiting block is the latch.
    if (preds < 2 || this->rpo.at(to) <= this->rpo.at(from)) return target;
    const std::size_t e = this->s.m.new_block();
    this->s.m.terminate(e, Term::br(target));
    this->edges.insert_or_assign(std::pair{from, to}, e);
    return e;
}

void Lower::fill_phis() {
    const Program& p = *this->x.p;
    // Lane-mask joins first (what an edge brings may ask for more joins, of masks and of
    // numbers): each edge's mask as the structure tracked it, a constant or a scalar-built one
    // included.
    const Structured& st = *this->x.st;
    while (!this->pending_masks.empty()) {
        const auto [v, b, i] = this->pending_masks.back();
        this->pending_masks.pop_back();
        const auto& phi = p.blocks.at(b).phis.at(i);
        const std::size_t offset = p.blocks.at(b).offset;
        auto found = st.phi_masks.find(phi.value);
        if (found == st.phi_masks.end()) unsupported(offset, "a lane-mask join of an untracked mask");
        const auto& args = found->second;
        std::vector<std::pair<V, std::size_t>> incoming;
        const auto& preds = p.blocks.at(b).preds;
        for (std::size_t n = 0; n < preds.size(); ++n) {
            const std::size_t q = preds[n];
            if (!this->tail.at(q)) continue;
            std::size_t qb = *this->tail.at(q);
            if (auto e = this->edges.find(std::pair{q, b}); e != this->edges.end()) qb = e->second;
            V x{};
            try {
                x = this->boolean(args.at(n), qb);
            } catch (const WhyFail& w) {
                unsupported(offset, w.why);
            }
            incoming.emplace_back(x, qb);
        }
        this->s.m.set_phi(v, std::move(incoming));
    }
    // A join that only another join reads is read wherever that one is. Left unmarked, a value
    // without a DXIL form coming into it became zero without a word (the looked-up tint of the
    // material resolve programs, whose size computation had no value, was zero in every lane that
    // way).
    // shortcut: only with `X2W_SHADERCFG_STRICT_JOINS` set (the offline pack listing sets it to count
    // them): 40 programs that translate today stop by name with it (vehicle paint pixel shaders
    // at "the first half of an interpolation", foliage mesh shaders); until those are lowered
    // they keep drawing with that zero.
    static const bool strict = std::getenv("X2W_SHADERCFG_STRICT_JOINS") != nullptr;
    std::vector<Value> work;
    if (strict) {
        // order: the work list only grows read_phis; its final contents do not depend on order.
        work.assign(this->read_phis.begin(), this->read_phis.end());
    }
    while (!work.empty()) {
        const Value v = work.back();
        work.pop_back();
        auto at = this->phi_at.find(v);
        if (at == this->phi_at.end()) continue;
        const auto [b, i] = at->second;
        for (const Value a : p.blocks.at(b).phis.at(i).args) {
            if (this->phi_at.contains(a) && this->read_phis.insert(a).second) work.push_back(a);
        }
    }
    auto data_phis = std::move(this->data_phis);
    this->data_phis.clear();
    for (const auto& [v, b, i] : data_phis) {
        const auto& phi = p.blocks.at(b).phis.at(i);
        const bool bit = phi.reg == Reg::Scc;
        // A join of numbers with dead registers (`dead`): the paths that bring those never read
        // the result, so they bring zero. Not for a join that is itself a lane mask (a ballot
        // gives that one as data).
        bool numbers = !bit && !st.masks.contains(phi.value) && !st.highs.contains(phi.value);
        if (numbers) {
            bool any = false;
            for (const Value a : phi.args) {
                auto it = this->vals.find(a);
                if (it != this->vals.end() && it->second.is_i32()) {
                    any = true;
                    break;
                }
            }
            numbers = any;
        }
        std::vector<std::pair<V, std::size_t>> incoming;
        const auto& preds = p.blocks.at(b).preds;
        for (std::size_t n = 0; n < preds.size(); ++n) {
            const std::size_t q = preds[n];
            if (!this->tail.at(q)) continue;
            std::size_t qb = *this->tail.at(q);
            if (auto e = this->edges.find(std::pair{q, b}); e != this->edges.end()) qb = e->second;
            const Value arg = phi.args.at(n);
            if (this->read_phis.contains(phi.value) && !this->vals.contains(arg)) {
                // A failed lazy barycentric lookup leaves the argument unresolved.
                try {
                    (void)this->lazy_barycentric(arg);
                } catch (const WhyFail&) {
                }
                // A root parameter nothing read before the join (user data is loaded on first
                // use, in the entry block).
                if (!this->vals.contains(arg)) {
                    if (auto pr = this->params.find(arg); pr != this->params.end()) {
                        const auto [param, dword] = pr->second;
                        Val val = this->user(param, dword);
                        this->vals.insert_or_assign(arg, val);
                    }
                }
            }
            std::optional<Val> got;
            if (auto it = this->vals.find(arg); it != this->vals.end()) got = it->second;
            const bool none_or_missing = !got || got->is_missing();
            V x{};
            if ((got && got->is_i32() && !bit) || (got && got->is_bit() && bit)) {
                x = got->v;
            } else if (none_or_missing && !bit && numbers && [&] {
                           std::vector<Value> seen;
                           return this->dead(arg, seen);
                       }()) {
                x = this->s.m.i32(0);
            } else if (none_or_missing && !bit && this->read_phis.contains(phi.value) &&
                       this->mask_data == MaskData::Console &&
                       (st.masks.contains(arg) || st.highs.contains(arg))) {
                // A lane mask joined with numbers (a traversal that keeps its lanes' masks on a
                // stack with its node numbers): the mask comes as the number it is, the ballot of
                // its lanes where it is written.
                try {
                    x = this->mask_dword(arg);
                } catch (const WhyFail& w) {
                    throw StopRetry{phi.value, w.why};
                }
            } else if (none_or_missing && bit && this->read_phis.contains(phi.value) &&
                       this->mask_data == MaskData::Console && st.scc_some.contains(arg)) {
                // SCC of a mask operation ("some lane is left") joined with a scalar test: the
                // compiler branches around code that sets SCC and tests SCC after it, the skip
                // bringing "no lane" along. As data it is "some lane of the wave has the bit" -
                // right only if the wave took that skip together, as the console does: a lane
                // that skipped alone would bring "some lane" where the console has the skipped
                // code's SCC. So the branch must be one the wave takes together; else its region
                // is named for `emit`'s retry to widen it.
                const LaneExit& qe = st.exits.at(q);
                const bool per_lane = !st.wide.contains(q) && qe.kind == LaneExit::Kind::Branch &&
                                      !all_uniform(st, qe.when);
                if (per_lane) {
                    for (const std::size_t other : p.blocks.at(b).preds) {
                        if (other != q) need(other);
                    }
                    throw StopRetry{phi.value, LANE_MASK};
                }
                try {
                    x = this->scc_some(arg);
                } catch (const WhyFail& w) {
                    throw StopRetry{phi.value, w.why};
                }
            } else if (got && got->is_bit() && !bit && this->read_phis.contains(phi.value)) {
                throw StopRetry{phi.value, "an SCC joined with SGPR values"};
            } else if (got && got->is_missing() && this->read_phis.contains(phi.value)) {
                throw StopRetry{phi.value, got->why};
            } else if (this->read_phis.contains(phi.value)) {
                throw StopRetry{phi.value, "a join of mismatched values"};
            } else if (bit) {
                x = this->s.m.bool_(false);
            } else {
                x = this->s.m.i32(0);
            }
            incoming.emplace_back(x, qb);
        }
        this->s.m.set_phi(v, std::move(incoming));
    }
    // order: set_phi only fills phis made before; no value is made here.
    auto counters = std::move(this->counters);
    this->counters.clear();
    for (auto& [hdr, entry] : counters) {
        (void)hdr;
        this->s.m.set_phi(entry.first, std::move(entry.second));
    }
}

}  // namespace x2w::vs::lower
