// normalized-format quotients, d16 channels, and the descriptors' handles and heap indices (handle,
// raw_twin, the descriptor and sampler phis, the slot words of heap views, per-lane heap indices).
//
// resource::Source is vs-front's tagged struct (kind + per-kind fields, leaves()).
#include <algorithm>

#include "x2w_shader/emit/lower_impl.h"
#include "x2w_shader/spill.h"

namespace x2w::vs::lower {

namespace {

// the shape of resource::Source.
using SK = Source::Kind;
bool is(const Source& s, SK k) { return s.kind == k; }

// The root parameter `param` of the layout, none past its end (CODE).
const native::RootParam* layout_get(const NativeProgram& n, std::size_t param) {
    return param < n.layout.size() ? &n.layout[param] : nullptr;
}

bool root_is(const RootType& r, RootType::Kind k) { return r.kind == k; }

}  // namespace

V Lower::norm_quotient(V a, std::uint32_t w) {
    if (w == 11 || w == 12) {
        return this->exact_norm(a, w);
    }
    std::size_t c = this->cur;
    std::uint64_t d = (std::uint64_t{1} << w) - 1;
    std::uint64_t kk = (std::uint64_t{1} << 32) / d;
    std::uint64_t r = (std::uint64_t{1} << 32) % d;
    auto k = [](Lower& l, std::uint64_t v) { return l.s.m.i32(static_cast<std::uint32_t>(v)); };
    V one = k(*this, 1);
    V kv = k(*this, kk);
    V q = this->s.m.bin(c, Bin::Mul, a, kv);
    if (r != 1) {
        V rv = k(*this, r);
        V dv = k(*this, d);
        V t = this->s.m.bin(c, Bin::Mul, a, rv);
        V more = this->s.m.bin(c, Bin::UDiv, t, dv);
        q = this->s.m.bin(c, Bin::Add, q, more);
    }
    // The bits to drop (1 to 8) from the exponent of `q >> 8`, which converts exactly (fewer than
    // 25 bits); none when `q` fits.
    V k8 = k(*this, 8);
    V top = this->s.m.bin(c, Bin::LShr, q, k8);
    top = this->s.m.cast(c, Cast::UiToFp, top, this->s.t.f32);
    top = this->as_int(top);
    V k23 = k(*this, 23);
    V e = this->s.m.bin(c, Bin::LShr, top, k23);
    V bias = k(*this, 127 + 15);
    V shift = this->s.m.bin(c, Bin::Sub, e, bias);
    V zero = k(*this, 0);
    V b24 = k(*this, 1 << 24);
    V fits = this->s.m.cmp(c, pred::IULT, q, b24);
    shift = this->s.m.select(c, fits, zero, shift);
    // Rounded up when the highest dropped bit is set: `a / d` is never exact, so a dropped half is
    // more than half, never a tie.
    V up = this->s.m.bin(c, Bin::Shl, q, one);
    up = this->s.m.bin(c, Bin::LShr, up, shift);
    up = this->s.m.bin(c, Bin::And, up, one);
    V m = this->s.m.bin(c, Bin::LShr, q, shift);
    m = this->s.m.bin(c, Bin::Add, m, up);
    // At most 2^24: exact as a float, and so its scale by 2^(shift - 32).
    V f = this->s.m.cast(c, Cast::UiToFp, m, this->s.t.f32);
    V b95 = k(*this, 127 - 32);
    V scale = this->s.m.bin(c, Bin::Add, shift, b95);
    scale = this->s.m.bin(c, Bin::Shl, scale, k23);
    scale = this->as_float(scale);
    f = this->s.m.bin(c, Bin::Mul, f, scale);
    return this->as_int(f);
}

// norm_quotient in integers only, for any width up to 16: `Q` as there, 8 more bits when
// that leaves fewer than 25 (codes of wide formats near 0), then rounded to 24 bits to nearest
// even with the remainder as sticky bit, and assembled as float bits.
V Lower::exact_norm(V a, std::uint32_t w) {
    std::size_t c = this->cur;
    std::uint64_t d = (std::uint64_t{1} << w) - 1;
    std::uint64_t kk = (std::uint64_t{1} << 32) / d;
    std::uint64_t r = (std::uint64_t{1} << 32) % d;
    auto k = [](Lower& l, std::uint64_t v) { return l.s.m.i32(static_cast<std::uint32_t>(v)); };
    V one = k(*this, 1);
    V zero = k(*this, 0);
    V dv = k(*this, d);
    V k8 = k(*this, 8);
    V kv = k(*this, kk);
    V q = this->s.m.bin(c, Bin::Mul, a, kv);
    V rem = a;
    if (r != 1) {
        V rv = k(*this, r);
        V t = this->s.m.bin(c, Bin::Mul, a, rv);
        V more = this->s.m.bin(c, Bin::UDiv, t, dv);
        q = this->s.m.bin(c, Bin::Add, q, more);
        rem = this->s.m.bin(c, Bin::URem, t, dv);
    }
    // The exponent's bias less the 32 (or 40) fraction bits of `q`; the shift below is added.
    V bias = k(*this, 127 + 23 - 32);
    if (kk < (std::uint64_t{1} << 24)) {
        // Fewer than 25 bits for the smallest codes: 8 more.
        V t = this->s.m.bin(c, Bin::Shl, rem, k8);
        V more = this->s.m.bin(c, Bin::UDiv, t, dv);
        V rest = this->s.m.bin(c, Bin::URem, t, dv);
        V wide = this->s.m.bin(c, Bin::Shl, q, k8);
        wide = this->s.m.bin(c, Bin::Or, wide, more);
        V limit = k(*this, 1 << 24);
        V small = this->s.m.cmp(c, pred::IULT, q, limit);
        q = this->s.m.select(c, small, wide, q);
        rem = this->s.m.select(c, small, rest, rem);
        V b40 = k(*this, 127 + 23 - 40);
        bias = this->s.m.select(c, small, b40, bias);
    }
    // Bits to drop: the bit length less 24, 1 to 8.
    V shift = one;
    for (std::uint32_t bit = 25; bit < 32; ++bit) {
        V at = k(*this, std::uint64_t{1} << bit);
        V ge = this->s.m.cmp(c, pred::IUGE, q, at);
        V add = this->s.m.select(c, ge, one, zero);
        shift = this->s.m.bin(c, Bin::Add, shift, add);
    }
    V mant = this->s.m.bin(c, Bin::LShr, q, shift);
    V below = this->s.m.bin(c, Bin::Sub, shift, one);
    V half = this->s.m.bin(c, Bin::LShr, q, below);
    half = this->s.m.bin(c, Bin::And, half, one);
    V low = this->s.m.bin(c, Bin::Shl, one, below);
    low = this->s.m.bin(c, Bin::Sub, low, one);
    low = this->s.m.bin(c, Bin::And, q, low);
    V rest = this->s.m.bin(c, Bin::Or, low, rem);
    V sticky = this->s.m.cmp(c, pred::NE, rest, zero);
    V odd = this->s.m.bin(c, Bin::And, mant, one);
    V tie = this->s.m.select(c, sticky, one, odd);
    V up = this->s.m.bin(c, Bin::And, half, tie);
    mant = this->s.m.bin(c, Bin::Add, mant, up);
    // Rounding up to 2^24 moves to the next exponent.
    V k24 = k(*this, 24);
    V carry = this->s.m.bin(c, Bin::LShr, mant, k24);
    mant = this->s.m.bin(c, Bin::LShr, mant, carry);
    V e = this->s.m.bin(c, Bin::Add, bias, shift);
    e = this->s.m.bin(c, Bin::Add, e, carry);
    V k23 = k(*this, 23);
    e = this->s.m.bin(c, Bin::Shl, e, k23);
    V fraction = k(*this, 0x7F'FFFF);
    V m = this->s.m.bin(c, Bin::And, mant, fraction);
    V bits = this->s.m.bin(c, Bin::Or, e, m);
    V is_zero = this->s.m.cmp(c, pred::EQ, a, zero);
    return this->s.m.select(c, is_zero, zero, bits);
}

// Channels of buffer format `format`, as format_load gives them, as d16 data: 16 bits
// each, two per dword, low half first (an odd last high half 0). Integer formats keep their low
// 16 bits; the others become f16.
std::vector<V> Lower::d16_channels(std::uint8_t format, std::vector<V> channels) {
    auto fmt = ok_or(buffer_format(format), "a buffer format");
    Numeric numeric = fmt.second;
    std::size_t c = this->cur;
    std::vector<V> halves;
    for (V x : channels) {
        if (numeric == Numeric::Uint || numeric == Numeric::Sint) {
            V m = this->s.m.i32(0xFFFF);
            halves.push_back(this->s.m.bin(c, Bin::And, x, m));
        } else {
            V f = this->as_float(x);
            Ty i32t = this->s.t.i32;
            halves.push_back(this->s.op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, std::vector<V>{f}));
        }
    }
    std::vector<V> out;
    for (std::size_t n = 0; n < halves.size(); n += 2) {
        if (n + 1 < halves.size()) {
            V lo = halves[n];
            V k16 = this->s.m.i32(16);
            V hi = this->s.m.bin(c, Bin::Shl, halves[n + 1], k16);
            out.push_back(this->s.m.bin(c, Bin::Or, lo, hi));
        } else {
            out.push_back(halves[n]);
        }
    }
    return out;
}

// The byte offset a buffer descriptor built around a root descriptor adds (constant and
// computed parts); none for other descriptors.
std::pair<std::uint32_t, std::optional<V>> Lower::built_offset(const Source& src) {
    if (is(src, SK::Built)) {
        const Address& address = src.address;
        std::optional<V> x;
        if (address.dynamic) {
            x = this->i32_of(*address.dynamic);
        }
        if (auto bias = this->root_bias(address.param)) {
            if (x) {
                x = this->s.m.bin(this->cur, Bin::Add, *x, *bias);
            } else {
                x = *bias;
            }
        }
        return {address.offset, x};
    }
    return {0, std::nullopt};
}

// For a root view the host binds as a raw view that may begin before the title's address (USER_VIEW_BIAS):
// the bytes to add to offsets through it. Every root UAV (root SRVs keep the views they have).
std::optional<V> Lower::root_bias(std::size_t param) {
    const native::RootParam* p = layout_get(*this->x.native, param);
    if (!p) return std::nullopt;
    native::RootParam rp = *p;
    // (Title memory by address where the user-data table is spilled: root_memory.)
    if (!root_is(rp.root, RootType::Kind::Descriptor) || spill::is_spilled(*this->x.native)) {
        return std::nullopt;
    }
    std::optional<ResourceType> kind;
    for (const UsedResource& u : this->x.native->used) {
        if (u.user_data == rp.user_data) {
            kind = u.kind;
            break;
        }
    }
    if (kind && kind->kind == ResourceType::Kind::Uav) {
        return this->host_word(RootType::Other(USER_VIEW_BIAS), rp.slot, 0);
    }
    return std::nullopt;
}

// The used-resource entry a descriptor loaded from a table falls in.
const UsedResource* Lower::used_kind(const Source& src) const {
  
    if (is(src, SK::Phi) || is(src, SK::Select) || is(src, SK::Lanes)) {
        if (src.args.empty()) return nullptr;
        return this->used_kind(src.args.front());
    }
    if (is(src, SK::LanePhi)) {
        for (const auto& a : src.lane_args) {
            if (!a) continue;
            if (const UsedResource* u = this->used_kind(*a)) return u;
        }
        return nullptr;
    }
    if (is(src, SK::Built)) {
        const native::RootParam* rp = layout_get(*this->x.native, src.address.param);
        if (!rp) return nullptr;
        for (const UsedResource& u : this->x.native->used) {
            if (root_is(u.root, RootType::Kind::Descriptor) && u.user_data == rp->user_data) return &u;
        }
        return nullptr;
    }
    if (!is(src, SK::Loaded) && !is(src, SK::Fetched)) {
        return nullptr;
    }
    auto ai = this->scalar.find(src.load);
    if (ai == this->scalar.end()) return nullptr;
    const Address& a = ai->second;
    const native::RootParam* rp = layout_get(*this->x.native, a.param);
    if (!rp) return nullptr;
    if (!root_is(rp->root, RootType::Kind::Table)) {
        return nullptr;
    }
    std::uint32_t dword = (a.offset / 4) + std::uint32_t{src.first};
    // For equal maximum keys, prefer the last entry.
    const UsedResource* best = nullptr;
    for (const UsedResource& u : this->x.native->used) {
        if (root_is(u.root, RootType::Kind::Table) && u.user_data == rp->user_data &&
            u.table_offset_dwords <= dword) {
            if (!best || u.table_offset_dwords >= best->table_offset_dwords) best = &u;
        }
    }
    return best;
}

// A handle for the descriptor `src` read at `site`.
V Lower::handle(Site site, const Source& src, bool sampler, std::pair<std::uint32_t, std::uint32_t> props) {
    switch (src.kind) {
    case SK::Loaded: {
        auto key = std::make_tuple(this->cur, src.load, src.first, sampler, props);
        if (auto it = this->handles.find(key); it != this->handles.end()) {
            return it->second;
        }
        V index = this->heap_index(src.load, src.first, sampler);
        auto [twin, bounded] = this->raw_twin(src, index, sampler, props);
        // A waterfall loop's load at the first lane's offset reads each lane's own entry (see
        // `resource::Waterfall`).
        const auto& loads = this->x.res->waterfall.loads;
        bool per_lane = std::find(loads.begin(), loads.end(), src.load) != loads.end();
        V h = this->s.heap_handle_at(this->cur, twin, sampler, per_lane, props, bounded);
        this->handles.insert_or_assign(key, h);
        return h;
    }
    case SK::Constant: {
        if (sampler) {
            std::uint32_t n = this->register_sampler(src.constant, (props.first & 0x8000) != 0);
            return this->s.sampler_handle(this->cur, n);
        }
        fail("a constant resource descriptor");
    }
    case SK::Built: {
        if (this->root_copy) {
            return *this->root_copy;
        }
        const native::RootParam* p = layout_get(*this->x.native, src.address.param);
        if (!p) fail("a descriptor built around a table in the code");
        native::RootParam rp = *p;
        std::optional<ResourceType> kind;
        for (const UsedResource& u : this->x.native->used) {
            if (u.user_data == rp.user_data) {
                kind = u.kind;
                break;
            }
        }
        bool desc = root_is(rp.root, RootType::Kind::Descriptor);
        auto kind_is = [&](ResourceType::Kind k) { return kind && kind->kind == k; };
        if (desc && (kind_is(ResourceType::Kind::Cbv) || !kind) && props.first == 13) {
            return this->s.root_cbv_handle(rp.slot);
        }
        if (desc && kind_is(ResourceType::Kind::Srv) && props.first == 11) {
            return this->s.root_srv_handle(rp.slot);
        }
        // A root UAV: a raw buffer whether the access reads (11) or writes (11 with the UAV bit).
        // Indexed accesses reach here as raw ones at index times the entry's stride (`buffer`).
        if (desc && kind_is(ResourceType::Kind::Uav) && (props.first == 11 || props.first == 0x100b)) {
            return this->s.root_uav_handle(rp.slot);
        }
        if (desc && kind_is(ResourceType::Kind::Srv) && props.first == 0x100b) {
            fail("a write through a root SRV");
        }
        fail("a typed root view");
    }
    case SK::Phi: {
        (void)site;
        V phi = this->descriptor_phi(src.block, src.args, sampler, 8);
        auto [index, bounded] = this->raw_twin(src, phi, sampler, props);
        return this->s.heap_handle_at(this->cur, index, sampler, false, props, bounded);
    }
    case SK::Select: {
        // The heap index of each choice, picked by SCC (uniform).
        if (src.args.size() != 2 || !is(src.args[0], SK::Loaded) || !is(src.args[1], SK::Loaded)) {
            fail("a descriptor chosen by a condition");
        }
        Site a = src.args[0].load;
        std::uint8_t fa = src.args[0].first;
        Site b = src.args[1].load;
        std::uint8_t fb = src.args[1].first;
        Val g = this->get(src.scc);
        if (!g.is_bit()) fail("SCC without a value");
        V cond = g.v;
        V set = this->heap_index(a, fa, sampler);
        V clear = this->heap_index(b, fb, sampler);
        V index = this->s.m.select(this->cur, cond, set, clear);
        auto [twin, bounded] = this->raw_twin(src, index, sampler, props);
        return this->s.heap_handle_at(this->cur, twin, sampler, false, props, bounded);
    }
    case SK::Fetched:
    case SK::Lanes:
    case SK::LanePhi: {
        // A descriptor each lane has its own of (a waterfall loop reads it from the first active
        // lane): each lane's heap index, non-uniform.
        V index = this->lane_heap_index(src, sampler);
        auto [twin, bounded] = this->raw_twin(src, index, sampler, props);
        return this->s.heap_handle_at(this->cur, twin, sampler, true, props, bounded);
    }
    }
    std::abort();  // unreachable: every kind handled
}

// The heap index to read descriptor `src` at, `index` being the title's: for a buffer read
// as raw bytes, through a table or picked from the heap itself (or picked and read as a constant
// buffer), the host's raw twin of that slot (shader raw_views), whose reads end where the view
// does. Everything else (textures, typed buffers, tables' constant buffers) stays in place. And
// whether it is a twin's (inside the host's heap).
//
// Tables too: a view that does not begin on 16 bytes cannot be raw, and a structured view with an eight-byte stride may start at offset eight.
std::pair<V, bool> Lower::raw_twin(const Source& src, V index, bool sampler,
                                   std::pair<std::uint32_t, std::uint32_t> props) {
    // Raw and constant buffers (DXIL resource kinds 11 and 13).
    constexpr std::uint32_t RAW = 11;
    constexpr std::uint32_t CONSTANT = 13;
    std::uint32_t first = raw_views();
    auto picked = [&](const Source* leaf) {
        if (!is(*leaf, SK::Loaded) && !is(*leaf, SK::Fetched)) {
            return false;
        }
        auto ai = this->scalar.find(leaf->load);
        if (ai == this->scalar.end()) return false;
        const native::RootParam* p = layout_get(*this->x.native, ai->second.param);
        if (!p) return false;
        return root_is(p->root, RootType::Kind::ResourceHeap) ||
               (root_is(p->root, RootType::Kind::Table) && (props.first & 0xFF) == RAW);
    };
    std::uint32_t kind = props.first & 0xFF;
    if (first == 0 || sampler || !(kind == RAW || kind == CONSTANT)) {
        return {index, false};
    }
    std::vector<const Source*> leaves = src.leaves();
    if (!std::all_of(leaves.begin(), leaves.end(), picked)) {
        return {index, false};
    }
    std::size_t c = this->cur;
    V last = this->s.m.i32(first - 1);
    V firstv = this->s.m.i32(first);
    Ty i32t = this->s.t.i32;
    // (As `dx.op.binary` UMin, see `heap_handle`.)
    V clamped = this->s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, std::vector<V>{index, last});
    this->s.m.tag(clamped, Tag::Clamp);
    return {this->s.m.bin(c, Bin::Or, clamped, firstv), true};
}

// The heap index a descriptor phi picks. Each predecessor computes its heap index at its
// end and the join picks one with a phi; a predecessor whose descriptor is itself chosen by control
// flow (a phi of an earlier join, which dominates it) gives that join's phi, at most `depth` joins
// deep.
V Lower::descriptor_phi(std::size_t block, std::span<const Source> args, bool sampler, std::uint32_t depth) {
    constexpr Why WHY = "a descriptor chosen by control flow";
    const Program& p = *this->x.p;
    std::vector<std::pair<V, std::size_t>> incoming;
    const auto& preds = p.blocks.at(block).preds;
    for (std::size_t n = 0; n < preds.size() && n < args.size(); ++n) {
        std::size_t q = preds[n];
        const Source& arg = args[n];
        if (!this->block.at(q)) {
            continue;
        }
        std::size_t tail = ok_or(this->tail.at(q), WHY);
        auto ei = this->edges.find({q, block});
        std::size_t at = ei != this->edges.end() ? ei->second : tail;
        V index;
        if (is(arg, SK::Loaded)) {
            std::size_t here = std::exchange(this->cur, at);
            try {
                index = this->heap_index(arg.load, arg.first, sampler);
            } catch (const WhyFail&) {
                this->cur = here;
                throw;
            }
            this->cur = here;
        } else if (is(arg, SK::Phi) && depth > 0) {
            index = this->descriptor_phi(arg.block, arg.args, sampler, depth - 1);
        } else {
            fail(WHY);
        }
        incoming.emplace_back(index, at);
    }
    std::size_t join = ok_or(this->block.at(block), WHY);
    V phi = this->s.m.phi(join, this->s.t.i32);
    this->s.m.set_phi(phi, incoming);
    return phi;
}

// Registers an immediate sampler constant and returns its slot index: a constant used with
// and without comparison takes one slot for each.
std::uint32_t Lower::register_sampler(std::span<const std::uint32_t> words_in, bool cmp) {
    if (words_in.size() < 4) fail("a sampler constant");
    std::array<std::uint32_t, 4> words{words_in[0], words_in[1], words_in[2], words_in[3]};
    std::optional<std::size_t> slot;
    for (std::size_t n = 0; n < this->samplers.size(); ++n) {
        // (n < 64: at most 64 slots are ever pushed)
        if (this->samplers[n] == words && (((this->s.cmp_samplers >> n) & 1) != 0) == cmp) {
            slot = n;
            break;
        }
    }
    std::size_t n;
    if (slot) {
        n = *slot;
    } else {
        n = this->samplers.size();
        if (n >= 64) {
            fail("more than 64 immediate sampler slots");
        }
        this->samplers.push_back(words);
        this->s.cmp_samplers |= std::uint64_t{cmp} << n;
    }
    return static_cast<std::uint32_t>(n);
}

// Resolves a control-flow phi of immediate sampler constants into either a single static
// sampler register (`One`) or a uniform `i32` phi index over `regs` (`Many`) for branch dispatch at
// the sample site.
std::pair<std::optional<V>, std::vector<std::uint32_t>> Lower::sampler_phi_choice(
    std::size_t block, std::span<const Source> args, bool cmp, std::uint32_t depth) {
    constexpr Why WHY = "a descriptor chosen by control flow";
    std::vector<std::array<std::uint32_t, 4>> distinct;
    this->collect_constant_samplers(block, args, depth, distinct);
    if (distinct.empty()) {
        fail(WHY);
    }
    std::vector<std::uint32_t> regs;
    regs.reserve(distinct.size());
    for (const auto& w : distinct) {
        regs.push_back(this->register_sampler(w, cmp));
    }
    if (regs.size() == 1) {
        return {std::nullopt, regs};
    }
    V idx = this->sampler_index_phi(block, args, 0, distinct, depth);
    return {idx, regs};
}

void Lower::collect_constant_samplers(std::size_t block, std::span<const Source> args, std::uint32_t depth,
                                      std::vector<std::array<std::uint32_t, 4>>& distinct) const {
    constexpr Why WHY = "a descriptor chosen by control flow";
    const Program& p = *this->x.p;
    const auto& preds = p.blocks.at(block).preds;
    for (std::size_t n = 0; n < preds.size() && n < args.size(); ++n) {
        std::size_t q = preds[n];
        const Source& arg = args[n];
        if (!this->block.at(q)) {
            continue;
        }
        if (is(arg, SK::Constant) && arg.constant.size() >= 4) {
            std::array<std::uint32_t, 4> k{arg.constant[0], arg.constant[1], arg.constant[2], arg.constant[3]};
            if (std::find(distinct.begin(), distinct.end(), k) == distinct.end()) {
                distinct.push_back(k);
            }
        } else if (is(arg, SK::Phi) && depth > 0) {
            this->collect_constant_samplers(arg.block, arg.args, depth - 1, distinct);
        } else {
            fail(WHY);
        }
    }
}

V Lower::sampler_index_phi(std::size_t block, std::span<const Source> args, std::uint32_t lower,
                           std::span<const std::array<std::uint32_t, 4>> distinct, std::uint32_t depth) {
    constexpr Why WHY = "a descriptor chosen by control flow";
    const Program& p = *this->x.p;
    std::vector<std::pair<V, std::size_t>> incoming;
    const auto& preds = p.blocks.at(block).preds;
    for (std::size_t n = 0; n < preds.size() && n < args.size(); ++n) {
        std::size_t q = preds[n];
        const Source& arg = args[n];
        if (!this->block.at(q)) {
            continue;
        }
        std::size_t tail = ok_or(this->tail.at(q), WHY);
        auto ei = this->edges.find({q, block});
        std::size_t at = ei != this->edges.end() ? ei->second : tail;
        V idx;
        if (is(arg, SK::Constant) && arg.constant.size() >= 4) {
            std::array<std::uint32_t, 4> k{arg.constant[0], arg.constant[1], arg.constant[2], arg.constant[3]};
            auto it = std::find(distinct.begin(), distinct.end(), k);
            if (it == distinct.end()) fail(WHY);
            auto pos = static_cast<std::uint32_t>(it - distinct.begin());
            idx = this->s.m.i32(lower + pos);
        } else if (is(arg, SK::Phi) && depth > 0) {
            idx = this->sampler_index_phi(arg.block, arg.args, lower, distinct, depth - 1);
        } else {
            fail(WHY);
        }
        incoming.emplace_back(idx, at);
    }
    std::size_t join = ok_or(this->block.at(block), WHY);
    V phi = this->s.m.phi(join, this->s.t.i32);
    this->s.m.set_phi(phi, incoming);
    return phi;
}

// The bytes per element of the buffer view `src` names, if the program picks it from the
// view heap at an index it computes (root type ResourceHeap): read from the host's table by heap
// slot in the current block.
std::optional<V> Lower::slot_stride(const Source& src) {
    std::optional<V> word = this->slot_word(src);
    if (!word) return std::nullopt;
    V bits = this->s.m.i32(VIEW_STRIDE_BITS);
    return this->s.m.bin(this->cur, Bin::And, *word, bits);
}

// Whether the typed buffer view `src` names is one of an integer format, and whether a
// signed one, as slot_stride (none: a view the program does not pick from the heap).
std::optional<std::pair<V, V>> Lower::integer_view(const Source& src) {
    std::optional<V> word = this->slot_word(src);
    if (!word) {
        return std::nullopt;
    }
    std::size_t c = this->cur;
    V zero = this->s.m.i32(0);
    auto test = [&](std::uint32_t bit) {
        V bv = this->s.m.i32(bit);
        V x = this->s.m.bin(c, Bin::And, *word, bv);
        return this->s.m.cmp(c, pred::NE, x, zero);
    };
    V integer = test(VIEW_INTEGER);
    V signed_ = test(VIEW_SIGNED);
    return std::pair<V, V>{integer, signed_};
}

// The host's word for the buffer view `src` names: its stride and format kind
// (shader SLOT_STRIDES).
std::optional<V> Lower::slot_word(const Source& src) {
    if (!is(src, SK::Loaded)) {
        return std::nullopt;
    }
    Site load = src.load;
    std::uint8_t first = src.first;
    bool heap = false;
    if (auto ai = this->scalar.find(load); ai != this->scalar.end()) {
        const native::RootParam* p = layout_get(*this->x.native, ai->second.param);
        heap = p && root_is(p->root, RootType::Kind::ResourceHeap);
    }
    if (!heap) {
        return std::nullopt;
    }
    auto key = std::make_tuple(this->cur, load, first);
    if (auto it = this->strides.find(key); it != this->strides.end()) {
        return it->second;
    }
    std::size_t c = this->cur;
    V slot = this->heap_index(load, first, false);
    // Inside the table whatever the index (lanes that do not run compute one too): a root view has
    // no bounds of its own.
    V last = this->s.m.i32(SLOT_STRIDES_COUNT - 1);
    Ty i32t = this->s.t.i32;
    slot = this->s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, std::vector<V>{slot, last});
    this->s.m.tag(slot, Tag::Clamp);
    V two = this->s.m.i32(2);
    V byte = this->s.m.bin(c, Bin::Shl, slot, two);
    V h = this->s.root_srv_handle(SLOT_STRIDES);
    V iu = this->s.m.undef(this->s.t.i32);
    V mask = this->s.m.i8(1);
    V align = this->s.m.i32(4);
    V r = this->s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, this->s.t.res_ret, 139,
                     std::vector<V>{h, byte, iu, mask, align});
    V f = this->s.m.extract(c, r, 0);
    V v = this->as_int(f);
    this->strides.insert_or_assign(key, v);
    return v;
}

// The heap index of a descriptor that may differ between lanes, computed in the current
// block (or, chosen by control flow, at the join).
V Lower::lane_heap_index(const Source& src, bool sampler) {
    switch (src.kind) {
    case SK::Loaded:
        return this->heap_index(src.load, src.first, sampler);
    case SK::Fetched:
        return this->fetched_index(src.load, src.first, sampler);
    case SK::Lanes:
    case SK::Select: {
        Value mask = is(src, SK::Lanes) ? src.mask : src.scc;
        if (src.args.size() != 2) {
            fail("a descriptor chosen per lane");
        }
        const Source& a = src.args[0];
        const Source& b = src.args[1];
        V cond;
        if (is(src, SK::Lanes)) {
            auto mi = this->x.st->masks.find(mask);
            if (mi == this->x.st->masks.end()) fail("a descriptor chosen per lane by an untracked mask");
            Bool m = mi->second;
            cond = this->boolean(m, this->cur);
        } else {
            Val g = this->get(mask);
            if (!g.is_bit()) fail("SCC without a value");
            cond = g.v;
        }
        V set = this->lane_heap_index(a, sampler);
        V clear = this->lane_heap_index(b, sampler);
        return this->s.m.select(this->cur, cond, set, clear);
    }
    case SK::LanePhi: {
        // Each predecessor computes its heap index at its end (what it reads is there: the dwords
        // are, on that edge); the join picks the one of the edge the lane came by. An edge no lane
        // reading it takes gives index 0.
        std::size_t block = src.block;
        std::size_t join = ok_or(this->block.at(block), "a per-lane descriptor chosen by control flow");
        std::vector<std::pair<V, std::size_t>> incoming;
        const auto& preds = this->x.p->blocks.at(block).preds;
        for (std::size_t n = 0; n < preds.size() && n < src.lane_args.size(); ++n) {
            std::size_t q = preds[n];
            const auto& arg = src.lane_args[n];
            if (!this->block.at(q)) {
                continue;
            }
            std::size_t tail = ok_or(this->tail.at(q), "a per-lane descriptor joined from a later block");
            auto ei = this->edges.find({q, block});
            std::size_t at = ei != this->edges.end() ? ei->second : tail;
            V index;
            if (arg) {
                std::size_t here = std::exchange(this->cur, at);
                try {
                    index = this->lane_heap_index(*arg, sampler);
                } catch (const WhyFail&) {
                    this->cur = here;
                    throw;
                }
                this->cur = here;
            } else {
                index = this->s.m.i32(0);
            }
            incoming.emplace_back(index, at);
        }
        V phi = this->s.m.phi(join, this->s.t.i32);
        this->s.m.set_phi(phi, incoming);
        return phi;
    }
    default:
        fail("a per-lane descriptor from neither a heap load nor a choice of them");
    }
}

// The heap index of the descriptor at dword `first` of what the global load at `load`
// reads, in each lane: its address (base plus offsets) over the descriptor size.
V Lower::fetched_index(Site load, std::uint8_t first, bool sampler) {
    constexpr Why WHY = "a per-lane descriptor load";
    auto ai = this->scalar.find(load);
    if (ai == this->scalar.end()) fail("an untracked descriptor load");
    Address a = ai->second;
    const native::RootParam* rp = layout_get(*this->x.native, a.param);
    if (!rp || !(root_is(rp->root, RootType::Kind::Table) || root_is(rp->root, RootType::Kind::ResourceHeap) ||
                 root_is(rp->root, RootType::Kind::SamplerHeap))) {
        fail("a descriptor outside the heaps");
    }
    const Inst& li = this->x.p->blocks.at(load.first).insts.at(load.second);
    const auto* fl = std::get_if<gfx10::Flat>(&li.fields);
    if (!fl || !fl->addr || fl->addr->kind != Operand::Kind::Vector) {
        fail(WHY);
    }
    std::uint8_t reg = fl->addr->reg;
    std::optional<Operand> saddr = fl->saddr;
    std::int16_t offset = fl->offset;
    std::size_t c = this->cur;
    // The low dword of the address: SGPR base plus VGPR offset, or the VGPR pair's low half (heap
    // addresses fit in 32 bits).
    V at = this->i32_of(ok_or(li.read(Reg::V(reg)), WHY));
    if (saddr && saddr->kind == Operand::Kind::Scalar && saddr->code != gfx10::NULL_CODE) {
        V base = this->i32_of(ok_or(li.read(Reg::S(saddr->code)), WHY));
        at = this->s.m.bin(c, Bin::Add, base, at);
    }
    V k = this->s.m.i32(static_cast<std::uint32_t>(static_cast<std::int32_t>(offset)) + 4 * std::uint32_t{first});
    at = this->s.m.bin(c, Bin::Add, at, k);
    V shift = this->s.m.i32(sampler ? 4 : 5);
    return this->s.m.bin(c, Bin::LShr, at, shift);
}

// The heap index of the descriptor at dword `first` of what the scalar load at `load`
// reads, computed in the current block.
V Lower::heap_index(Site load, std::uint8_t first, bool sampler) {
    auto ai = this->scalar.find(load);
    if (ai == this->scalar.end()) fail("an untracked descriptor load");
    Address a = ai->second;
    const native::RootParam* rp = layout_get(*this->x.native, a.param);
    if (!rp || !(root_is(rp->root, RootType::Kind::Table) || root_is(rp->root, RootType::Kind::ResourceHeap) ||
                 root_is(rp->root, RootType::Kind::SamplerHeap))) {
        fail("a descriptor outside the heaps");
    }
    const Inst& li = this->x.p->blocks.at(load.first).insts.at(load.second);
    const auto* sm = std::get_if<gfx10::Smem>(&li.fields);
    if (!sm) {
        fail("a descriptor load");
    }
    if (sm->base.kind != Operand::Kind::Scalar) {
        fail("a descriptor load");
    }
    std::uint8_t code = sm->base.code;
    std::size_t c = this->cur;
    Value base = ok_or(li.read(Reg::S(code)), "a descriptor load");
    V addr = this->i32_of(base);
    V k = this->s.m.i32(static_cast<std::uint32_t>(sm->offset) + 4 * std::uint32_t{first});
    addr = this->s.m.bin(c, Bin::Add, addr, k);
    if (!(sm->soffset.kind == Operand::Kind::Scalar && sm->soffset.code == gfx10::NULL_CODE)) {
        V so = this->operand(li, sm->soffset);
        addr = this->s.m.bin(c, Bin::Add, addr, so);
    }
    V shift = this->s.m.i32(sampler ? 4 : 5);
    return this->s.m.bin(c, Bin::LShr, addr, shift);
}

}  // namespace x2w::vs::lower
