// The free functions and per-process / per-thread settings of the lowering.
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include "x2w_shader/emit/lower_impl.h"
#include "x2w_shader/spill.h"

namespace x2w::vs::lower {

namespace {

// std::env::var_os(name) as text.
std::optional<std::string> env_var(const char* name) {
#ifdef _MSC_VER
#pragma warning(suppress : 4996)
#endif
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string(v);
}

// The scalar operands' and an instruction's mask test of lane_uniform / reads_mask_as_data.
bool is_mask_of(const Structured& st, Value v) {
    return st.masks.contains(v) || st.highs.contains(v);
}

}  // namespace

Lowered lower(const Parts& parts) {
    bool counted = [&] {
        // counted_in_threads: whether the title dispatches the program by a count of threads
        // whose last group may be partial: the structure builder's programs of one dimension.
        // Their lanes past the count must not run. Not where a group barrier would wait for lanes
        // that left, nor where two host waves of 32 stand for a console wave.
        const auto* cs = std::get_if<native::ComputeIo>(&parts.native->io);
        if (!cs) return false;
        if (!(cs->threads[0] > 1 && cs->threads[1] == 1 && cs->threads[2] == 1)) return false;
        if (!spill::is_spilled(*parts.native)) return false;
        if (!(!parts.native->wave64 || host_max_wave() >= 64)) return false;
        for (const auto& b : parts.p->blocks) {
            for (const auto& i : b.insts) {
                if (i.name() == "s_barrier") return false;
            }
        }
        return true;
    }();
    return lower_counted(parts, counted);
}

namespace {
// VS_DUMP=<dir> writes each lowered module to <dir>/<name>__<n>.dxil. Set the name with
// test::set_dump_name. Each name has its own counter.
thread_local std::string dump_name = "main";
thread_local std::uint32_t dump_n = 0;
void dump(const std::vector<std::uint8_t>& dxil) {
    const char* dir = std::getenv("VS_DUMP");
    if (!dir) return;
    std::string path = std::string(dir) + "/" + dump_name + "__" + std::to_string(dump_n++) + ".dxil";
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(dxil.data(), 1, dxil.size(), f);
        std::fclose(f);
    }
}
}  // namespace

void test::set_dump_name(std::string_view name) {
    dump_name.assign(name);
    dump_n = 0;
}

Lowered lower_counted(const Parts& parts, bool counted) {
    HashMap<Value, Why> bad;
    for (;;) {
        try {
            Lower l(parts, bad);
            Lowered out = l.run(counted);
            dump(out.dxil);
            return out;
        } catch (const StopRetry& r) {
            // the value was already missing once.
            bool had = bad.contains(r.value);
            bad.insert_or_assign(r.value, r.why);
            if (had) {
                throw ShaderError::unsupported(0, "a phi that does not settle");
            }
        }
        // Stop::Fail: the ShaderError passes through.
    }
}

// The instructions inside a pixel program's ordered section (primitive-ordered pixel
// shading, which the console compiler makes of rasterizer ordered views): from the write of
// POPS_PACKER, along every path, to the message that ends the section or the program's end. The
// console waits there until the waves holding earlier primitives' samples of the same pixels have
// left their sections; a host's rasterizer ordered views wait by themselves, at their first
// access. Empty for a program without such a section.
HashSet<Site> ordered_section(const Program& p) {
    auto starts = [](const Inst& i) {
        if (i.name() != "s_setreg_b32") return false;
        const auto* sop = std::get_if<gfx10::Sop>(&i.fields);
        return sop && sop->imm && (*sop->imm & 0x3F) == POPS_PACKER;
    };
    auto ends = [](const Inst& i) {
        if (i.name() != "s_sendmsg") return false;
        const auto* sopp = std::get_if<gfx10::Sopp>(&i.fields);
        return sopp && (sopp->imm & 0xF) == ORDERED_PS_DONE;
    };
    HashSet<Site> section;
    std::vector<bool> entered(p.blocks.size(), false);
    std::vector<Site> work;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block = p.blocks[b];
        for (std::size_t i = 0; i < block.insts.size(); ++i) {
            if (starts(block.insts[i])) work.push_back({b, i});
        }
    }
    while (!work.empty()) {
        auto [b, from] = work.back();
        work.pop_back();
        const auto& block = p.blocks[b];
        bool done = false;
        for (std::size_t i = from; i < block.insts.size(); ++i) {
            if (ends(block.insts[i])) {
                done = true;
                break;
            }
            section.insert({b, i});
        }
        if (done) continue;
        for (std::size_t t = 0; t < p.blocks.size(); ++t) {
            const auto& next = p.blocks[t];
            bool pred = std::find(next.preds.begin(), next.preds.end(), b) != next.preds.end();
            if (pred && !entered[t]) {  // !std::mem::replace(&mut entered[t], true)
                entered[t] = true;
                work.push_back({t, 0});
            }
        }
    }
    return section;
}

// The f16 bits of inline float constant `c` (240..=248: 0.5, -0.5, 1.0, -1.0, 2.0, -2.0,
// 4.0, -4.0, 1/(2 pi)).
std::uint32_t inline_f16(std::uint8_t c) {
    static constexpr std::array<std::uint32_t, 9> K = {0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000,
                                                       0xC000, 0x4400, 0xC400, 0x3118};
    return K.at(static_cast<std::size_t>(c - 240));
}

// A GFX10 buffer format (the unified format of typed buffer loads): the channel widths from the lowest bits
// up, and how the channels convert. Formats are numbered in families (`8`, `16`, `8_8`,...: AMD names list
// channels from the highest bits), each with its numeric formats in the order unorm, snorm, uscaled,
// sscaled, uint, sint, float.
std::optional<std::pair<std::span<const std::uint32_t>, Numeric>> buffer_format(std::uint8_t format) {
    using N = Numeric;
    static constexpr N NORM[] = {N::Unorm, N::Snorm, N::Uscaled, N::Sscaled, N::Uint, N::Sint};
    static constexpr N ALL[] = {N::Unorm, N::Snorm, N::Uscaled, N::Sscaled, N::Uint, N::Sint, N::Float};
    static constexpr N WIDE[] = {N::Uint, N::Sint, N::Float};
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10 buffer formats have no
    // 5_9_9_9. Channel order from Mesa (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and
    // 10_10_10_2 has its 2-bit channel in the lowest bits.
    static constexpr std::uint32_t B8[] = {8};
    static constexpr std::uint32_t B16[] = {16};
    static constexpr std::uint32_t B8_8[] = {8, 8};
    static constexpr std::uint32_t B32[] = {32};
    static constexpr std::uint32_t B16_16[] = {16, 16};
    static constexpr std::uint32_t B10_11_11[] = {11, 11, 10};  // 10_11_11
    static constexpr std::uint32_t B11_11_10[] = {10, 11, 11};  // 11_11_10
    static constexpr std::uint32_t B10_10_10_2[] = {2, 10, 10, 10};  // 10_10_10_2
    static constexpr std::uint32_t B2_10_10_10[] = {10, 10, 10, 2};  // 2_10_10_10
    static constexpr std::uint32_t B8_8_8_8[] = {8, 8, 8, 8};
    static constexpr std::uint32_t B32_32[] = {32, 32};
    static constexpr std::uint32_t B16x4[] = {16, 16, 16, 16};
    static constexpr std::uint32_t B32x3[] = {32, 32, 32};
    static constexpr std::uint32_t B32x4[] = {32, 32, 32, 32};
    struct Family {
        std::span<const std::uint32_t> bits;
        std::span<const N> numerics;
    };
    static const Family FAMILIES[14] = {
        {B8, NORM},          {B16, ALL},          {B8_8, NORM},        {B32, WIDE},
        {B16_16, ALL},       {B10_11_11, ALL},    {B11_11_10, ALL},    {B10_10_10_2, NORM},
        {B2_10_10_10, NORM}, {B8_8_8_8, NORM},    {B32_32, WIDE},      {B16x4, ALL},
        {B32x3, WIDE},       {B32x4, WIDE},
    };
    if (format == 0) return std::nullopt;  // format.checked_sub(1)?
    std::size_t n = static_cast<std::size_t>(format - 1);
    for (const auto& f : FAMILIES) {
        if (n < f.numerics.size()) {
            return std::pair{f.bits, f.numerics[n]};
        }
        n -= f.numerics.size();
    }
    return std::nullopt;
}

// How far a float channel of `width` bits (16, or the unsigned 11- and 10-bit floats of the
// packed formats) shifts up to be a half: the small floats are a half's exponent (5 bits, biased 15)
// and the top of its mantissa, without the sign, so `legacyF16ToF32` widens them exactly,
// denormals, infinities and NaNs included.
std::uint32_t half_shift(std::uint32_t width) {
    assert((width == 10 || width == 11 || width == 16) && "float width");
    return width == 16 ? 0 : 15 - width;
}

namespace {

// Every per-lane branch (by its block) with its region: the blocks reachable from it that do
// not post-dominate it, which only some lanes run. Not the branches the wave takes together
// (Structured::wide), nor those on scalar conditions. None with an indirect exit.
std::optional<std::vector<std::pair<std::size_t, std::vector<bool>>>> lane_regions(const Structured& st) {
    const std::size_t n = st.exits.size();
    std::vector<std::vector<std::size_t>> succ(n);
    for (std::size_t b = 0; b < n; ++b) {
        const auto& e = st.exits[b];
        switch (e.kind) {
            case LaneExit::Kind::Jump:
                succ[b].push_back(e.target);
                break;
            case LaneExit::Kind::Branch:
                succ[b].push_back(e.taken);
                succ[b].push_back(e.not_taken);
                break;
            case LaneExit::Kind::End:
                break;
            case LaneExit::Kind::Indirect:
                return std::nullopt;
        }
    }
    // post-dominators (bitsets; a virtual exit after every End)
    const std::size_t words = div_ceil<std::size_t>(n, 64);
    const std::vector<std::uint64_t> full(words, ~std::uint64_t{0});
    std::vector<std::vector<std::uint64_t>> pdom;
    pdom.reserve(n);
    for (std::size_t b = 0; b < n; ++b) {
        if (succ[b].empty()) {
            std::vector<std::uint64_t> v(words, 0);
            v[b / 64] |= std::uint64_t{1} << (b % 64);
            pdom.push_back(std::move(v));
        } else {
            pdom.push_back(full);
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t bb = n; bb-- > 0;) {
            const std::size_t b = bb;
            if (succ[b].empty()) continue;
            auto v = full;
            for (std::size_t t : succ[b]) {
                for (std::size_t w = 0; w < words; ++w) v[w] &= pdom[t][w];
            }
            v[b / 64] |= std::uint64_t{1} << (b % 64);
            if (v != pdom[b]) {
                pdom[b] = std::move(v);
                changed = true;
            }
        }
    }
    std::vector<std::pair<std::size_t, std::vector<bool>>> regions;
    for (std::size_t d = 0; d < n; ++d) {
        const auto& e = st.exits[d];
        if (e.kind != LaneExit::Kind::Branch) continue;
        bool uniform = true;
        for (const auto& a : st.bools.support(e.when)) {
            if (a.kind != Atom::Kind::Uniform) {
                uniform = false;
                break;
            }
        }
        if (st.wide.contains(d) || uniform) {
            continue;  // a wave-wide or uniform branch: all lanes go the same way
        }
        // blocks reachable from the lane branch that do not post-dominate it
        std::vector<bool> seen(n, false);
        std::vector<std::size_t> stack = succ[d];
        while (!stack.empty()) {
            std::size_t b = stack.back();
            stack.pop_back();
            if (seen[b] || ((pdom[d][b / 64] >> (b % 64)) & 1) == 1) continue;
            seen[b] = true;
            stack.insert(stack.end(), succ[b].begin(), succ[b].end());
        }
        regions.emplace_back(d, std::move(seen));
    }
    return regions;
}

}  // namespace

// Per IR block: every lane of the wave that started the program runs it – no per-lane
// branch separates it from the entry (it lies in no lane branch's region). None with an indirect
// exit (not followed).
std::optional<std::vector<bool>> all_lane_blocks(const Structured& st) {
    const std::size_t n = st.exits.size();
    std::vector<bool> all(n, true);
    auto regions = lane_regions(st);
    if (!regions) return std::nullopt;
    for (const auto& [_, region] : *regions) {
        for (std::size_t b = 0; b < n; ++b) {
            if (region[b]) all[b] = false;
        }
    }
    return all;
}

std::optional<std::vector<std::size_t>> branches_around(const Structured& st,
                                                        const std::vector<std::size_t>& needs) {
    std::vector<std::size_t> out;
    auto regions = lane_regions(st);
    if (!regions) return std::nullopt;
    for (const auto& [d, region] : *regions) {
        bool any = false;
        for (std::size_t b : needs) {
            if (b < region.size() && region[b]) {
                any = true;
                break;
            }
        }
        if (any) {
            // st.wave_any.get(d).copied().flatten()?
            if (d >= st.wave_any.size() || !st.wave_any[d].has_value()) return std::nullopt;
            out.push_back(d);
        }
    }
    return out;
}

bool needs_every_lane(std::string_view what) {
    return what.ends_with("where not every lane runs") || what == std::string_view(MASK_ELSEWHERE);
}

namespace {
// thread_local! { NEEDS, LOCAL_WAVES, LOCAL_ANY }
thread_local std::vector<std::size_t> NEEDS;
thread_local bool LOCAL_WAVES = false;
thread_local bool LOCAL_ANY = false;
thread_local bool FIRST_HALF_PAIRS = false;
// Per-thread test choices.
thread_local std::optional<bool> TEST_F16_NEAREST;
thread_local std::optional<std::uint32_t> TEST_RDNA;
thread_local std::optional<std::uint32_t> TEST_HOST_WAVE;
// Test title-memory mapping.
thread_local std::optional<std::pair<std::uint32_t, std::uint64_t>> TEST_MEMORY;

// The widest wave the host GPU runs (0: not known – no host-specific forms).
std::atomic<std::uint32_t> HOST_MAX_WAVE{0};
std::atomic<bool> HOST_RDNA{false};
}  // namespace

std::vector<std::size_t> take_needs() { return std::exchange(NEEDS, {}); }

void need(std::size_t block) {
    if (std::find(NEEDS.begin(), NEEDS.end(), block) == NEEDS.end()) NEEDS.push_back(block);
}

// VGPR values that are the same in every lane holding them (`all` is all_lane_blocks):
// written by an instruction every arriving lane runs (no guard), lane by lane, from scalars,
// constants and such values; or loaded from an address made of them. The compiler keeps
// wave-uniform data in VGPRs and fetches it with `v_readfirstlane`.
// shortcut: a value joined by a phi counts only where every predecessor is a block all lanes run
// (no lane branch can have split the arriving lanes); loop-carried values never do.
HashSet<Value> lane_uniform(const Program& p, const Structured& st, const std::vector<bool>& all) {
    auto is_mask = [&](Value v) { return is_mask_of(st, v); };
    HashSet<Value> same;
    for (std::size_t b : ir::reverse_postorder(p.blocks)) {
        const auto& block = p.blocks[b];
        bool whole = std::all_of(block.preds.begin(), block.preds.end(),
                                 [&](std::size_t q) { return q < all.size() && all[q]; });
        for (const auto& phi : block.phis) {
            if (phi.reg.is_v() && whole &&
                std::all_of(phi.args.begin(), phi.args.end(), [&](Value a) { return same.contains(a); })) {
                same.insert(phi.value);
            }
        }
        for (std::size_t i = 0; i < block.insts.size(); ++i) {
            const auto& inst = block.insts[i];
            auto name = inst.name();
            const auto* v = std::get_if<gfx10::Valu>(&inst.fields);
            if (!(v && !v->dpp.has_value() && !v->sdwa.has_value() && !name.starts_with("v_mbcnt") &&
                  !name.starts_with("v_cmpx") && name.find("lane") == std::string_view::npos)) {
                continue;
            }
            const std::vector<Operand> ops = v->src;
            if (st.guards[b][i].has_value()) {
                continue;  // lanes outside EXEC keep another value
            }
            bool alike = std::all_of(ops.begin(), ops.end(), [&](const Operand& op) {
                if (op.kind == Operand::Kind::Vector) {
                    // every dword of the operand the instruction reads from that register on
                    std::vector<Value> read;
                    for (const auto& r : inst.reads) {
                        if (r.first.is_v() && r.first.n >= op.reg) read.push_back(r.second);
                    }
                    return !read.empty() &&
                           std::all_of(read.begin(), read.end(), [&](Value x) { return same.contains(x); });
                }
                // a scalar is one value; a lane mask is not, unless a compare of such values made it
                if (op.kind == Operand::Kind::Scalar && op.code != gfx10::NULL_CODE) {
                    auto x = inst.read(Reg::S(op.code));
                    return !x || !is_mask(*x) || same.contains(*x);
                }
                return true;
            });
            if (alike) {
                for (const auto& w : inst.writes) {
                    if (w.first != Reg::Scc) same.insert(w.second);
                }
            }
        }
    }
    return same;
}

namespace {

// What a register value holds, for wave_local.
struct Wave {
    enum class Kind : std::uint8_t {
        // Nothing that depends on how the threads are divided into waves.
        Plain,
        // The number of the wave's first lane in a mask (`s_ff1`).
        First,
        // A reduction over the wave's lanes: a count, a sum, a minimum...
        Total,
        // A count or sum over the lanes below this one.
        Prefix,
        // Derived from what a first-lane atomic returned: the slots the wave reserved.
        Base,
        // A mix the idiom does not make.
        Mixed,
    };
    Kind kind = Kind::Plain;
    wave::Op op = wave::Op::AddU;  // Used only for Total. Other operation kinds retain the default.

    static constexpr Wave of(Kind k) { return Wave{k, wave::Op::AddU}; }
    static constexpr Wave total(wave::Op o) { return Wave{Kind::Total, o}; }
    constexpr bool operator==(const Wave&) const = default;
};
constexpr Wave PLAIN = Wave::of(Wave::Kind::Plain);
constexpr Wave FIRST_W = Wave::of(Wave::Kind::First);
constexpr Wave PREFIX_W = Wave::of(Wave::Kind::Prefix);
constexpr Wave BASE_W = Wave::of(Wave::Kind::Base);
constexpr Wave MIXED_W = Wave::of(Wave::Kind::Mixed);

}  // namespace

// Whether a console wave64 program may run with each host wave as a wave of its own
// (with_local_waves): its wave-wide values are used only as the allocation idiom uses them, which
// does not care how the threads are divided into waves. A first-lane number only picks the first
// lane; a count or sum only goes into that lane's atomic on a shared counter; a prefix only goes
// onto the base that atomic returned; the first lane does nothing else. Anything else done with
// them takes one wave to be the whole thread group (a tile's result, say) and is refused, with the
// offset and what was found. Lane masks read as numbers by other instructions and lane reads
// outside whole-wave sections are stopped by the lowering itself.
// shortcut: `v_readfirstlane` of a value that differs between lanes is taken as it is (this host
// wave's first lane), and a guarded write of a plain value over a wave-dependent one is the only
// mix of old and new lanes that is looked at.
std::optional<std::pair<std::size_t, Why>> wave_local(const Program& p, const Structured& st) {
    using wave::Op;
    using K = Wave::Kind;
    static constexpr Why FIRST = "a wave's first-lane number used as data";
    static constexpr Why TOTAL = "a count or sum over the wave used as data";
    static constexpr Why PREFIX = "a count or sum over the lanes below used without a reserved base";
    static constexpr Why MIXED = "a value that depends on the wave on some paths only";
    static constexpr Why EFFECT = "a write by the wave's first lane other than its reservation";
    using KindMap = HashMap<Value, Wave>;
    auto of = [](const KindMap& kind, std::optional<Value> v) -> Wave {
        if (!v) return PLAIN;
        auto it = kind.find(*v);
        return it == kind.end() ? PLAIN : it->second;
    };
    auto put = [](KindMap& kind, Value v, Wave k) {
        if (k == PLAIN) {
            kind.erase(v);
        } else {
            kind.insert_or_assign(v, k);
        }
    };
    auto wild = [](Wave k) { return !(k.kind == K::Plain || k.kind == K::Base); };
    auto why = [](Wave k) -> Why {
        switch (k.kind) {
            case K::First:
                return FIRST;
            case K::Total:
                return TOTAL;
            case K::Prefix:
                return PREFIX;
            default:
                return MIXED;
        }
    };
    // a mask (by its low dword) under which only a wave's first lane runs
    auto first_mask = [&](const HashSet<Value>& firsts, std::optional<Value> m) {
        if (!m) return false;
        auto it = st.masks.find(*m);
        if (it == st.masks.end()) return false;
        for (const auto& a : st.bools.support(it->second)) {
            if (a.kind == Atom::Kind::Lane && firsts.contains(a.value)) return true;
        }
        return false;
    };
    // One instruction outside a section: the kind of what it writes, and whether it is a first
    // lane's reservation. Err: throws WhyFail.
    auto step = [&](const Inst& inst, const KindMap& kind, HashSet<Value>& firsts) -> std::pair<Wave, bool> {
        auto name = inst.name();
        auto val = [&](const Operand& op) -> std::optional<Value> {
            if (op.kind == Operand::Kind::Vector) return inst.read(Reg::V(op.reg));
            if (op.kind == Operand::Kind::Scalar && op.code != gfx10::NULL_CODE) return inst.read(Reg::S(op.code));
            return std::nullopt;
        };
        std::vector<Operand> ops;
        if (const auto* v = std::get_if<gfx10::Valu>(&inst.fields)) {
            ops = v->src;
        } else if (const auto* s = std::get_if<gfx10::Sop>(&inst.fields)) {
            ops = s->src;
            if (name == "s_addk_i32" || name == "s_mulk_i32") {
                if (s->dst) ops.push_back(*s->dst);
            }
        } else if (const auto* bf = std::get_if<gfx10::Buffer>(&inst.fields);
                   bf && (name.find("_atomic_") != std::string_view::npos ||
                          name.find("_store_") != std::string_view::npos)) {
            if (bf->data) ops.push_back(*bf->data);
        }
        std::vector<Wave> kinds;
        for (const auto& o : ops) kinds.push_back(of(kind, val(o)));
        // Everything else it reads (addresses, descriptors, SCC, the high dwords of wide operands)
        // must not depend on the wave. Not EXEC, and not the old value of a VGPR it writes, which
        // only the lanes outside EXEC keep.
        std::vector<Value> named;
        for (const auto& o : ops) {
            if (auto v = val(o)) named.push_back(*v);
        }
        for (const auto& [r, v] : inst.reads) {
            bool kept = r.is_v() && std::any_of(inst.writes.begin(), inst.writes.end(),
                                                [&](const auto& w) { return w.first == r; });
            if ((r.is_s() && (r.n == ir::EXEC || r.n == ir::EXEC + 1)) || kept ||
                std::find(named.begin(), named.end(), v) != named.end()) {
                continue;
            }
            Wave k = of(kind, v);
            if (wild(k)) fail(why(k));
        }
        Wave w = MIXED_W;
        for (Wave k : kinds) {
            if (wild(k)) {
                w = k;
                break;
            }
        }
        auto sum = [](Wave k) { return k == Wave::total(Op::AddU); };
        auto is_mask = [&](const Operand& o) {
            auto v = val(o);
            return v && is_mask_of(st, *v);
        };
        bool reserved = false;
        bool any_wild = std::any_of(kinds.begin(), kinds.end(), wild);
        auto is = [&](std::initializer_list<std::string_view> names) {
            return std::find(names.begin(), names.end(), name) != names.end();
        };
        Wave out;
        if (is({"s_ff1_i32_b64", "s_ff1_i32_b32"}) && !ops.empty() && is_mask(ops[0])) {
            out = FIRST_W;
        } else if (is({"s_bcnt1_i32_b64", "s_bcnt1_i32_b32"}) && !ops.empty() && is_mask(ops[0])) {
            out = Wave::total(Op::AddU);
        } else if (is({"v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"}) && kinds.size() == 2) {
            // bits of a mask below the lane; with -1 for the mask, the lane's number
            bool scalar = ops[0].kind == Operand::Kind::Scalar;
            Wave b = kinds[1];
            if (b.kind == K::Prefix || (scalar && (b.kind == K::Plain || b.kind == K::Base))) {
                out = PREFIX_W;
            } else if (!scalar && (b.kind == K::Plain || b.kind == K::Base)) {
                out = PLAIN;
            } else {
                fail(why(b));
            }
        } else if (!any_wild) {
            // nothing wave-dependent goes in; what derives from a reserved base stays marked
            bool derives = std::holds_alternative<gfx10::Valu>(inst.fields) ||
                           std::holds_alternative<gfx10::Sop>(inst.fields);
            out = derives && std::find(kinds.begin(), kinds.end(), BASE_W) != kinds.end() ? BASE_W : PLAIN;
        } else if (is({"v_mov_b32", "s_mov_b32"}) && kinds.size() == 1) {
            out = kinds[0];
        } else if (is({"v_cmp_eq_i32", "v_cmp_eq_u32", "v_cmpx_eq_i32", "v_cmpx_eq_u32"}) && kinds.size() == 2) {
            // the lane whose own number is the first lane's
            Wave a = kinds[0], b = kinds[1];
            const Operand* lane;
            if (a == FIRST_W && !wild(b)) {
                lane = &ops[1];
            } else if (b == FIRST_W && !wild(a)) {
                lane = &ops[0];
            } else {
                fail(why(w));
            }
            if (lane->kind != Operand::Kind::Vector) fail(FIRST);
            std::optional<Value> mask;
            if (name.starts_with("v_cmpx")) {
                mask = inst.write(Reg::S(ir::EXEC));
            } else if (const auto* v = std::get_if<gfx10::Valu>(&inst.fields)) {
                if (v->dst && v->dst->kind == Operand::Kind::Scalar) mask = inst.write(Reg::S(v->dst->code));
            }
            if (mask) firsts.insert(*mask);
            out = PLAIN;
        } else if (is({"v_add_nc_u32", "s_add_i32"}) && kinds.size() == 2) {
            Wave a = kinds[0], b = kinds[1];
            if ((a == PREFIX_W && b == BASE_W) || (a == BASE_W && b == PREFIX_W)) {
                out = BASE_W;
            } else if ((a == PREFIX_W && (b == PLAIN || b == PREFIX_W)) || (a == PLAIN && b == PREFIX_W)) {
                out = PREFIX_W;
            } else if (sum(a) && sum(b)) {
                out = a;
            } else {
                fail(why(w));
            }
        } else if (is({"v_sub_nc_u32", "s_sub_i32"}) && kinds.size() == 2) {
            Wave a = kinds[0], b = kinds[1];
            if (a == BASE_W && b == PREFIX_W) {
                out = BASE_W;
            } else if (a == PREFIX_W && b == PLAIN) {
                out = PREFIX_W;
            } else if (a == PLAIN && sum(b) && ops[0] == Operand::Const(128)) {
                out = b;
            } else {
                fail(why(w));
            }
        } else if (is({"s_mul_i32", "v_mul_lo_u32", "v_mul_u32_u24"}) && kinds.size() == 2) {
            // `(x, Plain) | (Plain, x) if sum(x) || x == Prefix`: the guard is tried per alternative
            Wave a = kinds[0], b = kinds[1];
            if (b == PLAIN && (sum(a) || a == PREFIX_W)) {
                out = a;
            } else if (a == PLAIN && (sum(b) || b == PREFIX_W)) {
                out = b;
            } else {
                fail(why(w));
            }
        } else if (name == "s_lshl_b32" && kinds.size() == 2 && kinds[1] == PLAIN &&
                   (sum(kinds[0]) || kinds[0] == PREFIX_W)) {
            out = kinds[0];
        } else if (name == "v_lshlrev_b32" && kinds.size() == 2 && kinds[0] == PLAIN &&
                   (sum(kinds[1]) || kinds[1] == PREFIX_W)) {
            out = kinds[1];
        } else if (name == "v_lshl_add_u32" && kinds.size() == 3 && kinds[0] == PREFIX_W && kinds[1] == PLAIN &&
                   !wild(kinds[2])) {
            out = kinds[2] == BASE_W ? BASE_W : PREFIX_W;
        } else if (kinds.size() == 1 && kinds[0].kind == K::Total && name.find("_atomic_") != std::string_view::npos) {
            // the reservation: the first lane adds its wave's total to a shared counter
            std::string_view want;
            switch (kinds[0].op) {
                case Op::AddU:
                    want = "buffer_atomic_add";
                    break;
                case Op::MinU:
                    want = "buffer_atomic_umin";
                    break;
                case Op::MaxU:
                    want = "buffer_atomic_umax";
                    break;
                case Op::MinI:
                    want = "buffer_atomic_smin";
                    break;
                case Op::MaxI:
                    want = "buffer_atomic_smax";
                    break;
                case Op::Or:
                    want = "buffer_atomic_or";
                    break;
                case Op::And:
                    want = "buffer_atomic_and";
                    break;
                case Op::Xor:
                    want = "buffer_atomic_xor";
                    break;
                default:
                    want = "";
                    break;
            }
            if (name != want || !first_mask(firsts, inst.read(Reg::S(ir::EXEC)))) fail(TOTAL);
            reserved = true;
            out = BASE_W;
        } else {
            fail(why(w));
        }
        return {out, reserved};
    };
    std::vector<std::unordered_map<std::size_t, wave::Section>> sections;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) sections.push_back(wave::sections(p, b));
    const auto order = ir::reverse_postorder(p.blocks);
    auto pass = [&](KindMap& kind, HashSet<Value>& firsts, bool check) -> std::optional<std::pair<std::size_t, Why>> {
        for (std::size_t b : order) {
            const auto& block = p.blocks[b];
            for (const auto& phi : block.phis) {
                Wave k = PLAIN;
                for (std::size_t n = 0; n < phi.args.size(); ++n) {
                    Wave x = of(kind, phi.args[n]);
                    if (n == 0) {
                        k = x;
                    } else if (k == x) {
                        // k
                    } else if (k == PLAIN && (x == BASE_W || x == PREFIX_W)) {
                        k = x;
                    } else if ((k == BASE_W || k == PREFIX_W) && x == PLAIN) {
                        // k
                    } else {
                        k = MIXED_W;
                    }
                }
                put(kind, phi.value, k);
            }
            for (std::size_t i = 0; i < block.insts.size(); ++i) {
                const auto& inst = block.insts[i];
                // inside a whole-wave section: what the simulation found each value to be
                // order: sections of one block do not overlap, at most one holds i
                const wave::Section* sec = nullptr;
                for (const auto& [s, se] : sections[b]) {
                    if (i > s && i < se.end) {
                        sec = &se;
                        break;
                    }
                }
                if (sec) {
                    for (const auto& [_, w] : inst.writes) {
                        Wave k;
                        auto it = sec->shapes.find(w);
                        if (it == sec->shapes.end()) {
                            k = MIXED_W;
                        } else {
                            const wave::Shape& sh = it->second;
                            switch (sh.kind) {
                                case wave::Shape::Kind::K:
                                    k = PLAIN;
                                    break;
                                case wave::Shape::Kind::Value:
                                    k = of(kind, sh.a);
                                    break;
                                case wave::Shape::Kind::Masked:
                                    k = of(kind, sh.a);
                                    break;
                                case wave::Shape::Kind::Active: {
                                    Wave s = of(kind, sh.a);
                                    if (s.kind == K::Plain || s.kind == K::Base) {
                                        k = Wave::total(sh.op);
                                    } else if (s.kind == K::Total && s.op == sh.op && first_mask(firsts, sh.b)) {
                                        // the first lane's own sum of a total is that total
                                        k = Wave::total(sh.op);
                                    } else {
                                        k = MIXED_W;
                                    }
                                    break;
                                }
                                case wave::Shape::Kind::Prefix:
                                    if (sh.op == Op::AddU) {
                                        Wave s = of(kind, sh.a);
                                        if (s.kind == K::Plain || s.kind == K::Base) {
                                            k = PREFIX_W;
                                        } else if (s == Wave::total(Op::AddU) && first_mask(firsts, sh.b)) {
                                            k = PREFIX_W;
                                        } else {
                                            k = MIXED_W;
                                        }
                                    } else {
                                        k = MIXED_W;
                                    }
                                    break;
                                default:
                                    k = MIXED_W;
                                    break;
                            }
                        }
                        put(kind, w, k);
                    }
                    continue;
                }
                auto name = inst.name();
                if (name.starts_with("s_cbranch")) {
                    continue;  // with the block's exit, below
                }
                Wave k;
                bool reserved;
                try {
                    std::tie(k, reserved) = step(inst, kind, firsts);
                } catch (const WhyFail& e) {
                    if (check) return std::pair{inst.machine.offset, e.why};
                    k = MIXED_W;
                    reserved = false;
                }
                // the same test as `Lower::inst`'s `effect`
                bool effect = std::holds_alternative<gfx10::Exp>(inst.fields) ||
                              (std::holds_alternative<gfx10::Flat>(inst.fields) && !name.starts_with("global_load_")) ||
                              name.find("store") != std::string_view::npos ||
                              name.find("atomic") != std::string_view::npos ||
                              (std::holds_alternative<gfx10::Ds>(inst.fields) &&
                               !(name.starts_with("ds_read") || name.starts_with("ds_swizzle") ||
                                 name.find("permute") != std::string_view::npos));
                if (check && effect && !reserved && first_mask(firsts, inst.read(Reg::S(ir::EXEC)))) {
                    return std::pair{inst.machine.offset, EFFECT};
                }
                bool guarded = st.guards[b][i].has_value();
                for (const auto& [r, w] : inst.writes) {
                    Wave kw = k;
                    // "some lane is counted" keeps the count's kind; any other SCC is one bit of data
                    if (r.is_scc() && !name.starts_with("s_bcnt1")) {
                        kw = wild(k) ? MIXED_W : PLAIN;
                    }
                    // a guarded write leaves the old value in the lanes outside EXEC
                    if (guarded && r.is_v() && kw == PLAIN) {
                        Wave old = of(kind, inst.read(r));
                        if (old.kind == K::First || old.kind == K::Total || old.kind == K::Prefix) kw = MIXED_W;
                    }
                    put(kind, w, kw);
                }
            }
            // a branch on "some lane is counted": fine where the structure made it the lane's own
            // bit (the mask is EXEC under a scalar condition)
            if (check && !block.insts.empty()) {
                const auto& last = block.insts.back();
                if (last.name().starts_with("s_cbranch_scc")) {
                    if (auto scc = last.read(Reg::Scc); scc && wild(of(kind, scc))) {
                        bool own = false;
                        const auto& e = st.exits[b];
                        if (e.kind == LaneExit::Kind::Branch) {
                            own = true;
                            for (const auto& a : st.bools.support(e.when)) {
                                if (a.kind == Atom::Kind::Uniform && a.value == *scc) {
                                    own = false;
                                    break;
                                }
                            }
                        }
                        if (!own) return std::pair{last.machine.offset, TOTAL};
                    }
                }
            }
        }
        return std::nullopt;
    };
    KindMap kind;
    HashSet<Value> firsts;
    // to a fixpoint (values carried around loops), then once more to say what is wrong
    bool settled = false;
    for (int n = 0; n < 16; ++n) {
        KindMap before = kind;
        std::size_t before_firsts = firsts.size();
        if (auto e = pass(kind, firsts, false)) return e;
        if (before == kind && before_firsts == firsts.size()) {
            settled = true;
            break;
        }
    }
    if (!settled) return std::pair{std::size_t{0}, MIXED};
    return pass(kind, firsts, true);
}

// The mode of a group barrier the translation adds (host waves meeting): a wait for the
// group with a shared-memory fence; the fence alone with X2W_SHADERCFG_GROUP_SYNC=0.
std::uint32_t meeting_mode() { return x2w::vs::group_sync() == 0 ? 8 : (1 | 8); }

// LDS accesses of a console wave run as two host waves, where some sit in blocks that only
// some lanes run (`all` is all_lane_blocks): per IR block, whether the host waves meet (a group
// barrier) at its start and before its exit. None if barriers cannot keep the console's order.
//
// The console runs the wave's lanes in step, so an LDS write is seen by every later read. Two host
// waves keep that order if they meet between any two accesses of which one writes. A barrier must
// be run by every thread, so it cannot stand in a lane branch's region; but every way into and out
// of such a region passes a block all lanes run, and the waves meet there: before the exit of an
// all-lane block into a region with LDS accesses, and at the start of an all-lane block entered
// from one. Accesses in all-lane blocks get their own barrier as before. What that cannot order is
// a write and a read inside one region (blocks joined without an all-lane block between them):
// refused.
// shortcut: two writes in one region are taken not to hit the same dword from both halves.
std::optional<std::vector<std::pair<bool, bool>>> lds_meetings(const Program& p, const Structured& st,
                                                               const std::vector<bool>& all) {
    const std::size_t n = st.exits.size();
    auto succ = [&](std::size_t b) -> std::vector<std::size_t> {
        const auto& e = st.exits[b];
        switch (e.kind) {
            case LaneExit::Kind::Jump:
                return {e.target};
            case LaneExit::Kind::Branch:
                return {e.taken, e.not_taken};
            default:
                return {};
        }
    };
    // regions: blocks not every lane runs, joined by the edges between them
    std::vector<std::size_t> region(n);
    for (std::size_t b = 0; b < n; ++b) region[b] = b;
    auto find = [](std::vector<std::size_t>& region, std::size_t b) {
        while (region[b] != b) {
            region[b] = region[region[b]];
            b = region[b];
        }
        return b;
    };
    for (std::size_t b = 0; b < n; ++b) {
        for (std::size_t t : succ(b)) {
            if (!all[b] && !all[t]) {
                std::size_t x = find(region, b);
                std::size_t y = find(region, t);
                region[x] = y;
            }
        }
    }
    // per region: LDS reads, LDS writes
    std::vector<bool> reads(n, false);
    std::vector<bool> writes(n, false);
    for (std::size_t b = 0; b < p.blocks.size() && b < n; ++b) {
        const auto& block = p.blocks[b];
        if (all[b]) continue;
        std::size_t r = find(region, b);
        for (const auto& i : block.insts) {
            auto name = i.name();
            if (!std::holds_alternative<gfx10::Ds>(i.fields) || name.starts_with("ds_swizzle") ||
                name.find("permute") != std::string_view::npos) {
                continue;
            }
            if (name.starts_with("ds_read")) {
                reads[r] = true;
            } else {
                writes[r] = true;
            }
        }
    }
    for (std::size_t r = 0; r < n; ++r) {
        if (reads[r] && writes[r]) return std::nullopt;
    }
    std::vector<std::pair<bool, bool>> meet(n, {false, false});
    for (std::size_t b = 0; b < n; ++b) {
        for (std::size_t t : succ(b)) {
            if (all[b] && !all[t]) {
                std::size_t r = find(region, t);
                meet[b].second = meet[b].second || reads[r] || writes[r];
            } else if (!all[b] && all[t]) {
                std::size_t r = find(region, b);
                meet[t].first = meet[t].first || reads[r] || writes[r];
            }
        }
    }
    return meet;
}

// Masks held as numbers or constant lane sets in guards or lane branches: the lane tests its
// own bit by its console lane.
bool lane_constant_masks(const Structured& st) {
    std::vector<Bool> work;
    for (const auto& per_block : st.guards) {
        for (const auto& g : per_block) {
            if (g) work.push_back(*g);
        }
    }
    for (const auto& e : st.exits) {
        if (e.kind == LaneExit::Kind::Branch) work.push_back(e.when);
    }
    // Through lane-mask joins too: an edge that brings a number or a constant lane set needs the
    // lane's own bit of it.
    // shortcut: masks only selects read (`v_cndmask_b32`, carries in) are not followed; such a
    // program still stops with "a constant lane mask" or "an opaque lane mask".
    std::vector<Bool> seen;  // a set (Bool has ==; no hash needed)
    HashSet<Value> joins;
    while (!work.empty()) {
        Bool g = work.back();
        work.pop_back();
        if (std::find(seen.begin(), seen.end(), g) != seen.end()) continue;
        seen.push_back(g);
        for (const auto& a : st.bools.support(g)) {
            if (a.kind == Atom::Kind::Opaque || a.kind == Atom::Kind::Constant) return true;
            if (a.kind == Atom::Kind::Phi && joins.insert(a.value).second) {
                auto it = st.phi_masks.find(a.value);
                if (it != st.phi_masks.end()) work.insert(work.end(), it->second.begin(), it->second.end());
            }
        }
    }
    return false;
}

void set_host_max_wave(std::uint32_t lanes) { HOST_MAX_WAVE.store(lanes, std::memory_order_relaxed); }

namespace detail {
bool replace_local_waves(bool on) { return std::exchange(LOCAL_WAVES, on); }
bool replace_local_any(bool on) { return std::exchange(LOCAL_ANY, on); }
bool replace_first_half_pairs(bool on) { return std::exchange(FIRST_HALF_PAIRS, on); }
}  // namespace detail

bool local_waves_on() { return LOCAL_WAVES; }
bool local_any_on() { return LOCAL_ANY; }
bool first_half_pairs_on() { return FIRST_HALF_PAIRS; }

// X2W_SHADERCFG_F16_NEAREST=1: the results of 16-bit float operations are rounded to nearest as
// the console rounds them (Lower::f16_nearest) instead of as the host's `f32tof16` does. Off by
// default.
bool f16_nearest_on() {
    if (TEST_F16_NEAREST) return *TEST_F16_NEAREST;
    static const bool ON = [] {
        auto v = env_var("X2W_SHADERCFG_F16_NEAREST");
        return v && *v == "1";
    }();
    return ON;
}

// X2W_SHADERCFG_WAVE_OWN_BITS=0: a wave OR where not every lane runs is refused as before.
bool own_bits_on() {
    static const bool ON = [] {
        auto v = env_var("X2W_SHADERCFG_WAVE_OWN_BITS");
        return !v || *v != "0";
    }();
    return ON;
}

// What the translator was told of the host, for a cache's key: the widest wave, the view
// heap's limit, where the raw twins of its views start, and what it writes for the console's own
// hardware.
std::array<std::uint32_t, 4> host_settings() {
    std::uint32_t a = host_max_wave();
    std::uint32_t b = shader::view_heap_limit();
    std::uint32_t c = shader::raw_views();
    std::uint32_t d = host_rdna();
    return {a, b, c, d};
}

void set_host_rdna(bool on) { HOST_RDNA.store(on, std::memory_order_relaxed); }

std::uint32_t host_rdna() {
    if (TEST_RDNA) return *TEST_RDNA;
    return HOST_RDNA.load(std::memory_order_relaxed) ? x2w::vs::rdna_float() : 0;
}

std::uint32_t host_max_wave() {
    if (TEST_HOST_WAVE) return *TEST_HOST_WAVE;
    return HOST_MAX_WAVE.load(std::memory_order_relaxed);
}

namespace test {
void set_f16_nearest(std::optional<bool> on) { TEST_F16_NEAREST = on; }
void set_rdna(std::optional<std::uint32_t> bits) { TEST_RDNA = bits; }
void set_host_wave(std::optional<std::uint32_t> lanes) { TEST_HOST_WAVE = lanes; }
void set_memory(std::optional<std::pair<std::uint32_t, std::uint64_t>> slot_and_address) {
    TEST_MEMORY = slot_and_address;
}
}  // namespace test

std::optional<std::pair<std::uint32_t, std::uint64_t>> test_memory() { return TEST_MEMORY; }

// Instructions that read a lane mask (not EXEC implicitly, not a lane's bit) as a number.
bool reads_mask_as_data(const Structured& st, const Inst& i) {
    std::vector<const Operand*> ops;
    if (const auto* sop = std::get_if<gfx10::Sop>(&i.fields)) {
        // A mask operation (its result a mask) is no data use.
        for (const auto& [r, w] : i.writes) {
            if (r.is_s() && is_mask_of(st, w)) return false;
        }
        for (const auto& o : sop->src) ops.push_back(&o);
    } else if (const auto* v = std::get_if<gfx10::Valu>(&i.fields);
               v && i.name() != "v_cndmask_b32" && i.name().find("_co_ci_") == std::string_view::npos) {
        for (const auto& o : v->src) ops.push_back(&o);
    } else {
        return false;
    }
    for (const Operand* op : ops) {
        if (op->kind == Operand::Kind::Scalar && op->code != gfx10::NULL_CODE) {
            auto x = i.read(Reg::S(op->code));
            if (x && is_mask_of(st, *x)) return true;
        }
    }
    return false;
}

// The parameters packing 16-bit rows two per dword and where each lays its elements out:
// (first row, [(dword, register, column, columns)]).
//
// The console allocates whole dwords to each 16-bit element. The first element of each row comes first. A
// short element packed into spare columns of an earlier register follows in the first parameter with whole
// dwords left, in row order. Packed pedestrian shaders use both layouts.
PackedLayout packed_layout(std::span<const SigElement> elements,
                           std::span<const std::pair<std::uint8_t, std::optional<std::uint8_t>>> packs_in) {
    auto of_row = [&](std::uint8_t r) {
        std::vector<const SigElement*> of;
        for (const auto& e : elements) {
            if (e.row == std::optional<std::uint32_t>(r)) of.push_back(&e);
        }
        std::stable_sort(of.begin(), of.end(), [](const SigElement* a, const SigElement* b) { return a->col < b->col; });
        return of;
    };
    std::vector<std::pair<std::uint8_t, std::optional<std::uint8_t>>> packs(packs_in.begin(), packs_in.end());
    std::sort(packs.begin(), packs.end());
    // dedup_by_key(|p| p.0): keeps the first of each run
    packs.erase(std::unique(packs.begin(), packs.end(), [](const auto& a, const auto& b) { return a.first == b.first; }),
                packs.end());
    // Per parameter: its first row, the dwords used, the elements.
    using Placed = std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>;
    std::vector<std::tuple<std::uint8_t, std::uint8_t, std::vector<Placed>>> layout;
    std::vector<std::tuple<std::uint8_t, std::uint8_t, std::uint8_t>> behind;
    for (const auto& [a, b] : packs) {
        std::vector<Placed> placed;
        std::uint8_t at = 0;
        if (!of_row(a).empty()) {
            std::vector<std::uint8_t> rows{a};
            if (b) rows.push_back(*b);
            for (std::uint8_t r : rows) {
                auto of = of_row(r);
                if (of.empty()) continue;
                const SigElement* lead = of[0];
                placed.emplace_back(at, r, lead->col, lead->cols);
                at = static_cast<std::uint8_t>(at + div_ceil<std::uint8_t>(lead->cols, 2));
                for (std::size_t k = 1; k < of.size(); ++k) behind.emplace_back(r, of[k]->col, of[k]->cols);
            }
        }
        layout.emplace_back(a, at, std::move(placed));
    }
    std::sort(behind.begin(), behind.end());
    for (const auto& [r, col, cols] : behind) {
        std::uint8_t need = div_ceil<std::uint8_t>(cols, 2);
        for (auto& p : layout) {
            if (std::get<1>(p) + need <= 4) {
                std::get<2>(p).emplace_back(std::get<1>(p), r, col, cols);
                std::get<1>(p) = static_cast<std::uint8_t>(std::get<1>(p) + need);
                break;
            }
        }
    }
    PackedLayout out;
    for (auto& [a, _, of] : layout) out.emplace_back(a, std::move(of));
    return out;
}

// Where half `h` (dword h / 2, the high half when odd) of the parameter whose first 16-bit
// row is `a` comes from: a register and column of `elements`, none for padding.
std::optional<std::pair<std::uint8_t, std::uint8_t>> packed_half(
    std::span<const SigElement> elements,
    std::span<const std::pair<std::uint8_t, std::optional<std::uint8_t>>> packs, std::uint8_t a,
    std::uint8_t h) {
    auto layout = packed_layout(elements, packs);
    auto it = std::find_if(layout.begin(), layout.end(), [&](const auto& p) { return p.first == a; });
    if (it == layout.end()) return std::nullopt;
    for (const auto& [dword, row, col, cols] : it->second) {
        std::uint8_t two = static_cast<std::uint8_t>(2 * dword);
        if (h < two) continue;  // h.checked_sub(2 * dword)?
        std::uint8_t k = static_cast<std::uint8_t>(h - two);
        if (k < cols) return std::pair{row, static_cast<std::uint8_t>(col + k)};
    }
    return std::nullopt;
}

// The pixel inputs packing two 16-bit rows per dword, as (first row, second row).
std::vector<std::pair<std::uint8_t, std::optional<std::uint8_t>>> input_packs(const StageIo& io,
                                                                             std::span<const SigElement> inputs) {
    const auto* ps = std::get_if<native::PsIo>(&io);
    if (!ps) return {};
    auto partners = half_input_partners(io, inputs);
    std::vector<std::pair<std::uint8_t, std::optional<std::uint8_t>>> out;
    for (const auto& i : ps->inputs) {
        if (!(i.fp16 && i.pack2)) continue;
        std::optional<std::uint8_t> b;
        if (i.row2 != i.row) {
            b = i.row2;
        } else if (auto it = partners.find(i.row); it != partners.end()) {
            b = it->second;
        }
        out.emplace_back(i.row, b);
    }
    return out;
}

// Vertex or mesh outputs of 16-bit values (see Lower::half_outputs): a packed row's partner
// is the next 16-bit row of the signature that no output entry names, taken in row order.
HashMap<std::uint8_t, std::pair<bool, std::optional<std::uint8_t>>> half_outputs(
    const StageIo& io, std::span<const SigElement> outputs) {
    const std::vector<native::VsOutput>* entries;
    if (const auto* vs = std::get_if<native::VsIo>(&io)) {
        entries = &vs->outputs;
    } else if (const auto* ms = std::get_if<native::MeshIo>(&io)) {
        entries = &ms->outputs;
    } else {
        return {};
    }
    std::vector<const native::VsOutput*> per_vertex;
    for (const auto& o : *entries) {
        if ((o.param & 0x80) == 0) per_vertex.push_back(&o);
    }
    std::vector<std::uint8_t> free;
    for (const auto& e : outputs) {
        if (!(e.half && e.kind == 0) || !e.row || *e.row > 0xFF) continue;
        auto r = static_cast<std::uint8_t>(*e.row);
        if (std::any_of(per_vertex.begin(), per_vertex.end(), [&](const auto* o) { return o->row == r; })) continue;
        free.push_back(r);
    }
    std::sort(free.begin(), free.end());
    free.erase(std::unique(free.begin(), free.end()), free.end());
    std::vector<const native::VsOutput*> halves;
    for (const auto* o : per_vertex) {
        if (o->fp16) halves.push_back(o);
    }
    std::stable_sort(halves.begin(), halves.end(), [](const auto* a, const auto* b) { return a->row < b->row; });
    HashMap<std::uint8_t, std::pair<bool, std::optional<std::uint8_t>>> out;
    for (const auto* o : halves) {
        std::optional<std::uint8_t> b;
        if (o->pack2) {
            auto it = std::find_if(free.begin(), free.end(), [&](std::uint8_t r) { return r > o->row; });
            if (it != free.end()) {
                b = *it;
                free.erase(it);
            }
        }
        out.insert_or_assign(o->row, std::pair{o->pack2, b});
    }
    return out;
}

// Pixel inputs packing two 16-bit rows per dword whose second row the metadata leaves out
// (`row2` = `row`): as for outputs (half_outputs), the next 16-bit row of the signature that no
// input entry names, taken in row order (rows the metadata does give are taken first).
HashMap<std::uint8_t, std::uint8_t> half_input_partners(const StageIo& io, std::span<const SigElement> inputs) {
    const auto* ps = std::get_if<native::PsIo>(&io);
    if (!ps) return {};
    const auto& entries = ps->inputs;
    auto packed = [&](const native::PsInput& i) { return i.fp16 && i.pack2; };
    std::vector<std::uint8_t> free;
    for (const auto& e : inputs) {
        if (!(e.half && e.kind == 0) || !e.row || *e.row > 0xFF) continue;
        auto r = static_cast<std::uint8_t>(*e.row);
        if (std::any_of(entries.begin(), entries.end(), [&](const auto& i) { return i.row == r; })) continue;
        if (std::any_of(entries.begin(), entries.end(),
                        [&](const auto& i) { return packed(i) && i.row2 != i.row && i.row2 == r; })) {
            continue;
        }
        free.push_back(r);
    }
    std::sort(free.begin(), free.end());
    free.erase(std::unique(free.begin(), free.end()), free.end());
    std::vector<const native::PsInput*> missing;
    for (const auto& i : entries) {
        if (packed(i) && i.row2 == i.row) missing.push_back(&i);
    }
    std::stable_sort(missing.begin(), missing.end(), [](const auto* a, const auto* b) { return a->row < b->row; });
    HashMap<std::uint8_t, std::uint8_t> partners;
    for (const auto* i : missing) {
        auto it = std::find_if(free.begin(), free.end(), [&](std::uint8_t r) { return r > i->row; });
        if (it != free.end()) {
            partners.insert_or_assign(i->row, *it);
            free.erase(it);
        }
    }
    return partners;
}

std::optional<std::uint8_t> interp_mode(Barycentric kind) {
    switch (kind) {
        case Barycentric::PerspectiveCenter:
            return 2;
        case Barycentric::PerspectiveCentroid:
            return 3;
        case Barycentric::PerspectiveSample:
            return 6;
        case Barycentric::LinearCenter:
            return 4;
        case Barycentric::LinearCentroid:
            return 5;
        case Barycentric::LinearSample:
            return 7;
        case Barycentric::PerspectivePullModel:
            return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace x2w::vs::lower
