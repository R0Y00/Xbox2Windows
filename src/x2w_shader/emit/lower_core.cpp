// The emitter's set-up, its run over the program, and the whole-wave sections.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#include "x2w_shader/emit/lower_impl.h"

namespace x2w::vs::lower {

namespace {

// std::env::var_os(name).is_some()
bool env_set(const char* name) { return std::getenv(name) != nullptr; }

// Multiply thread counts modulo 2^32.
std::uint32_t product3(const std::array<std::uint32_t, 3>& t) { return t[0] * t[1] * t[2]; }

// st.masks.get(&mask): the lane boolean of a mask by its low dword.
std::optional<Bool> mask_of(const Structured& st, Value mask) {
    auto it = st.masks.find(mask);
    if (it == st.masks.end()) return std::nullopt;
    return it->second;
}

}  // namespace

Lower::Lower(const Parts& parts, const HashMap<Value, Why>& bad_)
    : s(parts.native->stage == HwStage::Ps), x(parts), bad(bad_) {
    const Program& p = *x.p;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block_ = p.blocks[b];
        for (std::size_t i = 0; i < block_.phis.size(); ++i) {
            phi_at.insert_or_assign(block_.phis[i].value, std::pair{b, i});
        }
    }
    s.inputs = x.inputs;
    s.outputs = x.outputs;
    auto any_inst = [&](auto&& f) {
        for (const auto& blk : p.blocks) {
            for (const auto& i : blk.insts) {
                if (f(i)) return true;
            }
        }
        return false;
    };
    bool writes_uavs_ = any_inst([](const Inst& i) {
        auto n = i.name();
        return (std::holds_alternative<gfx10::Image>(i.fields) ||
                std::holds_alternative<gfx10::Buffer>(i.fields)) &&
               (n.find("store") != std::string_view::npos ||
                n.find("atomic") != std::string_view::npos);
    });
    bool console_lanes_ = false;
    // LDS accesses get a group barrier each (a console wave split in two host waves)
    bool split_lds_ = false;
    // with `split_lds`, LDS accesses in lane branches: the blocks every lane runs, and where the
    // host waves meet around those branches (`lds_meetings`)
    std::optional<std::pair<std::vector<bool>, std::vector<std::pair<bool, bool>>>> lds_meet_;
    // a console wave64 program in two host waves of 32 (`exchange`)
    std::optional<std::tuple<std::uint32_t, bool, std::vector<bool>>> exchange_;
    // a console wave64 program in host waves of 32: the LDS dword of `wave_base`'s counter
    std::optional<std::uint32_t> halves_;
    if (const auto* cs = std::get_if<native::ComputeIo>(&x.native->io)) {
        s.threads = cs->threads;
        s.lds_dwords = cs->lds_bytes() / 4;
        // Code reading other lanes needs the console's wave size and lane order; so do lane
        // masks read as numbers, and LDS shared within one wave, where the console compiler
        // leaves out barriers.
        console_lanes_ = any_inst([&](const Inst& i) {
                             auto n = i.name();
                             const auto* v = std::get_if<gfx10::Valu>(&i.fields);
                             return n.starts_with("ds_swizzle") ||
                                    n.find("permute") != std::string_view::npos ||
                                    n.starts_with("v_permlane") || n.starts_with("v_mbcnt") ||
                                    n == "v_readlane_b32" || n == "v_writelane_b32" ||
                                    (v && v->dpp.has_value()) || reads_mask_as_data(*x.st, i);
                         }) ||
                         lane_constant_masks(*x.st)
                         // (a branch the wave takes together asks the wave)
                         || !x.st->wide.empty();
        std::uint32_t wave_ = x.native->wave64 ? 64 : 32;
        std::uint32_t group = product3(cs->threads);
        bool one_wave = group <= wave_;
        // A host whose waves are at most 32 lanes (NVIDIA) refuses `WaveSize(64)`. A console
        // wave64 program whose cross-lane code stays inside 32-lane halves - DPP rows,
        // `ds_swizzle`, `v_permlane16` / `v_permlanex16`, which the ISA confines there - runs
        // exactly in waves of 32: the thread relabelling makes each host wave one half of a
        // console wave (lanes 32k..32k+31). Not when it crosses halves (`v_mbcnt`, `v_readlane`,
        // `v_writelane`, `ds_(b)permute`, masks as numbers or lane-constant masks), nor when LDS
        // may be shared within the console wave without barriers (a group over 32 threads with
        // LDS): those still need 64-lane waves.
        if (wave_ == 64 && host_max_wave() == 32 &&
            (console_lanes_ || (one_wave && s.lds_dwords > 0))) {
            bool crosses = any_inst([&](const Inst& i) {
                               auto n = i.name();
                               return n.starts_with("v_mbcnt") || n == "v_readlane_b32" ||
                                      n == "v_writelane_b32" || n.starts_with("ds_permute") ||
                                      n.starts_with("ds_bpermute") || reads_mask_as_data(*x.st, i);
                           }) ||
                           lane_constant_masks(*x.st) || !x.st->wide.empty();
            if (!crosses && (s.lds_dwords == 0 || group <= 32)) {
                wave_ = 32;
            } else {
                // (Groups of several console waves too: each console wave is two host waves,
                // which exchange through their own pair of dwords.)
                // LDS the console shares within its one wave without barriers: split into two
                // waves of 32, a group barrier before every LDS access keeps their order - where
                // every thread of the group gets there (no lane branch around it, no guard),
                // else a barrier would hang the group
                auto lds = [](const Inst& i) {
                    auto n = i.name();
                    return std::holds_alternative<gfx10::Ds>(i.fields) &&
                           !(n.starts_with("ds_swizzle") || n.find("permute") != std::string_view::npos);
                };
                if (auto all_ = all_lane_blocks(*x.st)) {
                    std::vector<bool> all = std::move(*all_);
                    auto all_at = [&](std::size_t b) { return b < all.size() && all[b]; };
                    bool ok = true;
                    for (std::size_t b = 0; b < p.blocks.size() && ok; ++b) {
                        for (const auto& inst : p.blocks[b].insts) {
                            if (!(!lds(inst) || all_at(b))) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    bool lds_ok = !(s.lds_dwords > 0 && group > 32) || ok;
                    if (!lds_ok) {
                        if (auto meet = lds_meetings(p, *x.st, all)) {
                            lds_meet_ = std::pair{all, std::move(*meet)};
                            lds_ok = true;
                        }
                    }
                    if (!lds_ok) {
                        // (the blocks whose LDS accesses not every lane runs: with the branches
                        // around them taken wave-wide they get their barriers)
                        for (std::size_t b = 0; b < p.blocks.size(); ++b) {
                            const auto& blk = p.blocks[b];
                            if (!all_at(b) && std::any_of(blk.insts.begin(), blk.insts.end(), lds)) {
                                need(b);
                            }
                        }
                    }
                    if (lds_ok) {
                        wave_ = 32;
                        split_lds_ = s.lds_dwords > 0 && group > 32;
                        if (crosses) {
                            // lanes 32..63 are the other host wave: masks, lane reads and lane
                            // numbers go through two exchange dwords past the program's LDS
                            exchange_ = std::tuple{s.lds_dwords, group <= 32, std::move(all)};
                            s.lds_dwords += 2 * div_ceil<std::uint32_t>(group, 64);
                        }
                    }
                }
            }
            // `X2W_SHADERCFG_WAVE_WHY=1`: why a program keeps 64-lane waves (one line per reason)
            if (wave_ == 64 && env_set("X2W_SHADERCFG_WAVE_WHY")) {
                std::vector<std::string_view> why;
                for (const auto& blk : p.blocks) {
                    for (const auto& i : blk.insts) {
                        auto n = i.name();
                        std::string_view r;
                        if (n.starts_with("v_mbcnt")) {
                            r = "v_mbcnt";
                        } else if (n == "v_readlane_b32") {
                            r = "v_readlane";
                        } else if (n == "v_writelane_b32") {
                            r = "v_writelane";
                        } else if (n.starts_with("ds_permute") || n.starts_with("ds_bpermute")) {
                            r = "ds_(b)permute";
                        } else if (reads_mask_as_data(*x.st, i)) {
                            r = "mask as data";
                        } else {
                            continue;
                        }
                        if (std::find(why.begin(), why.end(), r) == why.end()) why.push_back(r);
                    }
                }
                if (lane_constant_masks(*x.st)) why.push_back("lane-constant mask");
                if (s.lds_dwords > 0 && group > 32) {
                    why.push_back("LDS in a group over 32");
                    auto all = all_lane_blocks(*x.st);
                    if (!all) {
                        why.push_back("an indirect exit");
                    } else if (!lds_meetings(p, *x.st, *all)) {
                        why.push_back("an LDS read and write in one lane-branch region");
                    }
                }
                std::sort(why.begin(), why.end());
                std::string joined;
                for (std::size_t k = 0; k < why.size(); ++k) {
                    if (k) joined += ", ";
                    joined += why[k];
                }
                std::fprintf(stderr, "WAVE64-WHY group=%u lds=%s [%s]\n", group,
                             s.lds_dwords > 0 ? "true" : "false", joined.c_str());
            }
        }
        if (wave_ == 32 && x.native->wave64) {
            // console threads go to host waves in the order the waves take them (`wave_base`)
            halves_ = s.lds_dwords;
            s.lds_dwords += 1;
        }
        if (console_lanes_ || (one_wave && s.lds_dwords > 0)) {
            s.wave_size = wave_;
        }
    }
    // The title's barriers a thread of the group may never reach, which leaves the others
    // waiting on the host: those in blocks some lanes do not run, and all of them in a program
    // with a loop closed by a plain jump (a lane still in it at the iteration limit leaves the
    // program there, `abort_block`).
    HashSet<std::size_t> fence_only_;
    if (any_inst([](const Inst& i) { return i.name() == "s_barrier"; })) {
        auto order = ir::reverse_postorder(p.blocks);
        std::vector<std::size_t> place(p.blocks.size(), SIZE_MAX);
        for (std::size_t n = 0; n < order.size(); ++n) place[order[n]] = n;
        auto at = [&](std::size_t b) { return b < place.size() ? place[b] : SIZE_MAX; };
        auto back = [&](std::size_t from, std::size_t to) {
            return at(from) != SIZE_MAX && at(to) <= at(from);
        };
        bool cut_short = false;
        for (std::size_t b = 0; b < x.st->exits.size(); ++b) {
            const auto& e = x.st->exits[b];
            if (e.kind == LaneExit::Kind::Jump && back(b, e.target)) {
                cut_short = true;
                break;
            }
        }
        auto all = all_lane_blocks(*x.st);
        for (std::size_t b = 0; b < p.blocks.size(); ++b) {
            const auto& insts = p.blocks[b].insts;
            bool barrier = std::any_of(insts.begin(), insts.end(),
                                       [](const Inst& i) { return i.name() == "s_barrier"; });
            if (!barrier) continue;
            if (cut_short || !all || !(b < all->size() && (*all)[b])) fence_only_.insert(b);
        }
    }
    HashSet<std::size_t> no_barrier_;
    if (const auto* ms = std::get_if<native::MeshIo>(&x.native->io)) {
        s.prims = x.prims;
        s.threads = std::array<std::uint32_t, 3>{ms->threads, 1, 1};
        s.lds_dwords = div_ceil<std::uint32_t>(ms->lds_bytes, 4);
        s.mesh = Mesh{ms->max_vertices, ms->max_primitives, ms->topology};
        // Barriers order the lanes' LDS accesses; the console's allocation request needs them
        // too, but the host does not. One lane sends that request, and what it dominates (the
        // path for counts out of range) runs in that lane alone, so its barrier would hang the
        // host.
        auto order = ir::reverse_postorder(p.blocks);
        std::vector<std::size_t> index(p.blocks.size(), SIZE_MAX);
        for (std::size_t n = 0; n < order.size(); ++n) index[order[n]] = n;
        auto idom = ir::idoms(p.blocks, order, index);
        std::vector<std::size_t> alloc;
        for (std::size_t b = 0; b < p.blocks.size(); ++b) {
            for (const auto& i : p.blocks[b].insts) {
                const auto* sopp = std::get_if<gfx10::Sopp>(&i.fields);
                if (i.name() == "s_sendmsg" && sopp && (sopp->imm & 0xF) == GS_ALLOC_REQ) {
                    alloc.push_back(b);
                    break;
                }
            }
        }
        for (std::size_t b : order) {
            std::size_t d = b;
            bool dominated;
            for (;;) {
                if (std::find(alloc.begin(), alloc.end(), d) != alloc.end()) {
                    dominated = true;
                    break;
                }
                if (idom.at(d) == d || idom.at(d) == SIZE_MAX) {
                    dominated = false;
                    break;
                }
                d = idom.at(d);
            }
            if (ms->lds_bytes == 0 || dominated) no_barrier_.insert(b);
        }
    }
    // The struct literal, field by field (none of these make builder calls).
    block.assign(p.blocks.size(), std::nullopt);
    cur = 0;
    if (x.native->stage == HwStage::Ps) ordered = ordered_section(p);
    for (const auto& [v, pp, d] : x.res->params) params.insert_or_assign(v, std::pair{pp, d});
    for (const auto& d : x.res->descriptors) {
        descriptors.insert_or_assign(std::pair{d.site, d.role == Role::Sampler}, &d.source);
    }
    for (const auto& [site, a] : x.res->scalar) scalar.insert_or_assign(site, a);
    for (const auto& e : x.io->exports) exports.insert_or_assign(e.site, &e);
    for (const auto& [site, i] : x.io->interps) interps.insert_or_assign(site, &i);
    // A first half whose result is also read as data (an f16 second half keeps the other half of
    // its dst): evaluated as the pair it starts. Approximate for those other
    // bits (P0 + i P10 on the console); a first half only the second half reads is gone before
    // lowering. Only in the second translation of a program the first one refused for it
    // (with_first_half_pairs).
    if (first_half_pairs_on()) {
        const char* fh = std::getenv("X2W_SHADERCFG_FIRST_HALF");
        if (fh == nullptr || std::strcmp(fh, "0") != 0) {
            for (const auto& [site, i] : x.io->interps) {
                if (i.kind == io::Interp::Kind::Interpolated) {
                    for (const Site& f : i.first) interps.try_emplace(f, &i);
                }
            }
        }
    }
    exported.assign(p.blocks.size(), false);
    rpo.assign(p.blocks.size(), SIZE_MAX);
    tail.assign(p.blocks.size(), std::nullopt);
    writes_uavs = writes_uavs_;
    console_lanes = console_lanes_;
    split_lds = split_lds_;
    lds_meet = std::move(lds_meet_);
    local = exchange_.has_value() && local_waves_on();
    local_ok = false;
    local_any = local_any_on();
    if (exchange_ && !std::get<1>(*exchange_) && !local_waves_on()) {
        uniform = lane_uniform(p, *x.st, std::get<2>(*exchange_));
    }
    if (exchange_ && !std::get<1>(*exchange_) && !local_waves_on() && own_bits_on()) {
        own_bits = own_bits::find(p, *x.st, env_set("X2W_SHADERCFG_WAVE_WHY"));
    }
    exchange = std::move(exchange_);
    halves = halves_;
    no_barrier = std::move(no_barrier_);
    fence_only = std::move(fence_only_);
    half_outputs_ = half_outputs(x.native->io, x.outputs);
    switch (x.native->stage) {
        case HwStage::Cs:
            mask_data = MaskData::Console;
            break;
        case HwStage::Ps:
        case HwStage::Vs:
            mask_data = MaskData::Host;
            break;
        default:
            mask_data = MaskData::None;
            break;
    }
    mask_ok = false;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& blk = p.blocks[b];
        for (const auto& i : blk.insts) {
            for (const auto& w : i.writes) def_block.insert_or_assign(w.second, b);
        }
        for (const auto& phi : blk.phis) def_block.insert_or_assign(phi.value, b);
    }
    cur_ir = 0;
}

// The dword of the program's code at byte `byte` (a constant table the compiler appends
// to the code), as an internal constant array.
// shortcut: an index outside the table reads its last dword (the console would read whatever
// memory follows the code).
V Lower::code_dword(V byte) {
    V g;
    std::uint32_t base, len;
    if (this->code_table) {
        std::tie(g, base, len) = *this->code_table;
    } else {
        std::optional<std::uint32_t> first_;
        auto take = [&](const Address& a) {
            if (a.param != resource::CODE) return;
            std::uint32_t o = a.offset & ~3u;
            if (!first_ || o < *first_) first_ = o;
        };
        for (const auto& sc : this->x.res->scalar) take(sc.second);
        for (const auto& d : this->x.res->descriptors) {
            for (const Source* l : d.source.leaves()) {
                if (l->kind == Source::Kind::Built) take(l->address);
            }
        }
        std::uint32_t first = ok_or(first_, "a code table");
        const auto& code = this->x.native->code;
        if (first > code.size()) fail("a code table past the code");
        std::vector<std::uint32_t> words;
        for (std::size_t at = first; at < code.size(); at += 4) {
            std::uint8_t w[4] = {0, 0, 0, 0};
            for (std::size_t k = 0; k < 4 && at + k < code.size(); ++k) w[k] = code[at + k];
            words.push_back(std::uint32_t(w[0]) | std::uint32_t(w[1]) << 8 |
                            std::uint32_t(w[2]) << 16 | std::uint32_t(w[3]) << 24);
        }
        if (words.empty()) fail("a code table past the code");
        std::vector<V> elems;
        for (std::uint32_t w : words) elems.push_back(this->s.m.i32(w));
        Ty arr = this->s.m.array(static_cast<std::uint32_t>(words.size()), this->s.t.i32);
        V init = this->s.m.aggregate(arr, elems);
        g = this->s.m.constant_global("code", init);
        base = first;
        len = static_cast<std::uint32_t>(words.size());
        this->code_table = std::tuple{g, base, len};
    }
    std::size_t c = this->cur;
    V k = this->s.m.i32(base);
    V rel = this->s.m.bin(c, Bin::Sub, byte, k);
    V two = this->s.m.i32(2);
    V index = this->s.m.bin(c, Bin::LShr, rel, two);
    V last = this->s.m.i32(len - 1);
    V over = this->s.m.cmp(c, pred::IUGT, index, last);
    index = this->s.m.select(c, over, last, index);
    V at = this->s.m.element(c, g, index);
    return this->s.m.load(c, at);
}

// A global memory load (`global_load_*`): from the program's constant table, or from a
// root SRV at the address the title sets.
void Lower::flat(const Inst& inst, std::size_t b, std::size_t i) {
    const auto& fl = std::get<gfx10::Flat>(inst.fields);
    const auto& dst = fl.dst;
    this->root_copy = std::nullopt;
    auto name = inst.name();
    {
        auto it = this->scalar.find(Site{b, i});
        if (it != this->scalar.end() && it->second.param == resource::MEMORY) {
            return this->memory_flat(inst, b, i);
        }
    }
    if (!name.starts_with("global_load_")) fail("a global memory write");
    auto rest = name.substr(std::string_view("global_load_").size());
    std::uint32_t n;
    std::optional<std::pair<std::uint32_t, bool>> sub;
    if (rest == "dword") {
        n = 1;
    } else if (rest == "dwordx2") {
        n = 2;
    } else if (rest == "dwordx3") {
        n = 3;
    } else if (rest == "dwordx4") {
        n = 4;
    } else if (rest == "ubyte") {
        n = 1, sub = std::pair{8u, false};
    } else if (rest == "sbyte") {
        n = 1, sub = std::pair{8u, true};
    } else if (rest == "ushort") {
        n = 1, sub = std::pair{16u, false};
    } else if (rest == "sshort") {
        n = 1, sub = std::pair{16u, true};
    } else {
        fail(name.data());
    }
    Address a;
    {
        auto it = this->scalar.find(Site{b, i});
        if (it == this->scalar.end()) fail("an untracked global address");
        a = it->second;
    }
    std::optional<V> guard = this->guard(b, i);
    if (!dst || dst->kind != Operand::Kind::Vector) fail("a global load result");
    std::uint8_t reg = dst->reg;
    std::size_t c = this->cur;
    V off = this->s.m.i32(a.offset);
    if (a.dynamic) {
        V d = this->i32_of(*a.dynamic);
        off = this->s.m.bin(c, Bin::Add, off, d);
    }
    std::optional<V> h;
    if (a.param == resource::CODE) {
        h = std::nullopt;
    } else {
        Source src = Source::Built(a, 0, {0, 0});
        if (auto bias = this->root_bias(a.param)) {
            off = this->s.m.bin(c, Bin::Add, off, *bias);
        }
        if (auto mem = this->root_memory(src, off, guard, false)) {
            off = mem->second;
            h = mem->first;
        } else {
            h = this->handle(Site{b, i}, src, false, {11, 0});
        }
    }
    c = this->cur;
    auto dword = [&](Lower& l, V byte) -> V {
        if (!h) return l.code_dword(byte);
        std::size_t c = l.cur;
        V u = l.s.m.undef(l.s.t.i32);
        V mask = l.s.m.i8(1);
        V al = l.s.m.i32(4);
        V r = l.s.op(c, "rawBufferLoad.f32", Attr::ReadOnly, l.s.t.res_ret, 139,
                     std::vector<V>{*h, byte, u, mask, al});
        V xx = l.s.m.extract(c, r, 0);
        return l.as_int(xx);
    };
    for (std::uint32_t k = 0; k < n; ++k) {
        V xv;
        if (!sub) {
            V four = this->s.m.i32(4 * k);
            V at = this->s.m.bin(c, Bin::Add, off, four);
            xv = dword(*this, at);
        } else {
            auto [width, is_signed] = *sub;
            // shortcut: the field is taken not to cross a dword.
            V not3 = this->s.m.i32(~3u);
            V three = this->s.m.i32(3);
            V at = this->s.m.bin(c, Bin::And, off, not3);
            V w = dword(*this, at);
            V byte = this->s.m.bin(c, Bin::And, off, three);
            V sh = this->s.m.bin(c, Bin::Shl, byte, three);
            V xx = this->s.m.bin(c, Bin::LShr, w, sh);
            V left = this->s.m.i32(32 - width);
            xx = this->s.m.bin(c, Bin::Shl, xx, left);
            xv = this->s.m.bin(c, is_signed ? Bin::AShr : Bin::LShr, xx, left);
        }
        this->write(inst, Reg::V(static_cast<std::uint8_t>(reg + static_cast<std::uint8_t>(k))), xv, guard);
    }
}

// Finds the whole-wave sections and the values they leave.
void Lower::find_sections() {
    const Program& p = *this->x.p;
    HashMap<Value, std::tuple<std::size_t, std::size_t, std::size_t>> inside;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        auto found = wave::sections(p, b);
        // order: SSA values are written once, so `inside` gets the same entries in any order.
        for (const auto& [start, sec] : found) {
            for (std::size_t i = start + 1; i < sec.end; ++i) {
                for (const auto& w : p.blocks[b].insts[i].writes) {
                    inside.insert_or_assign(w.second, std::tuple{b, start, sec.end});
                }
            }
        }
        if (!found.empty()) this->sections.insert_or_assign(b, std::move(found));
    }
    if (inside.empty()) return;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& blk = p.blocks[b];
        for (const auto& phi : blk.phis) {
            for (const Value& a : phi.args) {
                if (inside.contains(a)) this->section_out.insert(a);
            }
        }
        for (std::size_t i = 0; i < blk.insts.size(); ++i) {
            for (const auto& rd : blk.insts[i].reads) {
                auto it = inside.find(rd.second);
                if (it == inside.end()) continue;
                auto [sb, start, end] = it->second;
                if (!(sb == b && i > start && i < end)) this->section_out.insert(rd.second);
            }
        }
    }
}

// An instruction inside a whole-wave section: the values that leave the section are made
// from what the simulation found them to be.
void Lower::section_inst(std::size_t b, std::size_t i, std::size_t start) {
    static constexpr Why UNKNOWN = "a whole-wave section step the simulation does not follow";
    const Inst& inst = this->x.p->blocks[b].insts[i];
    for (const auto& wr : inst.writes) {
        Value w = wr.second;
        Val val;
        if (this->section_out.contains(w)) {
            const auto& shapes = this->sections.at(b).at(start).shapes;
            std::optional<wave::Shape> shape;
            if (auto it = shapes.find(w); it != shapes.end()) shape = it->second;
            bool own = this->own_bits.totals.contains(w);
            if (!shape) {
                val = Val::Missing(UNKNOWN);
            } else {
                try {
                    val = Val::I32(this->shape(*shape, own));
                } catch (const WhyFail& e) {
                    val = Val::Missing(e.why);
                }
            }
        } else {
            val = Val::Missing(UNKNOWN);
        }
        this->vals.insert_or_assign(w, val);
    }
}

// A value a whole-wave section leaves, as DXIL wave operations over the host's active
// lanes: the console's result with console lanes; on another wave size, what the HLSL wave
// operation the console compiler expanded gives on that wave.
//
// `own`: a wave OR that only steers a loop over each lane's own bits (own_bits); where two host
// waves cannot exchange their halves' ORs it is this host wave's.
// wave::Shape is vs-emit's tagged struct (wave.h: a/b/c, k, flag per kind).
V Lower::shape(wave::Shape shape, bool own) {
    using wave::Op;
    using SK = wave::Shape::Kind;
    Op op;
    Value src, mask;
    std::uint32_t ident;
    std::optional<bool> prefix;
    switch (shape.kind) {
        case SK::K:
            return this->s.m.i32(shape.k);
        case SK::Value:
            return this->i32_of(shape.a);
        case SK::Active:
            op = shape.op, src = shape.a, mask = shape.b, ident = shape.k;
            break;
        case SK::ActiveF64:
            return this->active_f64_sum(shape.a, shape.b, shape.c, shape.flag);
        case SK::Masked: {
            std::size_t c = this->cur;
            Bool m = ok_or(mask_of(*this->x.st, shape.b), "a wave reduction's untracked mask");
            V on = this->boolean(m, c);
            V xv = this->i32_of(shape.a);
            V id = this->s.m.i32(shape.k);
            return this->s.m.select(c, on, xv, id);
        }
        case SK::Prefix:
        default:
            op = shape.op, src = shape.a, mask = shape.b, ident = shape.k;
            prefix = shape.flag;
            break;
    }
    // Mesh: no host wave size can be declared, so a reduction is over this host wave only. A
    // minimum or maximum is taken that way: the compiler's "next distinct value" loop (the wave's
    // minimum, then the lanes that hold it) serves every lane however the lanes are divided into
    // waves. A sum, a bit reduction or a prefix would be of part of a console wave: still refused.
    if (this->mask_data == MaskData::None &&
        (prefix.has_value() || !(op == Op::MinU || op == Op::MaxU || op == Op::MinI ||
                                 op == Op::MaxI || op == Op::MinF || op == Op::MaxF))) {
        fail("a wave reduction in a mesh shader");
    }
    std::size_t c = this->cur;
    Bool m = ok_or(mask_of(*this->x.st, mask), "a wave reduction's untracked mask");
    V on = this->boolean(m, c);
    V xv = this->i32_of(src);
    V id = this->s.m.i32(ident);
    xv = this->s.m.select(c, on, xv, id);
    this->s.flags |= flags::WAVE_OPS;
    if (op == Op::AddH) {
        if (prefix) fail("a wave prefix of an operation other than a sum");
        return this->half_sum(xv);
    }
    bool is_float = op == Op::AddF || op == Op::MinF || op == Op::MaxF;
    Ty i32t = this->s.t.i32;
    Ty f32t = this->s.t.f32;
    V arg = is_float ? this->as_float(xv) : xv;
    Ty ty = is_float ? f32t : i32t;
    std::string sfx = is_float ? "f32" : "i32";
    V r;
    if (prefix) {
        if (op == Op::AddU || op == Op::AddF) {
            bool inclusive = *prefix;
            V k0 = this->s.m.i8(0);
            V k1 = this->s.m.i8(1);
            r = this->s.op(c, "wavePrefixOp." + sfx, Attr::NoUnwind, ty, 121,
                           std::vector<V>{arg, k0, k1});
            if (inclusive) r = this->s.m.bin(c, Bin::Add, r, arg);
        } else {
            fail("a wave prefix of an operation other than a sum");
        }
    } else if (op == Op::Or || op == Op::And || op == Op::Xor) {
        V k = this->s.m.i8(op == Op::And ? 0 : op == Op::Or ? 1 : 2);
        r = this->s.op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, std::vector<V>{arg, k});
    } else {
        std::uint8_t code, is_unsigned;
        switch (op) {
            case Op::AddU: code = 0, is_unsigned = 1; break;
            case Op::AddF: code = 0, is_unsigned = 0; break;
            case Op::MinU: code = 2, is_unsigned = 1; break;
            case Op::MaxU: code = 3, is_unsigned = 1; break;
            case Op::MinI:
            case Op::MinF: code = 2, is_unsigned = 0; break;
            default: code = 3, is_unsigned = 0; break;
        }
        V k = this->s.m.i8(code);
        V u = this->s.m.i8(is_unsigned);
        r = this->s.op(c, "waveActiveOp." + sfx, Attr::NoUnwind, ty, 119, std::vector<V>{arg, k, u});
    }
    // two host waves of 32 hold the console's 64 lanes: their results combine
    if (this->local || (this->exchange && std::get<1>(*this->exchange))) {
        // this host wave's own result (and prefix): each wave on its own, or a group of 32
        // threads or fewer, which one host wave holds whole (combining with the 0 a missing half
        // "gives" broke AND, MIN and signed MAX)
    } else if (own && !prefix && op == Op::Or && !this->all_lanes(this->cur_ir)) {
        // each thread walks the bits of this value and works only at its own, which the host
        // wave's OR holds: the same run as with the console wave's OR
    } else if (this->exchange) {
        if (!this->all_lanes(this->cur_ir)) {
            need(this->cur_ir);
            fail("a wave reduction where not every lane runs");
        }
        auto bits = [&](Lower& l, V v) { return is_float ? l.as_int(v) : v; };
        auto back = [&](Lower& l, V v) { return is_float ? l.as_float(v) : v; };
        if (prefix) {
            // the upper half adds the lower half's total to its prefix
            V k0 = this->s.m.i8(0);
            V ku = this->s.m.i8(static_cast<std::uint8_t>(!is_float));
            V total = this->s.op(c, "waveActiveOp." + sfx, Attr::NoUnwind, ty, 119,
                                 std::vector<V>{arg, k0, ku});
            V tb = bits(*this, total);
            std::size_t c2 = this->cur;
            V lo = this->exchange_halves(c2, tb).first;
            lo = back(*this, lo);
            V cl = this->console_lane();
            V k32 = this->s.m.i32(32);
            V upper = this->s.m.cmp(c2, pred::IUGE, cl, k32);
            V zero = is_float ? this->s.m.f32(0) : this->s.m.i32(0);
            V add = this->s.m.select(c2, upper, lo, zero);
            r = this->s.m.bin(c2, Bin::Add, r, add);
        } else {
            V rb = bits(*this, r);
            std::size_t c2 = this->cur;
            auto [lo0, hi0] = this->exchange_halves(c2, rb);
            V lo = back(*this, lo0);
            V hi = back(*this, hi0);
            switch (op) {
                case Op::AddU:
                case Op::AddF: r = this->s.m.bin(c2, Bin::Add, lo, hi); break;
                case Op::Or: r = this->s.m.bin(c2, Bin::Or, lo, hi); break;
                case Op::And: r = this->s.m.bin(c2, Bin::And, lo, hi); break;
                case Op::Xor: r = this->s.m.bin(c2, Bin::Xor, lo, hi); break;
                default: {
                    // FMax 35, FMin 36, IMax 37, IMin 38, UMax 39, UMin 40
                    std::uint32_t code = op == Op::MaxF   ? 35
                                         : op == Op::MinF ? 36
                                         : op == Op::MinU ? 40
                                         : op == Op::MaxU ? 39
                                         : op == Op::MinI ? 38
                                                          : 37;
                    r = this->s.op(c2, "binary." + sfx, Attr::ReadNone, ty, code, std::vector<V>{lo, hi});
                    break;
                }
            }
        }
    }
    return is_float ? this->as_int(r) : r;
}

// `WaveActiveSum` of 64-bit floats (`f64(lo, hi)` in active lanes, `+0.0` elsewhere),
// returning the low (`!high`) or high (`high`) 32-bit word.
V Lower::active_f64_sum(Value lo_, Value hi_, Value mask, bool high) {
    if (this->mask_data == MaskData::None) fail("a wave reduction in a mesh shader");
    std::size_t c = this->cur;
    Bool m = ok_or(mask_of(*this->x.st, mask), "a wave reduction's untracked mask");
    V on = this->boolean(m, c);
    V lo = this->i32_of(lo_);
    V hi = this->i32_of(hi_);
    V z = this->s.m.i32(0);
    lo = this->s.m.select(c, on, lo, z);
    hi = this->s.m.select(c, on, hi, z);
    this->s.flags |= flags::WAVE_OPS | flags::DOUBLES;
    Ty f64t = this->s.m.double_();
    Ty i32t = this->s.t.i32;
    Ty split_ty = this->s.m.named_struct("dx.types.splitdouble", std::vector<Ty>{i32t, i32t});
    V arg = this->s.op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, std::vector<V>{lo, hi});
    V k0 = this->s.m.i8(0);
    V k0b = this->s.m.i8(0);
    V sum = this->s.op(c, "waveActiveOp.f64", Attr::NoUnwind, f64t, 119, std::vector<V>{arg, k0, k0b});
    V split = this->s.op(c, "splitDouble.f64", Attr::ReadNone, split_ty, 102, std::vector<V>{sum});
    V r_lo = this->s.m.extract(c, split, 0);
    V r_hi = this->s.m.extract(c, split, 1);
    if (this->exchange) {
        if (!this->all_lanes(this->cur_ir)) {
            need(this->cur_ir);
            fail("a wave reduction where not every lane runs");
        }
        auto [lo0, lo1] = this->exchange_halves(this->cur, r_lo);
        auto [hi0, hi1] = this->exchange_halves(this->cur, r_hi);
        std::size_t c2 = this->cur;
        V d0 = this->s.op(c2, "makeDouble.f64", Attr::ReadNone, f64t, 101, std::vector<V>{lo0, hi0});
        V d1 = this->s.op(c2, "makeDouble.f64", Attr::ReadNone, f64t, 101, std::vector<V>{lo1, hi1});
        V total = this->s.m.bin(c2, Bin::Add, d0, d1);
        V split2 = this->s.op(c2, "splitDouble.f64", Attr::ReadNone, split_ty, 102, std::vector<V>{total});
        r_lo = this->s.m.extract(c2, split2, 0);
        r_hi = this->s.m.extract(c2, split2, 1);
    }
    return high ? r_hi : r_lo;
}

// The sum of two 16-bit floats (the low halves of `a` and `b`) as the console's
// `v_add_f16` gives it: added in f32 and rounded to 16 bits, to nearest (f16_nearest).
V Lower::add_halves(V a, V b) {
    std::size_t c = this->cur;
    Ty f32t = this->s.t.f32;
    V low = this->s.m.i32(0xFFFF);
    auto wide = [&](Lower& l, V xv) {
        V h = l.s.m.bin(c, Bin::And, xv, low);
        return l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, std::vector<V>{h});
    };
    V fa = wide(*this, a);
    V fb = wide(*this, b);
    V f = this->s.m.bin(c, Bin::Add, fa, fb);
    return this->f16_nearest(f);
}

// `WaveActiveSum` of 16-bit floats (`x`: each lane's term in its low half, +0 in the lanes outside the
// mask), bit for bit as the console adds it: every addition rounds to 16 bits, so the order counts, and the
// compiler's is the balanced tree over the lanes in their order (lanes 2k and 2k+1, then pairs,... then the
// wave's two halves; wave::buddies). Each lane adds its partner's partial sum, read from the lane whose
// number differs in one bit, step by step; two host waves of 32 then exchange their halves' sums for the
// last addition. The high half of the result is lane 0's, which no `v_add_f16` writes. On a host wave of 64
// lanes where not every lane runs, the lanes that are elsewhere count as the console's do (it masks them to
// +0): a partner's partial sum is read from the first lane there is of the partner's run of lanes, and is
// +0 without one. Two host waves of 32 can only exchange where every lane runs: refused elsewhere, in a
// thread group that does not fill its waves, and for each half-wave on its own (half a sum is no sum).
V Lower::half_sum(V xv) {
    static constexpr Why WHERE = "a wave sum of 16-bit floats where not every lane runs";
    auto threads = ok_or(this->s.threads, "a wave sum of 16-bit floats outside compute");
    if (product3(threads) % 64 != 0) fail("a wave sum of 16-bit floats in a wave that is not full");
    if (this->local) fail("a wave sum of 16-bit floats in a half-wave");
    // two host waves of 32 that exchange, or one host wave of 64 lanes
    bool halves_;
    if (this->exchange && !std::get<1>(*this->exchange)) {
        halves_ = true;
    } else if (!this->exchange && this->s.wave_size == std::optional<std::uint32_t>(64) &&
               this->x.p->wave64) {
        halves_ = false;
    } else {
        fail("a wave sum of 16-bit floats on these waves");
    }
    bool every;
    if (halves_) {
        every = this->all_lanes(this->cur_ir);
    } else {
        auto all = all_lane_blocks(*this->x.st);
        every = all && this->cur_ir < all->size() && (*all)[this->cur_ir];
    }
    if (!every && halves_) {
        need(this->cur_ir);
        fail(WHERE);
    }
    std::size_t c = this->cur;
    Ty i32t = this->s.t.i32;
    V lane = this->s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, std::vector<V>{});
    V high = this->s.m.i32(0xFFFF0000u);
    V low = this->s.m.i32(0xFFFF);
    V zero = this->s.m.i32(0);
    // the lanes that are here, when not all are
    std::optional<std::pair<V, V>> here;
    if (!every) {
        this->s.flags |= flags::WAVE_OPS;
        Ty four = this->s.m.named_struct("dx.types.fouri32", std::vector<Ty>(4, i32t));
        V yes = this->s.m.bool_(true);
        V r = this->s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, std::vector<V>{yes});
        V e0 = this->s.m.extract(c, r, 0);
        V e1 = this->s.m.extract(c, r, 1);
        here = std::pair{e0, e1};
    }
    // `v` of the first lane here of the `n` (a power of two) lanes from `base`: +0 of none
    auto first_of = [&](Lower& l, V v, V base, std::uint32_t n) -> V {
        if (!here) return l.read_lane(v, base);
        auto [lo, hi] = *here;
        V k31 = l.s.m.i32(31);
        V k32 = l.s.m.i32(32);
        V lower_ = l.s.m.cmp(c, pred::IULT, base, k32);
        V word = l.s.m.select(c, lower_, lo, hi);
        V shift = l.s.m.bin(c, Bin::And, base, k31);
        word = l.s.m.bin(c, Bin::LShr, word, shift);
        V run = l.s.m.i32(0xFFFFFFFFu >> (32 - n));
        word = l.s.m.bin(c, Bin::And, word, run);
        V any = l.s.m.cmp(c, pred::NE, word, zero);
        V nth = l.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, std::vector<V>{word});
        V from = l.s.m.bin(c, Bin::Add, base, nth);
        // (no lane there: this one, which is here; the read is not used)
        from = l.s.m.select(c, any, from, lane);
        V got = l.read_lane(v, from);
        return l.s.m.select(c, any, got, zero);
    };
    V sum = xv;
    std::uint32_t steps = halves_ ? 5 : 6;
    for (std::uint32_t step = 0; step < steps; ++step) {
        V bit = this->s.m.i32(1u << step);
        V partner = this->s.m.bin(c, Bin::Xor, lane, bit);
        if (!every) {
            V run = this->s.m.i32(~((1u << step) - 1));
            partner = this->s.m.bin(c, Bin::And, partner, run);
        }
        V theirs = first_of(*this, sum, partner, 1u << step);
        sum = this->add_halves(sum, theirs);
    }
    V first = first_of(*this, xv, zero, 1);
    first = this->s.m.bin(c, Bin::And, first, high);
    sum = this->s.m.bin(c, Bin::And, sum, low);
    V mine = this->s.m.bin(c, Bin::Or, first, sum);
    if (!halves_) return mine;
    auto [lo, hi] = this->exchange_halves(c, mine);
    std::size_t c2 = this->cur;
    V total = this->add_halves(lo, hi);
    total = this->s.m.bin(c2, Bin::And, total, low);
    V first2 = this->s.m.bin(c2, Bin::And, lo, high);
    return this->s.m.bin(c2, Bin::Or, first2, total);
}

Lowered Lower::run(bool counted) {
    const Program& p = *this->x.p;
    if (counted) {
        // Before anything else, so that wave counts, reads of the first lane and stores know
        // only the lanes that run.
        std::size_t original = this->s.entry;
        if (!this->s.threads) throw std::logic_error("compute thread count");
        std::uint32_t width = (*this->s.threads)[0];
        V limit = this->host_word(RootType::Other(USER_DISPATCH_THREADS_X), 0, width);
        Val gv = this->compute_id("groupId.i32", 94, 0);
        if (!gv.is_i32()) throw std::logic_error("unreachable");
        V group = gv.v;
        Val tv = this->thread_id(0);
        if (!tv.is_i32()) throw std::logic_error("unreachable");
        V thread = tv.v;
        V widthv = this->s.m.i32(width);
        V first = this->s.m.bin(original, Bin::Mul, group, widthv);
        V xv = this->s.m.bin(original, Bin::Add, first, thread);
        V zero = this->s.m.i32(0);
        V all = this->s.m.cmp(original, pred::EQ, limit, zero);
        V inside = this->s.m.cmp(original, pred::IULT, xv, limit);
        V runs = this->s.m.bin(original, Bin::Or, all, inside);
        std::size_t body = this->s.m.new_block();
        std::size_t done = this->s.m.new_block();
        this->s.m.terminate(original, Term::cond_br(runs, body, done));
        this->s.m.terminate(done, Term::ret());
        this->s.entry = body;
    }
    this->find_coherent();
    auto order = ir::reverse_postorder(p.blocks);
    for (std::size_t n = 0; n < order.size(); ++n) this->rpo[order[n]] = n;
    for (std::size_t n = 0; n < order.size(); ++n) {
        this->block[order[n]] = n == 0 ? this->s.entry : this->s.m.new_block();
    }
    // Loop headers count iterations (`LOOP_LIMIT`).
    for (std::size_t b : order) {
        bool header = false;
        for (std::size_t q : p.blocks[b].preds) {
            if (this->block[q].has_value() && this->rpo[b] <= this->rpo[q]) {
                header = true;
                break;
            }
        }
        if (header) {
            std::size_t blk = this->block[b].value();
            V phi = this->s.m.phi(blk, this->s.t.i32);
            this->counters.insert_or_assign(b, std::pair{phi, std::vector<std::pair<V, std::size_t>>{}});
        }
    }
    this->cur = this->s.entry;
    this->find_sections();
    this->inputs();
    if (this->s.mesh) {
        Mesh mesh = *this->s.mesh;
        try {
            this->mesh_prologue(mesh);
        } catch (const WhyFail& e) {
            unsupported(0, e.why);
        }
        this->block[order[0]] = this->cur;
    }
    for (std::size_t b : order) {
        this->cur = this->block[b].value();
        this->cur_ir = b;
        std::vector<std::size_t> preds;
        for (std::size_t q : p.blocks[b].preds) {
            if (this->block[q].has_value()) preds.push_back(q);
        }
        this->exported[b] = !preds.empty() && std::all_of(preds.begin(), preds.end(), [&](std::size_t q) {
            return this->exported_out(q);
        });
        const auto& phis = p.blocks[b].phis;
        for (std::size_t i = 0; i < phis.size(); ++i) {
            const auto& phi = phis[i];
            Val val;
            if (phi.reg.is_s() && (phi.reg.n == ir::EXEC || phi.reg.n == ir::EXEC + 1)) {
                val = Val::Missing("a lane mask");
            } else if (auto it = this->bad.find(phi.value); it != this->bad.end()) {
                val = Val::Missing(it->second);
            } else if (phi.reg.is_scc()) {
                V v = this->s.m.phi(this->cur, this->s.t.i1);
                this->data_phis.emplace_back(v, b, i);
                val = Val::Bit(v);
            } else {
                V v = this->s.m.phi(this->cur, this->s.t.i32);
                this->data_phis.emplace_back(v, b, i);
                val = Val::I32(v);
            }
            this->vals.insert_or_assign(phi.value, val);
        }
        if (this->lds_meet && this->lds_meet->second.at(b).first) this->lds_meeting();
        std::size_t i = 0;
        while (i < p.blocks[b].insts.size()) {
            this->inst(b, i);
            std::optional<std::size_t> end;
            if (auto sb = this->sections.find(b); sb != this->sections.end()) {
                if (auto si = sb->second.find(i); si != sb->second.end()) end = si->second.end;
            }
            if (end) {
                for (std::size_t j = i + 1; j < *end; ++j) this->section_inst(b, j, i);
                i = *end;
                continue;
            }
            i += 1;
        }
        this->mask_ok = this->mask_data == MaskData::Console;
        if (this->lds_meet && this->lds_meet->second.at(b).second) this->lds_meeting();
        this->tail[b] = this->cur;
        this->exit(b);
    }
    this->fill_phis();
    std::vector<UserData> user_data_;
    for (const auto& [k, u] : this->user_reads) user_data_.push_back(u);
    this->s.samplers = static_cast<std::uint32_t>(this->samplers.size());
    std::vector<std::uint8_t> root_cbvs;
    for (const auto& r : this->s.root_cbvs) root_cbvs.push_back(r.first);
    std::vector<std::uint8_t> root_srvs;
    for (const auto& r : this->s.root_srvs) root_srvs.push_back(r.first);
    std::vector<std::uint8_t> root_uavs;
    for (const auto& r : this->s.root_uavs) root_uavs.push_back(r.first);
    Lowered out;
    out.wave_size = this->s.wave_size;
    out.cmp_samplers = this->s.cmp_samplers;
    out.root_cbvs = std::move(root_cbvs);
    out.root_srvs = std::move(root_srvs);
    out.root_uavs = std::move(root_uavs);
    out.nonuniform_handles = this->s.nonuniform_handles;
    out.wave_ops = (this->s.flags & flags::WAVE_OPS) != 0;
    out.dxil = std::move(this->s).finish();
    out.user_data = std::move(user_data_);
    out.samplers = std::move(this->samplers);
    return out;
}

bool Lower::exported_out(std::size_t b) const { return this->exported[b] || this->has_done_export(b); }

bool Lower::has_done_export(std::size_t b) const {
    for (std::size_t i = 0; i < this->x.p->blocks[b].insts.size(); ++i) {
        auto it = this->exports.find(Site{b, i});
        if (it != this->exports.end() && it->second->done) return true;
    }
    return false;
}

}  // namespace x2w::vs::lower
