// One-to-one SSA form of a native program.
#include "x2w_shader/ir.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace x2w::vs::ir {

using gfx10::Fields;
using gfx10::Flow;
using gfx10::Instruction;
using gfx10::Operand;

std::string Reg::debug() const {
    switch (kind) {
    case Kind::S: return "S(" + std::to_string(n) + ")";
    case Kind::Scc: return "Scc";
    case Kind::V: return "V(" + std::to_string(n) + ")";
    }
    return {};
}

std::string Value::debug() const { return "Value(" + std::to_string(n) + ")"; }

std::vector<std::size_t> Exit::successors() const {
    switch (kind) {
    case Kind::Jump: return {target};
    case Kind::Branch: return {taken, not_taken};
    default: return {};
    }
}

std::string Exit::debug() const {
    switch (kind) {
    case Kind::Jump: return "Jump(" + std::to_string(target) + ")";
    case Kind::Branch:
        return "Branch { taken: " + std::to_string(taken) + ", not_taken: " + std::to_string(not_taken) + " }";
    case Kind::End: return "End";
    case Kind::Indirect: return "Indirect";
    }
    return {};
}

std::string_view Inst::name() const { return machine.name().value_or(""); }

std::optional<Value> Inst::read(Reg reg) const {
    for (const auto& r : reads) {
        if (r.first == reg) return r.second;
    }
    return std::nullopt;
}

std::optional<Value> Inst::write(Reg reg) const {
    for (const auto& w : writes) {
        if (w.first == reg) return w.second;
    }
    return std::nullopt;
}

namespace {

using detail::Effects;
using detail::NONE;

// Scalar codes 0-255 (from PARKED up: parked scalars), SCC, VGPRs 0-255.
constexpr std::size_t REGS = 256 + 1 + 256;
constexpr std::size_t SET_WORDS = (REGS + 63) / 64;
using RegSet = std::array<std::uint64_t, SET_WORDS>;

std::size_t index(Reg reg) {
    switch (reg.kind) {
    case Reg::Kind::S: return reg.n;
    case Reg::Kind::Scc: return 256;
    case Reg::Kind::V: return 257 + std::size_t(reg.n);
    }
    return 0;
}

Reg reg_at(std::size_t i) {
    if (i < 256) return Reg::S(std::uint8_t(i));
    if (i == 256) return Reg::Scc;
    return Reg::V(std::uint8_t(i - 257));
}

void insert(RegSet& set, Reg reg) {
    std::size_t i = index(reg);
    set[i / 64] |= std::uint64_t{1} << (i % 64);
}

bool contains(const RegSet& set, Reg reg) {
    std::size_t i = index(reg);
    return ((set[i / 64] >> (i % 64)) & 1) != 0;
}

// In Reg order, which is also index order.
std::vector<Reg> members(const RegSet& set) {
    std::vector<Reg> out;
    for (std::size_t i = 0; i < REGS; ++i) {
        if (((set[i / 64] >> (i % 64)) & 1) != 0) out.push_back(reg_at(i));
    }
    return out;
}

// `outs` is sorted by register, and holds every register live out.
Value lookup(const std::vector<std::pair<Reg, Value>>& outs, Reg reg) {
    auto it = std::lower_bound(outs.begin(), outs.end(), reg,
                               [](const std::pair<Reg, Value>& e, Reg r) { return e.first < r; });
    if (it != outs.end() && it->first == reg) return it->second;
    return NONE;
}

// Registers live into and out of each block.
std::pair<std::vector<RegSet>, std::vector<RegSet>> liveness(const std::vector<Block>& blocks,
                                                             const std::vector<std::size_t>& order) {
    RegSet empty{};
    std::vector<RegSet> uses(blocks.size(), empty), defs(blocks.size(), empty);
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        for (const Inst& inst : blocks[b].insts) {
            for (const auto& [reg, _] : inst.reads) {
                if (!contains(defs[b], reg)) insert(uses[b], reg);
            }
            for (const auto& [reg, _] : inst.writes) insert(defs[b], reg);
        }
    }
    std::vector<RegSet> live_in(blocks.size(), empty), live_out(blocks.size(), empty);
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            std::size_t b = *it;
            RegSet out = empty;
            for (std::size_t s : blocks[b].exit.successors()) {
                for (std::size_t w = 0; w < SET_WORDS; ++w) out[w] |= live_in[s][w];
            }
            RegSet inn = empty;
            for (std::size_t w = 0; w < SET_WORDS; ++w) inn[w] = uses[b][w] | (out[w] & ~defs[b][w]);
            if (inn != live_in[b] || out != live_out[b]) {
                live_in[b] = inn;
                live_out[b] = out;
                changed = true;
            }
        }
    }
    return {std::move(live_in), std::move(live_out)};
}

// A loop among the `inside` blocks entered at more than one block, with its entries. Loops nested in
// a single-entry loop are searched with its header removed, so one is found whenever the control
// flow is irreducible.
std::optional<std::pair<std::vector<bool>, std::vector<std::size_t>>> multi_entry_loop(
    const std::vector<Block>& blocks, const std::vector<bool>& inside) {
    auto reach = [&](std::size_t from, bool forward) {
        std::vector<bool> seen(blocks.size(), false);
        std::vector<std::size_t> stack{from};
        while (!stack.empty()) {
            std::size_t b = stack.back();
            stack.pop_back();
            std::vector<std::size_t> next = forward ? blocks[b].exit.successors() : blocks[b].preds;
            for (std::size_t n : next) {
                if (inside[n] && !seen[n]) {
                    seen[n] = true;
                    stack.push_back(n);
                }
            }
        }
        return seen;
    };
    std::vector<bool> done(blocks.size(), false);
    for (std::size_t v = 0; v < blocks.size(); ++v) {
        if (!inside[v] || done[v]) continue;
        // Blocks on a cycle through `v`; `v` itself only if there is one.
        std::vector<bool> ahead = reach(v, true), behind = reach(v, false);
        std::vector<bool> region(blocks.size());
        for (std::size_t i = 0; i < blocks.size(); ++i) region[i] = ahead[i] && behind[i];
        if (!region[v]) continue;
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            if (region[b]) done[b] = true;
        }
        std::vector<std::size_t> entries;
        for (std::size_t b = 0; b < blocks.size(); ++b) {
            if (!region[b]) continue;
            for (std::size_t p : blocks[b].preds) {
                if (!region[p]) {
                    entries.push_back(b);
                    break;
                }
            }
        }
        if (entries.empty()) continue;
        if (entries.size() == 1) {
            std::vector<bool> nested = region;
            nested[entries[0]] = false;
            if (auto found = multi_entry_loop(blocks, nested)) return found;
            continue;
        }
        return std::make_pair(std::move(region), std::move(entries));
    }
    return std::nullopt;
}

// Scheduling hints, performance counters and tracing: no effect on results.
constexpr std::string_view DROPPED[] = {
    "s_nop",
    "s_waitcnt",
    "s_waitcnt_vscnt",
    "s_waitcnt_vmcnt",
    "s_waitcnt_expcnt",
    "s_waitcnt_lgkmcnt",
    "s_waitcnt_depctr",
    "s_wait_idle",
    "s_inst_prefetch",
    "s_clause",
    "s_setprio",
    "s_sleep",
    "s_ttracedata",
    "s_ttracedata_imm",
    "s_incperflevel",
    "s_decperflevel",
    "v_nop",
    "v_pipeflush",
    "v_clrexcp",
};

bool contains_str(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

// Scalar ALU opcodes that set SCC (carry, overflow, comparison, or "result is not zero").
bool writes_scc(gfx10::Encoding encoding, std::string_view name) {
    using gfx10::Encoding;
    if (encoding == Encoding::Sopc) return true;
    if (encoding == Encoding::Sopk) return name.starts_with("s_cmpk_") || name == "s_addk_i32";
    // name.split('_').nth(1)
    std::string_view stem;
    std::size_t a = name.find('_');
    if (a != std::string_view::npos) {
        std::size_t b = name.find('_', a + 1);
        stem = name.substr(a + 1, b == std::string_view::npos ? std::string_view::npos : b - a - 1);
    }
    static constexpr std::string_view STEMS[] = {
        "add",   "sub",   "addc",  "subb",  "min",   "max",   "and",   "or",    "xor",   "andn2",
        "orn2",  "nand",  "nor",   "xnor",  "lshl",  "lshr",  "ashr",  "bfe",   "absdiff", "lshl1",
        "lshl2", "lshl3", "lshl4", "not",   "wqm",   "bcnt0", "bcnt1", "quadmask", "abs",
    };
    return contains_str(name, "saveexec") || contains_str(name, "wrexec") ||
           std::find(std::begin(STEMS), std::end(STEMS), stem) != std::end(STEMS);
}

// What an atomic with `glc` returns: the old value (for compare-and-swap, half the data operand).
Operand atomic_result(const Operand& data, std::string_view name) {
    if (data.kind == Operand::Kind::Vector && contains_str(name, "cmpswap")) {
        return Operand::Vector(data.reg, std::uint8_t(data.dwords / 2));
    }
    return data;
}

// Appends the register dwords an operand names; reading `vccz`, `execz`, `scc` or `lds_direct` reads the
// register behind it. False: a range out of bounds.
bool add_regs(const Operand& op, std::uint8_t lanes, std::vector<Reg>& out) {
    switch (op.kind) {
    case Operand::Kind::Scalar: {
        if (op.code == gfx10::NULL_CODE) return true;
        unsigned end = unsigned(op.code) + unsigned(op.dwords);
        // A machine operand ends within the machine's codes, a parked scalar (`unspill`) within those
        // past them.
        unsigned limit = op.code < PARKED ? 128 : 256;
        if (end > limit) return false;
        for (unsigned c = op.code; c < end; ++c) out.push_back(Reg::S(std::uint8_t(c)));
        return true;
    }
    case Operand::Kind::Vector:
        if (unsigned(op.reg) + unsigned(op.dwords) > 256) return false;
        for (unsigned i = 0; i < op.dwords; ++i) out.push_back(Reg::V(std::uint8_t(op.reg + i)));
        return true;
    case Operand::Kind::Const:
        if (op.code == 251) {
            for (unsigned i = 0; i < lanes; ++i) out.push_back(Reg::S(std::uint8_t(VCC + i)));
        } else if (op.code == 252) {
            for (unsigned i = 0; i < lanes; ++i) out.push_back(Reg::S(std::uint8_t(EXEC + i)));
        } else if (op.code == 253) {
            out.push_back(Reg::Scc);
        } else if (op.code == 254) {
            out.push_back(Reg::S(M0));
        }
        return true;
    default:
        return true;
    }
}

template <class T>
bool vec_contains(const std::vector<T>& v, const T& x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

// (save, N, X, K) of `v_writelane_b32 vN, sX, K` / `v_readlane_b32 sX, vN, K`.
struct Park {
    bool save;
    std::uint8_t reg;
    std::uint8_t code;
    std::uint32_t lane;
    bool operator==(const Park&) const = default;
};

std::optional<Park> parked(const Instruction& inst, const Fields& fields) {
    const gfx10::Valu* v = std::get_if<gfx10::Valu>(&fields);
    if (!v) return std::nullopt;
    if (v->abs != 0 || v->neg != 0 || v->opsel != 0 || v->clamp || v->omod != 0 || v->sdwa || v->dpp) {
        return std::nullopt;
    }
    auto lane = [](const Operand& op) -> std::optional<std::uint32_t> {
        if (op.kind == Operand::Kind::Const && op.code >= 128 && op.code <= 191) return std::uint32_t(op.code) - 128;
        return std::nullopt;
    };
    auto name = inst.name();
    if (!name || !v->dst || v->src.size() != 2) return std::nullopt;
    const Operand& dst = *v->dst;
    const Operand& s0 = v->src[0];
    if (*name == "v_writelane_b32" && dst.kind == Operand::Kind::Vector && s0.kind == Operand::Kind::Scalar &&
        s0.code < 105) {
        auto k = lane(v->src[1]);
        if (!k) return std::nullopt;
        return Park{true, dst.reg, s0.code, *k};
    }
    if (*name == "v_readlane_b32" && dst.kind == Operand::Kind::Scalar && s0.kind == Operand::Kind::Vector &&
        dst.code < 105) {
        auto k = lane(v->src[1]);
        if (!k) return std::nullopt;
        return Park{false, s0.reg, dst.code, *k};
    }
    return std::nullopt;
}

struct Move {
    std::size_t index;
    std::uint8_t dst;
    std::uint8_t src;
    auto operator<=>(const Move&) const = default;
};

// Scalar registers the compiler parked in lanes of a VGPR, as scalar moves. A VGPR used only by
// `v_writelane_b32 vN, sX, K` / `v_readlane_b32 sX, vN, K` with constant lanes is a store of scalars; each
// (VGPR, lane) gets a spare register code and the save and restore become `s_mov_b32` (or `s_mov_b64` for
// lane pairs in wave64) to and from it. Returns the moves (index in `list`, destination code, source code)
// in order.
std::vector<Move> unspill(std::vector<Instruction>& list, const std::vector<Flow>& flows,
                          const std::vector<std::size_t>& leaders, bool wave64, std::uint8_t lanes,
                          std::uint16_t vgprs) {
    const std::size_t len = list.size();
    // VGPRs used by other than parking (the last such use), the scalar codes the program uses, and
    // per instruction the scalar codes it writes and reads.
    std::array<std::optional<std::size_t>, 256> other{};
    std::array<bool, 256> used{};
    std::vector<std::optional<Park>> parks;
    std::vector<std::vector<std::uint8_t>> written, read;
    parks.reserve(len);
    written.reserve(len);
    read.reserve(len);
    for (const Instruction& inst : list) {
        auto name = inst.name();
        if (!name) return {};
        Fields fields = inst.fields(wave64);
        std::optional<Park> park = parked(inst, fields);
        std::optional<Effects> effects;
        try {
            effects = detail::effects(inst, fields, *name, lanes, vgprs);
        } catch (const ShaderError&) {
            return {};
        }
        std::vector<std::uint8_t> codes, from;
        if (effects) {
            const auto& [reads, writes] = *effects;
            for (Reg reg : writes) {
                if (reg.is_s()) codes.push_back(reg.n);
            }
            for (Reg reg : reads) {
                if (reg.is_s()) from.push_back(reg.n);
            }
            // An M0-relative read (`v_movrels_b32`) names every VGPR from its base up; the compiler
            // indexes a small array with it, never the lanes it parks scalars in, so it does not
            // make those VGPRs ordinary ones. (Relative writes still do.)
            bool relative = *name == "v_movrels_b32";
            std::size_t n_reads = reads.size();
            for (std::size_t k = 0; k < reads.size() + writes.size(); ++k) {
                Reg reg = k < n_reads ? reads[k] : writes[k - n_reads];
                if (relative && k < n_reads && reg.is_v()) continue;
                if (reg.is_s()) {
                    used[reg.n] = true;
                } else if (reg.is_v()) {
                    // The parking instruction's VGPR acts as storage.
                    if (!(park && park->reg == reg.n)) other[reg.n] = parks.size();
                }
            }
        }
        parks.push_back(park);
        written.push_back(std::move(codes));
        read.push_back(std::move(from));
    }
    auto leader = [&](std::size_t i) { return std::binary_search(leaders.begin(), leaders.end(), list[i].offset); };
    for (std::size_t reg = 0; reg < 256; ++reg) {
        if (!other[reg]) continue;
        std::size_t last = *other[reg];
        auto of = [&](const std::optional<Park>& p) -> std::optional<Park> {
            if (p && p->reg == reg) return p;
            return std::nullopt;
        };
        std::optional<std::size_t> first;
        for (std::size_t i = 0; i < parks.size(); ++i) {
            auto p = of(parks[i]);
            if (p && p->save) {
                first = i;
                break;
            }
        }
        bool late = false;
        if (first) {
            std::size_t end = *first;
            for (std::size_t i = parks.size(); i-- > 0;) {
                if (of(parks[i])) {
                    end = i;
                    break;
                }
            }
            late = last < *first;
            if (late) {
                for (std::size_t i = *first + 1; i <= end; ++i) {
                    if (leader(i)) {
                        late = false;
                        break;
                    }
                }
            }
            if (late) {
                std::vector<std::uint32_t> saved;
                for (std::size_t i = *first; i <= end; ++i) {
                    auto p = of(parks[i]);
                    if (!p) continue;
                    if (p->save) saved.push_back(p->lane);
                    if (!vec_contains(saved, p->lane)) {
                        late = false;
                        break;
                    }
                }
            }
        }
        std::size_t from = late ? *first : parks.size();
        for (std::size_t i = 0; i < from; ++i) {
            if (of(parks[i])) parks[i] = std::nullopt;
        }
    }
    if (std::all_of(parks.begin(), parks.end(), [](const auto& p) { return !p; })) return {};
    // Lane pairs (register, lower lane) that always move together: each save of the lower lane is
    // followed, in the same block and before its register changes, by the save of the next register
    // into the next lane; each restore of the lower lane is followed at once by the restore of the
    // next lane into the next register. Only wave64 programs have pairs.
    // instruction -> the instruction that completes its pair
    auto partner = [&](std::size_t i) -> std::optional<std::size_t> {
        if (!parks[i]) return std::nullopt;
        auto [save, reg, code, lane] = *parks[i];
        Park want{save, reg, std::uint8_t(code + 1), lane + 1};
        for (std::size_t j = i + 1; j < len && !leader(j); ++j) {
            if (parks[j] == want) return j;
            const auto& p = parks[j];
            bool lanes_touched = p && p->reg == reg && (p->lane == lane || p->lane == lane + 1);
            if (save) {
                if (lanes_touched || vec_contains(written[j], code) || vec_contains(written[j], std::uint8_t(code + 1))) {
                    return std::nullopt;
                }
            } else {
                // The second restore may come a few instructions later: the pair is restored at the
                // first, so nothing between may read or write the second register, write the first,
                // or save into the lanes.
                if ((lanes_touched && p->save) || vec_contains(written[j], code) ||
                    vec_contains(written[j], std::uint8_t(code + 1)) || vec_contains(read[j], std::uint8_t(code + 1))) {
                    return std::nullopt;
                }
            }
        }
        return std::nullopt;
    };
    using Slot = std::pair<std::uint8_t, std::uint32_t>;
    auto slot = [&](std::size_t i) -> std::optional<Slot> {
        if (!parks[i]) return std::nullopt;
        return Slot{parks[i]->reg, parks[i]->lane};
    };
    auto at = [&](std::size_t offset) {
        return std::size_t(std::partition_point(list.begin(), list.end(),
                                                [&](const Instruction& x) { return x.offset < offset; }) -
                           list.begin());
    };
    // The store each parking instruction uses, and what the stores are lanes of: one per (VGPR,
    // lane), or (`split`) one per life of it.
    auto stores_of = [&](bool split) -> std::pair<std::vector<std::size_t>, std::vector<Slot>> {
        // Instructions of one store, by the first of them.
        std::vector<std::size_t> first(len);
        for (std::size_t i = 0; i < len; ++i) first[i] = i;
        if (split) {
            auto root = [&](std::size_t i) {
                while (first[i] != i) {
                    first[i] = first[first[i]];
                    i = first[i];
                }
                return i;
            };
            for (std::size_t save = 0; save < len; ++save) {
                if (!(parks[save] && parks[save]->save)) continue;
                std::unordered_set<std::size_t> seen;
                std::vector<std::size_t> work{save + 1};
                while (!work.empty()) {
                    std::size_t from = work.back();
                    work.pop_back();
                    for (std::size_t i = from; i < len; ++i) {
                        if (slot(i) && slot(i) == slot(save)) {
                            if (parks[i] && parks[i]->save) break;
                            std::size_t a = root(save), b = root(i);
                            first[std::max(a, b)] = std::min(a, b);
                        }
                        auto go = [&](std::size_t to) {
                            if (seen.insert(to).second) work.push_back(to);
                        };
                        const Flow& f = flows[i];
                        if (f.kind == Flow::Kind::Next) {
                            if (i + 1 < len && leader(i + 1)) {
                                go(i + 1);
                            } else {
                                continue;
                            }
                        } else if (f.kind == Flow::Kind::Jump) {
                            go(at(f.target));
                        } else if (f.kind == Flow::Kind::Branch) {
                            go(at(f.target));
                            go(i + 1);
                        }
                        break;
                    }
                }
            }
            for (std::size_t i = 0; i < len; ++i) first[i] = root(i);
        } else {
            std::unordered_map<std::uint64_t, std::size_t> of;
            for (std::size_t i = 0; i < len; ++i) {
                if (auto s = slot(i)) {
                    std::uint64_t key = std::uint64_t(s->first) << 32 | s->second;
                    first[i] = of.emplace(key, i).first->second;
                }
            }
        }
        std::vector<std::pair<Slot, std::size_t>> stores;
        for (std::size_t i = 0; i < len; ++i) {
            if (first[i] == i) {
                if (auto s = slot(i)) stores.push_back({*s, i});
            }
        }
        std::sort(stores.begin(), stores.end());
        auto number = [&](std::size_t i) -> std::optional<std::size_t> {
            for (std::size_t k = 0; k < stores.size(); ++k) {
                if (stores[k].second == first[i]) return k;
            }
            return std::nullopt;
        };
        std::vector<std::size_t> of(len, SIZE_MAX);
        for (std::size_t i = 0; i < len; ++i) {
            if (parks[i]) {
                if (auto k = number(i)) of[i] = *k;
            }
        }
        std::vector<Slot> slots;
        for (const auto& s : stores) slots.push_back(s.first);
        return {std::move(of), std::move(slots)};
    };
    struct Plan {
        std::vector<std::size_t> of;
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        std::vector<std::uint8_t> codes;
    };
    // The stores that move as pairs (lower, upper) and every store's code; none if the spare codes
    // do not suffice.
    auto plan = [&](bool split) -> std::optional<Plan> {
        auto [of, stores] = stores_of(split);
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        for (std::size_t n = 0; n < stores.size(); ++n) {
            auto [reg, lane] = stores[n];
            if (!wave64 || std::any_of(pairs.begin(), pairs.end(), [&](const auto& p) { return p.second == n; })) {
                continue;
            }
            std::vector<std::size_t> lower;
            for (std::size_t i = 0; i < len; ++i) {
                if (of[i] == n) lower.push_back(i);
            }
            std::vector<std::optional<std::size_t>> partners;
            for (std::size_t i : lower) partners.push_back(partner(i));
            std::vector<std::size_t> distinct;
            for (const auto& p : partners) {
                if (p) distinct.push_back(*p);
            }
            std::sort(distinct.begin(), distinct.end());
            distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
            if (distinct.empty()) continue;
            std::size_t upper = of.at(distinct[0]);
            bool all_some = std::all_of(partners.begin(), partners.end(), [](const auto& p) { return p.has_value(); });
            bool all_upper = std::all_of(distinct.begin(), distinct.end(), [&](std::size_t j) { return of[j] == upper; });
            std::size_t count = std::size_t(std::count(of.begin(), of.end(), upper));
            if (stores.at(upper) == Slot{reg, lane + 1} && all_some && distinct.size() == lower.size() && all_upper &&
                count == lower.size()) {
                pairs.push_back({n, upper});
            }
        }
        // Lives of one lane that move alike share a code, as they share the lane.
        using Role = std::pair<Slot, std::uint8_t>;
        auto role = [&](std::size_t n) {
            bool lo = std::any_of(pairs.begin(), pairs.end(), [&](const auto& p) { return p.first == n; });
            bool up = std::any_of(pairs.begin(), pairs.end(), [&](const auto& p) { return p.second == n; });
            return Role{stores[n], std::uint8_t((lo ? 1 : 0) + 2 * (up ? 1 : 0))};
        };
        std::vector<Role> shared;
        for (std::size_t n = 0; n < stores.size(); ++n) shared.push_back(role(n));
        std::vector<Role> all = shared;
        std::sort(shared.begin(), shared.end());
        shared.erase(std::unique(shared.begin(), shared.end()), shared.end());
        std::vector<std::size_t> place;
        for (const Role& r : all) {
            auto it = std::lower_bound(shared.begin(), shared.end(), r);
            if (it != shared.end() && *it == r) place.push_back(std::size_t(it - shared.begin()));
        }
        for (std::size_t& o : of) {
            if (o != SIZE_MAX) o = place[o];
        }
        for (auto& p : pairs) p = {place[p.first], place[p.second]};
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
        // Codes for the parked scalars: the trap temporaries (108..=123), then user SGPRs the
        // program never names, from the top, then the codes past the machine's (PARKED..=254; 255
        // stays unused so that the code after a parked one always exists).
        const std::size_t past = PARKED;
        std::array<bool, 256> free{};
        for (std::size_t c = 0; c < 105; ++c) free[c] = !used[c];
        for (std::size_t c = 108; c <= 123; ++c) free[c] = !used[c];
        for (std::size_t c = past; c < 255; ++c) free[c] = true;
        std::vector<std::optional<std::uint8_t>> codes(shared.size());
        for (const auto& [lo, up] : pairs) {
            std::optional<std::size_t> found;
            auto try_pair = [&](std::size_t c) {
                if (!found && free[c] && free[c + 1]) found = c;
            };
            for (std::size_t c = 108; c < 123; ++c) try_pair(c);
            for (std::size_t c = 104; c-- > 0;) try_pair(c);
            for (std::size_t c = past; c < 254; ++c) try_pair(c);
            // Not enough spare codes: leave the program as it is.
            if (!found) return std::nullopt;
            std::size_t c = *found;
            free[c] = false;
            free[c + 1] = false;
            codes[lo] = std::uint8_t(c);
            codes[up] = std::uint8_t(c + 1);
        }
        for (auto& code : codes) {
            if (code) continue;
            std::optional<std::size_t> found;
            auto try_one = [&](std::size_t c) {
                if (!found && free[c]) found = c;
            };
            for (std::size_t c = 108; c <= 123; ++c) try_one(c);
            for (std::size_t c = 105; c-- > 0;) try_one(c);
            for (std::size_t c = past; c < 255; ++c) try_one(c);
            if (!found) return std::nullopt;
            free[*found] = false;
            code = std::uint8_t(*found);
        }
        Plan out;
        out.of = std::move(of);
        out.pairs = std::move(pairs);
        for (const auto& c : codes) out.codes.push_back(*c);
        return out;
    };
    std::optional<Plan> chosen = plan(false);
    if (!chosen) return {};
    auto lonely = [&](const Plan& pl) {
        for (std::size_t i = 0; i < len; ++i) {
            if (parks[i] && parks[i]->save && partner(i) &&
                !std::any_of(pl.pairs.begin(), pl.pairs.end(),
                             [&](const auto& p) { return pl.of[i] == p.first || pl.of[i] == p.second; })) {
                return true;
            }
        }
        return false;
    };
    if (wave64 && lonely(*chosen)) {
        if (std::optional<Plan> split = plan(true); split && split->pairs.size() > chosen->pairs.size()) {
            chosen = std::move(split);
        }
    }
    const std::vector<std::size_t>& store = chosen->of;
    const auto& pairs = chosen->pairs;
    const std::vector<std::uint8_t>& codes = chosen->codes;
    // `s_mov_b32` / `s_mov_b64 dst, src` (SOP1, opcodes 3 and 4); `s_nop`. A code from PARKED up
    // does not fit the word: s0 stands in for it there.
    auto field = [](std::uint8_t code) -> std::uint32_t { return code < PARKED ? code : 0; };
    auto mov = [&](bool wide, std::uint8_t dst, std::uint8_t src) -> std::uint32_t {
        return 0xBE80'0000u | field(dst) << 16 | (3u + (wide ? 1u : 0u)) << 8 | field(src);
    };
    constexpr std::uint32_t NOP = 0xBF80'0000u;
    struct Edit {
        std::size_t at;
        std::optional<std::tuple<bool, std::uint8_t, std::uint8_t>> mov;
    };
    std::vector<Edit> edits;
    for (std::size_t i = 0; i < len; ++i) {
        if (!parks[i]) continue;
        bool save = parks[i]->save;
        std::uint8_t code = parks[i]->code;
        std::uint8_t slot_code = codes.at(store[i]);
        if (std::any_of(pairs.begin(), pairs.end(), [&](const auto& p) { return p.first == store[i]; })) {
            auto j = partner(i);
            if (!j) return {};
            // A save moves the pair where its second half is saved; a restore where its first half
            // is restored.
            std::size_t at_i = save ? *j : i, gone = save ? i : *j;
            std::uint8_t dst = save ? slot_code : code, src = save ? code : slot_code;
            edits.push_back({at_i, std::make_tuple(true, dst, src)});
            edits.push_back({gone, std::nullopt});
        } else if (std::any_of(pairs.begin(), pairs.end(), [&](const auto& p) { return p.second == store[i]; })) {
            // the upper half of a pair: moved with the lower
        } else {
            std::uint8_t dst = save ? slot_code : code, src = save ? code : slot_code;
            edits.push_back({i, std::make_tuple(false, dst, src)});
        }
    }
    std::vector<Move> moves;
    for (const Edit& e : edits) {
        if (e.mov) {
            auto [wide, dst, src] = *e.mov;
            list[e.at] = Instruction(list[e.at].offset, gfx10::Encoding::Sop1, std::uint16_t(3 + (wide ? 1 : 0)),
                                     {mov(wide, dst, src)});
            moves.push_back({e.at, dst, src});
        } else {
            list[e.at] = Instruction(list[e.at].offset, gfx10::Encoding::Sopp, 0, {NOP});
        }
    }
    std::sort(moves.begin(), moves.end());
    return moves;
}

std::vector<std::pair<Reg, Value>> unset(const std::vector<Reg>& regs) {
    std::vector<std::pair<Reg, Value>> out;
    out.reserve(regs.size());
    for (Reg r : regs) out.push_back({r, NONE});
    return out;
}

void retarget(Exit& exit, const std::function<std::size_t(std::size_t)>& to) {
    if (exit.kind == Exit::Kind::Jump) {
        exit.target = to(exit.target);
    } else if (exit.kind == Exit::Kind::Branch) {
        exit.taken = to(exit.taken);
        exit.not_taken = to(exit.not_taken);
    }
}

bool all_sopp(const Block& b) {
    return std::all_of(b.insts.begin(), b.insts.end(),
                       [](const Inst& i) { return std::holds_alternative<gfx10::Sopp>(i.fields); });
}

// Clears the phis and renames after the exits changed; nullopt if the flow became irreducible beyond
// repair.
std::optional<Program> relink(std::vector<Block> blocks, const Program& p) {
    for (Block& block : blocks) block.phis.clear();
    detail::link(blocks);
    try {
        detail::split_irreducible(blocks);
    } catch (const ShaderError&) {
        return std::nullopt;
    }
    return detail::rename(std::move(blocks), p.wave64, p.vgprs);
}

}  // namespace

namespace detail {

void link(std::vector<Block>& blocks) {
    for (Block& block : blocks) block.preds.clear();
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        for (std::size_t s : blocks[b].exit.successors()) blocks.at(s).preds.push_back(b);
    }
}

// Node splitting, before SSA. While a loop is entered at several blocks (a compiler-rotated loop
// whose peeled first test jumps into its body), the entry first in reverse postorder stays the
// header, and another entry's outside predecessors move to a copy of everything that entry reaches
// in the loop before the header. The copies run the same instructions, so behaviour is unchanged.
void split_irreducible(std::vector<Block>& blocks) {
    const std::size_t limit = 4 * blocks.size() + 64;
    for (;;) {
        std::vector<std::size_t> order = reverse_postorder(blocks);
        std::vector<std::size_t> idx(blocks.size(), SIZE_MAX);
        for (std::size_t i = 0; i < order.size(); ++i) idx[order[i]] = i;
        auto edge = irreducible_edge(blocks, order, idx);
        if (!edge) return;
        std::size_t from = edge->first;
        std::optional<std::pair<std::vector<bool>, std::vector<std::size_t>>> found;
        found = multi_entry_loop(blocks, std::vector<bool>(blocks.size(), true));
        if (!found || !(blocks.size() < limit)) {
            throw ShaderError::unsupported(blocks[from].offset, "irreducible control flow");
        }
        std::vector<bool> region = std::move(found->first);
        std::vector<std::size_t> entries = std::move(found->second);
        std::stable_sort(entries.begin(), entries.end(), [&](std::size_t a, std::size_t b) { return idx[a] < idx[b]; });
        std::size_t header = entries[0], entry = entries[1];

        // What `entry` reaches in the loop before the header, and its copies.
        std::vector<std::size_t> copy(blocks.size(), SIZE_MAX);
        std::vector<std::size_t> copied;
        std::vector<std::size_t> stack{entry};
        while (!stack.empty()) {
            std::size_t b = stack.back();
            stack.pop_back();
            if (copy[b] != SIZE_MAX) continue;
            copy[b] = blocks.size() + copied.size();
            copied.push_back(b);
            for (std::size_t s : blocks[b].exit.successors()) {
                if (region[s] && s != header) stack.push_back(s);
            }
        }
        for (std::size_t b : copied) {
            Block block = blocks[b];
            retarget(block.exit, [&](std::size_t s) { return copy[s] == SIZE_MAX ? s : copy[s]; });
            blocks.push_back(std::move(block));
        }
        std::vector<std::size_t> outside;
        for (std::size_t p : blocks[entry].preds) {
            if (!region[p]) outside.push_back(p);
        }
        for (std::size_t p : outside) {
            retarget(blocks[p].exit, [&](std::size_t s) { return s == entry ? copy[entry] : s; });
        }
        link(blocks);
    }
}

// Rename register reads and writes after changing edges in a reducible graph.
Program rename(std::vector<Block> blocks, bool wave64, std::uint16_t vgprs) {
    std::vector<std::size_t> order = reverse_postorder(blocks);
    auto [live_in, live_out] = liveness(blocks, order);
    std::uint32_t values = 0;
    auto new_value = [&]() { return Value{values++}; };
    std::vector<std::pair<Reg, Value>> inputs;
    std::vector<std::vector<std::pair<Reg, Value>>> outs(blocks.size());
    std::vector<Value> current(REGS, NONE);
    for (std::size_t b : order) {
        std::fill(current.begin(), current.end(), NONE);
        Block& block = blocks[b];
        for (Reg reg : members(live_in[b])) {
            Value v;
            if (block.preds.empty()) {
                v = new_value();
                inputs.push_back({reg, v});
            } else if (block.preds.size() == 1) {
                // A single predecessor comes earlier in reverse postorder.
                v = lookup(outs[block.preds[0]], reg);
            } else {
                v = new_value();
                block.phis.push_back(Phi{reg, v, {}});
            }
            current[index(reg)] = v;
        }
        for (Inst& inst : block.insts) {
            for (auto& [reg, v] : inst.reads) v = current[index(reg)];
            for (auto& [reg, v] : inst.writes) {
                v = new_value();
                current[index(reg)] = v;
            }
        }
        std::vector<std::pair<Reg, Value>> out;
        for (Reg reg : members(live_out[b])) out.push_back({reg, current[index(reg)]});
        outs[b] = std::move(out);
    }
    for (Block& block : blocks) {
        for (Phi& phi : block.phis) {
            phi.args.clear();
            for (std::size_t p : block.preds) phi.args.push_back(lookup(outs[p], phi.reg));
        }
    }
    remove_trivial_phis_steps(blocks, values);
    Program p;
    p.wave64 = wave64;
    p.vgprs = vgprs;
    p.blocks = std::move(blocks);
    p.inputs = std::move(inputs);
    return p;
}

// Phis are placed for every live register at every join; those whose arguments are all one value
// (or the phi itself) are replaced by it.
std::size_t remove_trivial_phis_steps(std::vector<Block>& blocks, std::uint32_t values) {
    std::vector<Value> replaced(values);
    for (std::uint32_t i = 0; i < values; ++i) replaced[i] = Value{i};
    std::size_t steps = 0;
    // An unrenamed path can leave NONE here. A checked lookup rejects the out-of-range value.
    auto find = [&](Value v) {
        Value root = v;
        while (replaced.at(root.n) != root) {
            ++steps;
            root = replaced[root.n];
        }
        Value cur = v;
        while (replaced.at(cur.n) != root) {
            ++steps;
            Value next = replaced[cur.n];
            replaced[cur.n] = root;
            cur = next;
        }
        return root;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        for (Block& block : blocks) {
            std::vector<Phi> kept;
            kept.reserve(block.phis.size());
            for (Phi& phi : block.phis) {
                std::optional<Value> only;
                bool keep = false;
                for (Value a0 : phi.args) {
                    Value a = find(a0);
                    if (a == phi.value || (only && *only == a)) continue;
                    if (only) {
                        keep = true;
                        break;
                    }
                    only = a;
                }
                if (keep || !only) {
                    kept.push_back(std::move(phi));
                    continue;
                }
                replaced.at(phi.value.n) = *only;
                changed = true;
            }
            block.phis = std::move(kept);
        }
    }
    for (Block& block : blocks) {
        for (Phi& phi : block.phis) {
            for (Value& a : phi.args) a = find(a);
        }
        for (Inst& inst : block.insts) {
            for (auto& [_, v] : inst.reads) v = find(v);
        }
    }
    return steps;
}

std::optional<Effects> effects(const Instruction& inst, const Fields& fields, std::string_view name,
                               std::uint8_t lanes, std::uint16_t vgprs) {
    if (std::find(std::begin(DROPPED), std::end(DROPPED), name) != std::end(DROPPED)) return std::nullopt;
    // `name` comes from the opcode tables (static, NUL-terminated).
    auto unsupported = [&]() { return ShaderError::unsupported(inst.offset, name.data()); };
    auto mask = [&](std::uint8_t code, std::vector<Reg>& out) {
        for (unsigned i = 0; i < lanes; ++i) out.push_back(Reg::S(std::uint8_t(code + i)));
    };
    std::vector<Reg> r, w;
    auto regs = [&](const Operand& op, std::vector<Reg>& out) {
        if (!add_regs(op, lanes, out)) throw ShaderError::isa(inst.offset, "register range out of bounds");
    };
    bool vector = !(std::holds_alternative<gfx10::Sop>(fields) || std::holds_alternative<gfx10::Sopp>(fields) ||
                    std::holds_alternative<gfx10::Smem>(fields));
    if (const auto* s = std::get_if<gfx10::Sop>(&fields)) {
        if (name.starts_with("s_movrel") || name.starts_with("s_subvector_loop") || name == "s_call_b64" ||
            name == "s_rfe_b64") {
            throw unsupported();
        }
        for (const Operand& o : s->src) regs(o, r);
        if (s->dst) {
            regs(*s->dst, w);
            // These change (or may keep) the destination's old value.
            if (name.starts_with("s_bitset") || name.starts_with("s_cmov") || name == "s_addk_i32" ||
                name == "s_mulk_i32") {
                regs(*s->dst, r);
            }
        }
        if (name == "s_addc_u32" || name == "s_subb_u32" || name == "s_cmovk_i32" || name.starts_with("s_cselect_") ||
            name.starts_with("s_cmov_")) {
            r.push_back(Reg::Scc);
        }
        if (contains_str(name, "saveexec") || contains_str(name, "wrexec")) {
            mask(EXEC, r);
            mask(EXEC, w);
        }
        if (writes_scc(inst.encoding, name)) w.push_back(Reg::Scc);
    } else if (std::holds_alternative<gfx10::Sopp>(fields)) {
        if (name == "s_cbranch_scc0" || name == "s_cbranch_scc1") {
            r.push_back(Reg::Scc);
        } else if (name == "s_cbranch_vccz" || name == "s_cbranch_vccnz") {
            mask(VCC, r);
        } else if (name == "s_cbranch_execz" || name == "s_cbranch_execnz") {
            mask(EXEC, r);
        } else if (name == "s_sendmsg" || name == "s_sendmsghalt") {
            r.push_back(Reg::S(M0));
        } else if (name == "s_endpgm" || name == "s_endpgm_saved" || name == "s_endpgm_ordered_ps_done" ||
                   name == "s_code_end" || name == "s_branch" || name == "s_barrier" || name == "s_icache_inv" ||
                   name == "s_wakeup" || name == "s_round_mode" || name == "s_denorm_mode") {
        } else {
            throw unsupported();
        }
    } else if (const auto* m = std::get_if<gfx10::Smem>(&fields)) {
        if (name == "s_memtime" || name == "s_memrealtime" || name == "s_get_waveid_in_workgroup") {
            if (!m->data) throw unsupported();
            regs(*m->data, w);
        } else if (name == "s_gl1_inv" || name == "s_dcache_inv" || name == "s_dcache_wb") {
        } else if (contains_str(name, "_load_") || contains_str(name, "_store_")) {
            regs(m->base, r);
            regs(m->soffset, r);
            if (!m->data) throw unsupported();
            regs(*m->data, contains_str(name, "_load_") ? w : r);
        } else {
            throw unsupported();
        }
    } else if (const auto* v = std::get_if<gfx10::Valu>(&fields)) {
        bool src_indexed = false, dst_indexed = false;
        if (name == "v_movrels_b32") {
            src_indexed = true;
        } else if (name == "v_movreld_b32") {
            dst_indexed = true;
        } else if (name == "v_movrelsd_b32" || name == "v_movrelsd_2_b32" || name == "v_swaprel_b32") {
            src_indexed = dst_indexed = true;
        }
        // v[reg + M0] for any M0 that stays inside the program's VGPRs.
        auto indexed = [&](const Operand& op, std::vector<Reg>& out) {
            if (op.kind == Operand::Kind::Vector && std::uint16_t(op.reg) < vgprs) {
                for (std::uint16_t x = op.reg; x < vgprs; ++x) out.push_back(Reg::V(std::uint8_t(x)));
                return;
            }
            throw ShaderError::isa(inst.offset, "relative VGPR base outside the program's VGPRs");
        };
        for (std::size_t i = 0; i < v->src.size(); ++i) {
            if (i == 0 && src_indexed) {
                indexed(v->src[i], r);
            } else {
                regs(v->src[i], r);
            }
        }
        // GFX10 compares into EXEC write nothing else.
        if (name.starts_with("v_cmpx")) {
            mask(EXEC, w);
        } else if (v->dst) {
            if (dst_indexed) {
                indexed(*v->dst, w);
            } else {
                regs(*v->dst, w);
            }
        }
        if (v->carry) regs(*v->carry, w);
        if (name == "v_swap_b32") {
            regs(v->src.at(0), w);
            if (v->dst) regs(*v->dst, r);
        } else if (name == "v_swaprel_b32") {
            indexed(v->src.at(0), w);
            if (v->dst) indexed(*v->dst, r);
        } else if (name == "v_div_fmas_f32" || name == "v_div_fmas_f64") {
            mask(VCC, r);
        }
        if (name.starts_with("v_interp_") || src_indexed || dst_indexed) r.push_back(Reg::S(M0));
    } else if (const auto* i = std::get_if<gfx10::Vintrp>(&fields)) {
        regs(i->src, r);
        r.push_back(Reg::S(M0));
        regs(i->dst, w);
    } else if (const auto* d = std::get_if<gfx10::Ds>(&fields)) {
        if (d->addr) regs(*d->addr, r);
        for (const Operand& o : d->data) regs(o, r);
        if (d->dst) regs(*d->dst, w);
        if (d->gds || name.starts_with("ds_gws") || contains_str(name, "addtid") || name == "ds_ordered_count" ||
            name == "ds_append" || name == "ds_consume") {
            r.push_back(Reg::S(M0));
        }
    } else if (const auto* b = std::get_if<gfx10::Buffer>(&fields)) {
        if (b->offen || b->idxen) regs(b->addr, r);
        regs(b->rsrc, r);
        regs(b->soffset, r);
        if (b->lds) r.push_back(Reg::S(M0));
        if (b->data) {
            const Operand& dd = *b->data;
            if (contains_str(name, "_load_")) {
                if (!b->lds) regs(dd, w);
            } else if (contains_str(name, "_store_")) {
                regs(dd, r);
            } else if (contains_str(name, "_atomic_")) {
                regs(dd, r);
                if (b->cache.glc) regs(atomic_result(dd, name), w);
            } else {
                throw unsupported();
            }
        }
    } else if (const auto* im = std::get_if<gfx10::Image>(&fields)) {
        for (const Operand& a : im->addr) regs(a, r);
        regs(im->rsrc, r);
        if (im->sampler) regs(*im->sampler, r);
        if (contains_str(name, "_store")) {
            regs(im->data, r);
        } else if (contains_str(name, "_atomic_")) {
            regs(im->data, r);
            if (im->cache.glc) regs(atomic_result(im->data, name), w);
        } else {
            regs(im->data, w);
        }
    } else if (const auto* f = std::get_if<gfx10::Flat>(&fields)) {
        if (f->addr) regs(*f->addr, r);
        if (f->data) regs(*f->data, r);
        if (f->saddr) regs(*f->saddr, r);
        if (f->dst) regs(*f->dst, w);
    } else if (const auto* e = std::get_if<gfx10::Exp>(&fields)) {
        for (const auto& s : e->src) {
            if (s) regs(*s, r);
        }
    }
    if (vector) mask(EXEC, r);
    std::vector<Reg> deduped;
    deduped.reserve(r.size());
    for (Reg reg : r) {
        if (!vec_contains(deduped, reg)) deduped.push_back(reg);
    }
    return Effects{std::move(deduped), std::move(w)};
}

}  // namespace detail

Program build(std::span<const std::uint8_t> code, bool wave64, std::uint16_t vgprs) {
    std::vector<Instruction> list = gfx10::decode(code);
    const std::uint8_t lanes = std::uint8_t(1 + (wave64 ? 1 : 0));
    auto isa = [](std::size_t offset, const char* reason) { return ShaderError::isa(offset, reason); };

    std::vector<Flow> flows;
    flows.reserve(list.size());
    std::vector<std::size_t> leaders{0};
    gfx10::Before before = [&](std::size_t at) -> const Instruction* {
        auto it = std::partition_point(list.begin(), list.end(), [&](const Instruction& i) { return i.offset < at; });
        return it == list.begin() ? nullptr : &*std::prev(it);
    };
    for (std::size_t i = 0; i < list.size(); ++i) {
        Flow f = gfx10::control(list[i], code.size(), before);
        if (f.kind == Flow::Kind::Jump || f.kind == Flow::Kind::Branch) leaders.push_back(f.target);
        if (f.kind != Flow::Kind::Next && i + 1 < list.size()) leaders.push_back(list[i + 1].offset);
        flows.push_back(f);
    }
    std::sort(leaders.begin(), leaders.end());
    leaders.erase(std::unique(leaders.begin(), leaders.end()), leaders.end());
    std::vector<Move> parked = unspill(list, flows, leaders, wave64, lanes, vgprs);
    auto block_at = [&](std::size_t offset) -> std::size_t {
        auto it = std::lower_bound(leaders.begin(), leaders.end(), offset);
        if (it == leaders.end() || *it != offset) throw isa(offset, "jump into a block");
        return std::size_t(it - leaders.begin());
    };

    std::vector<Block> blocks;
    blocks.reserve(leaders.size());
    for (std::size_t offset : leaders) {
        Block b;
        b.offset = offset;
        b.exit = Exit::End;
        blocks.push_back(std::move(b));
    }
    std::vector<std::optional<std::size_t>> next_offsets(list.size());
    for (std::size_t i = 0; i + 1 < list.size(); ++i) next_offsets[i] = list[i + 1].offset;
    for (std::size_t n = 0; n < list.size(); ++n) {
        Instruction& machine = list[n];
        const Flow flow = flows[n];
        const std::optional<std::size_t> next = next_offsets[n];
        std::size_t b = std::size_t(std::upper_bound(leaders.begin(), leaders.end(), machine.offset) - leaders.begin()) - 1;
        auto name_opt = machine.name();
        if (!name_opt) throw ShaderError::unsupported(machine.offset, "unknown opcode");
        std::string_view name = *name_opt;
        Fields fields = machine.fields(wave64);
        // A move to or from a scalar parked in a VGPR lane (`unspill`): its register codes may be
        // ones the machine word cannot hold.
        auto k = std::lower_bound(parked.begin(), parked.end(), n, [](const Move& m, std::size_t v) { return m.index < v; });
        if (k != parked.end() && k->index == n) {
            if (auto* sop = std::get_if<gfx10::Sop>(&fields)) {
                if (sop->dst && sop->dst->kind == Operand::Kind::Scalar && !sop->src.empty() &&
                    sop->src[0].kind == Operand::Kind::Scalar) {
                    sop->dst->code = k->dst;
                    sop->src[0].code = k->src;
                }
            }
        }
        auto fallthrough = [&]() -> std::size_t {
            if (!next) throw isa(machine.offset, "falls off the code");
            return block_at(*next);
        };
        // A conditional SOPP branch to fallthrough has no register effects and contributes one edge
        // to the join. Calls also use Flow::Branch but write a return address.
        if (machine.encoding == gfx10::Encoding::Sopp && flow.kind == Flow::Kind::Branch && next &&
            flow.target == *next) {
            blocks[b].exit = Exit::Jump(fallthrough());
            continue;
        }
        bool ends = flow.kind != Flow::Kind::Next || !next ||
                    std::binary_search(leaders.begin(), leaders.end(), *next);
        if (ends) {
            switch (flow.kind) {
            case Flow::Kind::Next:
                blocks[b].exit = Exit::Jump(fallthrough());
                break;
            case Flow::Kind::Jump:
                blocks[b].exit = Exit::Jump(block_at(flow.target));
                break;
            case Flow::Kind::Branch: {
                std::size_t taken = block_at(flow.target);
                std::size_t not_taken = fallthrough();
                blocks[b].exit = Exit::Branch(taken, not_taken);
                break;
            }
            case Flow::Kind::Stop:
                blocks[b].exit = name == "s_setpc_b64" ? Exit::Indirect : Exit::End;
                break;
            }
        }
        if (auto e = detail::effects(machine, fields, name, lanes, vgprs)) {
            auto& [reads, writes] = *e;
            // Lanes disabled in EXEC keep the old value.
            for (Reg w : writes) {
                if (w.is_v() && !vec_contains(reads, w)) reads.push_back(w);
            }
            Inst inst;
            inst.machine = std::move(machine);
            inst.fields = std::move(fields);
            inst.reads = unset(reads);
            inst.writes = unset(writes);
            blocks[b].insts.push_back(std::move(inst));
        }
    }
    detail::link(blocks);
    if (!blocks[0].preds.empty()) throw ShaderError::unsupported(0, "a branch to the program entry");
    detail::split_irreducible(blocks);

    return detail::rename(std::move(blocks), wave64, vgprs);
}

Program thread_edges(const Program& p,
                     const std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>& edges) {
    std::vector<Block> blocks = p.blocks;
    for (Block& block : blocks) block.phis.clear();
    for (const auto& [pred, join, dest] : edges) {
        Block copy = blocks.at(join);
        copy.exit = Exit::Jump(dest);
        std::size_t target = blocks.size();
        Exit& exit = blocks.at(pred).exit;
        if (exit.kind == Exit::Kind::Jump) {
            exit.target = target;
        } else if (exit.kind == Exit::Kind::Branch) {
            if (exit.taken == join) exit.taken = target;
            if (exit.not_taken == join) exit.not_taken = target;
        } else {
            throw std::logic_error("threading an incoming edge");
        }
        blocks.push_back(std::move(copy));
    }
    detail::link(blocks);
    detail::split_irreducible(blocks);
    return detail::rename(std::move(blocks), p.wave64, p.vgprs);
}

std::optional<Program> without_early_outs(const Program& p) {
    auto ends = [&](std::size_t b) { return p.blocks[b].exit == Exit::End && all_sopp(p.blocks[b]); };
    std::vector<Block> blocks = p.blocks;
    bool any = false;
    for (Block& block : blocks) {
        if (block.exit.kind != Exit::Kind::Branch) continue;
        std::size_t taken = block.exit.taken, not_taken = block.exit.not_taken;
        if (ends(taken) && !block.insts.empty() && block.insts.back().name() == "s_cbranch_execz") {
            block.insts.pop_back();
            block.exit = Exit::Jump(not_taken);
            any = true;
        }
    }
    if (!any) return std::nullopt;
    return relink(std::move(blocks), p);
}

std::optional<Program> without_skips(const Program& p, bool deep) {
    const std::size_t n = p.blocks.size();
    auto ends = [&](std::size_t b) { return p.blocks[b].exit == Exit::End && all_sopp(p.blocks[b]); };
    // Scalars moved from register to register (a scalar the compiler parked in a VGPR lane comes back
    // by such a move, `unspill`) are the value they copy.
    std::unordered_map<Value, Value> copy;
    // Values something reads, directly or through joins. A scalar operation none of whose results is
    // read reads nothing either (the compiler restores EXEC from a saved mask and overwrites it at
    // once where two branches end together).
    std::unordered_set<Value> live;
    auto idle = [&](const Inst& inst) {
        std::string_view name = inst.name();
        return std::holds_alternative<gfx10::Sop>(inst.fields) &&
               !(name.starts_with("s_setpc") || name.starts_with("s_swappc") || name.starts_with("s_call") ||
                 name.starts_with("s_rfe") || name.starts_with("s_movrel") || name.starts_with("s_subvector")) &&
               std::all_of(inst.writes.begin(), inst.writes.end(), [&](const auto& w) { return !live.contains(w.second); });
    };
    for (const Block& block : p.blocks) {
        for (const Inst& inst : block.insts) {
            if ((inst.name() == "s_mov_b32" || inst.name() == "s_mov_b64") && inst.reads.size() == inst.writes.size() &&
                std::all_of(inst.reads.begin(), inst.reads.end(), [](const auto& r) { return r.first.is_s(); }) &&
                std::all_of(inst.writes.begin(), inst.writes.end(), [](const auto& r) { return r.first.is_s(); })) {
                for (std::size_t i = 0; i < inst.writes.size(); ++i) copy[inst.writes[i].second] = inst.reads[i].second;
            }
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (const Block& block : p.blocks) {
            for (const Inst& inst : block.insts) {
                if (!idle(inst)) {
                    for (const auto& r : inst.reads) changed |= live.insert(r.second).second;
                }
            }
            for (const Phi& phi : block.phis) {
                if (live.contains(phi.value)) {
                    for (Value a : phi.args) changed |= live.insert(a).second;
                }
            }
        }
    }
    auto root = [&](Value v) {
        while (deep) {
            auto it = copy.find(v);
            if (it == copy.end()) break;
            v = it->second;
        }
        return v;
    };
    // The region a skip of `b` to `target` leaves out, if it is a pure skip.
    auto skipped = [&](std::size_t b, std::size_t target, std::size_t first) -> std::optional<std::vector<bool>> {
        std::vector<bool> inside(n, false);
        std::vector<std::size_t> stack{first};
        while (!stack.empty()) {
            std::size_t x = stack.back();
            stack.pop_back();
            if (x == target || inside[x]) continue;
            if (x == b) return std::nullopt;  // a loop back to the branch
            inside[x] = true;
            const Exit& exit = p.blocks[x].exit;
            if (exit.kind == Exit::Kind::End || exit.kind == Exit::Kind::Indirect) return std::nullopt;
            for (std::size_t s : exit.successors()) stack.push_back(s);
        }
        // entered only through the branch
        for (std::size_t x = 0; x < n; ++x) {
            if (!inside[x]) continue;
            for (std::size_t q : p.blocks[x].preds) {
                if (q != b && !inside[q]) return std::nullopt;
            }
        }
        // the join sees no scalar the region changed
        const Block& join = p.blocks[target];
        for (const Phi& phi : join.phis) {
            bool scalar = phi.reg.is_s() ? (phi.reg.n != 126 && phi.reg.n != 127) : phi.reg.is_scc();
            // (a join nothing reads shows no one the difference)
            if (!scalar || (deep && !live.contains(phi.value))) continue;
            std::vector<Value> before;
            for (std::size_t k = 0; k < join.preds.size() && k < phi.args.size(); ++k) {
                if (join.preds[k] == b) before.push_back(root(phi.args[k]));
            }
            for (std::size_t k = 0; k < join.preds.size() && k < phi.args.size(); ++k) {
                if (inside[join.preds[k]] && !vec_contains(before, root(phi.args[k]))) return std::nullopt;
            }
        }
        // Other ways into the join than the branch and the region (an inner branch that ends
        // together with an outer one) change nothing in that: the lanes that would have skipped
        // arrive from the region instead, with the same scalars.
        if (!deep) {
            for (std::size_t q : join.preds) {
                if (q != b && !inside[q]) return std::nullopt;
            }
        }
        return inside;
    };
    std::vector<Block> blocks = p.blocks;
    bool any = false;
    for (std::size_t b = 0; b < n; ++b) {
        if (p.blocks[b].exit.kind != Exit::Kind::Branch) continue;
        std::size_t taken = p.blocks[b].exit.taken, not_taken = p.blocks[b].exit.not_taken;
        if (p.blocks[b].insts.empty() || p.blocks[b].insts.back().name() != "s_cbranch_execz") continue;
        if (ends(taken) || skipped(b, taken, not_taken)) {
            blocks[b].insts.pop_back();
            blocks[b].exit = Exit::Jump(not_taken);
            any = true;
        }
    }
    if (!any) return std::nullopt;
    return relink(std::move(blocks), p);
}

std::vector<Reg> operand_reads(const Program& p, const Inst& inst) {
    std::uint8_t lanes = std::uint8_t(1 + (p.wave64 ? 1 : 0));
    try {
        auto e = detail::effects(inst.machine, inst.fields, inst.name(), lanes, p.vgprs);
        if (e) return std::move(e->first);
    } catch (const ShaderError&) {
    }
    return {};
}

std::vector<std::size_t> reverse_postorder(std::span<const Block> blocks) {
    std::vector<bool> seen(blocks.size(), false);
    std::vector<std::size_t> post;
    post.reserve(blocks.size());
    std::vector<std::pair<std::size_t, std::size_t>> stack{{0, 0}};
    seen[0] = true;
    while (!stack.empty()) {
        auto& top = stack.back();
        auto [b, i] = top;
        top.second += 1;
        std::vector<std::size_t> succ = blocks[b].exit.successors();
        if (i < succ.size()) {
            std::size_t s = succ[i];
            if (!seen[s]) {
                seen[s] = true;
                stack.push_back({s, 0});
            }
        } else {
            post.push_back(b);
            stack.pop_back();
        }
    }
    std::reverse(post.begin(), post.end());
    return post;
}

std::optional<std::pair<std::size_t, std::size_t>> irreducible_edge(std::span<const Block> blocks,
                                                                    std::span<const std::size_t> order,
                                                                    std::span<const std::size_t> index) {
    if (order.empty()) return std::nullopt;
    std::size_t entry = order[0];
    std::vector<std::size_t> idom = idoms(blocks, order, index);
    for (std::size_t b : order) {
        for (std::size_t s : blocks[b].exit.successors()) {
            if (index[s] > index[b]) continue;
            std::size_t d = b;
            while (d != s && d != entry) d = idom[d];
            if (d != s) return std::make_pair(b, s);
        }
    }
    return std::nullopt;
}

std::vector<std::size_t> idoms(std::span<const Block> blocks, std::span<const std::size_t> order,
                               std::span<const std::size_t> index) {
    std::vector<std::size_t> idom(blocks.size(), SIZE_MAX);
    if (order.empty()) return idom;
    std::size_t entry = order[0];
    idom[entry] = entry;
    auto intersect = [&](std::size_t a, std::size_t b) {
        while (a != b) {
            while (index[a] > index[b]) a = idom[a];
            while (index[b] > index[a]) b = idom[b];
        }
        return a;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t k = 1; k < order.size(); ++k) {
            std::size_t b = order[k];
            std::size_t nw = SIZE_MAX;
            for (std::size_t q : blocks[b].preds) {
                if (idom[q] == SIZE_MAX) continue;
                nw = nw == SIZE_MAX ? q : intersect(q, nw);
            }
            if (idom[b] != nw) {
                idom[b] = nw;
                changed = true;
            }
        }
    }
    return idom;
}

}  // namespace x2w::vs::ir
