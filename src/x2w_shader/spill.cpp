// Spilled user-data tables. See spill.h.
#include "x2w_shader/spill.h"

#include <algorithm>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "x2w_shader/front_hash.h"

namespace x2w::vs::spill {

namespace {

using gfx10::Operand;
using ir::Program;
using ir::Reg;
using ir::Value;
using Site = std::pair<std::size_t, std::size_t>;
using ValueSet = std::unordered_set<Value, fhash::ValueHash>;

// The user-data index of the table's first dword.
constexpr std::uint32_t FIRST = 16;

bool contains(const std::vector<Value>& v, Value x) { return std::find(v.begin(), v.end(), x) != v.end(); }
bool contains(const std::vector<Site>& v, Site x) { return std::find(v.begin(), v.end(), x) != v.end(); }

// Values some instruction or phi reads; with `skip`, not counting the instructions it lists.
ValueSet reads_of(const Program& p, const std::vector<Site>* skip)
{
    ValueSet used;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block = p.blocks[b];
        for (const auto& phi : block.phis) used.insert(phi.args.begin(), phi.args.end());
        for (std::size_t i = 0; i < block.insts.size(); ++i) {
            if (skip && contains(*skip, Site{b, i})) continue;
            for (const auto& r : block.insts[i].reads) used.insert(r.second);
        }
    }
    return used;
}

}  // namespace

namespace detail {

std::optional<std::vector<std::pair<std::uint32_t, std::uint32_t>>> spilled(const native::NativeProgram& native)
{
    const auto* cs = std::get_if<native::ComputeIo>(&native.io);
    if (!cs) return std::nullopt;
    if (native.stage != native::HwStage::Cs) return std::nullopt;
    const std::uint32_t registers = cs->user_sgprs();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> out;
    for (const auto& e : native.layout)
        if (e.sgprs > 0 && std::uint32_t{e.user_data} >= registers) out.push_back({e.user_data, e.sgprs});
    // Only the one arrangement seen: the table starts at index 16.
    std::optional<std::uint32_t> min;
    for (const auto& e : out)
        if (!min || e.first < *min) min = e.first;
    if (min == FIRST && registers >= 2) return out;
    return std::nullopt;
}

}  // namespace detail

Program build(const native::NativeProgram& native)
{
    const std::vector<std::uint8_t> code = native.patched_code();
    Program p = ir::build(code, native.wave64, native.vgprs);
    user_data(p, native);
    return p;
}

bool is_spilled(const native::NativeProgram& native) { return detail::spilled(native).has_value(); }

std::size_t user_data(Program& p, const native::NativeProgram& native)
{
    const auto spilled = detail::spilled(native);
    if (!spilled) return 0;
    auto entry = [&](Reg reg) -> std::optional<Value> {
        for (const auto& i : p.inputs)
            if (i.first == reg) return i.second;
        return std::nullopt;
    };
    const auto low = entry(Reg::S(0));
    const auto high = entry(Reg::S(1));
    if (!low || !high) return 0;

    // The pointer's halves: `s0` and its copies, `s1 & 0xff` and its copies.
    std::vector<Value> lows{*low};
    std::vector<Value> highs;
    std::vector<Value> raw_highs{*high};
    // Instructions that only form the pointer: dropped when nothing reads them any more.
    std::vector<Site> forming;
    for (;;) {
        bool changed = false;
        for (std::size_t b = 0; b < p.blocks.size(); ++b) {
            const auto& block = p.blocks[b];
            for (std::size_t i = 0; i < block.insts.size(); ++i) {
                const ir::Inst& inst = block.insts[i];
                const auto* f = std::get_if<gfx10::Sop>(&inst.fields);
                if (!f) continue;
                if (!f->dst || !f->dst->is_scalar()) continue;
                const auto out = inst.write(Reg::S(f->dst->code));
                if (!out) continue;
                auto read = [&](const Operand& op) -> std::optional<Value> {
                    if (op.is_scalar() && op.code != gfx10::NULL_CODE) return inst.read(Reg::S(op.code));
                    return std::nullopt;
                };
                std::vector<Value>* found = nullptr;
                const std::string_view name = inst.name();
                if (name == "s_mov_b32" && f->src.size() == 1) {
                    const auto v = read(f->src[0]);
                    if (v && contains(lows, *v))
                        found = &lows;
                    else if (v && contains(highs, *v))
                        found = &highs;
                    else if (v && contains(raw_highs, *v))
                        found = &raw_highs;
                } else if (name == "s_and_b32" && f->src.size() == 2) {
                    auto masked = [&](const Operand& a, const Operand& m) {
                        if (!(m.is_literal() && m.literal == 0xff)) return false;
                        const auto v = read(a);
                        return v && contains(raw_highs, *v);
                    };
                    if (masked(f->src[0], f->src[1]) || masked(f->src[1], f->src[0])) found = &highs;
                }
                if (found && !contains(*found, *out)) {
                    found->push_back(*out);
                    forming.push_back({b, i});
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }
    if (highs.empty()) return 0;

    // Values some instruction or phi reads.
    const ValueSet read_somewhere = reads_of(p, nullptr);
    std::uint32_t next = 0;
    {
        std::optional<std::uint32_t> max;
        auto see = [&](Value v) {
            const std::uint32_t n = v.n + 1;
            if (!max || n > *max) max = n;
        };
        for (const auto& i : p.inputs) see(i.second);
        for (const auto& b : p.blocks) {
            for (const auto& phi : b.phis) see(phi.value);
            for (const auto& i : b.insts)
                for (const auto& w : i.writes) see(w.second);
        }
        next = max.value_or(0);
    }

    // Each load through the pointer: its dwords become entry values.
    std::unordered_map<Value, Value, fhash::ValueHash> replaced;
    std::vector<Site> gone;
    auto inputs = p.inputs;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block = p.blocks[b];
        for (std::size_t i = 0; i < block.insts.size(); ++i) {
            const ir::Inst& inst = block.insts[i];
            const auto* f = std::get_if<gfx10::Smem>(&inst.fields);
            if (!f) continue;
            if (!f->data || !f->data->is_scalar() || !f->base.is_scalar()) continue;
            const std::uint8_t first = f->data->code;
            const std::uint8_t base = f->base.code;
            const std::int32_t offset = f->offset;
            const Operand& soffset = f->soffset;
            const bool soffset_none = (soffset.is_scalar() && soffset.code == gfx10::NULL_CODE) ||
                                      (soffset.is_const() && soffset.code == 128);
            auto in = [&](std::uint8_t code, const std::vector<Value>& set) {
                const auto v = inst.read(Reg::S(code));
                return v && contains(set, *v);
            };
            if (!inst.name().starts_with("s_load_dword") || !soffset_none || offset < 0 || offset % 4 != 0 ||
                !in(base, lows) || !in(static_cast<std::uint8_t>(base + 1), highs))
                continue;
            std::vector<std::pair<Reg, Value>> dwords;
            bool skip = false;
            for (const auto& [reg, v] : inst.writes) {
                if (!reg.is_s() || reg.n < first) {
                    skip = true;
                    break;
                }
                const std::uint32_t k = static_cast<std::uint32_t>(reg.n - first);
                const std::uint32_t index = FIRST + static_cast<std::uint32_t>(offset) / 4 + k;
                const bool declared = std::any_of(spilled->begin(), spilled->end(), [&](const auto& e) {
                    return index >= e.first && index < e.first + e.second;
                });
                if (!declared) {
                    if (read_somewhere.count(v)) {
                        skip = true;
                        break;
                    }
                    continue;
                }
                // SGPR codes end at 105; no root signature reaches there.
                if (index > 105) {
                    skip = true;
                    break;
                }
                dwords.push_back({Reg::S(static_cast<std::uint8_t>(index)), v});
            }
            if (skip) continue;
            for (const auto& [reg, v] : dwords) {
                Value value;
                const auto it =
                    std::find_if(inputs.begin(), inputs.end(), [&](const auto& e) { return e.first == reg; });
                if (it != inputs.end()) {
                    value = it->second;
                } else {
                    value = Value{next};
                    next += 1;
                    inputs.push_back({reg, value});
                }
                replaced.insert_or_assign(v, value);
            }
            gone.push_back({b, i});
        }
    }
    if (gone.empty()) return 0;
    p.inputs = std::move(inputs);
    auto renamed = [&](Value& v) {
        if (const auto it = replaced.find(v); it != replaced.end()) v = it->second;
    };
    for (auto& block : p.blocks) {
        for (auto& phi : block.phis)
            for (auto& a : phi.args) renamed(a);
        for (auto& inst : block.insts)
            for (auto& r : inst.reads) renamed(r.second);
    }
    // The loads go, then what only formed the pointer (last first, so a copy goes before what it
    // copied).
    std::vector<Site> drop = gone;
    for (;;) {
        const ValueSet used = reads_of(p, &drop);
        const std::size_t before = drop.size();
        for (const Site& site : forming) {
            const ir::Inst& inst = p.blocks[site.first].insts[site.second];
            const bool unread = std::all_of(inst.writes.begin(), inst.writes.end(),
                                            [&](const auto& w) { return !used.count(w.second); });
            if (!contains(drop, site) && unread) drop.push_back(site);
        }
        if (drop.size() == before) break;
    }
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        auto& insts = p.blocks[b].insts;
        std::vector<ir::Inst> kept;
        kept.reserve(insts.size());
        for (std::size_t i = 0; i < insts.size(); ++i)
            if (!contains(drop, Site{b, i})) kept.push_back(std::move(insts[i]));
        insts = std::move(kept);
    }
    // The pointer's registers are no inputs once nothing reads them.
    const ValueSet read = reads_of(p, nullptr);
    std::erase_if(p.inputs, [&](const std::pair<Reg, Value>& e) {
        const bool pointer = e.first == Reg::S(0) || e.first == Reg::S(1);
        return pointer && !read.count(e.second);
    });
    return gone.size();
}

}  // namespace x2w::vs::spill
