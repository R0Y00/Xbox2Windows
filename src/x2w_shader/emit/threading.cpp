// Jump threading for uniform diamonds. One arm can set a flag that fixes the join's next branch. Registers
// recycled on that arm then never reach the other successor, even if an ordinary SSA phi joins their
// values.

#include "x2w_shader/emit/threading.h"

#include <algorithm>
#include <map>
#include <optional>
#include <variant>

#include "x2w_shader/emit/wave.h"
#include "x2w_shader/structure.h"

namespace x2w::vs::threading {

using gfx10::Fields;
using gfx10::Operand;
using gfx10::Valu;
using ir::Exit;
using ir::Program;
using ir::Reg;
using ir::Value;
using structure::Atom;
using structure::Bool;
using structure::LaneExit;
using structure::Structured;

namespace {

bool plain(const Valu& v) {
    return v.abs == 0 && v.neg == 0 && v.opsel == 0 && !v.clamp && v.omod == 0 && !v.sdwa && !v.dpp;
}

std::optional<std::uint32_t> immediate(const Operand& op) {
    switch (op.kind) {
        case Operand::Kind::Const: return wave::detail::inline_constant(op.code);  // resource::inline
        case Operand::Kind::Literal: return op.literal;
        default: return std::nullopt;
    }
}

}  // namespace

std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> constant_edges(const Program& p,
                                                                               const Structured& st) {
    std::map<Value, std::uint32_t> constants;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const auto& block = p.blocks[b];
        for (std::size_t i = 0; i < block.insts.size(); ++i) {
            const auto& inst = block.insts[i];
            if (inst.name() != "v_mov_b32" || st.guards.at(b).at(i).has_value()) continue;
            const auto* v = std::get_if<Valu>(&inst.fields);
            if (!v || !plain(*v) || v->src.size() != 1) continue;
            const auto word = immediate(v->src[0]);
            if (!word || inst.writes.size() != 1) continue;
            constants.insert_or_assign(inst.writes[0].second, *word);
        }
    }
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> edges;
    for (std::size_t join = 0; join < p.blocks.size(); ++join) {
        const auto& block = p.blocks[join];
        const LaneExit& ex = st.exits.at(join);
        if (ex.kind != LaneExit::Kind::Branch) continue;
        const Bool when = ex.when;
        const std::size_t taken = ex.taken;
        const std::size_t not_taken = ex.not_taken;
        if (when == Bool::True || when == Bool::False) continue;
        // Keep memory accesses and wave operations at their original reconvergence point. Only
        // duplicate scalar ALU and comparisons.
        const bool scalar_only = std::all_of(block.insts.begin(), block.insts.end(), [](const ir::Inst& i) {
            if (std::holds_alternative<gfx10::Sop>(i.fields)) return true;
            if (std::holds_alternative<gfx10::Sopp>(i.fields)) return i.name().starts_with("s_cbranch_");
            if (std::holds_alternative<Valu>(i.fields)) return i.name().starts_with("v_cmp_");
            return false;
        });
        if (!scalar_only) continue;
        if (block.preds.size() != 2) continue;
        const std::size_t a = block.preds[0];
        const std::size_t b = block.preds[1];
        // A triangle has a direct edge from the fork to the join. A full diamond has an
        // unconditional jump at the end of each arm.
        auto arm_from = [&](std::size_t arm) -> std::optional<std::size_t> {
            const auto& blk = p.blocks.at(arm);
            if (blk.preds.size() == 1 && blk.exit == Exit::Jump(join)) return blk.preds[0];
            return std::nullopt;
        };
        const auto fa = arm_from(a);
        const auto fb = arm_from(b);
        std::size_t fork = 0;
        if (fa && fb && *fa == *fb) {
            fork = *fa;
        } else if (fb && *fb == a) {
            fork = *fb;
        } else if (fa && *fa == b) {
            fork = *fa;
        } else {
            continue;
        }
        if (fork == join || a == b) continue;
        const LaneExit& fx = st.exits.at(fork);
        if (fx.kind != LaneExit::Kind::Branch) continue;
        const Bool entry = fx.when;
        const auto sup = st.bools.support(entry);
        if (!std::all_of(sup.begin(), sup.end(), [](const Atom& at) { return at.kind == Atom::Kind::Uniform; })) {
            continue;
        }
        for (std::size_t edge = 0; edge < block.preds.size(); ++edge) {
            const std::size_t pred = block.preds[edge];
            auto value = [&](Value v) -> std::optional<std::uint32_t> {
                Value input = v;
                for (const auto& phi : block.phis) {
                    if (phi.value == v) {
                        input = phi.args.at(edge);
                        break;
                    }
                }
                const auto it = constants.find(input);
                if (it == constants.end()) return std::nullopt;
                return it->second;
            };
            // Restrict the branch's decision diagram to known comparisons. Unknown tests stop the
            // proof. Keep the original join for the other arm, which may still go either way.
            Bool cond = when;
            while (true) {
                const auto node = st.bools.node(cond);
                if (!node || std::get<0>(*node).kind != Atom::Kind::Lane) break;
                const Value id = std::get<0>(*node).value;
                const Bool lo = std::get<1>(*node);
                const Bool hi = std::get<2>(*node);
                const ir::Inst* inst = nullptr;
                for (const auto& i : block.insts) {
                    if (std::any_of(i.writes.begin(), i.writes.end(), [&](const auto& w) { return w.second == id; })) {
                        inst = &i;
                        break;
                    }
                }
                if (!inst) break;
                const auto* v = std::get_if<Valu>(&inst->fields);
                if (!v) break;
                if (v->src.size() != 2) break;
                auto operand = [&](const Operand& op) -> std::optional<std::uint32_t> {
                    if (op.kind == Operand::Kind::Vector && op.dwords == 1) {
                        const auto r = inst->read(Reg::V(op.reg));
                        if (!r) return std::nullopt;
                        return value(*r);
                    }
                    return immediate(op);
                };
                const auto x = operand(v->src[0]);
                const auto y = operand(v->src[1]);
                if (!x || !y) break;
                if (!plain(*v)) break;
                const std::string_view name = inst->name();
                bool yes = false;
                if (name == "v_cmp_eq_u32" || name == "v_cmp_eq_i32") {
                    yes = *x == *y;
                } else if (name == "v_cmp_ne_u32" || name == "v_cmp_ne_i32") {
                    yes = *x != *y;
                } else {
                    break;
                }
                cond = yes ? hi : lo;
            }
            if (cond == Bool::True || cond == Bool::False) {
                edges.emplace_back(pred, join, cond == Bool::True ? taken : not_taken);
                break;
            }
        }
    }
    return edges;
}

}  // namespace x2w::vs::threading
