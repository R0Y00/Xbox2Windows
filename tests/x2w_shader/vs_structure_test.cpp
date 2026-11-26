// Port of the upstream Rust shader tests in src/structure.rs.
#include "x2w_shader/structure.h"

#include "vs_front_test.h"

#include <algorithm>
#include <limits>

using namespace x2w::vs;
using namespace x2w::vs::structure;
using ir::Program;
using ir::Reg;

namespace {

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

Program build(const std::vector<std::uint32_t>& words, bool wave64)
{
    std::vector<std::uint8_t> code;
    for (std::uint32_t w : words)
        for (int i = 0; i < 4; ++i) code.push_back(static_cast<std::uint8_t>(w >> (8 * i)));
    return ir::build(code, wave64, 256);
}

std::size_t guarded(const Structured& s)
{
    std::size_t n = 0;
    for (const auto& row : s.guards)
        for (const auto& g : row) n += g.has_value();
    return n;
}

using Node = std::optional<std::tuple<Atom, Bool, Bool>>;
Node node(Atom a, Bool lo, Bool hi) { return std::tuple<Atom, Bool, Bool>{a, lo, hi}; }

bool support_is(const Structured& s, Bool b, Atom::Kind kind)
{
    const auto sup = s.bools.support(b);
    return sup.size() == 1 && sup[0].kind == kind;
}

Node mask_node(const Structured& s, ir::Value v)
{
    const auto it = s.masks.find(v);
    if (it == s.masks.end()) return std::nullopt;
    return s.bools.node(it->second);
}

const ir::Phi* phi_of(const Program& p, Reg reg)
{
    for (const auto& b : p.blocks)
        for (const auto& phi : b.phis)
            if (phi.reg == reg) return &phi;
    return nullptr;
}

bool contains(const std::vector<Bool>& v, Bool b) { return std::find(v.begin(), v.end(), b) != v.end(); }

// The search of Bools::meet with every case split made anew, as it was before Meets.
bool meet_anew(const Bools& b, std::vector<std::pair<Bool, bool>> all, std::size_t& budget)
{
    for (const auto& [x, holds] : all)
        if (x == (holds ? Bool::False : Bool::True)) return false;
    std::erase_if(all, [](const auto& e) { return e.first == Bool::False || e.first == Bool::True; });
    if (all.empty() || budget == 0) return true;
    budget -= 1;
    std::uint32_t v = std::numeric_limits<std::uint32_t>::max();
    for (const auto& e : all) v = std::min(v, b.top(e.first));
    for (const bool value : {false, true}) {
        std::vector<std::pair<Bool, bool>> next;
        for (const auto& [x, holds] : all) {
            const auto [lo, hi] = b.split(x, v);
            next.push_back({value ? hi : lo, holds});
        }
        if (meet_anew(b, next, budget)) return true;
    }
    return false;
}

}  // namespace

VS_TEST(a_search_that_remembers_answers_as_one_made_anew)
{
    // Functions of 12 variables put together at random, and conditions of two to five of them
    // (repeats and constants among them), under budgets that run out early, late and never: the
    // answer, and where no lane is found the budget left, are those of the search that remembers
    // nothing; with what one Meets gathered over all of them, and with an empty one.
    Bools b;
    std::uint64_t state = 0x9E37'79B9'7F4A'7C15ull;
    auto random = [&state](std::size_t n) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<std::size_t>(state % static_cast<std::uint64_t>(n));
    };
    std::vector<Bool> f;
    for (std::uint64_t i = 0; i < 12; ++i) f.push_back(b.atom(Atom::Constant(i + 2), 0));
    f.push_back(Bool::False);
    f.push_back(Bool::True);
    for (int k = 0; k < 120; ++k) {
        const Bool x = f[random(f.size())];
        const Bool y = f[random(f.size())];
        Bool made;
        switch (random(4)) {
        case 0: made = b.and_(x, y); break;
        case 1: made = b.or_(x, y); break;
        case 2: made = b.xor_(x, y); break;
        default: made = b.not_(x); break;
        }
        f.push_back(made);
    }
    Meets seen;
    int none = 0, some = 0, gave_up = 0;
    for (int k = 0; k < 1500; ++k) {
        const std::size_t count = 2 + random(4);
        std::vector<std::pair<Bool, bool>> all;
        for (std::size_t j = 0; j < count; ++j) {
            const Bool x = f[random(f.size())];
            const bool holds = random(2) == 0;
            all.push_back({x, holds});
        }
        for (const std::size_t budget : {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5},
                                         std::size_t{9}, std::size_t{20}, std::size_t{60}, std::size_t{250},
                                         std::size_t{1000}, MAX_MEET}) {
            std::size_t left = budget;
            const bool anew = meet_anew(b, all, left);
            for (const bool fresh : {false, true}) {
                Meets empty;
                Meets& use = fresh ? empty : seen;
                std::vector<std::uint32_t> work;
                for (const auto& [x, holds] : all) work.push_back((x.n << 1) | (holds ? 1u : 0u));
                std::size_t mine = budget;
                const bool found = b.meet_in(0, work, mine, use);
                VS_CHECK(found == anew);
                VS_CHECK(work.empty());
                if (!found) VS_CHECK(mine == left);
            }
            VS_CHECK(b.meet_within(all, budget, seen) == anew);
            if (!anew) {
                ++none;
            } else {
                std::size_t unlimited = std::numeric_limits<std::size_t>::max();
                if (meet_anew(b, all, unlimited))
                    ++some;
                else
                    ++gave_up;
            }
        }
    }
    // All three outcomes came up, and searches were remembered.
    VS_CHECK(none > 100 && some > 100 && gave_up > 100);
    VS_CHECK(seen.none.size() > 100);
}

VS_TEST(divergent_if_becomes_a_lane_branch)
{
    const Program p = build(
        {
            0x7D84'0080,  // v_cmp_eq_u32 vcc, 0, v0
            0xBE80'3C6A,  // s_and_saveexec_b32 s0, vcc_lo
            0xBF88'0001,  // s_cbranch_execz -> 16
            0x7E02'0280,  // v_mov_b32 v1, 0
            0x887E'007E,  // 16: s_or_b32 exec_lo, exec_lo, s0
            0x7E04'0280,  // v_mov_b32 v2, 0
            S_ENDPGM,
        },
        false);
    const Structured s = structure::structure(p);
    VS_REQUIRE(s.exits[0].is_branch() && s.exits[0].taken == 2 && s.exits[0].not_taken == 1);
    const Bool when = s.exits[0].when;
    // Skipped exactly where the compare fails.
    const ir::Value cmp = *p.blocks[0].insts[0].write(Reg::S(ir::VCC));
    VS_CHECK(s.bools.node(when) == node(Atom::Lane(cmp), Bool::True, Bool::False));
    VS_CHECK(guarded(s) == 0);
}

VS_TEST(exec_narrowed_without_a_branch_leaves_a_guard)
{
    const Program p = build(
        {
            0x7DA4'0080,  // v_cmpx_eq_u32 exec, 0, v0
            0x7E02'0280,  // v_mov_b32 v1, 0
            S_ENDPGM,
        },
        true);
    const Structured s = structure::structure(p);
    VS_CHECK(!s.guards[0][0].has_value());  // the compare runs in every lane
    VS_REQUIRE(s.guards[0][1].has_value());
    VS_CHECK(support_is(s, *s.guards[0][1], Atom::Kind::Lane));
}

VS_TEST(guards_follow_exec_and_block_reachability)
{
    const Program p = build(
        {
            0x7DA4'0080,  // v_cmpx_eq_u32 exec, 0, v0
            0x7E02'0280,  // v_mov_b32 v1, 0
            0x7E04'0280,  // v_mov_b32 v2, 0
            0xBF88'0005,  // s_cbranch_execz -> end
            0x7E06'0280,  // v_mov_b32 v3, 0
            0x7DA4'0880,  // v_cmpx_eq_u32 exec, 0, v4
            0x7E06'0280,  // v_mov_b32 v3, 0
            0xBE82'0480,  // s_mov_b64 s[2:3], 0
            0x7E0A'0280,  // v_mov_b32 v5, 0
            S_ENDPGM,
        },
        true);
    const Structured s = structure::structure(p);
    auto lane = [&](std::size_t block, std::size_t i) {
        return Atom::Lane(*p.blocks[block].insts[i].write(Reg::S(ir::EXEC)));
    };
    VS_REQUIRE(s.guards[0][1].has_value() && s.guards[1][2].has_value());
    const Bool a = *s.guards[0][1];
    const Bool b = *s.guards[1][2];
    VS_CHECK(s.bools.node(a) == node(lane(0, 0), Bool::False, Bool::True));
    VS_CHECK((s.guards[0] == std::vector<std::optional<Bool>>{std::nullopt, a, a, std::nullopt}));
    // The fallthrough already implies the first compare. Only the second compare needs testing, and
    // the scalar move never needs a guard.
    VS_CHECK(s.bools.node(b) == node(lane(1, 1), Bool::False, Bool::True));
    VS_CHECK((s.guards[1] ==
              std::vector<std::optional<Bool>>{std::nullopt, std::nullopt, b, std::nullopt, b}));
}

VS_TEST(divergent_loop_exits_per_lane)
{
    const Program p = build(
        {
            0xBE82'0480,  // s_mov_b64 s[2:3], 0
            0x4A00'0081,  // 4: v_add_nc_u32 v0, 1, v0
            0x7D84'0080,  // v_cmp_eq_u32 vcc, 0, v0
            0x8882'026A,  // s_or_b64 s[2:3], vcc, s[2:3]
            0x8AFE'027E,  // s_andn2_b64 exec, exec, s[2:3]
            0xBF89'FFFB,  // s_cbranch_execnz -> 4
            0x88FE'027E,  // s_or_b64 exec, exec, s[2:3]
            0x7E02'0280,  // v_mov_b32 v1, 0
            S_ENDPGM,
        },
        true);
    const Structured s = structure::structure(p);
    VS_CHECK(guarded(s) == 0);
    VS_REQUIRE(s.exits[1].is_branch() && s.exits[1].taken == 1 && s.exits[1].not_taken == 2);
    // Back while the lane's compare fails.
    const auto n = s.bools.node(s.exits[1].when);
    VS_REQUIRE(n.has_value());
    VS_CHECK(std::get<0>(*n).kind == Atom::Kind::Lane && std::get<1>(*n) == Bool::True &&
             std::get<2>(*n) == Bool::False);
}

// `s_cselect` on the SCC a lane loop ended with (the blend-shape skinning programs): the loop is left
// when no lane remains, so the select takes its second operand; it was a uniform atom whose value no
// lane has.
VS_TEST(a_select_on_the_condition_a_lane_loop_ended_with_takes_its_second_operand)
{
    const std::uint32_t cnd = 256 | (256 << 9) | (3 << 18);
    const Program p = build(
        {
            0xBE83'037E,  // 0: s_mov_b32 s3, exec_lo
            0xBE9C'037E,  // s_mov_b32 s28, exec_lo
            0x7E3A'0507,  // 8: v_readfirstlane_b32 s29, v7
            0x7DA4'0E1D,  // v_cmpx_eq_u32 exec_lo, s29, v7
            0xBE9C'471C,  // s_andn2_wrexec_b32 s28, s28
            0xBF89'FFFC,  // s_cbranch_execnz -> 8
            0xBEFE'0303,  // s_mov_b32 exec_lo, s3
            0x8503'807E,  // s_cselect_b32 s3, exec_lo, 0
            0xD501'0001,
            cnd,  // v_cndmask_b32 v1, v0, v0, s3
            S_ENDPGM,
        },
        false);
    const Structured s = structure::structure(p);
    const ir::Inst* select = nullptr;
    for (const auto& b : p.blocks)
        for (const auto& i : b.insts)
            if (!select && i.name() == "s_cselect_b32") select = &i;
    VS_REQUIRE(select != nullptr);
    const ir::Value mask = *select->write(Reg::S(3));
    const auto it = s.masks.find(mask);
    VS_CHECK(it != s.masks.end() && it->second == Bool::False);
}

VS_TEST(scalar_branches_stay_uniform)
{
    const Program p = build(
        {
            0xBF06'0100,  // s_cmp_eq_u32 s0, s1
            0xBF85'0001,  // s_cbranch_scc1 -> 12
            0x7E02'0280,  // v_mov_b32 v1, 0
            S_ENDPGM,     // 12
        },
        true);
    const Structured s = structure::structure(p);
    VS_REQUIRE(s.exits[0].is_branch());
    VS_CHECK(support_is(s, s.exits[0].when, Atom::Kind::Uniform));
    VS_CHECK(guarded(s) == 0);
}

VS_TEST(a_mask_whose_high_dword_is_rewritten_is_not_the_compare)
{
    for (const std::uint32_t reader : {
             0x0202'0480u,  // v_cndmask_b32 v1, 0, v2, vcc
             0x5002'0480u,  // v_add_co_ci_u32 v1, vcc, 0, v2, vcc
         }) {
        const Program p = build(
            {
                0x7D84'0080,  // v_cmp_eq_u32 vcc, 0, v0
                0xBEEB'0380,  // s_mov_b32 vcc_hi, 0
                reader,
                S_ENDPGM,
            },
            true);
        const Structured s = structure::structure(p);
        const ir::Value lo = *p.blocks[0].insts[2].read(Reg::S(ir::VCC));
        VS_CHECK(p.blocks[0].insts[0].write(Reg::S(ir::VCC)) == lo);
        // The reader sees (compare low dword, 0): lanes 32-63 are clear. The mask exported for that
        // low dword must not be the compare.
        VS_CHECK(mask_node(s, lo) != node(Atom::Lane(lo), Bool::False, Bool::True));
    }
}

VS_TEST(a_mask_joined_over_a_back_edge_with_a_rewritten_high_dword_is_not_the_compare)
{
    const Program p = build(
        {
            0x7D84'0080,  // v_cmp_eq_u32 vcc, 0, v0
            0x0202'0480,  // 4: v_cndmask_b32 v1, 0, v2, vcc
            0x7D84'0280,  // v_cmp_eq_u32 vcc, 0, v1
            0xBEEB'0380,  // s_mov_b32 vcc_hi, 0
            0xBF85'FFFC,  // s_cbranch_scc1 -> 4
            S_ENDPGM,
        },
        true);
    const Structured s = structure::structure(p);
    // The select reads the loop header's join; over the back edge it takes (compare low dword, 0),
    // which only the join sees.
    const ir::Value lo = *p.blocks[1].insts[1].write(Reg::S(ir::VCC));
    VS_CHECK(mask_node(s, lo) != node(Atom::Lane(lo), Bool::False, Bool::True));
}

VS_TEST(irreducible_loops_are_split)
{
    const Program p = build(
        {
            0xBF06'0100,  // s_cmp_eq_u32 s0, s1
            0xBF85'0001,  // s_cbranch_scc1 -> 12, into the loop's middle
            0x8002'8102,  // 8: s_add_u32 s2, s2, 1
            0xBF84'FFFE,  // 12: s_cbranch_scc0 -> 8
            S_ENDPGM,
        },
        false);
    // The loop keeps 12 as its header; the entry into 8 gets a copy of it.
    VS_REQUIRE(p.blocks.size() > 4);
    const ir::Block& copy = p.blocks[4];
    VS_CHECK(copy.offset == 8 && copy.exit == ir::Exit::Jump(2));
    VS_CHECK((p.blocks[0].exit.successors() == std::vector<std::size_t>{2, 4}));
    VS_CHECK((p.blocks[1].preds == std::vector<std::size_t>{2}));
    VS_CHECK(try_shader([&] { return structure::structure(p); }).has_value());
}

VS_TEST(mask_phi_joining_constant_and_compare_records_edge_bools)
{
    const std::uint32_t cnd = 258 | (259 << 9) | (2 << 18);
    const Program p = build(
        {
            0xBF06'8000,  // 0: s_cmp_eq_u32 s0, 0
            0xBF85'0002,  // 4: s_cbranch_scc1 -> 16
            0xBE82'0380,  // 8: s_mov_b32 s2, 0
            0xBF82'0002,  // 12: s_branch -> 24
            0xD4C2'0002,
            0x0002'0080,  // 16: v_cmp_eq_u32 s2, 0, v0
            0xD501'0001,
            cnd,  // 24: v_cndmask_b32 v1, v2, v3, s2
            S_ENDPGM,
        },
        false);
    const Structured s = structure::structure(p);
    const ir::Phi* phi = phi_of(p, Reg::S(2));
    VS_REQUIRE(phi != nullptr);
    VS_CHECK(mask_node(s, phi->value) == node(Atom::Phi(phi->value), Bool::False, Bool::True));
    const auto it = s.phi_masks.find(phi->value);
    VS_REQUIRE(it != s.phi_masks.end());
    const auto& edges = it->second;
    VS_CHECK(edges.size() == 2);
    VS_CHECK(contains(edges, Bool::False));
    VS_CHECK(std::any_of(edges.begin(), edges.end(), [&](Bool e) { return support_is(s, e, Atom::Kind::Lane); }));
}

VS_TEST(loop_mask_accumulator_from_zero_tracks_phi_and_edges)
{
    const std::uint32_t cnd = 258 | (259 << 9) | (2 << 18);
    const Program p = build(
        {
            0xBE82'0480,  // 0: s_mov_b64 s[2:3], 0
            0xBE84'0380,  // 4: s_mov_b32 s4, 0
            0xD501'0001,
            cnd,          // 8: v_cndmask_b32 v1, v2, v3, s[2:3]
            0x7D84'0080,  // 16: v_cmp_eq_u32 vcc, 0, v0
            0x8882'026A,  // 20: s_or_b64 s[2:3], vcc, s[2:3]
            0x8004'8104,  // 24: s_add_u32 s4, s4, 1
            0xBF0A'8404,  // 28: s_cmp_lt_u32 s4, 4
            0xBF85'FFF9,  // 32: s_cbranch_scc1 -> 8
            0xD501'0004,
            cnd,  // 36: v_cndmask_b32 v4, v2, v3, s[2:3]
            S_ENDPGM,
        },
        true);
    const Structured s = structure::structure(p);
    const ir::Phi* phi = phi_of(p, Reg::S(2));
    VS_REQUIRE(phi != nullptr);
    VS_CHECK(s.masks.count(phi->value) == 1);
    const auto it = s.phi_masks.find(phi->value);
    VS_REQUIRE(it != s.phi_masks.end());
    const auto& edges = it->second;
    VS_CHECK(edges.size() == 2);
    VS_CHECK(contains(edges, Bool::False));
    VS_CHECK(std::any_of(edges.begin(), edges.end(), [&](Bool e) {
        const auto sup = s.bools.support(e);
        const bool has_phi = std::find(sup.begin(), sup.end(), Atom::Phi(phi->value)) != sup.end();
        const bool has_lane =
            std::any_of(sup.begin(), sup.end(), [](const Atom& a) { return a.kind == Atom::Kind::Lane; });
        return has_phi && has_lane;
    }));
}

VS_TEST(loop_mask_phi_reset_to_constant_on_back_edge)
{
    const std::uint32_t cnd = 258 | (259 << 9) | (2 << 18);
    const Program p = build(
        {
            0xD4C2'0002,
            0x0002'0080,  // 0: v_cmp_eq_u32 s2, 0, v0
            0xBE84'0380,  // 8: s_mov_b32 s4, 0
            0xD501'0001,
            cnd,          // 12: v_cndmask_b32 v1, v2, v3, s2
            0xBE82'0380,  // 20: s_mov_b32 s2, 0
            0x8004'8104,  // 24: s_add_u32 s4, s4, 1
            0xBF0A'8404,  // 28: s_cmp_lt_u32 s4, 4
            0xBF85'FFFA,  // 32: s_cbranch_scc1 -> 12
            S_ENDPGM,
        },
        false);
    const Structured s = structure::structure(p);
    const ir::Phi* phi = phi_of(p, Reg::S(2));
    VS_REQUIRE(phi != nullptr);
    VS_CHECK(s.masks.count(phi->value) == 1);
    const auto it = s.phi_masks.find(phi->value);
    VS_REQUIRE(it != s.phi_masks.end());
    const auto& edges = it->second;
    VS_CHECK(edges.size() == 2);
    VS_CHECK(contains(edges, Bool::False));
    VS_CHECK(std::any_of(edges.begin(), edges.end(), [&](Bool e) { return support_is(s, e, Atom::Kind::Lane); }));
}

VS_TEST_MAIN()
