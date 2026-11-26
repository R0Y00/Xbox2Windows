// Port of the tests in the upstream Rust module emit/wave.rs.
//
// The tests that build programs need vs-core's ir::build (X2W_VS_HAVE_IR, set by wave.cmake when
// the ir implementation is in the tree); the others run without it.

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "test_util.h"
#include "x2w_shader/emit/bitcode.h"
#include "x2w_shader/emit/wave.h"

using namespace x2w::vs;
using wave::Op;
using wave::Shape;
using wave::Sym;
using wave::detail::dpp_lane;
using wave::detail::per_lane;

namespace {

void dpp_lanes_follow_the_isa() {
    // row_shr:1 with bound_ctrl: lane 16 reads 0, lane 17 lane 16.
    const gfx10::Dpp shr = gfx10::DppRow{0x111, 0xF, 0xF, true, false};
    CHECK((dpp_lane(shr, 16) == std::tuple{16u, true, true}));
    CHECK((dpp_lane(shr, 17) == std::tuple{16u, false, true}));
    // row_mirror: lane 3 reads lane 12.
    const gfx10::Dpp mirror = gfx10::DppRow{0x140, 0xF, 0xF, true, false};
    CHECK((dpp_lane(mirror, 3) == std::tuple{12u, false, true}));
    // dpp8 [1,0,3,2,5,4,7,6] swaps neighbours.
    const gfx10::Dpp swap = gfx10::DppLanes{1 | 3 << 6 | 2 << 9 | 5 << 12 | 4 << 15 | 7 << 18 | 6 << 21, false};
    CHECK((dpp_lane(swap, 8) == std::tuple{9u, false, true}));
}

void prefixes_are_recognised() {
    const ir::Value x{1}, m{2};
    auto red = [&](std::uint64_t lanes) { return Sym::red(Op::AddU, x, m, 0, lanes); };
    std::vector<Sym> inclusive;
    for (std::uint32_t l = 0; l < 64; ++l) inclusive.push_back(red(~std::uint64_t{0} >> (63 - l)));
    CHECK(per_lane(inclusive, ~std::uint64_t{0}) == Shape::prefix(Op::AddU, x, m, 0, true));
    std::vector<Sym> exclusive;
    for (std::uint32_t l = 0; l < 64; ++l) exclusive.push_back(red((std::uint64_t{1} << l) - 1));
    exclusive[0] = Sym::konst(0);
    CHECK(per_lane(exclusive, ~std::uint64_t{0}) == Shape::prefix(Op::AddU, x, m, 0, false));
    CHECK(per_lane(std::vector<Sym>(64, red(~std::uint64_t{0})), ~std::uint64_t{0}) ==
          Shape::active(Op::AddU, x, m, 0));
}

void buddies_are_the_two_halves_of_an_aligned_run() {
    using wave::buddies;
    CHECK(buddies(0b01, 0b10));
    CHECK(buddies(0b1100, 0b0011));
    CHECK(buddies(0xF0, 0x0F));
    CHECK(buddies(std::uint64_t{0xFFFF'FFFFu}, std::uint64_t{0xFFFF'FFFFu} << 32));
    CHECK(!buddies(0b010, 0b100));     // lanes 1 and 2: not one pair
    CHECK(!buddies(0b0011, 0b110000));  // pairs 0 and 2
    CHECK(!buddies(0b0101, 0b1010));    // not runs
    CHECK(!buddies(0b01, 0b1100));      // unequal
    CHECK(!buddies(0b11, 0b11));
}

void per_lane_prefix_soundness() {
    const ir::Value v0{1}, v_other{2}, m0{3}, m_other{4};
    auto red = [](std::optional<Op> op, ir::Value src, ir::Value mask, std::uint32_t ident, std::uint64_t lanes) {
        return Sym::red(op, src, mask, ident, lanes);
    };
    for (const auto& [wave, all] : {std::pair<std::size_t, std::uint64_t>{32, 0xFFFF'FFFFu},
                                    std::pair<std::size_t, std::uint64_t>{64, ~std::uint64_t{0}}}) {
        auto below_mask = [](std::size_t l) { return l == 0 ? std::uint64_t{0} : (std::uint64_t{1} << l) - 1; };

        std::vector<Sym> valid_inc;
        for (std::size_t l = 0; l < wave; ++l)
            valid_inc.push_back(red(Op::AddU, v0, m0, 0, below_mask(l) | (std::uint64_t{1} << l)));
        valid_inc[0] = red(std::nullopt, v0, m0, 0, 1);
        CHECK(per_lane(valid_inc, all) == Shape::prefix(Op::AddU, v0, m0, 0, true));

        std::vector<Sym> valid_exc;
        for (std::size_t l = 0; l < wave; ++l) valid_exc.push_back(red(Op::AddU, v0, m0, 0, below_mask(l)));
        valid_exc[0] = Sym::konst(0);
        valid_exc[1] = red(std::nullopt, v0, m0, 0, 1);
        CHECK(per_lane(valid_exc, all) == Shape::prefix(Op::AddU, v0, m0, 0, false));

        auto mixed_other_src = valid_exc;
        mixed_other_src[0] = red(std::nullopt, v_other, m0, 0, 1);
        CHECK(per_lane(mixed_other_src, all) == std::nullopt);

        auto mixed_same_src = valid_exc;
        mixed_same_src[0] = red(std::nullopt, v0, m0, 0, 1);
        CHECK(per_lane(mixed_same_src, all) == std::nullopt);

        auto exc_bad_lane1_src = valid_exc;
        exc_bad_lane1_src[1] = red(std::nullopt, v_other, m0, 0, 1);
        CHECK(per_lane(exc_bad_lane1_src, all) == std::nullopt);

        auto exc_bad_lane1_set = valid_exc;
        exc_bad_lane1_set[1] = red(std::nullopt, v0, m0, 0, 0b11);
        CHECK(per_lane(exc_bad_lane1_set, all) == std::nullopt);

        for (const Sym& bad0 : {red(std::nullopt, v0, m0, 1, 1), red(std::nullopt, v_other, m0, 0, 1),
                                red(std::nullopt, v0, m_other, 0, 1), red(std::nullopt, v0, m0, 0, 0), Sym::konst(0)}) {
            auto bad_inc = valid_inc;
            bad_inc[0] = bad0;
            CHECK(per_lane(bad_inc, all) == std::nullopt);
        }

        std::vector<Sym> bad_ident;
        for (std::size_t l = 0; l < wave; ++l)
            bad_ident.push_back(red(Op::AddU, v0, m0, 1, below_mask(l) | (std::uint64_t{1} << l)));
        bad_ident[0] = red(std::nullopt, v0, m0, 1, 1);
        CHECK(per_lane(bad_ident, all) == std::nullopt);
    }
}

#ifdef X2W_VS_HAVE_IR

using ir::EXEC;
using ir::Program;
using ir::Reg;

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

constexpr std::uint32_t sop1(std::uint8_t sdst, std::uint8_t op, std::uint8_t ssrc0) {
    return 0xBE80'0000u | std::uint32_t(sdst) << 16 | std::uint32_t(op) << 8 | std::uint32_t(ssrc0);
}

constexpr std::uint32_t vop1(std::uint8_t vdst, std::uint8_t op, std::uint16_t src0) {
    return (0x3Fu << 25) | std::uint32_t(vdst) << 17 | std::uint32_t(op) << 9 | (std::uint32_t(src0) & 0x1FF);
}

constexpr std::uint32_t vop2(std::uint8_t vdst, std::uint8_t op, std::uint16_t src0, std::uint8_t vsrc1) {
    return std::uint32_t(op) << 25 | std::uint32_t(vdst) << 17 | std::uint32_t(vsrc1) << 9 | (std::uint32_t(src0) & 0x1FF);
}

constexpr std::uint32_t dpp16(std::uint8_t vsrc0, std::uint16_t ctrl, std::uint8_t row_mask, std::uint8_t bank_mask,
                              bool bound_ctrl) {
    return std::uint32_t(row_mask) << 28 | std::uint32_t(bank_mask) << 24 | std::uint32_t(bound_ctrl) << 19 |
           std::uint32_t(ctrl) << 8 | std::uint32_t(vsrc0);
}

constexpr std::array<std::uint32_t, 2> vop3(std::uint8_t vdst, std::uint16_t op, std::uint16_t src0, std::uint16_t src1,
                                            std::uint16_t src2) {
    return {0xD400'0000u | std::uint32_t(op) << 16 | std::uint32_t(vdst),
            (std::uint32_t(src0) & 0x1FF) | ((std::uint32_t(src1) & 0x1FF) << 9) | ((std::uint32_t(src2) & 0x1FF) << 18)};
}

Program build(const std::vector<std::uint32_t>& words, bool wave64) {
    std::vector<std::uint8_t> code;
    auto put = [&](std::uint32_t w) {
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(w >> (8 * k)));
    };
    for (std::uint32_t w : words) put(w);
    put(S_ENDPGM);
    return ir::build(code, wave64, 256);
}

ir::Value input(const Program& p, Reg r) {
    for (const auto& [reg, v] : p.inputs) {
        if (reg == r) return v;
    }
    x2w::vs::bitcode::bug("no such input");
}

void constant_decodes_width_and_inline_kinds() {
    const std::uint8_t b64 = 0x04;
    const std::uint8_t b32 = 0x03;
    auto check = [](const std::vector<std::uint32_t>& words, bool wave64) {
        const Program p = build(words, wave64);
        return wave::detail::constant(p.blocks[0].insts[0]);
    };
    using O = std::optional<std::uint64_t>;
    CHECK(check({sop1(EXEC, b64, 194)}, true) == O{0xFFFF'FFFF'FFFF'FFFEull});
    CHECK(check({sop1(EXEC, b64, 196)}, true) == O{0xFFFF'FFFF'FFFF'FFFCull});
    CHECK(check({sop1(EXEC, b64, 208)}, true) == O{0xFFFF'FFFF'FFFF'FFF0ull});
    CHECK(check({sop1(EXEC, b64, 193)}, true) == O{~std::uint64_t{0}});
    CHECK(check({sop1(EXEC, b64, gfx10::NULL_CODE)}, true) == O{0});
    CHECK(check({sop1(EXEC, b64, 243)}, true) == std::nullopt);
    CHECK(check({sop1(EXEC, b64, 255), 0x1234'5678}, true) == O{0x1234'5678});

    CHECK(check({sop1(EXEC, b32, 194)}, false) == O{0xFFFF'FFFE});
    CHECK(check({sop1(EXEC, b32, 243)}, false) == O{0xBF80'0000});
    CHECK(check({sop1(EXEC, b32, gfx10::NULL_CODE)}, false) == O{0});
    CHECK(check({sop1(EXEC, b32, 255), 0x1234'5678}, false) == O{0x1234'5678});
    CHECK(check({sop1(EXEC, b32, 0)}, false) == std::nullopt);
}

void exec_minus_two_masks_lane_zero_and_enforces_dpp_guard() {
    for (bool wave64 : {false, true}) {
        std::vector<std::uint32_t> w;
        const std::uint8_t save_op = wave64 ? 0x25 : 0x3D;
        const std::uint8_t mov_exec = wave64 ? 0x04 : 0x03;
        auto push_exec = [&](std::uint8_t k) { w.push_back(sop1(EXEC, mov_exec, k)); };
        auto push_exec_lit = [&](std::uint32_t lit) {
            w.push_back(sop1(EXEC, mov_exec, 255));
            w.push_back(lit);
        };
        auto extend = [&](std::array<std::uint32_t, 2> a) { w.insert(w.end(), a.begin(), a.end()); };

        w.push_back(sop1(0, save_op, 193));
        extend(vop3(1, 0x101, 128, 256, 0));
        for (int shift : {1, 2, 4, 8}) {
            w.push_back(vop2(1, 0x25, 0xFA, 1));
            w.push_back(dpp16(1, static_cast<std::uint16_t>(0x110 + shift), 0xF, 0xF, false));
        }
        extend(vop3(2, 0x360, 257, 128 + 15, 0));
        push_exec_lit(0xFFFF'0000);
        w.push_back(vop2(1, 0x25, 2, 1));

        if (wave64) {
            extend(vop3(2, 0x360, 257, 128 + 31, 0));
            w.push_back(sop1(EXEC, 0x03, 128));
            w.push_back(sop1(EXEC + 1, 0x03, 255));
            w.push_back(0x0000'FFFF);
            w.push_back(vop2(1, 0x25, 2, 1));

            extend(vop3(2, 0x360, 257, 128 + 47, 0));
            w.push_back(sop1(EXEC, 0x03, 128));
            w.push_back(sop1(EXEC + 1, 0x03, 255));
            w.push_back(0xFFFF'0000);
            w.push_back(vop2(1, 0x25, 2, 1));
        }

        push_exec(193);
        w.push_back(vop1(3, 0x01, 257));
        w.push_back(vop1(3, 0x01, 0xFA));
        w.push_back(dpp16(1, 0x111, 0xF, 0xF, false));

        extend(vop3(2, 0x360, 257, 128 + 15, 0));
        push_exec_lit(0x0001'0000);
        w.push_back(vop1(3, 0x01, 2));

        if (wave64) {
            extend(vop3(2, 0x360, 257, 128 + 31, 0));
            w.push_back(sop1(EXEC, 0x03, 128));
            w.push_back(sop1(EXEC + 1, 0x03, 129));
            w.push_back(vop1(3, 0x01, 2));

            extend(vop3(2, 0x360, 257, 128 + 47, 0));
            w.push_back(sop1(EXEC, 0x03, 128));
            w.push_back(sop1(EXEC + 1, 0x03, 255));
            w.push_back(0x0001'0000);
            w.push_back(vop1(3, 0x01, 2));
        }

        push_exec(193);
        w.push_back(vop1(2, 0x01, 128));
        w.push_back(vop1(5, 0x01, 135));
        push_exec(194);
        w.push_back(vop1(2, 0x01, 259));
        w.push_back(vop1(4, 0x01, 0xFA));
        w.push_back(dpp16(5, 0xE4, 0xF, 0xF, false));
        w.push_back(sop1(EXEC, mov_exec, 0));

        const Program p = build(w, wave64);
        const auto& insts = p.blocks[0].insts;
        const auto saved = *insts[0].write(Reg::S(0));
        const auto v0_in = input(p, Reg::V(0));
        const auto v2_val = *insts[insts.size() - 4].write(Reg::V(2));
        const auto v4_val = *insts[insts.size() - 3].write(Reg::V(4));
        const auto secs = wave::sections(p, 0);
        const auto& sec = secs.at(0);
        const auto s2 = sec.shapes.find(v2_val);
        CHECK(s2 != sec.shapes.end() && s2->second == Shape::prefix(Op::AddU, v0_in, saved, 0, false));
        CHECK(!sec.shapes.contains(v4_val));
    }
}

// A wave OR whose last step stays in the section's VGPR: the lower lanes add the first half's to
// themselves, so only the upper ones hold the whole, and the last lane is read after the section
// (an upscaler's programs).
void one_lane_of_what_a_section_leaves_is_read_after_it() {
    auto words = [](std::uint32_t lane) {
        std::vector<std::uint32_t> w(wave::WAVE_OR_V7_TO_S7.begin(), wave::WAVE_OR_V7_TO_S7.begin() + 14);
        w.insert(w.end(), {
                              0xD760'0007,
                              0x0001'0101,  // v_readlane_b32 s7, v1, 0
                              0x3802'0207,  // v_or_b32 v1, s7, v1
                              0xBEFE'0404,  // s_mov_b64 exec, s[4:5]
                              0xD760'0006,
                              0x0001'0101 | lane << 9,  // v_readlane_b32 s6, v1, lane
                              S_ENDPGM,
                          });
        return w;
    };
    auto read = [&](std::uint32_t lane) {
        const Program p = build(words(lane), true);
        const auto& insts = p.blocks[0].insts;
        const auto saved = *insts[0].write(Reg::S(4));
        const auto v7 = input(p, Reg::V(7));
        const auto got = *insts[insts.size() - 2].write(Reg::S(6));
        const auto secs = wave::sections(p, 0);
        const auto& sec = secs.at(0);
        std::optional<Shape> found;
        if (const auto it = sec.reads.find(got); it != sec.reads.end()) found = it->second;
        return std::pair{found, Shape::active(Op::Or, v7, saved, 0)};
    };
    const auto [found, whole] = read(63);
    CHECK(found == whole);
    // Lane 0 has the first half's alone: still a lane read.
    CHECK(read(0).first == std::nullopt);
}

// A 16-bit floating-point wave sum uses the tree the lowering
// reproduces. A tree in another order is not followed.
void a_sum_of_halves_over_the_wave_is_followed_in_the_order_of_its_tree() {
    using W2 = std::array<std::uint32_t, 2>;
    auto words = [](W2 first, W2 second) {
        return std::vector<std::uint32_t>{
            0xBE84'25C1,  // s_or_saveexec_b64 s[4:5], -1
            0xD501'0001,
            0x0012'0E80,  // v_cndmask_b32 v1, 0, v7, s[4:5]
            first[0],
            first[1],
            second[0],
            second[1],
            0x6402'02E9,
            0x0009'2401,  // v_add_f16 v1, v1, v1 dpp8:[4,4,4,4,0,0,0,0]
            0x6402'02FA,
            0xFF09'4001,  // v_add_f16 v1, v1, v1 row_mirror bound_ctrl:1
            0xD778'0002,
            0x0201'0101,  // v_permlanex16_b32 v2, v1, 0, 0
            0x6402'0302,  // v_add_f16 v1, v2, v1
            0xD760'0007,
            0x0001'4101,  // v_readlane_b32 s7, v1, 32
            0x6402'0207,  // v_add_f16 v1, s7, v1
            0xD760'0007,
            0x0001'0101,  // v_readlane_b32 s7, v1, 0
            0xBEFE'0404,  // s_mov_b64 exec, s[4:5]
        };
    };
    const W2 pairs = {0x6402'02E9, 0xDE54'C101};  // v_add_f16 v1, v1, v1 dpp8:[1,0,3,2,5,4,7,6]
    const W2 quads = {0x6402'02E9, 0x9360'1201};  // v_add_f16 v1, v1, v1 dpp8:[2,2,0,0,6,6,4,4]
    auto shape = [](const std::vector<std::uint32_t>& w) {
        const Program p = build(w, true);
        const auto& insts = p.blocks[0].insts;
        const auto saved = *insts[0].write(Reg::S(4));
        const auto v7 = input(p, Reg::V(7));
        // ..., v_readlane_b32 s7, v1, 0; s_mov_b64 exec, s[4:5]; s_endpgm
        const auto total = *insts[insts.size() - 3].write(Reg::S(7));
        const auto secs = wave::sections(p, 0);
        const auto& sec = secs.at(0);
        std::optional<Shape> found;
        if (const auto it = sec.shapes.find(total); it != sec.shapes.end()) found = it->second;
        return std::pair{found, Shape::active(Op::AddH, v7, saved, 0)};
    };
    const auto [found, want] = shape(words(pairs, quads));
    CHECK(found == want);
    // lanes two apart first, neighbours second: every lane counted once, but
    // (x0 + x2) + (x1 + x3) rounds differently from (x0 + x1) + (x2 + x3)
    CHECK(shape(words(quads, pairs)).first == std::nullopt);
}

void wave32_mov_b64_all_lanes_keeps_exec_masked_for_dpp() {
    const std::vector<std::uint32_t> w = {
        sop1(0, 0x3D, 193),
        vop1(1, 0x01, 135),
        sop1(EXEC, 0x04, 193),
        vop1(2, 0x01, 0xFA),
        dpp16(1, 0xE4, 0xF, 0xF, false),
        sop1(EXEC + 1, 0x03, 193),
        vop1(3, 0x01, 0xFA),
        dpp16(1, 0xE4, 0xF, 0xF, false),
        sop1(EXEC, 0x03, 0),
    };
    const Program p = build(w, false);
    const auto& insts = p.blocks[0].insts;
    const auto v2_val = *insts[3].write(Reg::V(2));
    const auto v3_val = *insts[5].write(Reg::V(3));
    const auto secs = wave::sections(p, 0);
    const auto& sec = secs.at(0);
    const auto a = sec.shapes.find(v2_val);
    const auto b = sec.shapes.find(v3_val);
    CHECK(a != sec.shapes.end() && a->second == Shape::konst(7));
    CHECK(b != sec.shapes.end() && b->second == Shape::konst(7));
}

#endif  // X2W_VS_HAVE_IR

}  // namespace

int main() {
    dpp_lanes_follow_the_isa();
    prefixes_are_recognised();
    buddies_are_the_two_halves_of_an_aligned_run();
    per_lane_prefix_soundness();
#ifdef X2W_VS_HAVE_IR
    constant_decodes_width_and_inline_kinds();
    exec_minus_two_masks_lane_zero_and_enforces_dpp_guard();
    one_lane_of_what_a_section_leaves_is_read_after_it();
    a_sum_of_halves_over_the_wave_is_followed_in_the_order_of_its_tree();
    wave32_mov_b64_all_lanes_keeps_exec_masked_for_dpp();
    std::printf("wave_test: 9 of 9 tests ran\n");
#else
    std::printf("wave_test: 4 of 9 tests ran (no ir::build: X2W_VS_HAVE_IR unset)\n");
#endif
    return test_util::finish("wave_test");
}
