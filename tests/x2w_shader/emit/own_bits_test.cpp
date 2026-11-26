// Port of the tests in the upstream Rust module emit/own_bits.rs (needs vs-core's ir, vs-front's structure and
// vs-lower's lower::reads_mask_as_data).

#include <algorithm>
#include <cstdint>
#include <set>
#include <string_view>
#include <vector>

#include "test_util.h"
#include "x2w_shader/emit/own_bits.h"
#include "x2w_shader/emit/wave.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/structure.h"

using namespace x2w::vs;
using ir::Program;
using ir::Reg;
using own_bits::OwnBits;

namespace {

constexpr std::uint32_t ADD_BIT_NUMBER = 0x4A0A'0A0A;  // v_add_nc_u32 v5, s10, v5

// A synthetic scalarised loop takes the wave OR of v7 in s7, a loop over its bits with the lowest one's number in s10 and
// the bit in s11, and `body` for the lanes whose v7 has the bit (`own`) or for every lane; `latch`
// runs in every lane after the body, `after` after the loop.
Program light_loop(bool own, const std::vector<std::uint32_t>& body, const std::vector<std::uint32_t>& latch,
                   const std::vector<std::uint32_t>& after) {
    std::vector<std::uint32_t> w(wave::WAVE_OR_V7_TO_S7.begin(), wave::WAVE_OR_V7_TO_S7.end());
    const std::vector<std::uint32_t> test = own ? std::vector<std::uint32_t>{
                                                      0x3608'0E0B,  // v_and_b32 v4, s11, v7
                                                      0xBE8C'047E,  // s_mov_b64 s[12:13], exec
                                                      0x7DAA'0880,  // v_cmpx_ne_u32 exec, 0, v4
                                                  }
                                                : std::vector<std::uint32_t>{0xBE8C'047E};  // s_mov_b64 s[12:13], exec
    // from the loop's head to its last branch
    const std::size_t head = 3 + test.size() + std::size_t(own) + body.size() + 1 + latch.size() + 3;
    w.insert(w.end(), {
                          0xBF06'8007,                                   // s_cmp_eq_u32 s7, 0
                          0xBF85'0000u | static_cast<std::uint32_t>(3 + head),  // s_cbranch_scc1 past the loop
                          0x7E06'0207,                                   // v_mov_b32 v3, s7
                          0xBF82'0001,                                   // s_branch to the loop's head
                          0x7E06'0209,                                   // again: v_mov_b32 v3, s9
                          0x7E10'0503,                                   // head: v_readfirstlane_b32 s8, v3
                          0xBE8A'1308,                                   // s_ff1_i32_b32 s10, s8
                          0x8F0B'0A81,                                   // s_lshl_b32 s11, 1, s10
                      });
    w.insert(w.end(), test.begin(), test.end());
    if (own) w.push_back(0xBF88'0000u | static_cast<std::uint32_t>(body.size()));  // s_cbranch_execz past the body
    w.insert(w.end(), body.begin(), body.end());
    w.push_back(0xBEFE'040C);  // s_mov_b64 exec, s[12:13]
    w.insert(w.end(), latch.begin(), latch.end());
    w.insert(w.end(), {
                          0x8909'080B,                                                  // s_xor_b32 s9, s11, s8
                          0xBF06'0B08,                                                  // s_cmp_eq_u32 s8, s11
                          0xBF84'0000u | static_cast<std::uint32_t>(0x1'0000 - head - 1),  // s_cbranch_scc0 again
                      });
    w.insert(w.end(), after.begin(), after.end());
    w.push_back(0xBF81'0000);  // s_endpgm
    std::vector<std::uint8_t> code;
    for (std::uint32_t x : w) {
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(x >> (8 * k)));
    }
    return ir::build(code, true, 256);
}

OwnBits found(const Program& p) {
    const auto st = structure::structure(p);
    return own_bits::find(p, st, false);
}

std::vector<const ir::Inst*> insts(const Program& p) {
    std::vector<const ir::Inst*> out;
    for (const auto& b : p.blocks) {
        for (const auto& i : b.insts) out.push_back(&i);
    }
    return out;
}

bool has(const Program& p, std::string_view name) {
    const auto all = insts(p);
    return std::any_of(all.begin(), all.end(), [&](const ir::Inst* i) { return i->name() == name; });
}

void a_wave_or_that_only_steers_a_loop_over_own_bits_is_found() {
    const Program p = light_loop(true, {ADD_BIT_NUMBER}, {}, {});
    for (std::string_view n : {"s_ff1_i32_b32", "s_lshl_b32", "v_and_b32", "v_cmpx_ne_u32", "s_xor_b32",
                               "v_readfirstlane_b32", "s_cbranch_execz"}) {
        CHECK(has(p, n));
    }
    // a loop: its last branch goes back to the copy of the bits still to visit
    bool back = false;
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        for (std::size_t s : p.blocks[b].exit.successors()) back |= s < b;
    }
    CHECK(back);
    const OwnBits own = found(p);
    // the OR is the section's last lane read, and the bits still to visit (v3) are alike
    const auto all = insts(p);
    std::optional<ir::Value> total;
    for (auto it = all.rbegin(); it != all.rend(); ++it) {
        if ((*it)->name() == "v_readlane_b32") {
            total = (*it)->write(Reg::S(7));
            break;
        }
    }
    CHECK(total.has_value());
    CHECK(total && own.totals == std::set<ir::Value>{*total});
    const ir::Inst* first = nullptr;
    for (const ir::Inst* i : all) {
        if (i->name() == "v_readfirstlane_b32") {
            first = i;
            break;
        }
    }
    CHECK(first != nullptr);
    CHECK(first && first->read(Reg::V(3)) && own.alike.contains(*first->read(Reg::V(3))));
}

void a_loop_whose_lanes_all_work_at_every_bit_needs_the_whole_waves_or() {
    // (the same title's volume loops, 0xa668 on: no test of the lane's own mask)
    const Program p = light_loop(false, {ADD_BIT_NUMBER}, {}, {});
    CHECK(found(p).totals.empty());
}

void a_bit_number_read_after_the_loop_is_not_the_lanes_own() {
    // v_mov_b32 v6, s10: the last bit of the OR, which another lane may have set
    const Program p = light_loop(true, {ADD_BIT_NUMBER}, {}, {0x7E0C'020A});
    CHECK(found(p).totals.empty());
}

void the_bits_still_to_visit_read_in_the_body_are_not_the_lanes_own() {
    // v_add_nc_u32 v5, s8, v5: s8 holds the other lanes' bits too
    const Program p = light_loop(true, {0x4A0A'0A08}, {}, {});
    CHECK(found(p).totals.empty());
}

void a_count_of_the_rounds_is_not_the_lanes_own() {
    // s_add_i32 s14, s14, 1 where every lane passes, read after the loop (v_mov_b32 v6, s14): one
    // round for each bit any lane has
    Program p = light_loop(true, {ADD_BIT_NUMBER}, {0x810E'810E}, {0x7E0C'020E});
    CHECK(has(p, "s_add_i32"));
    CHECK(found(p).totals.empty());
    // (reading s14 there without counting in it is nothing of the loop's)
    p = light_loop(true, {ADD_BIT_NUMBER}, {}, {0x7E0C'020E});
    CHECK(found(p).totals.size() == 1);
}

void an_effect_where_every_lane_passes_is_not_the_lanes_own() {
    // ds_write_b32 v5, v5 after the body: once for each bit any lane has
    const Program p = light_loop(true, {ADD_BIT_NUMBER}, {0xD834'0000, 0x0000'0505}, {});
    CHECK(has(p, "ds_write_b32"));
    CHECK(found(p).totals.empty());
}

}  // namespace

int main() {
    a_wave_or_that_only_steers_a_loop_over_own_bits_is_found();
    a_loop_whose_lanes_all_work_at_every_bit_needs_the_whole_waves_or();
    a_bit_number_read_after_the_loop_is_not_the_lanes_own();
    the_bits_still_to_visit_read_in_the_body_are_not_the_lanes_own();
    a_count_of_the_rounds_is_not_the_lanes_own();
    an_effect_where_every_lane_passes_is_not_the_lanes_own();
    return test_util::finish("own_bits_test");
}
