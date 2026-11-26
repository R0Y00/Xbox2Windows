// Port of the tests in the upstream Rust module emit/threading.rs (needs vs-core's ir and vs-front's structure).

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

#include "test_util.h"
#include "x2w_shader/emit/threading.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/structure.h"

using namespace x2w::vs;
using ir::Exit;
using ir::Program;
using ir::Reg;
using threading::constant_edges;

namespace {

// s8 initially holds a descriptor address. One arm overwrites it with a buffer load and sets v1.
// Only that arm can use s8 as the stored value.
constexpr std::array<std::uint32_t, 19> WORDS = {
    0xbe810380,              // s1 = 0 (table address high)
    0xf4080200, 0xfa000000,  // load SRV into s[8:11]
    0xf4080300, 0xfa000020,  // load UAV into s[12:15]
    0x7e020280,              // v1 = 0
    0xbf068002, 0xbf850003,  // if s2 == 0, jump to join
    0xf4200204, 0xfa00000c,  // s8 = buffer.Load(12)
    0x7e020281,              // v1 = 1
    0x7d840280, 0xbf870002,  // join: if v1 == 0, jump to constant
    0x7e060208, 0xbf820001,  // v3 = s8, jump to store
    0x7e060293,              // constant: v3 = 19
    0xe0700000, 0x80030300,  // store v3 at byte 0
    0xbf810000,              // end
};

Program program(const std::array<std::uint32_t, 19>& words) {
    std::vector<std::uint8_t> code;
    for (std::uint32_t w : words) {
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(w >> (8 * k)));
    }
    return ir::build(code, false, 4);
}

void only_the_buffer_load_reaches_the_numeric_read() {
    Program p = program(WORDS);
    const auto st = structure::structure(p);
    const auto edges = constant_edges(p, st);
    using E = std::tuple<std::size_t, std::size_t, std::size_t>;
    CHECK((edges == std::vector<E>{E{0, 2, 4}}));
    p = ir::thread_edges(p, edges);
    const auto& load = p.blocks.at(1).insts.at(0);
    CHECK(p.blocks.at(3).insts.at(0).read(Reg::S(8)) == load.write(Reg::S(8)));
    CHECK(p.blocks.at(0).exit == Exit::Branch(6, 1));
    CHECK(p.blocks.at(6).exit == Exit::Jump(4));
    CHECK(p.blocks.at(6).insts.size() == p.blocks.at(2).insts.size());
}

void leaves_unknown_flags_and_divergent_forks_alone() {
    auto words = WORDS;
    words[5] = 0x7e020202;   // v1 = s2
    words[10] = 0x7e020203;  // v1 = s3
    Program p = program(words);
    CHECK(constant_edges(p, structure::structure(p)).empty());
    words = WORDS;
    words[6] = 0x7d840080;  // compare 0 with lane-dependent v0
    words[7] = 0xbf870003;  // branch on VCC
    p = program(words);
    CHECK(constant_edges(p, structure::structure(p)).empty());
}

}  // namespace

int main() {
    only_the_buffer_load_reaches_the_numeric_read();
    leaves_unknown_flags_and_divergent_forks_alone();
    return test_util::finish("threading_test");
}
