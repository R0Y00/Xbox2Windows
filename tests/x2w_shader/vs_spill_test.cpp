// Port of the upstream Rust shader tests in src/spill.rs.
#include "x2w_shader/spill.h"

#include "vs_front_test.h"

#include <algorithm>

using namespace x2w::vs;
using namespace x2w::vs::native;
using ir::Reg;

namespace {

RootParam param(RootType root, std::uint8_t user_data, std::uint32_t sgprs, std::uint8_t slot)
{
    RootParam r;
    r.root = root;
    r.table_offset_dwords = 0;
    r.sgprs = sgprs;
    r.base_register = 0;
    r.user_data = user_data;
    r.slot = slot;
    return r;
}

// A compute program with `registers` user SGPRs and the group id after them.
NativeProgram native_program(const std::vector<std::uint32_t>& words, std::uint32_t registers,
                             std::vector<RootParam> layout)
{
    NativeProgram n;
    n.stage = HwStage::Cs;
    n.wave64 = false;
    n.vgprs = 256;
    n.layout = std::move(layout);
    n.io = ComputeIo{(registers << 1) | (1u << 7), {1, 1, 1}};
    for (std::uint32_t w : words)
        for (int i = 0; i < 4; ++i) n.code.push_back(static_cast<std::uint8_t>(w >> (8 * i)));
    return n;
}

// s_and_b32 s3, s1, 0xff; s_mov_b32 s2, s0; s_load_dword s5, s[2:3], 0x10;
// s_load_dwordx2 s[0:1], s[2:3], 0; s_endpgm.
const std::vector<std::uint32_t> TABLE = {
    0x8703'FF01, 0x0000'00FF, 0xBE82'0300, 0xF400'0141, 0xFA00'0010, 0xF404'0001, 0xFA00'0000, 0xBF81'0000,
};

std::size_t loads(const ir::Program& p)
{
    std::size_t n = 0;
    for (const auto& b : p.blocks)
        for (const auto& i : b.insts)
            if (i.name().starts_with("s_load_")) ++n;
    return n;
}

}  // namespace

VS_TEST(loads_of_the_spilled_table_become_entry_registers)
{
    const NativeProgram n = native_program(TABLE, 4,
                                           {param(RootType::Descriptor(), 16, 2, 0), param(RootType::Descriptor(), 18, 2, 1),
                                            param(RootType::Constants(), 20, 16, 2)});
    ir::Program p = ir::build(n.code, false, 256);
    VS_CHECK(spill::user_data(p, n) == 2);
    VS_CHECK(loads(p) == 0);
    // The pointer is gone with the loads; byte 0x10 is index 20, the first root constant, and bytes
    // 0-7 are the first root view.
    std::vector<Reg> regs;
    for (const auto& i : p.inputs) regs.push_back(i.first);
    std::sort(regs.begin(), regs.end());
    VS_CHECK((regs == std::vector<Reg>{Reg::S(16), Reg::S(17), Reg::S(20)}));
}

VS_TEST(parameters_in_registers_are_left_alone)
{
    // The same code where s0 and s1 are a root view the layout declares.
    const NativeProgram n = native_program(TABLE, 6, {param(RootType::Descriptor(), 0, 2, 0)});
    ir::Program p = ir::build(n.code, false, 256);
    const ir::Program before = p;
    VS_CHECK(spill::user_data(p, n) == 0);
    VS_CHECK(p == before);
}

VS_TEST(a_used_dword_no_parameter_declares_keeps_its_load)
{
    // As above with `s_mov_b32 s6, s5` before the end: the dword at byte 0x10 (index 20) is read,
    // and here nothing declares it.
    std::vector<std::uint32_t> words = TABLE;
    words.insert(words.begin() + 7, 0xBE86'0305);
    const NativeProgram n = native_program(words, 4, {param(RootType::Descriptor(), 16, 2, 0)});
    ir::Program p = ir::build(n.code, false, 256);
    VS_CHECK(spill::user_data(p, n) == 1);
    VS_CHECK(loads(p) == 1);
    // The pointer is still read, by the load that stays.
    VS_CHECK(std::any_of(p.inputs.begin(), p.inputs.end(), [](const auto& i) { return i.first == Reg::S(0); }));
}

VS_TEST_MAIN()
