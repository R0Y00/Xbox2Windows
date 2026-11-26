// Unit tests of the vs-core modules of x2w_shader, ported from the #[cfg(test)] blocks of the upstream Rust shader crate,
// specifically dxbc.rs, dxil.rs, gfx10/{mod,opcodes,sp3}.rs and ir.rs (same names, same cases).
// `x2w_shader_core_tests <name>` runs one test, no argument runs all, `--list` lists them.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "x2w_shader/gfx10/mod.h"
#include "x2w_shader/gfx10/opcodes.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/lib.h"

using namespace x2w::vs;

namespace {

struct TestCase {
    const char* name;
    void (*fn)();
};

std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Register {
    Register(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

int failures = 0;

void fail(const char* file, int line, const std::string& what) {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, what.c_str());
    ++failures;
}

#define TEST(name)                               \
    void name();                                 \
    const Register register_##name{#name, name}; \
    void name()

#define CHECK(c)                                     \
    do {                                             \
        if (!(c)) fail(__FILE__, __LINE__, #c);      \
    } while (0)

#define CHECK_EQ(a, b)                                       \
    do {                                                     \
        if (!((a) == (b))) fail(__FILE__, __LINE__, #a " == " #b); \
    } while (0)

// A ShaderError of `expr`, or nullopt.
#define ERROR_OF(expr) catch_shader_error([&] { (void)(expr); })

std::vector<std::uint8_t> bytes(const std::vector<std::uint32_t>& words) {
    std::vector<std::uint8_t> out;
    for (std::uint32_t w : words) {
        for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(w >> (8 * i)));
    }
    return out;
}

dxbc::Bytes span(const std::vector<std::uint8_t>& v) { return dxbc::Bytes(v.data(), v.size()); }

// ---------------------------------------------------------------- dxbc.rs

TEST(dxbc_round_trips_parts) {
    std::vector<std::uint8_t> a = {1, 2, 3, 4}, h(20, 9);
    auto data = dxbc::build({{FourCC::SFI0, span(a)}, {FourCC::HASH, span(h)}});
    Container c = Container::parse(span(data));
    CHECK_EQ(c.parts.size(), 2u);
    CHECK(c.part(FourCC::SFI0) && std::vector<std::uint8_t>(c.part(FourCC::SFI0)->data.begin(),
                                                             c.part(FourCC::SFI0)->data.end()) == a);
    CHECK(c.part(FourCC::HASH) && c.part(FourCC::HASH)->data.size() == 20);
    CHECK(c.part(FourCC::DXIL) == nullptr);
}

TEST(dxbc_rejects_bad_input) {
    std::vector<std::uint8_t> nope = {'n', 'o', 'p', 'e'};
    CHECK(ERROR_OF(Container::parse(span(nope))) == ShaderError::not_dxbc());
    std::vector<std::uint8_t> z(8, 0);
    auto data = dxbc::build({{FourCC::SFI0, span(z)}});
    std::size_t last = data.size() - 1;
    data[0x20] = 0xFF;  // corrupt the first part offset
    CHECK(ERROR_OF(Container::parse(span(data))).has_value());
    data.resize(last);
    CHECK(ERROR_OF(Container::parse(span(data))).has_value());
}

// ---------------------------------------------------------------- dxil.rs

const std::uint8_t MAGIC[4] = {0x42, 0x43, 0xC0, 0xDE};

std::vector<std::uint8_t> fake_bitcode() {
    std::vector<std::uint8_t> b(MAGIC, MAGIC + 4);
    b.insert(b.end(), 12, 0xAA);
    return b;
}

TEST(dxil_parses_standard_layout) {
    auto bc = fake_bitcode();
    auto data = dxil::build_standard(0, {6, 0}, 0x100, span(bc));
    DxilProgram p = DxilProgram::parse(span(data));
    CHECK(p.kind == ShaderKind::Pixel);
    CHECK(p.shader_model == std::make_pair(std::uint8_t(6), std::uint8_t(0)));
    CHECK(p.is_standard_layout());
    CHECK(p.xbox_extras() == XboxExtras{});
}

TEST(dxil_detects_xbox_layout) {
    // Xbox layout: XDAT table before the bitcode, xbop blob after it.
    auto bitcode = fake_bitcode();
    const std::uint8_t xdat[] = {'X', 'D', 'A', 'T', 1, 0, 0, 0};
    const std::uint8_t trailer[] = {'x', 'b', 'o', 'p', 2, 0, 0, 0};
    std::vector<std::uint8_t> data;
    auto put32 = [&](std::uint32_t v) {
        for (int i = 0; i < 4; ++i) data.push_back(std::uint8_t(v >> (8 * i)));
    };
    std::size_t size = 24 + sizeof xdat + bitcode.size() + sizeof trailer;
    put32((5u << 16) | 0x65);  // cs_6_5
    put32(std::uint32_t(size / 4));
    data.insert(data.end(), {'D', 'X', 'I', 'L'});
    put32(0x105);
    put32(16 + sizeof xdat);
    put32(std::uint32_t(bitcode.size()));
    data.insert(data.end(), std::begin(xdat), std::end(xdat));
    data.insert(data.end(), bitcode.begin(), bitcode.end());
    data.insert(data.end(), std::begin(trailer), std::end(trailer));

    DxilProgram p = DxilProgram::parse(span(data));
    CHECK(p.kind == ShaderKind::Compute);
    CHECK(p.shader_model == std::make_pair(std::uint8_t(6), std::uint8_t(5)));
    CHECK(!p.is_standard_layout());
    CHECK((p.xbox_extras() == XboxExtras{true, true}));
    CHECK(std::vector<std::uint8_t>(p.bitcode.begin(), p.bitcode.end()) == bitcode);
}

TEST(dxil_rejects_missing_bitcode_magic) {
    std::vector<std::uint8_t> z(16, 0);
    auto data = dxil::build_standard(0, {6, 0}, 0x100, span(z));
    CHECK(ERROR_OF(DxilProgram::parse(span(data))).has_value());
}

// ---------------------------------------------------------------- gfx10/mod.rs

constexpr std::uint32_t S_ENDPGM = 0xBF81'0000;

// s_branch / s_cbranch_scc0 with a dword offset relative to the next instruction.
constexpr std::uint32_t sopp(std::uint32_t op, std::int16_t simm16) {
    return 0xBF80'0000u | op << 16 | std::uint16_t(simm16);
}

struct Summary {
    std::size_t offset;
    std::size_t words;
    std::string name;
    bool operator==(const Summary&) const = default;
};

std::vector<Summary> summary(const std::vector<std::uint32_t>& code) {
    std::vector<Summary> out;
    for (const auto& i : gfx10::decode(bytes(code))) {
        out.push_back({i.offset, i.words.size(), std::string(i.name().value_or("?"))});
    }
    return out;
}

TEST(gfx10_flat_offsets_ignore_bit_eleven) {
    auto offset = [](std::uint32_t word0) -> int {
        auto code = bytes({word0, 0x0000'0001, S_ENDPGM});
        auto fields = gfx10::decode(code)[0].fields(false);
        const auto* f = std::get_if<gfx10::Flat>(&fields);
        if (!f) {
            fail(__FILE__, __LINE__, "not FLAT fields");
            return -1;
        }
        return f->offset;
    };
    // flat_load_dword v0, v[1:2]: the FLAT offset is 11-bit and unsigned, so bit 11 is ignored
    // (0x800 -> 0, 0xfff -> 0x7ff).
    CHECK_EQ(offset(0xDC30'0800), 0);
    CHECK_EQ(offset(0xDC30'0FFF), 0x7FF);
}

TEST(gfx10_tables_are_sorted) {
    namespace o = gfx10::opcodes;
    for (const o::Table& t : {o::SOP2, o::SOPK, o::SOP1, o::SOPC, o::SOPP, o::SMEM, o::VOP2, o::VOP1, o::VOPC,
                              o::VOP3, o::VOP3P, o::VINTRP, o::DS, o::MUBUF, o::MTBUF, o::MIMG, o::FLAT, o::GLOBAL,
                              o::SCRATCH}) {
        CHECK(!t.empty());
        for (std::size_t i = 0; i + 1 < t.size(); ++i) CHECK(t[i].op < t[i + 1].op);
    }
}

TEST(gfx10_lengths_and_names) {
    std::vector<std::uint32_t> code = {
        0xBE80'03FF, 0x3F80'0000,               // s_mov_b32 s0, literal
        0x0600'0000,                            // v_add_f32 v0, s0, v0
        0xD54B'0000, 0x03FE'0000, 0x4000'0000,  // v_fma_f32 with a literal src2
        0xF080'0F0A, 0, 0,                      // image_sample (2D) with one extra address dword
        0xF800'080F, 0,                         // exp
        S_ENDPGM,
    };
    std::vector<Summary> want = {
        {0, 2, "s_mov_b32"}, {8, 1, "v_add_f32"}, {12, 3, "v_fma_f32"},
        {24, 3, "image_sample"}, {36, 2, "exp"}, {44, 1, "s_endpgm"},
    };
    CHECK(summary(code) == want);
}

TEST(gfx10_follows_branches_and_skips_data) {
    std::vector<std::uint32_t> code = {
        sopp(0x04, 1),  // s_cbranch_scc0 -> 8
        sopp(0x02, 1),  // s_branch -> 12
        S_ENDPGM, S_ENDPGM,
        S_ENDPGM,  // data follows the program end
        0xFFFF'FFFF,
    };
    std::vector<std::size_t> found;
    for (const auto& s : summary(code)) found.push_back(s.offset);
    CHECK((found == std::vector<std::size_t>{0, 4, 8, 12}));
}

TEST(gfx10_follows_far_jumps) {
    std::vector<std::uint32_t> code = {
        0xBE88'1F00,              // s_getpc_b64 s[8:9]
        0x8008'FF08, 20,          // s_add_u32 s8, s8, 20  -> 4 + 20 = 24
        0x8209'8009,              // s_addc_u32 s9, s9, 0
        0xBE80'2008,              // s_setpc_b64 s[8:9]
        0xFFFF'FFFF,              // not code
        S_ENDPGM,
    };
    std::vector<Summary> got = summary(code);
    std::vector<std::pair<std::size_t, std::string>> found;
    for (const auto& s : got) found.push_back({s.offset, s.name});
    std::vector<std::pair<std::size_t, std::string>> want = {
        {0, "s_getpc_b64"}, {4, "s_add_u32"}, {12, "s_addc_u32"}, {16, "s_setpc_b64"}, {24, "s_endpgm"},
    };
    CHECK(found == want);
    // Through other registers (a handover to another stage): the path ends.
    CHECK_EQ(summary({0xBE80'2018, S_ENDPGM}).size(), 1u);
}

TEST(gfx10_follows_far_jumps_backward) {
    // The compiler's backward far jump adds a negative offset: the low dword as a 32-bit literal,
    // the high dword as -1 with the carry.
    std::vector<std::uint32_t> code = {
        sopp(0x02, 1),            // s_branch -> 8, past the jump target
        S_ENDPGM,                 // the far jump's target: only it reaches here
        0xBE88'1F00,              // s_getpc_b64 s[8:9]
        0x8008'FF08, 0xFFFF'FFF8, // s_add_u32 s8, s8, -8 -> 12 - 8 = 4
        0x8209'C109,              // s_addc_u32 s9, s9, -1
        0xBE80'2008,              // s_setpc_b64 s[8:9]
    };
    std::vector<std::pair<std::size_t, std::string>> found;
    for (const auto& s : summary(code)) found.push_back({s.offset, s.name});
    std::vector<std::pair<std::size_t, std::string>> want = {
        {0, "s_branch"}, {4, "s_endpgm"}, {8, "s_getpc_b64"}, {12, "s_add_u32"}, {20, "s_addc_u32"}, {24, "s_setpc_b64"},
    };
    CHECK(found == want);
}

TEST(gfx10_rejects_bad_code) {
    std::vector<std::uint8_t> three(3, 0);
    CHECK(ERROR_OF(gfx10::decode(three)).has_value());                            // size not a multiple of 4
    CHECK(ERROR_OF(gfx10::decode(bytes({sopp(0x02, 100)}))).has_value());         // branch out of range
    CHECK(ERROR_OF(gfx10::decode(bytes({0xBE80'03FF}))).has_value());             // literal past the end
    CHECK(ERROR_OF(gfx10::decode(bytes({0xE400'0000, 0}))).has_value());          // unknown encoding
    // s_cbranch_scc0 into the literal of the s_mov_b32 at 4.
    CHECK(ERROR_OF(gfx10::decode(bytes({sopp(0x04, 1), 0xBE80'03FF, S_ENDPGM, S_ENDPGM}))).has_value());
}

// ---------------------------------------------------------------- gfx10/opcodes.rs

// `s_ttracedata_imm 0x1234` as LLVM's GFX10 disassembler tests have it (`gfx10_sopp.txt`:
// `0x34,0x12,0xa8,0xbf`), SOPP opcode 0x28 as in AMD's RDNA2 ISA. A thread-trace marker, which
// translation drops.
TEST(opcodes_s_ttracedata_imm_is_sopp_0x28) {
    auto code = bytes({0xBFA8'1234, 0xBF81'0000});
    auto decoded = gfx10::decode(code);
    CHECK(decoded[0].name() == "s_ttracedata_imm");
    CHECK(!ERROR_OF(ir::build(code, false, 1)).has_value());
}

// ---------------------------------------------------------------- gfx10/sp3.rs

// Text of each instruction in `words` (a straight-line program).
std::vector<std::string> texts(std::vector<std::uint32_t> words, bool wave64) {
    words.push_back(0xBF81'0000);  // s_endpgm
    auto insts = gfx10::decode(bytes(words));
    std::vector<std::string> out;
    for (std::size_t i = 0; i + 1 < insts.size(); ++i) out.push_back(insts[i].text(wave64));
    return out;
}

std::string one(const std::vector<std::uint32_t>& words, bool wave64) {
    auto t = texts(words, wave64);
    if (t.size() != 1) {
        fail(__FILE__, __LINE__, "expected one instruction");
        return {};
    }
    return t[0];
}

#define CHECK_TEXT(got, want)                                                                    \
    do {                                                                                         \
        std::string g_ = (got);                                                                  \
        if (g_ != (want)) fail(__FILE__, __LINE__, "got \"" + g_ + "\" want \"" + (want) + "\""); \
    } while (0)

TEST(sp3_scalar) {
    auto t = texts({0xBF88'0001, 0xBF8C'0070, 0xBE80'03FF, 0x1234'5678}, true);
    CHECK_EQ(t.size(), 3u);
    if (t.size() == 3) {
        CHECK_TEXT(t[0], std::string("s_cbranch_execz label_0008"));
        CHECK_TEXT(t[1], std::string("s_waitcnt vmcnt(0) & lgkmcnt(0)"));
        CHECK_TEXT(t[2], std::string("s_mov_b32 s0, lit(0x12345678)"));
    }
    CHECK_TEXT(one({0x8782'027E}, true), std::string("s_and_b64 s[2:3], exec, s[2:3]"));
    CHECK_TEXT(one({0xF408'0501, 0xFA00'0040}, true), std::string("s_load_dwordx4 s[20:23], s[2:3], 0x000040"));
    CHECK_TEXT(one({0xF420'0144, 0x0400'0000}, true), std::string("s_buffer_load_dword s5, s[8:11], s2"));
}

// `s_atc_probe` / `s_atc_probe_buffer`: SDATA is a 7-bit immediate, the buffer resource is 4
// dwords, and GLC/DLC are not operands. Words checked with llvm-mc -mcpu=gfx1030.
TEST(sp3_atc_probe_immediate_is_not_a_register) {
    struct Case {
        std::vector<std::uint32_t> words;
        std::string text;
        gfx10::Operand data, base;
    };
    using gfx10::Operand;
    std::vector<Case> cases = {
        {{0xF498'01C2, 0xFA00'0064}, "s_atc_probe 7, s[4:5], 0x000064", Operand::Literal(7), Operand::Scalar(4, 2)},
        {{0xF498'1FF4, 0x0A00'0001}, "s_atc_probe 0x7f, s[104:105], s5 offset:0x000001", Operand::Literal(0x7f),
         Operand::Scalar(104, 2)},
        {{0xF498'0000, 0xFA00'0000}, "s_atc_probe 0, s[0:1], null", Operand::Literal(0), Operand::Scalar(0, 2)},
        {{0xF49C'01C4, 0xFA00'0020}, "s_atc_probe_buffer 7, s[8:11], 0x000020", Operand::Literal(7),
         Operand::Scalar(8, 4)},
        {{0xF499'41C2, 0xFA00'0000}, "s_atc_probe 7, s[4:5], null", Operand::Literal(7), Operand::Scalar(4, 2)},
    };
    for (const Case& c : cases) {
        CHECK_TEXT(one(c.words, false), c.text);
        CHECK_TEXT(one(c.words, true), c.text);
        auto w = c.words;
        w.push_back(0xBF81'0000);
        auto fields = gfx10::decode(bytes(w))[0].fields(false);
        const auto* m = std::get_if<gfx10::Smem>(&fields);
        CHECK(m != nullptr);
        if (m) {
            CHECK(m->data == c.data);
            CHECK(m->base == c.base);
            CHECK(m->cache == gfx10::Cache{});
        }
    }
}

TEST(sp3_lane_masks_follow_the_wave_size) {
    std::vector<std::uint32_t> cmp = {0x7C08'06F0};  // v_cmp_gt_f32 with 0.5 and v3
    CHECK_TEXT(one(cmp, false), std::string("v_cmp_gt_f32 vcc_lo, 0.5, v3"));
    CHECK_TEXT(one(cmp, true), std::string("v_cmp_gt_f32 vcc, 0.5, v3"));
    std::vector<std::uint32_t> vop3 = {0xD401'0106, 0x0001'E501};  // v_cmp_lt_f32 to s6, |v1| < 1.0
    CHECK_TEXT(one(vop3, false), std::string("v_cmp_lt_f32 s6, abs(v1), 1.0"));
    CHECK_TEXT(one(vop3, true), std::string("v_cmp_lt_f32 s[6:7], abs(v1), 1.0"));
    CHECK_TEXT(one({0x0202'0480}, false), std::string("v_cndmask_b32 v1, 0, v2, vcc_lo"));
    CHECK_TEXT(one({0x5006'0801}, true), std::string("v_add_co_ci_u32 v3, vcc, s1, v4, vcc"));
}

TEST(sp3_vector_modifiers) {
    CHECK_TEXT(one({0x1004'0AFF, 0x3F00'0001}, true), std::string("v_mul_f32 v2, lit(0x3f000001), v5"));
    CHECK_TEXT(one({0xD54B'8004, 0x2BC0'0501}, true), std::string("v_fma_f32 v4, -v1, s2, 0.5 mul:2 clamp"));
    CHECK_TEXT(one({0x7E02'16F9, 0x0605'0602}, true), std::string("v_cvt_f32_f16 v1, v2 src0_sel:WORD_1"));
    CHECK_TEXT(one({0x7E06'02E9, 0x0539'7701}, true), std::string("v_mov_b32 v3, v1 dpp8:[7,6,5,4,3,2,1,0]"));
    CHECK_TEXT(one({0x7E0A'02FA, 0xFF08'B106}, true),
               std::string("v_mov_b32 v5, v6 quad_perm:[1,0,3,2] bound_ctrl:1"));
    CHECK_TEXT(one({0xCC20'0C01, 0x2C10'0702}, true), std::string("v_fma_mix_f32 v1, -sel_hi(v2), s3, abs(v4)"));
}

TEST(sp3_memory_and_export) {
    CHECK_TEXT(one({0xE034'7010, 0x0102'0402}, true),
               std::string("buffer_load_dwordx2 v[4:5], v[2:3], s[8:11], s1 idxen offen offset:16 glc"));
    CHECK_TEXT(one({0xD8DC'0302, 0x0600'0001}, true), std::string("ds_read2_b32 v[6:7], v1 offset0:2 offset1:3"));
    CHECK_TEXT(one({0xF800'1C0F, 0x0000'0302}, true), std::string("exp mrt0, v2, v2, v3, v3 done compr vm"));
}

TEST(sp3_one_sided_packed_negation_is_printed) {
    // Encodings from llvm-mc -mcpu=gfx1030. A source negated in both halves keeps the `-` prefix;
    // one-sided bits get LLVM's suffixes.
    CHECK_TEXT(one({0xCC0F'4001, 0x1A02'0702}, true), std::string("v_pk_add_f16 v1, v2, v3"));
    CHECK_TEXT(one({0xCC0F'4001, 0x3A02'0702}, true), std::string("v_pk_add_f16 v1, v2, v3 neg_lo:[1,0]"));
    CHECK_TEXT(one({0xCC0F'4201, 0x1A02'0702}, true), std::string("v_pk_add_f16 v1, v2, v3 neg_hi:[0,1]"));
    CHECK_TEXT(one({0xCC0F'4101, 0x7A02'0702}, true), std::string("v_pk_add_f16 v1, -v2, v3 neg_lo:[0,1]"));
}

TEST(sp3_fields_the_instruction_lacks_are_not_printed) {
    // Encodings from llvm-mc -mcpu=gfx1030; s_get_waveid_in_workgroup is gfx1010 only.
    CHECK_TEXT(one({0xF490'0500, 0}, true), std::string("s_memtime s[20:21]"));
    CHECK_TEXT(one({0xF4A8'0140, 0}, true), std::string("s_get_waveid_in_workgroup s5"));
    CHECK_TEXT(one({0xDC30'8000, 0x007D'0002}, true), std::string("global_load_dword v0, v[2:3], off"));
    CHECK_TEXT(one({0xE031'1000, 0x0001'0001}, true), std::string("buffer_load_dword v1, s[4:7], s0 offen lds"));
    CHECK_TEXT(one({0x7E02'02F9, 0x0005'0602}, true), std::string("v_mov_b32 v1, v2 src0_sel:WORD_1"));
}

TEST(sp3_image_d16_data_is_packed) {
    // Three f16 channels occupy two VGPRs; NSA lists the addresses.
    std::vector<std::uint32_t> words = {0xF09C'070A, 0x8024'0801, 0x0000'0005};
    CHECK_TEXT(one(words, true),
               std::string("image_sample_lz v[8:9], [v1,v5], s[16:23], s[4:7] dmask:0x7 dim:SQ_RSRC_IMG_2D d16"));
    auto w = words;
    w.push_back(0xBF81'0000);
    auto fields = gfx10::decode(bytes(w))[0].fields(true);
    const auto* im = std::get_if<gfx10::Image>(&fields);
    CHECK(im != nullptr);
    if (im) {
        CHECK(im->data == gfx10::Operand::Vector(8, 2));
        CHECK_EQ(im->addr.size(), 2u);
    }
}

// ---------------------------------------------------------------- ir.rs

using ir::Block;
using ir::Exit;
using ir::Inst;
using ir::Phi;
using ir::Program;
using ir::Reg;
using ir::Value;

ir::Program build_words(const std::vector<std::uint32_t>& words, bool wave64) {
    return ir::build(bytes(words), wave64, 256);
}

Value input(const Program& p, Reg reg) {
    for (const auto& i : p.inputs) {
        if (i.first == reg) return i.second;
    }
    fail(__FILE__, __LINE__, "input " + reg.debug());
    return Value{0xFFFF'FFFF};
}

// `if (v0 < s0) { ... }` whose body changes s6: `body` are its words, `after` what follows the join.
Program skip(const std::vector<std::uint32_t>& body, const std::vector<std::uint32_t>& after) {
    std::vector<std::uint32_t> w = {
        0xBE94'0306,                                // s_mov_b32 s20, s6
        0x7D88'0000,                                // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C'246A,                                // s_and_saveexec_b64 s[12:13], vcc
        0xBF88'0000u | std::uint32_t(body.size()),  // s_cbranch_execz past the body
    };
    w.insert(w.end(), body.begin(), body.end());
    w.push_back(0xBEFE'040C);  // s_mov_b64 exec, s[12:13]
    w.insert(w.end(), after.begin(), after.end());
    w.push_back(S_ENDPGM);
    return build_words(w, true);
}

std::size_t skips(const Program& p) {
    std::size_t n = 0;
    for (const Block& b : p.blocks) {
        if (!b.insts.empty() && b.insts.back().name() == "s_cbranch_execz") ++n;
    }
    return n;
}

TEST(ir_a_skip_is_left_out_when_the_scalars_it_changes_come_back_or_are_not_read) {
    const std::uint32_t SET = 0xBE86'0385;   // s_mov_b32 s6, 5
    const std::uint32_t USE = 0x7E02'0206;   // v_mov_b32 v1, s6
    const std::uint32_t BACK = 0xBE86'0314;  // s_mov_b32 s6, s20
    const std::uint32_t READ = 0x7E04'0206;  // v_mov_b32 v2, s6
    // s6 is changed in the body and read after the join: the branch stays
    Program kept = skip({SET, USE}, {READ});
    CHECK_EQ(skips(kept), 1u);
    CHECK(!ir::without_skips(kept, false));
    CHECK(!ir::without_skips(kept, true));
    // ... put back by a copy of what it held (as a scalar parked in a VGPR lane comes back): every
    // lane may walk the body
    Program back = skip({SET, USE, BACK}, {READ});
    CHECK(!ir::without_skips(back, false));
    auto b = ir::without_skips(back, true);
    CHECK(b && skips(*b) == 0);
    // ... or not read after the join at all
    Program unread = skip({SET, USE}, {});
    auto u0 = ir::without_skips(unread, false);
    CHECK(u0 && skips(*u0) == 0);
    auto u1 = ir::without_skips(unread, true);
    CHECK(u1 && skips(*u1) == 0);
    // ... or read only by a restore of EXEC that nothing uses before the next one
    // (s_mov_b64 exec, s[6:7]; s_mov_b64 exec, s[12:13]; v_mov_b32 v2, s0)
    Program idle = skip({SET, USE}, {0xBEFE'0406, 0xBEFE'040C, 0x7E04'0200});
    CHECK(!ir::without_skips(idle, false));
    auto i1 = ir::without_skips(idle, true);
    CHECK(i1 && skips(*i1) == 0);
}

TEST(ir_joins_get_phis_only_where_values_differ) {
    std::vector<std::uint32_t> code = {
        0xBF06'0100,  // s_cmp_eq_u32 s0, s1
        0xBF85'0001,  // s_cbranch_scc1 -> 12
        0xBE82'0380,  // s_mov_b32 s2, 0
        0x7E00'0202,  // 12: v_mov_b32 v0, s2
        0x7E02'0200,  // v_mov_b32 v1, s0
        S_ENDPGM,
    };
    for (bool wave64 : {false, true}) {
        Program p = build_words(code, wave64);
        CHECK_EQ(p.blocks.size(), 3u);
        if (p.blocks.size() != 3) continue;
        const Block &b0 = p.blocks[0], &b1 = p.blocks[1], &b2 = p.blocks[2];
        CHECK(b0.exit == Exit::Branch(2, 1));
        CHECK(b1.exit == Exit::Jump(2));
        CHECK((b2.preds == std::vector<std::size_t>{0, 1}));
        CHECK(b0.insts[1].read(Reg::Scc) == b0.insts[0].write(Reg::Scc));

        // s2 differs per path; s0, EXEC, v0 and v1 do not.
        CHECK_EQ(b2.phis.size(), 1u);
        if (b2.phis.size() != 1) continue;
        const Phi& phi = b2.phis[0];
        CHECK(phi.reg == Reg::S(2));
        CHECK((phi.args == std::vector<Value>{input(p, Reg::S(2)), *b1.insts[0].write(Reg::S(2))}));
        CHECK_EQ(b2.insts.size(), 3u);
        if (b2.insts.size() != 3) continue;
        const Inst &mov0 = b2.insts[0], &mov1 = b2.insts[1], &end = b2.insts[2];
        CHECK(mov0.read(Reg::S(2)) == phi.value);
        CHECK(mov1.read(Reg::S(0)) == input(p, Reg::S(0)));
        // The old v1 is read: lanes outside EXEC keep it.
        CHECK(mov1.read(Reg::V(1)) == input(p, Reg::V(1)));
        CHECK(mov1.read(Reg::S(ir::EXEC)) == input(p, Reg::S(ir::EXEC)));
        CHECK_EQ(mov1.read(Reg::S(ir::EXEC + 1)).has_value(), wave64);
        CHECK(end.name() == "s_endpgm");
        CHECK(b2.exit == Exit::End);
    }
}

TEST(ir_in_place_scalar_ops_read_their_destination) {
    Program p = build_words(
        {
            0xBE8D'1D94,  // s_bitset1_b32 s13, 20
            0xB78E'0010,  // s_addk_i32 s14, 0x10
            0xBE8F'0380,  // s_mov_b32 s15, 0
            S_ENDPGM,
        },
        false);
    const auto& insts = p.blocks[0].insts;
    CHECK_EQ(insts.size(), 4u);
    if (insts.size() != 4) return;
    CHECK(insts[0].read(Reg::S(13)) == input(p, Reg::S(13)));
    CHECK(insts[1].read(Reg::S(14)) == input(p, Reg::S(14)));
    CHECK(insts[2].read(Reg::S(15)) == std::nullopt);
}

TEST(ir_loops_carry_values_through_phis) {
    Program p = build_words(
        {
            0xBE80'0380,  // s_mov_b32 s0, 0
            0x8000'8100,  // 4: s_add_u32 s0, s0, 1
            0xBF07'8400,  // s_cmp_lg_u32 s0, 4
            0xBF85'FFFD,  // s_cbranch_scc1 -> 4
            S_ENDPGM,
        },
        true);
    CHECK(p.inputs.empty());
    CHECK_EQ(p.blocks.size(), 3u);
    if (p.blocks.size() != 3) return;
    const Block &b0 = p.blocks[0], &b1 = p.blocks[1], &b2 = p.blocks[2];
    CHECK((b1.preds == std::vector<std::size_t>{0, 1}));
    CHECK(b1.exit == Exit::Branch(1, 2));
    CHECK_EQ(b1.phis.size(), 1u);
    if (b1.phis.size() != 1) return;
    const Phi& phi = b1.phis[0];
    const Inst& add = b1.insts[0];
    CHECK(phi.reg == Reg::S(0));
    CHECK((phi.args == std::vector<Value>{*b0.insts[0].write(Reg::S(0)), *add.write(Reg::S(0))}));
    CHECK(add.read(Reg::S(0)) == phi.value);
    CHECK(b1.insts[1].read(Reg::S(0)) == add.write(Reg::S(0)));
    CHECK(b2.insts[0].name() == "s_endpgm");
}

TEST(ir_drops_scheduling_and_follows_far_jumps) {
    Program p = build_words(
        {
            0xBF80'0000,      // s_nop 0
            0xBE88'1F00,      // s_getpc_b64 s[8:9]
            0x8008'FF08, 20,  // s_add_u32 s8, s8, 20  -> 8 + 20 = 28
            0x8209'8009,      // s_addc_u32 s9, s9, 0
            0xBE80'2008,      // s_setpc_b64 s[8:9]
            0xFFFF'FFFF,      // not code
            S_ENDPGM,
        },
        false);
    std::vector<std::string> names;
    for (const Inst& i : p.blocks[0].insts) names.push_back(std::string(i.name()));
    CHECK((names == std::vector<std::string>{"s_getpc_b64", "s_add_u32", "s_addc_u32", "s_setpc_b64"}));
    CHECK(p.blocks[0].exit == Exit::Jump(1));
    CHECK_EQ(p.blocks[1].offset, 28u);
    // s_addc_u32 reads the carry s_add_u32 left in SCC.
    const auto& i = p.blocks[0].insts;
    CHECK(i[2].read(Reg::Scc) == i[1].write(Reg::Scc));
}

TEST(ir_reports_what_it_cannot_express) {
    auto code = bytes({0xBE80'2E01, S_ENDPGM});  // s_movrels_b32 s0, s1
    auto e = ERROR_OF(ir::build(code, false, 256));
    CHECK(e == ShaderError::unsupported(0, "s_movrels_b32"));
    if (e) CHECK_TEXT(e->to_string(), std::string("GFX10 code at 0x0: s_movrels_b32 is not supported"));
}

TEST(ir_relative_vgpr_access_spans_up_to_the_vgpr_count) {
    auto code = bytes({
        0x7E04'8501,  // v_movreld_b32 v2, v1: v[2 + m0] = v1
        0x7E00'8704,  // v_movrels_b32 v0, v4: v0 = v[4 + m0]
        S_ENDPGM,
    });
    Program p = ir::build(code, false, 6);
    CHECK_EQ(p.blocks[0].insts.size(), 3u);
    if (p.blocks[0].insts.size() != 3) return;
    const Inst &movreld = p.blocks[0].insts[0], &movrels = p.blocks[0].insts[1];
    auto vgprs = [](const std::vector<std::pair<Reg, Value>>& regs) {
        std::vector<int> out;
        for (const auto& r : regs) {
            if (r.first.is_v()) out.push_back(r.first.n);
        }
        return out;
    };
    CHECK((vgprs(movreld.writes) == std::vector<int>{2, 3, 4, 5}));
    // v1 is the source; v2-v5 are read for the lanes and indices not written.
    CHECK((vgprs(movreld.reads) == std::vector<int>{1, 2, 3, 4, 5}));
    CHECK(movreld.read(Reg::S(ir::M0)).has_value());
    CHECK((vgprs(movrels.writes) == std::vector<int>{0}));
    CHECK((vgprs(movrels.reads) == std::vector<int>{4, 5, 0}));
    CHECK(movrels.read(Reg::V(4)) == movreld.write(Reg::V(4)));
    CHECK(ERROR_OF(ir::build(code, false, 4)).has_value());  // v4 is past 4 VGPRs
}

constexpr std::uint32_t s_branch(std::int16_t simm16) { return 0xBF82'0000u | std::uint16_t(simm16); }
constexpr std::uint32_t s_cbranch_scc1(std::int16_t simm16) { return 0xBF85'0000u | std::uint16_t(simm16); }

TEST(ir_reverse_order_diamond_chain_eliminates_trivial_phis) {
    const std::size_t LAYERS = 4096;
    std::vector<std::uint32_t> reverse = {
        0xBE81'0381,  // s_mov_b32 s1, 1
        0xBE82'0382,  // s_mov_b32 s2, 2
        s_branch(std::int16_t((LAYERS - 1) * 4)),
    };
    for (std::size_t k = 0; k < LAYERS; ++k) {
        reverse.insert(reverse.end(), {
                                          0xBF06'0201,  // s_cmp_eq_u32 s1, s2
                                          s_cbranch_scc1(1),
                                          0xBF80'0000,  // s_nop 0
                                          k == 0 ? s_branch(std::int16_t(LAYERS * 4 - 4)) : s_branch(-8),
                                      });
    }
    reverse.insert(reverse.end(), {0x8000'0201, S_ENDPGM});  // s_add_u32 s0, s1, s2

    std::vector<std::uint32_t> forward = {
        0xBE81'0381,  // s_mov_b32 s1, 1
        0xBE82'0382,  // s_mov_b32 s2, 2
        s_branch(0),
    };
    for (std::size_t k = 0; k < LAYERS; ++k) {
        forward.insert(forward.end(), {0xBF06'0201, s_cbranch_scc1(1), 0xBF80'0000, s_branch(0)});
    }
    forward.insert(forward.end(), {0x8000'0201, S_ENDPGM});

    Program rev = build_words(reverse, false);
    Program fwd = build_words(forward, false);
    CHECK(rev.inputs.empty());
    CHECK(fwd.inputs.empty());
    for (const Block& b : rev.blocks) CHECK(b.phis.empty());
    for (const Block& b : fwd.blocks) CHECK(b.phis.empty());
    CHECK_EQ(rev.blocks.size(), fwd.blocks.size());

    for (const Program* p : {&rev, &fwd}) {
        Value s1 = *p->blocks[0].insts[0].write(Reg::S(1));
        Value s2 = *p->blocks[0].insts[1].write(Reg::S(2));
        const Block& exit = p->blocks.back();
        CHECK(exit.insts[0].name() == "s_add_u32");
        CHECK(exit.insts[0].read(Reg::S(1)) == s1);
        CHECK(exit.insts[0].read(Reg::S(2)) == s2);
    }
}

TEST(ir_remove_trivial_phis_compresses_reverse_chain_in_bounded_steps) {
    const std::size_t CHAIN = 4096;
    Inst tmpl = build_words({0xBE80'0380, S_ENDPGM}, false).blocks[0].insts[0];
    tmpl.writes.clear();
    std::vector<Block> blocks;
    blocks.reserve(CHAIN);
    for (std::size_t k = 0; k < CHAIN; ++k) {
        Value v1{std::uint32_t(2 + 2 * k)}, v2{std::uint32_t(3 + 2 * k)};
        Value n1{0}, n2{1};
        if (k + 1 < CHAIN) {
            n1 = Value{std::uint32_t(2 + 2 * (k + 1))};
            n2 = Value{std::uint32_t(3 + 2 * (k + 1))};
        }
        Inst inst = tmpl;
        inst.reads = {{Reg::S(1), v1}, {Reg::S(2), v2}};
        Block b;
        b.offset = k * 4;
        b.phis = {Phi{Reg::S(1), v1, {n1, n1}}, Phi{Reg::S(2), v2, {n2, n2}}};
        b.insts = {inst};
        b.exit = Exit::End;
        blocks.push_back(std::move(b));
    }
    std::size_t steps = ir::detail::remove_trivial_phis_steps(blocks, std::uint32_t(2 + 2 * CHAIN));
    std::vector<std::pair<Reg, Value>> want = {{Reg::S(1), Value{0}}, {Reg::S(2), Value{1}}};
    bool all_empty = true, all_reads = true;
    for (const Block& b : blocks) {
        all_empty = all_empty && b.phis.empty();
        all_reads = all_reads && b.insts[0].reads == want;
    }
    CHECK(all_empty);
    CHECK(all_reads);
    if (!(steps < 100'000)) fail(__FILE__, __LINE__, "steps = " + std::to_string(steps));
}

TEST(ir_non_trivial_phis_preserve_compressed_representatives) {
    const std::size_t LAYERS = 4;
    std::vector<std::uint32_t> code = {
        0xBE81'0381,  // s_mov_b32 s1, 1
        0xBE82'0382,  // s_mov_b32 s2, 2
        0xBF06'8000,  // s_cmp_eq_u32 s0, 0
        s_cbranch_scc1(std::int16_t(2 + (LAYERS - 1) * 4)),
        0xBE81'0383,  // s_mov_b32 s1, 3
        s_branch(std::int16_t(LAYERS * 4)),
    };
    for (std::size_t k = 0; k < LAYERS; ++k) {
        code.insert(code.end(), {0xBF06'0201, s_cbranch_scc1(1), 0xBF80'0000,
                                 k == 0 ? s_branch(std::int16_t(LAYERS * 4 - 4)) : s_branch(-8)});
    }
    code.insert(code.end(), {0x8000'0201, S_ENDPGM});  // s_add_u32 s0, s1, s2

    Program p = build_words(code, false);
    Value entry_s1 = *p.blocks[0].insts[0].write(Reg::S(1));
    Value entry_s2 = *p.blocks[0].insts[1].write(Reg::S(2));
    Value arm_s1 = *p.blocks[1].insts[0].write(Reg::S(1));
    const Block& merge = p.blocks.back();
    for (std::size_t b = 0; b + 1 < p.blocks.size(); ++b) CHECK(p.blocks[b].phis.empty());
    CHECK_EQ(merge.phis.size(), 1u);
    if (merge.phis.size() == 1) {
        const Phi& phi = merge.phis[0];
        CHECK(phi.reg == Reg::S(1));
        CHECK((phi.args == std::vector<Value>{arm_s1, entry_s1}));
        CHECK(merge.insts[0].read(Reg::S(1)) == phi.value);
        CHECK(merge.insts[0].read(Reg::S(2)) == entry_s2);
    }

    std::vector<std::uint32_t> loop_code = {
        0xBE81'0381,  // s_mov_b32 s1, 1
        0xBE82'0382,  // s_mov_b32 s2, 2
        s_branch(std::int16_t((LAYERS - 1) * 4)),
    };
    for (std::size_t k = 0; k < LAYERS; ++k) {
        loop_code.insert(loop_code.end(), {0xBF06'0201, s_cbranch_scc1(1), 0xBF80'0000,
                                           k == 0 ? s_branch(std::int16_t(LAYERS * 4 - 4)) : s_branch(-8)});
    }
    loop_code.insert(loop_code.end(), {
                                          0x8001'0201,  // s_add_u32 s1, s1, s2
                                          0xBF07'8401,  // s_cmp_lg_u32 s1, 4
                                          s_cbranch_scc1(-3),
                                          0x8000'0201,  // s_add_u32 s0, s1, s2
                                          S_ENDPGM,
                                      });

    Program lp = build_words(loop_code, false);
    Value lp_s1 = *lp.blocks[0].insts[0].write(Reg::S(1));
    Value lp_s2 = *lp.blocks[0].insts[1].write(Reg::S(2));
    const Block& header = lp.blocks[lp.blocks.size() - 2];
    const Block& exit = lp.blocks.back();
    for (std::size_t b = 0; b + 2 < lp.blocks.size(); ++b) CHECK(lp.blocks[b].phis.empty());
    CHECK(exit.phis.empty());
    CHECK_EQ(header.phis.size(), 1u);
    if (header.phis.size() == 1) {
        const Phi& loop_phi = header.phis[0];
        Value next_s1 = *header.insts[0].write(Reg::S(1));
        CHECK(loop_phi.reg == Reg::S(1));
        CHECK((loop_phi.args == std::vector<Value>{lp_s1, next_s1}));
        CHECK(header.insts[0].read(Reg::S(1)) == loop_phi.value);
        CHECK(header.insts[0].read(Reg::S(2)) == lp_s2);
        CHECK(exit.insts[0].read(Reg::S(1)) == next_s1);
        CHECK(exit.insts[0].read(Reg::S(2)) == lp_s2);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--list") == 0) {
        for (const auto& t : registry()) std::printf("%s\n", t.name);
        return 0;
    }
    int ran = 0;
    for (const auto& t : registry()) {
        if (argc > 1 && std::strcmp(argv[1], t.name) != 0) continue;
        ++ran;
        int before = failures;
        try {
            t.fn();
        } catch (const std::exception& e) {
            fail(__FILE__, __LINE__, std::string(t.name) + " threw: " + e.what());
        }
        std::printf("%s %s\n", failures == before ? "PASS" : "FAIL", t.name);
    }
    if (ran == 0) {
        std::fprintf(stderr, "no test named %s\n", argc > 1 ? argv[1] : "(any)");
        return 2;
    }
    return failures == 0 ? 0 : 1;
}
