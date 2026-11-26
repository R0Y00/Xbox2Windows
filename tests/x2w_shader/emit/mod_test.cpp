// Port of the tests in the upstream Rust module emit/mod.rs (#[cfg(test)] mod tests, rs:956), in Rust order.
//
// Each Rust #[test] is a function here; main runs them all, each under its own dump name
// (lower::test::set_dump_name, "emit__tests__<name>", as lower_test.cpp names its own). A test
// that throws (a Rust panic or `.unwrap()` of an error) counts as failed and the others still run.
//
// Rust `.unwrap()` of a ShaderError is a panic, `?` a returned error: where a test looks at an
// error (is_err, unwrap_err), the steps Rust unwraps go through unwrap() below, which turns a
// ShaderError into std::logic_error so that it fails the test instead of passing as the error.
//
// The GPU tests (Rust #[cfg(windows)]) skip, and say so, without dxc.exe / dxil.dll / an SM 6.6
// device as the Rust ones do. The upstream valu_tests::ONE_AT_A_TIME lock is not needed: the tests of
// this executable run one after another on one thread.

#include <windows.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test_util.h"
#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/bitcode.h"
#include "x2w_shader/emit/container.h"
#include "x2w_shader/emit/emit.h"
#include "x2w_shader/emit/host.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/opt.h"
#include "x2w_shader/emit/sampler.h"
#include "x2w_shader/emit/settings.h"
#include "x2w_shader/emit/shader.h"
#include "x2w_shader/emit/validator.h"
#include "x2w_shader/error.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/structure.h"

namespace fs = std::filesystem;
namespace vs = x2w::vs;
using vs::HostShader;
using vs::ShaderError;
using vs::container::SigElement;
using vs::dxbc::Bytes;
using vs::dxbc::Container;
using vs::dxbc::FourCC;
using vs::native::ComputeIo;
using vs::native::HwStage;
using vs::native::MeshIo;
using vs::native::NativeProgram;
using vs::native::PsInput;
using vs::native::PsIo;
using vs::native::ResourceType;
using vs::native::RootParam;
using vs::native::RootType;
using vs::native::UsedResource;
using vs::native::Usage;
using vs::validator::Validator;

// CHECK with a message printed when it fails (Rust assert!'s format arguments).
#define CHECK_MSG(expr, msg)                                                               \
    do {                                                                                   \
        const bool check_ok_ = static_cast<bool>(expr);                                    \
        if (!check_ok_) std::fprintf(stderr, "  %s\n", std::string(msg).c_str());          \
        ::test_util::check(check_ok_, #expr, __FILE__, __LINE__);                          \
    } while (0)

namespace {

std::vector<std::uint8_t> le_bytes(std::span<const std::uint32_t> words) {
    std::vector<std::uint8_t> code;
    code.reserve(words.size() * 4);
    for (std::uint32_t w : words)
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(w >> (8 * k)));
    return code;
}

// Rust `.unwrap()` of a Result<_, ShaderError>: the error is a panic.
template <class F>
auto unwrap(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const ShaderError& e) {
        throw std::logic_error("called `Result::unwrap()` on an `Err` value: " + e.to_string());
    }
}

std::string hex(std::uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof b, "%#llx", static_cast<unsigned long long>(v));
    return b;
}

bool contains(std::string_view s, std::string_view what) { return s.find(what) != std::string_view::npos; }

// `c.part(fourcc).unwrap().data`.
Bytes part_data(const Container& c, FourCC fourcc) {
    const vs::dxbc::Part* p = c.part(fourcc);
    if (!p) throw std::logic_error("no " + fourcc.to_string() + " part");
    return p->data;
}

// `u64::from_le_bytes(data.try_into().unwrap())`.
std::uint64_t u64_exact(Bytes d) {
    if (d.size() != 8) throw std::logic_error("not 8 bytes: " + std::to_string(d.size()));
    std::uint64_t v = 0;
    for (std::size_t k = 0; k < 8; ++k) v |= std::uint64_t(d[k]) << (8 * k);
    return v;
}

// The SFI0 features of a translated container.
std::uint64_t sfi0_of(const std::vector<std::uint8_t>& dxil) {
    const Container host = unwrap([&] { return Container::parse(dxil); });
    return u64_exact(part_data(host, FourCC::SFI0));
}

// `v.validate(&mut dxil).unwrap()`: in place.
void validate_in_place(Validator& v, std::vector<std::uint8_t>& dxil) {
    if (std::optional<std::string> e = v.validate(dxil)) CHECK_MSG(!e, "validator: " + *e);
}

// `if let Ok(mut v) = Validator::load(Validator::default_path()) { v.validate(&mut dxil.clone()).unwrap() }`.
void validate_if_loaded(const std::vector<std::uint8_t>& dxil) {
    std::optional<Validator> v = Validator::load(Validator::default_path());
    if (!v) return;
    std::vector<std::uint8_t> copy = dxil;
    validate_in_place(*v, copy);
}

// `if let Some(path) = Validator::default_path() { Validator::load(Some(&path)).unwrap().validate(...).unwrap() }`.
void validate_if_sdk(const std::vector<std::uint8_t>& dxil) {
    std::optional<fs::path> path = Validator::default_path();
    if (!path) return;
    std::string why;
    std::optional<Validator> v = Validator::load(path, &why);
    if (!v) throw std::logic_error("Validator::load: " + why);
    std::vector<std::uint8_t> copy = dxil;
    validate_in_place(*v, copy);
}

SigElement sig(std::string name, std::uint8_t kind, std::uint8_t comp, std::uint8_t interp,
               std::optional<std::uint32_t> row, std::uint8_t col, std::uint8_t cols, std::uint8_t usage,
               bool half) {
    SigElement e;
    e.name = std::move(name);
    e.index = 0;
    e.kind = kind;
    e.comp = comp;
    e.interp = interp;
    e.row = row;
    e.col = col;
    e.cols = cols;
    e.usage = usage;
    e.half = half;
    return e;
}

// The usual SV_Target element of the tests.
SigElement sv_target() { return sig("SV_Target", 16, 9, 0, 0u, 0, 4, 15, false); }

PsInput generic_input(std::uint8_t mask, bool fp16) {
    PsInput i;
    i.usage = Usage::Generic();
    i.row = 0;
    i.mask = mask;
    i.attr = 0;
    i.flat = false;
    i.fp16 = fp16;
    i.pack2 = false;
    i.row2 = 0;
    return i;
}

PsIo ps_io(std::vector<PsInput> inputs, std::uint32_t input_addr) {
    PsIo io;
    io.inputs = std::move(inputs);
    io.input_addr = input_addr;
    io.db_shader_control = 0;
    return io;
}

NativeProgram program(HwStage stage, bool wave64, std::uint16_t vgprs, std::vector<RootParam> layout,
                      std::vector<UsedResource> used, vs::native::StageIo io, std::span<const std::uint32_t> words) {
    NativeProgram n;
    n.stage = stage;
    n.wave64 = wave64;
    n.vgprs = vgprs;
    n.layout = std::move(layout);
    n.used = std::move(used);
    n.io = std::move(io);
    n.code = le_bytes(words);
    return n;
}

RootParam table_param() { return RootParam{RootType::Table(), 0, 1, 0, 0, 0}; }

// The front end as most tests run it: ir::build of the patched code, structure, resource::track
// with the structure, io::raise; then emit. Throws ShaderError (Rust `?`).
HostShader translate_front(const Container& c, const NativeProgram& native) {
    const std::vector<std::uint8_t> code = native.patched_code();
    const vs::ir::Program p = vs::ir::build(code, native.wave64, native.vgprs);
    const vs::structure::Structured st = vs::structure::structure(p);
    const vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, &st);
    const vs::io::Io io = vs::io::raise(p, native);
    return vs::emit(c, native, p, st, io, res);
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::logic_error("cannot read " + path.string());
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::optional<std::string> env_var(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string(v);
}

// The LLVM text `dxc -dumpbin` prints for a DXIL container, when dxc is installed.
std::optional<std::string> dump_dxil(const std::vector<std::uint8_t>& container) {
    fs::path dxc;
    if (std::optional<std::string> p = env_var("X2W_SHADERCFG_PC_DXC")) {
        dxc = *p;
    } else {
        std::optional<fs::path> d = Validator::default_path();
        if (!d) return std::nullopt;
        dxc = d->replace_filename("dxc.exe");
    }
    std::error_code ec;
    if (!fs::exists(dxc, ec)) return std::nullopt;
    const std::string pid = std::to_string(GetCurrentProcessId());
    const fs::path file = fs::temp_directory_path() / ("gpu-interp-" + pid + ".dxil");
    const fs::path out = fs::temp_directory_path() / ("gpu-interp-" + pid + ".out");
    const fs::path err = fs::temp_directory_path() / ("gpu-interp-" + pid + ".err");
    {
        std::ofstream f(file, std::ios::binary);
        if (!f) return std::nullopt;
        f.write(reinterpret_cast<const char*>(container.data()), static_cast<std::streamsize>(container.size()));
        if (!f) return std::nullopt;
    }
    // The whole command quoted once more: cmd.exe strips the outer pair.
    const std::string cmd = "\"\"" + dxc.string() + "\" -dumpbin \"" + file.string() + "\" > \"" + out.string() +
                            "\" 2> \"" + err.string() + "\"\"";
    const int rc = std::system(cmd.c_str());
    fs::remove(file, ec);
    std::vector<std::uint8_t> text;
    std::vector<std::uint8_t> errors;
    try {
        text = read_file(out);
    } catch (const std::logic_error&) {
    }
    try {
        errors = read_file(err);
    } catch (const std::logic_error&) {
    }
    fs::remove(out, ec);
    fs::remove(err, ec);
    if (rc != 0)
        throw std::logic_error("dxc -dumpbin failed: " + std::string(errors.begin(), errors.end()));
    return std::string(text.begin(), text.end());
}

// ---- rs:989 -------------------------------------------------------------------------------------
void interpolated_f16_writes_the_op_sel_half_with_modifiers() {
    // p1_f32 v2, v0, attr0.x; then a p2_f16 completing it into v3, which holds a value
    // (v2's own is no number here: a first half is evaluated with its second).
    auto p2 = [](bool dst_high, std::uint32_t omod, bool clamp) -> std::array<std::uint32_t, 2> {
        auto vgpr = [](std::uint32_t n) { return 256 + n; };
        std::uint32_t w0 = 0x35u << 26 | 0x35au << 16 | 3;
        if (dst_high) w0 |= 1u << 14;  // op_sel[3]: the high destination half.
        if (clamp) w0 |= 1u << 15;
        const std::uint32_t w1 = vgpr(1) << 9 | vgpr(2) << 18;
        return {w0, w1 | omod << 27};
    };
    auto dump = [&](bool dst_high, std::uint32_t omod, bool clamp) {
        const auto [w0, w1] = p2(dst_high, omod, clamp);
        const std::array<std::uint32_t, 7> words = {
            0x7E06'0301u,  // v_mov_b32 v3, v1
            0xC808'0000,   // v_interp_p1_f32 v2, v0, attr0.x
            w0,
            w1,  // v_interp_p2_f16 v3, attr0, v1, v2
            0xF800'180F,
            0x0303'0303,  // exp mrt0, v3, v3, v3, v3 done vm
            0xBF81'0000,
        };
        const NativeProgram native =
            program(HwStage::Ps, false, 4, {}, {}, ps_io({generic_input(0b1111, true)}, 2), words);
        auto el = [](const char* name, std::uint8_t kind, std::uint8_t usage, bool half) {
            return sig(name, kind, 9, 0, 0u, 0, 4, usage, half);
        };
        const std::vector<SigElement> in = {el("TEXCOORD", 0, 0, true)};
        const std::vector<SigElement> outs = {el("SV_Target", 16, 15, false)};
        const std::vector<std::uint8_t> isg1 = vs::container::signature(in, true);
        const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
        const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::ISG1, isg1}, {FourCC::OSG1, osg1}});
        return unwrap([&] {
            const Container c = Container::parse(bytes);
            return translate_front(c, native).dxil;
        });
    };
    const std::optional<std::string> low = dump_dxil(dump(false, 0, false));
    const std::optional<std::string> high = dump_dxil(dump(true, 1, true));
    if (!low || !high) {
        std::fprintf(stderr, "no dxc.exe; DXIL check skipped\n");
        return;
    }
    // The low-half write keeps the destination's high half.
    CHECK_MSG(contains(*low, "-65536"), "low write keeps the high half");
    // The high-half write shifts the result up by 16.
    CHECK_MSG(contains(*high, "shl i32"), "op_sel[3] writes the high half");
    // omod mul:2 and clamp (dx.op.unary.f32, opcode 7) are applied.
    CHECK_MSG(contains(*high, "2.000000e+00"), "omod multiplies by two");
    CHECK_MSG(contains(*high, "dx.op.unary.f32(i32 7"), "clamp saturates the result");
    CHECK(!contains(*low, "dx.op.unary.f32(i32 7"));
}

// ---- rs:1083 ------------------------------------------------------------------------------------
void the_clean_up_switch_parses_or_runs_every_pass() {
    const std::uint32_t all = vs::bitcode::opt::ALL;
    const std::uint32_t usual = vs::bitcode::opt::DEFAULT;
    CHECK_MSG(usual == 63, "the passes of the builds before the wide loads");
    CHECK(vs::parse_shader_opt(std::nullopt) == usual);
    CHECK(vs::parse_shader_opt("0") == 0);
    CHECK(vs::parse_shader_opt(" 1 ") == 1);
    CHECK(vs::parse_shader_opt("0x1") == 1);
    CHECK(vs::parse_shader_opt("127") == all);
    CHECK(vs::parse_shader_opt("0xffffffff") == all);
    CHECK(vs::parse_shader_opt("on") == usual);
}

// ---- rs:1096 ------------------------------------------------------------------------------------
void the_loop_limit_switch_parses_or_falls_back() {
    using vs::lower::LOOP_LIMIT;
    CHECK(vs::parse_loop_limit(std::nullopt) == LOOP_LIMIT);
    CHECK(vs::parse_loop_limit("4096") == 4096);
    CHECK(vs::parse_loop_limit(" 0x1000 ") == 4096);
    CHECK(vs::parse_loop_limit("1") == 1);
    for (const char* bad : {"", "0", "-5", "lots", "0x", "4294967296"})
        CHECK_MSG(vs::parse_loop_limit(bad) == LOOP_LIMIT, std::string("\"") + bad + "\"");
}

// ---- rs:1111 ------------------------------------------------------------------------------------
// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`). DXC sets ShaderFlags.StencilRef
// (bit 11) for a pixel shader whose output signature has SV_StencilRef, and the feature bit
// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks both against the module.
void a_stencil_export_declares_the_stencil_ref_feature() {
    const std::array<std::uint32_t, 5> words = {
        0x7E00'02F2u,  // v_mov_b32 v0, 1.0
        0x7E02'0285,   // v_mov_b32 v1, 5
        0xF800'1883,
        0x0000'0100,  // exp mrtz v0, v1, off, off done vm
        0xBF81'0000,
    };
    const NativeProgram native = program(HwStage::Ps, false, 2, {}, {}, ps_io({}, 0), words);
    auto el = [](const char* name, std::uint8_t kind, std::uint8_t comp) {
        return sig(name, kind, comp, 0, std::nullopt, 0, 1, 1, false);
    };
    const std::vector<SigElement> outs = {el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)};
    const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
    const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::OSG1, osg1}});
    const HostShader out = unwrap([&] {
        const Container c = Container::parse(bytes);
        return translate_front(c, native);
    });
    // dxil.dll, where there is one, checks the flags against the module.
    validate_if_loaded(out.dxil);
    const std::uint64_t features = sfi0_of(out.dxil);
    CHECK_MSG((features & 0x200) != 0, "SFI0 " + hex(features) + ": no StencilRef");
}

// ---- rs:1173 ------------------------------------------------------------------------------------
// A pixel program with an ordered section, as the console compiler makes one of rasterizer ordered
// views (the title's hair cards keep their nearest layers that way): the wait for overlapped waves
// is skipped, the views inside the section are the host's ordered ones, the one after it is not.
void an_ordered_section_uses_rasterizer_ordered_views() {
    const std::array<std::uint32_t, 37> section = {
        0xBE82'0300u,  // s_mov_b32 s2, s0
        0xBE83'0380,   // s_mov_b32 s3, 0
        0xF40C'0201,   // s_load_dwordx8 s[8:15], s[2:3], 0
        0xFA00'0000,
        0x9390'FF01,  // s_bfe_u32 s16, s1, 0x2001c
        0x0002'001C,
        0x9391'FF01,  // s_bfe_u32 s17, s1, 0xa0000
        0x000A'0000,
        0x8F10'8110,  // s_lshl_b32 s16, s16, 1
        0x9392'FF01,  // s_bfe_u32 s18, s1, 0xa0010
        0x000A'0010,
        0xBE91'0711,  // s_not_b32 s17, s17
        0x9013'9F01,  // s_lshr_b32 s19, s1, 31
        0x8810'8110,  // s_or_b32 s16, s16, 1
        0x8112'1112,  // s_add_i32 s18, s18, s17
        0xBF06'8013,  // s_cmp_eq_u32 s19, 0
        0xB990'1019,  // s_setreg_b32 hwreg(HW_REG_POPS_PACKER, 0, 3), s16
        0x8514'807E,  // s_cselect_b32 s20, exec_lo, 0
        0x8115'11EF,  // s_add_i32 s21, src_pops_exiting_wave_id, s17
        0xBF08'1215,  // s_cmp_gt_u32 s21, s18
        0x8516'807E,  // s_cselect_b32 s22, exec_lo, 0
        0x8814'1614,  // s_or_b32 s20, s20, s22
        0x876A'147E,  // s_and_b32 vcc_lo, exec_lo, s20
        0xBF87'0004,  // s_cbranch_vccnz past the wait
        0xBF8E'FFFF,  // s_sleep 0xffff
        0x8115'11EF,  // s_add_i32 s21, src_pops_exiting_wave_id, s17
        0xBF0A'1512,  // s_cmp_lt_u32 s18, s21
        0xBF84'FFFC,  // s_cbranch_scc0 back to the sleep
        0x7E00'0280,  // v_mov_b32 v0, 0
        0x7E02'0280,  // v_mov_b32 v1, 0
        0xF000'3F88,  // image_load v[2:5], v[0:1], s[8:15] dmask:0xf 2D glc dlc
        0x0002'0200,
        0x0604'04F2,  // v_add_f32 v2, 1.0, v2
        0xF020'3F08,  // image_store v[2:5], v[0:1], s[8:15] dmask:0xf 2D glc
        0x0002'0200,
        0xBEFC'0380,  // s_mov_b32 m0, 0
        0xBF90'0007,  // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
    };
    const std::array<std::uint32_t, 3> end = {
        0xF800'180Fu,  // exp mrt0 v2, v2, v2, v2 done vm
        0x0202'0202,
        0xBF81'0000,  // s_endpgm
    };
    // A store after the section, through an ordinary view.
    const std::array<std::uint32_t, 2> after = {0xF020'3F08u, 0x0002'0200};
    // Throws ShaderError where Rust's closure returns Err.
    auto lower_words = [](const std::vector<std::uint32_t>& words) -> std::uint64_t {
        const NativeProgram native = program(HwStage::Ps, false, 8, {table_param()},
                                             {UsedResource{ResourceType::Uav(), RootType::Table(), 0, 0, 4, 8, 0, 0}},
                                             ps_io({}, 0), words);
        const std::vector<SigElement> outs = {sv_target()};
        const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
        const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::OSG1, osg1}});
        const Container c = unwrap([&] { return Container::parse(bytes); });
        const HostShader out = translate_front(c, native);
        {
            std::optional<Validator> v = Validator::load(Validator::default_path());
            if (v) {
                std::vector<std::uint8_t> copy = out.dxil;
                if (std::optional<std::string> e = v->validate(copy))
                    throw std::logic_error("validator: " + *e);
            }
        }
        return sfi0_of(out.dxil);
    };
    auto concat = [](std::initializer_list<std::span<const std::uint32_t>> parts) {
        std::vector<std::uint32_t> v;
        for (std::span<const std::uint32_t> p : parts) v.insert(v.end(), p.begin(), p.end());
        return v;
    };
    const std::span<const std::uint32_t> all(section);
    // D3D_SHADER_REQUIRES_ROVS
    std::uint64_t features = unwrap([&] { return lower_words(concat({all, end})); });
    CHECK_MSG((features & 0x1000) != 0, "SFI0 " + hex(features) + ": no ROVs");
    unwrap([&] { return lower_words(concat({all, after, end})); });
    // The same accesses outside a section: ordinary views, no feature.
    const std::vector<std::uint32_t> plain = concat({all.subspan(0, 4), all.subspan(28, 35 - 28), end});
    features = unwrap([&] { return lower_words(plain); });
    CHECK_MSG((features & 0x1000) == 0, "SFI0 " + hex(features) + ": ROVs");
    // Without the section's start the wait stays unknown.
    std::vector<std::uint32_t> unmarked = concat({all, end});
    unmarked.erase(unmarked.begin() + 16);
    CHECK(vs::catch_shader_error([&] { lower_words(unmarked); }).has_value());
}

// ---- rs:1301 ------------------------------------------------------------------------------------
void immediate_sampler_slots_distinguish_comparison_uses() {
    const std::array<std::uint32_t, 23> words = {
        0xBE81'0380u,  // s_mov_b32 s1, 0
        0xF40C'0000,
        0xFA00'0000,  // s_load_dwordx8 s[0:7], s[0:1], null
        0xBE88'0380,  // s_mov_b32 s8, 0
        0xBE89'0380,  // s_mov_b32 s9, 0
        0xBE8A'0380,  // s_mov_b32 s10, 0
        0xBE8B'0380,  // s_mov_b32 s11, 0
        0x7E00'02F0,  // v_mov_b32 v0, 0.5 (comparison reference)
        0x7E02'02F0,  // v_mov_b32 v1, 0.5 (u)
        0x7E04'02F0,  // v_mov_b32 v2, 0.5 (v)
        0xBF8C'0070,  // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xF080'0108,
        0x0040'0301,  // image_sample v3, v[1:2], s[0:7], s[8:11]
        0xF0A0'0108,
        0x0040'0400,  // image_sample_c v4, v[0:2], s[0:7], s[8:11]
        0xF0BC'0108,
        0x0040'0500,  // image_sample_c_lz v5, v[0:2], s[0:7], s[8:11]
        0xF080'0108,
        0x0040'0601,  // image_sample v6, v[1:2], s[0:7], s[8:11]
        0xBF8C'3F70,  // s_waitcnt vmcnt(0)
        0xF800'180F,
        0x0605'0403,  // exp mrt0, v3, v4, v5, v6 done vm
        0xBF81'0000,
    };
    const NativeProgram native = program(HwStage::Ps, false, 7, {table_param()},
                                         {UsedResource{ResourceType::Srv(), RootType::Table(), 0, 0, 0, 8, 0, 0}},
                                         ps_io({}, 0), words);
    const std::vector<SigElement> outs = {sv_target()};
    const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
    const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::OSG1, osg1}});
    const HostShader out = unwrap([&] {
        const Container c = Container::parse(bytes);
        return translate_front(c, native);
    });

    const vs::StaticSampler sampler = vs::StaticSampler::from_words({0, 0, 0, 0});
    vs::StaticSampler comparison = sampler;
    comparison.filter = sampler.filter | 0x80;
    const std::vector<std::array<std::uint32_t, 4>> zeros(2, std::array<std::uint32_t, 4>{});
    CHECK(out.samplers == zeros);
    CHECK((out.static_samplers == std::vector<vs::StaticSampler>{sampler, comparison}));
    validate_if_sdk(out.dxil);
}

// ---- rs:1393 ------------------------------------------------------------------------------------
// A wave64 compute program that writes only EXEC's high dword (`s_mov_b32 exec_hi, 0`) before a
// vector instruction. The new mask was opaque and keyed by the unchanged low dword, so building its
// lane bit needed the mask's own value: lowering recursed until the stack overflowed, which takes
// the whole process down (found by random-input testing). Now that low dword, seen with two high
// dwords, has no mask, and the instruction fails like any other that cannot be expressed: its
// result is missing, and only a side effect that uses it would stop the translation.
void an_exec_high_write_does_not_overflow_the_stack() {
    // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
    const std::array<std::uint32_t, 3> words = {0xBEFF'0380u, 0x7E02'0300, 0xBF81'0000};
    const NativeProgram native = program(HwStage::Cs, true, 2, {}, {}, ComputeIo{0, {64, 1, 1}}, words);
    const std::vector<std::uint8_t> code = native.patched_code();
    const vs::ir::Program p = unwrap([&] { return vs::ir::build(code, native.wave64, native.vgprs); });
    const vs::structure::Structured st = unwrap([&] { return vs::structure::structure(p); });
    const vs::resource::Resources res =
        unwrap([&] { return vs::resource::track(p, native.layout, native.stage, &st); });
    const vs::io::Io io = unwrap([&] { return vs::io::raise(p, native); });
    const std::vector<std::uint8_t> container = vs::dxbc::build({});
    const Container c = unwrap([&] { return Container::parse(container); });
    try {
        (void)vs::emit(c, native, p, st, io, res);
    } catch (const ShaderError& e) {
        CHECK_MSG(contains(e.to_string(), "its own value"), e.to_string());
    }
}

// rs:1429: a wave64 compute program: `body`, then a store of v1 to a UAV. The error of emit (Rust's
// Result); the front end's errors are panics.
std::optional<ShaderError> stores_v1(std::span<const std::uint32_t> body) {
    // s_mov_b32 s1, 0; s_load_dwordx4 s[8:11], s[0:1], 0
    std::vector<std::uint32_t> words = {0xBE81'0380, 0xF408'0200, 0xFA00'0000};
    words.insert(words.end(), body.begin(), body.end());
    // v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen; s_endpgm
    for (std::uint32_t w : {0x3406'0082u, 0xE070'1000u, 0x8002'0103u, 0xBF81'0000u}) words.push_back(w);
    const NativeProgram native = program(HwStage::Cs, true, 4, {table_param()},
                                         {UsedResource{ResourceType::Uav(), RootType::Table(), 0, 0, 0, 4, 0, 0}},
                                         ComputeIo{2, {64, 1, 1}}, words);
    const std::vector<std::uint8_t> code = native.patched_code();
    const vs::ir::Program p = unwrap([&] { return vs::ir::build(code, native.wave64, native.vgprs); });
    const vs::resource::Resources res =
        unwrap([&] { return vs::resource::track(p, native.layout, native.stage, nullptr); });
    const vs::io::Io io = unwrap([&] { return vs::io::raise(p, native); });
    const vs::structure::Structured st = unwrap([&] { return vs::structure::structure(p); });
    const std::vector<std::uint8_t> container = vs::dxbc::build({});
    const Container c = unwrap([&] { return Container::parse(container); });
    return vs::catch_shader_error([&] { (void)vs::emit(c, native, p, st, io, res); });
}

std::string words_hex(std::span<const std::uint32_t> body) {
    std::string s = "[";
    for (std::size_t k = 0; k < body.size(); ++k) s += (k ? ", " : "") + hex(body[k]);
    return s + "]";
}

// ---- rs:1476 ------------------------------------------------------------------------------------
// The high dword of a wave64 mask rewritten alone (`s_mov_b32 vcc_hi` or `exec_hi`): lanes 32-63
// took the mask the low dword was first written with (the compare's, EXEC's on entry). Now the
// store refuses; without the write it translates.
void a_mask_whose_high_dword_is_rewritten_is_refused() {
    // v_cmp_eq_u32 vcc, 0, v0; v_mov_b32 v2, 11; v_cndmask_b32 v1, 0, v2, vcc
    const std::vector<std::uint32_t> select = {0x7D84'0080, 0x7E04'028B, 0x0202'0480};
    // The same with v_add_co_ci_u32 v1, vcc, 0, v2, vcc.
    const std::vector<std::uint32_t> carry = {0x7D84'0080, 0x7E04'028B, 0x5002'0480};
    // v_mov_b32 v1, 7
    const std::vector<std::uint32_t> mov = {0x7E02'0287};
    for (const std::vector<std::uint32_t>* body : {&select, &carry, &mov}) {
        const std::optional<ShaderError> e = stores_v1(*body);
        if (e) throw std::logic_error("called `Result::unwrap()` on an `Err` value: " + e->to_string());
    }
    // s_mov_b32 vcc_hi, 0 after the compare.
    const std::vector<std::uint32_t> select_hi = {0x7D84'0080, 0xBEEB'0380, 0x7E04'028B, 0x0202'0480};
    const std::vector<std::uint32_t> carry_hi = {0x7D84'0080, 0xBEEB'0380, 0x7E04'028B, 0x5002'0480};
    // s_mov_b32 exec_hi, 0; v_mov_b32 v1, 7, then the same after s_mov_b64 s[2:3], exec.
    const std::vector<std::uint32_t> exec_hi = {0xBEFF'0380, 0x7E02'0287};
    const std::vector<std::uint32_t> exec_hi_saved = {0xBE82'047E, 0xBEFF'0380, 0x7E02'0287};
    for (const std::vector<std::uint32_t>* body : {&select_hi, &carry_hi, &exec_hi, &exec_hi_saved})
        CHECK_MSG(stores_v1(*body).has_value(), words_hex(*body));
}

// ---- rs:1503 ------------------------------------------------------------------------------------
// `ds_bpermute_b32` stores nothing: each lane reads the lane its address names. It stopped as "LDS
// the program does not allocate" (sticky lips) or "an LDS operation" (the checkerboard shadow
// masks).
void a_backward_permute_reads_the_lane_its_address_names() {
    // v_mov_b32 v2, 11; ds_bpermute_b32 v1, v0, v2
    const std::array<std::uint32_t, 3> body = {0x7E04'028B, 0xDACC'0000, 0x0100'0200};
    const std::optional<ShaderError> e = stores_v1(body);
    if (e) throw std::logic_error("called `Result::unwrap()` on an `Err` value: " + e->to_string());
}

// ---- rs:1512 ------------------------------------------------------------------------------------
// A carry-in reading a join of compares after a divergent branch. A lane that skipped the inner
// compare has VCC clear there (VOPC clears inactive lanes), not the first compare's bit, so the
// join is no per-lane phi: lowering refuses it.
void a_carry_in_from_a_divergent_join_is_refused() {
    const std::array<std::uint32_t, 7> body = {
        0x7E04'028B,  // v_mov_b32 v2, 11
        0x7D84'0080,  // v_cmp_eq_u32 vcc, 0, v0
        0xBE82'376A,  // s_andn1_saveexec_b64 s[2:3], vcc
        0xBF88'0001,  // s_cbranch_execz 1
        0x7D84'0081,  // v_cmp_eq_u32 vcc, 1, v0
        0x88FE'027E,  // s_or_b64 exec, exec, s[2:3]
        0x5002'0480,  // v_add_co_ci_u32 v1, vcc, 0, v2, vcc
    };
    const std::optional<ShaderError> err = stores_v1(body);
    if (!err) throw std::logic_error("called `Result::unwrap_err()` on an `Ok` value");
    const std::string e = err->to_string();
    CHECK_MSG(contains(e, "a carry-in from an untracked mask"), e);
}

// rs:1531: a pixel shader writing a constant colour.
std::vector<std::uint8_t> constant_color() {
    vs::shader::Shader s(true);
    s.outputs.push_back(sv_target());
    const std::size_t b = s.entry;
    const std::array<float, 4> colour = {1.0f, 0.5f, 0.25f, 1.0f};
    for (std::size_t c = 0; c < colour.size(); ++c) {
        const float v = colour[c];
        // Braced: evaluated left to right, as Rust's array.
        const std::array<vs::bitcode::V, 4> args = {
            s.m.i32(0),
            s.m.i32(0),
            s.m.i8(static_cast<std::uint8_t>(c)),
            s.m.f32(std::bit_cast<std::uint32_t>(v)),
        };
        s.op(b, "storeOutput.f32", vs::bitcode::Attr::NoUnwind, s.t.void_, 5, std::span<const vs::bitcode::V>(args));
    }
    s.m.terminate(b, vs::bitcode::Term::ret());
    return s.finish();
}

HostShader host_shader(HwStage stage, std::vector<std::uint8_t> dxil) {
    HostShader h;
    h.stage = stage;
    h.dxil = std::move(dxil);
    return h;
}

// ---- rs:1561 ------------------------------------------------------------------------------------
void stub_pipeline_is_created() {
    std::optional<vs::host::Host> host;
    try {
        host.emplace(false);
    } catch (const vs::host::Error& e) {
        std::fprintf(stderr, "no SM 6.6 device (%s); skipped\n", e.what());
        return;
    }
    const HostShader fallback = host_shader(HwStage::Ps, constant_color());
    auto file = [](const char* var, HwStage stage) -> std::optional<HostShader> {
        std::optional<std::string> p = env_var(var);
        if (!p) return std::nullopt;
        return host_shader(stage, read_file(*p));
    };
    const HostShader ps = file("X2W_SHADERCFG_REF_PS", HwStage::Ps).value_or(fallback);
    std::optional<HostShader> vs_file = file("X2W_SHADERCFG_REF_VS", HwStage::Vs);
    const HostShader vertex = vs_file ? std::move(*vs_file) : vs::host::stub_vs(ps);
    validate_if_loaded(vertex.dxil);
    (void)host->pipeline(vertex, ps);
    // The runtime checks the container hash: no hash or a wrong one is refused. (Recent runtimes
    // also take sixteen 0x01 bytes, a bypass value; ours works with older ones too.)
    if (ps.dxil.size() < 20) throw std::logic_error("container shorter than its header");
    std::vector<std::uint8_t> wrong(ps.dxil.begin() + 4, ps.dxil.begin() + 20);
    wrong[0] ^= 1;
    const std::vector<std::uint8_t> zero(16, 0);
    for (const std::vector<std::uint8_t>* digest : std::array<const std::vector<std::uint8_t>*, 2>{&zero, &wrong}) {
        HostShader other = ps;
        std::copy(digest->begin(), digest->end(), other.dxil.begin() + 4);
        bool refused = false;
        try {
            (void)host->pipeline(vertex, other);
        } catch (const vs::host::Error&) {
            refused = true;
        }
        std::string text = "digest";
        for (std::uint8_t b : *digest) {
            char h[4];
            std::snprintf(h, sizeof h, " %02x", b);
            text += h;
        }
        CHECK_MSG(refused, text + " accepted");
    }
}

// ---- rs:1619 ------------------------------------------------------------------------------------
void constant_color_validates() {
    std::vector<std::uint8_t> c = constant_color();
    const std::optional<fs::path> path = Validator::default_path();
    std::optional<Validator> v = Validator::load(path);
    if (!v) {
        std::fprintf(stderr, "no dxil.dll; skipped\n");
        return;
    }
    const std::vector<std::uint8_t> ours(c.begin() + 4, c.begin() + 20);
    validate_in_place(*v, c);
    CHECK_MSG(std::equal(ours.begin(), ours.end(), c.begin() + 4), "our hash differs from the validator's");
}

// rs:1633: a pixel program of `words` with no user data.
NativeProgram emit_native(std::span<const std::uint32_t> words, std::uint32_t input_addr) {
    return program(HwStage::Ps, false, 4, {}, {}, ps_io({}, input_addr), words);
}

// Triangle (0, 1, 2), each corner's 9-bit index in its 10-bit field.
constexpr std::uint32_t TRIANGLE = 1u << 10 | 2u << 20;
// The edge flags of the three corners (bits 9, 19 and 29).
constexpr std::uint32_t EDGE_FLAGS = 1u << 9 | 1u << 19 | 1u << 29;
// The culling bit.
constexpr std::uint32_t CULL = 1u << 31;

// rs:1784: the triangle with its far corners at `reach` - 1. Throws ShaderError.
HostShader triangle_mesh_to(std::uint32_t prim, float reach) {
    // (the second corner's y comes out as 2 x its lane bit: half of it)
    const std::uint32_t x = std::bit_cast<std::uint32_t>(reach);
    const std::uint32_t y = std::bit_cast<std::uint32_t>(reach / 2.0f);
    const std::array<std::uint32_t, 21> words = {
        0xD765'0001u,  // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x0001'00C1,
        0x3604'0281,  // v_and_b32 v2, 1, v1
        0x3606'0282,  // v_and_b32 v3, 2, v1
        0x7E04'0D02,  // v_cvt_f32_u32 v2, v2
        0x7E06'0D03,  // v_cvt_f32_u32 v3, v3
        0x1008'04FF,  // v_mul_f32 v4, reach, v2
        x,
        0x0608'08F3,  // v_add_f32 v4, -1.0, v4
        0x100A'06FF,  // v_mul_f32 v5, reach / 2, v3
        y,
        0x060A'0AF3,  // v_add_f32 v5, -1.0, v5
        0x7E0C'02F0,  // v_mov_b32 v6, 0.5
        0x7E0E'02F2,  // v_mov_b32 v7, 1.0
        0x7E12'02FF,  // v_mov_b32 v9, prim
        prim,
        0xF800'0941,  // exp prim v9, off, off, off done
        0x0000'0009,
        0xF800'08CF,  // exp pos0 v4, v5, v6, v7 done
        0x0706'0504,
        0xBF81'0000,  // s_endpgm
    };
    MeshIo mesh;
    mesh.clip_cntl = 0;
    mesh.threads = 32;
    mesh.max_vertices = 32;
    mesh.max_primitives = 32;
    mesh.topology = 2;
    mesh.lds_bytes = 0;
    const NativeProgram native = program(HwStage::Ms, false, 10, {}, {}, mesh, words);
    const std::vector<SigElement> outs = {sig("SV_Position", 3, 9, 0, 0u, 0, 4, 0xF, false)};
    const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
    const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::OSG1, osg1}});
    const Container c = Container::parse(bytes);
    return translate_front(c, native);
}

// rs:1776: a one-wave (32-lane) triangle mesh program whose lanes all export primitive `prim`
// and, as vertex `lane`, the position (lane & 1 ? 3 : -1, lane & 2 ? 3 : -1, 0.5, 1): triangle
// (0, 1, 2) covers the whole target.
HostShader triangle_mesh(std::uint32_t prim) { return triangle_mesh_to(prim, 4.0f); }

// ---- rs:1651 ------------------------------------------------------------------------------------
// A pixel program that takes its neighbour's position across its quad (`ds_swizzle_b32`, as the
// console's compiler makes derivatives) and exports the differences in x and y: 1 and -1 in every
// pixel of a triangle, those at its edges too, whose neighbours are helpers.
void quad_neighbours_at_a_triangles_edge_are_read() {
    using vs::host::Draw;
    using vs::host::Host;
    using vs::host::Root;
    // v2, v3: position x, y (input address: POS_X and POS_Y after the two perspective barycentrics).
    auto swizzle = [](std::uint32_t dst, std::uint32_t src, std::uint32_t pattern) -> std::array<std::uint32_t, 2> {
        return {0xD800'0000u | 0x35u << 18 | 0x8000u | pattern, dst << 24 | src};
    };
    std::vector<std::uint32_t> words = {0xBEFC'0380u};  // s_mov_b32 m0, 0
    for (const auto& pair : {swizzle(4, 2, 0xB1),        // lanes 1 0 3 2
                             swizzle(5, 3, 0x4E),        // lanes 2 3 0 1
                             swizzle(6, 2, 0xA0)})       // lanes 0 0 2 2
        words.insert(words.end(), pair.begin(), pair.end());
    for (std::uint32_t w : {
             0x0808'0504u,  // v_sub_f32 v4, v4, v2
             0x080A'0705u,  // v_sub_f32 v5, v5, v3
             0x080C'0506u,  // v_sub_f32 v6, v6, v2
             0xF800'180Fu,
             0x0302'0504u,  // exp mrt0 v4, v5, v2, v3 done vm
             0xBF81'0000u,
         })
        words.push_back(w);
    NativeProgram native = emit_native(words, 1u << 1 | 1u << 8 | 1u << 9);
    native.vgprs = 8;
    const std::vector<SigElement> outs = {sv_target()};
    const std::vector<SigElement> ins = {sig("SV_Position", 3, 9, 4, 0u, 0, 4, 3, false)};
    const std::vector<std::uint8_t> isg1 = vs::container::signature(ins, true);
    const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
    const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::ISG1, isg1}, {FourCC::OSG1, osg1}});
    const HostShader ps = unwrap([&] {
        const Container c = Container::parse(bytes);
        return translate_front(c, native);
    });
    validate_if_sdk(ps.dxil);
    const HostShader ms = unwrap([&] { return triangle_mesh_to(TRIANGLE, 1.3f); });
    for (const auto& [label, warp] : {std::pair<const char*, bool>{"hardware", false}, {"warp", true}}) {
        std::optional<Host> host;
        try {
            host.emplace(warp);
        } catch (const vs::host::Error&) {
            continue;
        }
        vs::host::ComPtr<ID3D12RootSignature> root;
        try {
            root = host->root_signature_for({{ms, D3D12_SHADER_VISIBILITY_MESH}, {ps, D3D12_SHADER_VISIBILITY_PIXEL}});
        } catch (const vs::host::Error&) {
            continue;
        }
        vs::host::ComPtr<ID3D12PipelineState> pso;
        try {
            pso = host->mesh_pipeline_with(root.Get(), ms.dxil, ps.dxil);
        } catch (const vs::host::Error&) {
            std::fprintf(stderr, "no mesh pipeline on %s; skipped\n", label);
            continue;
        }
        Draw d;
        d.roots.emplace_back(0u, Root::constants({0}));
        d.roots.emplace_back(1u, Root::constants({0}));
        d.stride = 0;
        d.size = {16, 16};
        d.mesh_groups = 1;
        std::vector<std::array<float, 4>> texels;
        try {
            texels = host->draw(root.Get(), pso.Get(), d);
        } catch (const vs::host::Error& e) {
            throw std::logic_error(std::string(label) + ": draw: " + e.what());
        }
        std::vector<std::pair<std::size_t, std::array<float, 4>>> lit;
        for (std::size_t n = 0; n < texels.size(); ++n)
            if (texels[n][2] != 0.0f) lit.emplace_back(n, texels[n]);
        CHECK_MSG(lit.size() >= 20 && lit.size() < 240,
                  std::string(label) + ": " + std::to_string(lit.size()) + " texels lit");
        std::vector<std::pair<std::size_t, std::array<float, 4>>> bad;
        for (const auto& [n, t] : lit) {
            const float x = n % 2 == 0 ? 1.0f : -1.0f;
            const float y = n / 16 % 2 == 0 ? 1.0f : -1.0f;
            if (t[0] != x || t[1] != y) bad.emplace_back(n, t);
        }
        if (!bad.empty()) {
            char first[160];
            std::snprintf(first, sizeof first, "(%zu, [%g, %g, %g, %g])", bad[0].first, bad[0].second[0],
                          bad[0].second[1], bad[0].second[2], bad[0].second[3]);
            CHECK_MSG(bad.empty(), std::string(label) + ": " + std::to_string(bad.size()) + " of " +
                                       std::to_string(lit.size()) + " texels: " + first);
        }
    }
}

// ---- rs:1850 ------------------------------------------------------------------------------------
// A primitive export's corners are 9-bit indices: the edge flag above each (bits 9, 19, 29) is no
// part of it. Masking ten bits made a flagged corner index 512 or more, past any mesh program's
// vertices (the draw test below shows the triangle). Flagged, unflagged and culled exports
// translate, and dxil.dll, where there is one, takes each module.
void primitive_export_indices_drop_the_edge_flags() {
    std::optional<Validator> validator = Validator::load(Validator::default_path());
    for (std::uint32_t prim : {TRIANGLE, TRIANGLE | EDGE_FLAGS, TRIANGLE | EDGE_FLAGS | CULL}) {
        HostShader ms;
        try {
            ms = triangle_mesh(prim);
        } catch (const ShaderError& e) {
            throw std::logic_error(hex(prim) + ": " + e.to_string());
        }
        if (validator) {
            std::vector<std::uint8_t> copy = ms.dxil;
            if (std::optional<std::string> e = validator->validate(copy))
                throw std::logic_error(hex(prim) + ": " + *e);
        }
    }
}

// ---- rs:1877 ------------------------------------------------------------------------------------
// A triangle mesh program on the host GPU and on WARP (where they have mesh shaders): triangle
// (0, 1, 2) covers the target with its corners' edge flags set as without them; with the culling
// bit, nothing is lit.
void edge_flagged_triangles_draw_as_unflagged_ones() {
    using vs::host::Draw;
    using vs::host::Host;
    using vs::host::Root;
    const std::vector<std::uint8_t> ps = constant_color();
    int ran = 0;
    for (const auto& [label, warp] : {std::pair<const char*, bool>{"hardware", false}, {"warp", true}}) {
        std::optional<Host> host;
        try {
            host.emplace(warp);
        } catch (const vs::host::Error&) {
            std::fprintf(stderr, "no SM 6.6 %s device; skipped\n", label);
            continue;
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
        const HRESULT queried = host->device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options,
                                                                  static_cast<UINT>(sizeof options));
        if (FAILED(queried) || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED) {
            std::fprintf(stderr, "no mesh shaders on the %s device; skipped\n", label);
            continue;
        }
        for (const auto& [prim, culled] : {std::pair<std::uint32_t, bool>{TRIANGLE, false},
                                           {TRIANGLE | EDGE_FLAGS, false},
                                           {TRIANGLE | EDGE_FLAGS | CULL, true}}) {
            const HostShader ms = unwrap([&] { return triangle_mesh(prim); });
            const vs::host::ComPtr<ID3D12RootSignature> root =
                host->root_signature_for({{ms, D3D12_SHADER_VISIBILITY_MESH}});
            vs::host::ComPtr<ID3D12PipelineState> pso;
            try {
                pso = host->mesh_pipeline_with(root.Get(), ms.dxil, ps);
            } catch (const vs::host::Error& e) {
                throw std::logic_error(std::string(label) + ": mesh pipeline: " + e.what());
            }
            Draw d;
            d.roots.emplace_back(0u, Root::constants({0}));
            d.stride = 0;
            d.size = {16, 16};
            d.mesh_groups = 1;
            std::vector<std::array<float, 4>> texels;
            try {
                texels = host->draw(root.Get(), pso.Get(), d);
            } catch (const vs::host::Error& e) {
                throw std::logic_error(std::string(label) + ": draw: " + e.what());
            }
            const auto lit = std::count_if(texels.begin(), texels.end(), [](const auto& t) { return t[3] != 0.0f; });
            if (culled) {
                CHECK_MSG(lit == 0, std::string(label) + ", " + hex(prim) + ": a culled triangle lit texels");
                continue;
            }
            const std::array<float, 4> colour = {1.0f, 0.5f, 0.25f, 1.0f};
            CHECK_MSG(std::all_of(texels.begin(), texels.end(), [&](const auto& t) { return t == colour; }),
                      std::string(label) + ", " + hex(prim) + ": " + std::to_string(lit) + " of 256 texels lit");
        }
        std::fprintf(stderr, "%s: edge-flagged triangles drew\n", label);
        ran += 1;
    }
    if (ran == 0) std::fprintf(stderr, "no device with mesh shaders; skipped\n");
}

// ---- rs:1956 ------------------------------------------------------------------------------------
// A pixel shader interpolating at sample positions (PERSP_SAMPLE barycentrics, `SPI_PS_INPUT_ADDR`
// bit 0) runs at sample frequency: DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte
// 1) whenever an input is interpolated per sample.
void sample_interpolated_inputs_set_psv_sample_frequency() {
    const std::array<std::uint32_t, 5> words = {
        0xC808'0000u,  // v_interp_p1_f32 v2, v0, attr0.x
        0xC809'0001,   // v_interp_p2_f32 v2, v1, attr0.x
        0xF800'180F,
        0x0202'0202,  // exp mrt0, v2, v2, v2, v2 done vm
        0xBF81'0000,
    };
    auto psv_of = [&](std::uint32_t input_addr) {
        const NativeProgram native =
            program(HwStage::Ps, false, 4, {}, {}, ps_io({generic_input(0b1111, false)}, input_addr), words);
        auto el = [](const char* name, std::uint8_t kind, std::uint8_t usage) {
            return sig(name, kind, 9, 0, 0u, 0, 4, usage, false);
        };
        const std::vector<SigElement> in = {el("TEXCOORD", 0, 0)};
        const std::vector<SigElement> outs = {el("SV_Target", 16, 15)};
        const std::vector<std::uint8_t> isg1 = vs::container::signature(in, true);
        const std::vector<std::uint8_t> osg1 = vs::container::signature(outs, false);
        const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::ISG1, isg1}, {FourCC::OSG1, osg1}});
        const HostShader out = unwrap([&] {
            const Container c = Container::parse(bytes);
            return translate_front(c, native);
        });
        // dxil.dll, where there is one, checks PSV0 against the module.
        validate_if_loaded(out.dxil);
        const Container host = unwrap([&] { return Container::parse(out.dxil); });
        const Bytes psv = part_data(host, FourCC::PSV0);
        return std::vector<std::uint8_t>(psv.begin(), psv.end());
    };
    // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
    const std::vector<std::uint8_t> center = psv_of(1u << 1);
    if (center.size() <= 5) throw std::logic_error("PSV0 too short");
    CHECK_MSG(center[4 + 1] == 0, "center interpolation is per pixel");
    const std::vector<std::uint8_t> sample = psv_of(1u << 0);
    if (sample.size() <= 5) throw std::logic_error("PSV0 too short");
    CHECK_MSG(sample[4 + 1] == 1, "PSInfo.SampleFrequency");
}

// ---- rs:2033 ------------------------------------------------------------------------------------
// A pixel shader reading channel z of an input whose element starts at w. Looking the channel up
// subtracted the element's first channel before checking the channel lies in it, so in a build with
// overflow checks (tests, debug) any read below an element that starts at z or w panicked,
// whichever row it was in (found by random-input testing).
void reading_a_channel_below_an_input_element_does_not_underflow() {
    // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
    const std::array<std::uint32_t, 2> words = {0xC80E'0202u, 0xBF81'0000};
    const NativeProgram native =
        program(HwStage::Ps, false, 4, {}, {}, ps_io({generic_input(0b1000, false)}, 0), words);
    const std::vector<SigElement> in = {sig("TEXCOORD", 0, 9, 0, 0u, 3, 1, 0, false)};
    const std::vector<std::uint8_t> isg1 = vs::container::signature(in, true);
    const std::vector<std::uint8_t> bytes = vs::dxbc::build({{FourCC::ISG1, isg1}});
    const Container c = unwrap([&] { return Container::parse(bytes); });
    const std::vector<std::uint8_t> code = native.patched_code();
    const vs::ir::Program p = unwrap([&] { return vs::ir::build(code, native.wave64, native.vgprs); });
    const vs::structure::Structured st = unwrap([&] { return vs::structure::structure(p); });
    const vs::resource::Resources res =
        unwrap([&] { return vs::resource::track(p, native.layout, native.stage, &st); });
    const vs::io::Io io = unwrap([&] { return vs::io::raise(p, native); });
    // Translated or refused with a reason; it must not panic.
    (void)vs::catch_shader_error([&] { (void)vs::emit(c, native, p, st, io, res); });
}

// The `Parts` of a compute program without signatures (rs:2167, rs:2324).
vs::lower::Lowered lower_compute(const NativeProgram& native, const vs::ir::Program& p,
                                 const vs::structure::Structured& st, const vs::io::Io& io,
                                 const vs::resource::Resources& res) {
    vs::lower::Parts parts;
    parts.native = &native;
    parts.p = &p;
    parts.st = &st;
    parts.io = &io;
    parts.res = &res;
    parts.mesh_stride = 0;
    return vs::lower::lower(parts);
}

// ---- rs:2088 ------------------------------------------------------------------------------------
// `s_buffer_load_dwordx4`, `x8` and `x16` from a raw SRV read the dwords the console's
// range-checked load reads, a call a dword or (`X2W_SHADERCFG_SHADER_OPT` with 64) a call for every four:
// also where a call's four dwords lie across the view's end (the dwords out of it are 0, the others
// are read). On the software adapter only: run the test once plain and once with
// `X2W_SHADERCFG_SHADER_OPT=127`.
void scalar_buffer_loads_read_the_same_dwords_one_call_or_four() {
    // (first register, dwords) of the three loads, in order.
    constexpr std::array<std::pair<std::uint32_t, std::uint32_t>, 3> LOADS = {{{16, 8}, {24, 4}, {32, 16}}};
    std::vector<std::uint32_t> words = {
        0xBE85'0380,  // s_mov_b32 s5, 0
        0xF408'0202,
        0xFA00'0000,  // s_load_dwordx4 s[8:11], s[4:5], 0x0 (the SRV)
        0xF408'0302,
        0xFA00'0020,  // s_load_dwordx4 s[12:15], s[4:5], 0x20 (the UAV)
        0xF42C'0404,
        0x0000'0000,  // s_buffer_load_dwordx8 s[16:23], s[8:11], s0
        0xF428'0604,
        0xFA00'0008,  // s_buffer_load_dwordx4 s[24:27], s[8:11], 0x8
        0xF430'0804,
        0x0200'0004,  // s_buffer_load_dwordx16 s[32:47], s[8:11], s1, 0x4
    };
    std::uint32_t out_dword = 0;
    for (const auto& [first, n] : LOADS) {
        for (std::uint32_t k = 0; k < n; ++k) {
            // v_mov_b32 v1, s<first + k>
            words.push_back(0x7E00'0000u | 1u << 17 | 0x001u << 9 | (first + k));
            // buffer_store_dword v1, s[12:15], offset 4 * out_dword
            words.push_back(0x38u << 26 | 0x1Cu << 18 | (4 * out_dword));
            words.push_back(0x80u << 24 | (12u >> 2) << 16 | 1u << 8);
            out_dword += 1;
        }
    }
    words.push_back(0xBF81'0000);  // s_endpgm
    auto table = [](ResourceType kind, std::uint32_t at) {
        return UsedResource{kind, RootType::Table(), at, 0, 4, 4, 4, 1};
    };
    const NativeProgram native = program(
        HwStage::Cs, false, 4, {RootParam{RootType::Constants(), 0, 4, 0, 0, 0}, RootParam{RootType::Table(), 0, 1, 0, 4, 1}},
        {table(ResourceType::Srv(), 0), table(ResourceType::Uav(), 8)}, ComputeIo{5u << 1, {1, 1, 1}}, words);
    const vs::ir::Program p = unwrap([&] { return vs::ir::build(native.code, native.wave64, native.vgprs); });
    const vs::resource::Resources res =
        unwrap([&] { return vs::resource::track(p, native.layout, native.stage, nullptr); });
    const vs::io::Io io = unwrap([&] { return vs::io::raise(p, native); });
    const vs::structure::Structured st = unwrap([&] { return vs::structure::structure(p); });
    vs::lower::Lowered l = unwrap([&] { return lower_compute(native, p, st, io, res); });
    // The calls that read the SRV: 28 a dword, 7 with four a call.
    const bool wide = (vs::shader_opt() & vs::bitcode::opt::WIDE) != 0;
    std::vector<std::uint8_t> dxil = std::move(l.dxil);
    if (std::optional<Validator> v = Validator::load(Validator::default_path())) validate_in_place(*v, dxil);
    std::optional<vs::host::Host> host;
    try {
        host.emplace(true);
    } catch (const vs::host::Error&) {
        std::fprintf(stderr, "no software adapter; the dispatch is skipped\n");
        return;
    }
    HostShader cs = host_shader(HwStage::Cs, std::move(dxil));
    cs.user_data = l.user_data;
    cs.root_cbvs = l.root_cbvs;
    cs.root_srvs = l.root_srvs;
    cs.root_uavs = l.root_uavs;
    const auto [root, pso] = host->compute_pipeline(cs);
    // Twelve dwords: the loads below run across the view's end.
    std::vector<std::uint32_t> data;
    for (std::uint32_t i = 0; i < 12; ++i) data.push_back(0x1111'1111u * (i + 1) ^ 0x8000'0000u);
    auto dword = [&](std::uint64_t byte) -> std::uint32_t {
        if (byte % 4 == 0 && byte + 4 <= 4 * std::uint64_t(data.size())) return data[static_cast<std::size_t>(byte / 4)];
        return 0;
    };
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 6> offsets = {
        {{0, 0}, {16, 0}, {24, 4}, {40, 28}, {48, 44}, {8, 12}}};
    for (const auto& [s0, s1] : offsets) {
        const std::array<vs::host::View, 2> views = {vs::host::View::buffer(data, vs::host::Stride::raw()),
                                                     vs::host::View::raw_uav(32)};
        vs::host::Dispatch d;
        d.roots.emplace_back(0u, vs::host::Root::constants({s0, s1, 0, 0, 0}));
        d.views = views;
        d.groups = {1, 1, 1};
        const std::vector<std::vector<std::uint32_t>> out = host->dispatch(root.Get(), pso.Get(), d);
        const std::array<std::uint64_t, 3> starts = {std::uint64_t(s0), 8, std::uint64_t(s1) + 4};
        std::vector<std::uint32_t> want;
        for (std::size_t m = 0; m < LOADS.size(); ++m)
            for (std::uint64_t k = 0; k < LOADS[m].second; ++k) want.push_back(dword(starts[m] + 4 * k));
        if (out.empty() || out[0].size() < 28) throw std::logic_error("the UAV holds fewer than 28 dwords");
        const std::vector<std::uint32_t> got(out[0].begin(), out[0].begin() + 28);
        std::string text = "offsets " + std::to_string(s0) + " and " + std::to_string(s1) +
                           ", four dwords a call: " + (wide ? "true" : "false");
        std::fprintf(stderr, "%s: %s\n", text.c_str(), words_hex(got).c_str());
        CHECK_MSG(got == want, text);
    }
}

// ---- rs:2238 ------------------------------------------------------------------------------------
void valu_div_f32_sequence_executes_on_gpu() {
    const std::array<std::uint32_t, 28> words = {
        0xBE85'0380u,  // s_mov_b32 s5, 0
        0xF408'0202,
        0xFA00'0000,  // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408'0302,
        0xFA00'0020,  // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x340E'0083,  // v_lshlrev_b32 v7, 3, v0
        0xE034'1000,
        0x8002'0107,  // buffer_load_dwordx2 v[1:2], v7, s[8:11], 0 offen
        0xD56D'6A03,
        0x0406'0502,  // v_div_scale_f32 v3, vcc_lo, v2, v2, v1
        0xD56D'1008,
        0x0406'0501,  // v_div_scale_f32 v8, s16, v1, v2, v1
        0x7E08'5503,  // v_rcp_f32 v4, v3
        0xD54B'0005,
        0x23CA'0903,  // v_fma_f32 v5, -v3, v4, 1.0
        0xD54B'0004,
        0x0412'0905,  // v_fma_f32 v4, v5, v4, v4
        0x100A'0908,  // v_mul_f32 v5, v8, v4
        0xD54B'0006,
        0x2422'0B03,  // v_fma_f32 v6, -v3, v5, v8
        0xD56F'0004,
        0x0416'0906,  // v_div_fmas_f32 v4, v6, v4, v5
        0xD55F'0001,
        0x0406'0504,  // v_div_fixup_f32 v1, v4, v2, v1
        0x340E'0082,  // v_lshlrev_b32 v7, 2, v0
        0xE070'1000,
        0x8003'0107,  // buffer_store_dword v1, v7, s[12:15], 0 offen
        0xBF81'0000,  // s_endpgm
    };
    const NativeProgram native = program(
        HwStage::Cs, false, 9, {RootParam{RootType::Constants(), 0, 4, 0, 0, 0}, RootParam{RootType::Table(), 0, 1, 0, 4, 1}},
        {UsedResource{ResourceType::Srv(), RootType::Table(), 0, 0, 4, 4, 4, 1},
         UsedResource{ResourceType::Uav(), RootType::Table(), 8, 0, 4, 4, 4, 1}},
        ComputeIo{5u << 1, {7, 1, 1}}, words);
    const vs::ir::Program p = unwrap([&] { return vs::ir::build(native.code, native.wave64, native.vgprs); });
    const vs::resource::Resources res =
        unwrap([&] { return vs::resource::track(p, native.layout, native.stage, nullptr); });
    const vs::io::Io io = unwrap([&] { return vs::io::raise(p, native); });
    const vs::structure::Structured st = unwrap([&] { return vs::structure::structure(p); });
    vs::lower::Lowered l = unwrap([&] { return lower_compute(native, p, st, io, res); });
    std::vector<std::uint8_t> dxil = std::move(l.dxil);
    if (std::optional<Validator> v = Validator::load(Validator::default_path())) validate_in_place(*v, dxil);
    std::optional<vs::host::Host> host;
    try {
        host.emplace(false);
    } catch (const vs::host::Error& e) {
        // (Rust skips this silently.)
        std::fprintf(stderr, "no SM 6.6 device (%s); the dispatch is skipped\n", e.what());
        return;
    }
    HostShader cs = host_shader(HwStage::Cs, std::move(dxil));
    cs.user_data = l.user_data;
    cs.root_cbvs = l.root_cbvs;
    cs.root_srvs = l.root_srvs;
    cs.root_uavs = l.root_uavs;
    const auto [root, pso] = host->compute_pipeline(cs);
    std::vector<std::uint32_t> inputs;
    for (float f : {7.0f, 3.0f, -1.0f, 1e-40f, 1.0f, 1e35f, 1e-35f, 1e-36f, 0.0f, 0.0f, -1.0f, 0.0f, -1e35f, 1e35f})
        inputs.push_back(std::bit_cast<std::uint32_t>(f));
    const std::array<vs::host::View, 2> views = {vs::host::View::buffer(inputs, vs::host::Stride::raw()),
                                                 vs::host::View::raw_uav(64)};
    vs::host::Dispatch d;
    d.roots.emplace_back(0u, vs::host::Root::constants({0, 0, 0, 0, 0}));
    d.views = views;
    d.groups = {1, 1, 1};
    const std::vector<std::vector<std::uint32_t>> out = host->dispatch(root.Get(), pso.Get(), d);
    if (out.empty() || out[0].size() < 7) throw std::logic_error("the UAV holds fewer than 7 dwords");
    std::array<float, 7> got{};
    for (std::size_t k = 0; k < 7; ++k) got[k] = std::bit_cast<float>(out[0][k]);
    auto lane = [&](std::size_t k) {
        char b[96];
        std::snprintf(b, sizeof b, "lane %zu got %g (%#x)", k, static_cast<double>(got[k]), out[0][k]);
        return std::string(b);
    };
    // Computed at run time (Rust's f32 arithmetic; MSVC folds a constant overflow with a warning).
    volatile float minus_one = -1.0f;
    volatile float tiny = 1e-40f;
    volatile float zero = 0.0f;
    CHECK_MSG(std::fabs(got[0] - 7.0f / 3.0f) < 1e-6f, lane(0));
    CHECK(out[0][1] == std::bit_cast<std::uint32_t>(static_cast<float>(minus_one / tiny)));
    CHECK_MSG(std::fabs(got[2] - 1e-35f) <= 1e-40f, lane(2));
    CHECK_MSG(std::fabs(got[3] - 10.0f) < 1e-5f, lane(3));
    CHECK(out[0][4] == 0xFFC0'0000u);
    CHECK(out[0][5] == std::bit_cast<std::uint32_t>(static_cast<float>(minus_one / zero)));
    CHECK_MSG(std::fabs(got[6] - (-1.0f)) < 1e-6f, lane(6));
}

int run_failed = 0;

void run(const char* name, void (*test)()) {
    x2w::vs::lower::test::set_dump_name(std::string("emit__tests__") + name);
    const int before = test_util::failures();
    try {
        test();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: panicked: %s\n", name, e.what());
        ++test_util::failures();
    }
    if (test_util::failures() != before) {
        ++run_failed;
        std::fprintf(stderr, "%s: FAILED\n", name);
    }
}

}  // namespace

#define RUN(test) run(#test, test)

int main() {
    RUN(interpolated_f16_writes_the_op_sel_half_with_modifiers);
    RUN(the_clean_up_switch_parses_or_runs_every_pass);
    RUN(the_loop_limit_switch_parses_or_falls_back);
    RUN(a_stencil_export_declares_the_stencil_ref_feature);
    RUN(an_ordered_section_uses_rasterizer_ordered_views);
    RUN(immediate_sampler_slots_distinguish_comparison_uses);
    RUN(an_exec_high_write_does_not_overflow_the_stack);
    RUN(a_mask_whose_high_dword_is_rewritten_is_refused);
    RUN(a_backward_permute_reads_the_lane_its_address_names);
    RUN(a_carry_in_from_a_divergent_join_is_refused);
    RUN(stub_pipeline_is_created);
    RUN(constant_color_validates);
    RUN(quad_neighbours_at_a_triangles_edge_are_read);
    RUN(primitive_export_indices_drop_the_edge_flags);
    RUN(edge_flagged_triangles_draw_as_unflagged_ones);
    RUN(sample_interpolated_inputs_set_psv_sample_frequency);
    RUN(reading_a_channel_below_an_input_element_does_not_underflow);
    RUN(scalar_buffer_loads_read_the_same_dwords_one_call_or_four);
    RUN(valu_div_f32_sequence_executes_on_gpu);
    if (run_failed) std::fprintf(stderr, "mod_test: %d test(s) failed\n", run_failed);
    return test_util::finish("mod_test");
}
