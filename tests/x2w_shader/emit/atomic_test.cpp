// Port of the test in the upstream Rust module emit/mod.rs `mod atomic_tests` (rs:2408).
//
// A test that throws (a Rust panic or `.unwrap()` of an error) fails; dxil.dll checks the modules
// where it is installed, as in Rust.

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_util.h"
#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/emit.h"
#include "x2w_shader/emit/lower.h"
#include "x2w_shader/emit/validator.h"
#include "x2w_shader/error.h"
#include "x2w_shader/io.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/native.h"
#include "x2w_shader/resource.h"
#include "x2w_shader/structure.h"

namespace vs = x2w::vs;
using vs::dxbc::Container;
using vs::dxbc::FourCC;
using vs::native::ComputeIo;
using vs::native::HwStage;
using vs::native::NativeProgram;
using vs::native::RootParam;
using vs::native::RootType;
using vs::validator::Validator;

namespace {

std::vector<std::uint8_t> le_bytes(std::span<const std::uint32_t> words) {
    std::vector<std::uint8_t> code;
    code.reserve(words.size() * 4);
    for (std::uint32_t w : words)
        for (int k = 0; k < 4; ++k) code.push_back(static_cast<std::uint8_t>(w >> (8 * k)));
    return code;
}

// ---- rs:2413 ------------------------------------------------------------------------------------
void heap_texture_atomic_widths() {
    // A synthetic compute shader loads a 2D UAV from a root table, compares or adds at (0, 0), then
    // uses the returned register(s).
    struct Case {
        std::uint32_t opcode;
        std::uint32_t mask;
        bool wide;
    };
    for (const Case& k : {Case{16, 3, false}, Case{16, 15, true}, Case{17, 1, false}, Case{17, 3, true}}) {
        const std::uint32_t opcode = k.opcode;
        const std::uint32_t mask = k.mask;
        const bool wide = k.wide;
        const std::array<std::uint32_t, 14> words = {
            0xBE81'0380u,  // s_mov_b32 s1, 0 (table address high)
            0xF40C'0200,
            0xFA00'0000,  // s_load_dwordx8 s[8:15], s[0:1], 0
            0x7E00'0280,
            0x7E02'0280,  // v0, v1 = coordinates
            0x7E04'0281,
            0x7E06'0282,
            0x7E08'0283,
            0x7E0A'0284,  // v2..v5 = 1..4
            0xF000'2008u | opcode << 18 | mask << 8,
            0x0002'0200,
            // Use the old value in a second atomic at the same location.
            0xF044'0008u | (wide ? 3u : 1u) << 8,
            0x0002'0200,
            0xBF81'0000,
        };
        NativeProgram native;
        native.stage = HwStage::Cs;
        native.wave64 = false;
        native.vgprs = 6;
        native.layout = {RootParam{RootType::Table(), 0, 1, 0, 0, 0}};
        native.io = ComputeIo{2, {32, 1, 1}};
        native.code = le_bytes(words);
        const vs::ir::Program p = vs::ir::build(native.code, false, native.vgprs);
        const std::vector<std::uint8_t> bytes = vs::dxbc::build({});
        // emit's arguments in Rust order.
        const Container c = Container::parse(bytes);
        const vs::structure::Structured st = vs::structure::structure(p);
        const vs::io::Io io = vs::io::raise(p, native);
        const vs::resource::Resources res = vs::resource::track(p, native.layout, native.stage, nullptr);
        const vs::HostShader shader = vs::emit(c, native, p, st, io, res);
        const Container container = Container::parse(shader.dxil);
        const vs::dxbc::Part* sfi0 = container.part(FourCC::SFI0);
        if (!sfi0) throw std::logic_error("no SFI0 part");
        if (sfi0->data.size() != 8) throw std::logic_error("SFI0 is not 8 bytes");
        std::uint64_t features = 0;
        for (std::size_t b = 0; b < 8; ++b) features |= std::uint64_t(sfi0->data[b]) << (8 * b);
        const std::uint64_t required = 0x1040'8000;  // heap atomics, typed atomics, Int64Ops
        CHECK((features & required) == (wide ? required : 0));
        if (std::optional<Validator> v = Validator::load(Validator::default_path())) {
            std::vector<std::uint8_t> copy = shader.dxil;
            if (std::optional<std::string> e = v->validate(copy)) {
                std::fprintf(stderr, "  validator: %s\n", e->c_str());
                CHECK(!e);
            }
        }
    }
}

}  // namespace

int main() {
    x2w::vs::lower::test::set_dump_name("emit__atomic_tests__heap_texture_atomic_widths");
    try {
        heap_texture_atomic_widths();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "heap_texture_atomic_widths: panicked: %s\n", e.what());
        ++test_util::failures();
    }
    return test_util::finish("atomic_test");
}
