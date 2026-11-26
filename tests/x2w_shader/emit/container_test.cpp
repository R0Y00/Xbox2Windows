// Port of the upstream Rust module emit/container.rs tests and emit/rect.rs tests (the ones without a device; the
// validator checks of the rectangle stage are in rect_test).

#include <array>
#include <cstdio>
#include <string>
#include <vector>

#include "test_util.h"
#include "x2w_shader/dxbc.h"
#include "x2w_shader/emit/container.h"
#include "x2w_shader/emit/rect.h"
#include "x2w_shader/emit/signature.h"
#include "x2w_shader/error.h"



using namespace x2w::vs;
using container::SigElement;

namespace {

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const ShaderError&) {
        return true;
    }
    return false;
}

void packed_vertex_outputs_preserve_name_order_and_declared_interpolation() {
    SigElement integer;
    integer.name = "TEXCOORD";
    integer.index = 1;
    integer.kind = 0;
    integer.comp = 5;
    integer.interp = 1;
    integer.row = 0;
    integer.col = 2;
    integer.cols = 2;
    integer.usage = 3;
    integer.half = false;
    SigElement flat_float = integer;
    flat_float.name = "COLOR";
    flat_float.index = 0;
    flat_float.comp = 9;
    flat_float.col = 0;
    SigElement smooth_float = integer;
    smooth_float.index = 2;
    smooth_float.row = 1;
    smooth_float.interp = 2;
    smooth_float.comp = 9;
    const std::vector<SigElement> outputs = {integer, flat_float, smooth_float};
    std::vector<SigElement> recovered = signature(container::signature(outputs, false));
    // DXC orders the row by semantic-name offset even though TEXCOORD occupies z/w and COLOR x/y.
    CHECK(recovered[0].name == "TEXCOORD");
    CHECK(recovered[1].name == "COLOR");
    // PSV records can be in a different order again.
    const std::vector<SigElement> reordered = {outputs[2], outputs[1], outputs[0]};
    container::Psv p;
    p.stage = 1;
    p.flag = false;
    p.outputs = reordered;
    const std::vector<std::uint8_t> psv = container::psv(p);
    recovered[1].interp = 2;
    vertex_output_interpolation(psv, recovered);
    CHECK(recovered.size() == 3 && recovered[0].interp == 1 && recovered[1].interp == 1 && recovered[2].interp == 2);
    CHECK(throws([&] {
        vertex_output_interpolation(std::span<const std::uint8_t>(psv.data(), psv.size() - 1), recovered);
    }));
}

void md5_core_matches_the_standard() {
    std::vector<std::uint8_t> block = {'a', 'b', 'c', 0x80};
    block.resize(56, 0);
    for (int k = 0; k < 8; ++k) block.push_back(std::uint8_t(std::uint64_t(24) >> (8 * k)));
    std::array<std::uint32_t, 4> state = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    container::md5_block_for_test(state, block.data());
    std::string hex;
    for (std::uint32_t s : state)
        for (int k = 0; k < 4; ++k) {
            char b[3];
            std::snprintf(b, sizeof b, "%02x", unsigned((s >> (8 * k)) & 0xff));
            hex += b;
        }
    CHECK(hex == "900150983cd24fb0d6963f7d28e17f72");
}

SigElement element(const char* name, std::uint32_t index, std::uint8_t kind, std::uint8_t comp, std::uint32_t row,
                   std::uint8_t cols) {
    SigElement e;
    e.name = name;
    e.index = index;
    e.kind = kind;
    e.comp = comp;
    e.interp = 0;
    e.row = row;
    e.col = 0;
    e.cols = cols;
    e.usage = 0;
    e.half = false;
    return e;
}

std::vector<SigElement> outputs() {
    return {element("SV_Position", 0, 3, 9, 0, 4), element("TEXCOORD", 0, 0, 9, 1, 4), element("TEXCOORD", 1, 0, 5, 2, 2)};
}

void the_rectangle_stage_is_a_geometry_shader() {
    const std::vector<SigElement> wanted = {element("SV_PrimitiveID", 0, 10, 5, 3, 1)};
    const std::vector<std::uint8_t> gs = rect_build(outputs(), wanted);
    const dxbc::Container c = dxbc::Container::parse(gs);
    auto data = c.part(dxbc::FourCC::DXIL)->data;
    // Program version: geometry (2), shader model 6.6.
    CHECK(data[0] == 0x66 && data[1] == 0 && data[2] == 2 && data[3] == 0);
    auto psv = c.part(dxbc::FourCC::PSV0)->data;
    const std::uint8_t a[] = {3, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 1};
    bool same = true;
    for (int k = 0; k < 13; ++k) same = same && psv[4 + k] == a[k];
    CHECK(same);
    const std::uint8_t b[] = {2, 0, 6, 0, 3, 4};
    same = true;
    for (int k = 0; k < 6; ++k) same = same && psv[28 + k] == b[k];
    CHECK(same);
    const auto ins = signature(c.part(dxbc::FourCC::ISG1)->data);
    const auto outs = signature(c.part(dxbc::FourCC::OSG1)->data);
    CHECK(ins.size() == 3);
    CHECK(outs.size() == 4);
    for (std::size_t k = 0; k < ins.size() && k < outs.size(); ++k) {
        CHECK(ins[k].name == outs[k].name && ins[k].index == outs[k].index && ins[k].row == outs[k].row);
        CHECK(ins[k].col == outs[k].col && ins[k].cols == outs[k].cols && ins[k].comp == outs[k].comp);
    }
    if (outs.size() == 4) CHECK(outs[3].kind == 10 && outs[3].row == std::optional<std::uint32_t>(3));
}

void rectangle_stages_that_cannot_be_made_are_refused() {
    auto o = outputs();
    CHECK(throws([&] { rect_build(std::vector<SigElement>(o.begin() + 1, o.end()), {}); }));
    const std::vector<SigElement> taken = {element("SV_PrimitiveID", 0, 10, 5, 1, 1)};
    CHECK(throws([&] { rect_build(outputs(), taken); }));
    std::vector<SigElement> many = {element("SV_Position", 0, 3, 9, 0, 4)};
    for (std::uint32_t row = 1; row <= 43; ++row) many.push_back(element("TEXCOORD", row - 1, 0, 9, row, 4));
    CHECK(throws([&] { rect_build(many, {}); }));
}

}  // namespace

int main() {
    packed_vertex_outputs_preserve_name_order_and_declared_interpolation();
    md5_core_matches_the_standard();
    the_rectangle_stage_is_a_geometry_shader();
    rectangle_stages_that_cannot_be_made_are_refused();
    return test_util::finish("container_test");
}
