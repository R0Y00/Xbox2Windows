// Byte-for-byte comparison with the upstream Rust emitter: the same fixed inputs as the reference dump
// (ref/ref_dump.rs, run in a temporary checkout of the upstream crate),
// whose outputs are read from the directory in X2W_REF_DIR (out/ for the default clean-up passes,
// out_opt0/ for X2W_SHADERCFG_SHADER_OPT=0; the test must run with the same setting).

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "x2w_shader/emit/container.h"
#include "x2w_shader/emit/rect.h"
#include "x2w_shader/emit/shader.h"
#include "test_util.h"

using namespace x2w::vs;
using namespace x2w::vs::bitcode;
using container::Binding;
using container::Psv;
using container::SigElement;
using shader::Shader;

namespace {

std::string dir() {
    const char* d = std::getenv("X2W_REF_DIR");
    return d ? d : "";
}

std::vector<std::uint8_t> read(const std::string& name) {
    std::ifstream f(dir() + "/" + name + ".bin", std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

void same(const char* name, const std::vector<std::uint8_t>& got) {
    const auto want = read(name);
    if (want.empty()) {
        std::fprintf(stderr, "%s: no reference\n", name);
        CHECK(!"reference missing");
        return;
    }
    if (got == want) {
        std::printf("%s: identical (%zu bytes)\n", name, got.size());
        return;
    }
    std::size_t at = 0;
    while (at < got.size() && at < want.size() && got[at] == want[at]) ++at;
    std::fprintf(stderr, "%s: DIFFERS (got %zu bytes, want %zu; first difference at %zu)\n", name, got.size(),
                 want.size(), at);
    std::string path = dir() + "/" + name + ".cpp.bin";
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(got.data()), std::streamsize(got.size()));
    CHECK(!"bytes differ");
}

SigElement el(const char* name, std::uint32_t index, std::uint8_t kind, std::uint8_t comp, std::optional<std::uint32_t> row,
              std::uint8_t col, std::uint8_t cols, std::uint8_t interp, std::uint8_t usage) {
    SigElement e;
    e.name = name;
    e.index = index;
    e.kind = kind;
    e.comp = comp;
    e.interp = interp;
    e.row = row;
    e.col = col;
    e.cols = cols;
    e.usage = usage;
    return e;
}

void containers() {
    const std::vector<SigElement> ins = {
        el("SV_Position", 0, 3, 9, 0, 0, 4, 4, 0xf),   el("TEXCOORD", 0, 0, 9, 1, 0, 2, 2, 3),
        el("TEXCOORD", 1, 0, 5, 1, 2, 2, 1, 1),        el("COLOR", 0, 0, 9, 2, 0, 4, 2, 0),
        el("SV_IsFrontFace", 0, 13, 5, 3, 0, 1, 1, 1), el("SV_SampleIndex", 0, 12, 5, std::nullopt, 0, 1, 0, 1),
    };
    const std::vector<SigElement> outs = {
        el("SV_Target", 0, 16, 9, 0, 0, 4, 0, 0xf),
        el("SV_Target", 1, 16, 5, 1, 0, 2, 0, 3),
        el("SV_Depth", 0, 17, 9, std::nullopt, 0, 1, 0, 1),
    };
    same("sig_in", container::signature(ins, true));
    same("sig_out", container::signature(outs, false));
    const Binding bindings[] = {{2, 0, 0, 13}, {1, 0, 3, 14}, {4, 1, 7, 11}, {7, 1, 9, 11}};
    {
        Psv p;
        p.stage = 0;
        p.flag = true;
        p.inputs = ins;
        p.outputs = outs;
        p.bindings = bindings;
        same("psv_ps", container::psv(p));
    }
    const std::vector<SigElement> prims = {el("PRIMDATA", 0, 0, 5, 1, 0, 3, 1, 7)};
    {
        shader::Mesh mesh{64, 126, 2};
        Psv p;
        p.stage = 13;
        p.outputs = std::span<const SigElement>(ins).first(4);
        p.prims = prims;
        p.mesh = &mesh;
        p.mesh_lds_bytes = 4096;
        p.threads = {128, 1, 1};
        same("psv_ms", container::psv(p));
    }
    {
        Psv p;
        p.stage = 5;
        p.bindings = std::span<const Binding>(bindings).subspan(2);
        p.threads = {8, 8, 1};
        p.wave_size = 64;
        same("psv_cs", container::psv(p));
    }
    {
        Psv p;
        p.stage = 2;
        p.flag = true;
        p.inputs = std::span<const SigElement>(ins).first(3);
        p.outputs = std::span<const SigElement>(ins).first(3);
        p.geometry = 6;
        same("psv_gs", container::psv(p));
    }
    std::vector<std::uint8_t> hashes;
    for (std::size_t n = 0; n <= 200; ++n) {
        std::vector<std::uint8_t> data(n);
        for (std::size_t i = 0; i < n; ++i) data[i] = std::uint8_t(i * 7 + 3);
        auto h = container::hash(data);
        hashes.insert(hashes.end(), h.begin(), h.end());
    }
    same("hashes", hashes);
    const std::uint8_t a[3] = {1, 2, 3};
    std::uint8_t b[37];
    std::fill(std::begin(b), std::end(b), std::uint8_t(9));
    const container::Part parts[] = {
        {{'S', 'F', 'I', '0'}, a}, {{'X', 'X', 'X', 'X'}, b}, {{'E', 'M', 'P', 'T'}, {}}};
    same("build", container::build(parts));
}

std::vector<std::uint8_t> cs_basic() {
    Shader s(false);
    s.threads = std::array<std::uint32_t, 3>{64, 1, 1};
    s.wave_size = 32;
    s.lds_dwords = 256;
    s.user_data = 6;
    const std::size_t b = s.entry;
    const Ty i32t = s.t.i32;
    V ud = s.user_data_handle();
    V zero = s.m.i32(0);
    V row = s.op(b, "cbufferLoadLegacy.i32", Attr::ReadOnly, s.t.cbuf_ret, 59, {ud, zero});
    V x = s.m.extract(b, row, 1);
    V tid = s.op(b, "threadId.i32", Attr::ReadNone, i32t, 93, {zero});
    V four = s.m.i32(4);
    V off = s.m.bin(b, Bin::Mul, tid, four);
    V srv = s.root_srv_handle(3);
    V undef = s.m.undef(i32t);
    V one8 = s.m.i8(1);
    V ld = s.op(b, "rawBufferLoad.f32", Attr::ReadOnly, s.t.res_ret, 139, {srv, off, undef, one8, four});
    V vf = s.m.extract(b, ld, 0);
    V v = s.m.cast(b, Cast::Bitcast, vf, i32t);
    V sum = s.m.bin(b, Bin::Add, v, x);
    // A dead value and a constant fold for the clean-up.
    V k = s.m.bin(b, Bin::Add, zero, four);
    (void)s.m.bin(b, Bin::Mul, sum, k);
    V lds = s.lds();
    V k255 = s.m.i32(255);
    V idx = s.m.bin(b, Bin::And, tid, k255);
    V p = s.m.element(b, lds, idx);
    s.m.store(b, p, sum);
    V p2 = s.m.element(b, lds, idx);
    V back = s.m.load(b, p2);
    V old = s.m.atomic(b, Rmw::Add, p2, back);
    V exch = s.m.cmpxchg(b, p2, old, back);
    // A loop: i from 0 while i < x.
    std::size_t head = s.m.new_block();
    std::size_t body = s.m.new_block();
    std::size_t exit = s.m.new_block();
    s.m.terminate(b, Term::br(head));
    V i = s.m.phi(head, i32t);
    V acc = s.m.phi(head, i32t);
    V c = s.m.cmp(head, pred::IULT, i, x);
    s.m.terminate(head, Term::cond_br(c, body, exit));
    V one = s.m.i32(1);
    V i2 = s.m.bin(body, Bin::Add, i, one);
    V zero8 = s.m.i8(0);
    V wsum = s.op(body, "waveActiveOp.i32", Attr::NoUnwind, i32t, 119, {acc, zero8, one8});
    V acc2 = s.m.bin(body, Bin::Xor, wsum, exch);
    s.m.terminate(body, Term::br(head));
    s.m.set_phi(i, {{zero, b}, {i2, body}});
    s.m.set_phi(acc, {{sum, b}, {acc2, body}});
    s.flags |= shader::flags::WAVE_OPS;
    V two = s.m.i32(2);
    V heap_index = s.m.bin(exit, Bin::LShr, acc, two);
    V h = s.heap_handle(exit, heap_index, false, true, {11 | 0x1000, 0});
    V f = s.m.cast(exit, Cast::Bitcast, acc, s.t.f32);
    V fi = s.m.cast(exit, Cast::FpToUi, f, i32t);
    s.op(exit, "rawBufferStore.i32", Attr::NoUnwind, s.t.void_, 140, {h, off, undef, fi, undef, undef, undef, one8, four});
    V uav = s.root_uav_handle(5);
    V sel = s.m.select(exit, c, fi, acc);
    s.op(exit, "rawBufferStore.i32", Attr::NoUnwind, s.t.void_, 140, {uav, off, undef, sel, undef, undef, undef, one8, four});
    V cbv = s.root_cbv_handle(2);
    V crow = s.op(exit, "cbufferLoadLegacy.i32", Attr::ReadOnly, s.t.cbuf_ret, 59, {cbv, zero});
    V cx = s.m.extract(exit, crow, 2);
    s.op(exit, "rawBufferStore.i32", Attr::NoUnwind, s.t.void_, 140, {uav, cx, undef, cx, undef, undef, undef, one8, four});
    s.m.terminate(exit, Term::ret());
    return s.finish();
}

std::vector<std::uint8_t> ps_basic() {
    Shader s(true);
    s.inputs = {
        el("SV_Position", 0, 3, 9, 0, 0, 4, 4, 0x3),
        el("TEXCOORD", 0, 0, 9, 1, 0, 2, 2, 3),
        el("TEXCOORD", 1, 0, 5, 3, 0, 1, 1, 0),
        el("COLOR", 0, 0, 9, 2, 0, 4, 2, 0),
    };
    s.outputs = {el("SV_Target", 0, 16, 9, 0, 0, 4, 0, 0xf)};
    s.samplers = 2;
    s.cmp_samplers = 2;
    s.user_data = 1;
    const std::size_t b = s.entry;
    const Ty f32t = s.t.f32;
    std::vector<V> uv;
    for (std::uint8_t col = 0; col < 2; ++col) {
        Ty ty = s.t.i32;
        const V args[4] = {s.m.i32(1), s.m.i32(0), s.m.i8(col), s.m.undef(ty)};
        uv.push_back(s.op(b, "loadInput.f32", Attr::ReadNone, f32t, 4, args));
    }
    V ud = s.user_data_handle();
    V zero = s.m.i32(0);
    V row = s.op(b, "cbufferLoadLegacy.i32", Attr::ReadOnly, s.t.cbuf_ret, 59, {ud, zero});
    V slot = s.m.extract(b, row, 0);
    V tex = s.heap_handle(b, slot, false, false, {2 | (9 << 8), 0x0409});
    V smp = s.sampler_handle(b, 0);
    V undef = s.m.undef(f32t);
    Ty i32t = s.t.i32;
    V ui = s.m.undef(i32t);
    V r = s.op(b, "sample.f32", Attr::ReadOnly, s.t.res_ret, 60, {tex, smp, uv[0], uv[1], undef, undef, ui, ui, ui, undef});
    V half = s.m.f32(0x3f000000);
    for (std::uint32_t col = 0; col < 4; ++col) {
        V v = s.m.extract(b, r, col);
        V m = s.m.bin(b, Bin::Mul, v, half);
        const V args[4] = {s.m.i32(0), s.m.i32(0), s.m.i8(std::uint8_t(col)), m};
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void_, 5, args);
    }
    s.m.terminate(b, Term::ret());
    return s.finish();
}

std::vector<SigElement> rect_outputs() {
    return {
        el("SV_Position", 0, 3, 9, 0, 0, 4, 0, 0),
        el("TEXCOORD", 0, 0, 9, 1, 0, 4, 0, 0),
        el("TEXCOORD", 1, 0, 5, 2, 0, 2, 0, 0),
    };
}

}  // namespace

int main() {
    if (dir().empty()) {
        std::fprintf(stderr, "ref_test: set X2W_REF_DIR\n");
        return 1;
    }
    containers();
    same("cs_basic", cs_basic());
    same("ps_basic", ps_basic());
    const SigElement wanted[] = {el("SV_PrimitiveID", 0, 10, 5, 3, 0, 1, 0, 0)};
    same("rect_prim", rect_build(rect_outputs(), wanted));
    same("rect_plain", rect_build(rect_outputs(), {}));
    return test_util::finish("ref_test");
}
