// Port of the upstream Rust shader tests in src/native.rs.
//
// The Rust tests compress with zlib-rs; the repository has no deflate compressor, so xbch() here
// writes a zlib stream of stored blocks (header, stored blocks, Adler-32), which inflates the same.
#include "x2w_shader/native.h"

#include "vs_front_test.h"

#include <cstring>

#include "x2w_shader/dxbc.h"

using namespace x2w::vs;
using namespace x2w::vs::native;

namespace {

using Buf = std::vector<std::uint8_t>;

constexpr std::size_t XDAT_OFFSET = 0x18;
constexpr std::uint32_t KIND_NATIVE = 0x0d;
constexpr std::uint32_t XBCH_VERSION = 3;
constexpr std::size_t XBCH_HEADER = 0x18;
constexpr std::uint32_t FSSM_VERSION = 0x101;
constexpr std::size_t FSSM_HEADER = 0x10;
constexpr std::uint64_t GRAPHICS_SLOT_COUNT = 6;
constexpr std::pair<HwStage, std::size_t> SLOTS[8] = {
    {HwStage::Ls, 0x08}, {HwStage::Hs, 0x10}, {HwStage::Es, 0x18}, {HwStage::Gs, 0x20},
    {HwStage::Vs, 0x28}, {HwStage::Ps, 0x30}, {HwStage::Ms, 0x48}, {HwStage::Cs, 0x60},
};
constexpr std::uint64_t ABSENT = 0xDEAD'BEEF'DEAD'BEEFull;
constexpr std::size_t RECORD_SIZE = 0x54;
constexpr std::uint32_t FLAG_WAVE32 = 1u << 5;
constexpr std::size_t RECORD_REGS = 0x50;
constexpr std::size_t RECORD_LAYOUT = 0x28;
constexpr std::size_t RECORD_USED = 0x30;
constexpr std::size_t RECORD_COUNTS = 0x38;
constexpr std::uint64_t ENTRY_SIZE = 20;
constexpr std::size_t PS_INPUTS = 0x24;
constexpr std::size_t PS_INPUT_COUNT = 0x124;
constexpr std::size_t PS_INPUT_ADDR = 0x134;
constexpr std::size_t PS_PATCH_SIZE = 0x148;
constexpr std::size_t PS_PATCHES = 0x14c;
constexpr std::size_t PATCH_SLOT_DWORDS = 10;
constexpr std::size_t PATCH_FULL = 9;
constexpr std::size_t PATCH_BYTES = 4 + 10 * PATCH_SLOT_DWORDS * 4;
constexpr std::size_t MRTS = 8;
constexpr std::size_t VS_INPUT_COUNT = 0xc4;
constexpr std::size_t VS_OUTPUTS = 0xc8;
constexpr std::size_t VS_OUTPUT_COUNT = 0x188;
constexpr std::size_t RECORD_LDS = 0x4c;
constexpr std::size_t MS_CLIP_CNTL = 0x194;
constexpr std::size_t MS_INFO = 0x448;

const Buf VS_CODE{1, 2, 3, 4, 5, 6, 7, 8};
const Buf PS_CODE{9, 10, 11, 12};
constexpr std::uint32_t VGPRS = 28;

std::size_t slot(HwStage stage)
{
    for (const auto& s : SLOTS)
        if (s.first == stage) return s.second;
    return 0;
}

void put32(Buf& b, std::size_t at, std::uint32_t v)
{
    for (int i = 0; i < 4; ++i) b[at + i] = static_cast<std::uint8_t>(v >> (8 * i));
}
void put64(Buf& b, std::size_t at, std::uint64_t v)
{
    for (int i = 0; i < 8; ++i) b[at + i] = static_cast<std::uint8_t>(v >> (8 * i));
}
void push32(Buf& b, std::uint32_t v)
{
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
void push64(Buf& b, std::uint64_t v)
{
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
void push_bytes(Buf& b, std::initializer_list<std::uint8_t> v) { b.insert(b.end(), v.begin(), v.end()); }
void set_bytes(Buf& b, std::size_t at, std::initializer_list<std::uint8_t> v)
{
    std::size_t i = 0;
    for (auto x : v) b[at + i++] = x;
}
std::uint64_t get64(const Buf& b, std::size_t at)
{
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= std::uint64_t{b[at + i]} << (8 * i);
    return v;
}

// Stage data: no semantics, and every export patch absent.
constexpr std::size_t STAGE_DATA = PS_PATCHES + MRTS * PATCH_BYTES;

Buf empty_stage_data()
{
    Buf data(STAGE_DATA, 0);
    for (std::size_t mrt = 0; mrt < MRTS; ++mrt) put32(data, PS_PATCHES + mrt * PATCH_BYTES, 0xFFFF'FFFFu);
    return data;
}

// Where a one-program blob's stage data starts.
std::size_t stage_data_at(const Buf& blob, HwStage stage)
{
    return FSSM_HEADER + static_cast<std::size_t>(get64(blob, FSSM_HEADER + slot(stage))) + RECORD_SIZE;
}

// FSSM blob with `programs` as (stage, code) pairs.
Buf fssm(const std::vector<std::pair<HwStage, Buf>>& programs)
{
    Buf payload;
    push64(payload, GRAPHICS_SLOT_COUNT);
    payload.resize(slot(HwStage::Cs) + 8, 0);
    for (const auto& s : SLOTS) put64(payload, s.second, ABSENT);
    for (const auto& [stage, code] : programs) {
        const std::size_t at = slot(stage);
        const std::uint64_t record = payload.size();
        put64(payload, at, record);
        const std::uint64_t code_at = record + RECORD_SIZE + STAGE_DATA;
        push64(payload, code_at);
        push64(payload, code.size());
        const std::uint32_t flags = stage == HwStage::Ps ? FLAG_WAVE32 : 0;
        push32(payload, 0);
        push32(payload, flags);
        payload.resize(record + RECORD_REGS, 0);
        push32(payload, VGPRS | (31u << 20));
        const Buf data = empty_stage_data();
        payload.insert(payload.end(), data.begin(), data.end());
        payload.insert(payload.end(), code.begin(), code.end());
    }
    Buf out{'F', 'S', 'S', 'M'};
    push32(out, FSSM_VERSION);
    push64(out, payload.size());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

std::uint32_t adler32(const Buf& data)
{
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t x : data) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

// A zlib stream of stored blocks.
Buf zlib_stored(const Buf& data)
{
    Buf out{0x78, 0x01};
    std::size_t at = 0;
    do {
        const std::size_t n = std::min<std::size_t>(data.size() - at, 65535);
        const bool last = at + n == data.size();
        out.push_back(last ? 1 : 0);
        out.push_back(static_cast<std::uint8_t>(n));
        out.push_back(static_cast<std::uint8_t>(n >> 8));
        out.push_back(static_cast<std::uint8_t>(~n));
        out.push_back(static_cast<std::uint8_t>(~n >> 8));
        out.insert(out.end(), data.begin() + at, data.begin() + at + n);
        at += n;
    } while (at < data.size());
    const std::uint32_t adler = adler32(data);
    for (int i = 3; i >= 0; --i) out.push_back(static_cast<std::uint8_t>(adler >> (8 * i)));
    return out;
}

Buf xbch(const Buf& unpacked)
{
    const Buf stream = zlib_stored(unpacked);
    Buf out{'X', 'B', 'C', 'H'};
    push32(out, XBCH_VERSION);
    push64(out, stream.size());
    push64(out, unpacked.size());
    out.insert(out.end(), stream.begin(), stream.end());
    return out;
}

// `DXIL` part data: program header, XDAT index, then the blobs.
Buf part(const std::vector<std::pair<std::uint32_t, Buf>>& blobs)
{
    Buf out(XDAT_OFFSET, 0);
    push_bytes(out, {'X', 'D', 'A', 'T'});
    push32(out, static_cast<std::uint32_t>(blobs.size()));
    push32(out, 0x24);
    std::size_t offset = out.size() + 12 * blobs.size();
    for (const auto& [kind, blob] : blobs) {
        push32(out, kind);
        push32(out, static_cast<std::uint32_t>(offset));
        push32(out, static_cast<std::uint32_t>(blob.size()));
        offset += blob.size();
    }
    for (const auto& b : blobs) out.insert(out.end(), b.second.begin(), b.second.end());
    return out;
}

bool fails(const Buf& data)
{
    return catch_shader_error([&] { native_programs(data); }).has_value();
}

const Buf XBOP{'x', 'b', 'o', 'p'};

}  // namespace

VS_TEST(finds_graphics_programs)
{
    const Buf native = xbch(fssm({{HwStage::Vs, VS_CODE}, {HwStage::Ps, PS_CODE}}));
    const Buf data = part({{0x0a, XBOP}, {KIND_NATIVE, native}});
    const auto found = native_programs(data);
    NativeProgram vs;
    vs.stage = HwStage::Vs;
    vs.wave64 = true;
    vs.vgprs = 28;
    vs.io = VsIo{};
    vs.code = VS_CODE;
    NativeProgram ps;
    ps.stage = HwStage::Ps;
    ps.wave64 = false;
    ps.vgprs = 28;
    ps.io = PsIo{};
    ps.code = PS_CODE;
    VS_CHECK((found == std::vector<NativeProgram>{vs, ps}));
}

VS_TEST(reads_root_tables)
{
    Buf blob = fssm({{HwStage::Ps, PS_CODE}});
    const std::size_t record = FSSM_HEADER + static_cast<std::size_t>(get64(blob, FSSM_HEADER + slot(HwStage::Ps)));
    const std::uint64_t tables = blob.size() - FSSM_HEADER;
    // A table range in s9 (root slot 1), and an SRV read from it.
    for (std::uint32_t v : {3u, 2u, 1u, 5u}) push32(blob, v);
    push_bytes(blob, {9, 1, 0, 0, 0, 3, 0, 0});
    for (std::uint32_t v : {2u, 5u, 4u}) push32(blob, v);
    push_bytes(blob, {8, 9, 1, 0});
    const std::uint64_t payload = blob.size() - FSSM_HEADER;
    put64(blob, 8, payload);
    put64(blob, record + RECORD_LAYOUT, tables);
    put64(blob, record + RECORD_USED, tables + ENTRY_SIZE);
    set_bytes(blob, record + RECORD_COUNTS, {1, 0, 0, 0, 1, 0, 0, 0});

    const auto found = native_programs(part({{KIND_NATIVE, xbch(blob)}}));
    VS_REQUIRE(!found.empty());
    const NativeProgram& p = found[0];
    RootParam r;
    r.root = RootType::Table();
    r.table_offset_dwords = 2;
    r.sgprs = 1;
    r.base_register = 5;
    r.user_data = 9;
    r.slot = 1;
    VS_CHECK(p.layout == std::vector<RootParam>{r});
    UsedResource u;
    u.kind = ResourceType::Srv();
    u.root = RootType::Table();
    u.table_offset_dwords = 2;
    u.lower_bound = 5;
    u.stride = 4;
    u.dwords = 8;
    u.user_data = 9;
    u.slot = 1;
    VS_CHECK(p.used == std::vector<UsedResource>{u});

    blob[record + RECORD_COUNTS] = 9;
    VS_CHECK(fails(part({{KIND_NATIVE, xbch(blob)}})));
}

VS_TEST(reads_pixel_io_and_applies_the_full_export_patch)
{
    Buf bytes;
    for (int i = 0; i < 4; ++i) push32(bytes, 0xBF80'0000u);  // s_nop
    Buf blob = fssm({{HwStage::Ps, bytes}});
    const std::size_t e = stage_data_at(blob, HwStage::Ps);
    // TEXCOORD in row 1 (xy), read as attr0, flat.
    set_bytes(blob, e + PS_INPUTS, {6, 1, 3, 0, 0, 0, 1, 1});
    put32(blob, e + PS_INPUT_COUNT, 1);
    put32(blob, e + PS_INPUT_ADDR, 2);
    put32(blob, e + PS_PATCH_SIZE, 2);
    // MRT1 at dword 1: the 32_ABGR variant is `exp mrt1 v0-v3 done`.
    const std::size_t patch = e + PS_PATCHES + PATCH_BYTES;
    put32(blob, patch, 1);
    const std::size_t full = patch + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
    set_bytes(blob, full, {0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3});

    const auto found = native_programs(part({{KIND_NATIVE, xbch(blob)}}));
    VS_REQUIRE(!found.empty());
    const NativeProgram& p = found[0];
    const auto* ps = std::get_if<PsIo>(&p.io);
    VS_REQUIRE(ps != nullptr);
    PsInput input;
    input.usage = Usage::Generic();
    input.row = 1;
    input.mask = 3;
    input.attr = 0;
    input.flat = true;
    VS_CHECK(ps->inputs == std::vector<PsInput>{input});
    VS_CHECK(ps->input_addr == 2);
    VS_REQUIRE(ps->patches.size() == 1);
    VS_CHECK(ps->patches[0].mrt == 1 && ps->patches[0].offset == 1);
    const Buf patched = p.patched_code();
    VS_CHECK(std::equal(patched.begin(), patched.begin() + 4, p.code.begin()));
    VS_CHECK((Buf(patched.begin() + 4, patched.begin() + 12) == Buf{0x1f, 0x18, 0, 0xf8, 0, 1, 2, 3}));
    VS_CHECK(std::equal(patched.begin() + 12, patched.end(), p.code.begin() + 12));

    put32(blob, patch, 3);
    VS_CHECK(fails(part({{KIND_NATIVE, xbch(blob)}})));
}

VS_TEST(finds_mesh_and_compute_programs)
{
    for (HwStage stage : {HwStage::Ms, HwStage::Cs}) {
        const auto found = native_programs(part({{KIND_NATIVE, xbch(fssm({{stage, VS_CODE}}))}}));
        VS_REQUIRE(found.size() == 1);
        VS_CHECK(found[0].stage == stage);
    }
}

VS_TEST(reads_mesh_outputs_and_limits)
{
    Buf blob = fssm({{HwStage::Ms, VS_CODE}});
    const std::size_t e = stage_data_at(blob, HwStage::Ms);
    // param0 is row 1 per vertex, param2 row 0 of the primitive signature.
    set_bytes(blob, e + VS_OUTPUTS, {6, 1, 0, 0, 0, 0, 6, 0, 0x82, 0, 0, 0});
    put32(blob, e + VS_OUTPUT_COUNT, 2);
    put32(blob, e + MS_CLIP_CNTL, 0x0040'000Fu);
    // 128 lanes, 128 vertices, 64 primitives, 13 user SGPRs, triangles.
    set_bytes(blob, e + MS_INFO, {0x80, 0, 0x80, 0, 0x40, 0, 13, 2});
    const std::size_t record = e - RECORD_SIZE;
    put32(blob, record + RECORD_LDS, 16);
    const auto found = native_programs(part({{KIND_NATIVE, xbch(blob)}}));
    VS_REQUIRE(!found.empty());
    auto out = [](std::uint8_t row, std::uint8_t param) {
        VsOutput o;
        o.usage = Usage::Generic();
        o.row = row;
        o.param = param;
        return o;
    };
    MeshIo want;
    want.outputs = {out(1, 0), out(0, 0x82)};
    want.clip_cntl = 0x0040'000F;
    want.threads = 128;
    want.max_vertices = 128;
    want.max_primitives = 64;
    want.topology = 2;
    want.lds_bytes = 16;
    VS_CHECK(found[0].io == StageIo{want});
}

VS_TEST(mesh_stage_io_ignores_unused_vs_input_count)
{
    Buf blob = fssm({{HwStage::Ms, VS_CODE}});
    const std::size_t e = stage_data_at(blob, HwStage::Ms);
    // Mesh stage data has no vertex inputs; E+0xc4 is unused by MS.
    put32(blob, e + VS_INPUT_COUNT, 0xFFFF'FFFFu);
    const auto found = native_programs(part({{KIND_NATIVE, xbch(blob)}}));
    VS_REQUIRE(!found.empty());
    VS_CHECK(std::holds_alternative<MeshIo>(found[0].io));
}

VS_TEST(compute_dispatch_settings)
{
    // 5 user SGPRs, group id x and y, thread ids in v0-v1, 1 KiB LDS.
    ComputeIo cs{0x1098A, {16, 16, 1}};
    VS_CHECK(cs.user_sgprs() == 5);
    auto [ids, info] = cs.system_sgprs();
    VS_CHECK(ids[0] == std::optional<std::uint8_t>(std::uint8_t{5}) && ids[1] == std::optional<std::uint8_t>(std::uint8_t{6}) && !ids[2] && !info);
    VS_CHECK(cs.thread_id_vgprs() == 2 && cs.lds_bytes() == 1024);
    // 6 user SGPRs, all three group ids, v0-v2, no LDS.
    cs = ComputeIo{0x138C, {8, 8, 1}};
    auto [ids2, info2] = cs.system_sgprs();
    VS_CHECK(ids2[0] == std::optional<std::uint8_t>(std::uint8_t{6}) && ids2[1] == std::optional<std::uint8_t>(std::uint8_t{7}) &&
             ids2[2] == std::optional<std::uint8_t>(std::uint8_t{8}) && !info2);
    VS_CHECK(cs.thread_id_vgprs() == 3 && cs.lds_bytes() == 0);
}

VS_TEST(initialized_sgprs_covers_user_system_and_scratch)
{
    VS_CHECK((ComputeIo{0x1098A, {16, 16, 1}}.initialized_sgprs() == 7));
    VS_CHECK((ComputeIo{0x78C, {8, 8, 1}}.initialized_sgprs() == 10));
    VS_CHECK((ComputeIo{0x78D, {8, 8, 1}}.initialized_sgprs() == 11));
}

VS_TEST(reads_every_native_blob)
{
    const Buf es = xbch(fssm({{HwStage::Es, VS_CODE}}));
    const Buf vs = xbch(fssm({{HwStage::Vs, VS_CODE}}));
    const Buf data = part({{0x12, Buf(16, 0)}, {KIND_NATIVE, es}, {KIND_NATIVE, vs}});
    std::vector<HwStage> stages;
    for (const auto& p : native_programs(data)) stages.push_back(p.stage);
    VS_CHECK((stages == std::vector<HwStage>{HwStage::Es, HwStage::Vs}));
}

VS_TEST(pc_layout_has_none)
{
    VS_CHECK(native_programs(Buf(64, 0)).empty());
    VS_CHECK(native_programs(part({{0x0a, XBOP}})).empty());
}

VS_TEST(rejects_corrupt_blobs)
{
    const Buf good = xbch(fssm({{HwStage::Ls, VS_CODE}}));
    auto corrupt = [&](std::size_t at, std::uint8_t byte) {
        Buf blob = good;
        blob[at] = byte;
        return fails(part({{KIND_NATIVE, blob}}));
    };
    VS_CHECK(corrupt(4, 9));                  // unknown version
    VS_CHECK(corrupt(16, 0xFF));              // stated size
    VS_CHECK(corrupt(XBCH_HEADER + 2, 0xFF)); // zlib stream
    VS_CHECK(fails(part({{KIND_NATIVE, Buf(good.begin(), good.begin() + 20)}})));

    Buf bad_record = fssm({{HwStage::Ls, VS_CODE}});
    put64(bad_record, FSSM_HEADER + slot(HwStage::Ls), ~std::uint64_t{0});
    VS_CHECK(fails(part({{KIND_NATIVE, xbch(bad_record)}})));
}

VS_TEST(rejects_truncated_oversized_and_bad_checksum_xbch_streams)
{
    const Buf plain = fssm({{HwStage::Vs, VS_CODE}});
    const Buf good = xbch(plain);
    VS_CHECK(detail::unpack(good) == plain);
    auto unpack_fails = [](const Buf& b) { return catch_shader_error([&] { detail::unpack(b); }).has_value(); };

    // Truncated zlib stream (drop last byte, update packed length).
    Buf truncated(good.begin(), good.end() - 1);
    put64(truncated, 8, truncated.size() - XBCH_HEADER);
    VS_CHECK(unpack_fails(truncated));

    // Stated unpacked size smaller than the inflated output.
    Buf oversized = good;
    put64(oversized, 16, plain.size() - 1);
    VS_CHECK(unpack_fails(oversized));

    // Stated unpacked size larger than the inflated output.
    Buf underfilled = good;
    put64(underfilled, 16, plain.size() + 1);
    VS_CHECK(unpack_fails(underfilled));

    // Corrupt Adler-32 trailer.
    Buf bad_adler = good;
    bad_adler.back() ^= 1;
    VS_CHECK(unpack_fails(bad_adler));
}

VS_TEST_MAIN()
