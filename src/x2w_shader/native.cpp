// Read native program layouts documented in native.h.
#include "x2w_shader/native.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "startup_stubs/inflate.h"
#include "x2w_shader/dxbc.h"

namespace x2w::vs::native {

namespace {

using Bytes = std::span<const std::uint8_t>;
using dxbc::u32_at;
using dxbc::u64_at;

constexpr std::size_t XDAT_OFFSET = 0x18;
constexpr std::size_t MAX_XDAT_ENTRIES = 64;
constexpr std::uint32_t KIND_NATIVE = 0x0d;
constexpr std::uint32_t XBCH_VERSION = 3;
constexpr std::size_t XBCH_HEADER = 0x18;
constexpr std::uint32_t FSSM_VERSION = 0x101;
constexpr std::size_t FSSM_HEADER = 0x10;
constexpr std::uint64_t GRAPHICS_SLOT_COUNT = 6;
// Where each stage's record offset sits in the FSSM payload.
constexpr std::pair<HwStage, std::size_t> SLOTS[8] = {
    {HwStage::Ls, 0x08}, {HwStage::Hs, 0x10}, {HwStage::Es, 0x18}, {HwStage::Gs, 0x20},
    {HwStage::Vs, 0x28}, {HwStage::Ps, 0x30}, {HwStage::Ms, 0x48}, {HwStage::Cs, 0x60},
};
constexpr std::uint64_t ABSENT = 0xDEAD'BEEF'DEAD'BEEFull;
constexpr std::uint64_t RECORD_SIZE = 0x54;
constexpr std::size_t RECORD_FLAGS = 0x14;
constexpr std::uint32_t FLAG_WAVE32 = 1u << 5;
constexpr std::size_t RECORD_REGS = 0x50;
constexpr std::size_t RECORD_LAYOUT = 0x28;
constexpr std::size_t RECORD_USED = 0x30;
constexpr std::size_t RECORD_COUNTS = 0x38;
constexpr std::uint64_t ENTRY_SIZE = 20;
constexpr std::uint32_t MAX_SEMANTICS = 32;
constexpr std::size_t PS_INPUTS = 0x24;
constexpr std::size_t PS_INPUT_COUNT = 0x124;
constexpr std::size_t PS_INPUT_ADDR = 0x134;
constexpr std::size_t PS_DB_SHADER_CONTROL = 0x138;
constexpr std::size_t PS_PATCH_SIZE = 0x148;
constexpr std::size_t PS_PATCHES = 0x14c;
constexpr std::size_t PATCH_SLOT_DWORDS = 10;
// The variant that exports every channel as a raw 32-bit value.
constexpr std::size_t PATCH_FULL = 9;
constexpr std::size_t PATCH_BYTES = 4 + 10 * PATCH_SLOT_DWORDS * 4;
constexpr std::size_t MRTS = 8;
constexpr std::size_t VS_INPUTS = 0x24;
constexpr std::size_t VS_INPUT_COUNT = 0xc4;
constexpr std::size_t VS_OUTPUTS = 0xc8;
constexpr std::size_t VS_OUTPUT_COUNT = 0x188;
constexpr std::size_t CS_RSRC2 = 0x24;
constexpr std::size_t CS_THREADS = 0x28;
constexpr std::size_t RECORD_LDS = 0x4c;
constexpr std::size_t MS_CLIP_CNTL = 0x194;
constexpr std::size_t MS_INFO = 0x448;
constexpr std::uint32_t FSPX_VERSION = 1;
constexpr std::size_t FSPX_HEADER = 0x10;
constexpr std::uint32_t FSPX_PACKET = 0;
constexpr std::uint32_t FSPX_CODE = 6;
constexpr std::uint32_t FSPX_METADATA = 7;
constexpr std::size_t PACKET_THREADS = 0x0c;
constexpr std::size_t PACKET_FLAGS = 0x12;
constexpr std::uint16_t PACKET_WAVE32 = 1u << 4;
constexpr std::size_t PACKET_RSRC1 = 0x14;
constexpr std::size_t PACKET_RSRC2 = 0x18;
constexpr std::size_t FSPX_RECORDS = 0x18;
constexpr std::size_t FSPX_RECORD = 32;
// Unpacked blobs are a few KiB; refuse to allocate for absurd sizes.
constexpr std::uint64_t MAX_UNPACKED = 16ull << 20;

bool starts_with(Bytes data, const char (&magic)[5])
{
    return data.size() >= 4 && std::memcmp(data.data(), magic, 4) == 0;
}

// `data.get(at..at + 4) == Some(magic)`.
bool magic_at(Bytes data, std::size_t at, const char (&magic)[5])
{
    return at <= data.size() && data.size() - at >= 4 && std::memcmp(data.data() + at, magic, 4) == 0;
}

// For entries already sized by `as_chunks`.
std::uint32_t le32(const std::uint8_t* entry, std::size_t at)
{
    return std::uint32_t{entry[at]} | (std::uint32_t{entry[at + 1]} << 8) | (std::uint32_t{entry[at + 2]} << 16) |
           (std::uint32_t{entry[at + 3]} << 24);
}

// `data[offset..offset + len]` with untrusted 64-bit bounds.
std::optional<Bytes> bytes(Bytes data, std::uint64_t offset, std::uint64_t len)
{
    if (offset > std::numeric_limits<std::size_t>::max() || len > std::numeric_limits<std::size_t>::max())
        return std::nullopt;
    const auto start = static_cast<std::size_t>(offset);
    const auto n = static_cast<std::size_t>(len);
    if (n > std::numeric_limits<std::size_t>::max() - start) return std::nullopt;
    if (start + n > data.size()) return std::nullopt;
    return data.subspan(start, n);
}

Bytes need(std::optional<Bytes> b, const char* why)
{
    if (!b) throw ShaderError::malformed(why);
    return *b;
}

std::uint16_t fspx_u16_at(Bytes data, std::size_t at)
{
    if (at > data.size() || data.size() - at < 2) throw ShaderError::malformed("FSPX data out of bounds");
    return static_cast<std::uint16_t>(data[at] | (data[at + 1] << 8));
}

ResourceType resource_type(std::uint8_t raw)
{
    switch (raw) {
    case 0: return ResourceType::Srv();
    case 1: return ResourceType::Sampler();
    case 2: return ResourceType::Cbv();
    case 3: return ResourceType::Uav();
    default: return ResourceType::Other(raw);
    }
}

// Inflates a zlib stream to exactly `expected` bytes, rejecting truncated, corrupt, under-filled or
// oversized streams. Bytes after the stream's end are ignored, as zlib does.
std::optional<std::vector<std::uint8_t>> inflate_zlib_exact(Bytes source, std::size_t expected)
{
    // zlib's window is at most 32 KiB (CINFO 7); zlib-rs with 15 window bits refuses larger ones.
    if (source.size() >= 1 && (source[0] >> 4) > 7) return std::nullopt;
    std::vector<std::uint8_t> out(expected);
    std::size_t produced = 0;
    const auto status =
        x2w_inflate::inflate_zlib_ex(source.data(), source.size(), out.data(), out.size(), produced);
    if (status != x2w_inflate::InflateStatus::ok || produced != expected) return std::nullopt;
    return out;
}

std::vector<std::uint8_t> unpack_blob(Bytes blob)
{
    if (!starts_with(blob, "XBCH")) throw ShaderError::malformed("native blob lacks XBCH magic");
    if (u32_at(blob, 4) != XBCH_VERSION) throw ShaderError::malformed("unknown XBCH version");
    const std::uint64_t packed = u64_at(blob, 8);
    const std::uint64_t unpacked = u64_at(blob, 16);
    if (unpacked > MAX_UNPACKED) throw ShaderError::malformed("implausible XBCH unpacked size");
    const Bytes stream = need(bytes(blob, XBCH_HEADER, packed), "XBCH stream out of bounds");
    auto out = inflate_zlib_exact(stream, static_cast<std::size_t>(unpacked));
    if (!out) throw ShaderError::malformed("XBCH stream does not inflate to its stated size");
    return std::move(*out);
}

// The entries of `size` bytes from `at`, with their count at `count_at`.
Bytes semantics(Bytes data, std::size_t at, std::size_t size, std::size_t count_at)
{
    const std::uint32_t count = u32_at(data, count_at);
    if (count > MAX_SEMANTICS) throw ShaderError::malformed("implausible semantic count");
    const std::size_t len = size * count;
    if (at > data.size() || data.size() - at < len) throw ShaderError::malformed("stage data out of bounds");
    return data.subspan(at, len);
}

StageIo ps_io(Bytes data, std::size_t code_len)
{
    PsIo io;
    const Bytes in = semantics(data, PS_INPUTS, 8, PS_INPUT_COUNT);
    for (std::size_t k = 0; k + 8 <= in.size(); k += 8) {
        const std::uint8_t* e = in.data() + k;
        PsInput i;
        i.usage = Usage::from_raw(e[0]);
        i.row = e[1];
        i.mask = e[2] & 0xf;
        i.attr = e[3];
        i.flat = (e[7] & 1) != 0;
        i.fp16 = (e[7] & 8) != 0;
        i.pack2 = (e[7] & 0x10) != 0;
        i.row2 = e[5];
        io.inputs.push_back(i);
    }
    const std::size_t size = u32_at(data, PS_PATCH_SIZE);
    if (size > PATCH_SLOT_DWORDS) throw ShaderError::malformed("implausible export patch size");
    for (std::size_t mrt = 0; mrt < MRTS; ++mrt) {
        const std::size_t at = PS_PATCHES + mrt * PATCH_BYTES;
        const std::uint32_t offset = u32_at(data, at);
        if (offset == std::numeric_limits<std::uint32_t>::max()) continue;
        if ((std::size_t{offset} + size) * 4 > code_len) throw ShaderError::malformed("export patch outside the code");
        const std::size_t full = at + 4 + PATCH_FULL * PATCH_SLOT_DWORDS * 4;
        ExportPatch patch;
        patch.mrt = static_cast<std::uint8_t>(mrt);
        patch.offset = offset;
        for (std::size_t i = 0; i < size; ++i) patch.code.push_back(u32_at(data, full + 4 * i));
        io.patches.push_back(std::move(patch));
    }
    io.input_addr = u32_at(data, PS_INPUT_ADDR);
    io.db_shader_control = u32_at(data, PS_DB_SHADER_CONTROL);
    return io;
}

std::vector<VsOutput> vs_outputs(Bytes data)
{
    std::vector<VsOutput> out;
    const Bytes s = semantics(data, VS_OUTPUTS, 6, VS_OUTPUT_COUNT);
    for (std::size_t k = 0; k + 6 <= s.size(); k += 6) {
        const std::uint8_t* e = s.data() + k;
        VsOutput o;
        o.usage = Usage::from_raw(e[0]);
        o.row = e[1];
        o.param = e[2];
        o.fp16 = (e[5] & 1) != 0;
        o.pack2 = (e[5] & 2) != 0;
        out.push_back(o);
    }
    return out;
}

StageIo vs_io(Bytes data)
{
    VsIo io;
    const Bytes s = semantics(data, VS_INPUTS, 5, VS_INPUT_COUNT);
    for (std::size_t k = 0; k + 5 <= s.size(); k += 5) {
        const std::uint8_t* e = s.data() + k;
        VsInput i;
        i.usage = Usage::from_raw(e[0]);
        i.row = e[1];
        i.mask = e[2] & 0xf;
        i.vgpr = e[3];
        io.inputs.push_back(i);
    }
    io.outputs = vs_outputs(data);
    return io;
}

StageIo ms_io(Bytes data, std::uint32_t lds_bytes)
{
    MeshIo io;
    io.outputs = vs_outputs(data);
    if (data.size() < MS_INFO + 8) throw ShaderError::malformed("stage data out of bounds");
    const std::uint8_t* info = data.data() + MS_INFO;
    auto half = [&](std::size_t at) { return std::uint32_t{static_cast<std::uint16_t>(info[at] | (info[at + 1] << 8))}; };
    io.clip_cntl = u32_at(data, MS_CLIP_CNTL);
    io.threads = half(0);
    io.max_vertices = half(2);
    io.max_primitives = half(4);
    io.topology = info[7];
    io.lds_bytes = lds_bytes;
    return io;
}

std::vector<NativeProgram> programs(Bytes fssm)
{
    if (!starts_with(fssm, "FSSM")) throw ShaderError::malformed("native blob lacks FSSM magic");
    if (u32_at(fssm, 4) != FSSM_VERSION) throw ShaderError::malformed("unknown FSSM version");
    const Bytes payload = fssm.subspan(std::min(FSSM_HEADER, fssm.size()));
    if (u64_at(fssm, 8) != payload.size()) throw ShaderError::malformed("FSSM payload size disagrees");
    if (u64_at(payload, 0) != GRAPHICS_SLOT_COUNT) throw ShaderError::malformed("unexpected FSSM slot count");
    std::vector<NativeProgram> out;
    for (const auto& [stage, at] : SLOTS) {
        const std::uint64_t start = u64_at(payload, at);
        if (start == ABSENT) continue;
        const Bytes record = need(bytes(payload, start, RECORD_SIZE), "FSSM record out of bounds");
        const std::uint64_t code_at = u64_at(record, 0);
        const std::uint64_t code_size = u64_at(record, 8);
        const Bytes code = need(bytes(payload, code_at, code_size), "native code out of bounds");
        const auto vgprs = static_cast<std::uint16_t>(u32_at(record, RECORD_REGS) & 0x3FF);
        if (vgprs > 256) throw ShaderError::malformed("implausible VGPR count");
        auto table = [&](std::size_t table_at, std::size_t count_at) -> Bytes {
            const std::uint64_t len = ENTRY_SIZE * std::uint64_t{u32_at(record, count_at)};
            if (len == 0) return {};
            const std::uint64_t offset = u64_at(record, table_at);
            return need(bytes(payload, offset, len), "FSSM root table out of bounds");
        };
        NativeProgram p;
        p.stage = stage;
        const Bytes layout = table(RECORD_LAYOUT, RECORD_COUNTS);
        for (std::size_t k = 0; k + ENTRY_SIZE <= layout.size(); k += ENTRY_SIZE) {
            const std::uint8_t* e = layout.data() + k;
            RootParam r;
            r.root = RootType::from_raw(le32(e, 0));
            r.table_offset_dwords = le32(e, 4);
            r.sgprs = le32(e, 8);
            r.base_register = le32(e, 12);
            r.user_data = e[16];
            r.slot = e[17];
            p.layout.push_back(r);
        }
        const Bytes used = table(RECORD_USED, RECORD_COUNTS + 4);
        for (std::size_t k = 0; k + ENTRY_SIZE <= used.size(); k += ENTRY_SIZE) {
            const std::uint8_t* e = used.data() + k;
            UsedResource u;
            u.kind = resource_type(e[0]);
            u.root = RootType::from_raw(e[1]);
            u.table_offset_dwords = le32(e, 4);
            u.lower_bound = le32(e, 8);
            u.stride = le32(e, 12);
            u.dwords = e[16];
            u.user_data = e[17];
            u.slot = e[18];
            p.used.push_back(u);
        }
        // The record is in bounds, so its end is too.
        const Bytes stage_data = payload.subspan(static_cast<std::size_t>(start + RECORD_SIZE));
        switch (stage) {
        case HwStage::Ps: p.io = ps_io(stage_data, code.size()); break;
        case HwStage::Vs: p.io = vs_io(stage_data); break;
        case HwStage::Cs: {
            ComputeIo cs;
            cs.rsrc2 = u32_at(stage_data, CS_RSRC2);
            cs.threads[0] = u32_at(stage_data, CS_THREADS);
            cs.threads[1] = u32_at(stage_data, CS_THREADS + 4);
            cs.threads[2] = u32_at(stage_data, CS_THREADS + 8);
            p.io = cs;
            break;
        }
        case HwStage::Ms: p.io = ms_io(stage_data, u32_at(record, RECORD_LDS)); break;
        default: p.io = std::monostate{}; break;
        }
        p.wave64 = (u32_at(record, RECORD_FLAGS) & FLAG_WAVE32) == 0;
        p.vgprs = vgprs;
        p.code.assign(code.begin(), code.end());
        out.push_back(std::move(p));
    }
    return out;
}

}  // namespace

std::uint8_t user_sgpr(HwStage stage, std::uint8_t index)
{
    switch (stage) {
    case HwStage::Ls:
    case HwStage::Hs:
    case HwStage::Es:
    case HwStage::Gs:
    case HwStage::Ms: return static_cast<std::uint8_t>(index + 8);
    default: return index;
    }
}

Usage Usage::from_raw(std::uint8_t raw)
{
    switch (raw) {
    case 6: return Generic();
    case 7: return ClipDistance();
    default: return Other(raw);
    }
}

RootType RootType::from_raw(std::uint32_t raw)
{
    switch (raw) {
    case 0: return Constants();
    case 1: return Descriptor();
    case 3: return Table();
    case 4: return ImmediateSampler();
    case 5: return ResourceHeap();
    case 6: return SamplerHeap();
    default: return Other(raw);
    }
}

ResourceType ResourceType::from_raw(std::uint8_t raw) { return resource_type(raw); }

std::uint8_t ComputeIo::user_sgprs() const { return static_cast<std::uint8_t>((rsrc2 >> 1) & 0x1F); }

std::pair<std::array<std::optional<std::uint8_t>, 3>, std::optional<std::uint8_t>> ComputeIo::system_sgprs() const
{
    std::uint8_t next = user_sgprs();
    auto take = [&](unsigned bit) -> std::optional<std::uint8_t> {
        if (((rsrc2 >> bit) & 1) == 0) return std::nullopt;
        next = static_cast<std::uint8_t>(next + 1);
        return static_cast<std::uint8_t>(next - 1);
    };
    std::array<std::optional<std::uint8_t>, 3> ids;
    ids[0] = take(7);
    ids[1] = take(8);
    ids[2] = take(9);
    const auto info = take(10);
    return {ids, info};
}

std::uint8_t ComputeIo::initialized_sgprs() const
{
    const auto [ids, info] = system_sgprs();
    std::optional<std::uint8_t> end;
    for (const auto& s : {ids[0], ids[1], ids[2], info}) {
        if (!s) continue;
        const auto e = static_cast<std::uint8_t>(*s + 1);
        if (!end || e > *end) end = e;
    }
    return static_cast<std::uint8_t>(std::max(end.value_or(0), user_sgprs()) + (rsrc2 & 1));
}

std::uint8_t ComputeIo::thread_id_vgprs() const { return static_cast<std::uint8_t>(((rsrc2 >> 11) & 3) + 1); }

std::uint32_t ComputeIo::lds_bytes() const { return ((rsrc2 >> 15) & 0x1FF) * 512; }

std::vector<std::uint8_t> NativeProgram::patched_code() const
{
    std::vector<std::uint8_t> out = code;
    if (const auto* ps = std::get_if<PsIo>(&io)) {
        for (const ExportPatch& p : ps->patches) {
            const std::size_t at = std::size_t{p.offset} * 4;
            for (std::size_t i = 0; i < p.code.size(); ++i) {
                const std::uint32_t word = p.code[i];
                // ps_io validates that each patch falls within the code.
                out.at(at + 4 * i + 0) = static_cast<std::uint8_t>(word);
                out.at(at + 4 * i + 1) = static_cast<std::uint8_t>(word >> 8);
                out.at(at + 4 * i + 2) = static_cast<std::uint8_t>(word >> 16);
                out.at(at + 4 * i + 3) = static_cast<std::uint8_t>(word >> 24);
            }
        }
    }
    return out;
}

std::vector<NativeProgram> native_programs(std::span<const std::uint8_t> part)
{
    if (starts_with(part, "FSPX")) return {fspx_program(part)};
    if (!magic_at(part, XDAT_OFFSET, "XDAT")) return {};
    const std::size_t count = u32_at(part, XDAT_OFFSET + 4);
    if (count > MAX_XDAT_ENTRIES) throw ShaderError::malformed("implausible XDAT entry count");
    std::vector<NativeProgram> out;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t at = XDAT_OFFSET + 12 + 12 * i;
        if (u32_at(part, at) != KIND_NATIVE) continue;
        const std::uint64_t offset = u32_at(part, at + 4);
        const std::uint64_t size = u32_at(part, at + 8);
        const Bytes blob = need(bytes(part, offset, size), "XDAT entry out of bounds");
        const std::vector<std::uint8_t> fssm = unpack_blob(blob);
        for (NativeProgram& p : programs(fssm)) out.push_back(std::move(p));
    }
    return out;
}

NativeProgram fspx_program(std::span<const std::uint8_t> part)
{
    if (u32_at(part, 4) != FSPX_VERSION) throw ShaderError::malformed("unknown FSPX version");
    const std::size_t count = u32_at(part, 8);
    if (count > MAX_XDAT_ENTRIES) throw ShaderError::malformed("implausible FSPX entry count");
    auto entry = [&](std::uint32_t kind) -> Bytes {
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t at = FSPX_HEADER + 16 * i;
            if (u32_at(part, at) == kind) {
                const std::uint32_t offset = u32_at(part, at + 4);
                const std::uint32_t size = u32_at(part, at + 8);
                return need(bytes(part, offset, size), "FSPX entry out of bounds");
            }
        }
        throw ShaderError::malformed("FSPX entry missing");
    };
    const Bytes packet = entry(FSPX_PACKET);
    const Bytes code = entry(FSPX_CODE);
    const Bytes meta = entry(FSPX_METADATA);
    if (std::size_t{u32_at(meta, 8)} * 4 != code.size()) throw ShaderError::malformed("FSPX code size disagrees");
    const bool wave64 = (fspx_u16_at(packet, PACKET_FLAGS) & PACKET_WAVE32) == 0;
    // VGPRs in granules of 4 (wave64) or 8 (wave32).
    const std::uint32_t granules = (u32_at(packet, PACKET_RSRC1) & 0x3F) + 1;
    const auto vgprs = static_cast<std::uint16_t>(std::min<std::uint32_t>(granules * (wave64 ? 4 : 8), 256));
    NativeProgram p;
    p.stage = HwStage::Cs;
    p.wave64 = wave64;
    p.vgprs = vgprs;
    const Bytes records = meta.size() >= FSPX_RECORDS ? meta.subspan(FSPX_RECORDS) : Bytes{};
    for (std::size_t k = 0; k + FSPX_RECORD <= records.size(); k += FSPX_RECORD) {
        const std::uint8_t* e = records.data() + k;
        if (e[0] == 0x42) continue;
        if (e[0] == 0x43) break;
        if (e[0] != 0x41) throw ShaderError::malformed("unknown FSPX metadata record");
        const RootType root = RootType::from_raw(e[5]);
        UsedResource u;
        u.kind = resource_type(e[4]);
        u.root = root;
        u.table_offset_dwords = le32(e, 8);
        u.lower_bound = le32(e, 12);
        u.stride = 0;
        u.dwords = e[20];
        u.user_data = e[21];
        u.slot = e[22];
        std::uint32_t sgprs;
        switch (root.kind) {
        case RootType::Kind::Table: sgprs = 1; break;
        case RootType::Kind::Descriptor: sgprs = 2; break;
        // The code holds its words; no user data.
        case RootType::Kind::ImmediateSampler: sgprs = 0; break;
        default: throw ShaderError::malformed("an FSPX resource of an unknown root type");
        }
        const bool listed =
            std::any_of(p.layout.begin(), p.layout.end(), [&](const RootParam& r) { return r.slot == u.slot; });
        if (sgprs > 0 && !listed) {
            RootParam r;
            r.root = root;
            r.table_offset_dwords = 0;
            r.sgprs = sgprs;
            r.base_register = u.lower_bound;
            r.user_data = u.user_data;
            r.slot = u.slot;
            p.layout.push_back(r);
        }
        p.used.push_back(u);
    }
    ComputeIo cs;
    cs.rsrc2 = u32_at(packet, PACKET_RSRC2);
    cs.threads[0] = fspx_u16_at(packet, PACKET_THREADS);
    cs.threads[1] = fspx_u16_at(packet, PACKET_THREADS + 2);
    cs.threads[2] = fspx_u16_at(packet, PACKET_THREADS + 4);
    p.io = cs;
    p.code.assign(code.begin(), code.end());
    return p;
}

namespace detail {
std::vector<std::uint8_t> unpack(std::span<const std::uint8_t> blob) { return unpack_blob(blob); }
}  // namespace detail

}  // namespace x2w::vs::native
