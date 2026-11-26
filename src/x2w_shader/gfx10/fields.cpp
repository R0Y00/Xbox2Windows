// Operand fields of GFX10 instructions, by encoding.
#include <algorithm>
#include <bit>

#include "x2w_shader/gfx10/mod.h"

namespace x2w::vs::gfx10 {

namespace {

constexpr std::uint8_t VCC = 106;
constexpr std::uint8_t EXEC = 126;

// VOP3-only opcodes that take two sources (the rest take three).
constexpr std::string_view VOP3_TWO_SOURCES[] = {
    "v_add_f64",
    "v_mul_f64",
    "v_min_f64",
    "v_max_f64",
    "v_ldexp_f64",
    "v_mul_lo_u32",
    "v_mul_hi_u32",
    "v_mul_lo_i32",
    "v_mul_hi_i32",
    "v_trig_preop_f64",
    "v_lshlrev_b64",
    "v_lshrrev_b64",
    "v_ashrrev_i64",
    "v_add_nc_u16",
    "v_sub_nc_u16",
    "v_mul_lo_u16",
    "v_lshrrev_b16",
    "v_ashrrev_i16",
    "v_max_u16",
    "v_max_i16",
    "v_min_u16",
    "v_min_i16",
    "v_add_nc_i16",
    "v_sub_nc_i16",
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
    "v_pack_b32_f16",
    "v_cvt_pknorm_i16_f16",
    "v_cvt_pknorm_u16_f16",
    "v_lshlrev_b16",
    "v_interp_p1ll_f16",
    "v_readlane_b32",
    "v_writelane_b32",
    "v_ldexp_f32",
    "v_bfm_b32",
    "v_bcnt_u32_b32",
    "v_mbcnt_lo_u32_b32",
    "v_mbcnt_hi_u32_b32",
    "v_cvt_pknorm_i16_f32",
    "v_cvt_pknorm_u16_f32",
    "v_cvt_pk_u16_u32",
    "v_cvt_pk_i16_i32",
    "v_sub_nc_i32",
    "v_add_nc_i32",
};

// VOP3 opcodes whose bits 14:8 hold a scalar destination instead of `abs`/`opsel`.
constexpr std::string_view VOP3B[] = {
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
    "v_add_co_ci_u32",
    "v_sub_co_ci_u32",
    "v_subrev_co_ci_u32",
    "v_div_scale_f32",
    "v_div_scale_f64",
    "v_mad_u64_u32",
    "v_mad_i64_i32",
};

template <std::size_t N>
bool in(const std::string_view (&list)[N], std::string_view name) {
    return std::find(std::begin(list), std::end(list), name) != std::end(list);
}

bool is_any(std::string_view name, std::initializer_list<std::string_view> list) {
    return std::find(list.begin(), list.end(), name) != list.end();
}

std::vector<std::string_view> split(std::string_view s, char sep) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (;;) {
        std::size_t at = s.find(sep, start);
        if (at == std::string_view::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, at - start));
        start = at + 1;
    }
}

constexpr bool bit(std::uint32_t word, std::uint32_t n) { return ((word >> n) & 1) != 0; }

Operand scalar(std::uint8_t code, std::uint8_t dwords) { return Operand::Scalar(code, dwords); }

// the low 8 bits.
Operand vector(std::uint32_t reg, std::uint8_t dwords) { return Operand::Vector(std::uint8_t(reg), dwords); }

// A 9-bit source field (8-bit scalar fields are the same codes below 256).
Operand source(std::uint32_t code, std::uint8_t dwords, std::uint32_t literal) {
    if (code <= 127) return scalar(std::uint8_t(code), dwords);
    if (code == 255) return Operand::Literal(literal);
    if (code >= 256) return vector(code - 256, dwords);
    return Operand::Const(std::uint8_t(code));
}

// VALU destinations that are SGPRs.
Operand valu_dst(std::string_view name, std::uint32_t field, std::uint8_t dwords) {
    if (name == "v_readlane_b32" || name == "v_readfirstlane_b32") return scalar(std::uint8_t(field), 1);
    return vector(field, dwords);
}

// Dwords of a type suffix such as `b64` or `f32`.
std::uint8_t type_dwords(std::string_view token) {
    return is_any(token, {"b64", "u64", "i64", "f64"}) ? 2 : 1;
}

struct Salu {
    std::uint8_t dst, src0, src1;
};

// (sdst, ssrc0, ssrc1) dwords of a scalar ALU opcode.
Salu salu_dwords(std::string_view name) {
    if (is_any(name, {"s_lshl_b64", "s_lshr_b64", "s_ashr_i64", "s_bfe_u64", "s_bfe_i64", "s_bitcmp0_b64",
                      "s_bitcmp1_b64"})) {
        return {2, 2, 1};
    }
    if (is_any(name, {"s_bfm_b64", "s_bitset0_b64", "s_bitset1_b64", "s_bitreplicate_b64_b32"})) return {2, 1, 1};
    if (is_any(name, {"s_bcnt0_i32_b64", "s_bcnt1_i32_b64", "s_ff0_i32_b64", "s_ff1_i32_b64", "s_flbit_i32_b64",
                      "s_flbit_i32_i64"})) {
        return {1, 2, 2};
    }
    std::size_t at = name.rfind('_');
    std::uint8_t n = type_dwords(at == std::string_view::npos ? name : name.substr(at + 1));
    return {n, n, n};
}

struct ValuDwords {
    std::uint8_t dst;
    std::array<std::uint8_t, 3> src;
};

// (vdst, [src0, src1, src2]) dwords of a VALU opcode.
ValuDwords valu_dwords(std::string_view name) {
    if (is_any(name, {"v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64"})) return {2, {1, 2, 2}};
    if (is_any(name, {"v_ldexp_f64", "v_trig_preop_f64", "v_cmp_class_f64", "v_cmpx_class_f64"})) {
        return {2, {2, 1, 2}};
    }
    if (is_any(name, {"v_mad_u64_u32", "v_mad_i64_i32"})) return {2, {1, 1, 2}};
    if (is_any(name, {"v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8"})) return {2, {2, 1, 2}};
    if (name == "v_mqsad_u32_u8") return {4, {2, 1, 4}};
    if (name == "v_frexp_exp_i32_f64") return {1, {2, 2, 2}};
    auto parts = split(name, '_');
    if (parts.size() == 4 && parts[1] == "cvt") {
        std::uint8_t s = type_dwords(parts[3]);
        return {type_dwords(parts[2]), {s, s, s}};
    }
    std::uint8_t n = type_dwords(parts.back());
    return {n, {n, n, n}};
}

// Dwords of loaded, stored or atomic data, from the opcode name.
std::uint8_t data_dwords(std::string_view name) {
    std::uint8_t n = 1;
    for (std::string_view token : split(name, '_')) {
        if (is_any(token, {"dwordx2", "x2", "xy", "b64", "u64", "i64", "f64"})) {
            n = 2;
        } else if (is_any(token, {"dwordx3", "xyz", "b96"})) {
            n = 3;
        } else if (is_any(token, {"dwordx4", "xyzw", "b128"})) {
            n = 4;
        } else if (token == "dwordx8") {
            n = 8;
        } else if (token == "dwordx16") {
            n = 16;
        }
    }
    if (name.find("cmpswap") != std::string_view::npos) n = std::uint8_t(n * 2);
    if (name.find("format_d16") != std::string_view::npos) n = std::uint8_t((n + 1) / 2);
    return n;
}

// Address dwords of an image opcode: offset, bias, compare, derivatives, coordinates, then
// LOD/clamp/mip, packed in pairs with `a16`.
std::uint8_t image_addr_dwords(std::string_view name, std::uint8_t dim, bool a16) {
    // (coordinates, derivatives per direction) for 1D 2D 3D CUBE 1D_ARRAY 2D_ARRAY 2D_MSAA
    // 2D_MSAA_ARRAY.
    static constexpr std::uint8_t DIMS[8][2] = {{1, 1}, {2, 2}, {3, 3}, {3, 2}, {2, 1}, {3, 2}, {3, 2}, {4, 2}};
    std::uint8_t coords = DIMS[dim & 7][0], grads = DIMS[dim & 7][1];
    if (name == "image_bvh_intersect_ray") return a16 ? 8 : 11;
    if (name == "image_bvh64_intersect_ray") return a16 ? 9 : 12;
    if (name == "image_get_resinfo") return 1;
    unsigned whole = 0, derivs = 0, packed = coords;
    std::optional<std::string_view> rest;
    if (name.starts_with("image_sample")) {
        rest = name.substr(std::string_view("image_sample").size());
    } else if (name.starts_with("image_gather4")) {
        rest = name.substr(std::string_view("image_gather4").size());
    }
    if (rest) {
        for (std::string_view token : split(*rest, '_')) {
            if (is_any(token, {"o", "b", "c"})) {
                whole += 1;
            } else if (is_any(token, {"d", "cd"})) {
                derivs = grads;
            } else if (is_any(token, {"l", "cl"})) {
                packed += 1;
            }
        }
    } else if (name.find("_mip") != std::string_view::npos) {
        packed += 1;
    }
    if (a16) return std::uint8_t(whole + 2 * ((derivs + 1) / 2) + (packed + 1) / 2);
    return std::uint8_t(whole + 2 * derivs + packed);
}

}  // namespace

bool ds_pair(std::string_view name) {
    for (std::string_view t : split(name, '_')) {
        if (t.starts_with("read2") || t.starts_with("write2") || t.starts_with("wrxchg2")) return true;
    }
    return false;
}

Fields Instruction::fields(bool wave64) const {
    std::string_view name = this->name().value_or("");
    auto w = [&](std::size_t i) -> std::uint32_t { return i < words.size() ? words[i] : 0; };
    std::uint32_t w0 = w(0), w1 = w(1);
    auto cache = [](bool glc, bool slc, bool dlc) { return Cache{glc, slc, dlc}; };
    switch (encoding) {
    case Encoding::Sop2:
    case Encoding::Sop1:
    case Encoding::Sopc: {
        Salu d = salu_dwords(name);
        Operand sdst = scalar(std::uint8_t((w0 >> 16) & 0x7F), d.dst);
        auto ssrc = [&](std::uint32_t shift, std::uint8_t dwords) { return source((w0 >> shift) & 0xFF, dwords, w1); };
        Sop s;
        if (encoding == Encoding::Sop2) {
            s.dst = sdst;
            s.src = {ssrc(0, d.src0), ssrc(8, d.src1)};
        } else if (encoding == Encoding::Sopc) {
            s.src = {ssrc(0, d.src0), ssrc(8, d.src1)};
        } else if (name == "s_getpc_b64") {
            s.dst = sdst;
        } else if (name == "s_setpc_b64" || name == "s_rfe_b64") {
            s.src = {ssrc(0, d.src0)};
        } else {
            s.dst = sdst;
            s.src = {ssrc(0, d.src0)};
        }
        return s;
    }
    case Encoding::Sopk: {
        Operand sdst = scalar(std::uint8_t((w0 >> 16) & 0x7F), std::uint8_t(1 + (name == "s_call_b64" ? 1 : 0)));
        Sop s;
        if (name == "s_setreg_imm32_b32") {
            s.src = {Operand::Literal(w1)};
        } else if (name == "s_version") {
        } else if (name.starts_with("s_cmpk_") || name.starts_with("s_waitcnt_") || name == "s_setreg_b32") {
            s.src = {sdst};
        } else {
            s.dst = sdst;
        }
        s.imm = std::uint16_t(w0);
        return s;
    }
    case Encoding::Sopp:
        return Sopp{std::uint16_t(w0)};
    case Encoding::Smem: {
        std::uint8_t sdata = std::uint8_t((w0 >> 6) & 0x7F);
        // s_atc_probe*: SDATA is a 7-bit immediate, not an SGPR. The buffer form's resource is 4
        // dwords. GLC and DLC are not operands.
        bool probe = name == "s_atc_probe" || name == "s_atc_probe_buffer";
        Smem s;
        if (probe) {
            s.data = Operand::Literal(sdata);
        } else if (is_any(name, {"s_gl1_inv", "s_dcache_inv", "s_dcache_wb"}) || name.starts_with("s_dcache_discard")) {
        } else if (name == "s_memtime" || name == "s_memrealtime") {
            s.data = scalar(sdata, 2);
        } else {
            s.data = scalar(sdata, data_dwords(name));
        }
        std::uint8_t base =
            std::uint8_t(2 + 2 * ((name.starts_with("s_buffer_") || name == "s_atc_probe_buffer") ? 1 : 0));
        s.base = scalar(std::uint8_t((w0 & 0x3F) << 1), base);
        s.offset = std::int32_t(w1 << 11) >> 11;
        s.soffset = scalar(std::uint8_t(w1 >> 25), 1);
        s.cache = probe ? Cache{} : cache(bit(w0, 16), false, bit(w0, 14));
        return s;
    }
    case Encoding::Vop1:
    case Encoding::Vop2:
    case Encoding::Vopc:
    case Encoding::Vop3:
    case Encoding::Vop3p:
        return valu(name, wave64 ? 2 : 1);
    case Encoding::Vintrp: {
        std::uint8_t vsrc = std::uint8_t(w0 & 0xFF);
        Vintrp v;
        v.dst = vector(w0 >> 18, 1);
        v.src = name == "v_interp_mov_f32" ? Operand::Const(vsrc) : vector(vsrc, 1);
        v.attr = Operand::Attr(std::uint8_t((w0 >> 10) & 0x3F), std::uint8_t((w0 >> 8) & 3), false);
        return v;
    }
    case Encoding::Ds: {
        std::uint8_t ty = data_dwords(name);
        bool pair = ds_pair(name);
        bool returns = name.find("read") != std::string_view::npos || name.find("_rtn") != std::string_view::npos ||
                       is_any(name, {"ds_swizzle_b32", "ds_permute_b32", "ds_bpermute_b32", "ds_append",
                                     "ds_consume", "ds_ordered_count"});
        bool no_addr = name.starts_with("ds_gws") || name.find("addtid") != std::string_view::npos ||
                       is_any(name, {"ds_nop", "ds_append", "ds_consume"});
        unsigned data;
        if (name.find("read") != std::string_view::npos || name.find("src2") != std::string_view::npos ||
            is_any(name, {"ds_swizzle_b32", "ds_nop", "ds_append", "ds_consume", "ds_ordered_count",
                          "ds_gws_sema_v", "ds_gws_sema_p", "ds_gws_sema_release_all"})) {
            data = 0;
        } else if (pair || name.find("cmpst") != std::string_view::npos || name.find("mskor") != std::string_view::npos) {
            data = 2;
        } else {
            data = 1;
        }
        Ds d;
        if (returns) d.dst = vector(w1 >> 24, std::uint8_t(ty * (1 + (pair ? 1 : 0))));
        if (!no_addr) d.addr = vector(w1 & 0xFF, 1);
        for (unsigned i = 0; i < data; ++i) d.data.push_back(vector((w1 >> (8 + 8 * i)) & 0xFF, ty));
        d.offset0 = std::uint8_t(w0);
        d.offset1 = std::uint8_t(w0 >> 8);
        d.gds = bit(w0, 17);
        return d;
    }
    case Encoding::Mubuf:
    case Encoding::Mtbuf: {
        bool offen = bit(w0, 12), idxen = bit(w0, 13), tfe = bit(w1, 23);
        bool mtbuf = encoding == Encoding::Mtbuf;
        Buffer b;
        if (!name.ends_with("_inv")) {
            b.data = vector((w1 >> 8) & 0xFF, std::uint8_t(data_dwords(name) + (tfe ? 1 : 0)));
        }
        b.addr = vector(w1 & 0xFF, std::uint8_t(1 + ((offen && idxen) ? 1 : 0)));
        b.rsrc = scalar(std::uint8_t(((w1 >> 16) & 0x1F) << 2), 4);
        b.soffset = source(w1 >> 24, 1, 0);
        b.offset = std::uint16_t(w0 & 0xFFF);
        b.offen = offen;
        b.idxen = idxen;
        if (mtbuf) b.format = std::uint8_t((w0 >> 19) & 0x7F);
        b.lds = !mtbuf && bit(w0, 16);
        b.tfe = tfe;
        b.cache = cache(bit(w0, 14), bit(w1, 22), bit(w0, 15));
        return b;
    }
    case Encoding::Mimg: {
        std::uint8_t dim = std::uint8_t((w0 >> 3) & 7), dmask = std::uint8_t((w0 >> 8) & 0xF);
        bool a16 = bit(w1, 30), d16 = bit(w1, 31);
        bool r128 = bit(w0, 15), tfe = bit(w0, 16), lwe = bit(w0, 17);
        std::uint8_t data = name.starts_with("image_gather4")
                                ? 4
                                : std::uint8_t(std::max(std::popcount(unsigned(dmask)), 1));
        if (d16) data = std::uint8_t((data + 1) / 2);
        std::uint8_t count = image_addr_dwords(name, dim, a16);
        Image im;
        if (((w0 >> 1) & 3) == 0) {
            im.addr = {vector(w1 & 0xFF, count)};
        } else {
            // NSA: one VGPR per address, the first in dword 1, the rest in the extra dwords.
            for (std::size_t i = 0; i < count; ++i) {
                if (i == 0) {
                    im.addr.push_back(vector(w1 & 0xFF, 1));
                } else {
                    im.addr.push_back(vector((w(2 + (i - 1) / 4) >> (8 * ((i - 1) % 4))) & 0xFF, 1));
                }
            }
        }
        bool sampled = name.starts_with("image_sample") || name.starts_with("image_gather4") ||
                       name.starts_with("image_get_lod");
        im.data = vector((w1 >> 8) & 0xFF, std::uint8_t(data + ((tfe || lwe) ? 1 : 0)));
        im.rsrc = scalar(std::uint8_t(((w1 >> 16) & 0x1F) << 2), r128 ? 4 : 8);
        if (sampled) im.sampler = scalar(std::uint8_t(((w1 >> 21) & 0x1F) << 2), 4);
        im.dmask = dmask;
        im.dim = dim;
        im.unorm = bit(w0, 12);
        im.r128 = r128;
        im.a16 = a16;
        im.d16 = d16;
        im.tfe = tfe;
        im.lwe = lwe;
        im.cache = cache(bit(w0, 13), bit(w0, 25), bit(w0, 7));
        return im;
    }
    case Encoding::Flat:
    case Encoding::Global:
    case Encoding::Scratch: {
        std::uint8_t saddr = std::uint8_t((w1 >> 16) & 0x7F);
        bool glc = bit(w0, 16);
        std::uint8_t n = data_dwords(name);
        bool load = name.find("_load_") != std::string_view::npos;
        bool atomic = name.find("_atomic_") != std::string_view::npos;
        Flat f;
        if (encoding == Encoding::Flat) {
            f.addr = vector(w1 & 0xFF, 2);
        } else if (encoding == Encoding::Global) {
            f.addr = vector(w1 & 0xFF, std::uint8_t(1 + (saddr == NULL_CODE ? 1 : 0)));
        } else if (saddr == NULL_CODE) {
            // With an SGPR base, scratch takes no VGPR.
            f.addr = vector(w1 & 0xFF, 1);
        }
        std::uint8_t ret = (atomic && name.find("cmpswap") != std::string_view::npos) ? std::uint8_t(n / 2) : n;
        if (load || (atomic && glc)) f.dst = vector(w1 >> 24, ret);
        if (!load) f.data = vector((w1 >> 8) & 0xFF, n);
        if (encoding != Encoding::Flat) {
            f.saddr = scalar(saddr, std::uint8_t(1 + (encoding == Encoding::Global ? 1 : 0)));
        }
        // GFX10 FLAT offsets are 11-bit and unsigned: the hardware ignores bit 11.
        f.offset = encoding == Encoding::Flat ? std::int16_t(w0 & 0x7FF)
                                              : std::int16_t(std::int16_t(std::uint16_t(w0 << 4)) >> 4);
        f.lds = bit(w0, 13);
        f.cache = cache(glc, bit(w0, 17), bit(w0, 12));
        return f;
    }
    case Encoding::Exp: {
        bool compr = bit(w0, 10);
        Exp e;
        e.target = std::uint8_t((w0 >> 4) & 0x3F);
        for (std::uint32_t i = 0; i < 4; ++i) {
            std::uint32_t slot = compr ? i / 2 : i;
            if (bit(w0, i)) e.src[i] = vector((w1 >> (8 * slot)) & 0xFF, 1);
        }
        e.compr = compr;
        e.done = bit(w0, 11);
        e.vm = bit(w0, 12);
        return e;
    }
    }
    return Sopp{};
}

Valu Instruction::valu(std::string_view name, std::uint8_t lanes) const {
    // No operands; the console compiler stores a shader hash in their fields.
    if (is_any(name, {"v_nop", "v_pipeflush", "v_clrexcp"})) return Valu{};
    auto w = [&](std::size_t i) -> std::uint32_t { return i < words.size() ? words[i] : 0; };
    std::uint32_t w0 = w(0);
    ValuDwords n = valu_dwords(name);
    bool compare = encoding == Encoding::Vopc || (encoding == Encoding::Vop3 && opcode < 0x100);
    bool vop3b = in(VOP3B, name);
    bool carry_in = name.ends_with("_co_ci_u32") || name == "v_cndmask_b32";
    Valu v;
    switch (encoding) {
    case Encoding::Vop1:
    case Encoding::Vop2:
    case Encoding::Vopc: {
        std::uint32_t ext = w(1);
        std::uint32_t src0 = w0 & 0x1FF;
        std::uint32_t vsrc1 = 256 + ((w0 >> 9) & 0xFF);
        std::uint32_t src1 = vsrc1;
        if (src0 == 0xF9) {
            Sdwa sdwa;
            sdwa.dst_sel = std::uint8_t((ext >> 8) & 7);
            sdwa.dst_unused = std::uint8_t((ext >> 11) & 3);
            sdwa.src_sel = {std::uint8_t((ext >> 16) & 7), std::uint8_t((ext >> 24) & 7)};
            sdwa.sext = std::uint8_t(((ext >> 19) & 1) | ((ext >> 26) & 2));
            // The s0/s1 bits select an SGPR or constant instead of a VGPR.
            src0 = (ext & 0xFF) | (((ext >> 23) & 1) == 0 ? 256u : 0u);
            if (ext >> 31 == 1) src1 = vsrc1 - 256;
            v.neg = std::uint8_t(((ext >> 20) & 1) | ((ext >> 27) & 2));
            v.abs = std::uint8_t(((ext >> 21) & 1) | ((ext >> 28) & 2));
            if (compare) {
                if (((ext >> 15) & 1) == 1) v.dst = scalar(std::uint8_t((ext >> 8) & 0x7F), lanes);
                Sdwa c = sdwa;
                c.dst_sel = 6;
                c.dst_unused = 0;
                v.sdwa = c;
            } else {
                v.clamp = ((ext >> 13) & 1) == 1;
                v.omod = std::uint8_t((ext >> 14) & 3);
                v.sdwa = sdwa;
            }
        } else if (src0 == 0xFA) {
            src0 = 256 + (ext & 0xFF);
            v.neg = std::uint8_t(((ext >> 20) & 1) | ((ext >> 21) & 2));
            v.abs = std::uint8_t(((ext >> 21) & 1) | ((ext >> 22) & 2));
            v.dpp = DppRow{std::uint16_t((ext >> 8) & 0x1FF), std::uint8_t(ext >> 28), std::uint8_t((ext >> 24) & 0xF),
                           ((ext >> 19) & 1) == 1, ((ext >> 18) & 1) == 1};
        } else if (src0 == 0xE9 || src0 == 0xEA) {
            v.dpp = DppLanes{ext >> 8, src0 == 0xEA};
            src0 = 256 + (ext & 0xFF);
        }
        Operand op0 = source(src0, n.src[0], ext);
        Operand op1 = source(src1, n.src[1], ext);
        if (encoding == Encoding::Vopc) {
            std::uint8_t mask = name.starts_with("v_cmpx") ? EXEC : VCC;
            if (!v.dst) v.dst = scalar(mask, lanes);
            v.src = {op0, op1};
        } else if (encoding == Encoding::Vop1) {
            v.dst = valu_dst(name, (w0 >> 17) & 0xFF, n.dst);
            v.src = {op0};
        } else {
            v.dst = vector((w0 >> 17) & 0xFF, n.dst);
            if (name.ends_with("mk_f32") || name.ends_with("mk_f16")) {
                v.src = {op0, Operand::Literal(ext), op1};
            } else if (name.ends_with("ak_f32") || name.ends_with("ak_f16")) {
                v.src = {op0, op1, Operand::Literal(ext)};
            } else {
                v.src = {op0, op1};
            }
            if (carry_in) v.src.push_back(scalar(VCC, lanes));
            if (name.ends_with("_co_ci_u32")) v.carry = scalar(VCC, lanes);
        }
        break;
    }
    case Encoding::Vop3:
    case Encoding::Vop3p: {
        std::uint32_t w1 = w(1);
        auto field = [&](std::uint32_t i) { return (w1 >> (9 * i)) & 0x1FF; };
        std::size_t count;
        if (encoding == Encoding::Vop3p) {
            count = (name.starts_with("v_pk_") && name.find("_fma_") == std::string_view::npos &&
                     name.find("_mad_") == std::string_view::npos)
                        ? 2
                        : 3;
        } else if (opcode < 0x100) {
            count = 2;
        } else if (opcode < 0x140) {
            count = 2 + (carry_in ? 1 : 0);
        } else if (opcode >= 0x180 && opcode < 0x200) {
            count = 1;
        } else if (in(VOP3_TWO_SOURCES, name)) {
            count = 2;
        } else {
            count = 3;
        }
        for (std::size_t i = 0; i < count; ++i) {
            std::uint32_t code = field(std::uint32_t(i));
            if (i == 0 && name.starts_with("v_interp_")) {
                v.src.push_back(Operand::Attr(std::uint8_t(code & 0x3F), std::uint8_t((code >> 6) & 3), ((code >> 8) & 1) == 1));
            } else if (i == 2 && carry_in) {
                v.src.push_back(source(code, lanes, w(2)));
            } else {
                v.src.push_back(source(code, n.src[i], w(2)));
            }
        }
        v.clamp = ((w0 >> 15) & 1) == 1;
        v.neg = std::uint8_t(w1 >> 29);
        if (encoding == Encoding::Vop3p) {
            v.neg_hi = std::uint8_t((w0 >> 8) & 7);
            v.opsel = std::uint8_t((w0 >> 11) & 7);
            v.opsel_hi = std::uint8_t(((w1 >> 27) & 3) | ((w0 >> 12) & 4));
        } else {
            v.omod = std::uint8_t((w1 >> 27) & 3);
            if (vop3b) {
                v.carry = scalar(std::uint8_t((w0 >> 8) & 0x7F), lanes);
            } else {
                v.abs = std::uint8_t((w0 >> 8) & 7);
                v.opsel = std::uint8_t((w0 >> 11) & 0xF);
            }
        }
        std::uint32_t vdst = w0 & 0xFF;
        v.dst = compare ? scalar(std::uint8_t(vdst), lanes) : valu_dst(name, vdst, n.dst);
        break;
    }
    default:
        // unreachable!("not a VALU encoding")
        break;
    }
    return v;
}

}  // namespace x2w::vs::gfx10
