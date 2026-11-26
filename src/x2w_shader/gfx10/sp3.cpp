// Instruction text in AMD's SP3 assembly syntax.
#include "x2w_shader/gfx10/sp3.h"

#include <cstdio>
#include <utility>

#include "x2w_shader/gfx10/mod.h"

namespace x2w::vs::gfx10 {

namespace {

template <class... A>
std::string fmt(const char* f, A... a) {
    char buf[128];
    std::snprintf(buf, sizeof buf, f, a...);
    return buf;
}

std::string join(const std::vector<std::string>& ops, const char* sep = ", ") {
    std::string out;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (i) out += sep;
        out += ops[i];
    }
    return out;
}

std::string str(const Operand& o) { return o.to_string(); }

bool is_null(const Operand& o) { return o.kind == Operand::Kind::Scalar && o.code == NULL_CODE; }

// `s_atc_probe` immediate: 0..=64 in decimal, larger values in hex.
std::string atc_immediate(std::uint32_t imm) {
    if (imm <= 64) return std::to_string(imm);
    return fmt("0x%x", unsigned(imm));
}

// Cache flags; images write `dlc` first.
void flags(std::string& out, const Cache& cache, bool dlc_first) {
    std::pair<bool, const char*> dlc = {cache.dlc, " dlc"};
    std::pair<bool, const char*> none = {false, ""};
    std::pair<bool, const char*> first = dlc_first ? dlc : none;
    std::pair<bool, const char*> last = dlc_first ? none : dlc;
    for (auto [on, text] : {first, std::pair<bool, const char*>{cache.glc, " glc"},
                            std::pair<bool, const char*>{cache.slc, " slc"}, last}) {
        if (on) out += text;
    }
    if (!out.empty() && out[0] == ' ') out.erase(0, 1);
}

std::string label(const Instruction& inst, std::uint16_t imm) {
    std::int64_t target = std::int64_t(inst.offset) + 4 * std::int64_t(inst.words.size()) +
                          4 * std::int64_t(std::int16_t(imm));
    // Hex-format negative 64-bit values in two's complement.
    return fmt("label_%04llX", static_cast<unsigned long long>(target));
}

std::string hwreg(std::uint16_t imm) {
    unsigned id = imm & 0x3F, offset = (imm >> 6) & 0x1F, size = ((imm >> 11) & 0x1F) + 1;
    std::string name;
    switch (id) {
    case 1: name = "HW_REG_MODE"; break;
    case 20: name = "HW_REG_SHADER_FLAT_SCRATCH_LO"; break;
    case 21: name = "HW_REG_SHADER_FLAT_SCRATCH_HI"; break;
    case 25: name = "HW_REG_POPS_PACKER"; break;
    default: name = std::to_string(id); break;
    }
    if (offset == 0 && size == 32) return "hwreg(" + name + ")";
    return "hwreg(" + name + ", " + std::to_string(offset) + ", " + std::to_string(size) + ")";
}

std::string waitcnt(std::uint16_t imm) {
    unsigned vm = (imm & 0xF) | (unsigned(imm) >> 14) << 4;
    unsigned exp = (imm >> 4) & 7;
    unsigned lgkm = (imm >> 8) & 0x3F;
    std::vector<std::string> parts;
    if (vm != 0x3F) parts.push_back("vmcnt(" + std::to_string(vm) + ")");
    if (exp != 7) parts.push_back("expcnt(" + std::to_string(exp) + ")");
    if (lgkm != 0x3F) parts.push_back("lgkmcnt(" + std::to_string(lgkm) + ")");
    if (parts.size() == 0 || parts.size() == 3) return fmt("0x%04x", unsigned(imm));
    return join(parts, " & ");
}

std::string sendmsg(std::uint16_t imm) {
    unsigned msg = imm & 0xF, op = (imm >> 4) & 7, stream = (imm >> 8) & 3;
    std::string msg_name;
    switch (msg) {
    case 2: msg_name = "MSG_GS"; break;
    case 3: msg_name = "MSG_GS_DONE"; break;
    case 7: msg_name = "MSG_ORDERED_PS_DONE"; break;
    case 9: msg_name = "MSG_GS_ALLOC_REQ"; break;
    default: msg_name = std::to_string(msg); break;
    }
    static const char* const OPS[] = {"GS_OP_NOP", "GS_OP_CUT", "GS_OP_EMIT", "GS_OP_EMIT_CUT"};
    std::string op_name = op < 4 ? OPS[op] : std::to_string(op);
    if (msg == 3 && op == 0) return "sendmsg(" + msg_name + ", " + op_name + ")";
    return "sendmsg(" + msg_name + ", " + op_name + ", " + std::to_string(stream) + ")";
}

std::string buffer_format(std::uint8_t format) {
    static const char* const NAMES[78] = {
        "INVALID",
        "8_UNORM",
        "8_SNORM",
        "8_USCALED",
        "8_SSCALED",
        "8_UINT",
        "8_SINT",
        "16_UNORM",
        "16_SNORM",
        "16_USCALED",
        "16_SSCALED",
        "16_UINT",
        "16_SINT",
        "16_FLOAT",
        "8_8_UNORM",
        "8_8_SNORM",
        "8_8_USCALED",
        "8_8_SSCALED",
        "8_8_UINT",
        "8_8_SINT",
        "32_UINT",
        "32_SINT",
        "32_FLOAT",
        "16_16_UNORM",
        "16_16_SNORM",
        "16_16_USCALED",
        "16_16_SSCALED",
        "16_16_UINT",
        "16_16_SINT",
        "16_16_FLOAT",
        "10_11_11_UNORM",
        "10_11_11_SNORM",
        "10_11_11_USCALED",
        "10_11_11_SSCALED",
        "10_11_11_UINT",
        "10_11_11_SINT",
        "10_11_11_FLOAT",
        "11_11_10_UNORM",
        "11_11_10_SNORM",
        "11_11_10_USCALED",
        "11_11_10_SSCALED",
        "11_11_10_UINT",
        "11_11_10_SINT",
        "11_11_10_FLOAT",
        "10_10_10_2_UNORM",
        "10_10_10_2_SNORM",
        "10_10_10_2_USCALED",
        "10_10_10_2_SSCALED",
        "10_10_10_2_UINT",
        "10_10_10_2_SINT",
        "2_10_10_10_UNORM",
        "2_10_10_10_SNORM",
        "2_10_10_10_USCALED",
        "2_10_10_10_SSCALED",
        "2_10_10_10_UINT",
        "2_10_10_10_SINT",
        "8_8_8_8_UNORM",
        "8_8_8_8_SNORM",
        "8_8_8_8_USCALED",
        "8_8_8_8_SSCALED",
        "8_8_8_8_UINT",
        "8_8_8_8_SINT",
        "32_32_UINT",
        "32_32_SINT",
        "32_32_FLOAT",
        "16_16_16_16_UNORM",
        "16_16_16_16_SNORM",
        "16_16_16_16_USCALED",
        "16_16_16_16_SSCALED",
        "16_16_16_16_UINT",
        "16_16_16_16_SINT",
        "16_16_16_16_FLOAT",
        "32_32_32_UINT",
        "32_32_32_SINT",
        "32_32_32_FLOAT",
        "32_32_32_32_UINT",
        "32_32_32_32_SINT",
        "32_32_32_32_FLOAT",
    };
    if (format < 78) return std::string("BUF_FMT_") + NAMES[format];
    return std::to_string(format);
}

std::string valu(std::string_view name, const Valu& v, Encoding encoding) {
    auto bit = [](std::uint8_t mask, std::size_t i) { return ((mask >> i) & 1) != 0; };
    bool mix = name.starts_with("v_fma_mix");
    auto src = [&](std::size_t i) {
        const Operand& o = v.src[i];
        std::string s = o.to_string();
        if (v.sdwa && bit(v.sdwa->sext, i)) s = "sext(" + s + ")";
        if (encoding == Encoding::Vop3p) {
            if (mix) {
                if (bit(v.opsel_hi, i)) s = std::string(bit(v.opsel, i) ? "sel_hi" : "sel_lo") + "(" + s + ")";
                if (bit(v.neg_hi, i)) s = "abs(" + s + ")";
                if (bit(v.neg, i)) s = "-" + s;
            } else {
                if (bit(v.neg, i) && bit(v.neg_hi, i)) s = "-" + s;
                bool lo = bit(v.opsel, i), hi = bit(v.opsel_hi, i);
                if (!lo && !hi) {
                    s = "sel_lo(" + s + ")";
                } else if (lo && hi) {
                    s = "sel_hi(" + s + ")";
                } else if (lo && !hi) {
                    s = "sel_hi_lo(" + s + ")";
                }
            }
        } else {
            if (bit(v.abs, i)) s = "abs(" + s + ")";
            if (bit(v.neg, i)) s = "-" + s;
            // v_permlane*: the opsel bits are fi and bound_ctrl.
            if (bit(v.opsel, i) && !name.starts_with("v_permlane")) s = "sel_hi(" + s + ")";
        }
        return s;
    };
    std::vector<std::string> ops;
    if (v.dst) ops.push_back(str(*v.dst));
    if (v.carry) ops.push_back(str(*v.carry));
    std::vector<std::size_t> order;
    for (std::size_t i = 0; i < v.src.size(); ++i) order.push_back(i);
    if (name.starts_with("v_interp_") && order.size() > 1) {
        // dst, i/j, attribute, then the p1 result for p2.
        std::swap(order[0], order[1]);
    }
    for (std::size_t i : order) ops.push_back(src(i));
    std::string out = join(ops);
    auto suffix = [&](const std::string& s) {
        out += ' ';
        out += s;
    };
    if (!v.src.empty() && v.src[0].kind == Operand::Kind::Attr && v.src[0].high) suffix("high");
    static const char* const OMOD[] = {"", "mul:2", "mul:4", "div:2"};
    std::string omod = OMOD[v.omod & 3];
    if (name.starts_with("v_permlane")) {
        if (v.opsel & 1) suffix("fi:1");
        if (v.opsel & 2) suffix("bound_ctrl:1");
    }
    if (encoding == Encoding::Vop3p && !mix) {
        // `-src` above covers a source negated in both halves; LLVM syntax for the one-sided rest.
        std::size_t n = v.src.size();
        std::pair<const char*, std::uint8_t> sides[] = {
            {"neg_lo", std::uint8_t(v.neg & ~v.neg_hi)},
            {"neg_hi", std::uint8_t(v.neg_hi & ~v.neg)},
        };
        for (auto [field, only] : sides) {
            bool any = false;
            for (std::size_t i = 0; i < n; ++i) any = any || bit(only, i);
            if (any) {
                std::vector<std::string> bits;
                for (std::size_t i = 0; i < n; ++i) bits.push_back(bit(only, i) ? "1" : "0");
                suffix(std::string(field) + ":[" + join(bits, ",") + "]");
            }
        }
    }
    if (v.sdwa) {
        const Sdwa& sdwa = *v.sdwa;
        if (v.clamp) suffix("clamp");
        if (!omod.empty()) suffix(omod);
        static const char* const SEL[] = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD", "?"};
        if (sdwa.dst_sel != 6) suffix(std::string("dst_sel:") + SEL[sdwa.dst_sel & 7]);
        if (sdwa.dst_unused != 0) {
            static const char* const UNUSED[] = {"UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE", "?"};
            suffix(std::string("dst_unused:") + UNUSED[sdwa.dst_unused & 3]);
        }
        for (std::size_t i = 0; i < sdwa.src_sel.size() && i < v.src.size(); ++i) {
            std::uint8_t s = sdwa.src_sel[i];
            if (s != 6) suffix("src" + std::to_string(i) + "_sel:" + SEL[s & 7]);
        }
    } else {
        if (!omod.empty()) suffix(omod);
        if (v.clamp) suffix("clamp");
    }
    if (v.dpp) {
        if (const DppRow* r = std::get_if<DppRow>(&*v.dpp)) {
            unsigned ctrl = r->ctrl;
            std::string c;
            if (ctrl <= 0x0FF) {
                c = fmt("quad_perm:[%u,%u,%u,%u]", ctrl & 3, (ctrl >> 2) & 3, (ctrl >> 4) & 3, (ctrl >> 6) & 3);
            } else if (ctrl >= 0x101 && ctrl <= 0x10F) {
                c = fmt("row_shl:%u", ctrl & 0xF);
            } else if (ctrl >= 0x111 && ctrl <= 0x11F) {
                c = fmt("row_shr:%u", ctrl & 0xF);
            } else if (ctrl >= 0x121 && ctrl <= 0x12F) {
                c = fmt("row_ror:%u", ctrl & 0xF);
            } else if (ctrl == 0x140) {
                c = "row_mirror";
            } else if (ctrl == 0x141) {
                c = "row_half_mirror";
            } else if (ctrl >= 0x150 && ctrl <= 0x15F) {
                c = fmt("row_share:%u", ctrl & 0xF);
            } else if (ctrl >= 0x160 && ctrl <= 0x16F) {
                c = fmt("row_xmask:%u", ctrl & 0xF);
            } else {
                c = fmt("dpp_ctrl:0x%x", ctrl);
            }
            suffix(c);
            if (r->row_mask != 0xF) suffix(fmt("row_mask:0x%x", unsigned(r->row_mask)));
            if (r->bank_mask != 0xF) suffix(fmt("bank_mask:0x%x", unsigned(r->bank_mask)));
            if (r->bound_ctrl) suffix("bound_ctrl:1");
            if (r->fi) suffix("fi:1");
        } else {
            const DppLanes& l = std::get<DppLanes>(*v.dpp);
            std::vector<std::string> lanes;
            for (unsigned i = 0; i < 8; ++i) lanes.push_back(std::to_string((l.sel >> (3 * i)) & 7));
            suffix("dpp8:[" + join(lanes, ",") + "]");
            if (l.fi) suffix("fi:1");
        }
    }
    return out;
}

std::string operands(const Instruction& inst, std::string_view name, const Fields& fields) {
    std::string out;
    if (const Sop* s = std::get_if<Sop>(&fields)) {
        std::vector<std::string> list;
        if (s->dst) list.push_back(str(*s->dst));
        for (const Operand& o : s->src) list.push_back(str(o));
        std::string regs = join(list);
        if (name == "s_getreg_b32" && s->imm) {
            out = regs + ", " + hwreg(*s->imm);
        } else if ((name == "s_setreg_b32" || name == "s_setreg_imm32_b32") && s->imm) {
            out = hwreg(*s->imm) + ", " + regs;
        } else if (name == "s_call_b64" && s->imm) {
            out = regs + ", " + label(inst, *s->imm);
        } else if (s->imm && regs.empty()) {
            out = fmt("0x%04x", unsigned(*s->imm));
        } else if (s->imm) {
            out = regs + ", " + fmt("0x%04x", unsigned(*s->imm));
        } else {
            out = regs;
        }
    } else if (const Sopp* p = std::get_if<Sopp>(&fields)) {
        if (name == "s_endpgm" || name == "s_barrier" || name == "s_code_end" || name == "s_endpgm_saved" ||
            name == "s_endpgm_ordered_ps_done" || name == "s_wakeup" || name == "s_icache_inv" ||
            name == "s_ttracedata") {
            out = "";
        } else if (name == "s_branch" || name.starts_with("s_cbranch_")) {
            out = label(inst, p->imm);
        } else if (name == "s_waitcnt") {
            out = waitcnt(p->imm);
        } else if (name == "s_sendmsg" || name == "s_sendmsghalt") {
            out = sendmsg(p->imm);
        } else {
            out = fmt("0x%04x", unsigned(p->imm));
        }
    } else if (const Smem* m = std::get_if<Smem>(&fields)) {
        if (m->data) {
            std::string data;
            if ((name == "s_atc_probe" || name == "s_atc_probe_buffer") && m->data->kind == Operand::Kind::Literal) {
                data = atc_immediate(m->data->literal);
            } else {
                data = str(*m->data);
            }
            std::string off;
            bool null = is_null(m->soffset);
            if (m->offset == 0) {
                off = str(m->soffset);
            } else if (null) {
                off = fmt("0x%06x", unsigned(m->offset & 0x1FFFFF));
            } else {
                off = str(m->soffset) + fmt(" offset:0x%06x", unsigned(m->offset & 0x1FFFFF));
            }
            if (name == "s_memtime" || name == "s_memrealtime" || name == "s_get_waveid_in_workgroup") {
                // The data is a counter or the wave ID; these have no address.
                out = data;
            } else {
                out = data + ", " + str(m->base) + ", " + off;
            }
        }
        flags(out, m->cache, false);
    } else if (const Valu* v = std::get_if<Valu>(&fields)) {
        out = valu(name, *v, inst.encoding);
    } else if (const Vintrp* i = std::get_if<Vintrp>(&fields)) {
        std::string src;
        if (name == "v_interp_mov_f32" && i->src.kind == Operand::Kind::Const) {
            static const char* const P[] = {"p10", "p20", "p0"};
            src = i->src.code < 3 ? P[i->src.code] : "p?";
        } else {
            src = str(i->src);
        }
        out = str(i->dst) + ", " + src + ", " + str(i->attr);
    } else if (const Ds* d = std::get_if<Ds>(&fields)) {
        std::vector<std::string> list;
        if (d->dst) list.push_back(str(*d->dst));
        if (d->addr) list.push_back(str(*d->addr));
        for (const Operand& o : d->data) list.push_back(str(o));
        out = join(list);
        if (ds_pair(name)) {
            if (d->offset0 != 0) out += " offset0:" + std::to_string(d->offset0);
            if (d->offset1 != 0) out += " offset1:" + std::to_string(d->offset1);
        } else {
            unsigned offset = unsigned(d->offset1) << 8 | d->offset0;
            if (offset != 0) out += " offset:" + std::to_string(offset);
        }
        if (d->gds) out += " gds";
    } else if (const Buffer* b = std::get_if<Buffer>(&fields)) {
        if (b->data) {
            // An LDS load returns its data to LDS, so vdata is unused.
            if (b->lds && name.find("_load_") != std::string_view::npos) {
                out = str(b->addr) + ", " + str(b->rsrc) + ", " + str(b->soffset);
            } else {
                out = str(*b->data) + ", " + str(b->addr) + ", " + str(b->rsrc) + ", " + str(b->soffset);
            }
            if (b->idxen) out += " idxen";
            if (b->offen) out += " offen";
            if (b->offset != 0) out += " offset:" + std::to_string(b->offset);
            if (b->format) out += " format:[" + buffer_format(*b->format) + "]";
        }
        flags(out, b->cache, false);
        if (b->lds) out += " lds";
        if (b->tfe) out += " tfe";
    } else if (const Image* im = std::get_if<Image>(&fields)) {
        std::string addr;
        if (im->addr.size() == 1) {
            addr = str(im->addr[0]);
        } else {
            std::vector<std::string> list;
            for (const Operand& o : im->addr) list.push_back(str(o));
            addr = "[" + join(list, ",") + "]";
        }
        std::vector<std::string> list = {str(im->data), addr, str(im->rsrc)};
        if (im->sampler) list.push_back(str(*im->sampler));
        out = join(list);
        if (im->dmask != 1) out += fmt(" dmask:0x%x", unsigned(im->dmask));
        static const char* const DIMS[] = {"1D", "2D", "3D", "CUBE", "1D_ARRAY", "2D_ARRAY", "2D_MSAA", "2D_MSAA_ARRAY"};
        out += std::string(" dim:SQ_RSRC_IMG_") + DIMS[im->dim & 7];
        if (im->r128) out += " r128";
        if (im->unorm) out += " unorm";
        flags(out, im->cache, true);
        if (im->a16) out += " a16";
        if (im->d16) out += " d16";
        if (im->tfe) out += " tfe";
        if (im->lwe) out += " lwe";
    } else if (const Flat* f = std::get_if<Flat>(&fields)) {
        std::vector<std::string> list;
        if (f->dst) list.push_back(str(*f->dst));
        list.push_back(f->addr ? str(*f->addr) : "off");
        if (f->data) list.push_back(str(*f->data));
        if (f->saddr) list.push_back(is_null(*f->saddr) ? "off" : str(*f->saddr));
        out = join(list);
        if (f->offset != 0) out += " inst_offset:" + std::to_string(f->offset);
        flags(out, f->cache, false);
        if (f->lds) out += " lds";
    } else if (const Exp* e = std::get_if<Exp>(&fields)) {
        unsigned t = e->target;
        std::string target;
        if (t <= 7) {
            target = "mrt" + std::to_string(t);
        } else if (t == 8) {
            target = "mrtz";
        } else if (t == 9) {
            target = "null";
        } else if (t >= 12 && t <= 15) {
            target = "pos" + std::to_string(t - 12);
        } else if (t == 20) {
            target = "prim";
        } else if (t >= 32 && t <= 63) {
            target = "param" + std::to_string(t - 32);
        } else {
            target = "<target " + std::to_string(t) + ">";
        }
        std::vector<std::string> list = {target};
        for (const auto& s : e->src) list.push_back(s ? str(*s) : "off");
        out = join(list);
        if (e->done) out += " done";
        if (e->compr) out += " compr";
        if (e->vm) out += " vm";
    }
    return out;
}

}  // namespace

namespace sp3 {

std::string scalar_name(std::uint8_t code) {
    if (code <= 105) return "s" + std::to_string(code);
    if (code == 106) return "vcc_lo";
    if (code == 107) return "vcc_hi";
    if (code <= 123) return "ttmp" + std::to_string(code - 108);
    if (code == 124) return "m0";
    if (code == 125) return "null";
    if (code == 126) return "exec_lo";
    return "exec_hi";
}

std::string const_name(std::uint8_t code) {
    if (code >= 128 && code <= 192) return std::to_string(int(code) - 128);
    if (code >= 193 && code <= 208) return std::to_string(192 - int(code));
    switch (code) {
    case 240: return "0.5";
    case 241: return "-0.5";
    case 242: return "1.0";
    case 243: return "-1.0";
    case 244: return "2.0";
    case 245: return "-2.0";
    case 246: return "4.0";
    case 247: return "-4.0";
    case 248: return "0.15915494";
    case 235: return "src_shared_base";
    case 236: return "src_shared_limit";
    case 237: return "src_private_base";
    case 238: return "src_private_limit";
    case 239: return "src_pops_exiting_wave_id";
    case 251: return "src_vccz";
    case 252: return "src_execz";
    case 253: return "src_scc";
    case 254: return "src_lds_direct";
    default: return "<src " + std::to_string(code) + ">";
    }
}

}  // namespace sp3

std::string Operand::to_string() const {
    switch (kind) {
    case Kind::Scalar:
        if (dwords == 1) return sp3::scalar_name(code);
        if (code == 106 && dwords == 2) return "vcc";
        if (code == 126 && dwords == 2) return "exec";
        // Eight-bit arithmetic wraps.
        if (code < 106) return "s[" + std::to_string(code) + ":" + std::to_string(std::uint8_t(code + dwords - 1)) + "]";
        if (code >= 108 && code < 124) {
            return "ttmp[" + std::to_string(code - 108) + ":" + std::to_string(std::uint8_t(code - 108 + dwords - 1)) + "]";
        }
        return sp3::scalar_name(code);
    case Kind::Vector:
        if (dwords == 1) return "v" + std::to_string(reg);
        return "v[" + std::to_string(reg) + ":" + std::to_string(std::uint32_t(reg) + std::uint32_t(dwords) - 1u) + "]";
    case Kind::Const:
        return sp3::const_name(code);
    case Kind::Literal:
        return fmt("lit(0x%08x)", unsigned(literal));
    case Kind::Attr: {
        static const char XYZW[] = {'x', 'y', 'z', 'w'};
        return "attr" + std::to_string(index) + "." + XYZW[chan & 3];
    }
    }
    return {};
}

std::string Operand::debug() const {
    switch (kind) {
    case Kind::Scalar:
        return "Scalar { code: " + std::to_string(code) + ", dwords: " + std::to_string(dwords) + " }";
    case Kind::Vector:
        return "Vector { reg: " + std::to_string(reg) + ", dwords: " + std::to_string(dwords) + " }";
    case Kind::Const:
        return "Const(" + std::to_string(code) + ")";
    case Kind::Literal:
        return "Literal(" + std::to_string(literal) + ")";
    case Kind::Attr:
        return "Attr { index: " + std::to_string(index) + ", chan: " + std::to_string(chan) +
               ", high: " + (high ? "true" : "false") + " }";
    }
    return {};
}

std::string Instruction::text(bool wave64) const {
    std::string_view name = this->name().value_or("?");
    std::string ops = operands(*this, name, fields(wave64));
    if (ops.empty()) return std::string(name);
    return std::string(name) + " " + ops;
}

}  // namespace x2w::vs::gfx10
