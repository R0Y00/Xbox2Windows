// GFX10 machine code: instruction boundaries, encodings and opcodes.
#include "x2w_shader/gfx10/mod.h"

#include <algorithm>
#include <iterator>
#include <map>

#include "x2w_shader/gfx10/opcodes.h"

namespace x2w::vs::gfx10 {

namespace {

ShaderError isa(std::size_t offset, const char* reason) { return ShaderError::isa(offset, reason); }

bool name_is(const std::optional<std::string_view>& n, std::string_view s) { return n && *n == s; }

Flow flow(const Instruction& inst, std::size_t next, std::size_t len) {
    auto target = [&]() -> std::size_t {
        std::int64_t delta = 4 * std::int64_t(std::int16_t(std::uint16_t(inst.words[0])));
        std::int64_t t = std::int64_t(next) + delta;
        if (t < 0 || std::uint64_t(t) >= len) throw isa(inst.offset, "branch target outside the code");
        return std::size_t(t);
    };
    std::string_view name = inst.name().value_or("");
    if (inst.encoding == Encoding::Sopp) {
        if (name == "s_endpgm" || name == "s_endpgm_saved" || name == "s_endpgm_ordered_ps_done" ||
            name == "s_code_end") {
            return Flow::Stop();
        }
        if (name == "s_branch") return Flow::Jump(target());
        if (name.starts_with("s_cbranch_")) return Flow::Branch(target());
    }
    if (inst.encoding == Encoding::Sopk && name == "s_call_b64") return Flow::Branch(target());
    // Indirect jump; `decode` resolves the compiler's far-jump form.
    if (inst.encoding == Encoding::Sop1 && name == "s_setpc_b64") return Flow::Stop();
    return Flow::Next();
}

// Resolves the far jump compilers emit beyond the reach of a branch offset: `s_getpc_b64 s[n:n+1]`,
// `s_add_u32|s_sub_u32 sn, sn, literal`, a carry into s(n+1), `s_setpc_b64 s[n:n+1]`, looking back
// over the straight-line run just decoded. Any other `s_setpc_b64` (e.g. a handover to the next
// merged stage through registers) gives nullopt.
std::optional<std::size_t> far_jump(const Before& before, const Instruction& setpc, std::size_t len) {
    std::uint32_t n = setpc.words[0] & 0xFF;
    std::optional<std::int64_t> delta;
    std::size_t at = setpc.offset;
    for (int k = 0; k < 4; ++k) {
        const Instruction* prev = before(at);
        if (!prev) break;
        if (prev->offset + 4 * prev->words.size() != at) break;
        std::uint32_t w = prev->words[0];
        std::uint32_t sdst = (w >> 16) & 0x7F, ssrc0 = w & 0xFF, ssrc1 = (w >> 8) & 0xFF;
        auto pname = prev->name();
        if (prev->encoding == Encoding::Sop2 && (name_is(pname, "s_add_u32") || name_is(pname, "s_sub_u32")) &&
            sdst == n && ssrc0 == n && ssrc1 == 0xFF && !delta) {
            // The low dword of a signed 64-bit offset: a backward jump adds a negative one (its
            // high dword is -1).
            std::int64_t literal = std::int64_t(std::int32_t(prev->words.at(1)));
            delta = name_is(pname, "s_add_u32") ? literal : -literal;
        } else if (prev->encoding == Encoding::Sop1 && name_is(pname, "s_getpc_b64") && sdst == n) {
            if (!delta) break;
            // s_getpc_b64 yields the address of the instruction after it.
            std::int64_t t = std::int64_t(prev->offset) + 4 + *delta;
            if (t < 0 || std::uint64_t(t) >= len) throw isa(setpc.offset, "far jump target outside the code");
            return std::size_t(t);
        }
        at = prev->offset;
    }
    return std::nullopt;
}

Instruction instruction(std::span<const std::uint8_t> code, std::size_t offset) {
    auto word = [&](std::size_t i) -> std::uint32_t {
        std::size_t at = offset + 4 * i;
        if (at + 4 > code.size()) throw isa(offset, "instruction runs past the end of the code");
        return std::uint32_t(code[at]) | std::uint32_t(code[at + 1]) << 8 | std::uint32_t(code[at + 2]) << 16 |
               std::uint32_t(code[at + 3]) << 24;
    };
    std::uint32_t w = word(0);
    Encoding encoding = Encoding::Sopp;
    std::uint32_t opcode = 0;
    std::size_t len = 0;
    if (w >> 31 == 0) {
        std::uint32_t src0 = w & 0x1FF;
        // Literal constant, SDWA, DPP16, DPP8, DPP8 with fetch-inactive.
        std::size_t extra = (src0 == 0xFF || src0 == 0xF9 || src0 == 0xFA || src0 == 0xE9 || src0 == 0xEA) ? 1 : 0;
        std::uint32_t op = w >> 25;
        if (op == 0x3F) {
            encoding = Encoding::Vop1, opcode = (w >> 9) & 0xFF, len = 1 + extra;
        } else if (op == 0x3E) {
            encoding = Encoding::Vopc, opcode = (w >> 17) & 0xFF, len = 1 + extra;
        } else {
            auto n = name(Encoding::Vop2, std::uint16_t(op));
            bool mk_ak = n && (*n == "v_madmk_f32" || *n == "v_madak_f32" || *n == "v_fmamk_f32" ||
                               *n == "v_fmaak_f32" || *n == "v_madmk_f16" || *n == "v_madak_f16" ||
                               *n == "v_fmamk_f16" || *n == "v_fmaak_f16");
            encoding = Encoding::Vop2, opcode = op, len = 1 + std::max(extra, std::size_t(mk_ak));
        }
    } else if (w >> 30 == 0b10) {
        auto literal1 = [](std::uint32_t a) -> std::size_t { return 1 + (a == 0xFF ? 1 : 0); };
        auto literal2 = [](std::uint32_t a, std::uint32_t b) -> std::size_t {
            return 1 + ((a == 0xFF || b == 0xFF) ? 1 : 0);
        };
        std::uint32_t top = w >> 23;
        if (top == 0x17D) {
            encoding = Encoding::Sop1, opcode = (w >> 8) & 0xFF, len = literal1(w & 0xFF);
        } else if (top == 0x17E) {
            encoding = Encoding::Sopc, opcode = (w >> 16) & 0x7F, len = literal2(w & 0xFF, (w >> 8) & 0xFF);
        } else if (top == 0x17F) {
            encoding = Encoding::Sopp, opcode = (w >> 16) & 0x7F, len = 1;
        } else if (w >> 28 == 0b1011) {
            std::uint32_t op = (w >> 23) & 0x1F;
            bool imm32 = name_is(name(Encoding::Sopk, std::uint16_t(op)), "s_setreg_imm32_b32");
            encoding = Encoding::Sopk, opcode = op, len = 1 + (imm32 ? 1 : 0);
        } else {
            encoding = Encoding::Sop2, opcode = (w >> 23) & 0x7F, len = literal2(w & 0xFF, (w >> 8) & 0xFF);
        }
    } else {
        auto vop3 = [](std::uint32_t w1) -> std::size_t {
            bool lit = (w1 & 0x1FF) == 0xFF || ((w1 >> 9) & 0x1FF) == 0xFF || ((w1 >> 18) & 0x1FF) == 0xFF;
            return 2 + (lit ? 1 : 0);
        };
        switch (w >> 26) {
        case 0x32:
            encoding = Encoding::Vintrp, opcode = (w >> 16) & 0x3, len = 1;
            break;
        case 0x33:
            encoding = Encoding::Vop3p, opcode = (w >> 16) & 0x7F, len = vop3(word(1));
            break;
        case 0x35:
            encoding = Encoding::Vop3, opcode = (w >> 16) & 0x3FF, len = vop3(word(1));
            break;
        case 0x36:
            encoding = Encoding::Ds, opcode = (w >> 18) & 0xFF, len = 2;
            break;
        case 0x37:
            switch ((w >> 14) & 3) {
            case 0:
                encoding = Encoding::Flat;
                break;
            case 1:
                encoding = Encoding::Scratch;
                break;
            case 2:
                encoding = Encoding::Global;
                break;
            default:
                throw isa(offset, "reserved FLAT segment");
            }
            opcode = (w >> 18) & 0x7F, len = 2;
            break;
        case 0x38:
            encoding = Encoding::Mubuf, opcode = ((w >> 18) & 0x7F) | ((w >> 25) & 1) << 7, len = 2;
            break;
        case 0x3A:
            encoding = Encoding::Mtbuf, opcode = ((w >> 16) & 0x7) | ((word(1) >> 21) & 1) << 3, len = 2;
            break;
        // Bits 2:1 count extra dwords of non-sequential address registers.
        case 0x3C:
            encoding = Encoding::Mimg, opcode = ((w >> 18) & 0x7F) | (w & 1) << 7, len = 2 + ((w >> 1) & 3);
            break;
        case 0x3D:
            encoding = Encoding::Smem, opcode = (w >> 18) & 0xFF, len = 2;
            break;
        case 0x3E:
            encoding = Encoding::Exp, opcode = 0, len = 2;
            break;
        default:
            throw isa(offset, "unknown encoding");
        }
    }
    std::vector<std::uint32_t> words;
    words.reserve(len);
    for (std::size_t i = 0; i < len; ++i) words.push_back(word(i));
    return Instruction(offset, encoding, std::uint16_t(opcode), std::move(words));
}

}  // namespace

const char* debug(Encoding e) {
    static const char* const NAMES[] = {
        "Sop2", "Sopk", "Sop1", "Sopc", "Sopp", "Smem", "Vop2", "Vop1", "Vopc", "Vop3",
        "Vop3p", "Vintrp", "Ds", "Mubuf", "Mtbuf", "Mimg", "Exp", "Flat", "Global", "Scratch",
    };
    return NAMES[static_cast<int>(e)];
}

Instruction::Instruction(std::size_t offset_, Encoding encoding_, std::uint16_t opcode_,
                         std::vector<std::uint32_t> words_)
    : offset(offset_), encoding(encoding_), opcode(opcode_), words(std::move(words_)),
      name_(gfx10::name(encoding_, opcode_)) {}

std::vector<Instruction> decode(std::span<const std::uint8_t> code) {
    if (code.size() % 4 != 0) throw isa(code.size(), "code size is not a multiple of 4");
    std::map<std::size_t, Instruction> found;
    std::vector<std::size_t> pending{0};
    Before before = [&](std::size_t at) -> const Instruction* {
        auto it = found.lower_bound(at);
        if (it == found.begin()) return nullptr;
        return &std::prev(it)->second;
    };
    while (!pending.empty()) {
        std::size_t offset = pending.back();
        pending.pop_back();
        while (!found.contains(offset)) {
            Instruction inst = instruction(code, offset);
            std::size_t next = offset + 4 * inst.words.size();
            Flow f = control(inst, code.size(), before);
            found.emplace(offset, std::move(inst));
            if (f.kind == Flow::Kind::Stop) break;
            if (f.kind == Flow::Kind::Jump) {
                offset = f.target;
            } else {
                if (f.kind == Flow::Kind::Branch) pending.push_back(f.target);
                offset = next;
            }
        }
    }
    std::vector<Instruction> list;
    list.reserve(found.size());
    for (auto& entry : found) list.push_back(std::move(entry.second));
    for (std::size_t i = 0; i + 1 < list.size(); ++i) {
        if (list[i].offset + 4 * list[i].words.size() > list[i + 1].offset) {
            throw isa(list[i + 1].offset, "a branch lands inside an instruction");
        }
    }
    return list;
}

Flow control(const Instruction& inst, std::size_t len, const Before& before) {
    Flow f = flow(inst, inst.offset + 4 * inst.words.size(), len);
    if (f.kind == Flow::Kind::Stop && name_is(inst.name(), "s_setpc_b64")) {
        auto t = far_jump(before, inst, len);
        return t ? Flow::Jump(*t) : Flow::Stop();
    }
    return f;
}

std::optional<std::string_view> name(Encoding encoding, std::uint16_t opcode) {
    namespace t = opcodes;
    opcodes::Table table;
    std::uint16_t op = opcode;
    switch (encoding) {
    case Encoding::Sop2: table = t::SOP2; break;
    case Encoding::Sopk: table = t::SOPK; break;
    case Encoding::Sop1: table = t::SOP1; break;
    case Encoding::Sopc: table = t::SOPC; break;
    case Encoding::Sopp: table = t::SOPP; break;
    case Encoding::Smem: table = t::SMEM; break;
    case Encoding::Vop2: table = t::VOP2; break;
    case Encoding::Vop1: table = t::VOP1; break;
    case Encoding::Vopc: table = t::VOPC; break;
    // VOP3 also encodes the VOPC, VOP2 and VOP1 opcodes at fixed bases.
    case Encoding::Vop3:
        if (opcode < 0x100) {
            table = t::VOPC;
        } else if (opcode < 0x140) {
            table = t::VOP2, op = std::uint16_t(opcode - 0x100);
        } else if (opcode >= 0x180 && opcode < 0x200) {
            table = t::VOP1, op = std::uint16_t(opcode - 0x180);
        } else {
            table = t::VOP3;
        }
        break;
    case Encoding::Vop3p: table = t::VOP3P; break;
    case Encoding::Vintrp: table = t::VINTRP; break;
    case Encoding::Ds: table = t::DS; break;
    case Encoding::Mubuf: table = t::MUBUF; break;
    case Encoding::Mtbuf: table = t::MTBUF; break;
    case Encoding::Mimg: table = t::MIMG; break;
    case Encoding::Exp: return std::string_view("exp");
    case Encoding::Flat: table = t::FLAT; break;
    case Encoding::Global: table = t::GLOBAL; break;
    case Encoding::Scratch: table = t::SCRATCH; break;
    }
    auto it = std::lower_bound(table.begin(), table.end(), op,
                               [](const opcodes::Entry& e, std::uint16_t v) { return e.op < v; });
    if (it != table.end() && it->op == op) return it->name;
    return std::nullopt;
}

}  // namespace x2w::vs::gfx10
