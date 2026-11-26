// GFX10 machine code: instruction boundaries, encodings and opcodes.
//
// Instructions are found by following control flow from offset 0, not by a linear sweep: compilers
// place data (constant tables) in the code region after the end of the program.
// Instruction::fields decodes the operand fields and Instruction::text prints them in AMD's SP3
// syntax.
//
// Encodings by the top bits of the first dword:
//
//   0xxxxxxx  VOP2; 0111111 VOP1; 0111110 VOPC
//   10        SOP2; 1011 SOPK; 101111101 SOP1; 101111110 SOPC; 101111111 SOPP
//   110010 VINTRP  110011 VOP3P  110101 VOP3  110110 DS  110111 FLAT/GLOBAL/SCRATCH
//   111000 MUBUF   111010 MTBUF  111100 MIMG  111101 SMEM  111110 EXP
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "x2w_shader/error.h"
#include "x2w_shader/gfx10/fields.h"

namespace x2w::vs::gfx10 {

enum class Encoding : std::uint8_t {
    Sop2,
    Sopk,
    Sop1,
    Sopc,
    Sopp,
    Smem,
    Vop2,
    Vop1,
    Vopc,
    Vop3,
    Vop3p,
    Vintrp,
    Ds,
    Mubuf,
    Mtbuf,
    Mimg,
    Exp,
    Flat,
    Global,
    Scratch,
};

// Diagnostic spelling for Encoding: "Sop2", "Vop3p",...
const char* debug(Encoding e);

class Instruction {
public:
    // Byte offset in the program.
    std::size_t offset = 0;
    Encoding encoding = Encoding::Sopp;
    std::uint16_t opcode = 0;
    // Every dword of the instruction, including literals and extra dwords.
    std::vector<std::uint32_t> words;

    Instruction() = default;
    // An instruction of these words, its name resolved.
    Instruction(std::size_t offset, Encoding encoding, std::uint16_t opcode, std::vector<std::uint32_t> words);

    // AMD's name for the opcode, if it is in our table (static storage, NUL-terminated).
    std::optional<std::string_view> name() const { return name_; }

    // Decodes the operand fields. `wave64` sizes lane masks (fields.cpp).
    Fields fields(bool wave64) const;
    // The instruction in SP3 syntax, e.g. `v_add_f32 v1, 0.5, v8`. Branch targets are written
    // `label_<byte offset>` (sp3.cpp).
    std::string text(bool wave64) const;

    bool operator==(const Instruction&) const = default;

private:
    // name() of `encoding` and `opcode`, resolved once at decode time.
    std::optional<std::string_view> name_;
    Valu valu(std::string_view name, std::uint8_t lanes) const;
};

// Decodes every instruction reachable from offset 0, sorted by offset. Throws ShaderError::isa.
std::vector<Instruction> decode(std::span<const std::uint8_t> code);

// Control flow after an instruction.
struct Flow {
    enum class Kind : std::uint8_t {
        Next,
        // End of the program, or an indirect jump that is not a far jump.
        Stop,
        Jump,
        // Conditional: to the target or the next instruction.
        Branch,
    };
    Kind kind = Kind::Next;
    std::size_t target = 0;  // Jump, Branch

    static constexpr Flow Next() { return {Kind::Next, 0}; }
    static constexpr Flow Stop() { return {Kind::Stop, 0}; }
    static constexpr Flow Jump(std::size_t t) { return {Kind::Jump, t}; }
    static constexpr Flow Branch(std::size_t t) { return {Kind::Branch, t}; }
    bool operator==(const Flow&) const = default;
};

// `before(at)`: the decoded instruction with the highest offset below `at`, or nullptr.
using Before = std::function<const Instruction*(std::size_t)>;

// Control flow after `inst`, resolving far jumps (pub(crate)). Throws ShaderError::isa.
Flow control(const Instruction& inst, std::size_t len, const Before& before);

// The opcode table lookup behind Instruction::name.
std::optional<std::string_view> name(Encoding encoding, std::uint16_t opcode);

}  // namespace x2w::vs::gfx10
