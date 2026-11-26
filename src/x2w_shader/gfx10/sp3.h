// Instruction text in AMD's SP3 assembly syntax. The entry points are Instruction::text (mod.h) and
// Operand::to_string (fields.h); these are its helpers.
#pragma once

#include <cstdint>
#include <string>

namespace x2w::vs::gfx10::sp3 {

std::string scalar_name(std::uint8_t code);
std::string const_name(std::uint8_t code);

}  // namespace x2w::vs::gfx10::sp3
