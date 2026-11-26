// x2w_shaders without the C++ translator port: builtin_translator() is null. When library x2w_shader exists,
// CMake builds x2w_shaders_vs.cpp in place of this file, which adapts x2w::vs::translate to Translator.
#include "x2w_shaders_translate.h"

namespace shader_db {

const Translator* builtin_translator() { return nullptr; }

}  // namespace shader_db
