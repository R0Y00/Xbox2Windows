// Xbox Series shaders to host shaders.
//
// An Xbox Series shader is a standard DXBC container (dxbc.h). Its `DXIL` part (dxil.h) may carry
// Xbox additions around the LLVM bitcode: an `XDAT` table before it indexing blobs after it, among
// them `xbop` (compiler options) and the zlib-compressed GFX10 code. The bitcode may use console-only
// intrinsics, so the GFX10 code is translated in general.
//
// Translation failures throw ShaderError. Use catch_shader_error or try_shader to capture a refusal.
#pragma once

#include "x2w_shader/error.h"

// Public container, program and hardware stage types from the module headers, including native.h.
#include "x2w_shader/dxbc.h"
#include "x2w_shader/dxil.h"

namespace x2w::vs {
using dxbc::Container;
using dxbc::FourCC;
using dxbc::Part;
using dxil::DxilProgram;
using dxil::ShaderKind;
using dxil::XboxExtras;
}  // namespace x2w::vs

#if __has_include("x2w_shader/native.h")
#include "x2w_shader/native.h"

namespace x2w::vs {
using native::HwStage;
using native::native_programs;
using native::NativeProgram;
using native::ResourceType;
using native::RootParam;
using native::RootType;
using native::UsedResource;
}  // namespace x2w::vs
#endif
