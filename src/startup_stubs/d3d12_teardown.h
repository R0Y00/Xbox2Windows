// Lifetime boundary of the desktop D3D12 objects the stand-in keeps in function-local statics:
// the indexed-validation pipeline (startup_stubs/d3d12_draw.inl) and the rectangle-list argument
// converter (startup_stubs/rect_indirect_args.h). #2572.
//
// Both caches hold their objects for as long as this module is loaded -- a conversion or a validation
// may still be queued when the call that built the entry returns -- so they are destroyed during this
// module's CRT teardown. That teardown runs at two different moments:
//
//   process exit      Windows detaches every module after the display driver's DLL; releasing a
//                     D3D12 object then reaches a stopped driver (amdxc64, offset 0x35f2a19, through
//                     D3D12Core!CPipelineState::FinalRelease) and the loader's __fastfail(7) turns
//                     that into 0xC0000409. Nothing may be released.
//   FreeLibrary       the module is unloaded while the process and the driver stay alive: every owned
//                     object must be released then, or it would be leaked for the rest of the process.
//
// DllMain's lpReserved is what tells the two detaches apart: non-null means the process is
// terminating, null means the loader is unloading this module. src/d3d12_xs_shim.cpp sets the flag
// below from its DllMain, which runs before this module's atexit table -- the static runtime reaches
// the user entry before it runs the table on detach, as the cpu_native_crt_lifecycle fixture measures
// (docs/experiments.md) -- so the destructors read it without a lock.
//
// The variable is defined inline: each module that includes this header has its own copy, so the
// caches compiled into another module (the test executables include rect_indirect_args.h) keep
// releasing normally, and only the DLL that owns the DllMain can silence its own teardown. Being a
// zero-initialized bool, it needs no dynamic initialization and no destruction order.
#pragma once

namespace x2w_startup {

inline bool d3d12_process_terminating = false;

}  // namespace x2w_startup
