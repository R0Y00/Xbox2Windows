// Lifetime boundary of the desktop D3D12 objects the stand-in keeps in function-local statics:
// the indexed-validation pipeline (startup_stubs/d3d12_draw.inl), the rectangle-list argument
// converter (startup_stubs/rect_indirect_args.h), the placed page aliases of the translated
// indirect address route (startup_stubs/d3d12_xs_device.cpp) and the draw-dump readbacks
// (startup_stubs/draw_dump_tracking.h). #2572.
//
// These caches hold their objects for as long as this module is loaded -- a conversion or a validation
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
//
// The accounting below is what makes that boundary observable. startup_teardown_unload loads this
// module, warms one cache, and has to show both paths: that the cache releases what it owns when the
// module is unloaded, and that it detaches instead of releasing when the process exits. The two
// variables are read when a cache is destroyed and absent from an ordinary run:
//
//   X2W_TEARDOWN_REPORT=1        the cache prints one line on stderr as it is destroyed:
//                                  x2w-teardown <cache> detached=<n> released=<n>
//                                where n counts the owned references (objects), not the entries. The
//                                line is written with WriteFile straight to the stderr handle: these
//                                destructors run during this module's CRT teardown, so a stdio buffer
//                                would never be flushed. No file is written and no buffer is deferred.
//   X2W_TEARDOWN_NO_GUARD=<c>    disables the detach of cache <c> (pages, readbacks, or all): that
//                                cache then releases normally during process exit, which is the crash
//                                path this boundary exists for. Absent, every guard behaves as it did.
//                                This is the negative control startup_teardown_unload runs against
//                                itself: with the variable set, its termination check must fail.
//
// Neither read is cached in a static: a function-local static here would be initialized for the first
// time inside the destructor, during teardown, after the atexit table has already run.
#pragma once

#include <windows.h>

#include <cstdio>
#include <cstring>

namespace x2w_startup {

inline bool d3d12_process_terminating = false;

// True when the caller asked the named cache to release instead of detach during process exit.
inline bool teardown_guard_disabled(const char* cache) noexcept
{
    const DWORD saved = GetLastError();
    char value[64]{};
    const DWORD length = GetEnvironmentVariableA("X2W_TEARDOWN_NO_GUARD", value, sizeof(value));
    SetLastError(saved);
    if (!length || length >= sizeof(value)) return false;
    return std::strcmp(value, "all") == 0 || std::strcmp(value, cache) == 0;
}

// One stable line per destroyed cache, on stderr, only when X2W_TEARDOWN_REPORT is set.
inline void teardown_report(const char* cache, unsigned detached, unsigned released) noexcept
{
    const DWORD saved = GetLastError();
    char on[2]{};
    const bool wanted = GetEnvironmentVariableA("X2W_TEARDOWN_REPORT", on, sizeof(on)) != 0;
    if (wanted) {
        char line[128];
        const int length = std::snprintf(line, sizeof(line), "x2w-teardown %s detached=%u released=%u\r\n", cache,
                                         detached, released);
        if (length > 0) {
            DWORD written = 0;
            const int bytes = length < static_cast<int>(sizeof(line)) ? length : static_cast<int>(sizeof(line)) - 1;
            WriteFile(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(bytes), &written, nullptr);
        }
    }
    SetLastError(saved);
}

}  // namespace x2w_startup
