// Opt-in Tracy zones, memory pools and plots for the stand-ins (CMake option X2W_TRACY, a local measurement build).
// Without the option every macro expands to nothing or a no-op and no Tracy header is included.
#pragma once

#if X2W_TRACY
#include <tracy/Tracy.hpp>
#define X2W_ZONE(name) ZoneScopedN(name)
#define X2W_FRAME_MARK FrameMark
// A named memory pool: every X2W_ALLOC needs its X2W_FREE with the same pool name.
#define X2W_ALLOC(pointer, bytes, pool) TracyAllocN(pointer, bytes, pool)
#define X2W_FREE(pointer, pool) TracyFreeN(pointer, pool)
#define X2W_PLOT(name, value) TracyPlot(name, value)
#else
#define X2W_ZONE(name)
#define X2W_FRAME_MARK
#define X2W_ALLOC(pointer, bytes, pool) ((void)0)
#define X2W_FREE(pointer, pool) ((void)0)
#define X2W_PLOT(name, value) ((void)0)
#endif
