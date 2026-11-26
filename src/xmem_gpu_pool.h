#pragma once
#include <windows.h>

namespace x2w_gpu_pool {
enum class Mode { off, moves, all };
Mode mode() noexcept;
// The next unused pool section of at least bytes, opened read-write, or nullptr (pool off, not running,
// exhausted, owned by another process, or too small). The contents are whatever the previous user left: the caller
// overwrites or zeroes them.
HANDLE take(ULONG64 bytes) noexcept;
}  // namespace x2w_gpu_pool
