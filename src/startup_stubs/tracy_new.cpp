// X2W_TRACY builds only (CMakeLists.txt adds this file to each instrumented DLL): the C++ heap of the DLL as a
// Tracy memory pool. The array, sized and nothrow forms of the runtime forward to these four.
#include "profile_zone.h"

#include <cstdlib>
#include <malloc.h>
#include <new>

void* operator new(std::size_t bytes)
{
    void* block = std::malloc(bytes ? bytes : 1);
    if (!block) throw std::bad_alloc();
    X2W_ALLOC(block, bytes, X2W_TRACY_HEAP);
    return block;
}

void* operator new(std::size_t bytes, std::align_val_t alignment)
{
    void* block = _aligned_malloc(bytes ? bytes : 1, static_cast<std::size_t>(alignment));
    if (!block) throw std::bad_alloc();
    X2W_ALLOC(block, bytes, X2W_TRACY_HEAP);
    return block;
}

void operator delete(void* block) noexcept
{
    if (block) X2W_FREE(block, X2W_TRACY_HEAP);
    std::free(block);
}

void operator delete(void* block, std::align_val_t) noexcept
{
    if (block) X2W_FREE(block, X2W_TRACY_HEAP);
    _aligned_free(block);
}
