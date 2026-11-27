"""Check the physical-page snapshot the startup heartbeat reports, and the slab accounting behind it."""
import ctypes as C
from pathlib import Path
import sys

PAGE_BYTES = 65536
PHYSICAL_64K = 0x20400000
WIN_RESERVE = 0x2000
RESERVE_PLACEHOLDER = 0x40000
WIN_RELEASE = 0x8000
NOACCESS = 1


def bind(library, name, args, result):
    function = getattr(library, name)
    function.argtypes, function.restype = args, result
    return function


def main(path):
    library = C.WinDLL(str(Path(path).resolve()), use_last_error=True, winmode=0x900)
    kernel = C.WinDLL("kernelbase.dll", use_last_error=True)
    pointer, number = C.c_void_p, C.c_size_t
    stats = bind(library, "x2w_xmem_page_stats", [C.POINTER(C.c_ulonglong)], C.c_int)
    slab_stats = bind(library, "x2w_xmem_slab_stats", [C.POINTER(C.c_ulonglong)] * 3, C.c_int)
    allocate = bind(library, "XMemAllocatePhysicalPages", [number, C.POINTER(number), C.POINTER(number)], C.c_int)
    release = bind(library, "XMemFreePhysicalPages", [number, C.POINTER(number)], C.c_int)
    contiguous = bind(library, "XMemMapPhysicalPages", [pointer, number, C.POINTER(number)], pointer)
    write_physical = bind(library, "x2w_xmem_write_physical",
                          [C.POINTER(number), number, number, C.c_char_p, number], C.c_int)
    reserve = bind(kernel, "VirtualAlloc2", [pointer, pointer, number, C.c_ulong, C.c_ulong, pointer, C.c_ulong],
                   pointer)
    free = bind(kernel, "VirtualFree", [pointer, number, C.c_ulong], C.c_int)

    def placeholder():
        base = reserve(pointer(-1), None, PAGE_BYTES, WIN_RESERVE | RESERVE_PLACEHOLDER, NOACCESS, None, 0)
        assert base, C.get_last_error()
        return base

    def sections():
        value = C.c_ulonglong()
        assert stats(C.byref(value)), "x2w_xmem_page_stats refused a valid snapshot"
        return value.value

    def slabs():
        values = [C.c_ulonglong() for _ in range(3)]
        assert slab_stats(*map(C.byref, values)), "x2w_xmem_slab_stats refused a valid snapshot"
        return tuple(value.value for value in values)

    def snapshot(stage, expected, slab_counts):
        actual = sections()
        assert actual == expected, f"{stage}: {actual} live sections, expected {expected}"
        # (slabs open, slots in use, slots free). A slab holds 4096 slots, stays open while any slot is live,
        # and is closed once all of its slots are released.
        counts = slabs()
        assert counts == slab_counts, f"{stage}: slabs, in use, free = {counts}, expected {slab_counts}"
        assert counts[1] == actual, f"{stage}: {counts[1]} slots in use but {actual} pages hold a slot"
        return actual

    assert not stats(None), "x2w_xmem_page_stats accepted a null snapshot"
    assert not slab_stats(None, C.byref(C.c_ulonglong()), C.byref(C.c_ulonglong())), "a null slab count was accepted"
    assert not slab_stats(C.byref(C.c_ulonglong()), None, C.byref(C.c_ulonglong())), "a null in-use count was accepted"
    assert not slab_stats(C.byref(C.c_ulonglong()), C.byref(C.c_ulonglong()), None), "a null free count was accepted"
    assert slabs() == (0, 0, 0), "slabs exist before any page was used"
    first, second = placeholder(), placeholder()
    try:
        # A grant on its own names pages; no section exists until a page is used.
        count, pages = number(3), (number * 3)()
        assert allocate(PHYSICAL_64K, C.byref(count), pages) and count.value == 3
        snapshot("after the grant", 0, (0, 0, 0))

        # Mapping two pages backs them with two 64 KiB slots of one slab.
        one, other = number(pages[0]), number(pages[1])
        assert contiguous(first, 1, C.byref(one)) == first
        assert contiguous(second, 1, C.byref(other)) == second
        snapshot("two mapped pages", 2, (1, 2, 4094))

        # Unmapping releases the views, not the sections: the pages stay committed until the grant is.
        assert contiguous(first, 1, None) == first
        assert contiguous(second, 1, None) == second
        snapshot("after unmapping", 2, (1, 2, 4094))

        # A second grant backed by a page-token write shares the same slab.
        extra_count, extra_pages = number(1), (number * 1)()
        assert allocate(PHYSICAL_64K, C.byref(extra_count), extra_pages) and extra_count.value == 1
        token = number(extra_pages[0] << 4)
        assert write_physical(C.byref(token), 1, 0, b"section", len(b"section"))
        snapshot("write through page tokens", 3, (1, 3, 4093))

        # Releasing the first grant leaves the slab open while the second grant still holds a slot.
        assert release(3, pages)
        snapshot("after releasing first grant", 1, (1, 1, 4095))

        # Releasing the last slot in the slab closes and reclaims it.
        assert release(1, extra_pages)
        snapshot("after releasing the written grant", 0, (0, 0, 0))
    finally:
        free(first, 0, WIN_RELEASE)
        free(second, 0, WIN_RELEASE)
    print("physical pages: grant=0 mapped=2 unmapped=2 extra=1 first_released=1 all_released=0")


if __name__ == "__main__":
    main(sys.argv[1])
