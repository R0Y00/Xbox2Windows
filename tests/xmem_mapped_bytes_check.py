"""x2w_xmem_mapped_bytes answers from the page registry, without touching memory, inside xmem's reservations."""
import ctypes as C
from pathlib import Path
import sys

PAGE_BYTES = 65536
WIN_RESERVE, WIN_COMMIT, WIN_RELEASE = 0x2000, 0x1000, 0x8000
NOACCESS, READWRITE = 1, 4


def bind(library, name, args, result):
    function = getattr(library, name)
    function.argtypes, function.restype = args, result
    return function


def main(path):
    kernel = C.WinDLL("kernel32.dll", use_last_error=True)
    kernel.SetEnvironmentVariableW.argtypes = [C.c_wchar_p, C.c_wchar_p]
    assert kernel.SetEnvironmentVariableW("X2W_XMEM_PERMISSIVE", "1")
    library = C.WinDLL(str(Path(path).resolve()), use_last_error=True, winmode=0x900)
    pointer, number, dword = C.c_void_p, C.c_size_t, C.c_ulong
    mapped_bytes = bind(library, "x2w_xmem_mapped_bytes", [number, number, C.POINTER(number)], C.c_int)
    virtual_alloc = bind(library, "XMemVirtualAlloc", [pointer, number, dword, C.c_ulonglong, dword], pointer)
    allocate = bind(library, "XMemAllocatePhysicalPages", [number, C.POINTER(number), C.POINTER(number)], C.c_int)
    release = bind(library, "XMemFreePhysicalPages", [number, C.POINTER(number)], C.c_int)
    contiguous = bind(library, "XMemMapPhysicalPages", [pointer, number, C.POINTER(number)], pointer)
    free = bind(C.WinDLL("kernelbase.dll", use_last_error=True), "VirtualFree", [pointer, number, dword], C.c_int)

    def query(address, size):
        value = number(12345)
        answered = mapped_bytes(address, size, C.byref(value))
        return value.value if answered else None

    # Outside every reservation xmem has no answer: the caller probes.
    outside = (C.c_char * 16)()
    assert query(C.addressof(outside), 16) is None, "answered for ordinary process memory"
    assert not mapped_bytes(C.addressof(outside), 16, None), "accepted a null result"

    base = virtual_alloc(None, 4 * PAGE_BYTES, WIN_RESERVE, 0, NOACCESS)
    assert base, C.get_last_error()
    count, pages = number(2), (number * 2)()
    assert allocate(0x280, C.byref(count), pages) and count.value == 2
    try:
        assert query(base, PAGE_BYTES) == 0, "an unmapped reservation page was not reported unmapped"
        assert query(base + 0x1000, 0x2000) == 0
        # Pages 1 and 2 mapped: page 0 is still unmapped, a range from page 1 is mapped through page 2.
        tokens = (number * 2)(pages[0], pages[1])
        assert contiguous(base + PAGE_BYTES, 2, tokens) == base + PAGE_BYTES, C.get_last_error()
        assert query(base, 3 * PAGE_BYTES) == 0, "a range whose first page is unmapped was reported mapped"
        assert query(base + PAGE_BYTES + 0x10, 0x100) == 0x100
        assert query(base + PAGE_BYTES + 0x10, 3 * PAGE_BYTES - 0x10) == 2 * PAGE_BYTES - 0x10, "mapped prefix"
        C.memmove(base + PAGE_BYTES, b"mapped", 6)  # the mapped prefix is really there
        # A range past the end of the reservation has no answer.
        assert query(base + 3 * PAGE_BYTES, 2 * PAGE_BYTES) is None, "answered past the reservation"
        assert contiguous(base + PAGE_BYTES, 2, None) == base + PAGE_BYTES
        assert query(base + PAGE_BYTES, PAGE_BYTES) == 0, "an unmapped page was still reported mapped"
    finally:
        release(2, pages)
        free(base, 0, WIN_RELEASE)
    # Released: Windows no longer shows a reservation, so the registry's stale entry gives no answer.
    assert query(base, PAGE_BYTES) is None, "answered for a released reservation"
    print("mapped bytes: outside=no answer, unmapped=0, mapped prefix=0x1fff0, past the end=no answer, released=no answer")


if __name__ == "__main__":
    main(sys.argv[1])
