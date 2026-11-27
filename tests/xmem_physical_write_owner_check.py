"""Check that x2w_xmem_write_physical writes only into granted physical pages.

A physical page gets its 64 KiB section on first use and loses it when its grant is released
(docs/startup-runtime.md), and a fresh grant reads as zero. A page-token write to a page nobody owns
must therefore be refused the way XMemMapPhysicalPages refuses it (FALSE, ERROR_INVALID_PARAMETER):
otherwise it leaves a section outside any grant, and the next grant of that page number maps the
written bytes instead of zeros.

usage: xmem_physical_write_owner_check.py <xmem.dll>
Each case runs in a process of its own (--child <case>), so that its page numbers and the section
count start from a shim nothing else has used.
"""
import ctypes as C
import os
from pathlib import Path
import subprocess
import sys

PAGE_BYTES = 65536
PHYSICAL_64K = 0x20400000
WIN_RESERVE = 0x2000
RESERVE_PLACEHOLDER = 0x40000
WIN_RELEASE = 0x8000
NOACCESS = 1
ERROR_INVALID_PARAMETER = 87
CAPACITY_PAGES = 1 << 18  # era_physical::capacity_pages
CHUNK = 4096
CHECKS_FAILED = 2  # exit code of a case that ran to its end with failed checks


def bind(library, name, args, result):
    function = getattr(library, name)
    function.argtypes, function.restype = args, result
    return function


class Shim:
    def __init__(self, path):
        library = C.WinDLL(str(Path(path).resolve()), use_last_error=True, winmode=0x900)
        kernel = C.WinDLL("kernelbase.dll", use_last_error=True)
        pointer, number = C.c_void_p, C.c_size_t
        self.pointer, self.number = pointer, number
        self.stats = bind(library, "x2w_xmem_page_stats", [C.POINTER(C.c_ulonglong)], C.c_int)
        self.allocate = bind(library, "XMemAllocatePhysicalPages", [number, C.POINTER(number), C.POINTER(number)], C.c_int)
        self.release = bind(library, "XMemFreePhysicalPages", [number, C.POINTER(number)], C.c_int)
        self.contiguous = bind(library, "XMemMapPhysicalPages", [pointer, number, C.POINTER(number)], pointer)
        self.write_physical = bind(library, "x2w_xmem_write_physical",
                                   [C.POINTER(number), number, number, C.c_char_p, number], C.c_int)
        self.reserve = bind(kernel, "VirtualAlloc2",
                            [pointer, pointer, number, C.c_ulong, C.c_ulong, pointer, C.c_ulong], pointer)
        self.free = bind(kernel, "VirtualFree", [pointer, number, C.c_ulong], C.c_int)
        self.failures = 0

    def check(self, condition, message):
        if not condition:
            self.failures += 1
            print(f"  FAIL: {message}", flush=True)

    def sections(self):
        value = C.c_ulonglong()
        assert self.stats(C.byref(value)), "x2w_xmem_page_stats refused a valid snapshot"
        return value.value

    def grant(self, count):
        got, pages = self.number(count), (self.number * count)()
        assert self.allocate(PHYSICAL_64K, C.byref(got), pages) and got.value == count, C.get_last_error()
        return pages

    def write(self, tokens, offset, data):
        """One page-token write. Returns its result and the last error it left."""
        array = (self.number * len(tokens))(*tokens)
        C.set_last_error(0)
        result = self.write_physical(array, len(tokens), offset, data, len(data))
        return bool(result), C.get_last_error()

    def accepted(self, what, tokens, offset, data):
        ok, error = self.write(tokens, offset, data)
        self.check(ok, f"{what}: a write to a granted page was refused (last error {error})")

    def refused(self, what, tokens, offset, data):
        ok, error = self.write(tokens, offset, data)
        self.check(not ok, f"{what}: the write was accepted")
        self.check(ok or error == ERROR_INVALID_PARAMETER,
                   f"{what}: last error {error}, expected ERROR_INVALID_PARAMETER ({ERROR_INVALID_PARAMETER})")

    def live(self, what, expected):
        actual = self.sections()
        self.check(actual == expected, f"{what}: {actual} live sections, expected {expected}")

    def read(self, page, offset, size):
        """Maps one granted page into a placeholder of its own and returns size bytes at offset."""
        base = self.reserve(self.pointer(-1), None, PAGE_BYTES, WIN_RESERVE | RESERVE_PLACEHOLDER, NOACCESS, None, 0)
        assert base, C.get_last_error()
        try:
            token = self.number(page)
            assert self.contiguous(base, 1, C.byref(token)) == base, C.get_last_error()
            data = C.string_at(base + offset, size)
            assert self.contiguous(base, 1, None) == base, C.get_last_error()
            return data
        finally:
            self.free(base, 0, WIN_RELEASE)

    def grant_again(self, page):
        """Grants pages until the registry issues this page number (page numbers are reused)."""
        chunk = CHUNK
        while chunk:
            got, pages = self.number(chunk), (self.number * chunk)()
            if not self.allocate(PHYSICAL_64K, C.byref(got), pages):
                chunk //= 2  # fewer pages than that are left
            elif page in pages:
                return
        raise AssertionError(f"page {page} was not granted")

    def reads_zero(self, what, page, offset):
        data = self.read(page, offset, 5)
        self.check(data == bytes(5), f"{what}: the page reads {data!r}, expected zeros")


def never_granted(shim):
    """A page number that no grant has named yet."""
    held = shim.grant(1)[0]
    page = held + 1 if held < CAPACITY_PAGES else 1
    shim.refused("never granted", [page << 4], 16, b"stale")
    shim.live("never granted, after the refused write", 0)
    shim.grant_again(page)
    shim.reads_zero("never granted, first grant of the page", page, 16)


def released(shim):
    """A granted page takes the write (control). After the release the same token is refused."""
    pages = shim.grant(1)
    page = pages[0]
    shim.accepted("control", [page << 4], 16, b"owned")
    shim.live("control, after the write", 1)
    data = shim.read(page, 16, 5)
    shim.check(data == b"owned", f"control: the granted page reads {data!r}, expected b'owned'")
    assert shim.release(1, pages), C.get_last_error()
    shim.live("control, after the release", 0)
    shim.refused("released", [page << 4], 16, b"stale")
    shim.live("released, after the refused write", 0)
    shim.grant_again(page)
    shim.reads_zero("released, next grant of the page", page, 16)


def regrant(shim):
    """Control: with no write in between, a page written under one grant reads zero under the next."""
    pages = shim.grant(1)
    page = pages[0]
    shim.accepted("control", [page << 4], 16, b"owned")
    assert shim.release(1, pages), C.get_last_error()
    shim.live("control, after the release", 0)
    shim.grant_again(page)
    shim.reads_zero("control, next grant of the page", page, 16)


def second_token(shim):
    """Every token of a write is checked, not only the first."""
    first, second = shim.grant(1), shim.grant(1)
    page, owned = first[0], second[0]
    assert shim.release(1, first), C.get_last_error()
    shim.refused("second token", [owned << 4, page << 4], 0, bytes(PAGE_BYTES) + b"stale")
    shim.grant_again(page)
    shim.reads_zero("second token, next grant of the page", page, 0)


def permissive(shim):
    """X2W_XMEM_PERMISSIVE=1, the mode the DirectStorage stand-in runs in: grants are page tokens."""
    tokens = shim.grant(1)
    token = tokens[0]
    assert token and not token & 0xF, f"the permissive grant {token:#x} is not a page token"
    shim.accepted("permissive control", [token], 16, b"owned")
    data = shim.read(token, 16, 5)
    shim.check(data == b"owned", f"permissive control: the granted page reads {data!r}, expected b'owned'")
    assert shim.release(1, tokens), C.get_last_error()
    shim.refused("permissive, released", [token], 16, b"stale")
    shim.live("permissive, after the refused write", 0)


CASES = {"never-granted": (never_granted, {}), "released": (released, {}), "regrant": (regrant, {}),
         "second-token": (second_token, {}), "permissive": (permissive, {"X2W_XMEM_PERMISSIVE": "1"})}


def child(path, case):
    shim = Shim(path)
    CASES[case][0](shim)
    return CHECKS_FAILED if shim.failures else 0


def main(path):
    failed = []
    for case, (_, environment) in CASES.items():
        env = dict(os.environ)
        env.pop("X2W_XMEM_PERMISSIVE", None)
        env.update(environment)
        result = subprocess.run([sys.executable, "-B", __file__, "--child", case, str(path)], capture_output=True,
                                text=True, timeout=60, env=env)
        if result.returncode == 0:
            print(f"{case}: ok")
            continue
        failed.append(case)
        print(f"{case}: FAIL (exit code {result.returncode & 0xFFFFFFFF:#x})")
        print(result.stdout.rstrip() or "  (no output)")
        if result.returncode != CHECKS_FAILED:  # the case did not get to its end
            print(result.stderr.rstrip()[-800:])
    if failed:
        print(f"xmem_physical_write_owner: FAIL ({len(failed)} of {len(CASES)} cases: {', '.join(failed)})")
        return 1
    print(f"xmem_physical_write_owner: PASS ({len(CASES)} cases)")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        sys.exit(child(sys.argv[3], sys.argv[2]))
    sys.exit(main(sys.argv[1]))
