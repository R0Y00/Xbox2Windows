"""Guest wave64 groups whose waves branch apart: the lane-pair layout of the direct compute path.

Usage: native_wave64_lane_pairs_check.py controls|hardware COMPILE_DXIL NATIVE_STORE_REPLAY DXC_DIR

A guest wave64 group of more than one wave with a conditional branch and a mask write, lane read, LDS
access or s_barrier is translated as lane pairs: one host wave32 per guest wave64, host lane j carrying
guest lanes j and j + 32. Each program here is such a group with 128 threads (two guest waves). With 64
threads it is one guest wave and keeps the split-halves layout, which checks the same CPU model against
the layout that predates lane pairs. Thresholds on v0 are shifted by BASE = 64 * (waves - 1), so the
same shape applies to the last guest wave of the group.
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request


SENTINEL = 0x69696969
GUARD = 8
OUTPUT_WORDS = 2048 + 2 * GUARD
GROUPS = 2           # native_store_replay dispatches two groups
GROUP_WORDS = 512    # s_lshl_b32 s7, s5, 11: each group stores at a 2048-byte offset
THREAD_WORDS = 4     # v_lshlrev_b32 v8, 4, v0: each thread stores at a 16-byte offset
LOOP_REFUSAL = "lane-pair layout: s_barrier inside a guest loop"

# RSRC2: 5 user SGPRs (s4 = UAV table), TGID x in s5, TG_SIZE in s6; LDS 1 x 128 words for F3.
RSRC2 = (5 << 1) | 0x80 | 0x400
LDS_128_WORDS = 1 << 15

# Assembled with LLVM 15.0.7: clang -cc1as -triple amdgcn-amd-amdhal -target-cpu gfx1013
# -target-feature +wavefrontsize64. The listing prints wave32 spellings; s12 below is s[12:13].
PROLOGUE = (
    0xBE940304,              # s_mov_b32 s20, s4
    0xBE950380,              # s_mov_b32 s21, 0
    0xF408080A, 0xFA000020,  # s_load_dwordx4 s[32:35], s[20:21], 0x20
    0x8F078B05,              # s_lshl_b32 s7, s5, 11          (group byte offset)
    0x34100084,              # v_lshlrev_b32_e32 v8, 4, v0     (thread byte offset)
    0x4A101007,              # v_add_nc_u32_e32 v8, s7, v8
)
LAST_WAVE = (                # SCC = (TG_SIZE wave index == waves in group - 1)
    0x90099406,              # s_lshr_b32 s9, s6, 20
    0x8709BF09,              # s_and_b32 s9, s9, 63
    0x870ABF06,              # s_and_b32 s10, s6, 63
    0x808A810A,              # s_sub_u32 s10, s10, 1
    0xBF060A09,              # s_cmp_eq_u32 s9, s10
)


def movk(value):
    return 0xB00E0000 | value  # s_movk_i32 s14, value


def f1_program(threads, lane=40):
    base = shape(threads)
    return PROLOGUE + LAST_WAVE + (
        0xBF840011,                       # s_cbranch_scc0 END
        movk(base + 45),                  # s_movk_i32 s14, BASE+45
        0x34040088,                       # v_lshlrev_b32_e32 v2, 8, v0
        0x38040485,                       # v_or_b32_e32 v2, 5, v2
        0xD4C4000C, 0x0002000E,           # v_cmp_gt_u32_e64 s[12:13], s14, v0
        0xD7600010, 0x00000102 | (128 + lane) << 9,  # v_readlane_b32 s16, v2, 40
        0x7E06020C,                       # v_mov_b32_e32 v3, s12
        0x7E08020D,                       # v_mov_b32_e32 v4, s13
        0x7E0A0210,                       # v_mov_b32_e32 v5, s16
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xE0701000, 0x80080308,           # buffer_store_dword v3, v8, s[32:35], 0 offen
        0xE0701004, 0x80080408,           # buffer_store_dword v4, v8, s[32:35], 0 offen offset:4
        0xE0701008, 0x80080508,           # buffer_store_dword v5, v8, s[32:35], 0 offen offset:8
        0xBF810000,                       # END: s_endpgm
    )


def f2_program(threads, threshold=32):
    # F8 is folded in: after the v_cmpx, a compare that is true for every lane writes s[12:13].
    base = shape(threads)
    return PROLOGUE + (
        movk(base + threshold),           # s_movk_i32 s14, BASE+32
        0x34040088,                       # v_lshlrev_b32_e32 v2, 8, v0
        0x38040485,                       # v_or_b32_e32 v2, 5, v2
        0x7DA6000E,                       # v_cmpx_le_u32_e32 s14, v0
        0xBF88000D,                       # s_cbranch_execz END
        0x7E200502,                       # v_readfirstlane_b32 s16, v2
        0xD4C6000C, 0x00010100,           # v_cmp_ge_u32_e64 s[12:13], v0, 0   (true for every lane)
        0x7E060210,                       # v_mov_b32_e32 v3, s16
        0x7E08020C,                       # v_mov_b32_e32 v4, s12
        0x7E0A020D,                       # v_mov_b32_e32 v5, s13
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xE0701000, 0x80080308,           # buffer_store_dword v3, v8, s[32:35], 0 offen
        0xE0701004, 0x80080408,           # buffer_store_dword v4, v8, s[32:35], 0 offen offset:4
        0xE0701008, 0x80080508,           # buffer_store_dword v5, v8, s[32:35], 0 offen offset:8
        0xBF810000,                       # END: s_endpgm
    )


def f3_program(threads):
    del threads  # The last-wave test reads TG_SIZE; nothing depends on the thread count.
    return PROLOGUE + LAST_WAVE + (
        0xBF840011,                       # s_cbranch_scc0 JOIN
        0xD7650001, 0x000100C1,           # v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD7660001, 0x000202C1,           # v_mbcnt_hi_u32_b32 v1, -1, v1   (guest lane)
        0x34120282,                       # v_lshlrev_b32_e32 v9, 2, v1
        0x34040088,                       # v_lshlrev_b32_e32 v2, 8, v0
        0x38040485,                       # v_or_b32_e32 v2, 5, v2
        0xD8340000, 0x00000209,           # ds_write_b32 v9, v2
        0x4C1402BF,                       # v_sub_nc_u32_e32 v10, 63, v1
        0x34141482,                       # v_lshlrev_b32_e32 v10, 2, v10
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xD8D80000, 0x0300000A,           # ds_read_b32 v3, v10            (no barrier)
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xE0701000, 0x80080308,           # buffer_store_dword v3, v8, s[32:35], 0 offen
        0xBF8A0000,                       # JOIN: s_barrier
        0x361600BF,                       # v_and_b32_e32 v11, 63, v0
        0x34161682,                       # v_lshlrev_b32_e32 v11, 2, v11
        0xD8D80000, 0x0400000B,           # ds_read_b32 v4, v11
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xE0701004, 0x80080408,           # buffer_store_dword v4, v8, s[32:35], 0 offen offset:4
        0xBF810000,                       # s_endpgm
    )


SELECTORS = (1, 8, 3, 0, 15, 2, 9, 4, 8, 1, 14, 5, 2, 11, 6, 7)  # s12 lanes 0..7, s13 lanes 8..15


def f4_program(threads):
    base = shape(threads)
    # The branch comes from a VCC compare: DPP and permlane alone do not trip the split-halves gate.
    return PROLOGUE + (
        movk(base),                       # s_movk_i32 s14, BASE
        0x7D86000E,                       # v_cmp_le_u32_e32 vcc, s14, v0
        0xBF860017,                       # s_cbranch_vccz END
        0xD7650001, 0x000100C1,           # v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD7660001, 0x000202C1,           # v_mbcnt_hi_u32_b32 v1, -1, v1   (guest lane)
        0x4A040291,                       # v_add_nc_u32_e32 v2, 17, v1
        0xBE8C03FF, 0x492F0381,           # s_mov_b32 s12, 0x492f0381
        0xBE8D03FF, 0x76B25E18,           # s_mov_b32 s13, 0x76b25e18
        0x7E0A0280,                       # v_mov_b32_e32 v5, 0
        0x7E0602FA, 0xFF091102,           # v_mov_b32_dpp v3, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
        0x7E0802FA, 0xFF014002,           # v_mov_b32_dpp v4, v2 row_mirror row_mask:0xf bank_mask:0xf
        0xD7780005, 0x00341902,           # v_permlanex16_b32 v5, v2, s12, s13
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xE0701000, 0x80080308,           # buffer_store_dword v3, v8, s[32:35], 0 offen
        0xE0701004, 0x80080408,           # buffer_store_dword v4, v8, s[32:35], 0 offen offset:4
        0xE0701008, 0x80080508,           # buffer_store_dword v5, v8, s[32:35], 0 offen offset:8
        0xBF810000,                       # END: s_endpgm
    )


def f5_program(threads, dims):
    # Store v0, v1 (, v2) at linear index v0 + X*(v1 + Y*v2), so a wrong id lands in the wrong word.
    # The v_cmpx that no lane fails plus s_cbranch_execz makes the 128-thread form take the gate.
    x, y, z, _ = geometry(f"group_shape_{dims}d", threads)
    if dims == 3:
        index = (0x16120400 | (128 + y),  # v_mul_u32_u24_e32 v9, Y, v2
                 0x4A121301,              # v_add_nc_u32_e32 v9, v1, v9
                 0x16121200 | (128 + x))  # v_mul_u32_u24_e32 v9, X, v9
    else:
        index = (0x16120200 | (128 + x),)  # v_mul_u32_u24_e32 v9, X, v1
    stores = (0xE0701000, 0x80080009,     # buffer_store_dword v0, v9, s[32:35], 0 offen
              0xE0701004, 0x80080109,     # buffer_store_dword v1, v9, s[32:35], 0 offen offset:4
              0xE0701008, 0x80080209)     # buffer_store_dword v2, v9, s[32:35], 0 offen offset:8
    stores = stores[:2 * dims]
    return PROLOGUE[:5] + index + (
        0x4A121300,                       # v_add_nc_u32_e32 v9, v0, v9
        0x34121284,                       # v_lshlrev_b32_e32 v9, 4, v9
        0x4A121207,                       # v_add_nc_u32_e32 v9, s7, v9
        movk(0),                          # s_movk_i32 s14, 0
        0x7DA6000E,                       # v_cmpx_le_u32_e32 s14, v0      (no lane fails)
        0xBF880001 + len(stores),         # s_cbranch_execz END
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
    ) + stores + (0xBF810000,)            # END: s_endpgm


def f6_program(threads):
    del threads
    return PROLOGUE + LAST_WAVE + (
        0xBF84000A,                       # s_cbranch_scc0 END
        0x34040088,                       # v_lshlrev_b32_e32 v2, 8, v0
        0x38040485,                       # v_or_b32_e32 v2, 5, v2
        0xBE92047E,                       # s_mov_b64 s[18:19], exec
        0xBEFE0480,                       # s_mov_b64 exec, 0
        0x7E200502,                       # v_readfirstlane_b32 s16, v2    (empty EXEC)
        0xBEFE0412,                       # s_mov_b64 exec, s[18:19]
        0x7E060210,                       # v_mov_b32_e32 v3, s16
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xE0701000, 0x80080308,           # buffer_store_dword v3, v8, s[32:35], 0 offen
        0xBF810000,                       # END: s_endpgm
    )


def f7_program(threads):
    del threads
    return PROLOGUE + LAST_WAVE + (
        0xBF840010,                       # s_cbranch_scc0 END
        0xD7650001, 0x000100C1,           # v_mbcnt_lo_u32_b32 v1, -1, 0
        0xD7660001, 0x000202C1,           # v_mbcnt_hi_u32_b32 v1, -1, v1   (guest lane)
        0x3612029F,                       # v_and_b32_e32 v9, 31, v1
        0x34121282,                       # v_lshlrev_b32_e32 v9, 2, v9
        0x34040088,                       # v_lshlrev_b32_e32 v2, 8, v0
        0x38040485,                       # v_or_b32_e32 v2, 5, v2
        0xD8340000, 0x00000209,           # ds_write_b32 v9, v2            (lanes j and j+32: same word)
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xD8D80000, 0x03000009,           # ds_read_b32 v3, v9
        0xBF8CC07F,                       # s_waitcnt lgkmcnt(0)
        0xE0701000, 0x80080308,           # buffer_store_dword v3, v8, s[32:35], 0 offen
        0xBF810000,                       # END: s_endpgm
    )


ATOMIC_LANES = 48  # guest lanes 16..63 of the last wave: 16 of its low lane set and all 32 of its high set


def f8_program(threads):
    # Every lane left by the v_cmpx adds 1 to the first word of its group. A count is not idempotent, so a
    # lane set that ran twice shows as well as one that did not run.
    base = shape(threads)
    return PROLOGUE + (
        movk(base + 64 - ATOMIC_LANES),   # s_movk_i32 s14, BASE+16
        0x7DA6000E,                       # v_cmpx_le_u32_e32 s14, v0
        0xBF880004,                       # s_cbranch_execz END
        0x7E040281,                       # v_mov_b32_e32 v2, 1
        0x7E120207,                       # v_mov_b32_e32 v9, s7
        0xE0C81000, 0x80080209,           # buffer_atomic_add v2, v9, s[32:35], 0 offen
        0xBF810000,                       # END: s_endpgm
    )


# 128 threads. Each trip of the loop passes the s_barrier, and the v_cmp makes the group a lane-pair one.
BARRIER_IN_LOOP = PROLOGUE + (
    0x7E040280,              # v_mov_b32_e32 v2, 0
    0xBF8A0000,              # LOOP: s_barrier
    0x4A040481,              # v_add_nc_u32_e32 v2, 1, v2
    0x7D880482,              # v_cmp_gt_u32_e32 vcc, 2, v2
    0xBF87FFFC,              # s_cbranch_vccnz LOOP
    0xBF810000,              # s_endpgm
)


# F9 `exit_before_barrier`: 256 threads, four guest waves. No lane of wave 0 passes the v_cmpx, so wave 0
# leaves through the EXECZ skip before the s_barrier that waves 1..3 reach. Those write LDS, meet at the
# barrier and read the words of wave 3. Each thread stores two words.
EXIT_THREADS = 256
EXIT_BEFORE_BARRIER = (
    0xBE940304,              # s_mov_b32 s20, s4
    0xBE950380,              # s_mov_b32 s21, 0
    0xF408080A, 0xFA000020,  # s_load_dwordx4 s[32:35], s[20:21], 0x20
    0x8F078B05,              # s_lshl_b32 s7, s5, 11          (group byte offset)
    0x34100083,              # v_lshlrev_b32_e32 v8, 3, v0     (thread byte offset: two words)
    0x4A101007,              # v_add_nc_u32_e32 v8, s7, v8
    0x7DA200BF,              # v_cmpx_lt_u32_e32 63, v0        (no lane of wave 0 is left)
    0xBF880011,              # s_cbranch_execz END
    0x34040088,              # v_lshlrev_b32_e32 v2, 8, v0
    0x38040485,              # v_or_b32_e32 v2, 5, v2
    0x34060082,              # v_lshlrev_b32_e32 v3, 2, v0
    0xD8340000, 0x00000203,  # ds_write_b32 v3, v2             (LDS word = thread id)
    0xBF8A0000,              # s_barrier
    0x360800BF,              # v_and_b32_e32 v4, 63, v0
    0x4A0808FF, 0x000000C0,  # v_add_nc_u32_e32 v4, 0xc0, v4   (the same lane of wave 3)
    0x34080882,              # v_lshlrev_b32_e32 v4, 2, v4
    0xD8D80000, 0x05000004,  # ds_read_b32 v5, v4
    0xBF8CC07F,              # s_waitcnt lgkmcnt(0)
    0xE0701000, 0x80080508,  # buffer_store_dword v5, v8, s[32:35], 0 offen
    0xE0701004, 0x80080208,  # buffer_store_dword v2, v8, s[32:35], 0 offen offset:4
    0xBF810000,              # END: s_endpgm
)


def exit_request(directory):
    # TGID x in s5, no TG_SIZE; LDS 2 x 128 words, one word per thread.
    return write_request(directory, EXIT_BEFORE_BARRIER, directory.parent.joinpath("root.bin").read_bytes(),
                         sa(EXIT_THREADS, 1, 1, (5 << 1) | 0x80 | 2 << 15))


def exit_model(first_wave=1, source_wave=3):
    """FIRST_WAVE 0 is the result had wave 0 not left; SOURCE_WAVE None the result of reading one's own word."""
    out = blank()
    for group in range(GROUPS):
        for tid in range(64 * first_wave, EXIT_THREADS):
            at = GUARD + group * GROUP_WORDS + tid * 2
            out[at] = value(tid if source_wave is None else 64 * source_wave + (tid & 63))
            out[at + 1] = value(tid)
    return out


def geometry(name, threads):
    """(X, Y, Z, RSRC2 local axes field) of a request."""
    if name == "group_shape_2d":
        return 8, threads // 8, 1, 1      # 8x8x1 or 8x16x1
    if name == "group_shape_3d":
        return 4, 4, threads // 16, 2     # 4x4x4 or 4x4x8
    return threads, 1, 1, 0


# CPU models of guest lanes 0..63. With swap, guest lane l computes on the data of lane l ^ 32 but
# stores at its own address: the result a layout that exchanged the two 32-lane halves would give.
def value(tid):
    return (tid << 8) | 5


def blank():
    return [SENTINEL] * OUTPUT_WORDS


def put(out, group, tid, j, word):
    out[GUARD + group * GROUP_WORDS + tid * THREAD_WORDS + j] = word & 0xFFFFFFFF


def shape(threads):
    return 64 * (threads // 64 - 1)


def f1_model(threads, swap=False, lane=40):
    base, out = shape(threads), blank()
    data = lambda l: base + (l ^ 32 if swap else l)
    mask = sum(1 << l for l in range(64) if data(l) < base + 45)
    for group in range(GROUPS):
        for l in range(64):
            for j, word in enumerate((mask, mask >> 32, value(data(lane)))):
                put(out, group, base + l, j, word)
    return out


def f2_model(threads, swap=False, threshold=32, low=None):
    # F8: the all-true compare after v_cmpx clears the bits of inactive lanes. LOW overrides the
    # expected low mask word for the negative control.
    base, out = shape(threads), blank()
    data = lambda l: base + (l ^ 32 if swap else l)
    active = [l for l in range(64) if data(l) >= base + threshold]
    mask = sum(1 << l for l in active)
    words = (value(data(active[0])), mask if low is None else low, mask >> 32)
    for group in range(GROUPS):
        for l in active:
            for j, word in enumerate(words):
                put(out, group, base + l, j, word)
    return out


def f3_model(threads, swap=False):
    base, out = shape(threads), blank()
    data = lambda l: base + (l ^ 32 if swap else l)
    for group in range(GROUPS):
        for l in range(64):
            put(out, group, base + l, 0, value(data(63 - l)))
        for tid in range(threads):
            put(out, group, tid, 1, value(data(tid & 63)))
    return out


def f4_model(threads, swap=False):
    base, out = shape(threads), blank()
    v = [17 + (l ^ 32 if swap else l) for l in range(64)]
    for group in range(GROUPS):
        for l in range(64):
            row, k = l & ~15, l & 15
            shifted = v[l - 1] if k else 0                     # row_shr:1, bound_ctrl
            mirrored = v[row + 15 - k]                         # row_mirror
            other = (l & 32) | (16 - (l & 16))                 # permlanex16: the other row of the half
            crossed = v[other + SELECTORS[k]]
            for j, word in enumerate((shifted, mirrored, crossed)):
                put(out, group, base + l, j, word)
    return out


def f5_model(threads, dims):
    x, y, z, _ = geometry(f"group_shape_{dims}d", threads)
    out = blank()
    for group in range(GROUPS):
        for tid in range(threads):  # linear index v0 + X*(v1 + Y*v2) equals the flat thread id
            ids = (tid % x, tid // x % y, tid // (x * y))
            for j in range(dims):
                put(out, group, tid, j, ids[j])
    return out


def f6_model(threads):
    # RDNA returns lane 0 of the wave for v_readfirstlane_b32 under an empty EXEC.
    base, out = shape(threads), blank()
    for group in range(GROUPS):
        for l in range(64):
            put(out, group, base + l, 0, value(base))
    return out


def f7_model(threads):
    # Lane-pair layout only: lanes j and j+32 write LDS word j in one instruction and the host
    # thread writes lane j+32 last. Which lane the guest hardware keeps is not verified.
    base, out = shape(threads), blank()
    for group in range(GROUPS):
        for l in range(64):
            put(out, group, base + l, 0, value(base + (l & 31) + 32))
    return out


def f8_model(threads, lanes=ATOMIC_LANES):
    del threads
    out = blank()
    for group in range(GROUPS):
        out[GUARD + group * GROUP_WORDS] = SENTINEL + lanes
    return out


# name: (program(threads, **options), model(threads, **options), RSRC2 LDS bits)
PROGRAMS = {
    "mask_readlane": (f1_program, f1_model, 0),
    "readfirstlane_high": (f2_program, f2_model, 0),
    "lds_divergent": (f3_program, f3_model, LDS_128_WORDS),
    "dpp_permlane_divergent": (f4_program, f4_model, 0),
    "group_shape_2d": (lambda t: f5_program(t, 2), lambda t: f5_model(t, 2), 0),
    "group_shape_3d": (lambda t: f5_program(t, 3), lambda t: f5_model(t, 3), 0),
    "readfirstlane_empty": (f6_program, f6_model, 0),
    "lds_same_address": (f7_program, f7_model, LDS_128_WORDS),
    "atomic_count": (f8_program, f8_model, 0),
}
# Split halves race on the shared LDS word, so this result is asserted for lane pairs only.
PAIRS_ONLY = {"lds_same_address"}
# Negative-control programs, each checked against the unmodified expectation of its base program.
VARIANTS = {
    "mask_readlane/lane8": ("mask_readlane", {"lane": 8}),
    "readfirstlane_high/threshold31": ("readfirstlane_high", {"threshold": 31}),
}


def sa(x, y, z, rsrc2):
    return struct.pack("<7I", 0, 0, 0, x | (y << 16), z, 0, rsrc2)  # dword 4 bit 20 clear: wave64


def build(directory, name, threads):
    base_name, options = VARIANTS.get(name, (name, {}))
    program, _, lds = PROGRAMS[base_name]
    x, y, z, axes = geometry(base_name, threads)
    return write_request(directory, program(threads, **options), directory.parent.joinpath("root.bin").read_bytes(),
                         sa(x, y, z, RSRC2 | lds | axes << 11))


def compile_batch(cli, sdk, work, tag, requests):
    """Translate a request directory with its own cache and batch directory."""
    cache, batch = work / f"cache_{tag}", work / f"batch_{tag}"
    result = subprocess.run([cli, sdk, cache, "gfx1013-direct-cs@0", "--requests", requests, batch],
                            capture_output=True, text=True)
    return result, cache, batch


ADAPTER = []


def replay(tool, sdk, marker, cache, output):
    result = subprocess.run([tool, sdk, marker, cache, "0", output, "--direct"], capture_output=True, text=True)
    if result.returncode not in (0, 77):
        raise AssertionError((marker.name, result.stdout, result.stderr))
    if not ADAPTER:  # name the adapter once in the log, also ahead of a capability skip
        ADAPTER.extend(line for line in result.stderr.splitlines() if line.startswith(("adapter=", "driver=")))
        print("\n".join(ADAPTER))
    return result


def first_mismatch(actual, expected):
    for index, (got, want) in enumerate(zip(actual, expected)):
        if got != want:
            return index, got, want
    return None


def read_words(path):
    return struct.unpack(f"<{OUTPUT_WORDS}I", path.read_bytes())


def root_signature(sdk, work):
    shader, root = work / "root.hlsl", work / "root.bin"
    shader.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"')
    subprocess.run([Path(sdk) / "dxc.exe", "-T", "rootsig_1_1", "-E", "SIG", "-Fo", root, shader], check=True)


def dumpbin(sdk, cso):
    return subprocess.run([Path(sdk) / "dxc.exe", "-dumpbin", cso], check=True, capture_output=True, text=True).stdout


def size(name, threads):
    return "x".join(map(str, geometry(name, threads)[:3]))


def controls(cli, replay_tool, sdk, work):
    del replay_tool
    # Two guest waves: every program is a lane-pair module, with 64 host threads and no exchange memory.
    requests = work / "requests_128"
    requests.mkdir()
    markers = {name: build(requests, name, 128) for name in PROGRAMS}
    result, _, batch = compile_batch(cli, sdk, work, "128", requests)
    assert result.returncode == 0, ("128 threads", result.stdout, result.stderr)
    pairs = {name: dumpbin(sdk, batch / f"{marker.stem[4:]}.cs.cso") for name, marker in markers.items()}
    for name, ir in pairs.items():
        assert "NumThreads=(64,1,1)" in ir and "waveExchange" not in ir, (name, ir)
        print(f"{name}, {size(name, 128)} threads: lane pairs (64 host threads, no exchange memory)")
    # A guest lane may read LDS that another lane of its wave wrote one instruction earlier. Each LDS access
    # of a lane-pair module therefore takes its lane predicate from a ballot of its own, which brings the
    # host wave's lanes to that point together: lds_same_address has one ballot at entry and one per lane
    # set for each of its two LDS instructions.
    ballots = sum("call" in line and "waveActiveBallot" in line for line in pairs["lds_same_address"].splitlines())
    assert ballots == 5, (ballots, pairs["lds_same_address"])
    print("lds_same_address, 128x1x1 threads: a ballot per lane set before each LDS access")
    # One guest wave: the same program keeps split halves and their exchange memory.
    requests = work / "requests_64"
    requests.mkdir()
    marker = build(requests, "mask_readlane", 64)
    result, _, batch = compile_batch(cli, sdk, work, "64", requests)
    assert result.returncode == 0, ("64 threads", result.stdout, result.stderr)
    split = dumpbin(sdk, batch / f"{marker.stem[4:]}.cs.cso")
    assert "NumThreads=(64,1,1)" in split and "waveExchange" in split, split
    print("mask_readlane, 64x1x1 threads: split halves (exchange memory present)")
    # A wave that ends before an s_barrier the other waves still reach gets one barrier site in a loop
    # (@waveDone and a dispatch switch). A barrier that no wave can skip keeps the plain form.
    assert "waveDone" not in pairs["lds_divergent"], pairs["lds_divergent"]
    requests = work / "requests_exit"
    requests.mkdir()
    marker = exit_request(requests)
    result, _, batch = compile_batch(cli, sdk, work, "exit", requests)
    assert result.returncode == 0, ("exit_before_barrier", result.stdout, result.stderr)
    looped = dumpbin(sdk, batch / f"{marker.stem[4:]}.cs.cso")
    assert "waveDone" in looped and "switch i32" in looped and "NumThreads=(128,1,1)" in looped, looped
    print(f"exit_before_barrier, {EXIT_THREADS}x1x1 threads: lane pairs with the barrier loop; "
          "lds_divergent keeps the plain barrier")
    # A barrier inside a guest loop has no single place to resume and is refused.
    requests = work / "requests_loop"
    requests.mkdir()
    marker = write_request(requests, BARRIER_IN_LOOP, (work / "root.bin").read_bytes(), sa(128, 1, 1, RSRC2))
    result, _, batch = compile_batch(cli, sdk, work, "loop", requests)
    assert result.returncode, ("barrier_in_loop compiled", result.stdout)
    reason = (batch / f"{marker.stem[4:]}.fail").read_text(encoding="utf-8")
    assert LOOP_REFUSAL in reason and not (batch / f"{marker.stem[4:]}.cs.cso").exists(), reason
    print(f"barrier_in_loop, 128x1x1 threads: refused ({LOOP_REFUSAL})")
    return 0


def run_set(cli, tool, sdk, work, threads, names):
    """Compile and replay NAMES at THREADS. Returns {name: words} or 77."""
    requests = work / f"requests_{threads}"
    requests.mkdir()
    markers = {name: build(requests, name, threads) for name in names}
    result, cache, batch = compile_batch(cli, sdk, work, str(threads), requests)
    assert result.returncode == 0, (threads, result.stdout, result.stderr)
    readbacks = {}
    for name, marker in markers.items():
        assert (batch / f"{marker.stem[4:]}.cs.cso").is_file(), (threads, name)
        output = work / f"{threads}_{name.replace('/', '_')}.bin"
        if replay(tool, sdk, marker, cache, output).returncode == 77:
            return 77
        readbacks[name] = read_words(output)
    return readbacks


def negative_controls(layout, threads, readbacks):
    """Each control must report a mismatch."""
    cases = [(f"{name} halves swapped", name, PROGRAMS[name][1](threads, swap=True))
             for name in ("mask_readlane", "readfirstlane_high", "dpp_permlane_divergent")]
    cases.append(("readfirstlane_high inactive mask bits set", "readfirstlane_high",
                  f2_model(threads, low=0xFFFFFFFF)))  # F8
    cases.append(("atomic_count without the high lane set", "atomic_count", f8_model(threads, lanes=16)))
    for variant, (name, options) in VARIANTS.items():
        # The variant must equal its own model, so its mismatch below has the intended cause.
        own = first_mismatch(readbacks[variant], PROGRAMS[name][1](threads, **options))
        assert own is None, (layout, variant, "own model", own)
        cases.append((variant, variant, PROGRAMS[name][1](threads)))
    for title, readback, expected in cases:
        mismatch = first_mismatch(readbacks[readback], expected)
        assert mismatch, (layout, title, "no mismatch")
        index, got, want = mismatch
        print(f"negative, {threads} threads ({layout}), {title}: mismatch at word {index} "
              f"(got {got:#010x}, expected {want:#010x})")


def hardware(cli, tool, sdk, work):
    names = list(PROGRAMS) + list(VARIANTS)
    sets = (("lane pairs", 128, names),
            ("split halves", 64, [name for name in names if name not in PAIRS_ONLY]))
    for layout, threads, subset in sets:
        readbacks = run_set(cli, tool, sdk, work, threads, subset)
        if readbacks == 77:
            return 77
        for name in PROGRAMS:
            if name not in subset:
                print(f"{name}, {size(name, threads)} threads ({layout}): not asserted (lane-pair result only)")
                continue
            mismatch = first_mismatch(readbacks[name], PROGRAMS[name][1](threads))
            assert mismatch is None, (layout, threads, name, mismatch)
            print(f"{name}, {size(name, threads)} threads ({layout}): {OUTPUT_WORDS} words with guards matched the model")
        negative_controls(layout, threads, readbacks)
    # F9: guest wave 0 ends before the barrier that waves 1..3 meet at.
    requests = work / "requests_exit"
    requests.mkdir()
    marker = exit_request(requests)
    result, cache, _ = compile_batch(cli, sdk, work, "exit", requests)
    assert result.returncode == 0, ("exit_before_barrier", result.stdout, result.stderr)
    output = work / "exit.bin"
    if replay(tool, sdk, marker, cache, output).returncode == 77:
        return 77
    readback = read_words(output)
    mismatch = first_mismatch(readback, exit_model())
    assert mismatch is None, ("exit_before_barrier", mismatch)
    print(f"exit_before_barrier, {EXIT_THREADS}x1x1 threads (lane pairs): {OUTPUT_WORDS} words with guards matched the model")
    for title, expected in (("wave 0 not ended", exit_model(first_wave=0)), ("own LDS word", exit_model(source_wave=None))):
        mismatch = first_mismatch(readback, expected)
        assert mismatch, ("exit_before_barrier", title, "no mismatch")
        index, got, want = mismatch
        print(f"negative, {EXIT_THREADS} threads (lane pairs), exit_before_barrier {title}: mismatch at word {index} "
              f"(got {got:#010x}, expected {want:#010x})")
    return 0


def main(mode, cli, tool, sdk):
    with tempfile.TemporaryDirectory() as temp:
        work = Path(temp)
        root_signature(sdk, work)
        cli, tool = Path(cli).resolve(), Path(tool).resolve()
        return {"controls": controls, "hardware": hardware}[mode](cli, tool, sdk, work)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
