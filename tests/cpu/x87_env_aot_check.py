"""Differential, architectural-state, MXCSR-isolation, mutant, and refusal checks
for x87 environment/state AOT lifting (`tools/cpu_x87_env_aot.py`).

Exercises all 7 uncovered x87 mnemonics in 28-byte and 108-byte protected-mode forms:
  - FDECSTP, FNSTENV, FSTENV (WAIT+FNSTENV), FLDENV, FNSAVE, FSAVE (WAIT+FNSAVE), FRSTOR
plus bounded stack/control/status glue, translator positive/negative controls,
CLI negative/atomic-write controls, MXCSR DAZ/FTZ isolation, and altered-build controls.
"""

import ctypes
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import struct
import sys
import tempfile
from unittest import mock

from iced_x86 import (
    Code,
    Decoder,
    Encoder,
    Formatter,
    FormatterSyntax,
    Instruction,
    MemoryOperand,
    Register as R,
)
import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import bounded_code, export_rva  # noqa: E402
import cpu_x87_env_aot  # noqa: E402
from cpu_x87_env_aot import lift_code  # noqa: E402

EXPORTS = (
    "op_fdecstp_sw",
    "op_fdecstp_wrap_env",
    "op_fnstenv_pushed",
    "op_fstenv_pushed",
    "op_fnstenv_post_cw",
    "op_fldenv_fnstenv",
    "op_fldenv_wait_fnstenv",
    "op_fnsave_stack",
    "op_fnsave_post_cw",
    "op_fnsave_post_sw",
    "op_fsave_stack",
    "op_frstor_fnsave",
    "op_frstor_fdecstp_fnsave",
    "op_fld_m32_fstp_m64",
    "op_fld_m64_fstp_m32",
    "op_fld_m80_fstp_m80",
    "op_fstp_c1_fdecstp",
)

ALTERED_EXPORTS = (
    "op_fdecstp_sw",
    "op_fdecstp_wrap_env",
    "op_fstenv_pushed",
    "op_fldenv_fnstenv",
    "op_fnsave_stack",
    "op_fsave_stack",
    "op_frstor_fnsave",
)

GUARD_PRE = bytes(range(0xA0, 0xB0))
GUARD_POST = bytes(range(0xC0, 0xD0))

J = 1 << 63
QBIT = 1 << 62

SW_C1 = 0x0200
SW_TOP_MASK = 0x3800

POS_ZERO = (0, 0x0000)
NEG_ZERO = (0, 0x8000)
POS_ONE = (J, 0x3FFF)
NEG_ONE = (J, 0xBFFF)
POS_PI_APPROX = (0xC90FDAA22168C235, 0x4000)
POS_INF = (J, 0x7FFF)
NEG_INF = (J, 0xFFFF)
QNAN_A = (J | QBIT | 0x12345678, 0x7FFF)
SNAN_A = (J | 0x00005678, 0x7FFF)
SUB_POS_MIN = (1, 0x0000)
SUB_NEG_MID = (0x0000123456789ABC, 0x8000)
UNNORM = (0x0000123456789ABC, 0x3FFF)


def pack80(val):
    return struct.pack("<QH", val[0], val[1])


def unpack80(raw):
    return struct.unpack("<QH", raw[:10])


def pack16(v):
    return struct.pack("<H", v & 0xFFFF)


def unpack16(raw):
    return struct.unpack("<H", raw[:2])[0]


def pack_f32(v):
    return struct.pack("<I", v) if isinstance(v, int) else struct.pack("<f", v)


def pack_f64(v):
    return struct.pack("<Q", v) if isinstance(v, int) else struct.pack("<d", v)


def pack_env28(*, cw=0x037F, sw=0x0000, tw=0xFFFF, fip=0, fcs=0, fop=0, fdp=0, fds=0):
    return struct.pack(
        "<HHHHHHIHHIHH",
        cw & 0xFFFF, 0xFFFF,
        sw & 0xFFFF, 0xFFFF,
        tw & 0xFFFF, 0xFFFF,
        fip & 0xFFFFFFFF, fcs & 0xFFFF, fop & 0x07FF,
        fdp & 0xFFFFFFFF, fds & 0xFFFF, 0xFFFF,
    )


def unpack_env28(raw):
    cw, r0, sw, r1, tw, r2, fip, fcs, fop, fdp, fds, r3 = struct.unpack("<HHHHHHIHHIHH", raw[:28])
    return {
        "cw": cw, "r0": r0,
        "sw": sw, "r1": r1,
        "tw": tw, "r2": r2,
        "fip": fip, "fcs": fcs, "fop": fop,
        "fdp": fdp, "fds": fds, "r3": r3,
        "top": (sw >> 11) & 7,
    }


def unpack_state108(raw):
    env = unpack_env28(raw[:28])
    regs = [unpack80(raw[28 + i * 10:38 + i * 10]) for i in range(8)]
    return env, regs


def x87_tag(val):
    sig, se = val
    exp = se & 0x7FFF
    if exp == 0 and sig == 0:
        return 1
    if exp == 0 or exp == 0x7FFF or (sig & J) == 0:
        return 2
    return 0


def expected_ftw(occupied_by_phys):
    tw = 0xFFFF
    for phys, val in occupied_by_phys.items():
        tag = x87_tag(val)
        tw = (tw & ~(3 << (phys * 2))) | (tag << (phys * 2))
    return tw


def alloc_guarded(payload):
    size = 32 + len(GUARD_PRE) + len(payload) + len(GUARD_POST)
    raw = ctypes.create_string_buffer(size)
    base = ctypes.addressof(raw)
    data_addr = (base + len(GUARD_PRE) + 15) & ~15
    pre_off = (data_addr - len(GUARD_PRE)) - base
    data_off = data_addr - base
    post_off = data_off + len(payload)
    raw[pre_off:data_off] = GUARD_PRE
    raw[data_off:post_off] = payload
    raw[post_off:post_off + len(GUARD_POST)] = GUARD_POST
    return raw, pre_off, data_off, post_off, data_addr


def call3(fn, a_bytes, b_bytes, out_init, *, check_inputs=True):
    buf_a, a_pre, a_off, a_post, a_addr = alloc_guarded(a_bytes)
    buf_b, b_pre, b_off, b_post, b_addr = alloc_guarded(b_bytes)
    buf_o, o_pre, o_off, o_post, o_addr = alloc_guarded(out_init)
    fn(a_addr, b_addr, o_addr)
    assert buf_a.raw[a_pre:a_off] == GUARD_PRE and buf_a.raw[a_post:a_post + 16] == GUARD_POST, "A guards corrupted"
    assert buf_b.raw[b_pre:b_off] == GUARD_PRE and buf_b.raw[b_post:b_post + 16] == GUARD_POST, "B guards corrupted"
    assert buf_o.raw[o_pre:o_off] == GUARD_PRE and buf_o.raw[o_post:o_post + 16] == GUARD_POST, "out guards corrupted"
    if check_inputs:
        assert buf_a.raw[a_off:a_post] == a_bytes, "input A mutated"
        assert buf_b.raw[b_off:b_post] == b_bytes, "input B mutated"
    return buf_o.raw[o_off:o_off + len(out_init)]


def bind(path, prefix):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    funcs = {}
    for name in EXPORTS:
        fn = getattr(lib, prefix + name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        funcs[name] = fn
    return lib, funcs


def encode_seq(*instructions):
    enc = Encoder(64)
    rip = 0x1000
    for ins in instructions:
        rip += enc.encode(ins, rip)
    return enc.take_buffer()


def check_translator():
    ret = Instruction.create(Code.RETNQ)
    wait = Instruction.create(Code.WAIT)
    fdecstp = Instruction.create(Code.FDECSTP)
    fninit = Instruction.create(Code.FNINIT)
    fld1 = Instruction.create(Code.FLD1)
    fldz = Instruction.create(Code.FLDZ)
    fld_m32 = Instruction.create_mem(Code.FLD_M32FP, MemoryOperand(R.RCX))
    fld_m64 = Instruction.create_mem(Code.FLD_M64FP, MemoryOperand(R.RCX))
    fld_m80 = Instruction.create_mem(Code.FLD_M80FP, MemoryOperand(R.RCX))
    fstp_m32 = Instruction.create_mem(Code.FSTP_M32FP, MemoryOperand(R.R8))
    fstp_m64 = Instruction.create_mem(Code.FSTP_M64FP, MemoryOperand(R.R8))
    fstp_m80 = Instruction.create_mem(Code.FSTP_M80FP, MemoryOperand(R.R8))
    fstp_st0 = Instruction.create_reg(Code.FSTP_STI, R.ST0)
    fldcw_rdx = Instruction.create_mem(Code.FLDCW_M2BYTE, MemoryOperand(R.RDX))
    fnstcw_r8 = Instruction.create_mem(Code.FNSTCW_M2BYTE, MemoryOperand(R.R8))
    fnstsw_ax = Instruction.create_reg(Code.FNSTSW_AX, R.AX)
    mov_r8_ax = Instruction.create_mem_reg(Code.MOV_RM16_R16, MemoryOperand(R.R8), R.AX)
    fldenv_rcx = Instruction.create_mem(Code.FLDENV_M28BYTE, MemoryOperand(R.RCX))
    fldenv_rdx = Instruction.create_mem(Code.FLDENV_M28BYTE, MemoryOperand(R.RDX))
    fnstenv_r8 = Instruction.create_mem(Code.FNSTENV_M28BYTE, MemoryOperand(R.R8))
    frstor_rcx = Instruction.create_mem(Code.FRSTOR_M108BYTE, MemoryOperand(R.RCX))
    frstor_rdx = Instruction.create_mem(Code.FRSTOR_M108BYTE, MemoryOperand(R.RDX))
    fnsave_r8 = Instruction.create_mem(Code.FNSAVE_M108BYTE, MemoryOperand(R.R8))

    positive = [
        ("fdecstp_sw", (fninit, fdecstp, fnstsw_ax, mov_r8_ax, ret), "x87_fdecstp(x87);"),
        ("fldenv_fnstenv", (fldenv_rcx, fnstenv_r8, fninit, ret), "x87_fldenv(x87, rcx);"),
        ("fldenv_wait_fnstenv", (fldenv_rcx, wait, fnstenv_r8, fninit, ret), "x87_wait(x87);"),
        ("frstor_cw_sw_fnsave", (frstor_rcx, fldcw_rdx, fnstcw_r8, fnstsw_ax, mov_r8_ax, fnsave_r8, ret), "ax = x87_status_word(x87);"),
        ("fstenv_wait", (fninit, fld1, wait, fnstenv_r8, fstp_st0, ret), "x87_fnstenv(x87, r8);"),
        ("fnsave", (fninit, fld_m80, fnsave_r8, ret), "x87_fnsave(x87, r8);"),
        ("fsave_wait", (fninit, fld_m32, wait, fnsave_r8, ret), "x87_fnsave(x87, r8);"),
        ("frstor_rdx", (frstor_rdx, fnsave_r8, ret), "x87_frstor(x87, rdx);"),
        ("fldenv_rdx_fninit", (fldenv_rdx, fdecstp, fnstcw_r8, fninit, ret), "x87_fldenv(x87, rdx);"),
        ("m64_to_m32", (fninit, fldcw_rdx, fld_m64, fstp_m32, ret), "x87_store_f32(x87, r8);"),
        ("m32_to_m64", (fninit, fld_m32, fstp_m64, ret), "x87_store_f64(x87, r8);"),
        ("m80_to_m80", (fninit, fldz, fstp_st0, fld_m80, fstp_m80, ret), "x87_store_m80(x87, r8);"),
    ]
    for label, seq, needle in positive:
        src, _ = lift_code(label, 0x1000, encode_seq(*seq))
        assert needle in src, f"{label}: missing {needle!r}"
        assert "long double" not in src and "<cmath>" not in src, f"{label}: forbidden host FP type/header"

    rep_ret = Instruction.create(Code.RETNQ)
    rep_ret.has_rep_prefix = True
    fs_fld = Instruction.create_mem(Code.FLD_M80FP, MemoryOperand(R.RCX))
    fs_fld.segment_prefix = R.FS

    rejections = [
        # o16 (14-byte / 94-byte) forms refused
        ("o16_fnstenv", (fninit, Instruction.create_mem(Code.FNSTENV_M14BYTE, MemoryOperand(R.R8)), ret),
         "unsupported 16-bit x87 environment/state operand"),
        ("o16_fldenv", (Instruction.create_mem(Code.FLDENV_M14BYTE, MemoryOperand(R.RCX)), fninit, ret),
         "unsupported 16-bit x87 environment/state operand"),
        ("o16_fnsave", (fninit, Instruction.create_mem(Code.FNSAVE_M94BYTE, MemoryOperand(R.R8)), ret),
         "unsupported 16-bit x87 environment/state operand"),
        ("o16_frstor", (Instruction.create_mem(Code.FRSTOR_M94BYTE, MemoryOperand(R.RCX)), fnsave_r8, ret),
         "unsupported 16-bit x87 environment/state operand"),
        ("o16_fstenv_wait", (fninit, wait, Instruction.create_mem(Code.FNSTENV_M14BYTE, MemoryOperand(R.R8)), ret),
         "unsupported WAIT not immediately before FNSTENV/FNSAVE"),
        ("o16_fsave_wait", (fninit, wait, Instruction.create_mem(Code.FNSAVE_M94BYTE, MemoryOperand(R.R8)), ret),
         "unsupported WAIT not immediately before FNSTENV/FNSAVE"),
        # WAIT rules: only immediately before FNSTENV / FNSAVE
        ("wait_standalone_ret", (fninit, wait, ret), "unsupported WAIT not immediately before FNSTENV/FNSAVE"),
        ("wait_before_fninit", (wait, fninit, ret), "unsupported WAIT not immediately before FNSTENV/FNSAVE"),
        ("wait_before_fnstsw", (fninit, wait, fnstsw_ax, mov_r8_ax, ret),
         "unsupported WAIT not immediately before FNSTENV/FNSAVE"),
        ("wait_before_fdecstp", (fninit, wait, fdecstp, ret), "unsupported WAIT not immediately before FNSTENV/FNSAVE"),
        ("wait_at_eof", (fninit, wait), "unsupported WAIT not immediately before FNSTENV/FNSAVE"),
        # Stack overflow / underflow / non-empty at ret
        ("overflow_9", (*([fld1] * 9), *([fstp_st0] * 8), ret), "x87 stack overflow"),
        ("overflow_8_then_fdecstp", (fninit, *([fld1] * 8), fdecstp, fld1, *([fstp_st0] * 8), ret), "x87 stack overflow"),
        ("underflow_empty_fstp", (fninit, fstp_st0, ret), "x87 stack underflow"),
        ("underflow_fdecstp_fstp", (fninit, fld1, fdecstp, fstp_st0, ret), "x87 stack underflow"),
        ("nonempty_at_ret_fld1", (fninit, fld1, ret), "non-empty x87 stack at ret"),
        ("nonempty_at_ret_fdecstp", (fninit, fld1, fdecstp, ret), "non-empty x87 stack at ret"),
        ("nonempty_at_ret_fnstenv", (fninit, fld1, fnstenv_r8, ret), "non-empty x87 stack at ret"),
        ("nonempty_after_fldenv", (fldenv_rcx, ret), "non-empty x87 stack at ret"),
        ("nonempty_after_frstor", (frstor_rcx, ret), "non-empty x87 stack at ret"),
        ("fld_after_frstor", (frstor_rcx, fld1, fninit, ret), "x87 stack overflow"),
        ("fstp_after_fldenv", (fldenv_rcx, fstp_st0, fninit, ret), "x87 stack underflow"),
        # Memory & store guards
        ("displaced_fldenv", (Instruction.create_mem(Code.FLDENV_M28BYTE, MemoryOperand(R.RCX, displ=16, displ_size=1)), fninit, ret),
         "unsupported displaced memory operand"),
        ("displaced_fnstenv", (fninit, Instruction.create_mem(Code.FNSTENV_M28BYTE, MemoryOperand(R.R8, displ=16, displ_size=1)), ret),
         "unsupported displaced memory operand"),
        ("displaced_fnsave", (fninit, Instruction.create_mem(Code.FNSAVE_M108BYTE, MemoryOperand(R.R8, displ=16, displ_size=1)), ret),
         "unsupported displaced memory operand"),
        ("displaced_frstor", (Instruction.create_mem(Code.FRSTOR_M108BYTE, MemoryOperand(R.RCX, displ=16, displ_size=1)), fnsave_r8, ret),
         "unsupported displaced memory operand"),
        ("indexed_mem", (Instruction.create_mem(Code.FRSTOR_M108BYTE, MemoryOperand(R.RCX, index=R.RDX)), fnsave_r8, ret),
         "unsupported memory operand"),
        ("rip_rel_mem", (Instruction.create_mem(Code.FLDENV_M28BYTE, MemoryOperand(R.RIP, displ=0x1200, displ_size=4)), fninit, ret),
         "unsupported memory operand"),
        ("nonvol_base_rbx", (Instruction.create_mem(Code.FLDENV_M28BYTE, MemoryOperand(R.RBX)), fninit, ret),
         "nonvolatile GPR base"),
        ("non_arg_base_rax", (Instruction.create_mem(Code.FLDENV_M28BYTE, MemoryOperand(R.RAX)), fninit, ret),
         "unsupported memory base"),
        ("fnstenv_outside_r8_rcx", (fninit, Instruction.create_mem(Code.FNSTENV_M28BYTE, MemoryOperand(R.RCX)), ret),
         "store outside output buffer r8"),
        ("fnsave_outside_r8_rdx", (fninit, Instruction.create_mem(Code.FNSAVE_M108BYTE, MemoryOperand(R.RDX)), ret),
         "store outside output buffer r8"),
        ("fnstcw_outside_r8", (fninit, Instruction.create_mem(Code.FNSTCW_M2BYTE, MemoryOperand(R.RCX)), ret),
         "store outside output buffer r8"),
        ("fstp_outside_r8", (fninit, fld1, Instruction.create_mem(Code.FSTP_M80FP, MemoryOperand(R.RCX)), ret),
         "store outside output buffer r8"),
        # Uninitialized CW/SW/env reads & undefined condition codes after FNSTENV/FSTENV
        ("fnstcw_before_init", (fnstcw_r8, ret), "read of uninitialized x87 control word"),
        ("fnstsw_before_init", (fnstsw_ax, mov_r8_ax, ret), "read of uninitialized x87 status word"),
        ("fnstenv_before_init", (fnstenv_r8, ret), "read of uninitialized x87 environment"),
        ("fnsave_before_init", (fnsave_r8, ret), "read of uninitialized x87 environment"),
        ("fdecstp_before_init", (fdecstp, ret), "read of uninitialized x87 environment"),
        ("fnstsw_after_fldcw_only", (fldcw_rdx, fnstsw_ax, mov_r8_ax, ret), "read of uninitialized x87 status word"),
        ("fnstenv_after_fldcw_only", (fldcw_rdx, fnstenv_r8, ret), "read of uninitialized x87 environment"),
        ("fnstsw_after_fnstenv", (fninit, fnstenv_r8, fnstsw_ax, mov_r8_ax, ret), "undefined x87 condition codes after FNSTENV"),
        ("fnstenv_after_fnstenv", (fninit, fnstenv_r8, fnstenv_r8, ret), "undefined x87 condition codes after FNSTENV"),
        ("fnsave_after_fstenv", (fninit, wait, fnstenv_r8, fnsave_r8, ret), "undefined x87 condition codes after FNSTENV"),
        # SSE / VEX / EVEX / unsupported x87 / prefix / ax / control-flow
        ("sse_addps", (Instruction.create_reg_reg(Code.ADDPS_XMM_XMMM128, R.XMM0, R.XMM1), ret),
         "unsupported vector instruction"),
        ("vex_vaddps", (Instruction.create_reg_reg_reg(Code.VEX_VADDPS_YMM_YMM_YMMM256, R.YMM0, R.YMM0, R.YMM1), ret),
         "unsupported VEX encoding"),
        ("evex_vmovdqu32", (Instruction.create_reg_mem(Code.EVEX_VMOVDQU32_ZMM_K1Z_ZMMM512, R.ZMM0, MemoryOperand(R.RCX)), ret),
         "unsupported EVEX encoding"),
        ("fincstp_unsupported", (fninit, Instruction.create(Code.FINCSTP), ret), "unsupported instruction"),
        ("undef_ax", (mov_r8_ax, ret), "undefined register ax"),
        ("prefix_fs", (fs_fld, fstp_m80, ret), "unsupported instruction prefix"),
        ("prefix_rep_ret", (fninit, rep_ret), "unsupported instruction prefix"),
        ("branch_jmp", (Instruction.create_branch(Code.JMP_REL8_64, 0x1010), ret), "unsupported control-flow instruction"),
        ("missing_ret", (fninit, fdecstp), "missing terminal ret"),
        ("trailing_after_ret", (fninit, ret, fninit), "unreachable instruction after ret"),
    ]
    for label, seq, reason in rejections:
        try:
            lift_code(label, 0x1000, encode_seq(*seq))
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{label}: expected {reason!r}, got {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: translator accepted rejected sequence")

    return len(positive), len(rejections)


def check_cli_controls(original_path):
    """Verify atomic write and negative CLI controls on tools/cpu_x87_env_aot.py."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        out_cpp = base / "nested" / "lifted.cpp"
        exports_arg = ",".join(EXPORTS)

        def run_cli(argv):
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    rc = cpu_x87_env_aot.main(argv)
                except SystemExit as exc:
                    rc = exc.code
            return rc, stdout.getvalue(), stderr.getvalue()

        rc, out, err = run_cli([str(original_path), exports_arg, str(out_cpp)])
        assert rc == 0 and not err and out_cpp.is_file() and "lifted_op_frstor_fnsave" in out_cpp.read_text("utf-8"), (rc, err)
        assert sorted(p.name for p in out_cpp.parent.iterdir()) == ["lifted.cpp"]

        # Negative CLI controls: zero tracebacks, no leftover .tmp files.
        out_cpp.write_text("prior", encoding="utf-8")
        with mock.patch.object(Path, "replace", side_effect=OSError("replace failed")):
            rc_rep, _, err_rep = run_cli([str(original_path), exports_arg, str(out_cpp)])
        assert rc_rep == 1 and "Traceback" not in err_rep and "replace failed" in err_rep
        assert out_cpp.read_text("utf-8") == "prior"
        assert sorted(p.name for p in out_cpp.parent.iterdir()) == ["lifted.cpp"]

        dir_out = base / "dir.cpp"
        dir_out.mkdir()
        for argv, want_code in (
            ([str(base / "missing.dll"), exports_arg, str(base / "unused.cpp")], 1),
            ([str(original_path), "", str(base / "unused.cpp")], 1),
            ([str(original_path), exports_arg, str(base / "bad.txt")], 2),
            ([str(original_path), exports_arg, str(dir_out)], 1),
            ([str(original_path), exports_arg, str(original_path)], 2),
            ([str(original_path), exports_arg, str(out_cpp / "child.cpp")], 1),
        ):
            rc_bad, out_bad, err_bad = run_cli(argv)
            assert rc_bad == want_code and out_bad == "" and "Traceback" not in err_bad and err_bad.strip(), (argv, rc_bad, err_bad)


def assert_env28_deterministic(
    nat_raw, lif_raw, label, *, check_fip=True, check_fdp_fop=True, check_fcs_fds=None
):
    nat = unpack_env28(nat_raw)
    lif = unpack_env28(lif_raw)
    keys = ["cw", "r0", "sw", "r1", "tw", "r2", "r3"]
    if check_fcs_fds is None:
        check_fcs_fds = check_fdp_fop
    if check_fdp_fop:
        keys.extend(("fop", "fdp"))
    if check_fcs_fds:
        keys.extend(("fcs", "fds"))
    if check_fip:
        keys.append("fip")
    for key in keys:
        assert nat[key] == lif[key], f"{label}: {key} mismatch nat={nat[key]:#x} lif={lif[key]:#x}"


def _classify_fcs_fds(observed_profiles, nat_parsed, expected_fcs, expected_fds, label):
    if nat_parsed["fcs"] == expected_fcs and nat_parsed["fds"] == expected_fds:
        observed_profiles.add("preserved")
    elif nat_parsed["fcs"] == 0 and nat_parsed["fds"] == 0:
        observed_profiles.add("deprecated-zero")
    else:
        raise AssertionError(
            f"{label}: unexpected native FCS/FDS ({nat_parsed['fcs']:#x}, {nat_parsed['fds']:#x}) "
            f"for ({expected_fcs:#x}, {expected_fds:#x})"
        )


def assert_fnsave_nonempty_regs_match(nat_raw, lif_raw, label):
    nat_env, nat_regs = unpack_state108(nat_raw)
    lif_env, lif_regs = unpack_state108(lif_raw)
    top = lif_env["top"]
    tw = lif_env["tw"]
    assert nat_env["tw"] == tw and nat_env["top"] == top
    for i in range(8):
        phys = (top + i) & 7
        if ((tw >> (phys * 2)) & 3) != 3:
            assert nat_regs[i] == lif_regs[i], (
                f"{label}: ST({i}) (phys {phys}) mismatch nat={nat_regs[i]} lif={lif_regs[i]}"
            )


def recomputed_ftw(raw_tw, phys_vals):
    tw = 0xFFFF
    for phys in range(8):
        if ((raw_tw >> (phys * 2)) & 3) != 3:
            tag = x87_tag(phys_vals[phys])
            tw = (tw & ~(3 << (phys * 2))) | (tag << (phys * 2))
    return tw


def check_differential(native, lifted, altered, altered_lifted):
    total = 0
    dummy = b"\x00" * 16

    # 1. FDECSTP from FNINIT -> TOP wraps 0 -> 7, C1=0.
    nat_sw = unpack16(call3(native["op_fdecstp_sw"], dummy, dummy, b"\x5a" * 2))
    lif_sw = unpack16(call3(lifted["op_fdecstp_sw"], dummy, dummy, b"\x5a" * 2))
    assert nat_sw == lif_sw == 0x3800, (hex(nat_sw), hex(lif_sw))
    total += 1

    # 2. FDECSTP via FLDENV across all 8 TOPs (0..7) with C1=1 set in FSW.
    #    Empty FTW (0xFFFF) avoids reading caller-left physical registers (FNINIT does not zero regs).
    #    Native vs lifted checks architectural state strictly (cw/sw/tw/reserved); lifted strictly
    #    checks the `preserved-metadata` profile (fip/fcs/fop/fdp/fds preserved across FDECSTP).
    observed_fdecstp_profiles = set()
    observed_fcs_fds_profiles = set()
    for top in range(8):
        env_in = pack_env28(
            cw=0x037F,
            sw=(top << 11) | SW_C1 | 0x4501,
            tw=0xFFFF,
            fip=0x12345678, fcs=0x1234, fop=0x03AB,
            fdp=0x9ABCDEF0, fds=0x5678,
        )
        nat_env = call3(native["op_fdecstp_wrap_env"], env_in, dummy, b"\x5a" * 28)
        lif_env = call3(lifted["op_fdecstp_wrap_env"], env_in, dummy, b"\x5a" * 28)
        assert_env28_deterministic(
            nat_env, lif_env, f"fdecstp_wrap_top_{top}", check_fip=False, check_fdp_fop=False
        )
        parsed = unpack_env28(lif_env)
        assert (parsed["fip"], parsed["fcs"], parsed["fop"], parsed["fdp"], parsed["fds"]) == (
            0x12345678, 0x1234, 0x03AB, 0x9ABCDEF0, 0x5678
        ), f"fdecstp_wrap_top_{top}: preserved-metadata mismatch {parsed}"
        assert parsed["top"] == ((top - 1) & 7)
        assert (parsed["sw"] & SW_C1) == 0
        assert parsed["tw"] == 0xFFFF
        nat_parsed = unpack_env28(nat_env)
        assert nat_parsed["fdp"] == 0x9ABCDEF0
        assert nat_parsed["fds"] in (0x5678, 0), f"fdecstp_wrap_top_{top}: unexpected native FDS {nat_parsed['fds']:#x}"
        if nat_parsed["fop"] == 0x03AB and nat_parsed["fcs"] in (0x1234, 0):
            observed_fdecstp_profiles.add("preserved")
        elif nat_parsed["fop"] == 0x01F6:
            observed_fdecstp_profiles.add(f"updated(fop=0x1f6,fcs={nat_parsed['fcs']:#x})")
        else:
            raise AssertionError(f"fdecstp_wrap_top_{top}: unexpected native FOP {nat_parsed['fop']:#x}")
        total += 1
    assert len(observed_fdecstp_profiles) == 1, f"inconsistent host FDECSTP FOP behavior: {observed_fdecstp_profiles}"
    native_fdecstp_profile = next(iter(observed_fdecstp_profiles))

    # 3. FNSTENV and FSTENV after pushed values (checks FTW Valid=00, Zero=01, Special=10, Empty=11, TOP=5, FCW).
    for cw in (0x037F, 0x027F, 0x077F, 0x0B7F, 0x0F7F):
        for val in (POS_PI_APPROX, POS_ZERO, NEG_ZERO, POS_INF, QNAN_A, SUB_POS_MIN, UNNORM):
            nat_env = call3(native["op_fnstenv_pushed"], pack80(val), pack16(cw), b"\x5a" * 28)
            lif_env = call3(lifted["op_fnstenv_pushed"], pack80(val), pack16(cw), b"\x5a" * 28)
            assert_env28_deterministic(
                nat_env, lif_env, f"fnstenv_pushed_{cw:#x}_{val}", check_fip=False, check_fdp_fop=False
            )
            parsed = unpack_env28(lif_env)
            assert parsed["top"] == 5 and parsed["cw"] == cw
            assert parsed["tw"] == expected_ftw({7: val, 6: POS_ONE, 5: POS_ZERO})
            total += 1

        for f64_val in (0.0, -0.0, 1.5, float("inf"), 1e-310):
            nat_env = call3(native["op_fstenv_pushed"], pack_f64(f64_val), pack16(cw), b"\x5a" * 28)
            lif_env = call3(lifted["op_fstenv_pushed"], pack_f64(f64_val), pack16(cw), b"\x5a" * 28)
            assert_env28_deterministic(
                nat_env, lif_env, f"fstenv_pushed_{cw:#x}_{f64_val}", check_fip=False, check_fdp_fop=False
            )
            assert unpack_env28(lif_env)["top"] == 6
            total += 1

    # 4. FNSTENV masks FCW[5:0] after saving environment.
    for cw_in in (0x037F, 0x037E, 0x0340, 0x027F):
        nat_buf = call3(native["op_fnstenv_post_cw"], pack16(cw_in), dummy, b"\x5a" * 28)
        lif_buf = call3(lifted["op_fnstenv_post_cw"], pack16(cw_in), dummy, b"\x5a" * 28)
        assert unpack16(nat_buf[:2]) == unpack16(lif_buf[:2]) == (cw_in | 0x003F)
        assert nat_buf[2:12] == lif_buf[2:12]
        total += 1

    # 5. FLDENV -> FNSTENV and FLDENV -> WAIT -> FNSTENV deterministic 28-byte roundtrips
    #    (non-zero FCS/FDS classified per host, strictly preserved in lifted `preserved-metadata`;
    #    C0..C3 = 0x4700 preserved across WAIT under the measured x64 status profile).
    for meta in (
        (0x027F, 0x1800, 0xFFFF, 0x11223344, 0x1234, 0x0155, 0x55667788, 0x5678),
        (0x0F7F, 0x3820, 0xFFFF, 0xCAFEBABE, 0x002B, 0x07FF, 0xDEADBEEF, 0x0033),
    ):
        cw, sw, tw, fip, fcs, fop, fdp, fds = meta
        env_in = pack_env28(cw=cw, sw=sw, tw=tw, fip=fip, fcs=fcs, fop=fop, fdp=fdp, fds=fds)
        nat_env = call3(native["op_fldenv_fnstenv"], env_in, dummy, b"\x5a" * 28)
        lif_env = call3(lifted["op_fldenv_fnstenv"], env_in, dummy, b"\x5a" * 28)
        assert_env28_deterministic(
            nat_env, lif_env, "fldenv_fnstenv", check_fip=True, check_fdp_fop=True, check_fcs_fds=False
        )
        lif_parsed = unpack_env28(lif_env)
        assert (lif_parsed["fip"], lif_parsed["fcs"], lif_parsed["fop"], lif_parsed["fdp"], lif_parsed["fds"]) == (
            fip, fcs, fop, fdp, fds
        ), f"fldenv_fnstenv: preserved-metadata mismatch {lif_parsed}"
        _classify_fcs_fds(observed_fcs_fds_profiles, unpack_env28(nat_env), fcs, fds, "fldenv_fnstenv")
        total += 1

    for sw_val, fcs_val, fds_val in ((0x4700, 0x1234, 0x5678), (0x0000, 0x002B, 0x0033)):
        env_in = pack_env28(
            cw=0x037F, sw=sw_val, tw=0xFFFF,
            fip=0x11223344, fcs=fcs_val, fop=0x0155, fdp=0x55667788, fds=fds_val,
        )
        nat_env = call3(native["op_fldenv_wait_fnstenv"], env_in, dummy, b"\x5a" * 28)
        lif_env = call3(lifted["op_fldenv_wait_fnstenv"], env_in, dummy, b"\x5a" * 28)
        assert_env28_deterministic(
            nat_env, lif_env, f"fldenv_wait_fnstenv_{sw_val:#x}",
            check_fip=True, check_fdp_fop=True, check_fcs_fds=False,
        )
        lif_parsed = unpack_env28(lif_env)
        assert (lif_parsed["sw"] & 0x4700) == (sw_val & 0x4700)
        assert (lif_parsed["fcs"], lif_parsed["fds"]) == (fcs_val, fds_val)
        _classify_fcs_fds(
            observed_fcs_fds_profiles, unpack_env28(nat_env), fcs_val, fds_val, f"fldenv_wait_{sw_val:#x}"
        )
        total += 1

    # 6. FNSAVE and FSAVE: 28-byte env + logical ST(0)..ST(7) + post-FNSAVE FNINIT state.
    for val in (POS_PI_APPROX, NEG_ONE, SUB_NEG_MID, QNAN_A, POS_INF, POS_ZERO):
        nat_st = call3(native["op_fnsave_stack"], pack80(val), pack16(0x027F), b"\x5a" * 108)
        lif_st = call3(lifted["op_fnsave_stack"], pack80(val), pack16(0x027F), b"\x5a" * 108)
        assert_env28_deterministic(
            nat_st[:28], lif_st[:28], f"fnsave_{val}", check_fip=False, check_fdp_fop=False
        )
        assert_fnsave_nonempty_regs_match(nat_st, lif_st, f"fnsave_{val}")
        lif_regs = unpack_state108(lif_st)[1]
        assert lif_regs[0] == POS_ZERO and lif_regs[1] == POS_ONE and lif_regs[2] == val
        total += 1

        nat_fst = call3(native["op_fsave_stack"], pack80(val), pack16(0x037F), b"\x5a" * 108)
        lif_fst = call3(lifted["op_fsave_stack"], pack80(val), pack16(0x037F), b"\x5a" * 108)
        assert_env28_deterministic(
            nat_fst[:28], lif_fst[:28], f"fsave_{val}", check_fip=False, check_fdp_fop=False
        )
        assert_fnsave_nonempty_regs_match(nat_fst, lif_fst, f"fsave_{val}")
        total += 1

    assert unpack16(call3(native["op_fnsave_post_cw"], pack80(POS_ONE), pack16(0x027F), b"\x5a" * 108)[:2]) == \
           unpack16(call3(lifted["op_fnsave_post_cw"], pack80(POS_ONE), pack16(0x027F), b"\x5a" * 108)[:2]) == 0x037F
    assert unpack16(call3(native["op_fnsave_post_sw"], pack80(SNAN_A), pack16(0x027F), b"\x5a" * 108)[:2]) == \
           unpack16(call3(lifted["op_fnsave_post_sw"], pack80(SNAN_A), pack16(0x027F), b"\x5a" * 108)[:2]) == 0x0000
    total += 2

    # 7. FRSTOR -> FNSAVE and FRSTOR -> FDECSTP -> FNSAVE across non-zero TOPs and full 80-bit register payloads.
    sample_regs = [
        POS_PI_APPROX,
        NEG_ONE,
        POS_ZERO,
        NEG_ZERO,
        SUB_POS_MIN,
        SUB_NEG_MID,
        POS_INF,
        QNAN_A,
    ]
    for top in (0, 3, 7):
        raw_tw = 0x1B4E
        env_bytes = pack_env28(
            cw=0x0B7F,
            sw=(top << 11) | SW_C1 | 0x4100,
            tw=raw_tw,
            fip=0x01020304, fcs=0x002B, fop=0x02A5,
            fdp=0x05060708, fds=0x0033,
        )
        state_in = env_bytes + b"".join(pack80(sample_regs[(i + top) & 7]) for i in range(8))
        want_tw = recomputed_ftw(raw_tw, sample_regs)

        nat_out = call3(native["op_frstor_fnsave"], state_in, dummy, b"\x5a" * 108)
        lif_out = call3(lifted["op_frstor_fnsave"], state_in, dummy, b"\x5a" * 108)
        assert_env28_deterministic(
            nat_out[:28], lif_out[:28], f"frstor_fnsave_top_{top}",
            check_fip=True, check_fdp_fop=True, check_fcs_fds=False,
        )
        assert nat_out[28:] == lif_out[28:] == state_in[28:]
        lif_out_env = unpack_env28(lif_out[:28])
        assert lif_out_env["tw"] == want_tw
        assert (lif_out_env["fcs"], lif_out_env["fds"]) == (0x002B, 0x0033)
        _classify_fcs_fds(
            observed_fcs_fds_profiles, unpack_env28(nat_out[:28]), 0x002B, 0x0033, f"frstor_fnsave_top_{top}"
        )

        nat_dec = call3(native["op_frstor_fdecstp_fnsave"], state_in, dummy, b"\x5a" * 108)
        lif_dec = call3(lifted["op_frstor_fdecstp_fnsave"], state_in, dummy, b"\x5a" * 108)
        assert_env28_deterministic(
            nat_dec[:28], lif_dec[:28], f"frstor_fdecstp_top_{top}", check_fip=False, check_fdp_fop=False
        )
        assert nat_dec[28:] == lif_dec[28:]
        dec_env, dec_regs = unpack_state108(lif_dec)
        assert (dec_env["fip"], dec_env["fcs"], dec_env["fop"], dec_env["fdp"], dec_env["fds"]) == (
            0x01020304, 0x002B, 0x02A5, 0x05060708, 0x0033
        ), f"frstor_fdecstp_top_{top}: preserved-metadata mismatch {dec_env}"
        assert dec_env["top"] == ((top - 1) & 7)
        assert (dec_env["sw"] & SW_C1) == 0
        assert dec_env["tw"] == want_tw
        # Logical ST(0) after FDECSTP is the previous ST(7)!
        assert dec_regs[0] == sample_regs[(7 + top) & 7]
        assert dec_regs[1] == sample_regs[top & 7]
        total += 2

    assert len(observed_fcs_fds_profiles) == 1, f"inconsistent host FCS/FDS behavior: {observed_fcs_fds_profiles}"
    native_fcs_fds_profile = next(iter(observed_fcs_fds_profiles))

    # 8. Glue conversions (m32<->m64, m80->m80, FSTP C1 cleared by FDECSTP).
    for f32 in (0.0, -0.0, 1.25, -3.5, 0x00000001):
        assert call3(native["op_fld_m32_fstp_m64"], pack_f32(f32), pack16(0x037F), b"\x5a" * 8) == \
               call3(lifted["op_fld_m32_fstp_m64"], pack_f32(f32), pack16(0x037F), b"\x5a" * 8)
        total += 1
    for cw in (0x037F, 0x077F, 0x0B7F, 0x0F7F):
        for f64 in (0.0, -0.0, 1.0 + 2.0 ** -30, -1.0 - 2.0 ** -30, 0x0000000000000001):
            assert call3(native["op_fld_m64_fstp_m32"], pack_f64(f64), pack16(cw), b"\x5a" * 4) == \
                   call3(lifted["op_fld_m64_fstp_m32"], pack_f64(f64), pack16(cw), b"\x5a" * 4)
            total += 1
    assert call3(native["op_fld_m80_fstp_m80"], pack80(UNNORM), dummy, b"\x5a" * 10) == \
           call3(lifted["op_fld_m80_fstp_m80"], pack80(UNNORM), dummy, b"\x5a" * 10) == pack80(UNNORM)
    # FSTP rounding up under RU sets C1=1, then FDECSTP clears C1 and sets TOP=7!
    nat_c1 = unpack16(call3(native["op_fstp_c1_fdecstp"], pack80((J | 1, 0x3FFF)), pack16(0x0B7F), b"\x5a" * 4))
    lif_c1 = unpack16(call3(lifted["op_fstp_c1_fdecstp"], pack80((J | 1, 0x3FFF)), pack16(0x0B7F), b"\x5a" * 4))
    assert (nat_c1 & (SW_TOP_MASK | SW_C1)) == (lif_c1 & (SW_TOP_MASK | SW_C1)) == 0x3800
    total += 2

    # 9. Altered-build controls.
    env_probe = pack_env28(cw=0x037F, sw=0x1000, tw=0xFFFF)
    state_probe = pack_env28(cw=0x037F, sw=0x1000, tw=0x5555) + b"".join(pack80(sample_regs[i]) for i in range(8))
    altered_probes = (
        ("op_fdecstp_sw", dummy, dummy, b"\x5a" * 2),
        ("op_fdecstp_wrap_env", env_probe, dummy, b"\x5a" * 28),
        ("op_fstenv_pushed", pack_f64(1.0), pack16(0x037F), b"\x5a" * 28),
        ("op_fldenv_fnstenv", env_probe, dummy, b"\x5a" * 28),
        ("op_fnsave_stack", pack80(POS_PI_APPROX), pack16(0x027F), b"\x5a" * 108),
        ("op_fsave_stack", pack80(POS_PI_APPROX), pack16(0x037F), b"\x5a" * 108),
        ("op_frstor_fnsave", state_probe, dummy, b"\x5a" * 108),
    )
    for name, a_b, b_b, o_b in altered_probes:
        orig_n = call3(native[name], a_b, b_b, o_b)
        alt_n = call3(altered[name], a_b, b_b, o_b)
        alt_l = call3(altered_lifted[name], a_b, b_b, o_b)
        if len(o_b) == 2:
            assert alt_n == alt_l and alt_n != orig_n, f"altered mismatch on {name}"
        else:
            check_fip = name == "op_fldenv_fnstenv"
            check_fdp = name == "op_fldenv_fnstenv"
            assert_env28_deterministic(alt_n[:28], alt_l[:28], f"alt_{name}", check_fip=check_fip, check_fdp_fop=check_fdp)
            if name == "op_frstor_fnsave":
                assert alt_n[28:] == alt_l[28:]
            elif len(o_b) == 108:
                assert_fnsave_nonempty_regs_match(alt_n, alt_l, f"alt_{name}")
            assert alt_n[:10] != orig_n[:10], f"altered build did not diverge on {name}"
        total += 1

    # Verify FNINIT does not corrupt caller-left physical registers when followed by empty-FTW FLDENV.
    leave_one = pack_env28(cw=0x037F, sw=0x0000, tw=0xCFFF) + b"".join(
        pack80(POS_ONE if i == 6 else POS_ZERO) for i in range(8)
    )
    call3(native["op_frstor_fnsave"], leave_one, dummy, b"\x5a" * 108)
    env_empty = pack_env28(cw=0x037F, sw=0x0000, tw=0xFFFF, fip=0x11112222, fop=0x0111)
    nat_after = call3(native["op_fdecstp_wrap_env"], env_empty, dummy, b"\x5a" * 28)
    lif_after = call3(lifted["op_fdecstp_wrap_env"], env_empty, dummy, b"\x5a" * 28)
    assert_env28_deterministic(nat_after, lif_after, "fninit_after_phys_one", check_fip=False, check_fdp_fop=False)
    total += 1

    return total, native_fdecstp_profile, native_fcs_fds_profile


def check_runtime_traps(lifted):
    """Verify out-of-profile unmasked exceptions, SW.ES=1, and uninit-reg FLDENV trap cleanly via __ud2()."""
    dummy = b"\x00" * 16
    trap_cases = (
        (
            "unmasked_snan_fstenv",
            "op_fstenv_pushed",
            pack_f64(0x7FF0000000000001),
            pack16(0x037E),
            28,
        ),
        (
            "unmasked_sw_es_frstor_fdecstp",
            "op_frstor_fdecstp_fnsave",
            pack_env28(cw=0x037E, sw=0x0081, tw=0xFFFF) + b"\x00" * 80,
            dummy,
            108,
        ),
        (
            "sw_es_frstor",
            "op_frstor_fnsave",
            pack_env28(cw=0x037F, sw=0x0081, tw=0xFFFF) + b"\x00" * 80,
            dummy,
            108,
        ),
        (
            "sw_es_fldenv",
            "op_fldenv_fnstenv",
            pack_env28(cw=0x037F, sw=0x0081, tw=0xFFFF),
            dummy,
            28,
        ),
        (
            "uninit_reg_fldenv_tw_cfff",
            "op_fdecstp_wrap_env",
            pack_env28(cw=0x037F, sw=0x0000, tw=0xCFFF),
            dummy,
            28,
        ),
    )
    for label, name, a_b, b_b, out_len in trap_cases:
        sentinel = b"\xcc" * out_len
        buf_a, _, _, _, a_addr = alloc_guarded(a_b)
        buf_b, _, _, _, b_addr = alloc_guarded(b_b)
        buf_o, _, o_off, _, o_addr = alloc_guarded(sentinel)
        try:
            lifted[name](a_addr, b_addr, o_addr)
        except OSError as exc:
            if (getattr(exc, "winerror", 0) or 0) & 0xFFFFFFFF != 0xC000001D:
                raise AssertionError(f"{label}: unexpected SEH {exc!r}") from exc
        else:
            raise AssertionError(f"{label}: lifted {name} returned instead of trapping")
        assert buf_o.raw[o_off:o_off + out_len] == sentinel, f"{label}: wrote output before trapping"
        _ = (buf_a, buf_b)
    return len(trap_cases)


def check_mxcsr_isolation(native, lifted, lifted_lib):
    get_mxcsr = lifted_lib.host_get_mxcsr
    get_mxcsr.argtypes, get_mxcsr.restype = [], ctypes.c_uint32
    xchg_mxcsr = lifted_lib.host_exchange_mxcsr
    xchg_mxcsr.argtypes, xchg_mxcsr.restype = [ctypes.c_uint32], ctypes.c_uint32

    saved = get_mxcsr()
    modes = (0x1F80, 0x1FC0, 0x9FC0, 0x3FC0)
    probes = (
        ("op_fld_m32_fstp_m64", pack_f32(0x00000001), pack16(0x037F), b"\x5a" * 8),
        ("op_fld_m64_fstp_m32", pack_f64(0x0000000000000001), pack16(0x037F), b"\x5a" * 4),
        ("op_fnsave_stack", pack80(SUB_POS_MIN), pack16(0x027F), b"\x5a" * 108),
    )
    checked = 0
    try:
        baselines = {}
        for mxcsr in modes:
            xchg_mxcsr(mxcsr)
            for idx, (name, a_b, b_b, o_b) in enumerate(probes):
                assert get_mxcsr() == mxcsr
                lif = call3(lifted[name], a_b, b_b, o_b)
                assert get_mxcsr() == mxcsr, f"lifted {name} clobbered MXCSR"
                nat = call3(native[name], a_b, b_b, o_b)
                assert get_mxcsr() == mxcsr
                if len(o_b) == 108:
                    assert_fnsave_nonempty_regs_match(nat, lif, f"mxcsr_{name}_{mxcsr:#x}")
                else:
                    assert nat == lif
                if idx not in baselines:
                    baselines[idx] = lif if len(o_b) <= 8 else lif[28:58]
                else:
                    cur = lif if len(o_b) <= 8 else lif[28:58]
                    assert cur == baselines[idx], f"lifted {name} changed under MXCSR {mxcsr:#x}"
                checked += 1
    finally:
        xchg_mxcsr(saved)
    return checked


MUTANT_SPECS = (
    ("mut_fdecstp_updates_fop_wrap", "op_fdecstp_wrap_env", "x87_fdecstp(x87);",
     "x87_fdecstp(x87); x87.fop = 0x01f6u; x87.fcs = 0x0033u;"),
    ("mut_fdecstp_updates_fop_frstor", "op_frstor_fdecstp_fnsave", "x87_fdecstp(x87);",
     "x87_fdecstp(x87); x87.fop = 0x01f6u; x87.fcs = 0x0033u;"),
    ("mut_fdecstp_no_wrap", "op_fdecstp_sw", "x87_fdecstp(x87);",
     "x87_set_c1(x87.sw, false);"),
    ("mut_fdecstp_keep_c1", "op_fdecstp_wrap_env", "x87_fdecstp(x87);",
     "x87.top = (x87.top - 1u) & 7u;"),
    ("mut_fnstenv_no_mask", "op_fnstenv_post_cw", "x87_fnstenv(x87, r8);",
     "x87_store_env28(x87, r8);"),
    ("mut_fnsave_phys_order", "op_frstor_fdecstp_fnsave", "x87_fnsave(x87, r8);",
     "x87_store_env28(x87, r8); for (unsigned i = 0u; i < 8u; ++i) "
     "store80(r8 + 28ull + static_cast<word>(i) * 10ull, x87.regs[i]); x87_fninit(x87);"),
    ("mut_frstor_no_recompute", "op_frstor_fnsave", "x87_frstor(x87, rcx);",
     "{ const std::uint16_t raw_tw = x87_load_env28(x87, rcx); "
     "for (unsigned i = 0u; i < 8u; ++i) x87_st(x87, i) = load80(rcx + 28ull + static_cast<word>(i) * 10ull); "
     "x87.regs_defined = 0xffu; x87.tw = raw_tw; }"),
    ("mut_fnsave_no_fninit", "op_fnsave_post_cw", "x87_fnsave(x87, r8);",
     "x87_store_env28(x87, r8); for (unsigned i = 0u; i < 8u; ++i) "
     "store80(r8 + 28ull + static_cast<word>(i) * 10ull, x87_st(x87, i));"),
    ("mut_fldenv_zero_fcs_fds", "op_fldenv_fnstenv", "x87_fldenv(x87, rcx);",
     "x87_fldenv(x87, rcx); x87.fcs = 0u; x87.fds = 0u;"),
)


def emit_mutants(original_path, out_cpp):
    """Emit compiled C++ mutation controls from translated functions."""
    unique_exports = []
    for _, src_name, _, _ in MUTANT_SPECS:
        if src_name not in unique_exports:
            unique_exports.append(src_name)
    base_src, _ = cpu_x87_env_aot.translate(original_path, unique_exports)
    chunks = [base_src]
    with pefile.PE(str(original_path), fast_load=True) as pe:
        by_name = {}
        for name in unique_exports:
            entry = export_rva(pe, name)
            fn_src, _ = lift_code(name, entry, bounded_code(pe, entry))
            by_name[name] = fn_src
    for mut_name, src_name, needle, repl in MUTANT_SPECS:
        src = by_name[src_name]
        if needle not in src:
            raise AssertionError(f"mutant {mut_name}: needle {needle!r} not found in {src_name}")
        mutated = src.replace(f"lifted_{src_name}", mut_name, 1).replace(needle, repl, 1)
        chunks.append(mutated)
    cpu_x87_env_aot._write_atomic(out_cpp, "\n".join(chunks) + "\n")


def check_mutants(mutants_path):
    """Run compiled C++ mutation DLL exports and verify every mutant breaks its target invariant."""
    lib = ctypes.CDLL(str(Path(mutants_path).resolve()))
    mut = {}
    for mut_name, _, _, _ in MUTANT_SPECS:
        fn = getattr(lib, mut_name)
        fn.argtypes = [ctypes.c_uint64] * 3
        fn.restype = None
        mut[mut_name] = fn

    dummy = b"\x00" * 16
    caught = 0

    # 1. Profile-distinguishing control: updating FOP/FCS on FDECSTP breaks preserved-metadata.
    env_in = pack_env28(cw=0x037F, sw=(3 << 11) | SW_C1, tw=0xFFFF, fip=0x12345678, fcs=0, fop=0x03AB, fdp=0x9ABCDEF0, fds=0)
    m_wrap = unpack_env28(call3(mut["mut_fdecstp_updates_fop_wrap"], env_in, dummy, b"\x5a" * 28))
    assert m_wrap["fop"] == 0x01F6 and m_wrap["fcs"] == 0x0033 and m_wrap["fop"] != 0x03AB
    caught += 1

    # 2. Profile-distinguishing control on FRSTOR -> FDECSTP -> FNSAVE.
    regs_in = [pack80((J | i, 0x3FFF + i)) for i in range(8)]
    st_in = pack_env28(cw=0x0B7F, sw=(3 << 11), tw=0x1B4E, fip=0x01020304, fcs=0, fop=0x02A5, fdp=0x05060708, fds=0) + b"".join(regs_in)
    m_fr = unpack_env28(call3(mut["mut_fdecstp_updates_fop_frstor"], st_in, dummy, b"\x5a" * 108)[:28])
    assert m_fr["fop"] == 0x01F6 and m_fr["fcs"] == 0x0033 and m_fr["fop"] != 0x02A5
    caught += 1

    # 3. Ignoring FDECSTP TOP wrap (0 -> 7).
    assert unpack16(call3(mut["mut_fdecstp_no_wrap"], dummy, dummy, b"\x5a" * 2)) == 0x0000
    caught += 1

    # 4. Not clearing FSW.C1 in FDECSTP.
    got_env = unpack_env28(call3(mut["mut_fdecstp_keep_c1"], env_in, dummy, b"\x5a" * 28))
    assert (got_env["sw"] & SW_C1) != 0
    caught += 1

    # 5. Storing physical regs[0..7] instead of logical ST(0)..ST(7) in FNSAVE.
    # At top=3 -> FDECSTP -> top=2, unmutated ST(0) is regs[2]=regs_in[7] while physical regs[0]=regs_in[5].
    st_out = call3(mut["mut_fnsave_phys_order"], st_in, dummy, b"\x5a" * 108)
    got_st0 = unpack80(st_out[28:38])
    assert got_st0 == unpack80(regs_in[5]) and got_st0 != unpack80(regs_in[7])
    caught += 1

    # 6. Skipping FTW recompute on FRSTOR.
    bad_tw_in = pack_env28(cw=0x037F, sw=0, tw=0x5555) + b"".join(pack80(POS_ONE) for _ in range(8))
    m_tw = unpack_env28(call3(mut["mut_frstor_no_recompute"], bad_tw_in, dummy, b"\x5a" * 108)[:28])
    assert m_tw["tw"] == 0x5555 and m_tw["tw"] != 0x0000
    caught += 1

    # 7. Failing to re-initialize FCW/FSW after FNSAVE.
    assert unpack16(call3(mut["mut_fnsave_no_fninit"], pack80(POS_ONE), pack16(0x027F), b"\x5a" * 108)[:2]) == 0x027F
    caught += 1

    # 8. Failing to mask FCW[5:0] after FNSTENV.
    assert unpack16(call3(mut["mut_fnstenv_no_mask"], pack16(0x0340), dummy, b"\x5a" * 28)[:2]) == 0x0340
    caught += 1

    # 9. Zeroing FCS/FDS on FLDENV breaks preserved-metadata even on hosts with CPUID FCS/FDS deprecation.
    fcs_in = pack_env28(cw=0x027F, sw=0x1800, tw=0xFFFF, fip=0x11223344, fcs=0x1234, fop=0x0155, fdp=0x55667788, fds=0x5678)
    m_fcs = unpack_env28(call3(mut["mut_fldenv_zero_fcs_fds"], fcs_in, dummy, b"\x5a" * 28))
    assert m_fcs["fcs"] == 0 and m_fcs["fds"] == 0
    caught += 1

    return caught


def record_covered_forms(original_path):
    record = os.environ.get("FORM_COVERAGE_RECORD")
    if not record:
        return
    formatter = Formatter(FormatterSyntax.INTEL)
    forms = set()
    with pefile.PE(str(original_path), fast_load=True) as pe:
        for name in EXPORTS:
            entry = export_rva(pe, name)
            raw = bounded_code(pe, entry)
            prev = None
            for ins in Decoder(64, raw, ip=entry):
                forms.add(formatter.format_mnemonic(ins).lower())
                if prev is not None and prev.code == Code.WAIT:
                    if ins.code == Code.FNSTENV_M28BYTE:
                        forms.add("fstenv")
                    elif ins.code == Code.FNSAVE_M108BYTE:
                        forms.add("fsave")
                prev = ins
    out_path = Path(record)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(word + "\n" for word in sorted(forms)), encoding="utf-8")


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "emit-mutants":
        emit_mutants(Path(sys.argv[2]), Path(sys.argv[3]))
        return

    if os.environ.get("FORM_COVERAGE_RECORD"):
        Path(os.environ["FORM_COVERAGE_RECORD"]).unlink(missing_ok=True)

    original_path, lifted_path, altered_path, altered_lifted_path, mutants_path = (
        Path(p) for p in sys.argv[1:6]
    )

    accepted, rejected = check_translator()
    print(f"{accepted} x87 env sequences accepted; {rejected} rejection controls passed")

    check_cli_controls(original_path)

    _, native = bind(original_path, "")
    lifted_lib, lifted = bind(lifted_path, "lifted_")
    _, altered = bind(altered_path, "")
    _, altered_lifted = bind(altered_lifted_path, "lifted_")

    diff_total, fdecstp_profile, fcs_fds_profile = check_differential(native, lifted, altered, altered_lifted)
    trap_count = check_runtime_traps(lifted)
    mxcsr_total = check_mxcsr_isolation(native, lifted, lifted_lib)
    mutant_count = check_mutants(mutants_path)

    print(
        f"{diff_total} differential/state/roundtrip/altered cases matched "
        f"(native FDECSTP FOP={fdecstp_profile}, FCS/FDS={fcs_fds_profile}); "
        f"{trap_count} runtime trap controls passed; "
        f"{mxcsr_total} MXCSR DAZ/FTZ isolation checks passed; "
        f"{mutant_count} compiled C++ mutant controls + {len(ALTERED_EXPORTS)} altered-build controls caught"
    )
    record_covered_forms(original_path)


if __name__ == "__main__":
    main()
