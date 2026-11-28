"""Differential checks (/Od and /O2), C++ & Python guard mutations, and refusal negative controls
for bounded frame-slot pointers passed to non-recursive internal callees in cpu_switch_aot.py."""

import ctypes
from pathlib import Path
import random
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import cpu_switch_aot as aot

MASK8 = 0xff
MASK16 = 0xffff
MASK32 = 0xffffffff
EXPORTS = ("out_u32_rsp", "rmw_u64_rbp", "all_widths_and_regs", "non_anchor_rbp_arith")

REFUSALS = (
    ("neg_escape_ret_rax", "escapes or is unsupported"),
    ("neg_escape_store_mem", "escapes or is unsupported"),
    ("neg_escape_store_stack", "escapes or is unsupported"),
    ("neg_recursive_cycle", "recursive callee"),
    ("neg_mutual_cycle", "recursive callee"),
    ("neg_tail_jump", "frame address escapes"),
    ("neg_tail_rbp", "unsupported stack operand"),
    ("neg_var_index_lea", "frame address escapes"),
    ("neg_multi_arg_alias_same", "aliased frame-slot pointers"),
    ("neg_multi_arg_alias_overlap", "aliased frame-slot pointers"),
    ("neg_multi_arg_copy_alias", "frame address escapes"),
    ("neg_home_space_overlap", "overlaps callee home space"),
    ("neg_ret_addr_overlap", "return address slot accessed"),
    ("neg_past_home_end", "unsupported stack argument"),
    ("neg_uninit_callee_read", "undefined stack byte"),
    ("neg_uninit_caller_wider_read", "undefined stack byte"),
    ("neg_cond_write_uninit_read", "undefined stack byte"),
    ("neg_mixed_widths", "escapes or is unsupported"),
    ("neg_unaccessed_ptr", "escapes or is unsupported"),
    ("neg_ptr_arith", "escapes or is unsupported"),
    ("neg_unproven_rbp", "nonvolatile register rbp may hold the caller's value"),
    ("neg_split_anchor_rbp", "frame pointer differs at join"),
    ("neg_tail_exit_partial_write", "undefined stack byte"),
    ("neg_cmov32_ptr_trunc", "escapes or is unsupported"),
    ("neg_bsf_cond_write_escape", "escapes or is unsupported"),
    ("neg_bsr_cond_write_escape", "escapes or is unsupported"),
    ("neg_callee_tail_ptr_live", "escapes or is unsupported"),
    ("neg_nested_int_callee", "escapes or is unsupported"),
    ("neg_nested_multi_live", "escapes or is unsupported"),
    ("neg_nested_cmov_ptr", "escapes or is unsupported"),
    ("neg_lea_disp_callee", "escapes or is unsupported"),
    ("neg_lea_index_callee", "escapes or is unsupported"),
    ("neg_mem_disp_callee", "escapes or is unsupported"),
    ("neg_mem_index_callee", "escapes or is unsupported"),
    ("neg_mem_ecx_callee", "escapes or is unsupported"),
    ("neg_stos_ptr_rax", "escapes or is unsupported"),
    ("neg_movs_ptr_rsi", "escapes or is unsupported"),
    ("neg_sign_ext_callee", "escapes or is unsupported"),
    ("neg_mul_ptr_callee", "escapes or is unsupported"),
    ("neg_partial_write_callee", "escapes or is unsupported"),
    ("neg_xor8_partial_callee", "escapes or is unsupported"),
    ("neg_bit_test_reg_callee", "escapes or is unsupported"),
    ("neg_cmov32_mem_callee", "escapes or is unsupported"),
    ("neg_may_write_saved_reg", "undefined stack byte"),
    ("neg_read_saved_reg", "hold the saved rbx, read as data"),
    ("neg_callee_stack_arg_overlap", "overlaps callee home space or stack arguments"),
    ("neg_incoming_stack_slot_uninit", "stack argument 5 of the call"),
)


def trace(first=MASK, second=MASK, third=MASK):
    return (GUARDS[0], first & MASK, second & MASK, third & MASK, GUARDS[1])


def ref_out_u32_rsp(x, y):
    val = ((x ^ y) ^ 0xA55A1234) & MASK32
    ret = (val & MASK8) | 1
    return (ret + val) & MASK, trace(val, ret)


def ref_rmw_u64_rbp(x, y):
    new_val = ((x + y) & MASK) ^ 0x1122334455667788
    return (x + new_val) & MASK, trace(x, new_val)


def ref_all_widths_and_regs(x, y):
    out64 = (x + y) & MASK
    out32 = (x - y) & MASK32
    ret1 = x ^ y

    slot16 = (x & MASK16)
    ax16 = (slot16 + (x & MASK16)) & MASK16
    ax16 = (ax16 ^ (out64 & MASK16)) & MASK16
    slot8 = ax16 & MASK8
    t1 = ax16 | (slot8 << 16)

    nv_val = (((out32 * 3) + 7) & MASK) + x
    nv_val &= MASK

    fwd_slot = ((x & MASK32) + 9) & MASK32
    fwd_ret = (fwd_slot + 0x13) & MASK32
    acc = (fwd_ret + fwd_slot) & MASK

    cond_slot = x & MASK32
    cond_ret = out32 & 1
    if cond_ret:
        cond_slot = out32
    acc = (acc + ((cond_ret + cond_slot) & MASK32)) & MASK

    wtr_slot = x & MASK32
    wtr_ret = (wtr_slot + 5) & MASK32
    acc = (acc + wtr_ret + wtr_slot + nv_val) & MASK
    return acc, trace(ret1, t1, nv_val)


def ref_non_anchor_rbp_arith(x, y):
    del y
    arg = (x + 8) & MASK
    mixed = (arg * 3 + 7) & MASK
    return (mixed + x) & MASK, trace(mixed)


REFERENCES = {
    "out_u32_rsp": ref_out_u32_rsp,
    "rmw_u64_rbp": ref_rmw_u64_rbp,
    "all_widths_and_regs": ref_all_widths_and_regs,
    "non_anchor_rbp_arith": ref_non_anchor_rbp_arith,
}


def pairs():
    rng = random.Random(3170)
    base = input_pairs()
    return base + [(rng.getrandbits(64), rng.getrandbits(64)) for _ in range(64)]


def verify_dll(probe, dll_path, prefix, expected_map, test_inputs):
    lib = ctypes.CDLL(str(dll_path))
    try:
        for name in EXPORTS:
            fn = ctypes.cast(getattr(lib, prefix + name), ctypes.c_void_p).value
            for x, y in test_inputs:
                got = observe(probe, fn, x, y)
                want_rax, want_trace = expected_map[name](x, y)
                if got != (want_rax, want_trace, SENTINELS):
                    raise AssertionError(
                        f"{dll_path.name} {prefix}{name}({x:#x}, {y:#x}) = {got}, "
                        f"expected ({want_rax:#x}, {want_trace}, {SENTINELS})"
                    )
    finally:
        handle = ctypes.c_void_p(lib._handle)
        del lib
        ctypes.windll.kernel32.FreeLibrary(handle)


def compile_cpp_dll(cxx_path, cpp_path, dll_path, opt_flag="/O2"):
    obj_path = dll_path.with_suffix(".obj")
    cmd = [
        str(cxx_path),
        "/nologo",
        opt_flag,
        "/W4",
        "/LD",
        str(cpp_path),
        f"/Fo{obj_path}",
        f"/Fe{dll_path}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise AssertionError(f"failed to compile {cpp_path.name} ({opt_flag}):\n{res.stdout}\n{res.stderr}")


def check_structure(source):
    if re.search(r"\brsp\b", source):
        raise AssertionError("RSP leaked into generated C++")
    if "reinterpret_cast<word>(frame +" not in source:
        raise AssertionError("expected frame-slot pointer passed via reinterpret_cast<word>(frame + ...)")
    for name in EXPORTS:
        if f"lifted_{name}(" not in source:
            raise AssertionError(f"missing lifted_{name}")


def check_cpp_mutants(probe, cxx_path, source, test_inputs):
    """Mutate key emitted frame-pointer semantics and verify differential checks catch them."""
    # Mutant 1: shift the frame-slot pointer argument offset passed to fn_*.
    mut1, n1 = re.subn(
        r"(fn_[0-9a-f]+\([^)]*reinterpret_cast<word>\(frame \+ )(\d+)(\))",
        lambda m: f"{m.group(1)}{int(m.group(2)) + 4}{m.group(3)}",
        source,
        count=1,
    )
    if n1 != 1:
        raise AssertionError(f"expected 1 frame-arg replacement in mutant 1, got {n1}")

    # Mutant 2: drop the store inside the internal callee that writes through the pointer.
    mut2, n2 = re.subn(
        r"store<std::uint32_t>\(\(rcx\), [^;]+\);",
        "/* mutated out store */",
        source,
        count=1,
    )
    if n2 != 1:
        raise AssertionError(f"expected 1 store replacement in mutant 2, got {n2}")

    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        for label, mut_src in (("shift_frame_ptr", mut1), ("drop_callee_store", mut2)):
            cpp = tmp / f"{label}.cpp"
            dll = tmp / f"{label}.dll"
            cpp.write_text(mut_src, encoding="utf-8")
            compile_cpp_dll(cxx_path, cpp, dll, "/O2")
            try:
                verify_dll(probe, dll, "lifted_", REFERENCES, test_inputs[:16])
            except AssertionError as exc:
                print(f"cpp mutant {label} caught: {exc}")
            else:
                raise AssertionError(f"cpp mutant {label} unexpectedly passed")


def check_refusals_and_guard_mutations(native_path):
    for name, reason in REFUSALS:
        try:
            aot.translate(native_path, [name])
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{name}: expected {reason!r}, got {exc}") from exc
        else:
            raise AssertionError(f"{name}: unexpectedly accepted (expected {reason!r})")
        print(f"refused {name}: {reason}")

    # Guard mutation 1: disable post-call byte marking / pre-call check by replacing prove_frame
    # with a version that strips frame.ptr_calls during prove_frame.
    orig_prove = aot.prove_frame

    def prove_without_ptr_calls(function, order, effects, frame, stacks=None):
        saved_ptr_calls = frame.ptr_calls
        frame.ptr_calls = {}
        try:
            return orig_prove(function, order, effects, frame, stacks)
        finally:
            frame.ptr_calls = saved_ptr_calls

    aot.prove_frame = prove_without_ptr_calls
    try:
        try:
            aot.translate(native_path, ["neg_uninit_callee_read"])
        except ValueError as exc:
            raise AssertionError(
                f"neg_uninit_callee_read still refused without ptr_calls proof: {exc}"
            ) from exc
        print("guard mutation (disable prove_frame ptr_calls) confirmed: neg_uninit_callee_read accepted")
    finally:
        aot.prove_frame = orig_prove

    # Guard mutation 2: verify out_u32_rsp fails if ptr_calls post-call must_write marking is skipped.
    orig_resolve = aot.Frame.resolve_pointer_calls

    def resolve_without_must_write(self, callees, program, stacks):
        orig_resolve(self, callees, program, stacks)
        for call_ip, ptr_map in self.ptr_calls.items():
            self.ptr_calls[call_ip] = {
                r: (ins, off, aot.PointerParam(p.width, p.reads_before_write, False, False))
                for r, (ins, off, p) in ptr_map.items()
            }

    aot.Frame.resolve_pointer_calls = resolve_without_must_write
    try:
        try:
            aot.translate(native_path, ["out_u32_rsp"])
        except ValueError as exc:
            if "undefined stack byte" not in str(exc):
                raise AssertionError(f"unexpected error when must_write disabled: {exc}") from exc
            print("guard mutation (disable post-call must_write marking) confirmed: out_u32_rsp refused")
        else:
            raise AssertionError("out_u32_rsp accepted without post-call must_write marking")
    finally:
        aot.Frame.resolve_pointer_calls = orig_resolve


def main(probe_path, native_path, lifted_path, lifted_cpp_path, cxx_path):
    probe = register_probe(probe_path)
    test_inputs = pairs()
    source, _ = aot.translate(native_path, list(EXPORTS))
    built_cpp = Path(lifted_cpp_path).read_text(encoding="utf-8")
    if built_cpp != source:
        raise AssertionError(f"{lifted_cpp_path} does not match cpu_switch_aot.translate output")
    check_structure(source)

    verify_dll(probe, native_path, "", REFERENCES, test_inputs)
    verify_dll(probe, lifted_path, "lifted_", REFERENCES, test_inputs)

    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        for opt_flag in ("/Od", "/O2"):
            cpp = tmp / f"lifted_{opt_flag.strip('/')}.cpp"
            dll = tmp / f"lifted_{opt_flag.strip('/')}.dll"
            cpp.write_text(source, encoding="utf-8")
            compile_cpp_dll(cxx_path, cpp, dll, opt_flag)
            verify_dll(probe, dll, "lifted_", REFERENCES, test_inputs)
            print(f"verified {len(EXPORTS)} exports x {len(test_inputs)} inputs ({opt_flag})")

    check_cpp_mutants(probe, cxx_path, source, test_inputs)
    check_refusals_and_guard_mutations(native_path)
    print(f"cpu_switch_frame_ptr: all differential (/Od, /O2), mutant, and {len(REFUSALS)} refusal checks passed")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit("usage: switch_frame_ptr_check.py PROBE NATIVE LIFTED LIFTED_CPP CXX")
    main(*map(Path, sys.argv[1:5]), Path(sys.argv[5]))
