"""Differential returns and unwind ownership controls for a split x64 function."""

import ctypes
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Code, Decoder

from direct_check import SENTINELS, Word, observe, register_probe
from cpu_chained_aot import translate
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges


def check(native, compiled, harness, native_jne=None, compiled_jne=None,
          native_zero=None, compiled_zero=None, native_imm=None, compiled_imm=None,
          native_frame=None, compiled_frame=None):
    probe = register_probe(str(harness))
    pairs = [(native, compiled, 7)]
    if native_jne is not None and compiled_jne is not None:
        pairs.append((native_jne, compiled_jne, 7))
    if native_zero is not None and compiled_zero is not None:
        pairs.append((native_zero, compiled_zero, 0))
    if native_imm is not None and compiled_imm is not None:
        pairs.append((native_imm, compiled_imm, 7))
    if native_frame is not None and compiled_frame is not None:
        pairs.append((native_frame, compiled_frame, 7))
    for orig_path, lifted_path, null_expected in pairs:
        source = ctypes.CDLL(str(orig_path)).choose_value
        lifted = ctypes.CDLL(str(lifted_path)).try_lifted
        for value, expected in ((None, null_expected), (0, null_expected), (1, 21),
                                (0xffffffffffffffff, 21), (1 << 32, 21)):
            slot = Word(value or 0)
            address = 0 if value is None else ctypes.addressof(slot)
            result, trace, regs = observe(probe, source, address, 0x1122334455667788)
            status, actual, kept = observe(probe, lifted, address, 0x1122334455667788)
            if (result != expected or regs != SENTINELS or kept != regs or status != 1
                    or actual != (trace[0], result, *trace[2:]) or slot.value != (value or 0)):
                raise AssertionError((orig_path.name, value, result, actual, kept))

    with pefile.PE(str(native), fast_load=True) as pe:
        entry = export_rva(pe, "choose_value")
        neighbor = export_rva(pe, "neighbor")
        ranges = function_ranges(pe, entry)
        assert len(ranges) == 3 and all(not a <= neighbor < b for a, b in ranges)
        entry_offset = pe.get_offset_from_rva(entry)
        body = pe.get_data(entry, ranges[0][1] - entry)
        jump = next(i for i in Decoder(64, body, ip=entry) if i.code == Code.JE_REL8_64)
        jump_opcode_offset = pe.get_offset_from_rva(jump.ip)
        jump_offset = pe.get_offset_from_rva(jump.ip + 1)
        move = next(i for i in Decoder(64, body, ip=entry) if i.code == Code.MOV_R32_IMM32)
        move_offset = pe.get_offset_from_rva(move.ip)
        info = next(r.struct.UnwindData for r in pe.DIRECTORY_ENTRY_EXCEPTION
                    if r.struct.BeginAddress == ranges[1][0])
        header_offset = pe.get_offset_from_rva(info)
        parent_offset = pe.get_offset_from_rva(info + 4)

    data = native.read_bytes()
    assert data[header_offset] == (4 << 3) | 1
    expected_translation = translate(native, "choose_value")
    controls = [(jump_offset, struct.pack("b", target - jump.next_ip), "outside owned instruction")
                for target in (neighbor, ranges[1][0] + 1)]
    controls += [(parent_offset, struct.pack("<I", entry + 1), "invalid unwind chain"),
                 (parent_offset, struct.pack("<III", *ranges[1], info), "invalid unwind chain"),
                 (header_offset, bytes([(4 << 3) | 0]), "unsupported chained unwind header"),
                 (header_offset, bytes([(4 << 3) | 3]), "unsupported chained unwind header"),
                 (header_offset, bytes([(5 << 3) | 2]), "unsupported chained unwind header"),
                 (jump_opcode_offset, b"\x78", "expected JE or JNE after TEST"),
                 (move_offset, bytes.fromhex("31 c8 90 90 90"), "unsupported operation"),
                 (move_offset, bytes.fromhex("bf 15 00 00 00"), "unrestored state")]
    with TemporaryDirectory() as folder:
        for offset, patch, reason in controls:
            damaged = bytearray(data)
            damaged[offset:offset + len(patch)] = patch
            path = Path(folder) / "control.dll"
            path.write_bytes(damaged)
            try:
                translate(path, "choose_value")
            except ValueError as exc:
                if reason not in str(exc):
                    raise
            else:
                raise AssertionError(f"accepted control: {reason}")
        # Version-2 UNWIND_INFO keeps the same chained RUNTIME_FUNCTION trailer layout as version 1.
        v2 = bytearray(data)
        v2[header_offset] = (4 << 3) | 2
        v2_path = Path(folder) / "v2.dll"
        v2_path.write_bytes(v2)
        assert translate(v2_path, "choose_value") == expected_translation
        # Preserve instruction boundaries after a synthetic call through [rbx].
        damaged = bytearray(data)
        damaged[move_offset:move_offset + 5] = bytes.fromhex("ff 13 90 90 90")
        path.write_bytes(damaged)
        _, _, returns, stops = translate(path, "choose_value", partial=True)
        assert len(returns) == 2 and stops == {move.ip}
        try:
            translate(path, "choose_value")
        except ValueError as exc:
            if "unsupported operation" not in str(exc):
                raise
        else:
            raise AssertionError("strict translation accepted an indirect call")
        if native_frame is not None:
            with pefile.PE(str(native_frame), fast_load=True) as pe:
                frame_entry = export_rva(pe, "choose_value")
                frame_ranges = function_ranges(pe, frame_entry)
                primary = list(Decoder(64, pe.get_data(frame_entry, frame_ranges[0][1] - frame_entry), ip=frame_entry))
                finish = list(Decoder(64, pe.get_data(frame_ranges[2][0], frame_ranges[2][1] - frame_ranges[2][0]), ip=frame_ranges[2][0]))
                sub_ins = next(i for i in primary if i.code == Code.SUB_RM64_IMM32)
                assert any(i.code == Code.ADD_RM64_IMM32 for i in finish)
                imm_offset = pe.get_offset_from_rva(sub_ins.next_ip - 4)
            oversized = bytearray(native_frame.read_bytes())
            oversized[imm_offset:imm_offset + 4] = struct.pack("<i", 4096)
            over_path = Path(folder) / "oversized_frame.dll"
            over_path.write_bytes(oversized)
            try:
                translate(over_path, "choose_value")
            except ValueError as exc:
                if "unsupported stack adjustment" not in str(exc):
                    raise
            else:
                raise AssertionError("accepted stack frame exceeding 4096 bytes")
        # Translation-only controls. Use 40 bytes for shadow space and call alignment.
        def callback_image(slot, save="", result=""):
            code = bytes.fromhex(f"48 83 ec 28 {save} ff 51 {slot} {result} 48 83 c4 28 c3")
            assert len(code) <= len(body)
            patched = bytearray(data)
            patched[entry_offset:entry_offset + len(body)] = code.ljust(len(body), b"\x90")
            path.write_bytes(patched)

        callback_image("08")
        _, _, returns, stops = translate(path, "choose_value", callback_offset=8)
        assert len(returns) == 1 and not stops

        # Each control is refused at the instruction it changes: the call, the RDX read or the shadow-slot reload.
        for label, slot, save, result, at in (
            ("wrong table slot", "10", "", "", 4),
            ("volatile RDX after callback", "08", "", "48 89 d0", 7),
            ("first shadow slot after callback", "08", "48 89 14 24", "48 8b 04 24", 11),
        ):
            callback_image(slot, save, result)
            try:
                translate(path, "choose_value", callback_offset=8)
            except ValueError as exc:
                if f"unsupported operation at {entry + at:#x}" not in str(exc):
                    raise
            else:
                raise AssertionError(f"accepted callback control: {label}")
    print("two null paths, three nonzero inputs (JE, JNE, XOR zeroing, 32-bit register writes and 32-bit stack frames), RBX/RSI, guarded writes, version-2 chained unwind and fifteen rejection controls passed")


if __name__ == "__main__":
    check(*map(Path, sys.argv[1:]))
