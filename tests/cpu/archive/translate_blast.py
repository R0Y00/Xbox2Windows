"""Lift the scalar MSVC /Od blast PE, including its six-argument entry and two callback ABIs."""
import argparse
from pathlib import Path

import pefile
from iced_x86 import MemorySizeExt, OpKind, RegisterExt

from cpu_switch_aot import (Entries, Frame, Function, HostImport, Lifter, M, PRELUDE, R,
                            VOLATILE, check_call_sites, export_rva, extent, flag_producer, relocated)


class CodecFrame(Frame):
    frame_limit = 8192  # Includes blast's 4096-byte sliding window.
    incoming_end = 56   # Return slot, home space, left and in.

    def step(self, ins, rsp, anchor):
        if ins.mnemonic == M.MOV and ins.op1_kind == OpKind.REGISTER and ins.op1_register == R.RSP:
            if ins.op0_register != R.RDX:
                raise ValueError(f"unexpected frame pointer copy at {ins.ip:#x}")
            return rsp, anchor
        if ins.mnemonic == M.LEA and ins.memory_base == R.RSP:
            if ins.memory_index != R.NONE or ins.has_segment_prefix:
                raise ValueError(f"unsupported stack address at {ins.ip:#x}")
            self.access(ins, rsp, rsp + ins.memory_displacement, 1, False, False)
            return rsp, anchor
        if ins.memory_base == R.RSP and ins.memory_index != R.NONE:
            if ins.has_segment_prefix or not RegisterExt.is_gpr64(ins.memory_index):
                raise ValueError(f"unexpected indexed stack operand at {ins.ip:#x}")
            return rsp, anchor
        return super().step(ins, rsp, anchor)


class CodecLifter(Lifter):
    def address(self, ins):
        if ins.memory_base == R.RIP:
            section = self.image.get_section_by_rva(ins.ip_rel_memory_address)
            if section is None or section.Name.rstrip(b"\0") not in (b".data", b".rdata"):
                raise ValueError(f"unsupported image operand at {ins.ip:#x}")
        if ins.memory_base == R.RSP and ins.memory_index != R.NONE:
            delta = self.offsets[ins.ip]
            index = self.reg(ins.memory_index)
            return (f"indexed_slot(sp, {delta}, {ins.memory_displacement}ull + "
                    f"{index} * {ins.memory_index_scale}u, {MemorySizeExt.size(ins.memory_size)})")
        return super().address(ins)

    def slot(self, ins):
        return f"(sp + ({self.frame[ins.ip]}))"


def data_image(pe):
    """Materialize data and rebase its internal pointers, without including executable bytes."""
    size = pe.OPTIONAL_HEADER.SizeOfImage
    if size > 1 << 20:
        raise ValueError("unexpected codec image size")
    lines = [f"struct Data {{ alignas(16) unsigned char bytes[{size}]{{}}; Data() {{"]
    sections = [s for s in pe.sections if s.Name.rstrip(b"\0") in (b".data", b".rdata")]
    for section in sections:
        data = section.get_data()[:extent(section)].rstrip(b"\0")
        if data:
            literals = ", ".join(map(str, data))
            lines.append(f"static const unsigned char s_{section.VirtualAddress:x}[] = {{{literals}}};")
            lines.append(f"std::memcpy(bytes + {section.VirtualAddress}, s_{section.VirtualAddress:x}, {len(data)});")
    for slot in relocated(pe):
        dest = pe.get_qword_at_rva(slot) - pe.OPTIONAL_HEADER.ImageBase
        for address, n in ((slot, 8), (dest, 1)):
            if not any(s.VirtualAddress <= address and address + n <= s.VirtualAddress + extent(s) for s in sections):
                raise ValueError(f"non-data relocation at {slot:#x}")
        lines.append(f"store<word>(word(bytes + {slot}), word(bytes + {dest}));")
    lines.append("} };")
    return "\n".join(lines)


SIGNATURE = "static word fn_{:x}(Context& ctx, word sp, word rcx, word rdx, word r8, word r9)"


def emit(pe, fn, imports):
    edges = fn.explore(imports, callback_calls=True)
    frame = CodecFrame(fn, edges)
    check_call_sites(fn, frame.offsets)
    lifter = CodecLifter(pe)
    lifter.frame = {ip: offset for ip, (offset, *_) in frame.accesses.items()}
    lifter.offsets = frame.offsets
    lifter.adjustments = frame.adjustments
    branches = {ip: flag_producer(fn, edges, ins) for ip, ins in fn.instructions.items() if ins.is_jcc_short_or_near}
    flags = {ins.ip for ins in branches.values()}
    callees = set()
    lines = []
    for ip, ins in sorted(fn.instructions.items()):
        rsp = f"(sp + ({frame.offsets[ip]}))"
        if ins.has_lock_prefix or ins.has_rep_prefix or ins.has_repne_prefix or ins.has_segment_prefix:
            raise ValueError(f"unsupported prefix at {ip:#x}")
        if ip in fn.host_calls:
            name = fn.host_calls[ip].symbol
            if name == "save":
                code = f"""ctx.environment = rcx;
    if (setjmp(ctx.resume)) {{
        rax = ctx.jump_result;
        rcx = rdx = r8 = r9 = r10 = r11 = fa = fb = 0;
    }} else rax = 0;"""
            else:
                code = "if (!ctx.environment || rcx != ctx.environment) std::abort(); " \
                       "ctx.jump_result = static_cast<unsigned>(rdx); " \
                       "if (!ctx.jump_result) ctx.jump_result = 1; std::longjmp(ctx.resume, 1);"
        elif ip in fn.calls:
            target = fn.calls[ip]
            if target is None:
                if ins.memory_base == R.RSP:
                    if ins.memory_index != R.NONE or ins.has_segment_prefix:
                        raise ValueError(f"unexpected callback slot at {ip:#x}")
                    frame.access(ins, frame.offsets[ip], frame.offsets[ip] + ins.memory_displacement, 8, True, False)
                    lifter.frame[ip] = frame.accesses[ip][0]
                address = lifter.address(ins)
                code = f"rax = ctx.callback(load<word>({address}), rcx, rdx, r8);"
            else:
                callees.add(target)
                code = f"rax = fn_{target:x}(ctx, {rsp} - 8, rcx, rdx, r8, r9);"
        elif ip in branches:
            code = f"if ({lifter.condition(branches[ip], ins)}) goto loc_{ins.near_branch_target:x};"
        elif ins.is_jmp_short_or_near:
            code = f"goto loc_{ins.near_branch_target:x};"
        elif ins.mnemonic == M.CDQE:
            code = "rax = static_cast<std::int32_t>(rax);"
        elif ins.mnemonic == M.MOV and ins.op1_kind == OpKind.REGISTER and ins.op1_register == R.RSP:
            code = f"rdx = {rsp};"
        else:
            code = lifter.statement(ins, ip in flags)[0]
        lines.append(f"loc_{ip:x}: {code}")
    if fn.tails or lifter.used - VOLATILE:
        raise ValueError(f"unsupported tail or nonvolatile operand in function {fn.entry:#x}")
    prefix = [SIGNATURE.format(fn.entry) + " {",
              f"if (sp < word(ctx.stack) + {-frame.start} || sp + 56 > word(ctx.stack + sizeof ctx.stack)) std::abort();",
              "const word image = word(ctx.data.bytes);", "word rax=0, r10=0, r11=0, fa=0, fb=0;",
              f"goto loc_{fn.entry:x};"]
    return "\n".join(prefix + lines + ["}"]), callees


def translate(path):
    with pefile.PE(str(path)) as pe:
        bindings = {b"__intrinsic_setjmp": HostImport("save", (8, 8), 4),
                    b"longjmp": HostImport("resume", (8, 4), 0, noreturn=True)}
        imports = {}
        for module in pe.DIRECTORY_ENTRY_IMPORT:
            for item in module.imports:
                if module.dll.lower() != b"vcruntime140.dll" or item.name not in bindings:
                    raise ValueError(f"unexpected codec import {module.dll!r}!{item.name!r}")
                imports[item.address - pe.OPTIONAL_HEADER.ImageBase] = bindings[item.name]
        entry = export_rva(pe, "blast")
        pending = [entry]
        functions = {}
        entries = Entries(pe)
        while pending:
            target = pending.pop()
            if target not in functions:
                fn = Function(pe, target, entries)
                functions[target], calls = emit(pe, fn, imports)
                pending.extend(sorted(calls))
        runtime = Path(__file__).with_name("blast_runtime.hpp").read_text()
        wrapper = f"""
extern "C" int blast(blast_in input, void* inhow, blast_out output, void* outhow,
                                          unsigned* left, unsigned char** in) {{
    static thread_local Data data;
    Context ctx{{data, input, output}};
    word sp = word(ctx.stack + sizeof ctx.stack) - 56;
    store<word>(sp + 40, word(left));
    store<word>(sp + 48, word(in));
    return static_cast<int>(fn_{entry:x}(ctx, sp, word(input), word(inhow), word(output), word(outhow)));
}}
"""
        declarations = [SIGNATURE.format(rva) + ";" for rva in functions]
        return "\n".join([PRELUDE, data_image(pe), runtime, *declarations, *functions.values(), wrapper])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(translate(args.image), encoding="ascii")
