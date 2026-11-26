"""Translate the two Bink leaf functions that build grouped allocation descriptors."""
import argparse
import os
from pathlib import Path
import struct
import tempfile

import pefile
from iced_x86 import Mnemonic as M, OpKind as K, Register as R

from bink_heap_aot import leaf_body
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, checked_image, lower, region


BINK_SHA256 = "9eaf045d004cd7d6a8382aaff0c6ea23a29fec4ef8bfacd726b54f5bc26fafd0"


def frame_body(pe: pefile.PE, package: bool, bind_group: bool = False) -> str:
    if package:
        # One loop iteration after pitch selection. The cursor starts at plane 0's pitch.
        start, end = 0x2257, 0x2325
        body = region(pe, start, end)
        initial = ["rdi = frames; rbp = index; rbx = frames + 36 + index * 80;",
                   "r14 = pitches[0]; r15 = pitches[1];"]
        calls = {ins.ip: "observe(rdx, uint32_t(r8));" for ins in body if ins.mnemonic == M.CALL
                 and ins.op0_kind == K.NEAR_BRANCH64 and ins.near_branch_target == 0x25b20}
        exits = {end: "return 0;"}
    else:
        (bounds,) = function_ranges(pe, export_rva(pe, "frame_planes"))
        body = region(pe, *bounds)
        initial = ["rsp = reinterpret_cast<word>(frame + 12); rcx = frames; rdx = index;",
                   "r8 = reinterpret_cast<word>(pitches);"]
        binding = "rax = group_append(group, rcx, uint32_t(rdx));"
        if not bind_group:
            initial.append("r9 = reinterpret_cast<word>(observe);")
            binding = "reinterpret_cast<Observer>(r9)(rcx, uint32_t(rdx));"
        calls = {ins.ip: binding for ins in body
                 if ins.mnemonic == M.CALL and ins.op0_register == R.R9}
        exits = {}
    if len(calls) != 5:
        raise ValueError("expected five frame plane request sites")
    return lower(body, initial, {}, calls, exits)


def generate(image: Path, package: bool, frame_planes: bool = False) -> str:
    with (checked_image(image, BINK_SHA256) if package else pefile.PE(str(image), fast_load=True)) as pe:
        init, append = ((0x25b80, 0x25b20) if package else
                        (export_rva(pe, "group_init"), export_rva(pe, "group_append")))
        init_body = lower(leaf_body(pe, init, {}), ["rcx = group; rdx = capacity;"], {}, {}, {})
        append_code = leaf_body(pe, append, {})
        # Expose the capacity breakpoint as a status to the desktop test.
        stops = {ins.ip: "return 0;" for ins in append_code if ins.mnemonic == M.INT3}
        if len(stops) != 1:
            raise ValueError("expected one capacity breakpoint")
        stops.update({ins.ip: "return 1;" for ins in append_code if ins.mnemonic == M.RET})
        append_body = lower(append_code, ["rcx = group; rdx = destination; r8 = size;"],
                            {}, {}, {}, host_ops=stops)
        planes = ""
        if frame_planes:
            planes = ('using Observer = void (*)(word, uint32_t);\n'
                      'extern "C" __declspec(dllexport) word frame_planes(word frames, uint32_t index, '
                      'const uint32_t* pitches, Observer observe) {\n' + frame_body(pe, package) + '\n}\n')
            if not package:
                planes += ('extern "C" word append_frame(word frames, uint32_t index, '
                           'const uint32_t* pitches, word group) {\n'
                           + frame_body(pe, False, bind_group=True) + '\n}\n')
    return (INTEGER_HEADER
            + 'extern "C" __declspec(dllexport) word group_init(word group, uint32_t capacity) {\n'
            + init_body + '\n}\n'
            + 'extern "C" __declspec(dllexport) word group_append(word group, word destination, word size) {\n'
            + append_body + '\n}\n' + planes)


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.image.is_dir():
        raise IsADirectoryError(f"image path is a directory: {args.image}")
    if not args.image.is_file():
        raise FileNotFoundError(f"image file not found: {args.image}")
    if args.output.suffix.lower() != ".cpp":
        parser.error("choose a distinct .cpp output")
    if args.output.is_symlink():
        raise ValueError(f"output path must not be a symlink: {args.output}")
    if args.output.is_dir():
        raise IsADirectoryError(f"output path is a directory: {args.output}")
    for parent in args.output.parents:
        if parent.is_file() or ((parent.exists() or parent.is_symlink()) and not parent.is_dir()):
            raise NotADirectoryError(f"output parent is not a directory: {parent}")
    if args.output.resolve() == args.image.resolve() or (args.output.exists() and args.image.samefile(args.output)):
        parser.error("choose a distinct .cpp output")


def _write_atomic(path: Path, content: str) -> None:
    """Write ASCII content to a sibling temp file and atomically replace path, leaving 0 temp files on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--frame-planes", action="store_true")
    args = parser.parse_args(argv)
    try:
        _validate_args(args, parser)
        _write_atomic(args.output, generate(args.image, args.package, args.frame_planes))
    except (OSError, ValueError, struct.error, pefile.PEFormatError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
