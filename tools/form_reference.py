"""Regenerate the instruction reference lists in docs/coverage/ (rarely needed).

Usage: out/pe-env/Scripts/python.exe tools/form_reference.py <llvm-docs-dir> [--out-dir DIR]

<llvm-docs-dir> holds AMDGPUAsmGFX10.rst, AMDGPUAsmGFX1011.rst, AMDGPUAsmGFX1013.rst and
AMDGPUAsmGFX1030.rst from llvm-project tag llvmorg-15.0.7, llvm/docs/AMDGPU/ (Apache-2.0 WITH
LLVM-exception). These are generated from LLVM's instruction tables. The x64 list comes from the
iced-x86 package already in requirements-dev.txt. Output lines are "group<TAB>form".
"""
import argparse
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

OUT = Path(__file__).resolve().parents[1] / "docs" / "coverage"
LLVM_SOURCE = "llvm-project llvmorg-15.0.7 llvm/docs/AMDGPU/"
REQUIRED_GFX1013_RST = ("AMDGPUAsmGFX10.rst", "AMDGPUAsmGFX1011.rst", "AMDGPUAsmGFX1013.rst")
REQUIRED_RDNA2_RST = ("AMDGPUAsmGFX1030.rst",)
OUTPUT_TSV_NAMES = ("gfx1013.tsv", "rdna2.tsv", "x64_zen2.tsv")

# Xbox Series CPUs are AMD Zen 2. These iced CPUID groups are what Zen 2 runs in user mode; the group
# names are what the dashboard shows. AVX-512, XOP/FMA4/TBM, 3DNow!, AMX and vendor extras are left out.
ZEN2 = {
    "Integer": "INTEL8086 INTEL186 INTEL286 INTEL386 INTEL486 X64 CMOV CX8 CMPXCHG16B MOVBE POPCNT LZCNT "
               "MULTIBYTENOP PAUSE TSC RDTSCP CPUID RDRAND RDSEED RDPID SYSCALL PREFETCHW CLFSH CLFLUSHOPT "
               "CLWB CLZERO MONITORX FSGSBASE RDPRU",
    "BMI/ADX": "BMI1 BMI2 ADX",
    "x87": "FPU FPU287 FPU387",
    "MMX": "MMX",
    "SSE": "SSE FXSR",
    "SSE2": "SSE2",
    "SSE3-4": "SSE3 SSSE3 SSE4_1 SSE4_2 SSE4A",
    "Crypto": "AES PCLMULQDQ SHA",
    "AVX": "AVX F16C",
    "AVX2": "AVX2",
    "FMA": "FMA",
    "XSAVE": "XSAVE XSAVEOPT XSAVEC XSAVES",
}


def llvm_forms(*files):
    forms, section = set(), None
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if i + 1 < len(lines) and re.fullmatch(r"-{2,}", lines[i + 1]) and re.fullmatch(r"[A-Z0-9_]+", line):
                section = line
            elif line.startswith(".. toctree::"):
                section = None  # the operand-page index at the end isn't instructions
            elif section and (m := re.match(r"    ([a-z][a-z0-9_]+)(\s|$)", line)):
                forms.add((section, m.group(1)))
    # Guard against filing forms under the wrong section heading.
    expect = {"DPP8": "_dpp$", "DPP16": "_dpp$", "SDWA": "_sdwa$", "DS": "^ds_", "SMEM": "^s_", "MIMG": "^image_"}
    bad = [f for f in forms if f[0] in expect and not re.search(expect[f[0]], f[1])]
    assert not bad, bad[:5]
    return forms


def x64_forms():
    import iced_x86 as iced

    def names(module):
        return {v: k for k, v in vars(module).items() if not k.startswith("_") and isinstance(v, int)}

    group_of = {feature: group for group, features in ZEN2.items() for feature in features.split()}
    feature_name, mnemonic_name = names(iced.CpuidFeature), names(iced.Mnemonic)
    forms = {}
    for code in sorted(names(iced.Code)):
        info = iced.OpCodeInfo(code)
        if not info.is_instruction or not info.mode64 or info.is_privileged:
            continue
        instruction = iced.Instruction()
        instruction.code = code
        groups = [group_of.get(feature_name[f]) for f in instruction.cpuid_features()]
        if None not in groups:
            forms.setdefault(mnemonic_name[info.mnemonic].lower(), groups[0])
    return {(group, mnemonic) for mnemonic, group in forms.items()}


def _validate_docs_dir(docs):
    """Validate that docs is a directory containing all required AMDGPUAsmGFX*.rst regular files."""
    if not docs.is_dir():
        raise ValueError(f"LLVM docs path is not a directory: {docs}")
    gfx1013_files = [docs / name for name in REQUIRED_GFX1013_RST]
    rdna2_files = [docs / name for name in REQUIRED_RDNA2_RST]
    for rst in (*gfx1013_files, *rdna2_files):
        if not rst.is_file():
            raise ValueError(f"missing or non-regular LLVM doc file: {rst}")
    return gfx1013_files, rdna2_files


def _validate_output_file(path):
    """Reject symlink, directory, or file-parent output paths before writing."""
    if path.is_symlink():
        raise ValueError(f"output path must not be a symlink: {path}")
    if path.exists() and path.is_dir():
        raise ValueError(f"output path must not be a directory: {path}")
    for ancestor in path.parents:
        if ancestor.exists() and not ancestor.is_dir():
            raise ValueError(f"output parent is not a directory: {ancestor}")


def _validate_out_dir_and_targets(out_dir):
    """Validate --out-dir and each target .tsv path upfront before parsing inputs."""
    if out_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {out_dir}")
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"output directory path is not a directory: {out_dir}")
    for ancestor in out_dir.parents:
        if ancestor.exists() and not ancestor.is_dir():
            raise ValueError(f"output parent is not a directory: {ancestor}")
    for name in OUTPUT_TSV_NAMES:
        _validate_output_file(out_dir / name)


def _write_atomic(path, content):
    """Write via a sibling temp file, preserving POSIX output permissions."""
    _validate_output_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
        if os.name == "posix":
            try:
                mode = stat.S_IMODE(path.stat().st_mode)
            except FileNotFoundError:
                mode = 0o666 & ~_umask()
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _umask():
    # Temporarily restrict permissions rather than relaxing the existing mask.
    current = os.umask(0o777)
    os.umask(current)
    return current


def write(name, source, forms, out_dir=OUT):
    body = "".join(f"{g}\t{f}\n" for g, f in sorted(forms))
    content = (f"# {len(forms)} forms. Source: {source}. Regenerate: tools/form_reference.py\n"
               + body)
    _write_atomic(Path(out_dir) / name, content)
    print(name, len(forms))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("llvm_docs_dir", help="directory containing AMDGPUAsmGFX*.rst docs")
    parser.add_argument("--out-dir", default=None, help="output directory for generated .tsv files")
    args = parser.parse_args(argv)
    try:
        if not args.llvm_docs_dir.strip():
            raise ValueError("llvm_docs_dir must not be empty")
        if args.out_dir is not None and not args.out_dir.strip():
            raise ValueError("--out-dir must not be empty")
        docs = Path(args.llvm_docs_dir)
        out_dir = OUT if args.out_dir is None else Path(args.out_dir)
        gfx1013_files, rdna2_files = _validate_docs_dir(docs)
        _validate_out_dir_and_targets(out_dir)
        gfx1013 = llvm_forms(*gfx1013_files)
        rdna2 = llvm_forms(*rdna2_files)
        x64 = x64_forms()
        write("gfx1013.tsv", LLVM_SOURCE + "AMDGPUAsmGFX10/1011/1013.rst", gfx1013, out_dir=out_dir)
        write("rdna2.tsv", LLVM_SOURCE + "AMDGPUAsmGFX1030.rst", rdna2, out_dir=out_dir)
        write("x64_zen2.tsv", "iced-x86 1.21.0 mnemonics, Zen 2 user-mode CPUID groups", x64, out_dir=out_dir)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"form_reference: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
