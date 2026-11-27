#!/usr/bin/env python3
"""Run several `emit` variants of one CPU fixture check script in a single Python process.

usage: emit_variants.py <script.py> <image> <variant>=<output> [<variant>=<output> ...]

Each pair does what `python <script.py> emit <image> <output> <variant>` does, in the order given, so the
interpreter, the lifter modules and the PE reader load once instead of once per variant. The script is
loaded as a module (its `__main__` block does not run) and its `emit(image, output, control)` is called.
"""
import argparse
import importlib.util
from pathlib import Path
import sys


def _same_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def load(script):
    path = Path(script).resolve()
    parent_str = str(path.parent)
    if not sys.path or sys.path[0] != parent_str:
        sys.path.insert(0, parent_str)  # as when the script itself is the program
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(path.stem, None)
        raise
    return module


def _validate_regular_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.is_dir() or not path.is_file():
        raise ValueError(f"{label} path must be a regular file: {path}")


def _parse_and_validate_jobs(
    pairs: list[str],
    script_path: Path,
    image_path: Path,
) -> list[tuple[str, Path]]:
    seen_variants: set[str] = set()
    jobs: list[tuple[str, Path]] = []
    for pair in pairs:
        variant, separator, raw_output = pair.partition("=")
        if not separator or not variant or not raw_output:
            raise ValueError(f"expected <variant>=<output>, got {pair!r}")
        if variant in seen_variants:
            raise ValueError(f"duplicate variant {variant!r}")
        seen_variants.add(variant)
        output = Path(raw_output)
        if output.exists() and output.is_dir():
            raise ValueError(f"output path for {variant!r} is an existing directory: {output}")
        if _same_path(output, script_path):
            raise ValueError(f"output path for {variant!r} collides with script: {output}")
        if _same_path(output, image_path):
            raise ValueError(f"output path for {variant!r} collides with image: {output}")
        output_resolved = output.resolve()
        for prev_variant, prev_output in jobs:
            if _same_path(output, prev_output):
                raise ValueError(
                    f"output path for {variant!r} collides with {prev_variant!r}: {output}"
                )
            prev_resolved = prev_output.resolve()
            if output_resolved in prev_resolved.parents or prev_resolved in output_resolved.parents:
                raise ValueError(
                    f"output paths for {variant!r} and {prev_variant!r} cannot contain one another"
                )
        jobs.append((variant, output))
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="CPU fixture script exposing emit(image, output, control)")
    parser.add_argument("image", type=Path, help="PE image passed to emit()")
    parser.add_argument(
        "pairs",
        nargs="+",
        metavar="variant=output",
        help="variant=output pairs to emit",
    )
    args = parser.parse_args(argv)

    try:
        _validate_regular_file(args.script, "script")
        _validate_regular_file(args.image, "image")
        jobs = _parse_and_validate_jobs(args.pairs, args.script, args.image)
        module = load(args.script)
        emit = getattr(module, "emit", None)
        if not callable(emit):
            raise ValueError(
                f"script {args.script} does not define callable emit(image, output, control)"
            )
        for _, output in jobs:
            output.parent.mkdir(parents=True, exist_ok=True)
        for variant, output in jobs:
            emit(args.image, output, variant)
    except (OSError, SyntaxError, ImportError, ValueError) as exc:
        print(f"emit_variants: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
