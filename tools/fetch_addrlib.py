#!/usr/bin/env python3
"""Fetch the pinned AMD addrlib source used by the optional xg_swizzle_select test.

addrlib is MIT-licensed. Mesa carries it under src/amd/addrlib, with a 64-bit type definition for MSVC
(_WIN32) that the GPUOpen PAL copy lacks. This script makes a shallow, sparse, blob-filtered fetch of that
one directory at the Mesa 25.1.0 commit under out/deps/mesa (gitignored) and prints the path to pass as
XBOX2WINDOWS_ADDRLIB_DIR. Nothing is vendored into this repository.

    python tools/fetch_addrlib.py            # fetch out/deps/mesa at the pinned commit
    python tools/fetch_addrlib.py --dest DIR # another location
"""
import argparse
import pathlib
import subprocess
import sys

URL = "https://gitlab.freedesktop.org/mesa/mesa.git"
COMMIT = "5c142e46f3f6e752ed745fd48912ebb8fad67145"  # tag mesa-25.1.0
SUBDIR = "src/amd/addrlib"


def git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def git_output(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def require_clean(dest):
    dirty = git_output("status", "--porcelain", "--untracked-files=all", "--ignored", cwd=dest)
    if dirty:
        sys.exit(f"{dest} has local changes. Keep them and choose an empty --dest.\n{dirty}")


def main(argv=None) -> int:
    root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=pathlib.Path, default=root / "out" / "deps" / "mesa")
    args = parser.parse_args(argv)
    try:
        dest = args.dest.resolve()
        if dest.exists() and not dest.is_dir():
            raise ValueError(f"{dest} is not a directory")

        if (dest / ".git").is_dir():
            if git_output("remote", "get-url", "origin", cwd=dest) != URL:
                sys.exit(f"{dest} is not the Mesa dependency checkout. Choose an empty --dest.")
            require_clean(dest)
        else:
            dest.mkdir(parents=True, exist_ok=True)
            if sorted(dest.iterdir()):
                sys.exit(f"{dest} is not empty. Choose an empty --dest.")
            git("init", "-q", cwd=dest)
            git("remote", "add", "origin", URL, cwd=dest)
        git("sparse-checkout", "set", "--cone", SUBDIR, cwd=dest)
        git("fetch", "--depth", "1", "--filter=blob:none", "origin", COMMIT, cwd=dest)
        git("checkout", "-q", "--detach", COMMIT, cwd=dest)
        require_clean(dest)
        head = git_output("rev-parse", "HEAD", cwd=dest)
        if head != COMMIT:
            sys.exit(f"checked out {head}, expected {COMMIT}")
        addrlib = dest / SUBDIR
        if not (addrlib / "inc" / "addrinterface.h").is_file():
            sys.exit(f"{addrlib} has no inc/addrinterface.h")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"fetch_addrlib: {exc}\n")
    print(f"addrlib {COMMIT} at {addrlib}")
    print(f"configure with -DXBOX2WINDOWS_ADDRLIB_DIR={addrlib.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
