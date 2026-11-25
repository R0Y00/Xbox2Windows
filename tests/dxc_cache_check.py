"""Check the opt-in DXC compile cache with a fake compiler (no DXC needed)."""
import json
import os
import stat
import subprocess
import sys
import tempfile
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dxc_cache  # noqa: E402

FAKE = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
counter = args[args.index("-Fo") + 1] + ".calls"
open(counter, "a").write("x")
src = open(args[-1], "rb").read()
if b"BROKEN" in src:
    sys.stderr.write("error in " + args[-1] + "\\n")
    sys.exit(3)
open(args[args.index("-Fo") + 1], "wb").write(b"OBJ:" + src + " ".join(args[:-1]).encode().split(b"-Fo")[0])
sys.stdout.write("compiled " + args[-1] + "\\n")
"""


def fail(message):
    raise SystemExit("dxc_cache_check: " + message)


class Rig:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = str(self.root / "cache")
        self.compiler = str(self.root / "fakedxc")
        Path(self.compiler).write_text(FAKE)
        os.chmod(self.compiler, os.stat(self.compiler).st_mode | stat.S_IEXEC)
        self.n = 0

    def compile(self, source, extra=(), cache=True, check=False, out=None):
        """Compile into a fresh (or given) output path; return (result, object bytes)."""
        self.n += 1
        src = self.root / "in.hlsl"
        src.write_text(source)
        out = out or self.root / ("out%d.cso" % self.n)
        if cache:
            os.environ[dxc_cache.ENV_VAR] = self.cache
        else:
            os.environ.pop(dxc_cache.ENV_VAR, None)
        result = dxc_cache.run_dxc(self.compiler, ["-T", "cs_6_0", *extra, "-Fo", str(out), str(src)],
                                   check=check, capture_output=True, text=True)
        obj = out.read_bytes() if out.exists() else None
        return result, obj

    def calls(self):
        return sum(len(p.read_text()) for p in self.root.glob("*.calls"))


def main():
    with tempfile.TemporaryDirectory() as root:
        rig = Rig(root)
        first, obj1 = rig.compile("void main(){}")
        second, obj2 = rig.compile("void main(){}")
        if rig.calls() != 1:
            fail("identical compile ran %d real calls, expected 1" % rig.calls())
        if (obj1, first.returncode) != (obj2, second.returncode) or obj1 is None:
            fail("cache hit differs from the real result")
        if first.stdout.replace("in.hlsl", "") != second.stdout.replace("in.hlsl", "") or "compiled" not in second.stdout:
            fail("stdout not replayed")

        rig.compile("void main(){ int x; }")
        if rig.calls() != 2:
            fail("changed source reused a stale entry")
        rig.compile("void main(){}", extra=["-Gis"])
        if rig.calls() != 3:
            fail("changed argument reused a stale entry")
        os.utime(rig.compiler, ns=(1, 1))
        rig.compile("void main(){}")
        if rig.calls() != 4:
            fail("changed compiler reused a stale entry")
        library = Path(rig.compiler).with_name("dxcompiler.dll")
        library.write_bytes(b"v1")
        rig.compile("void main(){}")
        before = rig.calls()
        rig.compile("void main(){}")
        if rig.calls() != before:
            fail("unchanged compiler library missed the cache")
        os.utime(library, ns=(1, 1))
        rig.compile("void main(){}")
        if rig.calls() != before + 1:
            fail("changed compiler library reused a stale entry")
        library.write_bytes(b"version two")
        rig.compile("void main(){}")
        if rig.calls() != before + 2:
            fail("resized compiler library reused a stale entry")

        bad1, none1 = rig.compile("BROKEN")
        bad2, none2 = rig.compile("BROKEN")
        if rig.calls() != before + 3 or bad1.returncode != 3 or bad2.returncode != 3:
            fail("failing compile not replayed with the same return code")
        if bad1.stderr != bad2.stderr or "in.hlsl" not in bad2.stderr:
            fail("failing compile stderr differs on replay")
        try:
            rig.compile("BROKEN", check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode != 3:
                fail("replayed failure raised the wrong return code")
        else:
            fail("replayed failure did not raise with check=True")

        stale = Path(root) / "stale.cso"
        _, seeded = rig.compile("void main(){ /*seed*/ }", out=stale)
        if seeded is None:
            fail("seed compile wrote no output")
        rig.compile("BROKEN stale", out=stale)
        absent = Path(root) / "absent.cso"
        replay, nothing = rig.compile("BROKEN stale", out=absent)
        if replay.returncode != 3 or nothing is not None or absent.exists():
            fail("failed compile replayed an output file the compiler never wrote")

        def seed_legacy_failure(source, version):
            """Write the failure record an older build stored: returncode 3 with a non-empty files map."""
            cache_dir = Path(rig.cache)
            known = set(cache_dir.glob("*.json"))
            with mock.patch.object(dxc_cache, "KEY_VERSION", version):
                rig.compile(source)
            (entry,) = set(cache_dir.glob("*.json")) - known
            record = json.loads(entry.read_text())
            record["files"] = {"-Fo": b"OLD-SHADER".hex()}
            entry.write_text(json.dumps(record))

        seed_legacy_failure("BROKEN legacy same key", dxc_cache.KEY_VERSION)
        old_path = Path(root) / "legacy_same.cso"
        replay, nothing = rig.compile("BROKEN legacy same key", out=old_path)
        if replay.returncode != 3 or nothing is not None or old_path.exists():
            fail("cached failure with files restored an output file")

        seed_legacy_failure("BROKEN legacy v1", "1")
        before = rig.calls()
        old_path = Path(root) / "legacy_v1.cso"
        replay, nothing = rig.compile("BROKEN legacy v1", out=old_path)
        if rig.calls() != before + 1 or replay.returncode != 3 or nothing is not None or old_path.exists():
            fail("a record from an older key version was reused")

        before = rig.calls()
        rig.compile("void main(){}", cache=False)
        rig.compile("void main(){}", cache=False)
        if rig.calls() != before + 2:
            fail("cache off must make every call real")

        before = rig.calls()
        rig.compile('#include "x.hlsli"\nvoid main(){}')
        rig.compile('#include "x.hlsli"\nvoid main(){}')
        rig.compile("void main(){}", extra=["-I", root])
        rig.compile("void main(){}", extra=["-I", root])
        if rig.calls() != before + 4:
            fail("#include or -I compile was cached")

        src = Path(root) / "stub.hlsl"
        src.write_text("void main(){}")
        os.environ[dxc_cache.ENV_VAR] = str(Path(root) / "cache-stub")
        sentinel = subprocess.CompletedProcess(["stub-dxc"], 0, "patched", "")
        with mock.patch.object(subprocess, "run", return_value=sentinel) as patched:
            got = dxc_cache.run_dxc("stub-dxc", ["-T", "cs_6_0", "-Fo", str(Path(root) / "stub.cso"), str(src)],
                                    check=True)
        if got is not sentinel or patched.call_count != 1:
            fail("a compiler that is not a file must run uncached through subprocess.run")
        os.environ.pop(dxc_cache.ENV_VAR, None)
    print("dxc_cache_check: ok")


if __name__ == "__main__":
    main()
