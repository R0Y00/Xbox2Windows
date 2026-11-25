"""Check the opt-in DXC compile cache with a fake compiler (no DXC needed)."""
import os
import stat
import subprocess
import sys
import tempfile
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

    def compile(self, source, extra=(), cache=True, check=False):
        """Compile into a fresh output path; return (result, object bytes, real call count)."""
        self.n += 1
        src = self.root / "in.hlsl"
        src.write_text(source)
        out = self.root / ("out%d.cso" % self.n)
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

        bad1, none1 = rig.compile("BROKEN")
        bad2, none2 = rig.compile("BROKEN")
        if rig.calls() != 5 or bad1.returncode != 3 or bad2.returncode != 3:
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
    print("dxc_cache_check: ok")


if __name__ == "__main__":
    main()
