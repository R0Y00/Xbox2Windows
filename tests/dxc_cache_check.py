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
        if os.name == "nt":
            # Windows cannot execute a shebang script: the compiler is a .cmd wrapper
            # that runs the fake script with this interpreter.
            script = self.root / "fakedxc_impl.py"
            script.write_text(FAKE)
            self.compiler = str(self.root / "fakedxc.cmd")
            Path(self.compiler).write_text('@"%s" "%s" %%*\n' % (sys.executable, script))
        else:
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

        # A hit must not create a directory where the real compiler would refuse the output.
        missing_output = rig.root / "missing" / "output.cso"
        rejected, obj = rig.compile("void main(){}", out=missing_output)
        if rejected.returncode == 0 or obj is not None or missing_output.parent.exists():
            fail("cache hit bypassed a missing output directory")

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

        before = rig.calls()
        bad1, _ = rig.compile("BROKEN")
        bad2, _ = rig.compile("BROKEN")
        if rig.calls() != before + 2 or bad1.returncode != 3 or bad2.returncode != 3:
            fail("a failing compile must run for real every time with its own return code")
        if "in.hlsl" not in bad2.stderr:
            fail("failing compile stderr lost")
        try:
            rig.compile("BROKEN", check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode != 3:
                fail("failure raised the wrong return code")
        else:
            fail("failure did not raise with check=True")

        missing = Path(root) / "no-such-dir" / "x.cso"
        rig.compile("void main(){ /*late dir*/ }", out=missing)
        (Path(root) / "no-such-dir").mkdir()
        _, late = rig.compile("void main(){ /*late dir*/ }", out=missing)
        if late is None:
            fail("a failure caused by a missing output directory poisoned the cache")

        cache_dir = Path(rig.cache)
        known = set(cache_dir.glob("*.json"))
        rig.compile("BROKEN seeded")
        if set(cache_dir.glob("*.json")) != known:
            fail("a failure was written to the cache")
        rig.compile("void main(){ /*seeded*/ }")
        (entry,) = set(cache_dir.glob("*.json")) - known
        record = json.loads(entry.read_text())
        record["returncode"] = 3
        entry.write_text(json.dumps(record))
        before = rig.calls()
        replay, _ = rig.compile("void main(){ /*seeded*/ }")
        if replay.returncode != 0 or rig.calls() != before + 1:
            fail("a stored failure record was replayed")

        wine = Path(root) / "winedxc"
        wine.write_text('#!/bin/bash\nexec wine "$DXC" "$@"\n' + FAKE.split("\n", 1)[1].replace("\n", "\n# "))
        os.chmod(wine, os.stat(wine).st_mode | stat.S_IEXEC)
        if dxc_cache._is_wrapper(str(wine)) is not True or dxc_cache._is_wrapper(rig.compiler) is not False:
            fail("a Wine launcher must bypass the cache, a plain compiler must not")

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
