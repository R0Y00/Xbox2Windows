#!/usr/bin/env python3
"""Check tools/startup/run_preset.py and tools/startup/run_presets.json without starting a run.

Every console x OS x GPU choice must merge, name only switches that run.ps1 and run.sh really have,
and give the console its ConsoleMode answer. Staging is checked on a synthetic package in a temporary
folder: only the executable's folder's PE files are staged, and nothing in the package changes.
Negative controls: a misspelt switch, an unknown key, a non-string value, an unknown choice, a drive
letter that is taken and a package without an executable must each be refused.
"""
import contextlib
import copy
import io
import itertools
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools" / "startup"))
import run_preset as rp  # noqa: E402

failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print(f"FAIL: {what}")


def refused(fn, what):
    try:
        fn()
    except rp.PresetError:
        return
    check(False, f"{what} was accepted")


def missing_flags(options):
    """Switch names in `options` that the scripts do not define."""
    ps1 = (ROOT / "tools" / "startup" / "run.ps1").read_text(encoding="utf-8")
    sh = (ROOT / "tools" / "startup" / "run.sh").read_text(encoding="utf-8")
    ps1_params = {m.lower() for m in re.findall(r"\]\s*\$(\w+)", ps1.split(")\n$ErrorActionPreference")[0])}
    out = []
    for key, (ps1_flag, sh_flag, _kind) in options.items():
        if ps1_flag and ps1_flag[1:].lower() not in ps1_params:
            out.append(f"{key}: run.ps1 has no {ps1_flag}")
        if sh_flag and not re.search(rf"(^|[\s|]){re.escape(sh_flag)}([)\s|]|$)", sh, re.M):
            out.append(f"{key}: run.sh has no {sh_flag}")
    return out


# --- the shipped presets ---------------------------------------------------------------------
presets = rp.load_presets()
check(missing_flags(rp.OPTIONS) == [], f"launcher switches missing from the scripts: {missing_flags(rp.OPTIONS)}")
bad = copy.deepcopy(rp.OPTIONS)
bad["render"] = ("-Rendering", "--rendering", "switch")
check(len(missing_flags(bad)) == 2, "control: a misspelt switch is not caught")

pkg = Path("/pkg") if sys.platform != "win32" else Path("C:/pkg")
combos = list(itertools.product(presets["os"], presets["console"], presets["gpu"]))
check(set(presets["console"]) == {"x", "s"} and set(presets["gpu"]) == {"nvidia", "amd"}
      and set(presets["os"]) == {"win11", "win10", "linux"}, "the choices are x/s, nvidia/amd, win11/win10/linux")
for os_name, console, gpu in combos:
    s = rp.merge(presets, {"os": os_name, "console": console, "gpu": gpu})
    where = f"{console}/{os_name}/{gpu}"
    check(s["env"].get("X2W_DEVICE_TYPE") == {"x": "7", "s": "6"}[console], f"{where}: X2W_DEVICE_TYPE")
    check(s["layers"] == ["base", f"os.{os_name}", f"console.{console}", f"gpu.{gpu}"], f"{where}: layer order")
    check(all(isinstance(v, str) for v in s["env"].values()), f"{where}: env values are strings")
    host = "linux" if os_name == "linux" else "windows"
    cmd = rp.build_command(host, s, pkg / "stage" / "t.exe", pkg, pkg / "log", pkg / "storage")
    if host == "linux":
        check(cmd[1].endswith("run.sh") and "--render" in cmd, f"{where}: run.sh command")
        check(s["env"].get("X2W_XMEM_SHARED_VIEWS") == "1", f"{where}: shared views on Linux")
    else:
        check(cmd[5].endswith("run.ps1") and "-Render" in cmd, f"{where}: run.ps1 command")
        check(s["run"].get("package_drive") == "G" and "-MapDrive" not in cmd, f"{where}: package on G through subst, not -MapDrive")
        check(cmd[cmd.index("-SuspendOthers") + 1] == "0", f"{where}: hook thread holding off")
        check("X2W_XMEM_LAZY_COMMIT" not in s["env"], f"{where}: lazy commit stays off on Windows")
    check(cmd[cmd.index("--storage" if host == "linux" else "-StoragePath") + 1] == str(pkg / "storage"),
          f"{where}: per-run storage")
for os_name in presets["os"]:
    envs = [rp.merge(presets, {"os": os_name, "console": "x", "gpu": g})["env"] for g in presets["gpu"]]
    check(all(e == envs[0] for e in envs), f"{os_name}: AMD and NVIDIA differ without a recorded reason")

# --- layer rules (controls) ------------------------------------------------------------------
mini = {"base": {"run": {"render": True}, "env": {"X2W_A": "1", "X2W_B": "1"}},
        "os": {"w": {"env": {"X2W_B": None}}}, "console": {"x": {"env": {"X2W_A": "2"}}}, "gpu": {"n": {}}}
m = rp.merge(mini, {"os": "w", "console": "x", "gpu": "n"})
check(m["env"] == {"X2W_A": "2"}, "a later layer overrides, null unsets")
off = {"run": {"render": False, "translate": True, "seconds": 5}, "env": {}}
cmd = rp.build_command("windows", off, pkg / "t.exe", pkg, pkg / "log", pkg / "storage")
check("-Render" not in cmd and "-Translate" in cmd and cmd[cmd.index("-Seconds") + 1] == "5",
      "a switch set to false is left out, values are passed")
refused(lambda: rp.merge(mini, {"os": "w", "console": "z", "gpu": "n"}), "an unknown console")
with tempfile.TemporaryDirectory() as tmp:
    for label, change in [("an unknown layer key", lambda d: d["base"].update(extra=1)),
                          ("an unknown run option", lambda d: d["base"]["run"].update(renderx=True)),
                          ("a non-X2W env name", lambda d: d["base"]["env"].update(PATH="x")),
                          ("a number as an env value", lambda d: d["base"]["env"].update(X2W_C=1)),
                          ("an empty choice list", lambda d: d.update(gpu={}))]:
        d = copy.deepcopy(mini)
        change(d)
        p = Path(tmp) / "p.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        refused(lambda: rp.load_presets(p), label)

# --- drive mapping with a fake subst (nothing is mapped for real) ----------------------------
class FakeSubst:
    def __init__(self, listing="", rc=0):
        self.listing, self.rc, self.calls = listing, rc, []

    def __call__(self, argv, **_kw):
        self.calls.append(argv)
        return type("R", (), {"stdout": self.listing if argv == ["subst"] else "", "returncode": self.rc})()


pkg_dir = r"D:\packages\one"
f = FakeSubst()
check(rp.mount("G", Path(pkg_dir), run=f, exists=lambda _p: False) is True
      and f.calls[-1] == ["subst", "G:", str(Path(pkg_dir))], "a free letter is mapped with subst")
listing = "G:\\: => " + pkg_dir + "\n"  # what `subst` prints for a mapped letter
f = FakeSubst(listing)
check(rp.mount("g:", Path(pkg_dir), run=f, exists=lambda _p: True) is False and len(f.calls) == 1,
      "a subst of the same package is reused and not removed")
check(rp.subst_target("H", FakeSubst(listing)) == "", "another letter's mapping is not this one")
refused(lambda: rp.mount("G", Path(pkg_dir), run=FakeSubst("G:\\: => D:\\other\n"), exists=lambda _p: True),
        "a letter mapped onto another folder")
refused(lambda: rp.mount("G", Path(pkg_dir), run=FakeSubst(), exists=lambda _p: True), "a letter in use by a volume")
refused(lambda: rp.mount("G", Path(pkg_dir), run=FakeSubst(rc=1), exists=lambda _p: False), "a failed subst")

# --- staging on a synthetic package ----------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    package = Path(tmp) / "package"
    (package / "bin").mkdir(parents=True)
    (package / "MicrosoftGame.config").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<Game configVersion="1" xmlns="http://schemas.microsoft.com/x">'
        '<ExecutableList><Executable Name="bin\\t.exe" Id="Game"/></ExecutableList></Game>\n', encoding="utf-8")
    for name, data in [("t.exe", b"MZ exe"), ("a.dll", b"MZ dll"), ("data.bin", b"data"), ("B.DLL", b"MZ up")]:
        (package / "bin" / name).write_bytes(data)
    (package / "other.dll").write_bytes(b"MZ elsewhere")
    before = sorted((str(f.relative_to(package)), f.read_bytes(), f.stat().st_mtime_ns) for f in package.rglob("*") if f.is_file())
    exe = rp.package_executable(package)
    check(exe == package / "bin" / "t.exe", "the executable comes from MicrosoftGame.config")
    staged, linked, copied = rp.stage(package, exe, Path(tmp) / "run" / "stage")
    names = sorted(f.name for f in staged.parent.iterdir())
    check(names == ["B.DLL", "a.dll", "t.exe"], f"only the executable folder's PE files are staged: {names}")
    check(linked + copied == 3 and staged.read_bytes() == b"MZ exe", "staged files match")
    rp.stage(package, exe, Path(tmp) / "run" / "stage")  # staging again replaces, never fails
    real_link = rp.os.link

    def no_link(*_a):
        raise OSError(18, "not the same device")  # another volume: hard links fail
    rp.os.link = no_link
    try:
        staged2, linked2, copied2 = rp.stage(package, exe, Path(tmp) / "run2" / "stage")
    finally:
        rp.os.link = real_link
    check((linked2, copied2) == (0, 3) and staged2.read_bytes() == b"MZ exe", "copies when hard links fail")
    after = sorted((str(f.relative_to(package)), f.read_bytes(), f.stat().st_mtime_ns) for f in package.rglob("*") if f.is_file())
    check(before == after, "the package is unchanged")

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = rp.main(["--package", str(package), "--console", "s", "--gpu", "amd", "--os", "win11", "--print",
                        "--runs", str(Path(tmp) / "runs")])
    check(code == 0 and "X2W_DEVICE_TYPE=6" in out.getvalue() and "run.ps1" in out.getvalue()
          and f"mount: subst G: {package}" in out.getvalue(), "--print output")
    check(not (Path(tmp) / "runs").exists(), "--print creates nothing")

    (package / "MicrosoftGame.config").write_text("<Game/>", encoding="utf-8")
    refused(lambda: rp.package_executable(package), "a config without an Executable")
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code = rp.main(["--package", str(package), "--console", "x", "--gpu", "nvidia", "--os", "win11", "--print"])
    check(code == 2 and "names no Executable" in err.getvalue(), "main reports a package without an executable")

print(f"{len(failures)} failure(s)" if failures else f"OK: {len(combos)} preset choices, staging and controls")
sys.exit(1 if failures else 0)
