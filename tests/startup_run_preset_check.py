#!/usr/bin/env python3
"""Check tools/startup/run_preset.py and tools/startup/run_presets.json without starting a run.

Every console x OS x GPU choice must merge, name only switches that run.ps1 and run.sh really have,
and give the console its ConsoleMode answer. Staging is checked on a synthetic package in a temporary
folder: only the executable's folder's PE files are staged, and nothing in the package changes, also
after the stand-ins are copied over the staged files the way run.ps1 and run.sh copy them.
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
    check(s["env"].get("X2W_HIDE_TITLE_WINDOW") == "1", f"{where}: the title's own window is hidden")
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


def unexplained_gpu_differences(presets):
    """Switches or env values that differ between GPU choices, where a GPU layer that sets the key does not
    name it in its `why`. A measured difference belongs in that layer (AGENTS.md), with its reason."""
    out = []
    for os_name, console in itertools.product(presets["os"], presets["console"]):
        merged = {g: rp.merge(presets, {"os": os_name, "console": console, "gpu": g}) for g in presets["gpu"]}
        for part in ("env", "run"):
            keys = set().union(*(m[part] for m in merged.values()))
            for key in sorted(keys):
                if len({json.dumps(m[part].get(key)) for m in merged.values()}) == 1:
                    continue
                setters = [g for g, body in presets["gpu"].items() if key in body.get(part, {})]
                for g in setters or ["(no gpu layer)"]:
                    if key not in presets["gpu"].get(g, {}).get("why", ""):
                        out.append(f"{console}/{os_name}: gpu.{g} {part} {key} without its reason in why")
    return out


check(unexplained_gpu_differences(presets) == [],
      f"AMD and NVIDIA differ without a recorded reason: {unexplained_gpu_differences(presets)}")
named = copy.deepcopy(presets)
named["gpu"]["amd"]["env"] = {"X2W_SPLIT_INDIRECT_DISPATCH": "1"}  # the amd why names it
check(unexplained_gpu_differences(named) == [], "a GPU difference named in the layer's why is accepted")
unnamed = copy.deepcopy(presets)
unnamed["gpu"]["amd"]["env"] = {"X2W_PRESET_CHECK_ONLY": "1"}
unnamed["gpu"]["nvidia"]["run"] = {"seconds": 60}
check(len(unexplained_gpu_differences(unnamed)) == 2 * len(presets["os"]) * len(presets["console"]),
      "control: a GPU difference without its reason is not caught")

# --- layer rules (controls) ------------------------------------------------------------------
mini = {"base": {"run": {"render": True}, "env": {"X2W_A": "1", "X2W_B": "1"}},
        "os": {"w": {"env": {"X2W_B": None}}}, "console": {"x": {"env": {"X2W_A": "2"}}}, "gpu": {"n": {}}}
m = rp.merge(mini, {"os": "w", "console": "x", "gpu": "n"})
check(m["env"] == {"X2W_A": "2"} and m["unset"] == ["X2W_B"], "a later layer overrides, null unsets")
caller = {"X2W_A": "caller", "X2W_B": "caller", "X2W_WIN_PYTHON": "py.exe", "WINEPREFIX": "/p", "PATH": "/bin"}
env, inherited = rp.child_env(m, caller)
check(env == {"X2W_A": "2", "X2W_WIN_PYTHON": "py.exe", "WINEPREFIX": "/p", "PATH": "/bin"},
      f"null removes a caller's value, layers override it, the rest is kept: {env}")
check(inherited == {"WINEPREFIX": "/p", "X2W_WIN_PYTHON": "py.exe"}, f"caller X2W_/WINE values are recorded: {inherited}")
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
def script_stand_ins():
    """The stand-in names run.ps1 and run.sh copy next to the executable."""
    ps1 = (ROOT / "tools" / "startup" / "run.ps1").read_text(encoding="utf-8")
    sh = (ROOT / "tools" / "startup" / "run.sh").read_text(encoding="utf-8")
    m1 = re.search(r'foreach \(\$dll in ([^)]*)\)', ps1)
    m2 = re.search(r'for dll in ([^;]*);', sh)
    return (tuple(re.findall(r'"([^"]+)"', m1.group(1))) if m1 else (),
            tuple(m2.group(1).split()) if m2 else ())


check(script_stand_ins() == (rp.STAND_INS, rp.STAND_INS), f"STAND_INS matches run.ps1 and run.sh: {script_stand_ins()}")


def copy_stand_ins(folder):
    """What run.ps1 (Copy-Item -Force) and run.sh (cp -f) do: an existing file is rewritten in place."""
    for name in rp.STAND_INS:
        with open(folder / f"{name}.dll", "wb") as out:
            out.write(b"MZ stand-in")


def snapshot(root):
    return sorted((str(f.relative_to(root)), f.read_bytes(), f.stat().st_mtime_ns) for f in root.rglob("*") if f.is_file())


with tempfile.TemporaryDirectory() as tmp:
    package = Path(tmp) / "package"
    (package / "bin").mkdir(parents=True)
    (package / "MicrosoftGame.config").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<Game configVersion="1" xmlns="http://schemas.microsoft.com/x">'
        '<ExecutableList><Executable Name="bin\\t.exe" Id="Game"/></ExecutableList></Game>\n', encoding="utf-8")
    for name, data in [("t.exe", b"MZ exe"), ("a.dll", b"MZ dll"), ("data.bin", b"data"), ("B.DLL", b"MZ up"),
                       ("GameInput.dll", b"MZ package GameInput"), ("XMEM.dll", b"MZ package xmem")]:
        (package / "bin" / name).write_bytes(data)
    (package / "other.dll").write_bytes(b"MZ elsewhere")
    before = snapshot(package)
    exe = rp.package_executable(package)
    check(exe == package / "bin" / "t.exe", "the executable comes from MicrosoftGame.config")
    staged, linked, copied = rp.stage(package, exe, Path(tmp) / "run" / "stage")
    names = sorted(f.name for f in staged.parent.iterdir())
    check(names == ["B.DLL", "GameInput.dll", "XMEM.dll", "a.dll", "t.exe"],
          f"only the executable folder's PE files are staged: {names}")
    check(linked + copied == 5 and copied >= 2 and staged.read_bytes() == b"MZ exe", "staged files match")
    copy_stand_ins(staged.parent)
    check(snapshot(package) == before, "the stand-in copy over the stage leaves the package's same-named files alone")
    rp.stage(package, exe, Path(tmp) / "run" / "stage")  # staging again replaces, never fails
    real_link = rp.os.link

    def no_link(*_a):
        raise OSError(18, "not the same device")  # another volume: hard links fail
    rp.os.link = no_link
    try:
        staged2, linked2, copied2 = rp.stage(package, exe, Path(tmp) / "run2" / "stage")
    finally:
        rp.os.link = real_link
    check((linked2, copied2) == (0, 5) and staged2.read_bytes() == b"MZ exe", "copies when hard links fail")
    check(snapshot(package) == before, "the package is unchanged")
    # Control: with stand-in names linked like the rest, the same copy writes into the package.
    real_stand_ins = rp.STAND_INS
    rp.STAND_INS = ()
    try:
        staged3, linked3, _ = rp.stage(package, exe, Path(tmp) / "run3" / "stage")
    finally:
        rp.STAND_INS = real_stand_ins
    if linked3 == 5:  # only where the temporary folder allows hard links
        copy_stand_ins(staged3.parent)
        check((package / "bin" / "GameInput.dll").read_bytes() == b"MZ stand-in",
              "control: a linked stand-in name is not caught")
        for name, data in [("GameInput.dll", b"MZ package GameInput"), ("XMEM.dll", b"MZ package xmem")]:
            (package / "bin" / name).write_bytes(data)

    out = io.StringIO()
    rp.os.environ["X2W_PRESET_CHECK_CALLER"] = "1"
    try:
        with contextlib.redirect_stdout(out):
            code = rp.main(["--package", str(package), "--console", "s", "--gpu", "amd", "--os", "win11", "--print",
                            "--runs", str(Path(tmp) / "runs")])
    finally:
        del rp.os.environ["X2W_PRESET_CHECK_CALLER"]
    check(code == 0 and "X2W_DEVICE_TYPE=6" in out.getvalue() and "run.ps1" in out.getvalue()
          and f"mount: subst G: {package}" in out.getvalue()
          and "X2W_PRESET_CHECK_CALLER=1 (from the caller's shell)" in out.getvalue(), "--print output")
    check(not (Path(tmp) / "runs").exists(), "--print creates nothing")

    # A run with the child start replaced: the environment it gets and what settings.json records.
    mini_path = Path(tmp) / "mini.json"
    mini_path.write_text(json.dumps(mini), encoding="utf-8")
    calls = []
    real_call = rp.subprocess.call
    rp.subprocess.call = lambda argv, **kw: calls.append((argv, kw.get("env"))) or 0
    rp.os.environ["X2W_B"] = "caller"
    rp.os.environ["X2W_PRESET_CHECK_CALLER"] = "1"
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = rp.main(["--package", str(package), "--console", "x", "--gpu", "n", "--os", "w",
                            "--presets", str(mini_path), "--run-dir", str(Path(tmp) / "run4")])
    finally:
        rp.subprocess.call = real_call
        del rp.os.environ["X2W_B"], rp.os.environ["X2W_PRESET_CHECK_CALLER"]
    recorded = json.loads((Path(tmp) / "run4" / "settings.json").read_text(encoding="utf-8"))
    child = calls[0][1] if calls else {}
    check(code == 0 and "X2W_B" not in child and child.get("X2W_A") == "2"
          and child.get("X2W_PRESET_CHECK_CALLER") == "1", "the run's environment: null removes the caller's value")
    check(recorded.get("inherited_env", {}).get("X2W_PRESET_CHECK_CALLER") == "1" and recorded.get("unset") == ["X2W_B"]
          and "X2W_B" not in recorded["inherited_env"], "settings.json records the caller's values and the unset keys")

    (package / "MicrosoftGame.config").write_text("<Game/>", encoding="utf-8")
    refused(lambda: rp.package_executable(package), "a config without an Executable")
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        code = rp.main(["--package", str(package), "--console", "x", "--gpu", "nvidia", "--os", "win11", "--print"])
    check(code == 2 and "names no Executable" in err.getvalue(), "main reports a package without an executable")

print(f"{len(failures)} failure(s)" if failures else f"OK: {len(combos)} preset choices, staging and controls")
sys.exit(1 if failures else 0)
