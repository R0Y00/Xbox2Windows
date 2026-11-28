#!/usr/bin/env python3
"""Run a console package with the shared real-run settings: pick the console, the host OS and the GPU vendor.

    python -B tools/startup/run_preset.py --package <package root> --console x|s --gpu nvidia|amd [--os win11|win10|linux]

The settings come from tools/startup/run_presets.json, not from the command line: its `base` layer, then
the chosen `os`, `console` and `gpu` layers, each overriding the one before. The launcher
  - finds the executable named in the package's MicrosoftGame.config,
  - stages it with the other PE files of its folder in <run>/stage (hard links where the volume allows, copies
    otherwise), because run.ps1 and run.sh copy the stand-in DLLs next to the executable and the package
    folder must never be written. A file named like a stand-in is always copied: the scripts' copy
    rewrites an existing file in place, which would write through a hard link into the package,
  - gives every run an empty storage folder (<run>/storage),
  - on Windows, maps the package on its console drive letter with `subst` for the run (and removes the mapping),
  - starts tools/startup/run.ps1 (Windows) or tools/startup/run.sh (Linux, under Wine) with the merged switches
    and environment, and
  - prints the allowlisted triage of the run's logs (tools/startup/triage_run.py).
Runs go to out/startup/runs/<time>-<console>-<os>-<gpu>/ (log/, stage/, storage/, settings.json).
The child starts from the caller's environment. X2W_* and WINE* values set there that reach the run are
recorded in settings.json (`inherited_env`); an env value of null removes the variable, also when the caller set it.

--print shows the merged settings and the command without staging or starting anything.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
PRESETS = Path(__file__).resolve().with_name("run_presets.json")
LAYERS = ("os", "console", "gpu")
PE_SUFFIXES = {".exe", ".dll"}
# The DLLs run.ps1 and run.sh copy next to the executable (their `foreach ($dll in ...)` / `for dll in` lists).
STAND_INS = ("xmem", "d3d12_xs", "AcpHal", "XCRDAPI", "chad", "xapu", "xg_xs", "GameInput")

# Launcher option -> (run.ps1 parameter, run.sh flag, kind). kind: "switch" (true/false), "value".
OPTIONS = {
    "seconds": ("-Seconds", "--seconds", "value"),
    "memory_gib": ("-MemoryGiB", "--memory-gib", "value"),
    "continue_asserts": ("-ContinueAsserts", "--continue-asserts", "value"),
    "assert_sites": ("-AssertSites", "--assert-sites", "value"),
    "input_script": ("-InputScript", "--input-script", "value"),
    "capture_every": ("-CaptureEvery", "--capture-every", "value"),
    "shader_cache": ("-ShaderCache", "--shader-cache", "value"),
    "build_dir": ("-BuildDir", "--build-dir", "value"),
    "suspend_others": ("-SuspendOthers", "--suspend-others", "value"),
    "render": ("-Render", "--render", "switch"),
    "fake_user": ("-FakeUser", "--fake-user", "switch"),
    "gpu_mirror": ("-GpuMirror", "--gpu-mirror", "switch"),
    "host_cpu": ("-HostCpu", "--host-cpu", "switch"),
    "translate": ("-Translate", "--translate", "switch"),
    "startup_below_40": ("-StartupBelow40", "--startup-below-40", "switch"),
    "hide_title_window": (None, "--hide-title-window", "switch"),
}
# Launcher-only keys: handled here, never passed through as a script switch.
OWN_KEYS = {"package_drive"}
# Caller variables that are recorded in settings.json when they reach the run.
RECORDED_PREFIXES = ("X2W_", "WINE")


class PresetError(ValueError):
    pass


def load_presets(path: Path = PRESETS) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for layer in ("base",) + LAYERS:
        if layer not in data:
            raise PresetError(f"{path.name}: no '{layer}' section")
    check_layer("base", data["base"])
    for layer in LAYERS:
        if not data[layer]:
            raise PresetError(f"{path.name}: '{layer}' has no choices")
        for name, body in data[layer].items():
            check_layer(f"{layer}.{name}", body)
    return data


def check_layer(where: str, body: dict) -> None:
    unknown = set(body) - {"why", "run", "env"}
    if unknown:
        raise PresetError(f"{where}: unknown key(s) {sorted(unknown)}")
    for key in body.get("run", {}):
        if key not in OPTIONS and key not in OWN_KEYS:
            raise PresetError(f"{where}: unknown run option '{key}'")
    for key, value in body.get("env", {}).items():
        if not key.startswith("X2W_") and key not in ("WINEDLLOVERRIDES", "WINEFSYNC", "WINEDEBUG"):
            raise PresetError(f"{where}: env '{key}' is not an X2W_* switch")
        if value is not None and not isinstance(value, str):
            raise PresetError(f"{where}: env '{key}' must be a string (or null to unset)")


def merge(presets: dict, choice: dict) -> dict:
    """{"run": {...}, "env": {...}, "unset": [...], "layers": [...]}: base, then each chosen layer in LAYERS
    order. `unset` names the env keys whose last value is null; they are removed from the run's environment."""
    run: dict = {}
    env: dict = {}
    layers = ["base"]
    bodies = [presets["base"]]
    for layer in LAYERS:
        name = choice[layer]
        if name not in presets[layer]:
            raise PresetError(f"unknown {layer} '{name}' (choices: {', '.join(presets[layer])})")
        layers.append(f"{layer}.{name}")
        bodies.append(presets[layer][name])
    for body in bodies:
        run.update(body.get("run", {}))
        env.update(body.get("env", {}))
    unset = sorted(k for k, v in env.items() if v is None)  # null unsets an earlier layer's or the caller's value
    env = {k: v for k, v in env.items() if v is not None}
    return {"run": run, "env": env, "unset": unset, "layers": layers}


def child_env(settings: dict, environ) -> tuple[dict, dict]:
    """(the run's environment, the recorded caller values that reach it unchanged)."""
    env = {k: v for k, v in environ.items() if k not in settings["unset"]}
    inherited = {k: v for k, v in sorted(env.items())
                 if k.startswith(RECORDED_PREFIXES) and k not in settings["env"]}
    env.update(settings["env"])
    return env, inherited


def host_os() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        build = sys.getwindowsversion().build
        return "win11" if build >= 22000 else "win10"
    return ""


def package_executable(package: Path) -> Path:
    config = package / "MicrosoftGame.config"
    if not config.is_file():
        raise PresetError("no MicrosoftGame.config at the package root")
    root = ET.parse(config).getroot()
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "Executable" and el.get("Name"):
            exe = package / el.get("Name").replace("\\", "/")
            if not exe.is_file():
                raise PresetError("the executable named in MicrosoftGame.config is not in the package")
            return exe
    raise PresetError("MicrosoftGame.config names no Executable")


def stage(package: Path, exe: Path, stage_dir: Path) -> tuple[Path, int, int]:
    """Link (or copy) the executable's folder's PE files into stage_dir, keeping the relative folder.
    Returns (staged exe, linked, copied). Nothing under the package is written, and nothing written later
    into the stage reaches it: a file named like a stand-in is copied, never linked, because Copy-Item -Force
    and cp -f overwrite an existing destination in place (measured on NTFS: the linked package file changed)."""
    rel = exe.parent.relative_to(package)
    dst = stage_dir / rel
    dst.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    stand_ins = {f"{n.lower()}.dll" for n in STAND_INS}
    for f in sorted(exe.parent.iterdir()):
        if not f.is_file() or f.suffix.lower() not in PE_SUFFIXES:
            continue
        target = dst / f.name
        if target.exists():
            target.unlink()
        if f.name.lower() not in stand_ins:
            try:
                os.link(f, target)
                linked += 1
                continue
            except OSError:
                pass
        shutil.copy2(f, target)
        copied += 1
    return dst / exe.name, linked, copied


def subst_target(letter: str, run=subprocess.run) -> str:
    """The folder `subst` maps on `letter`, or "" when it maps none."""
    out = run(["subst"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        left, sep, target = line.partition("=>")
        if sep and left.strip().upper().startswith(f"{letter.upper()}:"):
            return target.strip()
    return ""


def mount(letter: str, package: Path, run=subprocess.run, exists=os.path.exists) -> bool:
    """Map the package on its console drive letter with `subst`. Returns True when this call made the mapping
    (so the caller removes it afterwards), False when `subst` already maps the letter onto this package.
    run.ps1 -MapDrive is not used: it only rewrites opens that pass through the debugger's file hooks, and
    real-title runs with it stayed on their first screens where the same settings with a subst drive went on
    loading (2026-11-28, 420 s: 11 vs 76 pipelines translated)."""
    letter = letter.rstrip(":").upper()
    current = subst_target(letter, run)
    if current:
        if os.path.normcase(os.path.normpath(current)) == os.path.normcase(os.path.normpath(str(package))):
            return False
        raise PresetError(f"{letter}: is already mapped onto another folder (subst {letter}: /D removes it)")
    if exists(f"{letter}:\\"):
        raise PresetError(f"{letter}: is in use; free the letter, or map it onto the package yourself with subst")
    if run(["subst", f"{letter}:", str(package)], capture_output=True, text=True).returncode:
        raise PresetError(f"subst {letter}: failed")
    return True


def build_command(host: str, settings: dict, exe: Path, package: Path, log: Path, storage: Path) -> list[str]:
    run = settings["run"]
    if host == "linux":
        cmd = ["bash", str(REPO / "tools" / "startup" / "run.sh"), "--exe", str(exe), "--data", str(package),
               "--log", str(log), "--storage", str(storage)]
        col = 1
    else:
        shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell"
        cmd = [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(REPO / "tools" / "startup" / "run.ps1"),
               "-Exe", str(exe), "-DataDir", str(package), "-LogDir", str(log), "-StoragePath", str(storage)]
        col = 0
    for key, value in run.items():
        if key in OWN_KEYS:
            continue
        flag, kind = OPTIONS[key][col], OPTIONS[key][2]
        if flag is None:
            continue  # not a switch of this script
        if kind == "switch":
            if value:
                cmd.append(flag)
        elif value not in (None, ""):
            cmd += [flag, str(value)]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, help="the package root (the folder with MicrosoftGame.config)")
    ap.add_argument("--console", required=True, help="x (Series X) or s (Series S)")
    ap.add_argument("--gpu", required=True, help="nvidia or amd")
    ap.add_argument("--os", default="", help="win11, win10 or linux (default: this host)")
    ap.add_argument("--print", action="store_true", help="show the settings and the command, start nothing")
    # For tests and comparison runs only; everyday settings belong in run_presets.json.
    ap.add_argument("--presets", default=str(PRESETS), help=argparse.SUPPRESS)
    ap.add_argument("--runs", default=str(REPO / "out" / "startup" / "runs"), help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--set", action="append", default=[], metavar="OPTION=VALUE", help=argparse.SUPPRESS)
    ap.add_argument("--seed-storage", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    try:
        presets = load_presets(Path(a.presets))
        host = a.os or host_os()
        if not host:
            raise PresetError(f"no preset for this host ({platform.system()}); pass --os")
        choice = {"os": host, "console": a.console.lower(), "gpu": a.gpu.lower()}
        settings = merge(presets, choice)
        for item in a.set:
            key, _, value = item.partition("=")
            if key not in OPTIONS:
                raise PresetError(f"--set: unknown run option '{key}'")
            settings["run"][key] = (value.lower() in ("1", "true", "yes")) if OPTIONS[key][2] == "switch" else value
        package = Path(a.package).resolve()
        exe = package_executable(package)
    except (PresetError, OSError, ET.ParseError) as e:
        print(f"run_preset: {e}", file=sys.stderr)
        return 2

    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{choice['console']}-{choice['os']}-{choice['gpu']}"
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.runs) / name
    log, storage, stage_dir = run_dir / "log", run_dir / "storage", run_dir / "stage"
    staged_exe = stage_dir / exe.relative_to(package)
    cmd = build_command("linux" if host == "linux" else "windows", settings, staged_exe, package, log, storage)

    env, inherited = child_env(settings, os.environ)
    print(f"run_preset: {' + '.join(settings['layers'])}")
    for k, v in sorted(settings["env"].items()):
        print(f"  env {k}={v}")
    for k, v in inherited.items():
        print(f"  env {k}={v} (from the caller's shell)")
    for k in settings["unset"]:
        print(f"  unset {k}")
    if a.print:
        if host != "linux" and settings["run"].get("package_drive"):
            print(f"  mount: subst {settings['run']['package_drive'].rstrip(':').upper()}: {package}")
        print("  command: " + subprocess.list2cmdline(cmd))
        return 0

    storage.mkdir(parents=True, exist_ok=True)
    if a.seed_storage:
        shutil.copytree(a.seed_storage, storage, dirs_exist_ok=True)  # the seed folder is only read
    if host == "linux" and settings["run"].get("package_drive"):
        print(f"run_preset: run.sh has no drive mapping; dosdevices/{settings['run']['package_drive'].lower()}: "
              "in the prefix must point at the package root")
    staged_exe, linked, copied = stage(package, exe, stage_dir)
    (run_dir / "settings.json").write_text(
        json.dumps({"choice": choice, **settings, "inherited_env": inherited, "command": cmd}, indent=2),
        encoding="utf-8")
    print(f"run_preset: staged {linked} linked + {copied} copied PE file(s); run folder {run_dir}")
    letter = settings["run"].get("package_drive") if host != "linux" else None
    mounted = False
    try:
        if letter:
            mounted = mount(letter, package)
            print(f"run_preset: {letter.rstrip(':').upper()}: -> package root"
                  + ("" if mounted else " (mapping already in place, left as it is)"))
        code = subprocess.call(cmd, cwd=REPO, env=env)
    except PresetError as e:
        print(f"run_preset: {e}", file=sys.stderr)
        return 2
    finally:
        if mounted:
            subprocess.run(["subst", f"{letter.rstrip(':').upper()}:", "/D"], capture_output=True)
    print(f"run_preset: launcher exit {code}")
    py = sys.executable
    subprocess.call([py, "-B", str(REPO / "tools" / "startup" / "triage_run.py"), str(log),
                     "--shader-cache", str(REPO / settings["run"].get("shader_cache", "out/startup/shader_cache"))],
                    cwd=REPO)
    return code


if __name__ == "__main__":
    sys.exit(main())
