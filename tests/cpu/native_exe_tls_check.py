"""Compare an EXE's own TLS template and TLS callbacks under a Windows launch and under pe_map_run.

cpu_exe_tls is a CRT-free EXE whose image holds an initialized TLS template that spans a page boundary
(the int and 8 KiB array of tests/cpu/tls_value.cpp) and two TLS callbacks in table order. It prints one
ordered startup log: the callbacks on process attach, the entry point, the main thread's own
observations, then thread attach and detach of two workers that are alive at the same time, and the
exit code (tests/cpu/exe_tls.cpp describes the format). The test launches the EXE with Windows and, in a
child process, maps it with pe_map_image and starts it with pe_map_run. The log and the exit code must
be identical, and must match the expected log, which this script derives from the fixture's rules.

Two negative controls patch copies of the built EXE with the #883 helpers: a swapped callback table
and a template zeroed from its first page boundary. Each must be rejected under both loaders with a
diagnostic that names the callback order or the template.

Under pe_map, thread notifications become eligible after the process-attach callbacks have returned and
before the entry point runs. The child binds only kernel32.dll, to the host's kernel32, and gives the
callbacks four counters (exe_tls_mirror), one per reason, that it reads after release. Controls:
- cpu_exe_tls_early starts and joins a probe thread from its first process-attach callback. The probe
  must record its own observation and receive no EXE notification; a mapper that made notifications
  eligible before the process-attach callbacks finished fails here. Windows holds such a thread until
  process initialization ends, so this variant runs only under pe_map.
- An EXE that is mapped but never started, and one whose pe_map_run is refused (a null exit code), get no
  callback while two host threads attach and detach, and none at release.
- A started EXE's release sends no process detach: the counters show 2 process attaches, 4 thread
  attaches, 4 thread detaches and 0 process detaches.
"""

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from native_map_check import Mapper
from native_tls_value_check import TLS_VALUE, layout, swap_callbacks, written, zero_second_page

TIMEOUT = 30  # per process; longer than the 10 s waits inside the EXE, so its failure code is observed
MAIN_WRITE, A_WRITE, B_WRITE = 20, 50, 60
PROCESS_ATTACH, THREAD_ATTACH, THREAD_DETACH = "1", "2", "3"
# Callback calls by reason (process detach, process attach, thread attach, thread detach) for a started
# EXE: two callbacks on process attach and on each worker's attach and detach, and nothing at release.
STARTED_COUNTS = [0, 2, 4, 4]


def expected_log(early=False):
    """The log the fixture must print when every block starts from the template, in its fixed order.

    early: cpu_exe_tls_early, whose probe thread p records its own observation inside the first
    process-attach callback and receives no notification."""
    fresh, main, a, b = (f"{v:x}" for v in (TLS_VALUE, written(MAIN_WRITE), written(A_WRITE), written(B_WRITE)))

    def callbacks(thread, reason, value):
        return [f"{thread}{reason}1={value}", f"{thread}{reason}3={value}"]

    attach = callbacks("m", PROCESS_ATTACH, fresh)
    if early:
        attach.insert(1, f"pf={fresh}")
    events = [*attach, "E", f"mf={fresh}", f"mw={main}",
              *callbacks("a", THREAD_ATTACH, fresh), f"af={fresh}", f"aw={a}",
              *callbacks("b", THREAD_ATTACH, fresh), f"bf={fresh}", f"bw={b}",
              f"al={a}", *callbacks("a", THREAD_DETACH, a),
              f"bl={b}", *callbacks("b", THREAD_DETACH, b),
              f"ml={main}", f"X={main}"]
    return " ".join(events)


def problems(result, early=False):
    """Every way one launch departs from the expected log; empty when it matches.

    Each problem starts with "template" when a thread saw something other than its own copy of the
    template, "callback order" when the two TLS callbacks ran against the table order, or "notifications"
    when a thread was notified that should not have been, or the reverse."""
    if result["log"] is None:
        return [f"no startup log (exit {result['exit']:#x}): {result}"]
    expected = expected_log(early).split()
    events = result["log"].split()
    found = []
    # Callbacks: per thread and reason, the caller order and the value each one saw.
    calls, wanted = {}, {}
    for event, target in ((e, calls) for e in events):
        if len(event) > 3 and event[3] == "=" and event[1] in "0123":
            target.setdefault(event[:2], []).append((event[2], event[4:]))
    for event in expected:
        if len(event) > 3 and event[3] == "=" and event[1] in "0123":
            wanted.setdefault(event[:2], []).append((event[2], event[4:]))
    if set(calls) != set(wanted):
        found.append(f"notifications for {sorted(calls)}, expected {sorted(wanted)}")
    for key, seen in calls.items():
        callers = [caller for caller, _ in seen]
        if sorted(callers) != ["1", "3"]:
            found.append(f"{key}: callers {callers}, expected 1 and 3 once each")
        elif callers != ["1", "3"]:
            found.append(f"callback order: {key} ran callers {callers}; the table has 1 before 3")
        values = [value for _, value in seen]
        expected_values = [value for _, value in wanted.get(key, [])]
        if key in wanted and values != expected_values:
            found.append(f"template: callbacks on {key} observed {values}, expected {expected_values}")
    # The threads' own observations and the exit code.
    own = {e.split("=")[0]: e.split("=")[1] for e in events if "=" in e and e[1] in "fwlX="}
    for event in expected:
        if "=" in event and event[1] in "fwl=":
            name, value = event.split("=")
            if own.get(name) != value:
                found.append(f"template: {name} observed {own.get(name)}, expected {value}")
    if result["exit"] != written(MAIN_WRITE):
        found.append(f"template: exit code {result['exit']:#x}, expected {written(MAIN_WRITE):#x}")
    if not found and events != expected:
        found.append(f"log order: {result['log']!r}, expected {expected_log(early)!r}")
    return found


def observed(done):
    lines = done.stdout.decode().splitlines()
    log = next((line[4:].strip() for line in lines if line.startswith("LOG:")), None)
    return {"log": log, "exit": done.returncode & 0xffffffff, "stderr": done.stderr.decode()}


def native(exe):
    with TemporaryDirectory() as temp:
        copy = Path(temp) / "cpu_exe_tls.exe"
        shutil.copy(exe, copy)
        return observed(subprocess.run([str(copy)], capture_output=True, timeout=TIMEOUT))


def mapped(mapper, exe, mode="start"):
    """pe_map in a child process; its stdout belongs to the guest, the child's result goes to a file."""
    with TemporaryDirectory() as temp:
        result_file = Path(temp) / "result.json"
        done = subprocess.run([sys.executable, "-B", __file__, "--run", str(mapper), str(exe), str(result_file),
                               mode], capture_output=True, timeout=TIMEOUT)
        if not result_file.exists():
            return {**observed(done), "failure": f"child exited {done.returncode:#x} without a result"}
        child = json.loads(result_file.read_text())
        result = {**observed(done), **child}
        result["exit"] = child["exit"]  # the guest's exit code, returned by pe_map_run
        return result


def host_threads(kernel, count=2):
    """Start and join threads that only run kernel32!Sleep(0), so they attach and detach in this process."""
    sleep = ctypes.cast(kernel.Sleep, ctypes.c_void_p).value
    for _ in range(count):
        handle = kernel.CreateThread(None, 0, sleep, None, 0, None)
        if not handle or kernel.WaitForSingleObject(handle, 10000) != 0:
            raise AssertionError("host thread did not finish")
        kernel.CloseHandle(handle)


def run(mapper_path, exe, result_file, mode):
    """Child: map the EXE with kernel32.dll bound to the host's kernel32 and nothing else, then start it
    (mode "start"), leave it mapped but unstarted ("mapped-only") or have pe_map_run refuse it ("refused")
    while host threads attach and detach. The callbacks count their calls by reason in counters that this
    process owns, read after release."""
    mapper = Mapper(mapper_path)
    start = ctypes.CDLL(str(mapper_path)).pe_map_run
    start.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    start.restype = ctypes.c_int
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateThread.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    # Any other DLL name the image imports stays unresolved, so the map fails instead of loading it.
    base, status, detail, requests = mapper(Path(exe), {"kernel32.dll": kernel})
    result = {"failure": None, "started": None, "exit": 0xdeadbeef,
              "imported": sorted({dll.casefold() for dll, _ in requests})}
    if not base:
        result["failure"] = [status, detail]
        Path(result_file).write_text(json.dumps(result))
        return
    counts = (ctypes.c_long * 4)()
    ctypes.CFUNCTYPE(None, ctypes.c_void_p)(mapper.export(base, b"exe_tls_mirror"))(ctypes.addressof(counts))
    if mode == "start":
        exit_code = ctypes.c_uint32(0xdeadbeef)
        result["started"] = start(base, ctypes.byref(exit_code))
        result["exit"] = exit_code.value
    elif mode == "refused":
        result["started"] = start(base, None)  # a null exit code: pe_map_run runs nothing and returns 0
        host_threads(kernel)
    else:
        host_threads(kernel)
    sys.stdout.flush()
    result["released"] = mapper.release(base) == 1
    result["counts"] = list(counts)  # by reason: process detach, process attach, thread attach, thread detach
    Path(result_file).write_text(json.dumps(result))


def check_started(label, result):
    """pe_map facts of a started run: kernel32 only, started, released, and the per-reason counters."""
    found = []
    if result.get("imported") != ["kernel32.dll"]:
        found.append(f"imports: requested from {result.get('imported')}, expected kernel32.dll only")
    if result.get("failure") or result.get("started") != 1 or not result.get("released"):
        found.append(f"pe_map: failure {result.get('failure')}, started {result.get('started')}, "
                     f"released {result.get('released')}")
    elif result.get("counts") != STARTED_COUNTS:
        found.append(f"notifications: callback calls by reason {result.get('counts')}, expected {STARTED_COUNTS} "
                     "(process detach, process attach, thread attach, thread detach)")
    return [f"{label}: {problem}" for problem in found]


def check(mapper, exe, early):
    info = layout(exe)
    rva, length = info["template"]
    positions = [info["table"].index(c) if c in info["table"] else None for c in info["ours"]]
    if None in positions or positions[0] > positions[1]:
        raise AssertionError(f"callback table {[hex(c) for c in info['table']]} does not hold "
                             f"{[hex(c) for c in info['ours']]} in that order")
    print(f"template: RVA {rva:#x}, {length} bytes; the array crosses the page boundary at RVA "
          f"{info['boundary']:#x} with words {info['below']:#010x} | {info['above']:#010x}; "
          f"the callbacks are entries {positions[0]} and {positions[1]} of {len(info['table'])} in the table")

    nat, map_ = native(exe), mapped(mapper, exe)
    print(f"native: LOG:{nat['log']} exit {nat['exit']:#x}")
    print(f"mapped: LOG:{map_['log']} exit {map_['exit']:#x}")
    for label, result in (("native", nat), ("mapped", map_)):
        found = problems(result)
        if result["stderr"]:
            found.append(f"stderr {result['stderr']!r}")
        if label == "mapped":
            found += check_started(label, map_)
        if found:
            raise AssertionError(f"{label}: " + "; ".join(found))
    if (nat["log"], nat["exit"]) != (map_["log"], map_["exit"]):
        raise AssertionError(f"native and mapped differ: {nat} != {map_}")
    print("native and pe_map_run print the same log and exit code: both callbacks run in table order on process "
          "attach and on thread attach and detach of two concurrent workers, every thread starts from the "
          f"template and keeps its own writes ({MAIN_WRITE}, {A_WRITE}, {B_WRITE})")
    print(f"pe_map: imports bound only to kernel32.dll; callback calls by reason {map_['counts']} "
          "(process detach, process attach, thread attach, thread detach): release sent no process detach")

    # Eligibility: a thread that attaches during the process-attach callbacks is not notified.
    probe = mapped(mapper, early)
    found = problems(probe, early=True) + check_started("early", probe)
    if found:
        raise AssertionError("early probe thread: " + "; ".join(found))
    print(f"early: LOG:{probe['log']}")
    print("early: the probe thread started and joined inside the first process-attach callback ran (pf) without "
          "any EXE notification; notifications became eligible after process attach, before the entry point")

    # Mapped but never started, and refused by pe_map_run: no callback for host threads or at release.
    for mode in ("mapped-only", "refused"):
        quiet = mapped(mapper, exe, mode)
        expected_started = {"mapped-only": None, "refused": 0}[mode]
        if (quiet.get("failure"), quiet.get("started"), quiet.get("released"), quiet.get("counts"), quiet["log"]) \
                != (None, expected_started, True, [0, 0, 0, 0], None):
            raise AssertionError(f"{mode}: expected no callback and no log, got {quiet}")
        print(f"{mode}: two host threads attached and detached and the image was released with no callback "
              f"(counts {quiet['counts']}, pe_map_run returned {quiet['started']})")

    controls = {"swapped callback table": (swap_callbacks, "callback order"),
                "template zeroed from its page boundary": (zero_second_page, "template")}
    with TemporaryDirectory() as temporary:
        for label, (edit, reason) in controls.items():
            copy = Path(temporary) / f"cpu_exe_tls_{reason.replace(' ', '_')}.exe"
            edit(exe, copy, info)
            for target, result in (("native", native(copy)), ("mapped", mapped(mapper, copy))):
                found = problems(result)
                if not any(problem.startswith(reason) for problem in found):
                    raise AssertionError(f"negative control '{label}' was not detected under {target} "
                                         f"(expected a '{reason}' diagnostic): {found}")
                print(f"negative control '{label}' rejected under {target}: {found[0]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run(*sys.argv[2:6])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mapper", type=Path)
        parser.add_argument("exe", type=Path)
        parser.add_argument("early", type=Path, help="the fixture built with EARLY_THREAD")
        arguments = parser.parse_args()
        check(arguments.mapper, arguments.exe, arguments.early)
