"""Compare implicit TLS values seen in TLS callbacks on the main and worker threads.

The tls_value fixture keeps one __declspec(thread) int (7) and an 8 KiB
__declspec(thread) array of distinct words in its TLS data template, so the
template spans at least one 4 KiB page boundary. Two TLS callbacks (callers 1
and 3, in that order in the callback table) and the entry point (caller 2) each
report the observed state in the sink's reserved slot: the int in the low 16
bits and the number of array words that differ from the template above them, so
a block copied from the template reports 7.

Each case runs in a fresh child process: load or map the image, check that the
template in the loaded image matches the file, observe and write the main
thread's copy, then run two workers that are alive at the same time. Worker A
observes, writes 50 and waits; worker B starts after A's write, observes, writes
60 and waits; then both observe again. The workers are joined with a timeout,
the image is released and the sink events are reported with raw thread IDs
translated to semantic identities ("main", "a", "b") and kept in call order.

Expected (Refs #165, #870): under both loaders every thread starts from the
template, both callbacks run in table order on every notification, a write on
one thread reaches neither another thread's copy nor the template, and the two
loaders report the same events in the same order. pe_map serves the data
template through a carrier DLL that Windows loads (tools/pe_map.cpp). The
empty-template control (tls_route) must map under both loaders.

Two negative controls patch copies of the built fixture: one swaps the two
entries of the callback table and one zeroes the template from its first page
boundary on. Each must be rejected under both loaders with a diagnostic that
names the callback order or the template, so a harness that stopped checking
either would fail.
"""

import argparse
import ctypes
import json
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from native_map_check import Mapper
from native_tls_check import Call, load_sink

TLS, ENTRY, TLS_SECOND = 1, 2, 3
ATTACH, DETACH, THREAD_ATTACH, THREAD_DETACH = 1, 0, 2, 3
TLS_VALUE = 7
MAIN_WRITE, A_WRITE, B_WRITE = 20, 50, 60
SPAN_WORDS = 2048
PAGE = 0x1000
WAIT_MS = 10000  # shorter than the harness timeout so the exit path is observed
TIMEOUT = 60


def span_word(i):
    """span_word in tests/cpu/tls_value.cpp."""
    return (0xA5000000 ^ (i * 0x9E3779B1)) & 0xFFFFFFFF


def written(value):
    """observe() after tls_value_write(value): the int and two changed array words."""
    return (2 << 16) | value


class Worker(ctypes.Structure):
    """Worker in tests/cpu/tls_value.cpp."""
    _fields_ = [("wrote", ctypes.c_void_p), ("resume", ctypes.c_void_p), ("value", ctypes.c_int32),
                ("waited", ctypes.c_int32), ("first", ctypes.c_uint64), ("written", ctypes.c_uint64),
                ("last", ctypes.c_uint64)]


def template_of(image):
    """(rva, bytes) of the TLS data template in the file."""
    with pefile.PE(str(image)) as pe:
        tls = pe.DIRECTORY_ENTRY_TLS.struct
        rva = tls.StartAddressOfRawData - pe.OPTIONAL_HEADER.ImageBase
        return rva, pe.get_data(rva, tls.EndAddressOfRawData - tls.StartAddressOfRawData)


def layout(image):
    """Where the array lies in the template, the first page boundary inside it and the callback table."""
    rva, data = template_of(image)
    pattern = b"".join(struct.pack("<I", span_word(i)) for i in range(SPAN_WORDS))
    offset = data.find(pattern)
    if offset < 0:
        raise AssertionError(f"{Path(image).name}: the 8 KiB array is not in the TLS template")
    span_rva = rva + offset
    # Select the next boundary strictly inside the span, including an aligned start.
    boundary = (span_rva // PAGE + 1) * PAGE
    if not span_rva < boundary < span_rva + len(pattern):
        raise AssertionError(f"{Path(image).name}: the array at {span_rva:#x} does not cross a page boundary")
    index = (boundary - span_rva) // 4
    with pefile.PE(str(image)) as pe:
        tls = pe.DIRECTORY_ENTRY_TLS.struct
        table_rva = tls.AddressOfCallBacks - pe.OPTIONAL_HEADER.ImageBase
        table = []
        while True:
            entry = pe.get_qword_at_rva(table_rva + 8 * len(table))
            if not entry:
                break
            table.append(entry)
        export = next(e.address for e in pe.DIRECTORY_ENTRY_EXPORT.symbols if e.name == b"tls_value_callbacks")
        ours = [pe.get_qword_at_rva(export), pe.get_qword_at_rva(export + 8)]
    return {"template": (rva, len(data)), "boundary": boundary, "below": span_word(index - 1),
            "above": span_word(index), "table_rva": table_rva, "table": table, "ours": ours}


def translate(calls, module, names):
    """[caller, reason, semantic thread, value] for this module's events, in call order."""
    out = []
    for call in calls:
        if call.module != module or call.thread not in names:
            continue  # other images, or unrelated pool threads; same rule as native_tls_check
        out.append([call.caller, call.reason, names[call.thread], call.reserved])
    return out


def emit(report):
    print(json.dumps(report))
    sys.stdout.flush()


def run(target, image, sink_path, mapper_path):
    """Child process: one load, main-thread writes, two concurrent workers, release."""
    sink = load_sink(sink_path)
    sink.sink_clear()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentThreadId.restype = ctypes.c_uint32
    kernel.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    kernel.CreateEventW.restype = ctypes.c_void_p
    kernel.SetEvent.argtypes = [ctypes.c_void_p]
    kernel.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CreateThread.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    names = {kernel.GetCurrentThreadId(): "main"}

    if target == "native":
        kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
        kernel.LoadLibraryW.restype = ctypes.c_void_p
        kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
        kernel.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        kernel.GetProcAddress.restype = ctypes.c_void_p
        module = kernel.LoadLibraryW(str(image))
        if not module:
            return emit({"failure": ctypes.get_last_error()})
        release = lambda: kernel.FreeLibrary(module)  # noqa: E731
        export = lambda name: kernel.GetProcAddress(module, name)  # noqa: E731
    else:
        mapper = Mapper(mapper_path)
        # The fixture links the static C runtime, whose imports are bound to the host's system DLLs.
        bindings = {Path(sink_path).name.casefold(): sink, "kernel32.dll": kernel}
        module, status, detail, _ = mapper(Path(image), bindings)
        if not module:
            return emit({"failure": [status, detail]})
        release = lambda: mapper.release(module)  # noqa: E731
        export = lambda name: mapper.export(module, name)  # noqa: E731

    addresses = {name: export(name) for name in (b"tls_value_observe", b"tls_value_write", b"tls_value_worker")}
    if not all(addresses.values()):
        release()
        return emit({"failure": f"missing export: {[n.decode() for n, a in addresses.items() if not a]}"})
    observe = ctypes.CFUNCTYPE(ctypes.c_uint64)(addresses[b"tls_value_observe"])
    write = ctypes.CFUNCTYPE(None, ctypes.c_int)(addresses[b"tls_value_write"])

    rva, expected_template = template_of(image)

    def template():
        return ctypes.string_at(module + rva, len(expected_template)) == expected_template

    report = {"failure": None, "template_loaded": template()}
    report["main"] = observe()
    write(MAIN_WRITE)
    report["main_written"] = observe()

    workers, handles = {}, {}
    for name, value in (("a", A_WRITE), ("b", B_WRITE)):
        worker = Worker(kernel.CreateEventW(None, 1, 0, None), kernel.CreateEventW(None, 1, 0, None), value, -1)
        thread_id = ctypes.c_uint32(0)
        handle = kernel.CreateThread(None, 0, addresses[b"tls_value_worker"], ctypes.byref(worker), 0,
                                     ctypes.byref(thread_id))
        if not handle:
            return emit({"failure": f"create-thread {name}"})
        workers[name], handles[name] = worker, handle
        names[thread_id.value] = name
        # B starts only after A has written its copy, while A is still alive.
        if kernel.WaitForSingleObject(worker.wrote, WAIT_MS) != 0:
            return emit({"failure": f"worker {name} did not write"})  # it may still run module code
    for worker in workers.values():
        kernel.SetEvent(worker.resume)
    for name, handle in handles.items():
        if kernel.WaitForSingleObject(handle, WAIT_MS) != 0:
            return emit({"failure": f"worker {name} did not finish"})
        kernel.CloseHandle(handle)
    for name, worker in workers.items():
        report[name] = {"first": worker.first, "written": worker.written, "last": worker.last,
                        "waited": worker.waited}
        kernel.CloseHandle(worker.wrote)
        kernel.CloseHandle(worker.resume)
    report["main_end"] = observe()
    report["template_end"] = template()

    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    report["events"] = translate(buffer[:count], module, names)
    sink.sink_clear()
    report["released"] = bool(release())
    # Detach events stay in the sink after release (the sink outlives the image).
    count = sink.sink_read(buffer, len(buffer))
    report["detach"] = translate(buffer[:count], module, names)
    emit(report)


def run_control(target, image, sink_path, mapper_path):
    """Child process for the empty-template control: load, observe attach, release.

    The control fixture (tls_route) exports no tls_value_* symbols, so run()
    above cannot be reused here. Success is a clean load with the TLS callback
    and entry point each observed once with DLL_PROCESS_ATTACH; any load
    failure is reported, never masked."""
    sink = load_sink(sink_path)
    sink.sink_clear()
    if target == "native":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
        kernel.LoadLibraryW.restype = ctypes.c_void_p
        kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
        module = kernel.LoadLibraryW(str(image))
        if not module:
            return emit({"failure": ctypes.get_last_error()})
        release = lambda: kernel.FreeLibrary(module)  # noqa: E731
    else:
        mapper = Mapper(mapper_path)
        module, status, detail, _ = mapper(Path(image), {Path(sink_path).name.casefold(): sink})
        if not module:
            return emit({"failure": [status, detail]})
        release = lambda: mapper.release(module)  # noqa: E731
    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    attach = [[c.caller, c.reason, "module" if c.module == module else hex(c.module), c.reserved]
              for c in buffer[:count] if c.reason == ATTACH]
    release()
    emit({"failure": None, "attach": attach})


def child(target, image, sink, mapper, control=False):
    args = ["--run-control" if control else "--run", target, str(image), str(sink), str(mapper)]
    result = subprocess.run([sys.executable, "-B", __file__] + args, capture_output=True, text=True,
                            timeout=TIMEOUT)
    if result.returncode:
        raise AssertionError(f"{target} {Path(image).name}: child exited {result.returncode:#x}: {result.stderr}")
    return json.loads(result.stdout)


def groups(events):
    """{(thread, reason): [(caller, value), ...]} in call order."""
    out = {}
    for caller, reason, thread, value in events:
        out.setdefault((thread, reason), []).append((caller, value))
    return out


EXPECTED_STATE = {("main", ATTACH): TLS_VALUE, ("a", THREAD_ATTACH): TLS_VALUE, ("b", THREAD_ATTACH): TLS_VALUE,
                  ("a", THREAD_DETACH): written(A_WRITE), ("b", THREAD_DETACH): written(B_WRITE),
                  ("main", DETACH): written(MAIN_WRITE)}


def problems(report):
    """Every way one run departs from the expected behavior; empty when it matches.

    Each problem starts with "template" when a thread saw something other than its own copy of the
    template, or "callback order" when the two TLS callbacks ran against the table order."""
    if report["failure"] is not None:
        return [f"load failed: {report['failure']}"]
    found = []
    if not report["template_loaded"] or not report["template_end"]:
        found.append(f"template: the TLS template in the loaded image differs from the file "
                     f"(after load {report['template_loaded']}, at the end {report['template_end']})")
    values = {"main": (report["main"], TLS_VALUE), "main written": (report["main_written"], written(MAIN_WRITE)),
              "main end": (report["main_end"], written(MAIN_WRITE))}
    for name, value in (("a", A_WRITE), ("b", B_WRITE)):
        values[f"{name} first"] = (report[name]["first"], TLS_VALUE)
        values[f"{name} written"] = (report[name]["written"], written(value))
        values[f"{name} last"] = (report[name]["last"], written(value))
        if report[name]["waited"] != 0:
            found.append(f"worker {name} did not see resume (wait result {report[name]['waited']})")
    for name, (seen, expected) in values.items():
        if seen != expected:
            found.append(f"template: {name} observed {seen:#x}, expected {expected:#x} "
                         f"(value {seen & 0xffff}, {seen >> 16} array words differ from the template)")
    if not report["released"]:
        found.append("release failed")

    # Every notification: which thread, which reason and the state that thread's own block held.
    seen = groups(report["events"] + report["detach"])
    if set(seen) != set(EXPECTED_STATE):
        found.append(f"notifications for {sorted(seen)}, expected {sorted(EXPECTED_STATE)}")
    for key, calls in seen.items():
        callers = [caller for caller, _ in calls]
        if sorted(callers) != [TLS, ENTRY, TLS_SECOND]:
            found.append(f"{key}: callers {callers}, expected each of {TLS}, {TLS_SECOND} and {ENTRY} once")
        elif callers.index(TLS) > callers.index(TLS_SECOND):
            found.append(f"callback order: {key} ran callers {callers}; the table has {TLS} before {TLS_SECOND}")
        for caller, value in calls:
            if key in EXPECTED_STATE and value != EXPECTED_STATE[key]:
                found.append(f"template: caller {caller} on {key} observed {value:#x}, "
                             f"expected {EXPECTED_STATE[key]:#x}")
    return found


def swap_callbacks(source, target, info):
    """A copy whose callback table holds the fixture's two callbacks in the opposite order."""
    data = bytearray(Path(source).read_bytes())
    with pefile.PE(data=bytes(data)) as pe:
        slots = [pe.get_offset_from_rva(info["table_rva"] + 8 * info["table"].index(c)) for c in info["ours"]]
    first, second = (struct.unpack_from("<Q", data, slot)[0] for slot in slots)
    struct.pack_into("<Q", data, slots[0], second)
    struct.pack_into("<Q", data, slots[1], first)
    Path(target).write_bytes(data)


def zero_second_page(source, target, info):
    """A copy whose TLS template is zero from its first page boundary to its end."""
    data = bytearray(Path(source).read_bytes())
    rva, length = info["template"]
    with pefile.PE(data=bytes(data)) as pe:
        start = pe.get_offset_from_rva(info["boundary"])
        end = pe.get_offset_from_rva(rva) + length
    data[start:end] = bytes(end - start)
    Path(target).write_bytes(data)


def order(report):
    return {f"{thread}/{reason}": [caller for caller, _ in calls]
            for (thread, reason), calls in groups(report["events"] + report["detach"]).items()}


def check(value_image, control_image, sink, mapper):
    # Control first: the empty-template fixture must load under both loaders,
    # proving the harness itself can map and observe thread notifications.
    for target in ("native", "mapped"):
        observed = child(target, control_image, sink, mapper, control=True)
        if observed["failure"] is not None:
            raise AssertionError(f"control {target}: failed to load: {observed}")
        callers = [c for c, _, _, _ in observed["attach"]]
        if TLS not in callers or ENTRY not in callers:
            raise AssertionError(f"control {target}: missing attach callback: {observed}")
    print("control: empty-template TLS fixture loads under Windows and pe_map")

    info = layout(value_image)
    rva, length = info["template"]
    positions = [info["table"].index(c) if c in info["table"] else None for c in info["ours"]]
    if None in positions or positions[0] > positions[1]:
        raise AssertionError(f"callback table {[hex(c) for c in info['table']]} does not hold "
                             f"{[hex(c) for c in info['ours']]} in that order")
    print(f"template: RVA {rva:#x}, {length} bytes; the array crosses the page boundary at RVA "
          f"{info['boundary']:#x} with words {info['below']:#010x} | {info['above']:#010x}; "
          f"the callbacks are entries {positions[0]} and {positions[1]} of {len(info['table'])} in the table")

    reports = {}
    for target in ("native", "mapped"):
        report = child(target, value_image, sink, mapper)
        found = problems(report)
        if found:
            raise AssertionError(f"{target}: " + "; ".join(found) + f"\nreport: {report}")
        reports[target] = report
        print(f"{target}: callers in call order per thread/reason {order(report)}")
    # Per thread and reason, in call order. The two workers are resumed together, so which of them
    # exits first is not ordered and the interleaving across threads is not compared.
    native, mapped = (groups(reports[t]["events"] + reports[t]["detach"]) for t in ("native", "mapped"))
    if native != mapped:
        raise AssertionError(f"native and mapped notifications differ: {native} != {mapped}")
    print(f"native and mapped agree: the main thread and two concurrent workers start from the template "
          f"({TLS_VALUE}, no changed array words), both callbacks run in table order on every notification, "
          f"and writes of {MAIN_WRITE}, {A_WRITE} and {B_WRITE} stay on their own thread while the template in "
          f"the image is unchanged")

    # Negative controls: each damaged copy must be rejected under both loaders, for the named reason.
    controls = {"swapped callback table": (swap_callbacks, "callback order"),
                "template zeroed from its page boundary": (zero_second_page, "template")}
    with TemporaryDirectory() as temporary:
        for label, (edit, reason) in controls.items():
            copy = Path(temporary) / f"tls_value_{reason.replace(' ', '_')}.dll"
            edit(value_image, copy, info)
            for target in ("native", "mapped"):
                found = problems(child(target, copy, sink, mapper))
                if not any(problem.startswith(reason) for problem in found):
                    raise AssertionError(f"negative control '{label}' was not detected under {target} "
                                         f"(expected a '{reason}' diagnostic): {found}")
                print(f"negative control '{label}' rejected under {target}: {found[0]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--run", "--run-control"):
        (run_control if sys.argv[1] == "--run-control" else run)(*sys.argv[2:6])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for arg in ("value", "control", "sink", "mapper"):
            parser.add_argument(arg, type=Path)
        args = parser.parse_args()
        check(args.value, args.control, args.sink, args.mapper)
