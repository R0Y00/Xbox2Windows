"""Exercise the native trace filter and selective translator through their CLI."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from startup_compute_request_offline_check import CODE, DATA, DESC, G, REASON, ROOT, STATE_A, STATE_B

scanner = Path(sys.argv[1]).resolve()
translator = Path(__file__).resolve().parents[1] / "tools/startup/gfx2hlsl.py"
key = f"{G.compute_key(CODE, DATA, ROOT, DESC, STATE_A, STATE_B):016x}"
other = "1111111111111111"
draw = f"x2w-frame draw 3 NOT drawn target 1234 80x40 fmt 28 topology 0x4 key {other} "
compute = f"x2w-frame cx-uav #7 key {key} (1, 1, 1) pipeline 1234: "
with tempfile.TemporaryDirectory() as temp:
    cache = Path(temp)
    trace = cache / "trap.log"
    trace.write_bytes(("\r\n".join([
        compute + "skipped: no translation, no manifest: data",
        draw.replace("frame draw ", "frame draw-indexed ") + "no translation",
        compute + "skipped: no translation, no manifest: data",
        draw.replace(other, "0000000000000000") + "no translation record",
        draw.replace(other, other + "1") + "no translation",
        draw.replace("NOT drawn", "drawn") + "no translation",
        compute + "skipped: heap",
    ])).encode())
    rows = subprocess.check_output([scanner, trace], text=True).splitlines()
    assert rows == [f"{key}\t2", f"{other}\t1"], rows
    for suffix, data in dict(desc=DESC, sa=STATE_A, sb=STATE_B, data=DATA, root=ROOT, c67=CODE).items():
        (cache / f"req_{key}.{suffix}").write_bytes(data)
    # A second pending request is absent from the trace and must be left alone.
    (cache / "req_2222222222222222.a").touch()
    cmd = [sys.executable, "-B", translator, "--warm-trace", cache, trace, scanner]
    before = set(cache.iterdir())
    report = subprocess.check_output(cmd + ["--list"], text=True)
    assert report.splitlines() == ["2 distinct missed keys, 1 pending translator requests", key], report
    assert set(cache.iterdir()) == before
    env = dict(os.environ, X2W_TRANSLATE_JOBS="1")
    subprocess.run(cmd, check=True, env=env)
    assert set(cache.iterdir()) - before == {cache / f"{key}.fail"}
    assert (cache / f"{key}.fail").read_text().strip() == REASON
print("trace selection, CRLF, duplicate keys, preview and targeted translation passed")
