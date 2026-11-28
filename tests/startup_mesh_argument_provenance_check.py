"""Observe the source-built SRV -> GPU writer -> indirect Mesh chain.

The positive oracle requires matching resource AND byte interval, X > 0 and
the expected scene-color pixels. Controls must fail that same positive oracle.
"""
from pathlib import Path
import os
import subprocess
import sys


CASES = ("positive", "missing-copy", "stale-copy", "wrong-srv-offset", "wrong-argument-offset")


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def observations(stdout, prefix):
    return [fields(line) for line in stdout.splitlines() if line.startswith(prefix)]


def positive_oracle(records, result):
    before, after = records
    return (before["resource"].lower() == after["resource"].lower()
            and before["offset"] == after["offset"]
            and before["tuple"] == after["tuple"]
            and int(before["tuple"].split(",")[0]) > 0
            and before["guards"] == after["guards"] == "intact"
            and int(result["changed"]) == int(result["total"]) == 8)


def check_history(stdout, stderr, scenario):
    descriptor, = observations(stdout, "provenance descriptor ")
    history = observations(stderr, "x2w-descriptor-history seq=")
    assert history, "missing descriptor history"
    sequences = [int(event["seq"]) for event in history]
    assert sequences == list(range(1, len(history) + 1)), sequences
    destination = descriptor["guest"]
    source = descriptor["source"]
    source_writes = [event for event in history if event["guest"] == source and event["op"] == "create"]
    assert len(source_writes) == 2, source_writes
    initial, overwrite = source_writes
    assert initial["first"] == "8", initial
    assert overwrite["first"] == ("8" if scenario == "wrong-srv-offset" else "4"), overwrite
    assert overwrite["old"] == initial["new"], overwrite
    bound = [event for event in history if event["guest"] == destination]
    assert bound[0]["op"] == "create" and bound[0]["resource"] == "0", bound
    if scenario == "missing-copy":
        assert len(bound) == 1, bound
    else:
        assert bound[1]["op"] == "copy" and bound[1]["source"] == source, bound
        assert bound[1]["new"] == initial["new"], bound
        if scenario == "stale-copy":
            assert len(bound) == 2 and bound[-1]["new"] != overwrite["new"], bound
        else:
            assert bound[-1]["op"] == "copy" and bound[-1]["new"] == overwrite["new"], bound
            assert int(bound[-1]["seq"]) > int(overwrite["seq"]), bound
    final = bound[-1]
    assert final["desktop"] == descriptor["desktop"], final
    assert int(final["guest"], 16) != int(final["desktop"], 16), final
    magic, token = descriptor["token_bound"].split(",")
    assert magic == "56573258" and int(token) + 1 == int(final["new"]), descriptor
    for event in history:
        assert event["slot"] == "18" and event["materialized"] == "1", event
        assert event["guest_read"] == "1" and event["guest_bytes"] == "32", event
        if event["resource"] != "0":
            assert int(event["backing"], 16) == int(descriptor["input"], 16), event
        if event["old"] != "0":
            previous = event["guest_before"].split(",")
            assert previous[0] == "56573258" and int(previous[1], 16) + 1 == int(event["old"]), event


def check(executable, adapter, scenario, trace=True):
    env = dict(os.environ)
    env["X2W_MESH_PROVENANCE_CASE"] = scenario
    if trace:
        env["X2W_D3D12_DESCRIPTOR_HISTORY"] = "1"
    else:
        env.pop("X2W_D3D12_DESCRIPTOR_HISTORY", None)
    # Always inspect this process's stderr, not a caller's persistent trap log.
    env.pop("X2W_TRAP_LOG", None)
    result = subprocess.run([str(executable), adapter, "mesh-provenance"], env=env,
                            capture_output=True, text=True, timeout=45)
    if result.returncode == 77:
        print(result.stdout, end="")
        return False
    print(result.stdout, end="")
    assert result.returncode == 0, (scenario, result.returncode, result.stdout, result.stderr)
    assert "DEVICE REMOVED" not in result.stderr and "ExecuteIndirect refused" not in result.stderr, result.stderr
    records = observations(result.stdout, "provenance record ")
    assert len(records) == 2 and [r["phase"] for r in records] == ["post-write", "pre-consume"], records
    summary, = observations(result.stdout, "provenance result ")
    assert positive_oracle(records, summary) == (scenario == "positive"), (scenario, records, summary)
    assert summary["changed"] == ("8" if scenario == "positive" else "0"), summary
    fences = observations(result.stdout, "provenance fence ")
    assert [event["value"] for event in fences] == ["1", "2", "3"], fences
    assert len({event["queue"] for event in fences}) == len({event["fence"] for event in fences}) == 1, fences
    assert all(int(event["completed"]) >= int(event["value"]) for event in fences), fences
    producer, = observations(result.stdout, "provenance producer ")
    consumer, = observations(result.stdout, "provenance consumer ")
    assert producer["queue"] == consumer["queue"] == fences[0]["queue"], (producer, consumer)
    assert producer["resource"] == consumer["resource"] == records[0]["resource"] == records[1]["resource"], records
    assert producer["offset"] == records[0]["offset"] == "4", records
    assert consumer["offset"] == records[1]["offset"] == ("20" if scenario == "wrong-argument-offset" else "4"), records
    assert producer["bytes"] == consumer["bytes"] == "12", (producer, consumer)
    write, = observations(result.stdout, "provenance input-write ")
    scalar, = observations(result.stdout, "provenance input-record ")
    descriptor, = observations(result.stdout, "provenance descriptor ")
    assert write["queue"] == producer["queue"], write
    assert int(write["resource"], 16) == int(scalar["resource"], 16) == int(descriptor["input"], 16), scalar
    assert write["offset"] == scalar["offset"] == "16" and write["value"] == scalar["value"] == "2", scalar
    assert scalar["adjacent_offset"] == "32" and scalar["adjacent_value"] == "0", scalar
    assert result.stdout.index("provenance record phase=post-write") < result.stdout.index("provenance consumer ")
    if trace:
        check_history(result.stdout, result.stderr, scenario)
    else:
        assert "x2w-descriptor-history " not in result.stderr, result.stderr
    return True


def main():
    executable = Path(sys.argv[1]).resolve()
    adapter = sys.argv[2]
    for scenario in CASES:
        if not check(executable, adapter, scenario):
            return 77
    assert check(executable, adapter, "positive", trace=False)
    print("Mesh argument provenance: positive, four negative controls and trace opt-out passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
