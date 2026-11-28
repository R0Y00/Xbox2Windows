"""Direct compute wave32 carry-out to vcc_hi, with an independent vcc_lo seed."""
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

from native_compute_clear_check import write_request
from native_direct_container_check import compute, words
from native_image_atomic_check import render


SEED = 0x13579BDF
ODD_LANES = 0xAAAAAAAA
END = 0xBF810000
DIAGNOSTIC = 'unsupported carry mask destination'


def vop2(op, dst, src0, src1):
    return op << 25 | dst << 17 | src1 << 9 | src0


def vop3b(op, dst, sdst, src0, src1, src2=0):
    return [0xD4000000 | op << 16 | sdst << 8 | dst, src0 | src1 << 9 | src2 << 18]


def carry_in(sdst):
    # v_add_co_ci_u32_e64 v16, sdst, 0, v16, s105
    return vop3b(0x128, 16, sdst, 128, 256 + 16, 105)


def program():
    return [
        # s[0:1] is the table. Load the u0 descriptor into s[8:11].
        0xBE810380, 0xF4080200, 0xFA000000, 0xBF8CC07F,
        0xBE8003FF | 106 << 16, SEED,                # s_mov_b32 vcc_lo, seed
        vop2(0x1C, 16, 194, 0),                     # v_or_b32 v16, -2, v0
        *vop3b(0x30F, 17, 105, 129, 256 + 16),      # v_add_co_u32 v17, s105, 1, v16
        *carry_in(107),
        0x7E000200 | 18 << 17 | 107,                # v_mov_b32 v18, vcc_hi
        0x7E000200 | 19 << 17 | 106,                # v_mov_b32 v19, vcc_lo
        0xD5010000 | 20, 128 | 129 << 9 | 107 << 18,  # v_cndmask_b32_e64 v20, 0, 1, vcc_hi
        *vop3b(0x128, 20, 104, 128, 256 + 20, 107),  # v_add_co_ci_u32_e64 v20, s104, 0, v20, vcc_hi
        vop2(0x1A, 8, 132, 0),                      # v_lshlrev_b32 v8, 4, v0
        # buffer_store_dword, offen, s[8:11], offsets 0, 4, and 8.
        0xE0701000, 0x80020008 | 16 << 8,
        0xE0701004, 0x80020008 | 18 << 8,
        0xE0701008, 0x80020008 | 19 << 8,
        0xE070100C, 0x80020008 | 20 << 8,
        END,
    ]


def compile_shader(cli, sdk, work, source, profile):
    shader = source.with_suffix('.cso')
    run = subprocess.run([cli, sdk, work / 'cache', profile, source, shader],
                         capture_output=True, text=True, timeout=30)
    return shader, run


def compile_positive(cli, sdk, work):
    source = work / 'carry.cs'
    source.write_bytes(compute(words(*program()), table=True))
    shader, run = compile_shader(cli, sdk, work, source, 'gfx1013-container-cs')
    assert run.returncode == 0 and shader.is_file(), (run.stdout, run.stderr)
    return shader


def check_shape(dump):
    # Follow SSA dependencies through the direct emitter's register loads and stores.
    # Treat each ballot result as a distinct source, independent of its predicate.
    ballots, extracts, outputs = [], [], []
    values, slots = {}, {}
    for line in dump.splitlines():
        # DXC appends a "; ..." comment to intrinsic calls.
        line = line.split(";", 1)[0].strip()
        store = re.fullmatch(r'store i32 (\S+), i32\* (\S+), align 4', line)
        if store:
            value, slot = store.groups()
            slots[slot] = values.get(value, {value})
        raw = re.fullmatch(r'call void @dx\.op\.rawBufferStore\.i32\(i32 140, %dx\.types\.Handle \S+, '
                           r'i32 \S+, i32 undef, i32 (\S+), i32 undef, i32 undef, i32 undef, i8 1, i32 4\)'
                           r'(?: #\d+)?', line)
        if raw:
            value = raw[1]
            outputs.append(values.get(value, {value}))
        assign = re.fullmatch(r'(%[\w.]+) = (.+)', line)
        if not assign:
            continue
        name, rhs = assign.groups()
        load = re.fullmatch(r'load i32, i32\* (\S+), align 4', rhs)
        ballot = re.fullmatch(r'call %dx\.types\.fouri32 @dx\.op\.waveActiveBallot\(i32 116, i1 (\S+)\)'
                              r'(?: #\d+)?', rhs)
        extract = re.fullmatch(r'extractvalue %dx\.types\.fouri32 (\S+), (\d+)', rhs)
        if load:
            values[name] = slots.get(load[1], set())
        elif ballot:
            ballots.append((name, ballot[1]))
            values[name] = {name}
        else:
            if extract:
                extracts.append(extract.groups())
            values[name] = set().union(*(values.get(token, set()) for token in re.findall(r'%[\w.]+', rhs)))

    assert len(ballots) == 4 and len(outputs) == 4, (ballots, outputs, dump)
    exec_mask, carry_in_mask, carry_out_mask = [name for name, _ in ballots[:3]]
    assert extracts == [(name, '0') for name, _ in ballots], extracts
    # The last ballot is the carry-out of the add whose carry-in is vcc_hi.
    assert carry_out_mask in values[ballots[3][1]], ballots
    assert ballots[0][1] == 'true' and all(pred.startswith('%') for _, pred in ballots[1:]), ballots
    assert exec_mask in values[ballots[1][1]], ballots
    assert {exec_mask, carry_in_mask} <= values[ballots[2][1]], ballots
    assert {carry_in_mask, exec_mask} <= outputs[0] and carry_out_mask not in outputs[0], outputs
    assert carry_out_mask in outputs[1] and carry_in_mask not in outputs[1], outputs
    assert outputs[2].isdisjoint({carry_in_mask, carry_out_mask}), outputs
    assert carry_out_mask in outputs[3] and carry_in_mask not in outputs[3], outputs
    assert str(SEED) in outputs[2], outputs
    assert 'icmp ult i32' in dump and 'or i1' in dump and 'zext i1' in dump, dump
    assert '@waveExchange' not in dump, dump


def controls(cli, sdk, work):
    shader = compile_positive(cli, sdk, work)
    dump = subprocess.run([Path(sdk) / 'dxc.exe', '-dumpbin', shader], check=True,
                          capture_output=True, text=True, timeout=30).stdout
    check_shape(dump)

    # Raw requests exercise the narrow wave64 model. FSSM wave64 always selects
    # the full model, even with a group of 32 threads.
    signature, root = work / 'root.hlsl', work / 'root.bin'
    signature.write_text('#define SIG "DescriptorTable(UAV(u0,numDescriptors=8))"\n')
    subprocess.run([Path(sdk) / 'dxc.exe', '-T', 'rootsig_1_1', '-E', 'SIG', '-Fo', root, signature],
                   check=True, timeout=30)
    for label, threads, wave32, dst in (('full wave64', 64, False, 107), ('narrow wave64', 32, False, 107),
                                        ('m0', 32, True, 124), ('ttmp0', 32, True, 108)):
        state = words(0, 0, 0, threads | 1 << 16, 1 | (int(wave32) << 20), 0, 0x8A)
        # Make the carry input and VGPR numeric so the narrow-model mutation can lower.
        code = [0xBE800380 | 105 << 16, 0x7E0002C1 | 16 << 17, *carry_in(dst), END]
        marker = write_request(work, code, root.read_bytes(), state)
        output, run = compile_shader(cli, sdk, work, marker, 'gfx1013-direct-cs@0')
        assert run.returncode and not output.exists(), (label, run.stdout, run.stderr)
        assert f'native compute +0x8: {DIAGNOSTIC}' in run.stderr.splitlines(), (label, run.stderr)
    print('wave32 vcc_hi ballot and vcc_lo seed DXIL matched, and all four controls refused')


def hardware(cli, sdk, work, renderer, adapter):
    wave = render(renderer, adapter, '--shader-wave32')
    assert wave.returncode == 0, (wave.stdout, wave.stderr)
    shader = compile_positive(cli, sdk, work)
    expected = [0xC5C5C5C5] * 320
    for lane in range(32):
        expected[4 * lane:4 * lane + 4] = [0 if lane & 1 else 0xFFFFFFFE, ODD_LANES, SEED, 2 * (lane & 1)]
    case = work / 'carry.bin'
    case.write_bytes(words(1, *expected))
    run = render(renderer, adapter, '--compute-store-sized-case', shader, case)
    assert run.returncode == 0, (run.stdout, run.stderr)
    print(run.stdout, end='')


if __name__ == '__main__':
    mode, cli, sdk, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as temp:
        {'controls': controls, 'hardware': hardware}[mode](cli, sdk, Path(temp), *args)
