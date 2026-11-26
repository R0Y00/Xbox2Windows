"""Check mesh phase scheduling when only one wave enters the publisher."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/startup'))
import mesh2hlsl as M  # noqa: E402


def fixture():
    stage = M.MeshStage.__new__(M.MeshStage)
    stage.blocks = [
        (0, [], ('branch', 'tid < 64u', 4)),
        (1, ['++turns;'], ('branch', 'turns < 3u', 1)),
        (2, ['if (tid == 64u) mailbox[1] = 91u;'], ('barrier', 'console')),
        (3, [], ('jump', 7)),
        (4, ['if (tid == 0u) mailbox[0] = 47u;'], ('barrier', 'lane-scc')),
        (5, ['local = mailbox[0];'], ('barrier', 'console')),
        (6, [], ('jump', 7)),
        (7, ['dump.Store(4u * tid, local + turns + mailbox[tid < 64u ? 1u : 0u]);'], ('end',)),
    ]
    stage.uniform_skips = set()
    return stage


def check():
    stage = fixture()
    try:
        stage.barrier_count()
    except M.UnevenBarrierPaths:
        pass
    else:
        raise AssertionError('the original scheduler unexpectedly accepted the uneven paths')
    stage.barriers = stage.padded_barrier_count()
    stage.padded_phases = True
    assert stage.barriers == 2
    assert stage.phase_of[2] == stage.phase_of[5] == 1
    assert stage.phase_of[4] == 0 and stage.phase_of[7] == 2
    assert 'else { break; }' in '\n'.join(stage.body())
    for index, term in [(5, None), (4, ('branch', 'again', 1))]:
        bad = fixture()
        bad.blocks[index] = (index, [], term)
        try:
            bad.padded_barrier_count()
        except M.Unsupported:
            continue
        raise AssertionError('unequal console counts or a barrier inside a loop must refuse')
    return stage


if __name__ == '__main__':
    check()
    print('PASS: paired console barriers, one-arm publish, unequal counts and backward edges')
