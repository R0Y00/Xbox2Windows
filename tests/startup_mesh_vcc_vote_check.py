from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/startup'))
import mesh2hlsl as M


def stage(joined):
    vote = M.MeshStage.__new__(M.MeshStage)
    branch = ('branch', 'lane != 0', 3)
    vote.lane_edges = {id(branch)}
    vote.blocks = [
        (0, [], branch),
        (1, [], ('jump', 4 if joined else 5)),
        (3, [], None),
        (4, [], ('barrier', 'vcc-vote')),
        (5, [], ('end',)),
    ]
    return vote


if __name__ == '__main__':
    stage(True).check_vote_participation({4})
    try:
        stage(False).check_vote_participation({4})
    except M.Unsupported:
        pass
    else:
        raise AssertionError('partial-wave vote accepted')
    print('PASS: reconvergence and partial-wave bypass')
