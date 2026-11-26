"""Check the unsigned bound on a pixel search emitted as a dynamic loop."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'startup'))
import gfx2hlsl as G


def check(compare='scc = s15 >= s11;', step='s15 = s15 + 0x00000001u;', extra=(), expected=True):
    stage = G.Stage.__new__(G.Stage)
    stage.require_counted_search = True
    stage.blocks = [
        (0, [compare], ('branch', '!scc', 2)),
        (1, [], ('jump', 4)),
        (2, list(extra), ('branch', '!scc', 4)),
        (3, [step], ('jump', 0)),
        (4, [], ('end',)),
    ]
    try:
        body = stage.body()
    except G.Unsupported as reason:
        assert not expected and 'no proven unsigned iteration bound' in str(reason)
    else:
        assert expected and any('[loop] while' in line for line in body)
        assert all('guard <' not in line for line in body)


if __name__ == '__main__':
    check(extra=['s24 = s10 + s15;'])
    check(compare='scc = s15 > s11;', expected=False)
    check(compare='scc = s15 >= s15;', expected=False)
    check(step='s15 = s15 + 0x00000002u;', expected=False)
    check(extra=['s11 = s10 + s15;'], expected=False)
    check(extra=['s15 -= 4u;'], expected=False)
    check(extra=['foo(s11);'], expected=False)
    print('Counted search and six malformed-loop controls passed.')
