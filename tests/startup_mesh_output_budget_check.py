from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/startup'))
import mesh2hlsl as M


def check():
    pipe = SimpleNamespace(ps=SimpleNamespace(attrs={0: {'flat'}, 1: {'half', 'linear'}}))
    decl = ('    float4 param0 : PARAM0;\n    float4 param1 : PARAM1;\n'
            '    nointerpolation uint4 fparam0 : FPARAM0;\n    float4 hiparam1 : HIPARAM1;\n')
    ms = decl + 'if (exec) { o.param0 = raw; o.fparam0 = flat; o.param1 = mixed; }'
    ps = decl + 'return i.fparam0.x + i.hiparam1.y + i.param1.z;'
    compact, pixel = M.compact_mesh_interface(pipe, ms, ps)
    assert 'float4 param0 :' not in compact and 'o.param0 =' not in compact
    assert 'float4 param0 :' not in pixel
    assert 'o.fparam0 = flat;' in compact and 'o.param1 = mixed;' in compact
    assert pixel.endswith('return i.fparam0.x + i.hiparam1.y + i.param1.z;')


if __name__ == '__main__':
    check()
    print('PASS: flat-only and mixed half/linear mesh outputs')
