"""s_nor_b32 and s_nand_b32 on a lane mask in tools/startup/gfx2hlsl.py (Part of #1723).

Pure Python, no arguments. A lane mask is the per-lane bit the translator keeps for exec, vcc and the SGPRs that
hold a compare result. Each form is decoded and translated, and the emitted lines are run on the CPU with the integer
evaluator of startup_gfx2hlsl_bit_alu_check.py over every pair of lane bits. The reference is the ISA meaning on one
lane: D = ~(S0 op S1), kept to that lane's bit. The destination must also be recorded as a mask (or be exec/vcc), so
that a later mask operation reads it as one.

s_xnor_b32 on a lane mask stays refused: it is outside this change.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import startup_gfx2hlsl_sop2_complement_check as C  # noqa: E402

G = C.G
REFERENCE = {
    's_nand_b32': lambda a, b: ~(a & b) & 1,
    's_nor_b32': lambda a, b: ~(a | b) & 1,
}
# (form, destination variable, source variables): vcc_lo/exec_lo name themselves in the emitted lines.
FORMS = [
    ((3, C.VCC_LO, 2, None), 's3', ['vcc', 's2']),
    ((3, 1, C.EXEC_LO, None), 's3', ['s1', 'exec']),
    ((C.VCC_LO, 1, 2, None), 'vcc', ['s1', 's2']),
    ((C.EXEC_LO, 1, 2, None), 'exec', ['s1', 's2']),
]


def lane_value(stage, name, form, dest, srcs, a, b):
    env = {srcs[0]: a, srcs[1]: b, 'scc': C.SCC_UNSET}
    for source in ('s1', 's2', 'vcc', 'exec'):
        env.setdefault(source, 0)
    return C.run_lines(stage.lines, env)[dest]


def main():
    for name, reference in REFERENCE.items():
        wrong, refused = [], []
        for form, dest, srcs in FORMS:
            asm, stage = C.attempt(C.code(name, form))
            if stage.lines and stage.lines[0].startswith('refused: '):
                refused.append((asm, stage.lines[0]))
                continue
            for a in (0, 1):
                for b in (0, 1):
                    got = lane_value(stage, name, form, dest, srcs, a, b)
                    if got != reference(a, b):
                        wrong.append((asm, a, b, got, reference(a, b)))
        C.check('%s: vcc_lo/exec_lo forms are lowered' % name, not refused, str(refused[:2]))
        C.check('%s: lane-bit results equal the reference' % name, not wrong, str(wrong[:2]))

        stage = C.Probe()
        stage.masks.add(1)  # s1 holds a compare result
        try:
            stage.one(name, '%s s3, s1, s2' % name)
        except G.Unsupported as error:
            C.check('%s: an SGPR mask source is lowered' % name, False, str(error))
        else:
            C.check('%s: an SGPR mask source marks the destination as a mask' % name, 3 in stage.masks,
                    str(stage.masks))

    refused = C.attempt(C.code('s_xnor_b32', FORMS[0][0]))[1].lines
    C.check('s_xnor_b32 on a lane mask is still refused', refused == ['refused: s_xnor_b32 on a lane mask'],
            str(refused))

    if C.failures:
        print('FAIL')
        for text in C.failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
