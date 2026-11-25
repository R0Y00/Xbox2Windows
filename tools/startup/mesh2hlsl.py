"""gfx1013 NGG mesh-stage machine code -> HLSL mesh shader (SM 6.5) for a mesh + pixel pipeline (startup tools).

The console compiles a mesh shader to an NGG primitive shader (wave64, 128 threads, one thread per output vertex and
one per output primitive). This module translates that program with the pipeline translator's one-lane model
(gfx2hlsl.Stage: registers are uint variables, exec/vcc/masks hold the lane's bit) and wraps it in an HLSL mesh
shader:

  * one HLSL thread per console lane: SV_GroupIndex = the console's thread index in the group
    (wave id from merged_wave_info bits 27:24 times 64, plus v_mbcnt lane);
  * s_sendmsg 0x9 (GS_ALLOC_REQ) records m0 = vertices | primitives << 12 (Mesa ac_nir_prerast_utils.c
    ac_nir_ngg_alloc_vertices_and_primitives) in groupshared memory; after the program a barrier publishes it and
    SetMeshOutputCounts runs once, in uniform control flow;
  * `exp pos0` / `exp paramN` from thread t define output vertex t, `exp prim` (target 20) primitive t, with the
    gfx10 primitive export word: index i in bits [10*i+8 : 10*i], bit 31 = null primitive (Mesa ac_nir_lower_ngg.c
    "nir_ubfe_imm(..., 10 * v, 9)" and the "1u << 31" null export). A null or out-of-range primitive is emitted
    with SV_CullPrimitive;
  * the workgroup index: the console reads v5 (first ES input VGPR, the vertex id in a fast-launch NGG
    dispatch) of its first lane and shifts it right by 7. This translation sets v5 = SV_GroupID.x * 128 +
    SV_GroupIndex (INFERRED: consistent with every one of the 9282 archive shaders reading v5 >> 7 in each wave
    without LDS; a register dump of a mesh pipeline's GE/SPI state would settle it).

Cross-lane facts the one-lane model cannot see are checked and refused (Unsupported), never guessed:
  * v_readfirstlane_b32 needs a source that is the same in every lane (taint analysis over the emitted HLSL);
  * a per-lane branch (execz/execnz/vccz/vccnz) is taken per lane; that equals the wave-level branch only when
    no scalar register written in the skipped range is read after it, and, for vcc, when vcc is uniform over the
    active lanes (`exec & uniform`);
  * a branch on an SCC that depends on lane masks (the hardware sets SCC from the whole 64-bit result) is taken
    per lane only when that SCC is the same in every active lane and every running thread is an active lane;
    otherwise the wave's SCC is published through groupshared memory and a group barrier (MeshStage.lane_scc),
    or the program is refused where a barrier cannot be placed (X2W_MESH_LANE_SCC=0 refuses all of them).
LDS, DPP and permlane are not lowered (refused by the base translator); v_writelane / v_readlane only as the SGPR
spill idiom (per-wave scalar slots, MeshStage.lower_spills).
"""
import copy
import os
import re
import struct
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as G  # noqa: E402
from gfx2hlsl import Unsupported, reg_range, parse  # noqa: E402

GROUP = 128           # PSV0 numthreads (128, 1, 1) in all 9282 archive mesh records
MAX_VERTICES = 128    # PSV0 MaxOutputVertices
MAX_PRIMITIVES = 128  # PSV0 MaxOutputPrimitives
ALL_LANES = (1 << 64) - 1
# User SGPRs of the mesh stage start at s16 (MEASURED: in all 4045 distinct run programs the SGPRs read before any
# write lie in s16..s24, and the real root signatures (regblob g57_*.root) give 9 = 3 tables + 1 root SRV + 2
# constants + resource heap + sampler heap; s8..s15 are written first. Pixel and vertex stages start at s8.)
MESH_USER_SGPR = 16
FUSED = ('v_fma_f32', 'v_fmac_f32', 'v_fmamk_f32', 'v_fmaak_f32')
UNFUSED = ('v_mad_f32', 'v_mac_f32', 'v_madmk_f32', 'v_madak_f32')
FMA_MODE = os.environ.get('X2W_MESH_FMA', 'exact')
if os.environ.get('X2W_MESH_EXACT_FMA') == '0':
    FMA_MODE = 'precise'          # the round-1 switch, kept
if FMA_MODE not in ('exact', 'precise'):
    raise SystemExit('X2W_MESH_FMA: exact or precise, not %r' % FMA_MODE)
EXACT_FMA = FMA_MODE == 'exact'
LDS_DWORDS = 1024     # groupshared x2w_lds (4 KB of the 28 KB a D3D12 mesh shader may use); addresses are clamped
TOKEN = re.compile(r'\b(v\d+|s\d+|exec|vcc|m0|scc|a|w|t|q|tid|lane)\b')
SIMPLE = re.compile(r'^\s*(?:if \(exec & 1\) )?(v\d+|s\d+|exec|vcc|m0|scc|a|w|t|q)\s*=\s*(.*)$')

MESH_HELPERS = {
    # v_perm_b32 D, S0, S1, S2: byte i of D = selector byte i of S2 over the eight bytes {S0 (bytes 4-7), S1 (0-3)};
    # 8..11 replicate the sign bit of S1[15], S1[31], S0[15], S0[31]; 12 gives 0x00, 13..15 give 0xff
    # (RDNA 2 ISA, V_PERM_B32).
    'x2w_perm': ('uint x2w_perm_byte(uint s0, uint s1, uint sel) { if (sel < 4u) return (s1 >> (8u * sel)) & 255u; '
                 'if (sel < 8u) return (s0 >> (8u * (sel - 4u))) & 255u; if (sel == 8u) return ((s1 >> 15) & 1u) * 255u; '
                 'if (sel == 9u) return ((s1 >> 31) & 1u) * 255u; if (sel == 10u) return ((s0 >> 15) & 1u) * 255u; '
                 'if (sel == 11u) return ((s0 >> 31) & 1u) * 255u; if (sel == 12u) return 0u; return 255u; }\n'
                 'uint x2w_perm(uint s0, uint s1, uint s2) { uint r = 0; [unroll] for (uint i = 0; i < 4u; ++i) '
                 'r |= x2w_perm_byte(s0, s1, (s2 >> (8u * i)) & 255u) << (8u * i); return r; }'),
    # v_mbcnt_{lo,hi}_u32_b32 D, S0, S1 in a 64-lane wave: S1 + the number of set S0 bits below this lane in the
    # low (lanes 0-31) or high (lanes 32-63) half of the wave mask.
    'x2w_fclass': ('uint x2w_fclass(uint u) { uint e = (u >> 23) & 255u, m = u & 0x7fffffu, s = u >> 31; '
                   'if (e == 255u) return m != 0u ? ((m >> 22) != 0u ? 1u : 0u) : (s != 0u ? 2u : 9u); '
                   'if (e == 0u) return m == 0u ? (s != 0u ? 5u : 6u) : (s != 0u ? 4u : 7u); '
                   'return s != 0u ? 3u : 8u; }'),
    'x2w_fma':'float x2w_fma(float a, float b, float c) { return (float)fma((double)a, (double)b, (double)c); }',
    'x2w_madp': 'float x2w_madp(float a, float b, float c) { precise float p = a * b; precise float r = p + c; return r; }',
    'x2w_mbcnt_lo':'uint x2w_mbcnt_lo(uint m, uint lane) { return countbits(m & (lane >= 32u ? 0xffffffffu : ((1u << lane) - 1u))); }',
    'x2w_mbcnt_hi': 'uint x2w_mbcnt_hi(uint m, uint lane) { return lane < 32u ? 0u : countbits(m & ((1u << (lane - 32u)) - 1u)); }',
    'x2w_hword': ('ByteAddressBuffer x2w_vdesc : register(t2, space1);\n'
                  'uint x2w_hword(uint h, uint i) { uint id = x2w_vid(h); return id < %du ? x2w_vdesc.Load(id * 32u + 4u * i) : 0u; }'
                  % getattr(G, 'HEAP_IDS', 90000)),
}
# Mesh-only lowerings whose manifest lines need runtime support not in the draw path yet (comma list):
# heapwords = numeric reads of resource-heap descriptor words (x2w_hword, manifest `heapwords 1`).
# Exec-off conflicts (MeshStage.merge_exec_off): on by default; X2W_MESH_EXECOFF=0 restores the earlier refusals.
EXEC_OFF = os.environ.get('X2W_MESH_EXECOFF', '1') != '0'
# X2W_MESH_EXEC_COPIES=0: per-lane skipped ranges accept only exec copies taken inside the range (the f37ad9f8 rule);
# default on: also a pair proven equal to exec at the branch, and copies of copies (MeshStage.exec_copies_at).
EXEC_COPIES = os.environ.get('X2W_MESH_EXEC_COPIES', '1') != '0'
# X2W_MESH_BARRIER_RANGES=0: a group-uniform execz range is skipped when no lane of the group is active even when it
# holds a phase barrier (the f37ad9f8 form; barrier_count then refuses "paths cross different numbers of barriers").
BARRIER_RANGES = os.environ.get('X2W_MESH_BARRIER_RANGES', '1') != '0'
# X2W_MESH_RESTORED_SCALARS=0: a scalar written in a per-lane skipped range counts as live after it even when every
# exit of the range gives it back its value from before, and liveness reads VALU lane masks by their low SGPR only
# (the f37ad9f8 rule).
RESTORED_SCALARS = os.environ.get('X2W_MESH_RESTORED_SCALARS', '1') != '0'
# X2W_MESH_SPILLS=0: v_writelane_b32 / v_readlane_b32 stay refused (the 9eacac67 behaviour); default on: the SGPR
# spill idiom is lowered to per-wave scalar slots (MeshStage.lower_spills).
SPILLS = os.environ.get('X2W_MESH_SPILLS', '1') != '0'
# X2W_MESH_VCC_DATA=0: vcc stays the one-lane mask variable everywhere (32-bit data in vcc_lo is cut to bit 0, vcc_hi
# is refused); default on: vcc holding no lane mask is the SGPR pair s[106:107] (MeshStage.lower_vcc_data).
VCC_DATA = os.environ.get('X2W_MESH_VCC_DATA', '1') != '0'
# X2W_MESH_DATA_PAIRS=0: every s_mov_b64 between SGPR pairs is a lane-mask move (the base: bit 0 of the low word);
# default on: a pair that holds no lane mask is copied as its two 32-bit registers (MeshStage.scalar_inner).
DATA_PAIRS = os.environ.get('X2W_MESH_DATA_PAIRS', '1') != '0'
SLOT_BASE = 1000      # spill slot (i-th spill VGPR, lane k) = pseudo-SGPR s<SLOT_BASE + 64 * i + k>
# X2W_MESH_RFL_FIRST=0: v_readfirstlane_b32 of a lane-dependent value under a partial exec is refused (as before);
# default on: the wave's first ACTIVE lane is found through groupshared memory and a group barrier
# (MeshStage.first_active_broadcast), and an execz range holding such a read is run by the whole group.
RFL_FIRST = os.environ.get('X2W_MESH_RFL_FIRST', '1') != '0'
RFL_FIRST_SLOTS = 8   # first-active-lane reads per program (each: 2 x 128 groupshared words)
MESH_FEATURES = frozenset(f for f in os.environ.get('X2W_MESH_FEATURES', '').replace(' ', '').split(',') if f)
# descword (MeshStage.dw_*): a word of a TABLE descriptor read as a number becomes the runtime's entry value
# `udsrc <n> descword <param> <byte>` (d3d12_draw.inl, needs X2W_MESH_DESCWORD=1 there), computed lazily: only words
# some numeric use reaches get an entry value; arithmetic nobody reads stays dead (0u).
if MESH_FEATURES - {'heapwords', 'descword'}:
    raise SystemExit('X2W_MESH_FEATURES: unknown %s' % ', '.join(sorted(MESH_FEATURES - {'heapwords', 'descword'})))
# Lane-mask SCC (wave "any lane" value through groupshared memory, MeshStage.lane_scc): on unless
# X2W_MESH_LANE_SCC=0. X2W_MESH_TEST_DROP=publish|restore (TESTS ONLY, negative controls) leaves out the wave
# publish of such an SCC / the restore of hit scalars after a group-uniform execz range: both give wrong results.
LANE_SCC = os.environ.get('X2W_MESH_LANE_SCC', '1') != '0'
TEST_DROP = frozenset(os.environ.get('X2W_MESH_TEST_DROP', '').replace(' ', '').split(',')) - {''}
# Lane taint merged at joins of forward edges (MeshStage.merge): on unless X2W_MESH_JOIN_TAINT=0 (the linear scan).
JOIN_TAINT = os.environ.get('X2W_MESH_JOIN_TAINT', '1') != '0'
# The per-lane shortcut for an active-uniform lane-mask SCC with lanes off needs a mask result confined to the
# active lanes (MeshStage.one_body): on unless X2W_MESH_CONFINED_SCC=0 (tests only: the unrestricted shortcut).
CONFINED_SCC = os.environ.get('X2W_MESH_CONFINED_SCC', '1') != '0'
# A group-uniform execz skip may jump over every barrier the translation added (readfirstlane broadcasts, inner
# group-branch publishes, lane-mask SCC publishes), never over the program's s_barrier: on unless
# X2W_MESH_SKIP_ADDED=0 (then only lane-mask SCC publishes may be skipped, as at 135ee723).
SKIP_ADDED = os.environ.get('X2W_MESH_SKIP_ADDED', '1') != '0'
# Exec lineage (MeshStage.emit, lineage_ok): a value written from uniform values while exec was E stays the same in
# every active lane wherever exec is E or a narrowing of E (also after a restore to a saved copy of E or of a
# narrowing of it), not only until the next exec widening. OFF unless X2W_MESH_EXEC_LINEAGE=1: it moves no request key
# to ok yet (the keys it unblocks stop at a numeric descriptor word next) and no difftest covers a program needing it.
EXEC_LINEAGE = os.environ.get('X2W_MESH_EXEC_LINEAGE', '0') == '1'


class LaneScc(Unsupported):
    """A branch on an SCC computed from lane masks; `origins` = offsets of the SALU mask instructions it comes from
    (MeshPipeline translates again with those publishing the wave's value)."""

    def __init__(self, text, origins):
        super().__init__(text)
        self.origins = origins


class NeedGroupRange(Unsupported):
    """A lane-dependent v_readfirstlane_b32 inside the per-lane execz range at `off`: MeshPipeline translates again
    with that range run by the whole group (MeshStage.group_branches), so the read can use a group barrier."""

    def __init__(self, off, text):
        super().__init__(text)
        self.off = off

# Loop heads take the join of their entry and back edges, iterated until it settles (MeshStage.loop_one).
# X2W_MESH_LOOP_JOIN=0 restores the linear state at loop heads (the old behaviour: wrong for loop-carried values).
LOOP_JOIN = os.environ.get('X2W_MESH_LOOP_JOIN', '1') != '0'
LOOP_PASSES = 8
BOTTOM = 'x2w_bottom'   # key in a symbolic state: no path reaches this code in the current loop pass
# Waterfall loops (MeshStage.waterfalls): one pass per lane with the lane's own value. X2W_MESH_WATERFALL=0 refuses
# them as before (v_readfirstlane of a lane-dependent value with lanes off).
WATERFALL = os.environ.get('X2W_MESH_WATERFALL', '1') != '0'
# Instructions a waterfall body may not contain: cross-lane, wave-level, exec, memory ordering across lanes.
# Per-lane loops (MeshStage.perlane_loop): a back edge on "some lane still loops" becomes each thread's own loop.
# X2W_MESH_PERLANE_LOOPS=0 refuses them as before (branch on an SCC that depends on lane masks).
PERLANE_LOOPS = os.environ.get('X2W_MESH_PERLANE_LOOPS', '1') != '0'
PERLANE_BODY_REFUSED = re.compile(r'(v_readlane|v_writelane|v_permlane|ds_|s_barrier|s_sendmsg|exp$|s_setreg|s_store|'
                                  r's_atomic|s_buffer_store|s_buffer_atomic|.*_dpp$|s_endpgm)')
WATERFALL_BODY_REFUSED = re.compile(r'(v_readlane|v_readfirstlane|v_writelane|v_permlane|ds_|s_barrier|s_sendmsg|exp$|'
                                    r's_setreg|s_cbranch|s_branch|s_endpgm|.*_dpp$|.*_saveexec|.*_wrexec|v_cmpx)')


class LoopRetry(Exception):
    """A loop head's state was not the join of its edges; carries the back-edge states for the next pass."""

    def __init__(self, backs):
        super().__init__('loop state not settled')
        self.backs = backs


def scalar_regs(text):
    """SGPR numbers named in an operand string (s7, s[4:7])."""
    out = set()
    for m in re.finditer(r'\bs\[(\d+):(\d+)\]|\bs(\d+)\b', text):
        if m.group(3):
            out.add(int(m.group(3)))
        else:
            out.update(range(int(m.group(1)), int(m.group(2)) + 1))
    return out


CARRY = re.compile(r'v_(add|sub|subrev)_co(_ci)?_u32')

NO_DEST = ('s_cmp', 's_bitcmp', 's_cbranch', 's_branch', 's_waitcnt', 's_sendmsg', 's_barrier', 's_nop', 's_setreg',
           's_inst_prefetch', 's_endpgm', 's_clause', 's_setprio', 's_code_end', 's_sopp')


def reads_writes(name, asm):
    """(scalar registers read, scalar registers written, writes vcc, reads vcc) of one instruction (approximate:
    writes are only what the destination operand names)."""
    _, ops, mods = parse(asm)
    base = re.sub(r'_e(32|64)$', '', name)
    writes, reads = set(), set()
    wvcc = rvcc = False
    if base.startswith('s_') and not base.startswith(NO_DEST) and ops:
        writes = scalar_regs(ops[0])
        wvcc = ops[0].startswith('vcc')
        rest = ops[1:]
        if base.startswith(('s_load', 's_buffer_load')):
            rest = ops[1:]
    elif (base.startswith('v_cmp') and len(ops) == 3) or base == 'v_readfirstlane_b32' or base == 'v_readlane_b32':
        writes = scalar_regs(ops[0])
        wvcc = ops[0].startswith('vcc')
        rest = ops[1:]
    elif CARRY.fullmatch(base) and len(ops) >= 4:
        # VOP2/VOP3b carry-out: the disassembler names one SGPR (or vcc_lo); in wave64 the mask is the pair
        writes = scalar_regs(ops[1])
        writes |= {n + 1 for n in writes}
        wvcc = ops[1].startswith('vcc')
        rest = ops[2:]
    else:
        rest = ops
        if base.startswith('v_cmp') and len(ops) == 2:
            wvcc = not base.startswith('v_cmpx')
    for o in rest:
        reads |= scalar_regs(o)
        rvcc |= o.startswith('vcc')
    if base in ('v_cndmask_b32', 'v_addc_co_u32') and len(ops) == 3:
        rvcc = True
    if base in ('s_cbranch_vccz', 's_cbranch_vccnz'):
        rvcc = True
    return reads, writes, wvcc, rvcc


class MeshStage(G.Stage):
    """One NGG mesh stage. kind 'ms'."""

    def __init__(self, pipe, code):
        self.tainted = {'tid', 'lane', 'v5'}
        self.au, self.epoch = {}, 0
        # exec lineage: exec states form a tree (a narrowing is a child of the state it narrows); xs = the current
        # state, au_xs[reg] = (state, write serial) of an active-uniform value, xs_copy[sgpr] = the state it copies
        self.xs, self.xs_parent, self.au_xs, self.xs_copy, self.xs_serial = 0, {0: None}, {}, {}, 0
        self._xs_exec_seen = False
        self.au_scalars = {}
        self.broadcasts = 0
        self.literal_tables = {}
        self.first_slots = 0          # first-active-lane reads (x2w_rfv / x2w_rfe slots)
        self.divergent_until = -1
        self.exported_vertex = False
        self.exported_prim = False
        self.alloc_seen = False
        self.wave = 64
        self.exec_static = ALL_LANES  # exec as a known 64-bit lane mask (the same in every wave), else None
        self.static_copies = {}      # SGPR -> the static exec mask it holds
        self._static_set = False
        self.shadow = {}             # VGPR -> (unguarded variable, exec mask at the write); straight-line code only
        self.shadow_vars = []
        self.pending = []            # statements that start the next block (after a broadcast barrier)
        self.scc_from = {}           # register -> offsets of lane-mask SCC definitions its value derives from
        self.scc_any = []            # lane-mask SCCs published per wave (offsets): x2w_sany[k][wave]
        self.join_from = {}          # branch target -> scc_from union over the branches to it
        self.fell_off = False        # the previous instruction does not fall through (jump, end)
        self.running_active = False  # inside a per-lane execz range, no exec write since: running threads = active lanes
        self.scc_confined = set()    # lane-mask SCC offsets whose mask result is inside the active lanes (`x & exec`)
        self.lds_used = False
        self.lds_written_in_phase = False
        self.exec_off = set()        # conflict SGPRs whose conflicting paths all came with this lane's exec bit 0
        self._execz_edge = False
        self._execz_snaps = []       # snapshots of taken s_cbranch_execz edges (compared by identity)
        super().__init__(pipe, 'ms', code)
        if not self.alloc_seen:
            raise Unsupported('no GS_ALLOC_REQ (s_sendmsg 0x9): not an NGG mesh stage')
        self.barriers = self.barrier_count()

    def translate(self):
        # The program ends at the s_code_end padding; the words after it are data, not code.
        for i, x in enumerate(self.ins):
            if x[2] == 's_code_end':
                self.ins = self.ins[:i]
                break
        self.vcc_renamed = self.lower_vcc_data()
        self.spill_vgprs, self.spill_slots = self.lower_spills()
        self.swap_scc = self.lower_pair_swaps()
        self.has_lds = any(x[2].startswith('ds_') for x in self.ins)
        self.wave_any = []           # group-uniform execz branches (offsets): x2w_any[k][wave] = some lane active
        self.group_restore = {}      # range end -> statements restoring hit scalars in a wave that had no lane on
        self.group_checks = {}       # range end -> (register, symbolic value before the range, branch offset)
        self.uniform_skips = set()   # offsets of the inserted group-uniform skip blocks
        if LOOP_JOIN and self.loop_heads():
            self.loop_prepare()
            self.one = self.loop_one     # instance attribute: the base's scan calls it for every instruction
        result = super().translate()
        if LOOP_JOIN and self.loop_heads():
            self.loop_settled()
        for n in sorted(self.spill_vgprs):
            if n in self.used_v or n in self.written_v:
                raise Unsupported('v%d holds spilled scalars (v_writelane) and is also used as a VGPR' % n)
        # A group-uniform branch block ends in a barrier (the lanes publish their exec bit); the block after it
        # skips the range when no lane of the group was active (group_branch).
        # A range with a phase barrier inside (a console s_barrier, a broadcast or a nested group branch) is never
        # skipped: the skip is group-uniform, but a block after the range would then start in two different phases
        # (body() gives each block one). Running it with every lane off is what skipping does: vector writes,
        # exports, LDS writes and stores stay guarded by exec, and the restore gives each wave with no active lane
        # its scalars from before the range back.
        # (lane-mask SCC publish barriers are left out: 28020dea lets the uniform skip jump over those)
        barrier_at = [b[0] for b in self.blocks if b[2] and b[2][0] == 'barrier' and b[2][1:2] != ('lane-scc',)]
        blocks = []
        for b in self.blocks:
            if b[2] and b[2][0] == 'barrier' and len(b[2]) > 1 and b[2][1] == 'wave-branch':
                k = b[2][2]
                blocks.append((b[0], self.group_restore.get(b[0], []) + b[1], ('barrier',)))
                if not (BARRIER_RANGES and any(b[0] < o < b[2][3] for o in barrier_at)):
                    blocks.append((b[0] + 0.5, [], ('branch', '(x2w_any[%d][0] | x2w_any[%d][1]) == 0u' % (k, k), b[2][3])))
                    self.uniform_skips.add(b[0] + 0.5)
            elif b[0] in self.group_restore:
                blocks.append((b[0], self.group_restore[b[0]] + b[1], b[2]))
            else:
                blocks.append(b)
        missing = set(self.group_restore) - {b[0] for b in self.blocks}
        if missing:
            raise Unsupported('execz range end %#x is not a block start' % min(missing))
        self.blocks = blocks
        return result

    def lower_pair_swaps(self):
        """A ^= B; B ^= A; A ^= B swaps two disjoint SGPR pairs, including descriptors.
        Keep the original offsets for branches. No edge may enter the middle. The final
        XOR's wave-wide SCC must be dead because the three replacement moves leave it alone.
        """
        targets = {int(parse(asm)[1][0].split('_')[1], 16) for _, _, name, asm in self.ins
                   if name == 's_branch' or name.startswith('s_cbranch')}
        scc = set()
        for i, first in enumerate(self.ins):
            _, _, xor, _ = first
            if xor not in ('s_xor_b32', 's_xor_b64'):
                continue
            first_ops = parse(first[3])[1]
            if len(first_ops) != 3:
                continue
            touched = set().union(*(scalar_regs(o) for o in first_ops))
            sites = [i]
            for j in range(i + 1, min(i + 16, len(self.ins))):
                off, _, name, asm = self.ins[j]
                if off in targets:
                    break
                if name == xor:
                    sites.append(j)
                    if len(sites) == 3:
                        break
                else:
                    if not (name in ('s_mov_b32', 's_mov_b64') or
                            re.fullmatch(r'v_(?:add_f32|sub_f32|mul_f32|mov_b32)_e(?:32|64)', name)):
                        break
                    regs = set().union(*G.sgpr_access(name, asm))
                    if regs & touched or 'src_scc' in parse(asm)[1]:
                        break
            if len(sites) != 3:
                continue
            operands = [parse(self.ins[j][3])[1] for j in sites]
            if any(len(o) != 3 for o in operands):
                continue
            a, b = operands[0][0], operands[1][0]
            ra, rb = reg_range(a), reg_range(b)
            width = 2 if xor.endswith('64') else 1
            if not all(r and r[0] == 's' and r[2] == r[1] + width - 1 and r[2] < 128 for r in (ra, rb)):
                continue
            if abs(ra[1] - rb[1]) < width or operands[2][0] != a or any(set(o[1:]) != {a, b} for o in operands):
                continue
            # The architectural SGPRs are below 128; private lane spill slots start at SLOT_BASE.
            tmp = 's[%d:%d]' % (SLOT_BASE-2, SLOT_BASE-1) if width == 2 else 's%d' % (SLOT_BASE-2)
            move = 's_mov_b64' if width == 2 else 's_mov_b32'
            for j, (dst, src) in zip(sites, ((tmp, a), (a, b), (b, tmp))):
                off, n, _, _ = self.ins[j]
                self.ins[j] = (off, n, move, '%s %s, %s' % (move, dst, src))
            scc.add(self.ins[sites[-1]][0])
        return scc

    def check_scc(self, written, unknown):
        return super().check_scc(written, unknown | self.swap_scc)

    # ---- SGPR spills to VGPR lanes --------------------------------------------------------------
    # v_writelane_b32 vN, S, k writes the scalar S into lane k of vN and keeps the other lanes; v_readlane_b32 sD, vN, k
    # copies lane k of vN into sD. Neither reads exec (LLVM VOP2Instructions.td: "These are special and do not read
    # the exec mask", Uses = []; V_WRITELANE_B32 ties $vdst = $vdst_in). LLVM SIRegisterInfo::spillSGPR /
    # restoreSGPR and Mesa ACO spill a scalar register to one lane per 32-bit part with an immediate lane index, so
    # the 64 lanes of such a VGPR are 64 wave scalars: lane k is written and read only as a whole-wave value. In the
    # one-lane model that is a scalar register like any other; lower_spills renames it to the pseudo-SGPR
    # s<SLOT_BASE + 64 * i + k> and the instructions to scalar moves, so liveness, the per-lane skipped-range rules,
    # group-uniform branches, lane masks, symbolic descriptors and exec copies treat a slot exactly as an SGPR.
    @staticmethod
    def _half(tok):
        """(pair name for s_mov_b64, 0 = low / 1 = high half) of a 32-bit operand that is half of a 64-bit pair."""
        if tok in ('vcc_lo', 'vcc_hi'):
            return 'vcc', int(tok == 'vcc_hi')
        if tok in ('exec_lo', 'exec_hi'):
            return 'exec', int(tok == 'exec_hi')
        r = reg_range(tok)
        if r and r[0] == 's' and r[1] == r[2] and r[1] < SLOT_BASE:
            low = r[1] & ~1   # 64-bit SGPR operands are even-aligned
            return 's[%d:%d]' % (low, low + 1), r[1] & 1
        return None

    # ---- vcc as two 32-bit scalar registers ----------------------------------------------------------
    # VCC_LO / VCC_HI are SGPRs 106 / 107 (RDNA 2 ISA, scalar operand encoding); in wave64 the compiler uses them as
    # an ordinary SGPR pair when no lane mask needs them (32-bit loads, moves, float operands, spills). The one-lane
    # model keeps a lane MASK in the variable `vcc` (the lane's bit), so a 32-bit value there is cut to bit 0.
    @staticmethod
    def vcc_roles(name, asm):
        """[(operand index or None, halves, is_def, kind)] for each vcc operand; kind 'mask' (the wave64 lane mask
        the one-lane model keeps in `vcc`), 'data' (a 32-bit value) or 'any' (a 64-bit scalar move or logic op,
        a spill of a half: right either way)."""
        _, ops, mods = parse(asm)
        base = re.sub(r'_(e32|e64|sdwa|dpp)$', '', name)
        out = []
        if base in ('s_cbranch_vccz', 's_cbranch_vccnz'):
            return [(None, (0, 1), False, 'mask')]
        if base in ('v_div_fmas_f32', 'v_div_fmas_f64', 'v_div_scale_f32', 'v_div_scale_f64'):
            out.append((None, (0, 1), base.startswith('v_div_scale'), 'mask'))   # implicit vcc
        carry = re.fullmatch(r'v_(add|sub|subrev)_co(_ci)?_u32|v_(add|sub|subrev)c_co_u32', base)
        for i, tok in enumerate(ops):
            t = tok.lstrip('-|').rstrip('|')
            r = reg_range(t)
            if r and r[0] == 's' and r[1] <= 107 and r[2] >= 106:
                # the disassembler names vcc by number inside a register range (s_buffer_load_dwordx2 s[106:107])
                halves = tuple(h for h in (0, 1) if r[1] <= 106 + h <= r[2])
            elif t not in ('vcc', 'vcc_lo', 'vcc_hi'):
                continue
            else:
                halves = (0, 1) if t == 'vcc' else ((1,) if t == 'vcc_hi' else (0,))
            if base.startswith('s_'):
                is_def = i == 0 and not base.startswith(NO_DEST)
                kind = 'any' if (base.endswith('_b64') or 'saveexec' in base) else 'data'
            elif base.startswith('v_cmp') and not base.startswith('v_cmpx') and i == 0:
                is_def, kind, halves = True, 'mask', (0, 1)      # wave64 SDST: the disassembler names vcc_lo
            elif base == 'v_cndmask_b32' and i == len(ops) - 1:
                is_def, kind, halves = False, 'mask', (0, 1)
            elif carry and i == 1 and len(ops) >= 4:
                is_def, kind, halves = True, 'mask', (0, 1)
            elif carry and i == len(ops) - 1 and len(ops) >= 5:
                is_def, kind, halves = False, 'mask', (0, 1)
            elif base in ('v_writelane_b32', 'v_readlane_b32'):
                is_def, kind = base == 'v_readlane_b32' and i == 0, 'any'
            elif base in ('v_readfirstlane_b32',) and i == 0:
                is_def, kind = True, 'data'
            else:
                is_def, kind = False, 'data'
            out.append((i, halves, is_def, kind))
        # forms whose vcc the disassembler might leave implicit (VOPC / VOP2 carry / cndmask e32 without a vcc
        # operand): the lane mask
        if not any(j is not None for j, _, _, _ in out) and name.endswith('_e32'):
            if base.startswith('v_cmp') and not base.startswith('v_cmpx'):
                out.append((None, (0, 1), True, 'mask'))
            elif base == 'v_cndmask_b32' or (carry and '_ci' in base) or re.fullmatch(r'v_(add|sub|subrev)c_co_u32', base):
                out.append((None, (0, 1), False, 'mask'))
            if carry:
                out.append((None, (0, 1), True, 'mask'))
        return out

    def lower_vcc_data(self):
        """Rename vcc to the SGPR pair s[106:107] where it does not hold a lane mask. Webs of definitions and uses of
        each half (reaching definitions over the branch graph) are classified by their instructions: a web with a
        lane-mask use or definition keeps `vcc`; any other web (32-bit data, or only 64-bit moves) is renamed, which
        is exact (an SGPR pair takes every scalar form, including a lane mask moved through it). A web that is
        both a lane mask and 32-bit data is refused, unless its only 32-bit uses are scalar reads whose results
        nothing reads (dead_use; the web then keeps `vcc`, as before the rename). Returns the number of renamed
        operands."""
        if not VCC_DATA:
            return 0
        roles = {i: self.vcc_roles(x[2], x[3]) for i, x in enumerate(self.ins)}
        roles = {i: r for i, r in roles.items() if r}
        if not any(k == 'data' or (j is not None and k == 'any') for r in roles.values() for (j, h, d, k) in r):
            return 0
        parent = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            parent[find(a)] = find(b)
        kind_of = {}
        for i, r in roles.items():
            for n, (j, halves, is_def, kind) in enumerate(r):
                nodes = [(i, n, h) for h in halves]
                for a in nodes:
                    kind_of[a] = kind
                    find(a)
                for a in nodes[1:]:
                    union(nodes[0], a)
        offs = [x[0] for x in self.ins]
        at = {x[0]: i for i, x in enumerate(self.ins)}
        # Reaching definitions, with each path marked "exec is zero" after the taken edge of s_cbranch_execz (the
        # fall-through of execnz) until an exec write that can turn lanes on. A vector instruction does nothing in a
        # lane whose exec bit is off (in the one-lane model: its writes are guarded), so a definition that reaches a
        # vector data use only along such paths does not join that use's value.
        entry = (('entry', 0), ('entry', 1))
        for a in entry:
            kind_of[a] = 'any'
            find(a)

        def zero_edges(i):
            """(successor, path becomes exec-zero) pairs."""
            name, asm = self.ins[i][2], self.ins[i][3]
            out = []
            t = int(parse(asm)[1][0].split('_')[1], 16) if (name.startswith('s_cbranch') or name == 's_branch') else None
            if name not in ('s_endpgm', 's_branch') and i + 1 < len(self.ins):
                out.append((i + 1, name == 's_cbranch_execnz'))
            if t is not None and t in at:
                out.append((at[t], name == 's_cbranch_execz'))
            return out

        def narrows(name, asm):
            ops = parse(asm)[1]
            if name.startswith('v_cmpx') or name == 's_and_saveexec_b64':
                return True
            return bool(re.match(r's_(and|andn2)_b64$', name) and ops and ops[0] == 'exec' and 'exec' in ops[1:2]
                        or re.match(r's_and_b64$', name) and ops and ops[0] == 'exec' and 'exec' in ops[1:])
        # facts: (frozenset of path flags reaching, defs of half 0, defs of half 1); a def is (node, exec-zero flag)
        reach = {0: (frozenset([False]), frozenset([(entry[0], False)]), frozenset([(entry[1], False)]))} if self.ins else {}
        work = [0] if self.ins else []
        while work:
            i = work.pop()
            flags, d0, d1 = reach[i]
            facts = [d0, d1]
            name, asm = self.ins[i][2], self.ins[i][3]
            for n, (j, halves, is_def, kind) in enumerate(roles.get(i, ())):
                if not is_def:
                    vector_data = kind == 'data' and name.startswith('v_')
                    for h in halves:
                        for d, z in facts[h]:
                            if not (z and vector_data):
                                union((i, n, h), d)
            for n, (j, halves, is_def, kind) in enumerate(roles.get(i, ())):
                if is_def:
                    for h in halves:
                        facts[h] = frozenset(((i, n, h), z) for z in flags)
            if self.exec_written(name, parse(asm)[1]) and not narrows(name, asm):
                flags = frozenset([False])
                facts = [frozenset((d, False) for d, z in f) for f in facts]
            for s, zero in zero_edges(i):
                if zero:
                    out = (frozenset([True]),) + tuple(frozenset((d, True) for d, z in f) for f in facts)
                else:
                    out = (flags, facts[0], facts[1])
                old = reach.get(s)
                new = out if old is None else tuple(a | b for a, b in zip(old, out))
                if new != old:
                    reach[s] = new
                    work.append(s)
        webs = {}
        for a, k in kind_of.items():
            webs.setdefault(find(a), set()).add((k, a))
        all_reads = set()
        for x in self.ins:
            all_reads |= set(reads_writes(x[2], x[3])[0])

        def dead_use(i):
            """A scalar instruction reading vcc as data whose results nothing reads: its destination SGPRs are read
            by no instruction of the program, it writes no vcc / exec / m0, and its SCC (assumed written) is not
            read on any path before an s_cmp / s_bitcmp sets SCC again."""
            name, asm = self.ins[i][2], self.ins[i][3]
            if not name.startswith('s_') or name.startswith(('s_cbranch', 's_branch', 's_setpc', 's_swappc')):
                return False
            rd, wr, wv, rv = reads_writes(name, asm)
            ops = parse(asm)[1]
            if wv or (ops and ops[0].startswith(('exec', 'm0'))) or set(wr) & all_reads:
                return False
            seen, work = set(), [s for s, _ in zero_edges(i)]
            while work:
                k = work.pop()
                if k in seen:
                    continue
                seen.add(k)
                kn, ka = self.ins[k][2], self.ins[k][3]
                if G.reads_scc(kn, ka):
                    return False
                if kn.startswith(('s_cmp', 's_bitcmp')):
                    continue
                work += [s for s, _ in zero_edges(k)]
            return True
        rename = set()
        for root, members in webs.items():
            kinds = {k for k, _ in members}
            if 'mask' in kinds and 'data' in kinds:
                data = [a for k, a in members if k == 'data' and a[0] != 'entry']
                if all(not roles[a[0]][a[1]][2] and dead_use(a[0]) for a in data):
                    continue   # only dead 32-bit reads of a lane mask: nothing observes the cut value
                m = next(offs[a[0]] for k, a in members if k == 'mask' and a[0] != 'entry')
                d = next(offs[a[0]] for k, a in members if k == 'data' and a[0] != 'entry')
                raise Unsupported('vcc used as a lane mask (%#x) and as 32-bit data (%#x) in one value' % (m, d))
            if 'mask' not in kinds:
                rename |= {(a[0], a[1]) for k, a in members if a[0] != 'entry'}
        count = 0
        for i, n in sorted(rename):
            j = roles[i][n][0]
            if j is None:
                continue
            off, size, name, asm = self.ins[i]
            _, ops, mods = parse(asm)
            tok = ops[j]
            t = tok.lstrip('-|').rstrip('|')
            if t not in ('vcc', 'vcc_lo', 'vcc_hi'):
                continue   # already named s106 / s107 by number
            ops[j] = tok.replace(t, {'vcc': 's[106:107]', 'vcc_lo': 's106', 'vcc_hi': 's107'}[t])
            text = '%s %s' % (asm.split(' ', 1)[0], ', '.join(ops))
            if mods:
                text += ' ' + ' '.join(mods)
            self.ins[i] = (off, size, name, text)
            count += 1
        return count

    def lower_spills(self):
        """Rewrite the spill idiom in self.ins (before the translation and every analysis reads it) into scalar
        moves to pseudo-SGPR slots; refuse whatever is not that idiom. Returns (spill VGPR numbers, slot numbers).

        * v_writelane_b32 vN, S, k -> s_mov_b32 slot(N, k), S; v_readlane_b32 sD, vN, k -> s_mov_b32 sD, slot(N, k).
        * The two halves of a 64-bit pair (sX/sX+1, vcc_lo/hi, exec_lo/hi) spilled to lanes k, k+1, or reloaded from
          them, in one basic block with nothing in between reading or writing the registers or slots involved,
          become one s_mov_b64 at the later instruction (the earlier is a no-op): the value moved is the same, and
          the lane-mask and exec-copy rules see the pair. A half of vcc or exec on its own is refused (the one-lane
          model keeps only the lane's bit of a mask; vcc as 32-bit data is not modelled).
        * Refused: a lane select that is not a constant 0..63, a spill VGPR named by any other instruction (also
          checked against the translation's VGPR use afterwards), a slot read on some path before any write."""
        idx = [i for i, x in enumerate(self.ins) if x[2] in ('v_writelane_b32', 'v_readlane_b32')]
        if not idx or not SPILLS:
            return set(), set()
        info = {}
        vgprs = set()
        for i in idx:
            off, n, name, asm = self.ins[i]
            _, ops, mods = parse(asm)
            if mods or len(ops) != 3:
                raise Unsupported('%s form %s at %#x' % (name, asm, off))
            vtok = ops[0] if name == 'v_writelane_b32' else ops[1]
            v = reg_range(vtok)
            if not v or v[0] != 'v' or v[1] != v[2]:
                raise Unsupported('%s on %s at %#x' % (name, vtok, off))
            if not re.fullmatch(r'0x[0-9a-f]+|\d+', ops[2]) or int(ops[2], 0) > 63:
                raise Unsupported('%s lane select %s is not a constant lane at %#x' % (name, ops[2], off))
            info[i] = (name, v[1], int(ops[2], 0), ops[1] if name == 'v_writelane_b32' else ops[0])
            vgprs.add(v[1])
        order = sorted(vgprs)
        slot = {(v, k): SLOT_BASE + 64 * j + k for j, v in enumerate(order) for k in range(64)}
        # any other instruction naming a spill VGPR (single or in a range) uses it as per-lane data
        for i, (off, n, name, asm) in enumerate(self.ins):
            if i in info:
                continue
            for m in re.finditer(r'\bv(\d+)\b|\bv\[(\d+):(\d+)\]', asm):
                lo, hi = (int(m.group(1)),) * 2 if m.group(1) else (int(m.group(2)), int(m.group(3)))
                hit = [v for v in order if lo <= v <= hi]
                if hit:
                    raise Unsupported('v%d holds spilled scalars (v_writelane) and is also used by %s at %#x'
                                      % (hit[0], name, off))
        targets = self.branch_targets()

        def regs_of(name, asm):
            """(scalar registers read, written) with vcc = -1 and exec = -2."""
            r, w, wv, rv = reads_writes(name, asm)
            r, w = set(r), set(self.pair_writes(name, asm))
            ops = parse(asm)[1]
            if wv:
                w.add(-1)
            if rv:
                r.add(-1)
            if self.exec_written(name, ops):
                w.add(-2)
            if not name.startswith('s_') or any(t.startswith('exec') for t in ops[1:]) or 'saveexec' in name:
                r.add(-2)   # every vector instruction reads exec
            return r, w

        def tok_regs(tok):
            if tok.startswith('vcc'):
                return {-1}
            if tok.startswith('exec'):
                return {-2}
            r = reg_range(tok)
            return set(range(r[1], r[2] + 1)) if r and r[0] == 's' else set()

        def partner(i):
            """Index of the other half's instruction that may be fused with i, or None."""
            name, v, k, tok = info[i]
            half = self._half(tok)
            if not half:
                return None
            pair, h = half
            want_k = k + 1 if h == 0 else k - 1
            regs = tok_regs(tok) | tok_regs(pair)
            slots = {slot[(v, k)], slot.get((v, want_k), -9)}
            for j in range(i + 1, len(self.ins)):
                off, n, nm, asm = self.ins[j]
                if off in targets:
                    return None
                if j in info and info[j][0] == name and info[j][1] == v and info[j][2] == want_k:
                    other = self._half(info[j][3])
                    return j if other == (pair, 1 - h) else None
                if j in info:
                    jn, jv, jk, jt = info[j]
                    if slot[(jv, jk)] in slots:
                        return None
                    r, w = (tok_regs(jt), set()) if jn == 'v_writelane_b32' else (set(), tok_regs(jt))
                else:
                    r, w = regs_of(nm, asm)
                if w & regs or (name == 'v_readlane_b32' and r & regs):
                    return None
                if nm.startswith('s_cbranch') or nm in ('s_branch', 's_endpgm', 's_setpc_b64'):
                    return None
            return None

        new = list(self.ins)
        done = set()
        for i in idx:
            if i in done:
                continue
            name, v, k, tok = info[i]
            off, n = self.ins[i][0], self.ins[i][1]
            j = partner(i)
            if j is not None:
                pair, h = self._half(tok)
                lo_k = k if h == 0 else k - 1
                spair = 's[%d:%d]' % (slot[(v, lo_k)], slot[(v, lo_k)] + 1)
                asm = 's_mov_b64 %s, %s' % ((spair, pair) if name == 'v_writelane_b32' else (pair, spair))
                new[i] = (off, n, 's_nop', 's_nop 0')
                new[j] = (self.ins[j][0], self.ins[j][1], 's_mov_b64', asm)
                done |= {i, j}
                continue
            s = 's%d' % slot[(v, k)]
            if name == 'v_writelane_b32':
                if tok.startswith(('vcc', 'exec')) or (reg_range(tok) and reg_range(tok)[0] != 's'):
                    raise Unsupported('v_writelane_b32 of %s alone at %#x (half of a 64-bit pair)' % (tok, off))
                new[i] = (off, n, 's_mov_b32', 's_mov_b32 %s, %s' % (s, tok))
            else:
                r = reg_range(tok)
                if not r or r[0] != 's':
                    raise Unsupported('v_readlane_b32 into %s alone at %#x (half of a 64-bit pair)' % (tok, off))
                new[i] = (off, n, 's_mov_b32', 's_mov_b32 %s, %s' % (tok, s))
            done.add(i)
        self.ins = new
        slots = {s for (v, k), s in slot.items()}
        entry = self.liveness().get(self.ins[0][0], frozenset()) if self.ins else frozenset()
        unset = sorted(r for r in entry if isinstance(r, int) and r >= SLOT_BASE)
        if unset:
            j, k = divmod(unset[0] - SLOT_BASE, 64)
            raise Unsupported('v_readlane_b32 of v%d lane %d, which no v_writelane_b32 writes on some path to it'
                              % (order[j], k))
        return set(vgprs), slots
    # ---- loops: the state at a loop head is the join of every edge into it -----------------------------------
    # The base scan is linear and keeps, at a loop head, the state of the instruction above it: the back edges are
    # never seen (a constant before the loop was folded into the loop body: `s0 = 0; loop: s0 += 1` became
    # `s0 = 0u + 1u` on every iteration), and after an s_branch that state belongs to no path at all. Here the
    # stage is translated again until the state used at each loop head equals the join of its entry edges and
    # its back edges from the pass before (MeshPipeline retries on LoopRetry). A head no entry reaches in a pass
    # (entered through a forward branch into the body) is `bottom`: what is translated from it is not used and
    # its edges are left out of joins, and the next pass seeds it with the back edges.
    def loop_prepare(self):
        heads = self.loop_heads()
        self._loop_edges = {}   # head -> offsets of the branches to it
        targets = set()
        for off, _, name, asm in self.ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                t = int(parse(asm)[1][0].split('_')[1], 16)
                targets.add(t)
                if t in heads:
                    self._loop_edges.setdefault(t, []).append(off)
        # Unreachable alignment NOPs after a jump do not introduce another
        # entry edge into the following loop head.
        self._loop_fallthrough = set()
        falls_through = False
        for off, _, name, _ in self.ins:
            if falls_through and off in heads:
                self._loop_fallthrough.add(off)
            reachable = falls_through or off in targets or off == self.ins[0][0]
            falls_through = reachable and name not in ('s_branch', 's_endpgm')
        self._loop_branch = {o for v in self._loop_edges.values() for o in v}
        self._loop_after = {}   # branch offset -> loop state after it (this pass)
        self._loop_entry = {}   # head -> loop states of its entries (forward edges, fall-through), this pass
        self._loop_used = {}    # head -> the joined state used at the head, this pass
        self._loop_bottom = False

    def loop_state(self):
        return (self.snapshot(), frozenset(self.tainted), dict(self.au_scalars))

    def loop_join(self, states):
        """Sets the stage to the join of loop states (snapshot, tainted, au_scalars)."""
        self.merge([s[0] for s in states])
        self.tainted = set().union(*(s[1] for s in states))
        # values known uniform only over the active lanes are not followed around a loop: lane-dependent
        self.au = {}
        self.lineage_reset()
        stale = {r for s in states for r in s[2]}
        current = states[0][2]
        self.au_scalars = {r: (current[r] if all(s[2].get(r) == current.get(r) for s in states) else -1)
                           for r in stale}
        self.lane_s = set(range(128))   # the base marks every SGPR suspect at a loop head; keep it so

    @staticmethod
    def loop_key(state):
        snap, tainted, au_scalars = state
        return (snap[0], snap[1], snap[2], tainted, tuple(sorted(au_scalars.items())))

    def loop_one(self, name, asm):
        head = self.pc if self.pc in self._loop_edges else None
        if head is not None:
            entries = [self._loop_after[o] for o in self._loop_edges[head] if o < head and o in self._loop_after]
            if head in self._loop_fallthrough and BOTTOM not in self.sym:
                entries.append(self.loop_state())
            entries = [s for s in entries if BOTTOM not in s[0][0]]
            self._loop_entry[head] = entries
            seeds = [s for s in self.pipe.loop_seed.get(head, []) if BOTTOM not in s[0][0]]
            if entries or seeds:
                self.loop_join(entries + seeds)
            else:
                self.sym[BOTTOM] = True
                self._loop_bottom = True
            self._loop_used[head] = self.loop_state()
        if BOTTOM in self.sym:
            # not reached by any path in this pass: translated only to carry on the scan
            try:
                term = MeshStage.one(self, name, asm)
            except Unsupported:
                term = None
        else:
            term = MeshStage.one(self, name, asm)
        if self.pc in self._loop_branch:
            self._loop_after[self.pc] = self.loop_state()
        return term

    def loop_settled(self):
        """Raises LoopRetry with the back edges of this pass when some loop head's state was not the join."""
        backs = {h: [self._loop_after[o] for o in srcs if o >= h and o in self._loop_after]
                 for h, srcs in self._loop_edges.items()}
        saved = self.loop_state()
        settled = not self._loop_bottom
        for head, used in self._loop_used.items():
            states = [s for s in self._loop_entry.get(head, []) + backs[head] if BOTTOM not in s[0][0]]
            if not states:
                settled = False
                continue
            self.loop_join(states)
            if self.loop_key(self.loop_state()) != self.loop_key(used):
                settled = False
        self.loop_join([saved])
        if not settled:
            raise LoopRetry(backs)

    def next_offset(self):
        for off, n, _, _ in self.ins:
            if off == self.pc:
                return off + n
        return None

    def loop_heads(self):
        if getattr(self, '_heads', None) is None:
            self._heads = set()
            for off, _, name, asm in self.ins:
                if name.startswith('s_cbranch') or name == 's_branch':
                    t = int(parse(asm)[1][0].split('_')[1], 16)
                    if t <= off:
                        self._heads.add(t)
            # a waterfall is translated as one pass per lane (no loop): its head takes no loop join
            self._heads -= {h for h, info in self.waterfalls().items() if not info['refuse']}
        return self._heads

    # ---- waterfall loops: one pass per lane with the lane's own value ----------------------------------------
    # The compiler's idiom for a resource index that differs between lanes (LLVM AMDGPU SIInstrInfo
    # loadMBUFScalarOperandsFromVGPR / emitLoadScalarOpsFromVGPRLoop, "waterfall loop"):
    #     s_mov_b64 s[a], exec; s_mov_b64 s[m], exec          (copies of exec before the loop)
    #  H: v_readfirstlane_b32 sX, vY                           (the first remaining lane's value)
    #     v_cmpx_eq_u32/i32 sX, vY                             (exec = the remaining lanes holding that value)
    #     ... body using sX as a uniform value ...
    #     s_andn2_wrexec_b64 s[m], s[m]                        (m &= ~exec; exec = m; SCC = m != 0)
    #     s_cbranch_scc1 H                                     (again while lanes remain)
    #     s_mov_b64 exec, s[a]                                 (restore)
    # Each lane that is active at H runs the body exactly once, with exec holding it (and the other lanes of its
    # value), and sX equal to its own vY. In the one-lane model that is: sX = vY, the body under the lane's entry
    # exec bit, then on: s[m] = 0, SCC = 0, exec back to s[a] (RDNA 2 ISA S_ANDN2_WREXEC_B64, V_CMPX_EQ_U32,
    # V_READFIRSTLANE_B32). The hardware leaves sX, the body's scalar results and vcc with the LAST pass's values;
    # here every lane holds its own, so such a register must not be read after the loop (refused).
    def waterfalls(self):
        """head -> {role offset: role} plus 'refuse' (a reason) for each loop that has the waterfall's shape."""
        if getattr(self, '_wf', None) is not None:
            return self._wf
        self._wf = {}
        if not WATERFALL:
            return self._wf
        ins = self.ins
        index = {x[0]: i for i, x in enumerate(ins)}
        sources = {}
        for off, _, name, asm in ins:
            if name.startswith('s_cbranch') or name == 's_branch':
                sources.setdefault(int(parse(asm)[1][0].split('_')[1], 16), []).append(off)
        skip = ('s_nop', 's_waitcnt', 's_waitcnt_vmcnt', 's_waitcnt_lgkmcnt', 's_waitcnt_depctr')
        for i, (off, _, name, asm) in enumerate(ins):
            if name != 's_cbranch_scc1':
                continue
            head = int(parse(asm)[1][0].split('_')[1], 16)
            if head > off or head not in index or sources.get(head) != [off]:
                continue
            j = index[head]
            k = j
            while k < i and ins[k][2] in skip:
                k += 1
            if k + 2 >= i - 1 or ins[k][2] != 'v_readfirstlane_b32':
                continue
            rops = parse(ins[k][3])[1]
            cname, cops = ins[k + 1][2], parse(ins[k + 1][3])[1]
            wname, wops = ins[i - 1][2], parse(ins[i - 1][3])[1]
            sx, vy = reg_range(rops[0]), reg_range(rops[1])
            if not (sx and sx[0] == 's' and vy and vy[0] == 'v' and sx[2] == sx[1] and vy[2] == vy[1]):
                continue
            if not (re.fullmatch(r'v_cmpx_eq_[ui]32_e32', cname) and sorted(cops) == sorted(rops)):
                continue
            mr = reg_range(wops[0]) if wops else None
            if not (wname == 's_andn2_wrexec_b64' and len(wops) == 2 and wops[0] == wops[1] and mr and mr[0] == 's'):
                continue
            m = mr[1]
            info = {'head': head, 'rfl': ins[k][0], 'cmpx': ins[k + 1][0], 'wrexec': ins[i - 1][0], 'branch': off,
                    's': sx[1], 'v': vy[1], 'm': m}
            self._wf[head] = info
            info['refuse'] = self.waterfall_check(info, ins, index, sources, i, j, k)
        return self._wf

    def waterfall_check(self, info, ins, index, sources, i, j, k):
        """None when the matched loop is translatable as one pass per lane, else the reason."""
        m, vy = info['m'], info['v']
        body = ins[k + 2:i - 1]
        for off, _, name, asm in body:
            if WATERFALL_BODY_REFUSED.match(name) or self.exec_written(name, parse(asm)[1]):
                return '%s at %#x' % (name, off)
            if any(t in ('exec', 'exec_lo', 'exec_hi') for t in parse(asm)[1]):
                return 'exec read at %#x (the pass\'s lanes, not one lane)' % off
            if off in sources:
                return 'branch target %#x inside it' % off
            _, w, _, _ = reads_writes(name, asm)
            if w & {m, m + 1}:
                return 'the remaining-lanes mask s%d written at %#x' % (m, off)
            ops = parse(asm)[1]
            if name.startswith(('v_', 'buffer_', 'image_', 'global_', 'flat_', 'scratch_')) and ops:
                d = reg_range(ops[0])
                if d and d[0] == 'v' and d[1] <= vy <= d[2]:
                    return 'v%d (the index) written at %#x' % (vy, off)
        # before the head: s[m] = exec, and a copy s[a] = exec that the instruction after the loop restores, with
        # no exec write and no other way in between
        nxt = i + 1
        while nxt < len(ins) and ins[nxt][2] in ('s_nop',):
            nxt += 1
        if nxt >= len(ins):
            return 'no exec restore after it'
        rname, rops = ins[nxt][2], parse(ins[nxt][3])[1]
        ar = reg_range(rops[1]) if rname == 's_mov_b64' and len(rops) == 2 and rops[0] == 'exec' else None
        if not (ar and ar[0] == 's'):
            return 'no s_mov_b64 exec restore after it (%s)' % ins[nxt][3]
        a = ar[1]
        found = set()
        p = j - 1
        while p >= 0 and found != {'m', 'a'}:
            off, _, name, asm = ins[p]
            ops = parse(asm)[1]
            if ins[p + 1][0] in sources and ins[p + 1][0] != info['head']:
                return 'branch target %#x between the exec copies and the head' % ins[p + 1][0]
            if name == 's_mov_b64' and len(ops) == 2 and ops[1] == 'exec':
                r = reg_range(ops[0])
                if r and r[1] == m:
                    found.add('m')
                elif r and r[1] == a:
                    found.add('a')
                p -= 1
                continue
            if self.exec_written(name, ops) or name.startswith(('s_cbranch', 's_branch', 's_endpgm')):
                return 'exec written or a branch at %#x before the head' % off
            _, w, _, _ = reads_writes(name, asm)
            if w & {m, m + 1, a, a + 1}:
                return 'an exec copy overwritten at %#x' % off
            p -= 1
        if found != {'m', 'a'}:
            return 'no exec copies before the head'
        for off, _, name, asm in ins[j:nxt]:
            _, w, _, _ = reads_writes(name, asm)
            if w & {a, a + 1}:
                return 'the restored copy s%d written at %#x' % (a, off)
        # what the body leaves in scalar registers (each lane's own here, the last pass's on the hardware)
        written = set()
        for off, _, name, asm in ins[k:i - 1]:
            _, w, wv, _ = reads_writes(name, asm)
            written |= w | ({-1} if wv else set())
        live = self.liveness()[ins[nxt][0]] - {m, m + 1}
        hit = written & live
        if hit:
            return '%s written in it is live after it' % ('vcc' if min(hit) == -1 else 's%d' % min(hit))
        info['restore'] = ins[nxt][0]
        return None

    def waterfall_role(self):
        """(info, role) of the instruction at self.pc in a waterfall loop, else None."""
        if getattr(self, '_wf_roles', None) is None:
            self._wf_roles = {}
            for info in self.waterfalls().values():
                for role in ('rfl', 'cmpx', 'wrexec', 'branch'):
                    self._wf_roles[info[role]] = (info, role)
        return self._wf_roles.get(self.pc)

    def waterfall_step(self, info, role, ops):
        """The translation of a waterfall instruction; NotImplemented: translate it as any other instruction (a
        refused waterfall stays a loop, as before, unless its index is lane-dependent: then the reason is given)."""
        if info['refuse']:
            if role == 'rfl' and self.lane_dependent('v%d' % info['v']):
                raise Unsupported('waterfall loop %#x-%#x: %s' % (info['head'], info['branch'], info['refuse']))
            return NotImplemented
        if role == 'rfl':
            # this lane's pass: the value is its own (threads with exec 0 compute it too; their body is guarded)
            self.set_s(info['s'], self.u(ops[1]))
            self.au_scalars.pop(info['s'], None)
            return None
        if role == 'cmpx':
            return None          # sX == vY in this lane: exec keeps its bit
        if role == 'wrexec':
            # the lane's pass is over: s[m] holds no lane any more (all passes done), SCC = 0
            self.set_s(info['m'], '0u', sym=('const', 0))
            self.set_s(info['m'] + 1, '0u', sym=('const', 0))
            self.emit('scc = false;')
            self.scc_from.pop('scc', None)
            return None
        return None              # the back edge: no further pass for this lane

    def perlane_loop(self, head, off):
        """None when the loop head..off (an scc1 back edge on a lane-mask SCC) is translatable per lane, else the
        reason."""
        cache = self.__dict__.setdefault('_perlane', {})
        if off in cache:
            return cache[off]
        ins = self.ins
        index = {x[0]: i for i, x in enumerate(ins)}
        i, j = index[off], index.get(head)
        why = None
        prev = parse(ins[i - 1][3])[1] if i else []
        pname = ins[i - 1][2] if i else ''
        if ins[i][2] != 's_cbranch_scc1':
            why = '%s back edge' % ins[i][2]
        elif not ((pname == 's_andn2_b64' and len(prev) == 3 and prev[0] == prev[1] and prev[2] == 'exec') or
                  (pname == 's_andn2_wrexec_b64' and len(prev) == 2 and prev[0] == prev[1])):
            why = 'SCC not from the remaining-lanes mask (%s)' % ins[i - 1][3]
        mr = reg_range(prev[0]) if prev else None
        if why is None and not (mr and mr[0] == 's'):
            why = 'mask %s' % (prev[0] if prev else '?')
        if why is None:
            m = mr[1]
            written = set()
            for o, _, name, asm in ins[j:i + 1]:
                ops = parse(asm)[1]
                if PERLANE_BODY_REFUSED.match(name):
                    why = '%s at %#x' % (name, o)
                    break
                # the first active lane differs between a thread's iteration and the wave's: only a translated
                # waterfall (each lane's own value) may read one
                if name == 'v_readfirstlane_b32' and not any(
                        info['rfl'] == o and not info['refuse'] for info in self.waterfalls().values()):
                    why = 'v_readfirstlane_b32 at %#x' % o
                    break
                if name.startswith('s_cbranch') or name == 's_branch':
                    t = int(ops[0].split('_')[1], 16)
                    if not head <= t <= off + 4:
                        why = 'a branch out of the loop at %#x' % o
                        break
                _, w, wv, _ = reads_writes(name, asm)
                written |= w | ({-1} if wv else set())
            nxt = i + 1
            while why is None and nxt < len(ins) and ins[nxt][2] == 's_nop':
                nxt += 1
            if why is None:
                rops = parse(ins[nxt][3])[1] if nxt < len(ins) else []
                ar = reg_range(rops[1]) if (nxt < len(ins) and ins[nxt][2] == 's_mov_b64' and len(rops) == 2
                                            and rops[0] == 'exec') else None
                if not (ar and ar[0] == 's'):
                    why = 'no s_mov_b64 exec restore after it'
                elif written & {ar[1], ar[1] + 1}:
                    why = 'the restored exec copy s%d written in it' % ar[1]
            if why is None:
                hit = written & (self.liveness()[ins[nxt][0]] - {m, m + 1})
                if hit:
                    why = '%s written in it is live after it' % ('vcc' if min(hit) == -1 else 's%d' % min(hit))
        cache[off] = why
        return why

    def branch_targets(self):
        if getattr(self, '_targets', None) is None:
            self._targets = {int(parse(asm)[1][0].split('_')[1], 16) for _, _, name, asm in self.ins
                             if name.startswith('s_cbranch') or name == 's_branch'}
        return self._targets

    def body(self):
        """With barriers: the program runs as barriers + 1 phases, written out one after the other with a group
        barrier between them in straight-line code (no barrier inside a loop). barrier_count() put every block in
        exactly one phase; within a phase a thread runs its blocks until it reaches a barrier (pc gets
        0x10000 | the block after it) or the end (0x20000). Each console barrier is thus the phase boundary every
        wave of the group reaches."""
        if not any(b[2] and b[2][0] == 'barrier' for b in self.blocks):
            return super().body()
        index = {b[0]: i for i, b in enumerate(self.blocks)}

        def goto(off):
            if off not in index:
                raise Unsupported('branch into the middle of a block (+0x%x)' % off)
            return index[off]
        out = ['    uint pc = 0;']
        for phase in range(self.barriers + 1):
            members = [i for i in range(len(self.blocks)) if self.phase_of.get(i) == phase]
            loop = any(t and t[0] in ('jump', 'branch') and goto(t[-1]) <= i
                       for i in members for t in [self.blocks[i][2]])
            out.append('    // phase %d' % phase)
            ind = '    '
            if loop:
                out.append('    [loop] for (uint guard = 0; guard < 65536 && pc < 0x10000u; ++guard) {')
                ind = '        '
            for k, i in enumerate(members):
                off, lines, term = self.blocks[i]
                if loop and term and term[0] in ('jump', 'branch') and self.phase_of.get(goto(term[-1])) != phase:
                    # (a group-uniform skip into a later phase: the loop would spin on that pc until its guard)
                    raise Unsupported('jump from a looping phase into another phase at %#x' % int(off))
                out.append('%s%sif (pc == %d) {' % (ind, 'else ' if loop and k else '', i))
                out += [ind + '    ' + line for line in lines]
                last = i + 1 >= len(self.blocks)
                nxt = '0x20000u' if last else '%d' % (i + 1)
                if not term:
                    out.append('%s    pc = %s;' % (ind, nxt))
                elif term[0] == 'end':
                    out.append('%s    pc = 0x20000u;' % ind)
                elif term[0] == 'barrier':
                    out.append('%s    pc = %s;' % (ind, '0x20000u' if last else '0x10000u | %d' % (i + 1)))
                elif term[0] == 'jump':
                    out.append('%s    pc = %d;' % (ind, goto(term[1])))
                else:
                    out.append('%s    pc = (%s) ? %d : %s;' % (ind, term[1], goto(term[2]), nxt))
                out.append('%s}' % ind)
            if loop:
                out.append('    }')
            if phase < self.barriers:
                out += ['    GroupMemoryBarrierWithGroupSync();',
                        '    if (pc < 0x20000u) pc &= 0xffffu;']
        return out

    def barrier_count(self):
        """Every path from the entry to s_endpgm must cross the same number of barriers (the hardware needs every
        wave to arrive at each barrier). Returns that number; the body runs as that many + 1 phases."""
        blocks = self.blocks
        index = {b[0]: i for i, b in enumerate(blocks)}
        memo = {}
        cmemo = {}

        def console(i, depth=0):
            """Sorted counts of console barriers (not lane-mask SCC publishes) from block i to the end."""
            if i in cmemo:
                return cmemo[i]
            if depth > 10000:
                raise Unsupported('block graph too deep')
            cmemo[i] = ()
            off, lines, term = blocks[i]
            # only the program's s_barrier: the translation's own barriers (lane-mask SCC publish, readfirstlane
            # broadcast, group-branch publish) are reached or skipped by the whole group together
            here = 1 if term and term[0] == 'barrier' and term[1:2] == ('console',) and SKIP_ADDED else 0
            if term and term[0] == 'barrier' and not SKIP_ADDED and term[1:2] != ('lane-scc',):
                here = 1
            nxt = [i + 1] if (term is None or term[0] in ('barrier', 'branch')) and i + 1 < len(blocks) else []
            if term and term[0] in ('jump', 'branch'):
                if term[0] == 'jump':
                    nxt = []
                if term[-1] > off:
                    nxt.append(index[term[-1]])
            got = sorted({here + c for k in nxt for c in console(k, depth + 1)}) if nxt else [here]
            cmemo[i] = tuple(got)
            return cmemo[i]

        def counts(i, depth=0):
            if i in memo:
                return memo[i]
            if depth > 10000:
                raise Unsupported('block graph too deep')
            off, lines, term = blocks[i]
            here = 1 if term and term[0] == 'barrier' else 0
            if off in self.uniform_skips:
                # group-uniform skip of an execz range (group_branch): every thread of the group goes the same
                # way, so the skip may jump over barriers inside the range (body() runs the target in its later
                # phase; the threads still pass every phase barrier in between, doing nothing)
                through, skip = counts(i + 1, depth + 1), counts(index[term[-1]], depth + 1)
                if len(through) != 1 or len(skip) != 1 or min(skip) > min(through):
                    raise Unsupported('group-uniform execz range at %#x: barrier counts %s / %s'
                                      % (int(off), sorted(through), sorted(skip)))
                # only barriers this translation added (lane-mask SCC publishes) may be skipped: a console
                # s_barrier inside the range would be skipped by a wave without active lanes on the hardware
                if console(i + 1) != console(index[term[-1]]):
                    raise Unsupported('s_barrier inside a group-uniform execz range at %#x (console barriers ahead %s '
                                      'through, %s skipping)' % (int(off), list(console(i + 1)),
                                                                list(console(index[term[-1]]))))
                memo[i] = through
                return through
            nxt = []
            if term is None or term[0] in ('barrier', 'branch'):
                if i + 1 < len(blocks):
                    nxt.append(i + 1)
            if term and term[0] in ('jump', 'branch'):
                if term[-1] <= off:
                    if any(blocks[k][2] and blocks[k][2][0] == 'barrier' for k in range(index[term[-1]], i + 1)):
                        raise Unsupported('barrier inside a loop')
                    nxt = [k for k in nxt if k != index[term[-1]]]   # loop back edge: counted once
                else:
                    nxt.append(index[term[-1]])
            result = set()
            for k in nxt:
                result |= {here + c for c in counts(k, depth + 1)}
            if not nxt:
                result = {here}
            memo[i] = result
            return result
        found = counts(0)
        if len(found) != 1:
            raise Unsupported('paths cross different numbers of barriers: %s' % sorted(found))
        total = min(found)
        # Every reachable block then has one count of barriers still ahead (two would give two totals), so it runs
        # in exactly one phase: total - that count.
        self.phase_of = {}
        for i, c in memo.items():
            if len(c) != 1:
                raise Unsupported('block %#x reaches the end across different numbers of barriers' % blocks[i][0])
            self.phase_of[i] = total - min(c)
        return total

    # ---- entry ---------------------------------------------------------------------------------
    def setup_entry(self):
        root = self.pipe.root
        reg = self.pipe.user_sgpr
        for index, p in enumerate(root['params']):
            if p['type'] == 0:
                self.sym[reg] = ('ptr', ('table', index))
                reg += 1
            elif p['type'] == 1:
                for i in range(p['count']):
                    self.sym[reg] = ('ud', self.pipe.ud(('rootconst', index, i)))
                    reg += 1
            else:
                self.sym[reg] = ('ptr', ('root', index))
                self.sym[reg + 1] = ('ptrhi', ('root', index))
                reg += 2
        if root['flags'] & 0x400:
            self.sym[reg] = ('ptr', ('resheap',))
            reg += 1
        if root['flags'] & 0x800:
            self.sym[reg] = ('ptr', ('smpheap',))
            reg += 1
        self.user_end = reg
        # s3: merged_wave_info (bits 27:24 = wave index in the group, Mesa radv_shader_args.c / ac_unpack_param).
        self.sym[3] = ('waveinfo',)

    # ---- taint (lane dependence) ---------------------------------------------------------------
    # Three levels per register: uniform (not in `tainted`: the same in every lane of the wave), active-uniform
    # (in `tainted` and au[reg] == epoch: the same in every lane active since the last widening of exec; e.g. a
    # value computed from uniform values under a partial exec, or `exec & uniform`), and lane-dependent.
    # `epoch` counts exec writes that can turn lanes back on (restores); narrowing writes (v_cmpx, `exec & x`, any
    # write while every lane was active) keep active-uniform values valid.
    def emit(self, text):
        self.lines.append(text)
        for stmt in text.split(';'):
            m = SIMPLE.match(stmt)
            if not m:
                continue
            target, expr = m.group(1), m.group(2)
            deps = set(TOKEN.findall(expr))
            if stmt.strip().startswith('if (exec & 1)'):
                deps.add('exec')
            origins = frozenset().union(*(self.scc_from.get(d, ()) for d in deps))
            if stmt.strip().startswith('if '):
                origins |= self.scc_from.get(target, frozenset())   # a guarded write keeps the old value elsewhere
            if origins:
                self.scc_from[target] = origins
            else:
                self.scc_from.pop(target, None)
            if target == 'exec':
                narrow = expr.startswith('(exec & ') or getattr(self, '_known_at_start', True)
                if not narrow:
                    self.epoch += 1
                self.xs = self.lineage_exec(expr)
                self._xs_exec_seen = True
            elif self.LINEAGE_COPY.match(expr.strip()) and not stmt.strip().startswith('if '):
                self.xs_copy[target] = self.xs
            else:
                self.xs_copy.pop(target, None)
            # (a scalar computed inside a per-lane region from uniform values is what the wave computes there: the
            # lanes that skip the region never see it, and check_skipped_scalars refuses it being live after)
            if not deps & self.tainted:
                self.tainted.discard(target)
                self.au.pop(target, None)
                self.au_xs.pop(target, None)
                continue
            # (the sources as they were before this statement: `s12 = scc ? s12 : s24` reads the old, untainted s12)
            lane = {d for d in deps & self.tainted
                    if d != 'exec' and self.au.get(d) != self.epoch and not self.lineage_ok(d)}
            self.tainted.add(target)
            if lane or target == 'exec':
                self.au.pop(target, None)
                self.au_xs.pop(target, None)
            else:
                self.au[target] = self.epoch
                self.xs_serial += 1
                self.au_xs[target] = (self.xs, self.xs_serial)

    def is_tainted(self, expr):
        return bool(set(TOKEN.findall(expr)) & self.tainted)

    def lane_dependent(self, name):
        """Neither uniform nor uniform over the active lanes."""
        return name in self.tainted and self.au.get(name) != self.epoch and not self.lineage_ok(name)

    # ---- exec lineage ------------------------------------------------------------------------------
    # The epoch rule above forgets every active-uniform value at the first exec write that may widen exec. The
    # compiler's usual shape restores a SAVED exec instead: `s[26:27] = exec ... exec = s26 & x (narrow) ... exec =
    # s26 (restore) ... exec = s26 & y`. A value written from uniform values under E is the same in every lane of E;
    # every later exec that is E or a subset of E (a narrowing of E or of a restore of E) has only such lanes active.
    # States: 0 at entry (all lanes), a child per narrowing, the saved state per restore of an unchanged copy, a
    # fresh root for any other exec write, at loop heads, and where joined paths have no common ancestor.
    def lineage_new(self, parent):
        n = len(self.xs_parent)
        self.xs_parent[n] = parent
        return n

    def lineage_reset(self):
        self.xs = self.lineage_new(None)
        self.au_xs, self.xs_copy = {}, {}

    def lineage_chain(self, x):
        out = []
        while x is not None:
            out.append(x)
            x = self.xs_parent.get(x)   # (a loop seed's state from an earlier pass: loop_join resets after it)
        return out

    def lineage_ok(self, name):
        """name's active-uniform value was written under the current exec state or an ancestor of it."""
        got = self.au_xs.get(name) if EXEC_LINEAGE else None
        return got is not None and got[0] in self.lineage_chain(self.xs)

    LINEAGE_COPY = re.compile(r'^\(?exec\)?(?: & 1)?$')
    LINEAGE_INNER = re.compile(r'^\((.+)\) & 1$')

    def lineage_exec(self, expr):
        """The exec state after `exec = expr` (the one-lane forms: `(s26) & 1`, `(s18 & exec) & 1`,
        `(exec & xsave) & 1`, `(s18 & ~exec) & 1`)."""
        e = expr.strip()
        m = self.LINEAGE_INNER.match(e)
        terms = [t.strip() for t in m.group(1).split(' & ')] if m else [e]
        if len(terms) == 1 and terms[0] in self.xs_copy:
            return self.xs_copy[terms[0]]              # a restore of an unchanged copy
        if len(terms) >= 2 and all(re.fullmatch(r'~?\w+', t) for t in terms):
            if 'exec' in terms:
                return self.lineage_new(self.xs)       # exec & x: a narrowing of the current state
            for t in terms:
                if t in self.xs_copy:
                    return self.lineage_new(self.xs_copy[t])   # copy & x (x may be ~exec): inside the copy's lanes
        return self.lineage_new(None)

    def lineage_merge(self, states):
        """Join of the lineage fields (xs, au_xs, xs_copy) of the incoming paths."""
        chains = [self.lineage_chain(s[0]) for s in states]
        common = [x for x in chains[0] if all(x in c for c in chains[1:])]
        self.xs = common[0] if common else self.lineage_new(None)
        first = states[0]
        self.xs_copy = {r: v for r, v in first[2].items() if all(s[2].get(r) == v for s in states)}
        self.au_xs = {r: v for r, v in first[1].items() if all(s[1].get(r) == v for s in states)}

    # ---- instructions --------------------------------------------------------------------------
    def one(self, name, asm):
        """one_body, plus the lane-mask SCC provenance (scc_from) carried along branch edges: the scan is linear,
        so a branch target gets the union of the states at the branches to it (and at its fall-through when the
        previous instruction falls through). Backward edges are not followed (a missed origin only refuses)."""
        if self.pc in self.join_from:
            saved = self.join_from.pop(self.pc)
            if self.fell_off:
                self.scc_from = dict(saved)
            else:
                for k, v in saved.items():
                    self.scc_from[k] = self.scc_from.get(k, frozenset()) | v
        self._xs_exec_seen = False
        self._dw_keep = set()
        before = len(self.lines)
        term = self.one_body(name, asm)
        if 'descword' in MESH_FEATURES:
            # an SGPR this instruction wrote other than by an in-place descriptor edit (dw_mark_edit) holds a new value
            for r in G.sgpr_access(name, asm)[1] - self._dw_keep:
                self.sym.pop(self.dw_edit_key(r), None)
        if not self._xs_exec_seen and (self.exec_written(name, parse(asm)[1]) or
                                       any(re.search(r'exec = ', l) for l in self.lines[before:])):
            self.xs = self.lineage_new(None)   # an exec write emit() did not see: no lineage across it
        self.fell_off = bool(term) and term[0] in ('jump', 'end')
        target = None
        if term and term[0] in ('jump', 'branch'):
            target = term[-1]
        elif term and term[0] == 'barrier' and len(term) > 3:
            target = term[3]      # group-uniform execz: the skip edge
        if target is not None and target > self.pc:
            into = self.join_from.setdefault(target, {})
            for k, v in self.scc_from.items():
                into[k] = into.get(k, frozenset()) | v
        return term

    def one_body(self, name, asm):
        _, ops, mods = parse(asm)
        base = re.sub(r'_e(32|64)$', '', name)
        if self.pending:
            self.lines.extend(self.pending)
            self.pending = []
        if base == 's_cbranch_execz':
            self._execz_edge = True   # the snapshot of this branch's taken edge (base translate) is an execz edge
        if self.pc in self.loop_heads():
            self.exec_off = set()
        for r, before, at in self.group_checks.get(self.pc, ()):
            # A merge that left ('conflict',) is refused by any later use (exec_off_use: an exec-masked mask op
            # excepted), so only another change is refused here.
            if self.sym.get(r) != before and not (EXEC_OFF and self.sym.get(r) == ('conflict',)):
                raise Unsupported('execz range %#x-%#x changes s%d from %s to %s' % (at, self.pc, r, before[0], self.sym.get(r)))
        if self.pc in self.branch_targets():
            self.shadow.clear()      # another path may arrive here without the shadow assignments
        if self.pc >= self.divergent_until or self.pc in self.loop_heads():
            self.running_active = False
        if self.pc in self.loop_heads():
            # the linear scan reaches a loop head with the state from before the loop; a later iteration can
            # arrive with any exec
            self.exec_static = None
            self.static_copies.clear()
            self.lineage_reset()     # a back edge may arrive with any exec
            # the base marks s0..s127 lane-suspect at a loop head; spill slots are scalars too
            self.lane_s |= self.spill_slots
        if base in ('s_waitcnt_depctr',):
            return None
        if base.startswith('s_cbranch_exec') and self.exec_static is not None:
            # exec is a known mask (the same in every wave): the wave-level branch is decided here
            taken = (self.exec_static == 0) == (base == 's_cbranch_execz')
            return ('jump', int(ops[0].split('_')[1], 16)) if taken else None
        if base.startswith('ds_'):
            return self.lds(base, ops, mods)
        if base in FUSED or base in UNFUSED:
            # the multiply-add rule at FMA_MODE: fused ops exact (x2w_fma) or precise; unfused ops always precise
            arity, _ = G.VEC_F[base]
            args = [self.f(o) for o in ops[1:1 + arity]]
            if base in ('v_fmac_f32', 'v_mac_f32'):
                args.append(self.f(ops[0]))
            helper = 'x2w_fma' if base in FUSED and EXACT_FMA else 'x2w_madp'
            return self.set_vf(reg_range(ops[0])[1], '%s(%s, %s, %s)' % ((helper,) + tuple(args)), mods)
        if base == 's_sendmsg':
            if ops != ['0x9']:
                raise Unsupported('s_sendmsg %s' % ops)
            self.alloc_seen = True
            if self.au_scalars.get('m0', self.epoch) != self.epoch:
                raise Unsupported('m0 (from v_readfirstlane under a partial exec) sent after exec was widened')
            # Wave-level message: the wave sends it when it reaches it. Lanes active here all hold the same m0.
            self.emit('if (exec & 1) x2w_alloc = m0;')
            self.alloc_guarded = True
            return None
        if base == 's_barrier':
            if not self.has_lds:
                # A barrier orders LDS (and memory) traffic between the waves; a program without LDS instructions
                # writes nothing another wave reads here (the archive form's barriers sit in per-lane regions
                # around the allocation, which the one-lane model could not place anyway)
                return None
            # s_barrier runs whatever exec is; a lane that skipped it per lane would arrive at a different one
            if self.pc < self.divergent_until:
                raise Unsupported('s_barrier inside a per-lane skipped range at %#x' % self.pc)
            self.lds_written_in_phase = False
            return ('barrier', 'console')   # the program's own barrier (others are added by the translation)
        if base in ('s_waitcnt_vmcnt', 's_waitcnt_lgkmcnt', 's_waitcnt_expcnt', 's_waitcnt_vscnt'):
            return None
        # Every scalar destination stops being a lane mask unless the handler marks it again (the base
        # translator only updates the mask set in its ALU paths, so a load into a former mask register kept
        # the mark and a later s_or of it was cut to bit 0).
        reads, writes, _, _ = reads_writes(name, asm)
        masks_before = set(self.masks)
        for r in writes:
            if isinstance(r, int):
                # a copy is named by its low SGPR: a write to either half ends it (a copy this instruction makes is
                # entered again by emit)
                self.xs_copy.pop('s%d' % r, None)
                self.xs_copy.pop('s%d' % (r - 1), None)
        self.masks -= writes
        if base == 's_sendmsg' or 'm0' in ops[1:]:
            reads = reads | {'m0'}
        if ops and ops[0] == 'm0':
            writes = writes | {'m0'}
        # A scalar read from the first active lane while some lanes were off holds, in the one-lane model, each
        # lane's own value: right in the lanes that were active, stale in the others. Once exec has been widened
        # those other lanes run again, so such a register must not be read any more.
        stale = [r for r in reads if r in self.au_scalars and self.au_scalars[r] != self.epoch]
        if stale:
            raise Unsupported('%s (from v_readfirstlane under a partial exec) read after exec was widened at %#x'
                              % (stale[0], self.pc))
        derived = any(r in self.au_scalars for r in reads)
        for r in writes:
            if derived and base.startswith('s_'):
                self.au_scalars[r] = self.epoch   # computed from such a value: the same caveat
            else:
                self.au_scalars.pop(r, None)
        self._known_at_start = self.exec_known
        self._static_set = False
        for r in writes:
            self.static_copies.pop(r, None)
            if isinstance(r, int):
                self.static_copies.pop(r - 1, None)   # the high half of a 64-bit copy
        role = self.waterfall_role()
        if role is not None:
            term = self.waterfall_step(role[0], role[1], ops)
            if term is not NotImplemented:
                return term
        before = len(self.lines)
        term = super().one(name, asm)
        if self.exec_written(name, ops) or any(re.search(r'\bexec = ', l) for l in self.lines[before:]):
            self.exec_off = set()     # lanes may have been turned on: an exec-off register is plain conflict again
        if not self._static_set and (self.exec_written(name, ops) or
                                     any(re.search(r'\bexec = ', l) for l in self.lines[before:])):
            self.exec_static = None   # exec changed in a way not modelled as a constant mask
        if self.exec_written(name, ops) or any(re.search(r'\bexec = ', l) for l in self.lines[before:]):
            self.running_active = False   # threads with exec 0 may run on from here
        mask_read = any(t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or (reg_range(t) and reg_range(t)[0] == 's' and
                                                                          reg_range(t)[1] in masks_before)
                        for t in ops[1:])
        if self.mask_scc(base, ops, masks_before, self.lines[before:]):
            self.scc_from['scc'] = frozenset([self.pc])
            if re.match(r's_and_b(32|64)$', base) and ('exec' in ops[1:] or 'exec_lo' in ops[1:]):
                self.scc_confined.add(self.pc)   # result inside the active lanes (see the per-lane shortcut below)
            if self.pc in getattr(self.pipe, 'scc_publish', ()):
                if term is not None:
                    raise Unsupported('lane-mask SCC at %#x ends a block' % self.pc)
                return self.lane_scc()
        elif (base.startswith('s_') and mask_read and 'scc' in self.tainted
              and any(re.search(r'\bscc = ', l) for l in self.lines[before:])):
            # SCC of a non-mask SALU op on a lane mask (e.g. s_lshl_b32 of vcc_lo): the one-lane bit has no
            # per-wave rule here (an OR of the threads' results is not the op on the 64-bit mask): origin -(pc+1)
            self.scc_from['scc'] = frozenset([-(self.pc + 1)])
        if term and term[0] == 'branch':
            cond = term[1]
            off = self.pc
            target = term[2]
            per_lane = base[10:] in ('execz', 'execnz', 'vccz', 'vccnz')
            lane_cond = self.is_tainted(cond)
            if base[10:] in ('scc0', 'scc1') and lane_cond:
                tokens = TOKEN.findall(cond)
                origins = frozenset().union(*(self.scc_from.get(d, ()) for d in tokens))
                # with lanes off, a lane-mask SCC is the active lanes' value only when its mask result was confined
                # to them (`x & exec`): `saved & ~exec` has 0 in every active lane and the wave's SCC is 1
                confined = self.exec_known or origins <= self.scc_confined or not CONFINED_SCC
                if (LANE_SCC and not any(self.lane_dependent(t) for t in tokens)
                        and (self.exec_known or self.running_active) and confined):
                    # SCC the same in every active lane (e.g. `vcc & exec` with vcc from a uniform compare), and
                    # every thread that runs here is an active lane (all lanes on, or right inside a per-lane
                    # skipped range with no exec write since): each thread's SCC is the wave's, so the per-lane
                    # branch is the wave's branch
                    return term
                if target <= off and PERLANE_LOOPS:
                    why = self.perlane_loop(target, off)
                    if why is None:
                        return term     # per-lane loop: each thread iterates until its own lane leaves
                    raise Unsupported('per-lane loop %#x-%#x: %s' % (target, off, why))
                raise LaneScc('branch on an SCC that depends on lane masks at %#x' % off, origins)
            if per_lane and lane_cond:
                if target <= off:
                    raise Unsupported('per-lane loop branch %s at %#x' % (base, off))
                # vcc must be the same in every active lane (vcc = exec & uniform): inactive lanes then read 0 and
                # take the branch the way execz would, which the skipped-range check below covers
                if base[10:] in ('vccz', 'vccnz') and self.lane_dependent('vcc'):
                    raise Unsupported('%s on a lane-dependent vcc at %#x' % (base, off))
                if base == 's_cbranch_execz' and off in self.group_branches():
                    return self.group_branch(off, target)
                self.check_skipped_scalars(off, target)
                self.divergent_until = max(self.divergent_until, target)
                if base == 's_cbranch_execz':
                    self.running_active = True   # the threads that go on are exactly the active lanes
                elif base[10:] in ('execnz', 'vccnz'):
                    self.running_active = False  # the threads that go on may be inactive lanes
        return term

    def group_branch(self, off, target):
        """s_cbranch_execz over a range that writes a scalar live at its end (or encloses such a range).

        Per lane, the lanes that skip would keep the old scalar where the wave computed a new one. Instead every
        lane publishes its exec bit (x2w_any[k][wave]), a group barrier follows, and the range is skipped only when
        no lane of the GROUP is active; otherwise all 128 threads run it, the inactive lanes with exec 0 (vector
        writes, exports, LDS and buffer stores stay guarded by exec; inner per-lane branches still skip per lane).
        A whole wave with no active lane (which the hardware skips) then gets back, at the end of the range, the
        values the hit scalars had before it. Because every thread of the group runs the range, barriers inside
        it are allowed. Refused: exec writes the per-lane model cannot follow (as before), the branch inside a
        per-lane range, a hit scalar holding a descriptor before the range, and other ways into the target."""
        if off < self.divergent_until or self.exec_known or self.pending:
            raise Unsupported('group-uniform execz at %#x inside a per-lane range' % off)
        hits = self.skipped_scalars(off, target)
        restore = []
        # the end may also be reached by other branches: only a lane that came through this branch restores
        # (x2w_u<seen> = 1 here, cleared at the end)
        seen = 'x2w_u%d' % len(self.shadow_vars)
        self.shadow_vars.append(seen)
        k = len(self.wave_any)
        self.wave_any.append(off)
        for r in sorted(hits):
            reg = 'vcc' if r == -1 else 's%d' % r
            if r != -1 and r not in self.sym and r not in self.used_s:
                continue   # no value before the range: a wave that skips it reads an undefined register
            before = self.sym.get(r) if r != -1 else None
            if before is not None and before[0] not in ('const', 'fixed_buffer_word'):
                # a symbolic value (descriptor, pointer): no run-time copy exists. Right only when the range leaves
                # the same symbolic value (a reload of the same descriptor): checked at the target.
                self.group_checks.setdefault(target, []).append((r, before, off))
                continue
            try:
                value = self.u(reg)
            except Unsupported:
                raise Unsupported('execz range %#x-%#x writes %s, which holds %s before it' % (off, target, reg, before))
            var = 'x2w_u%d' % len(self.shadow_vars)
            self.shadow_vars.append(var)
            self.emit('%s = %s;' % (var, value))
            if r != -1:
                self.used_s.add(r)
            restore.append('if (%s != 0u && x2w_any[%d][wave] == 0u) %s = %s;' % (seen, k, reg, var))
        if restore:
            self.emit('%s = 1u;' % seen)
            restore.append('%s = 0u;' % seen)
        self.emit('if (exec & 1) x2w_any[%d][wave] = 1u;' % k)
        if 'restore' not in TEST_DROP:
            self.group_restore.setdefault(target, []).extend(restore)
        return ('barrier', 'wave-branch', k, target)

    MASK_OPS = ('and', 'or', 'xor', 'andn2', 'orn2', 'nand', 'nor', 'xnor', 'not')

    def mask_scc(self, base, ops, masks_before, lines):
        """The SALU instruction just translated set SCC from a lane mask, lane-dependently: SCC = (64-bit result
        != 0) is one bit for the whole wave (RDNA 2 ISA: S_AND_B64 etc. "SCC = (D != 0)"; S_*_SAVEEXEC_B64
        "SCC = (EXEC != 0)"), while the one-lane model computed it from this lane's bit only."""
        if not base.startswith('s_') or not any(re.search(r'\bscc = ', l) for l in lines):
            return False
        if 'scc' not in self.tainted:
            return False
        if 'saveexec' in base or 'wrexec' in base:
            return True
        if base.split('_')[1] not in self.MASK_OPS:
            return False
        for t in ops[1:]:
            r = reg_range(t)
            if t in ('exec', 'exec_lo', 'vcc', 'vcc_lo') or (r and r[0] == 's' and r[1] in masks_before):
                return True
        return False

    def lane_scc(self):
        """Publish the wave's SCC of a lane-mask instruction: every thread whose bit of the result is set marks
        x2w_sany[k][wave], a group barrier follows (a phase boundary, as the readfirstlane broadcast), and the next
        block starts with scc = the wave's value, which is then the same in every lane of the wave (not tainted).
        Inactive lanes take part with their own bits: in a group-uniform execz range they run with exec 0, so a
        result masked with exec has 0 there, as on the hardware. Every thread of the group must reach the barrier:
        refused inside a per-lane skipped range, before a branch target, and (barrier_count) inside loops or on one
        side of a branch only."""
        if self.pc < self.divergent_until:
            raise Unsupported('lane-mask SCC at %#x inside a per-lane skipped range' % self.pc)
        if self.pending:
            raise Unsupported('lane-mask SCC at %#x right after a broadcast' % self.pc)
        if self.next_offset() in self.branch_targets():
            raise Unsupported('lane-mask SCC at %#x followed by a branch target' % self.pc)
        self.scc_from.pop('scc', None)
        self.tainted.discard('scc')
        self.au.pop('scc', None)
        self.au_xs.pop('scc', None)
        if 'publish' in TEST_DROP:
            return None            # negative control: the lane's own bit is taken as the wave's SCC
        k = len(self.scc_any)
        if 2 * (k + 1) > GROUP:
            raise Unsupported('more than %d lane-mask SCCs (the slots are cleared by one thread each)' % (GROUP // 2))
        self.scc_any.append(self.pc)
        self.lines.append('if (scc) x2w_sany[%d][wave] = 1u;' % k)
        self.pending.append('scc = x2w_sany[%d][wave] != 0u;' % k)
        self.lds_written_in_phase = False
        return ('barrier', 'lane-scc')

    @staticmethod
    def exec_written(name, ops):
        base = re.sub(r'_e(32|64)$', '', name)
        return ('saveexec' in base or 'wrexec' in base or base.startswith('v_cmpx') or
                (base.startswith('s_') and ops and ops[0] in ('exec', 'exec_lo')))

    def exec_mask_expr(self, mask):
        """This lane's bit of a constant 64-bit exec mask."""
        lo, hi = mask & 0xffffffff, mask >> 32
        if hi == 0 and lo == 1:
            return '(lane == 0u ? 1u : 0u)'
        return '((lane < 32u ? (0x%08xu >> lane) : (0x%08xu >> (lane - 32u))) & 1u)' % (lo, hi)

    def set_v(self, n, expr, *, low16_only=False):
        # Shadow: under a constant partial exec, a value computed only from wave-uniform inputs is also kept in an
        # unguarded variable, so a later v_readfirstlane can read it although the HLSL thread's lane was off
        # (the hardware reads the first active lane, which held that same value if it was active at the write).
        sub = expr
        if self.exec_static is not None and not self.exec_known:
            for v in set(re.findall(r'\bv(\d+)\b', expr)):
                if int(v) in self.shadow:
                    sub = re.sub(r'\bv%s\b' % v, self.shadow[int(v)][0], sub)
        super().set_v(n, expr, low16_only=low16_only)
        self.shadow.pop(n, None)
        if self.exec_static is None or self.exec_known or self.exec_static == 0:
            return
        if set(TOKEN.findall(sub)) & (self.tainted | {'exec', 'lane', 'tid'}):
            return
        var = 'x2w_u%d' % len(self.shadow_vars)
        self.shadow_vars.append(var)
        self.lines.append('%s = %s;' % (var, sub))
        self.shadow[n] = (var, self.exec_static)

    def declarations(self):
        lines = super().declarations()
        for (offset, size), name in self.literal_tables.items():
            words = struct.unpack_from('<%dI' % (size // 4), self.raw, offset)
            values = ', '.join('0x%08xu' % word for word in words)
            lines.append('    static const uint %s[%d] = {%s};' % (name, len(words), values))
        if self.shadow_vars:
            lines.append('    uint %s;' % ', '.join('%s = 0' % v for v in self.shadow_vars))
        return lines

    def snapshot(self):
        snap = super().snapshot() + ((self.xs, dict(self.au_xs), dict(self.xs_copy)),
                                     self.exec_static, dict(self.static_copies))
        if JOIN_TAINT:
            snap += (frozenset(self.tainted), dict(self.au), self.epoch)
        if self._execz_edge:
            self._execz_edge = False
            self._execz_snaps.append(snap)
        return snap

    def merge(self, states):
        # edges from code no path reaches in this loop pass (loop_one) do not count, unless all are
        states = [s for s in states if BOTTOM not in s[0]] or states
        dw_join = {}
        if 'descword' in MESH_FEATURES:
            # a value computed from table descriptor words on some path, where the paths disagree: when every path
            # holds a run-time number there (None / RUNTIME_BACKED), the join is a run-time number that may depend on
            # those words: ('dwnum', union), marked used only by a later numeric read. Otherwise (a conflict, which an
            # exec-off read may still take as its variable) the words are needed now (conservative).
            for reg in set().union(*(s[0] for s in states)):
                values = [s[0].get(reg) for s in states]
                if all(v == values[0] for v in values) or not any(v and v[0] == 'dwnum' for v in values):
                    continue
                deps = frozenset().union(*(v[1] for v in values if v and v[0] == 'dwnum'))
                if all(v is None or v[0] in self.RUNTIME_BACKED for v in values):
                    dw_join[reg] = ('dwnum', deps)
                else:
                    self.dw_use(deps)
        n_base = len(G.Stage.snapshot(self))
        super().merge(states)
        self.sym.update(dw_join)
        self.lineage_merge([s[n_base] for s in states])   # right after the base fields
        # the mesh fields are the last ones of the snapshot, whatever the base snapshot holds before them
        n = 5 if JOIN_TAINT else 2
        statics = [s[-n] for s in states]
        self.exec_static = statics[0] if all(x == statics[0] for x in statics) else None
        copies = [s[-n + 1] for s in states]
        self.static_copies = {r: m for r, m in copies[0].items() if all(c.get(r) == m for c in copies)}
        if JOIN_TAINT:
            # Taint at a join of forward edges (the base merges only its own fields; the linear scan otherwise kept
            # the state of the instruction above, which after an unconditional jump is no incoming path at all):
            # tainted on any incoming path; active-uniform only where every path that taints the register has it
            # active-uniform at the current epoch (no exec widening the linear scan saw since that branch).
            self.tainted = set().union(*(s[-3] for s in states))
            au = {}
            for r in self.tainted:
                if all(r not in s[-3] or (s[-2].get(r) == s[-1] == self.epoch) for s in states):
                    au[r] = self.epoch
            self.au = au
        self.merge_exec_off(states)

    def merge_exec_off(self, states):
        """Registers that the merge made ('conflict',) only because of taken s_cbranch_execz edges.

        A lane takes an execz edge only with its exec bit 0 (per lane: `(exec & 1) == 0`; group-uniform: no lane of
        the group active), so on those paths the lane's bit is 0 at the join; every other path assigned the
        register at run time. The run programs use this for a lane mask merged after a divergent if (v_cmp_e64
        into a reused SGPR inside the range, then `s_nand_b64 / s_orn2_b64 sX, sX, exec` after it; INFERRED to be the
        compiler's i1 phi, whose skipped side is undefined). Until exec is written again
        (exec_off_use), a mask op whose lane bit does not depend on the register when exec is 0 may read its HLSL
        variable."""
        self.exec_off = set()
        if not EXEC_OFF:
            return
        backed = self.RUNTIME_BACKED
        for reg, value in self.sym.items():
            if value != ('conflict',):
                continue
            bad = [s for s in states if s[0].get(reg) is not None and s[0].get(reg)[0] not in backed]
            if bad and len(bad) < len(states) and all(any(s is e for e in self._execz_snaps) for s in bad):
                self.exec_off.add(reg)

    def exec_off_use(self, base, ops):
        """Exec-off conflict registers (merge_exec_off) a 64-bit mask op may read: the op takes exec as the other
        operand and its lane bit is the same for both values of the register's bit when the lane's exec bit is 0
        (s_and X, exec = 0; s_nand = 1; s_andn2 exec, X = 0; s_orn2 X, exec = 1). Empty when it does not apply."""
        if not self.exec_off or base not in G.SCA or len(ops) != 3 or not base.endswith('_b64'):
            return set()
        srcs = ops[1:]
        if 'exec' not in srcs:
            return set()
        k = 1 - srcs.index('exec')
        r = reg_range(srcs[k])
        if not r or r[0] != 's' or r[1] not in self.exec_off or self.sym.get(r[1]) != ('conflict',):
            return set()
        text = G.SCA[base][0]
        if '{2}' in text or not re.fullmatch(r'[~(){}01&|^ ]+', text):
            return set()
        bits = set()
        for x in (0, 1):
            args = ['0', '0']
            args[k] = str(x)
            bits.add(eval(text.format(*args)) & 1)   # noqa: S307 (a fixed SCA formula over two constant bits)
        return {r[1]} if len(bits) == 1 else set()

    def format_load(self, ops, mods, count, name):
        """buffer_load_format_* through a resource-heap descriptor (X2W_DRAW_FEATURES heap + bufword3): the format
        and swizzle come from the view's word 3 (x2w_hword, X2W_MESH_FEATURES=heapwords); table descriptors take
        the base path."""
        key = self.descriptor(ops[2])
        if key[0][0] != 'heapdyn':
            return super().format_load(ops, mods, count, name)
        if not {'heap', 'bufword3'} <= set(getattr(self.pipe, 'features', ())) or 'heapwords' not in MESH_FEATURES:
            raise Unsupported('%s through heapdyn' % name)
        self.pipe.heapwords = True
        for m in mods:
            if not (m in ('idxen', 'offen', 'glc', 'slc', 'dlc') or m.startswith('offset:')):
                raise Unsupported('%s modifier %s' % (name, m))
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, mods)
        self.emit('a = %s;' % address)
        for i in range(count):
            self.set_v(dst[1] + i, 'x2w_fmt_load(%s, a, x2w_hword(h%d, 3u), %du)' % (buf, key[0][1], i))

    # ---- table descriptor words read as numbers (X2W_MESH_FEATURES=descword) ---------------------------------------
    # A word of a descriptor loaded from a root descriptor TABLE (('desc', ('table', p), off, w)) is known only at draw
    # time: the runtime passes it as entry value `udsrc <n> descword <p> <off + 4 w>` (d3d12_draw.inl draw_descword_value:
    # the title's dword, or the synthesized T# / V# word of a view token, as compute's descword). Scalar arithmetic over
    # such words is emitted with a placeholder X2W_DW<k> per word and the result register holds ('dwnum', {k, ...})
    # (a run-time number). Only a numeric use (S(), or a join where paths disagree) marks its words used; build()
    # then makes used placeholders entry values and the others 0u (their arithmetic is dead code, as before: a V#
    # NUM_RECORDS x STRIDE bounds computation nobody reads adds no entry value).
    RUNTIME_BACKED = G.Stage.RUNTIME_BACKED + ('dwnum',)

    @staticmethod
    def dw_edit_key(n):
        return 'x2w_dwedit_%d' % n   # key in the symbolic state (snapshotted and merged with it), as BOTTOM

    def dw_mark_edit(self, n):
        """s<n> holding a descriptor word was edited in place (the base keeps its symbol): marked until an instruction
        writes s<n> otherwise (one(): a fresh load of the same entry is the table's word again)."""
        self.sym[self.dw_edit_key(n)] = ('dwedit',)
        self._dw_keep = getattr(self, '_dw_keep', set()) | {n}

    def dw_word(self, s, n):
        """Placeholder index for symbol s of s<n> when it is a table descriptor word the runtime can pass, else None."""
        if 'descword' not in MESH_FEATURES or not s or s[0] != 'desc' or not isinstance(s[1], tuple):
            return None
        if s[1][0] != 'table' or s[2] is None or not isinstance(s[2], int) or not isinstance(s[3], int):
            return None
        if self.sym.get(self.dw_edit_key(n)):
            # edited in place since it was loaded (or on some path into a join): the table's word is not its value
            raise Unsupported('numeric use of an edited descriptor word (s%d: table %d +%d word %d)' % (n, s[1][1], s[2], s[3]))
        src = ('descword', s[1][1], s[2] + 4 * s[3])
        if src not in self.pipe.dw_src:
            self.pipe.dw_src.append(src)
        return self.pipe.dw_src.index(src)

    def dw_scalar(self, base, ops):
        """A 32-bit scalar op over table descriptor words (and numbers): emitted with placeholders, result ('dwnum',
        deps). None when the op is not one (the caller goes on as before)."""
        if 'descword' not in MESH_FEATURES or base not in G.SCA or base.endswith('_b64') or len(ops) < 2:
            return None
        d = reg_range(ops[0])
        if not d or d[0] != 's' or d[1] != d[2]:
            return None
        syms = []
        for t in ops[1:]:
            r = reg_range(t)
            if r and r[0] == 's' and r[1] != r[2]:
                return None
            syms.append(self.sym.get(r[1]) if r and r[0] == 's' else None)
        kinds = [s[0] if s else None for s in syms]
        if not any(k in ('desc', 'dwnum') for k in kinds):
            return None
        if base == 's_mov_b32' and kinds[0] == 'desc':
            # a descriptor word moved: it keeps its descriptor meaning (a resource operand, as before), and its edit mark
            if self.sym.get(self.dw_edit_key(reg_range(ops[1])[1])):
                self.dw_mark_edit(d[1])
            return None
        if any(k not in (None, 'const', 'ud', 'fixed_buffer_word', 'desc', 'dwnum') for k in kinds):
            return None
        own = self.sym.get(d[1])
        if own and own[0] == 'desc' and own in syms:
            # an in-place edit of a descriptor word: the base records it (and keeps the symbol); noted for dw_word
            self.dw_mark_edit(d[1])
            return None
        if base == 's_movk_i32':
            return None
        regs = [reg_range(t) for t in ops[1:]]
        if any(k == 'desc' and self.dw_word(s, r[1]) is None for k, s, r in zip(kinds, syms, regs)):
            return None
        deps, values = set(), []
        for t, s in zip(ops[1:], syms):
            r = reg_range(t)
            if s and s[0] == 'desc':
                k = self.dw_word(s, r[1])
                deps.add(k)
                values.append('X2W_DW%d' % k)
            elif s and s[0] == 'dwnum':
                deps |= s[1]
                self.used_s.add(r[1])
                values.append('s%d' % r[1])
            else:
                values.append(self.u(t))
        text, scc = G.SCA[base]
        self.set_s(d[1], text.format(*values))
        self.sym[d[1]] = ('dwnum', frozenset(deps))
        if scc:
            self.emit('scc = %s;' % scc.replace('$', 's%d' % d[1]))
        return True

    def dw_use(self, deps):
        self.pipe.dw_used |= set(deps)

    def S(self, n):
        """A word of a resource-heap descriptor read as a number (an image's width bits, a null test): the
        runtime's synthesized word (x2w_hword), with X2W_MESH_FEATURES=heapwords; a table descriptor word or a
        value computed from such words (X2W_MESH_FEATURES=descword): its entry value / variable, marked used;
        otherwise as the base."""
        s = self.sym.get(n)
        if s and s[0] == 'hword':
            return 'x2w_hword(h%d, %du)' % (s[1], s[2])
        if s and s[0] == 'dwnum':
            self.dw_use(s[1])
            self.used_s.add(n)
            return 's%d' % n
        k = self.dw_word(s, n)
        if k is not None:
            self.dw_use({k})
            return 'X2W_DW%d' % k
        if (s and s[0] == 'desc' and isinstance(s[1], tuple) and s[1][0] == 'heapdyn' and s[2] == 0
                and 'heapwords' in MESH_FEATURES and 'heap' in getattr(self.pipe, 'features', ())):
            self.pipe.heapwords = True
            return 'x2w_hword(h%d, %du)' % (s[1][1], s[3])
        return super().S(n)

    def heap_sampler(self, tok):
        # run-time offset (base) or a constant entry of the sampler heap (const_heap_sampler)
        return super().heap_sampler(tok) or const_heap_sampler(self, tok)

    def set_exec_static(self, mask):
        """exec = a constant lane mask (s_mov_b64 exec, <constant> or a copy of one)."""
        mask &= ALL_LANES
        self._static_set = True
        self.exec_static = mask
        if mask == ALL_LANES:
            self.emit('exec = 1;')
            self.exec_known = True
        else:
            self.emit('exec = %s;' % self.exec_mask_expr(mask))
            self.exec_known = False

    def op_s_cselect_b32(self, ops, mods):
        # the base handles register / exec / vcc destinations; the run form also selects into m0
        if ops[0] == 'm0':
            low = [('s%d' % reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops[1:3]]
            self.emit('m0 = scc ? %s : %s;' % (self.u(low[0]), self.u(low[1])))
            return None
        if not reg_range(ops[0]) and ops[0] not in ('exec', 'exec_lo', 'vcc', 'vcc_lo'):
            raise Unsupported('s_cselect_b32 to %s' % ops[0])
        return super().op_s_cselect_b32(ops, mods)

    def op_s_and_b32(self, ops, mods):
        # merged_wave_info & mask: the wave index field (bits 27:24) is the only part this translation models
        for i, j in ((1, 2), (2, 1)):
            src = reg_range(ops[i])
            if src and src[0] == 's' and self.sym.get(src[1]) == ('waveinfo',):
                if not re.fullmatch(r'0x[0-9a-f]+|\d+', ops[j]) or int(ops[j], 0) & ~0x0f000000:
                    raise Unsupported('merged_wave_info & %s' % ops[j])
                d = reg_range(ops[0])
                if not d or d[0] != 's':
                    raise Unsupported('s_and_b32 of merged_wave_info to %s' % ops[0])
                self.masks.discard(d[1])
                self.set_s(d[1], '((wave << 24) & 0x%08xu)' % int(ops[j], 0), scc='$ != 0u')
                return None
        return self.scalar('s_and_b32', ops)

    def lds(self, base, ops, mods):
        """ds_read_b32 / ds_write_b32 / _b64 with offset:N as groupshared uint x2w_lds[] (byte address / 4).
        Lanes of one wave run independently between barriers here, so an LDS read after an LDS write in the same
        barrier phase (lane-to-lane traffic inside a wave) is refused."""
        size = {'ds_read_b32': 1, 'ds_write_b32': 1, 'ds_read_b64': 2, 'ds_write_b64': 2}.get(base)
        if size is None:
            raise Unsupported(base)
        off = 0
        for m in mods:
            if m.startswith('offset:'):
                off = int(m[7:], 0)
            else:
                raise Unsupported('%s with %s' % (base, m))
        if off + 4 * size > 4 * LDS_DWORDS:
            raise Unsupported('%s offset %d beyond the modelled LDS' % (base, off))
        self.lds_used = True
        if base.startswith('ds_write'):
            addr, data = ops[0], reg_range(ops[1])
            # Several lanes storing to one address are ordered on the console (the highest lane wins) but race in
            # HLSL: a uniform address is only taken under a constant single-lane exec.
            single = self.exec_static is not None and bin(self.exec_static).count('1') == 1
            ar = reg_range(addr)
            if not single and not (ar and ar[0] == 'v' and self.lane_dependent('v%d' % ar[1])):
                raise Unsupported('%s to a wave-uniform address with more than one lane active at %#x' % (base, self.pc))
            self.emit('a = (%s + %du) >> 2;' % (self.u(addr), off))
            for i in range(size):
                self.emit('if (exec & 1) x2w_lds[min(a + %du, %du)] = v%d;' % (i, LDS_DWORDS - 1, data[1] + i))
                self.used_v.add(data[1] + i)
            self.lds_written_in_phase = True
            return None
        if self.lds_written_in_phase:
            raise Unsupported('LDS read after an LDS write without a barrier at %#x' % self.pc)
        dst, addr = reg_range(ops[0]), ops[1]
        ar = reg_range(addr)
        if ar and ar[0] == 'v' and dst[1] <= ar[1] < dst[1] + size - 1:   # a later word would read the new value
            raise Unsupported('%s overwrites its address register' % base)
        for i in range(size):
            # the address expression is repeated (no temporary), so a shadow (see set_v) can substitute into it
            self.set_v(dst[1] + i, 'x2w_lds[min(((%s + %du) >> 2) + %du, %du)]' % (self.u(addr), off, i, LDS_DWORDS - 1))
        return None

    def liveness(self):
        """Scalar registers (numbers; -1 = vcc) live on entry to each instruction offset (backward dataflow over
        fall-through and branch edges)."""
        if getattr(self, '_live', None) is not None:
            return self._live
        order = [x[0] for x in self.ins]
        info = {x[0]: (x[2], x[3]) for x in self.ins}
        following = dict(zip(order, order[1:]))
        rw = {}
        succ = {}
        for off in order:
            name, asm = info[off]
            r, w, wv, rv = reads_writes(name, asm)
            if RESTORED_SCALARS:
                # wave64 lane masks that a VALU op names by their low SGPR: a v_cmp SDST writes the pair; a
                # v_cndmask / carry-in mask operand reads it (AMD RDNA 2 ISA, VOP3 SDST / SSRC2 in wave64)
                w = self.pair_writes(name, asm)
                base = re.sub(r'_e(32|64)$', '', name)
                last = parse(asm)[1][-1:] if base.startswith('v_cndmask') or (CARRY.fullmatch(base) and '_ci' in base) else []
                if last and re.fullmatch(r's\d+', last[0]):
                    r = set(r) | {int(last[0][1:]) + 1}
            rw[off] = (frozenset(r | ({-1} if rv else set())), frozenset(w | ({-1} if wv else set())))
            nxt = [] if name in ('s_endpgm', 's_branch') else ([following[off]] if off in following else [])
            if name.startswith('s_cbranch') or name == 's_branch':
                nxt.append(int(parse(asm)[1][0].split('_')[1], 16))
            succ[off] = [o for o in nxt if o in info]
        live = {off: frozenset() for off in order}
        changed = True
        while changed:
            changed = False
            for off in reversed(order):
                out = frozenset().union(*(live[s] for s in succ[off])) if succ[off] else frozenset()
                r, w = rw[off]
                new = r | (out - w)
                if new != live[off]:
                    live[off], changed = new, True
        self._live = live
        return live

    def check_skipped_scalars(self, off, target):
        """Refuse when a scalar register written in (off, target) is live at target (some path reads it before a
        write): lanes that skip the range would see the old value where the wave saw the new one."""
        hit = self.skipped_scalars(off, target)
        if hit:
            raise Unsupported('scalar %s written in the per-lane skipped range %#x-%#x is live at its end'
                              % ('vcc' if min(hit) == -1 else 's%d' % min(hit), off, target))

    def range_successors(self):
        if getattr(self, '_succ', None) is None:
            order = [x[0] for x in self.ins]
            info = {x[0]: (x[2], x[3]) for x in self.ins}
            following = dict(zip(order, order[1:]))
            self._succ = {}
            for o in order:
                name, asm = info[o]
                nxt = [] if name in ('s_endpgm', 's_branch') else ([following[o]] if o in following else [])
                if name.startswith('s_cbranch') or name == 's_branch':
                    nxt.append(int(parse(asm)[1][0].split('_')[1], 16))
                self._succ[o] = [x for x in nxt if x in info]
        return self._succ

    def restored_scalars(self, off, target, hits):
        """Of the SGPRs in `hits` (written in the range (off, target)), those that hold their value from before the
        range again on every exit of the range: the register-allocator shuffle `s_mov_b32 sT, sR; ... sR = ...;
        ... s_mov_b32 sR, sT`. Forward copy tracking over the range's own paths (state: register -> the register
        whose entry value it holds, None = unknown; a register not in the state holds its own entry value),
        intersected at joins; s_mov_b32 / s_mov_b64 copy, any other write ends what the destination held."""
        if not RESTORED_SCALARS:
            return set()
        succ = self.range_successors()
        info = {x[0]: (x[2], x[3]) for x in self.ins}
        inside = [o for o, _, _, _ in self.ins if off < o < target]
        if not inside:
            return set()
        state = {inside[0]: {}}
        exits = []
        work = [inside[0]]
        seen_edges = 0
        while work:
            seen_edges += 1
            if seen_edges > 100000:
                return set()
            o = work.pop()
            st = dict(state[o])
            name, asm = info[o]
            ops = parse(asm)[1]
            w = self.pair_writes(name, asm)
            d = reg_range(ops[0]) if ops else None
            s = reg_range(ops[1]) if len(ops) == 2 else None
            if name in ('s_mov_b32', 's_mov_b64') and d and d[0] == 's' and s and s[0] == 's':
                n = 2 if name == 's_mov_b64' else 1
                vals = [st.get(s[1] + i, s[1] + i) for i in range(n)]
                for i in range(n):
                    st[d[1] + i] = vals[i]
            else:
                for r in w:
                    if isinstance(r, int):
                        st[r] = None
            for nx in succ[o]:
                if not off < nx < target:
                    if nx <= off:
                        return set()   # a loop back above the branch: not modelled
                    exits.append(st)
                    continue
                if nx not in state:
                    state[nx] = st
                    work.append(nx)
                else:
                    merged = {r: v for r, v in state[nx].items()}
                    for r in set(merged) | set(st):
                        a, b = merged.get(r, r), st.get(r, r)
                        merged[r] = a if a == b else None
                    if merged != state[nx]:
                        state[nx] = merged
                        work.append(nx)
        return {r for r in hits if isinstance(r, int) and r >= 0 and exits and all(e.get(r, r) == r for e in exits)}

    def group_branches(self):
        """execz branches (offsets) translated as group-uniform branches: those whose range writes a scalar that
        is live at its end or holds a published lane-mask SCC (pipe.scc_publish), and every execz range enclosing
        one of them (the inner barrier needs every thread of the group to reach it). Exec-write refusals are left
        to the translation."""
        if getattr(self, '_group', None) is None:
            ranges = []
            for o, n, name, asm in self.ins:
                if name == 's_cbranch_execz':
                    t = int(parse(asm)[1][0].split('_')[1], 16)
                    if t > o:
                        ranges.append((o, t))
            # ranges that hold a lane-dependent v_readfirstlane_b32 (NeedGroupRange, set by MeshPipeline)
            need = set(o for o, t in ranges if o in getattr(self.pipe, 'force_group', ()))
            for o, t in ranges:
                try:
                    if self.skipped_scalars(o, t):
                        need.add(o)
                except Unsupported:
                    pass
            # a lane-mask SCC published inside the range: the publish barrier needs every thread of the group, and
            # the wave's SCC is then the OR over all 64 lanes' bits (the SALU op reads every bit, whatever exec is;
            # the inactive lanes run the range with exec 0, their compare bits 0, as VOPC writes them)
            publish = getattr(self.pipe, 'scc_publish', ()) if LANE_SCC else ()
            for o, t in ranges:
                if any(o < p < t for p in publish):
                    need.add(o)
            for o, t in ranges:
                if any(o < o2 < t for o2 in need):
                    need.add(o)
            self._group = {o: t for o, t in ranges if o in need}
        return self._group

    @staticmethod
    def pair_writes(name, asm):
        """SGPRs an instruction writes, with the high half of a wave64 lane mask that a compare names by its low
        register only (v_cmp / v_cmp_class SDST: the disassembler prints one SGPR; carry-outs are already pairs in
        reads_writes; readlane / readfirstlane write one SGPR)."""
        w = set(reads_writes(name, asm)[1])
        if name.startswith('v_cmp'):
            w |= {x + 1 for x in w}
        return w

    def exec_copies_at(self):
        """Offset -> SGPR pairs (low register numbers) that hold exactly the wave's exec on entry to that
        instruction on EVERY path from the entry: `s_mov_b64 sX, exec`, `s_mov_b64 exec, sX` and moves of such a
        pair. Forward dataflow over fall-through and branch edges (loops included), intersected at joins; any other
        exec write, or a write to the pair, ends the fact. Static (no translation state), so the pre-pass of
        group_branches sees the same facts as the translation."""
        if not EXEC_COPIES:
            return {}
        if getattr(self, '_exec_copies', None) is None:
            order = [x[0] for x in self.ins]
            info = {x[0]: (x[2], x[3]) for x in self.ins}
            following = dict(zip(order, order[1:]))
            succ = {}
            for off in order:
                name, asm = info[off]
                nxt = [] if name in ('s_endpgm', 's_branch') else ([following[off]] if off in following else [])
                if name.startswith('s_cbranch') or name == 's_branch':
                    nxt.append(int(parse(asm)[1][0].split('_')[1], 16))
                succ[off] = [o for o in nxt if o in info]

            def transfer(off, eq):
                # eq: 32-bit halves (register, 0 = exec_lo / 1 = exec_hi) holding that half of the current exec
                name, asm = info[off]
                ops = parse(asm)[1]
                w = self.pair_writes(name, asm)
                hv = [self.mask_halves(t) for t in ops[:2]]
                if ops and ops[0] in ('exec', 'exec_lo', 'exec_hi') or self.exec_written(name, ops):
                    if name == 's_mov_b64' and ops[0] == 'exec' and isinstance(hv[1], list) and len(hv[1]) == 2:
                        return frozenset(hv[1])
                    return frozenset()
                add = set()
                d = hv[0] if hv and isinstance(hv[0], list) else None
                if d and len(ops) == 2 and name in ('s_mov_b64', 's_mov_b32'):
                    s = hv[1]
                    if name == 's_mov_b64' and len(d) == 2:
                        if s == 'exec':
                            add = set(d)
                        elif isinstance(s, list) and len(s) == 2:
                            add = {(d[i][0], s[i][1]) for i in range(2) if s[i] in eq}
                    elif name == 's_mov_b32' and len(d) == 1:
                        if isinstance(s, tuple):
                            add = {(d[0][0], s[1])}
                        elif isinstance(s, list) and len(s) == 1:
                            add = {(d[0][0], h) for (rr, h) in eq if rr == s[0][0]}
                return frozenset({x for x in eq if x[0] not in w} | add)
            facts = {order[0]: frozenset()} if order else {}   # None (absent) = not reached yet
            work = list(order[:1])
            while work:
                off = work.pop()
                out = transfer(off, facts[off])
                for s in succ[off]:
                    new = out if s not in facts else facts[s] & out
                    if facts.get(s) != new:
                        facts[s] = new
                        work.append(s)
            self._exec_copies = {o: frozenset(r for (r, h) in eq if h == 0 and (r + 1, 1) in eq) for o, eq in facts.items()}
        return self._exec_copies

    def skipped_scalars(self, off, target):
        """Scalar registers (-1 = vcc) written in (off, target) and live at target; Unsupported for exec writes
        the per-lane model cannot follow."""
        if EXEC_COPIES:
            written = self.exec_walk(off, target)
            hits = written & self.liveness().get(target, frozenset())
            return hits - self.restored_scalars(off, target, hits) if hits else hits
        written = set()
        copies = set()   # registers holding exec as it was inside the range (nested if/else masks)
        for o, n, name, asm in self.ins:
            if off < o < target:
                r, w, wv, _ = reads_writes(name, asm)
                written |= w | ({-1} if wv else set())
                ops = parse(asm)[1]
                low = [reg_range(t)[1] if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops]
                if name.startswith('s_') and 'saveexec' in name:
                    copies.add(low[0])
                    continue
                if name in ('s_mov_b64', 's_mov_b32') and len(ops) == 2 and ops[1] in ('exec', 'exec_lo'):
                    copies.add(low[0])
                    continue
                if (re.match(r's_(and|andn2)_b(32|64)$', name) and len(ops) == 3 and ops[0] not in ('exec', 'exec_lo')
                        and low[1] in copies):
                    copies.add(low[0])   # a subset of a copy taken in the range (`copy & ~exec`: the else lanes)
                    continue
                if name.startswith('s_') and ops and ops[0] in ('exec', 'exec_lo'):
                    # A wave-level exec write can turn on lanes the per-lane model left behind, unless it only
                    # narrows exec (and / andn2 with exec) or restores / complements a copy taken in the range.
                    srcs = low[1:]
                    if re.match(r's_(and|andn2)_b(32|64)$', name) and 'exec' in ops[1:]:
                        pass
                    elif name in ('s_mov_b64', 's_mov_b32') and srcs[0] in copies:
                        pass
                    elif re.match(r's_andn2_b(32|64)$', name) and srcs[0] in copies:
                        pass   # else branch: copy & ~exec is inside the copy
                    else:
                        raise Unsupported('exec written by %s inside the per-lane skipped range %#x-%#x'
                                          % (name, off, target))
                for t in w:
                    copies.discard(t)
        return written & self.liveness().get(target, frozenset())

    @staticmethod
    def mask_halves(tok):
        """A scalar operand as 32-bit halves of a wave64 mask: 's[30:31]' -> [(30, 0), (31, 1)], 's30' -> [(30, None)]
        (a 32-bit move keeps whichever half it holds), 'exec' -> 'exec', 'exec_lo' / 'exec_hi' -> ('exec', h)."""
        if tok == 'exec':
            return 'exec'
        if tok in ('exec_lo', 'exec_hi'):
            return ('exec', 0 if tok == 'exec_lo' else 1)
        r = reg_range(tok)
        if not r or r[0] != 's':
            return None
        if r[2] == r[1] + 1:
            return [(r[1], 0), (r[1] + 1, 1)]
        if r[2] == r[1]:
            return [(r[1], None)]
        return None

    def xor_swaps(self):
        """XOR swaps of two SGPRs or two disjoint SGPR pairs: `s_xor_b32 sA, sA, sB; s_xor_b32 sB, sA, sB;
        s_xor_b32 sA, sA, sB` (or the s_xor_b64 form; sources in either order) inside one basic block, where the
        instructions between them touch neither A nor B. Afterwards A holds B's old value and B holds A's
        (x ^ y ^ y = x). Returns ({offset of the first: {register: register it moves to}}, {offsets of the other
        two}): exec_walk exchanges the facts of A and B at the first and leaves them alone at the others."""
        if getattr(self, '_xor_swaps', None) is None:
            swaps, rest = {}, set()
            targets = self.branch_targets()

            def form(x):
                """(width, destination base, {source bases}) of an s_xor_b32 / s_xor_b64 on SGPRs."""
                ops = parse(x[3])[1]
                width = {'s_xor_b32': 1, 's_xor_b64': 2}.get(x[2])
                if not width or len(ops) != 3:
                    return None
                # vcc is s[106:107] on the console (the register allocator swaps through it as a data pair)
                r = [('s', 106, 107) if t == 'vcc' else reg_range(t) for t in ops]
                if not all(rr and rr[0] == 's' and rr[2] - rr[1] + 1 == width for rr in r):
                    return None
                return width, r[0][1], {r[1][1], r[2][1]}
            for i in range(len(self.ins) - 2):
                f0 = form(self.ins[i])
                if not f0 or len(f0[2]) != 2 or f0[1] not in f0[2]:
                    continue
                width, a = f0[0], f0[1]
                b = (f0[2] - {a}).pop()
                if abs(a - b) < width:
                    continue   # overlapping pairs are no swap
                regs = set(range(a, a + width)) | set(range(b, b + width))
                # the next two instructions that touch sA or sB must be the other two xors; instructions between
                # them touch neither, and none of them is a branch or a branch target (one basic block)
                found = []
                for x in self.ins[i + 1:i + 17]:
                    if x[0] in targets or x[2].startswith(('s_branch', 's_cbranch', 's_setpc', 's_endpgm')):
                        break
                    rd, wr, wv, rv = reads_writes(x[2], x[3])
                    touched = set(rd) | set(wr) | ({106, 107} if wv or rv else set())
                    if not (regs & touched):
                        continue
                    want = (width, b, {a, b}) if not found else (width, a, {a, b})
                    if form(x) != want:
                        break
                    found.append(x[0])
                    if len(found) == 2:
                        swaps[self.ins[i][0]] = {a + k: b + k for k in range(width)} | \
                            {b + k: a + k for k in range(width)}
                        rest |= set(found)
                        break
            self._xor_swaps = (swaps, rest)
        return self._xor_swaps

    def exec_walk(self, off, target):
        """The exec rule of skipped_scalars as a forward dataflow over the paths inside (off, target): returns the
        scalar registers (-1 = vcc) written there, or raises Unsupported for an exec write that could turn on a lane
        that was off at the branch (the one-lane model left that lane at the target).

        State: the 32-bit halves (register, 0 = lanes 0-31 / 1 = lanes 32-63) that hold part of a lane mask inside
        the lanes active at the branch ("in-range masks"). Exec itself always is one (every accepted exec write keeps
        it inside). Seeded with the pairs equal to exec at the branch (exec_copies_at); saveexec destinations,
        copies of exec, moves (64- and 32-bit: register shuffles and SGPR spill slots), `and` with an in-range mask,
        `andn2` of one and `or` of two are in-range masks; any other write ends what the register held. Intersected
        at joins (a register is an in-range mask only if it is one on every path). Accepted exec writes: v_cmpx
        (narrows), s_and with exec or an in-range mask, s_andn2 / andn2_saveexec / andn2_wrexec of an in-range mask
        (an else mask), s_or / or_saveexec of in-range masks, and moves of an in-range pair or half."""
        info = {x[0]: (x[2], x[3]) for x in self.ins}
        succ = self.range_successors()
        inside = [o for o, _, _, _ in self.ins if off < o < target]
        written = set()
        if not inside:
            return written
        seed = set()
        for x in self.exec_copies_at().get(off, ()):
            seed |= {(x, 0), (x + 1, 1)}
        state = {inside[0]: frozenset(seed)}
        work = [inside[0]]
        steps = 0
        swaps, swap_rest = self.xor_swaps()

        def refuse(name):
            raise Unsupported('exec written by %s inside the per-lane skipped range %#x-%#x (at %#x)'
                              % (name, off, target, o))
        while work:
            steps += 1
            if steps > 200000:
                raise Unsupported('exec dataflow of the per-lane range %#x-%#x does not settle' % (off, target))
            o = work.pop()
            st = set(state[o])
            name, asm = info[o]
            ops = parse(asm)[1]
            halves = [self.mask_halves(t) for t in ops]

            def inrange(h):
                """Operand h (from mask_halves) is an in-range 64-bit mask."""
                return h == 'exec' or (isinstance(h, list) and len(h) == 2 and all(x in st for x in h))
            add = set()
            base = name
            if base.startswith('s_') and ops and ops[0] in ('exec', 'exec_lo', 'exec_hi'):
                src = halves[1:]
                if base in ('s_mov_b64',) and inrange(src[0]):
                    pass
                elif base == 's_mov_b32' and isinstance(src[0], list) and src[0][0][1] is None and \
                        any((src[0][0][0], h) in st for h in (0, 1)) and \
                        (src[0][0][0], 0 if ops[0] == 'exec_lo' else 1) in st:
                    pass   # a half restored into its own half of exec
                elif base == 's_and_b64' and (inrange(src[0]) or inrange(src[1])):
                    pass
                elif base == 's_andn2_b64' and inrange(src[0]):
                    pass   # else mask: an in-range mask & ~exec
                elif base == 's_or_b64' and inrange(src[0]) and inrange(src[1]):
                    pass
                else:
                    refuse(name)
            elif base.startswith('s_') and ('saveexec' in base or 'wrexec' in base):
                src = halves[1]
                if not (base.startswith('s_and_') or inrange(src)):
                    refuse(name)   # or / andn2 forms take lanes from the source
                d = halves[0]
                if isinstance(d, list) and len(d) == 2:
                    add = set(d)   # the old exec (saveexec) or the new one (wrexec): inside the branch's lanes
            elif base.startswith('s_') and len(ops) >= 2 and isinstance(halves[0], list):
                d = halves[0]
                if base == 's_mov_b64' and len(d) == 2:
                    s = halves[1]
                    if s == 'exec':
                        add = set(d)
                    elif isinstance(s, list) and len(s) == 2:
                        add = {(d[i][0], s[i][1]) for i in range(2) if s[i] in st}
                elif base == 's_mov_b32' and len(d) == 1:
                    s = halves[1]
                    if isinstance(s, tuple):
                        add = {(d[0][0], s[1])}                         # exec_lo / exec_hi
                    elif isinstance(s, list) and len(s) == 1:
                        add = {(d[0][0], h) for (rr, h) in st if rr == s[0][0]}
                elif base == 's_and_b64' and len(d) == 2 and (inrange(halves[1]) or inrange(halves[2])):
                    add = set(d)
                elif base == 's_andn2_b64' and len(d) == 2 and inrange(halves[1]):
                    add = set(d)
                elif base == 's_or_b64' and len(d) == 2 and inrange(halves[1]) and inrange(halves[2]):
                    add = set(d)
            if o in swaps:
                # the register allocator's XOR swap (three s_xor): A and B exchange their values
                st = {(swaps[o].get(r, r), h) for (r, h) in st}
            elif o not in swap_rest:
                pw = self.pair_writes(name, asm)
                st = {x for x in st if x[0] not in pw} | add
            out = frozenset(st)
            for nx in succ[o]:
                if not off < nx < target:
                    continue
                new = out if nx not in state else state[nx] & out
                if state.get(nx) != new:
                    state[nx] = new
                    work.append(nx)
        for o in inside:
            name, asm = info[o]
            _, w, wv, _ = reads_writes(name, asm)
            written |= w | ({-1} if wv else set())
            ops = parse(asm)[1]
            # an instruction the walk did not reach from the branch (entered from elsewhere): no exec write there
            if o not in state and name.startswith('s_') and (ops and ops[0] in ('exec', 'exec_lo', 'exec_hi') or
                                                             'saveexec' in name or 'wrexec' in name):
                refuse(name)
        return written

    def scalar(self, base, ops):
        """With X2W_MESH_FEATURES=heapwords, words of a resource-heap descriptor that a scalar op reads as numbers
        (not an in-place edit of the descriptor) are the runtime's synthesized words (x2w_hword): they are
        swapped for a numeric symbol for this instruction, so the base computes the result at run time
        instead of keeping a symbolic `descnum`."""
        swapped = {}
        off_regs = self.exec_off_use(base, ops)
        for n in off_regs:
            # (merge_exec_off) the register's run-time variable: right on the paths that assigned it, and the
            # result does not depend on it in a lane that came with exec 0
            self.sym.pop(n)
        if 'heapwords' in MESH_FEATURES and 'heap' in getattr(self.pipe, 'features', ()):
            d = reg_range(ops[0])
            own = self.sym.get(d[1]) if d and d[0] == 's' else None
            for t in ops[1:]:
                r = reg_range(t)
                if not r or r[0] != 's':
                    continue
                for n in range(r[1], r[2] + 1):
                    s = self.sym.get(n)
                    if (s and s[0] == 'desc' and isinstance(s[1], tuple) and s[1][0] == 'heapdyn' and s[2] == 0
                            and s != own):
                        swapped[n] = s
                        self.sym[n] = ('hword', s[1][1], s[3])
        if swapped:
            self.pipe.heapwords = True
        edits = sum(len(v) for v in getattr(self.pipe, 'modified_words', {}).values())
        try:
            term = self.scalar_inner(base, ops)
            if 'descword' in MESH_FEATURES and sum(len(v) for v in getattr(self.pipe, 'modified_words', {}).values()) > edits:
                # the base recorded an in-place edit of a descriptor word (whatever the other operands): s<d> keeps the
                # symbol but not the table's value (dw_word refuses its numeric use until it is written otherwise)
                d = reg_range(ops[0])
                if d and d[0] == 's':
                    for n in range(d[1], d[2] + 1):
                        self.dw_mark_edit(n)
            return term
        finally:
            for n, s in swapped.items():
                if self.sym.get(n) == ('hword', s[1][1], s[3]):
                    self.sym[n] = s
            for n in off_regs:
                if n not in scalar_regs(ops[0]):
                    self.sym[n] = ('conflict',)

    def scalar_inner(self, base, ops):
        dst = reg_range(ops[0])
        src = reg_range(ops[1]) if len(ops) >= 2 else None
        if (base == 's_mov_b32' and len(ops) == 2 and dst and src and dst[0] == src[0] == 's'
                and dst[1] == dst[2] and src[1] == src[2] and self.sym.get(src[1]) == ('conflict',)):
            # Also used for either dword of an SGPR pair. Copy its numeric backing where it
            # exists, retaining the conflict so numeric users still need a proof on every path.
            masked = src[1] in self.masks
            self.used_s.add(src[1])
            self.set_s(dst[1], 's%d' % src[1])
            self.sym[dst[1]] = ('conflict',)
            self.exec_off.discard(dst[1])
            getattr(self, 'execzero', {}).pop(dst[1], None)
            if masked:
                self.masks.add(dst[1])
            return None
        symbol = self.sym.get(src[1]) if src and src[0] == 's' else None
        if dst and dst[0] == 's' and symbol and symbol[0] in ('code_lo', 'code_hi'):
            if base == 's_mov_b32':
                self.set_s(dst[1], None, symbol)
                return
            if base == 's_add_u32' and symbol[0] == 'code_lo' and re.fullmatch(r'0x[0-9a-f]+|\d+', ops[2]):
                self.set_s(dst[1], None, ('code_lo', symbol[1] + int(ops[2], 0)))
                return
        if base == 's_bcnt1_i32_b32' and len(ops) == 2 and self.is_tainted(ops[1]):
            # a scalar that depends on lane masks holds this lane's share of a wave value: its count is not the wave's
            raise Unsupported('s_bcnt1_i32_b32 of a value that depends on lane masks')
        dname = ops[0]
        r0 = reg_range(dname)
        if dname == 'm0':
            text, _ = G.SCA[base]
            srcs = [('s%d' % reg_range(t)[1]) if reg_range(t) and reg_range(t)[0] == 's' else t for t in ops[1:]]
            self.emit('m0 = %s;' % text.format(*[self.u(t) for t in srcs]))
            return None
        # restoring exec from a copy taken while every lane was active makes every lane active again;
        # s_mov_b64 exec, -1 (the program's first instruction) activates all 64 lanes, which all exist
        # (128 threads = two full waves)
        if dname == 'exec' and base == 's_mov_b64':
            if re.fullmatch(r'-?(0x[0-9a-f]+|\d+)', ops[1]):
                self.set_exec_static(int(ops[1], 0))
                return None
            src = reg_range(ops[1])
            if src and src[0] == 's' and src[1] in self.static_copies:
                self.set_exec_static(self.static_copies[src[1]])
                return None
            if src and src[0] == 's' and self.sym.get(src[1]) == ('execcopy',):
                self.set_exec_static(ALL_LANES)
                return None
        if base == 's_mov_b64' and len(ops) == 2 and ops[1] == 'exec' and self.exec_static is not None:
            d = reg_range(dname)
            if d and d[0] == 's':
                mask = self.exec_static
                term = super().scalar(base, ops)
                self.static_copies[d[1]] = mask
                return term
        if r0 and r0[0] == 's':
            for t in ops[1:]:
                rr = reg_range(t)
                if rr and rr[0] == 's' and self.sym.get(rr[1]) == ('waveinfo',):
                    raise Unsupported('merged_wave_info used by %s' % base)
        if dname == 'null':
            # the null destination discards the result; only SCC would change, and the moves seen write none
            if base not in ('s_mov_b32', 's_mov_b64'):
                raise Unsupported('%s to null' % base)
            return None
        if not r0 and dname not in ('exec', 'exec_lo', 'vcc', 'vcc_lo'):
            raise Unsupported('%s to %s' % (base, dname))
        if self.dw_scalar(base, ops):
            return None
        if DATA_PAIRS and base == 's_mov_b64' and r0 and r0[0] == 's' and len(ops) == 2:
            src = reg_range(ops[1])
            if (src and src[0] == 's' and src[2] == src[1] + 1 and r0[2] == r0[1] + 1
                    and (src[1] not in self.masks or self.sym.get(src[2]) == ('conflict',))):
                # Copy both halves. A modeled lane mask keeps its lane bit in the low register;
                # an unresolved upper half remains unreadable as numeric data after the move.
                # If the ranges overlap, read the shared source before overwriting it.
                order = (1, 0) if r0[1] == src[2] else (0, 1)
                for offset in order:
                    self.scalar_inner('s_mov_b32', ['s%d' % (r0[1] + offset), 's%d' % (src[1] + offset)])
                return None
        return super().scalar(base, ops)

    def op_s_getpc_b64(self, ops, mods):
        # Keep the two address halves symbolic. Numeric uses remain unsupported,
        # since the guest code's original GPU virtual address is unknown.
        dst = reg_range(ops[0])
        if not dst or dst[0] != 's' or dst[2] != dst[1] + 1:
            raise Unsupported('getpc destination is not a scalar register pair')
        self.set_s(dst[1], None, ('code_lo', self.pc + 4))
        self.set_s(dst[2], None, ('code_hi',))

    def copied_constant(self, reg):
        """Follow SGPR copies over every predecessor to the load. Every reaching
        definition must agree on a literal. An entry input or other definition is unknown.
        """
        if self.is_const(reg) is not None:
            return self.is_const(reg)
        if not hasattr(self, '_literal_predecessors'):
            if any(name.startswith(('s_call', 's_setpc', 's_swappc', 's_rfe', 's_movrel'))
                   for _, _, name, _ in self.ins):
                return None
            self._literal_index = {off: i for i, (off, _, _, _) in enumerate(self.ins)}
            self._literal_predecessors = {off: [] for off in self._literal_index}
            self._literal_predecessors[self.ins[0][0]].append(None)
            for off, targets in self.range_successors().items():
                for target in targets:
                    self._literal_predecessors[target].append(off)
        todo = [(off, reg) for off in self._literal_predecessors[self.pc]]
        seen, values = set(), set()
        while todo:
            off, source = todo.pop()
            if off is None:
                return None
            if (off, source) in seen:
                continue
            seen.add((off, source))
            _, _, name, asm = self.ins[self._literal_index[off]]
            if source in (106, 107) and reads_writes(name, asm)[2]:
                return None
            if source in self.pair_writes(name, asm):
                ops = parse(asm)[1]
                if name not in ('s_mov_b32', 's_mov_b64', 's_movk_i32') or len(ops) != 2:
                    return None
                dst, src = map(reg_range, ops)
                if not dst or not dst[1] <= source <= dst[2]:
                    return None
                if src and src[0] == 's' and src[2] - src[1] == dst[2] - dst[1]:
                    source = src[1] + source - dst[1]
                elif name != 's_mov_b64' and re.fullmatch(r'0x[0-9a-f]+|-?[0-9]+', ops[1]):
                    literal = G.sopk_simm(ops[1]) if name == 's_movk_i32' else ops[1]
                    values.add(int(literal, 0) & 0xffffffff)
                    if len(values) > 1:
                        return None
                    continue
                else:
                    return None
            todo.extend((previous, source) for previous in self._literal_predecessors[off])
        return next(iter(values)) if len(values) == 1 else None

    def scalar_literal_index_bits(self, index):
        """Accept a contiguous BFE, LSHL-by-two, DWORD-load sequence with no
        intervening index writes or branches into the sequence after the BFE.
        """
        targets = self.branch_targets()
        i = next(i for i, ins in enumerate(self.ins) if ins[0] == self.pc)
        if self.pc in targets:
            return None
        i -= 1
        while i >= 0 and self.ins[i][2] == 's_buffer_load_dword':
            off, _, _, asm = self.ins[i]
            if off in targets or index == parse(asm)[1][0]:
                return None
            i -= 1
        if i < 1:
            return None
        off, _, shift, asm = self.ins[i]
        if off in targets or shift != 's_lshl_b32' or parse(asm)[1] != [index, index, '2']:
            return None
        _, _, extract, asm = self.ins[i - 1]
        ops = parse(asm)[1]
        if extract != 's_bfe_u32' or ops[0] != index or not re.fullmatch(r'0x[0-9a-f]+', ops[2]):
            return None
        return (int(ops[2], 0) >> 16) & 0x7f

    def scalar_literal_load(self, ops, mods, count, desc, base):
        register = desc[1]
        size = self.copied_constant(register + 2)
        if (count != 1 or desc[2] != register + 3 or self.sym.get(register + 1) != ('code_hi',)
                or self.copied_constant(register + 3) != 0x31016fac or not size or size % 4
                or base < 0 or base % 4 or base + size > len(self.raw)):
            raise Unsupported('incomplete scalar PC-relative raw buffer descriptor at %#x' % self.pc)
        if len(mods) > 1 or any(not re.fullmatch(r'offset:0x[0-9a-f]+', m) for m in mods):
            raise Unsupported('unsupported scalar literal table modifiers')
        extra = int(mods[0][7:], 0) if mods else 0
        bits = self.scalar_literal_index_bits(ops[2])
        if bits is None or bits > 30 or extra % 4 or extra + (1 << bits) * 4 > size:
            raise Unsupported('scalar literal table index unproven or outside descriptor at %#x' % self.pc)
        table = self.literal_tables.setdefault((base, size), 'x2w_literals_%x_%x' % (base, size))
        self.set_s(reg_range(ops[0])[1], '%s[(%s >> 2u) + %du]' % (table, self.u(ops[2]), extra // 4))

    def op_buffer_load_dword(self, ops, mods, count=1):
        desc = reg_range(ops[2])
        low = self.sym.get(desc[1]) if desc else None
        if not low or low[0] != 'code_lo':
            return super().op_buffer_load_dword(ops, mods, count)
        reg = desc[1]
        size = self.is_const(reg + 2)
        if (self.sym.get(reg + 1) != ('code_hi',) or self.is_const(reg + 3) != 0x31016fac
                or not size or size % 4 or low[1] % 4 or low[1] < 0 or low[1] + size > len(self.raw)):
            raise Unsupported('incomplete PC-relative raw buffer descriptor at %#x' % self.pc)
        offsets = [int(m[7:], 0) for m in mods if m.startswith('offset:')]
        if (count != 1 or ops[3] != '0' or 'offen' not in mods
                or any(m != 'offen' and not m.startswith('offset:') for m in mods)):
            raise Unsupported('PC-relative buffer load uses unsupported addressing')
        key = (low[1], size)
        table = self.literal_tables.setdefault(key, 'x2w_literals_%x_%x' % key)
        offset = sum(offsets)
        if offset % 4:
            raise Unsupported('unaligned literal table instruction offset')
        # RDNA2 raw DWORD addressing ignores the low two address bits.
        # Compare before adding the immediate so a large VGPR cannot wrap.
        self.emit('a = %s >> 2u;' % self.u(ops[1]))
        value = '0u' if offset >= size else '(a < %du ? %s[min(a, %du) + %du] : 0u)' % (
            (size - offset) // 4, table, (size - offset) // 4 - 1, offset // 4)
        self.set_v(reg_range(ops[0])[1], value)

    def op_s_bfe_u32(self, ops, mods):
        src = reg_range(ops[1])
        if src and src[0] == 's' and self.sym.get(src[1]) == ('waveinfo',):
            if ops[2] != '0x40018':
                raise Unsupported('merged_wave_info field %s' % ops[2])
            self.set_s(reg_range(ops[0])[1], 'wave')
            return None
        return self.scalar('s_bfe_u32', ops)

    def op_s_bfe_i32(self, ops, mods):
        # D = sext(S0 >> S1[4:0]) of width S1[22:16]; a width of 0 gives 0 (RDNA 2 ISA, S_BFE_I32).
        a, b = self.u(ops[1]), self.u(ops[2])
        expr = ('((({1} >> 16) & 127u) == 0u ? 0u : asuint(asint({0} << (32u - min(({1} & 31u) + (({1} >> 16) & 127u), 32u))) '
                '>> (32u - min(({1} & 31u) + (({1} >> 16) & 127u), 32u) + ({1} & 31u))))').format(a, b)
        self.set_s(reg_range(ops[0])[1], expr, scc='$ != 0')

    def op_s_bitset1_b32(self, ops, mods):
        d = reg_range(ops[0])[1]
        own = self.sym.get(d)
        if own and own[0] in ('ptrhi', 'descnum'):
            # a descriptor word built from a root pointer (stride / swizzle bits): keeps its descriptor meaning
            self.sym[d] = ('descnum',) if own[0] == 'descnum' else own
            return None
        self.set_s(d, '%s | (1u << (%s & 31u))' % (self.S(d), self.u(ops[1])))

    def op_s_bitset0_b32(self, ops, mods):
        # RDNA 2 ISA S_BITSET0_B32: D.u[S0.u[4:0]] = 0 (no SCC); a descriptor word is refused (no rule for it)
        d = reg_range(ops[0])[1]
        own = self.sym.get(d)
        if own and own[0] not in ('const', 'fixed_buffer_word'):
            raise Unsupported('s_bitset0_b32 on s%d holding %s' % (d, own[0]))
        self.set_s(d, '%s & ~(1u << (%s & 31u))' % (self.S(d), self.u(ops[1])))

    # s_pack_{lh,hl,hh}_b32_b16: D = low half from S0 (l = bits 15:0, h = bits 31:16), high half from S1
    # (RDNA 2 ISA, S_PACK_*). s_pack_ll is in the base table.
    def pack(self, ops, text):
        d = reg_range(ops[0])
        if not d or d[0] != 's':
            raise Unsupported('s_pack to %s' % ops[0])
        self.masks.discard(d[1])
        self.set_s(d[1], text.format(self.u(ops[1]), self.u(ops[2])))

    def op_s_pack_lh_b32_b16(self, ops, mods):
        self.pack(ops, '(({0} & 0xffffu) | ({1} & 0xffff0000u))')

    def op_s_pack_hl_b32_b16(self, ops, mods):
        self.pack(ops, '(({0} >> 16) | ({1} << 16))')

    def op_s_pack_hh_b32_b16(self, ops, mods):
        self.pack(ops, '(({0} >> 16) | ({1} & 0xffff0000u))')

    # v_min3 / v_max3 / v_med3 on integers (RDNA 2 ISA): three-way min, max, median
    def three(self, ops, fn, signed):
        a, b, c = [self.u(o) for o in ops[1:4]]
        if signed:
            a, b, c = ['asint(%s)' % x for x in (a, b, c)]
        expr = {'min': 'min(min({0}, {1}), {2})', 'max': 'max(max({0}, {1}), {2})',
                'med': 'max(min({0}, {1}), min(max({0}, {1}), {2}))'}[fn].format(a, b, c)
        self.set_v(reg_range(ops[0])[1], 'asuint(%s)' % expr if signed else expr)

    def op_v_min3_u32(self, ops, mods):
        self.three(ops, 'min', False)

    def op_v_max3_u32(self, ops, mods):
        self.three(ops, 'max', False)

    def op_v_med3_u32(self, ops, mods):
        self.three(ops, 'med', False)

    def op_v_min3_i32(self, ops, mods):
        self.three(ops, 'min', True)

    def op_v_max3_i32(self, ops, mods):
        self.three(ops, 'max', True)

    def op_v_med3_i32(self, ops, mods):
        self.three(ops, 'med', True)

    SEL = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8), 'WORD_0': (0, 16),
           'WORD_1': (16, 16), 'DWORD': (0, 32)}

    def sdwa_src(self, tok, sel):
        sx = tok.startswith('sext(') and tok.endswith(')')
        u = self.u(tok[5:-1] if sx else tok)
        sh, w = self.SEL[sel]
        if w == 32:
            return u
        if sx:
            return 'asuint(asint(%s << %du) >> %du)' % (u, 32 - sh - w, 32 - w)
        return '((%s >> %du) & 0x%xu)' % (u, sh, (1 << w) - 1)

    def op_v_mov_b32_sdwa(self, ops, mods):
        """D = the selected field of S0 placed at dst_sel; other bits zero (UNUSED_PAD) or kept (UNUSED_PRESERVE)
        (RDNA 2 ISA, SDWA)."""
        f = dict(m.partition(':')[::2] for m in mods)
        if set(f) - {'dst_sel', 'dst_unused', 'src0_sel'} or len(ops) != 2:
            raise Unsupported('v_mov_b32_sdwa with %s' % mods)
        d = reg_range(ops[0])[1]
        sh, w = self.SEL[f.get('dst_sel', 'DWORD')]
        val = '((%s & 0x%xu) << %du)' % (self.sdwa_src(ops[1], f.get('src0_sel', 'DWORD')), (1 << w) - 1 if w < 32 else 0xffffffff, sh)
        unused = f.get('dst_unused', 'UNUSED_PAD')
        if unused == 'UNUSED_PRESERVE':
            keep = 0xffffffff & ~((((1 << w) - 1) << sh))
            val = '(%s | (v%d & 0x%08xu))' % (val, d, keep)
            self.used_v.add(d)
        elif unused != 'UNUSED_PAD':
            raise Unsupported('v_mov_b32_sdwa %s' % unused)
        self.set_v(d, val)

    def op_v_cmp_class_f32_sdwa(self, ops, mods):
        """V_CMP_CLASS_F32 (SDWA, VOPC into an SGPR mask): bit class(S0) of the selected S1 field; class bits 0 sNaN,
        1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf (RDNA 2 ISA)."""
        f = dict(m.partition(':')[::2] for m in mods)
        if set(f) - {'src0_sel', 'src1_sel'} or f.get('src0_sel', 'DWORD') != 'DWORD' or len(ops) != 3:
            raise Unsupported('v_cmp_class_f32_sdwa with %s %s' % (ops, mods))
        x = self.u(ops[1])
        mask = self.sdwa_src(ops[2], f.get('src1_sel', 'DWORD'))
        return self.write_compare(False, ops[0], '((%s >> x2w_fclass(%s)) & 1u)' % (mask, x))

    def op_v_cmp_class_f16_sdwa(self, ops, mods):
        """V_CMP_CLASS_F16 (SDWA, VOPC into an SGPR mask): the selected 16-bit field of S0 with its abs/neg applied
        to bit 15 (as the e64 form, base x2w_class_f16), S1 the class mask (DWORD)."""
        f = dict(m.partition(':')[::2] for m in mods)
        if set(f) - {'src0_sel', 'src1_sel'} or f.get('src1_sel', 'DWORD') != 'DWORD' or len(ops) != 3:
            raise Unsupported('v_cmp_class_f16_sdwa with %s %s' % (ops, mods))
        tok = ops[1]
        neg, ab = tok.startswith('-'), '|' in tok
        reg = tok.strip('-|')
        if not reg_range(reg) or f.get('src0_sel', 'DWORD') not in ('WORD_0', 'WORD_1', 'DWORD'):
            raise Unsupported('v_cmp_class_f16_sdwa source %s %s' % (tok, mods))
        x = '(%s & 0xffffu)' % self.sdwa_src(reg, f.get('src0_sel', 'DWORD'))
        if ab:
            x = '(%s & 0x7fffu)' % x
        if neg:
            x = '(%s ^ 0x8000u)' % x
        return self.write_compare(False, ops[0], 'x2w_class_f16(%s, %s)' % (x, self.u(ops[2])))

    def compare(self, base, ops, mods=()):
        """16-bit integer compares (v_cmp[x]_<cond>_{i16,u16}, e32/e64 without modifiers): the low halves,
        sign- or zero-extended, compared as 32-bit integers; everything else as the base translator."""
        m = re.fullmatch(r'v_cmp(x?)_(\w+?)_(i16|u16)', base)
        if not m:
            return super().compare(base, ops, mods)
        if mods or any(t.startswith(('-', '|')) for t in ops):
            raise Unsupported('%s with modifiers %s' % (base, mods))
        writes_exec, cond, typ = m.group(1) == 'x', m.group(2), m.group(3)
        if len(ops) == 3:
            dst, a, b = ops
        elif len(ops) == 2:
            dst, (a, b) = ('exec' if writes_exec else 'vcc'), ops
        else:
            raise Unsupported('%s with %d operands' % (base, len(ops)))
        if typ == 'i16':
            x, y = ('(asint(%s << 16u) >> 16)' % self.u(t) for t in (a, b))
        else:
            x, y = ('(%s & 0xffffu)' % self.u(t) for t in (a, b))
        return self.compare_result(base.replace(typ, 'i32' if typ == 'i16' else 'u32'), cond, writes_exec, dst, x, y)

    def image(self, ops, mods, how):
        """d16 image results: the channels the base computes as binary32 are packed two per VGPR as binary16,
        the first in the low half, the unused half of an odd last register 0 (gfx10 packed D16). Only for the
        float views the base binds (Texture*<float4>); the base refuses the other result modifiers."""
        if 'd16' not in mods:
            return super().image(ops, mods, how)
        dmask = int([m for m in mods if m.startswith('dmask:')][0][6:], 0)
        n = bin(dmask & 15).count('1')
        dst = reg_range(ops[0])
        if not dst or dst[0] != 'v' or dst[2] - dst[1] + 1 != (n + 1) // 2:
            raise Unsupported('d16 image result %s for dmask %#x' % (ops[0], dmask))
        tmp = 1000                                           # HLSL-only scratch VGPRs (the console has 256)
        super().image([('v[%d:%d]' % (tmp, tmp + n - 1)) if n > 1 else 'v%d' % tmp] + list(ops[1:]), [m for m in mods if m != 'd16'], how)
        for j in range((n + 1) // 2):
            lo = 'f32tof16(asfloat(v%d))' % (tmp + 2 * j)
            hi = '(f32tof16(asfloat(v%d)) << 16)' % (tmp + 2 * j + 1) if 2 * j + 1 < n else '0u'
            self.set_v(dst[1] + j, '(%s | %s)' % (lo, hi))

    def op_v_bfm_b32(self, ops, mods):
        """V_BFM_B32: D = ((1 << S0[4:0]) - 1) << S1[4:0] (RDNA 2 ISA)."""
        if mods or len(ops) != 3:
            raise Unsupported('v_bfm_b32 form %s %s' % (ops, mods))
        self.set_v(reg_range(ops[0])[1], '(((1u << (%s & 31u)) - 1u) << (%s & 31u))' % (self.u(ops[1]), self.u(ops[2])))

    def carry_op(self, base, ops, mods):
        """v_add_co_u32 / v_sub_co_u32 / v_subrev_co_u32 and the _ci forms: D, carry-out mask, S0, S1[, carry-in mask].
        Add: sum = S0 + S1 + cin (mod 2^32), carry = cin ? sum <= S0 : sum < S0. Sub: d = S0 - S1 - bin, borrow =
        bin ? d >= S0 : d > S0 (subrev swaps the sources). Both read only S0, the result and the carry-in, so the
        carry-in may be the destination mask. The carry is the lane's bit of the destination mask."""
        m = CARRY.fullmatch(base)
        if mods or len(ops) != (5 if m.group(2) else 4):
            raise Unsupported('%s form %s %s' % (base, ops, mods))
        x, y = self.u(ops[2]), self.u(ops[3])
        if m.group(1) == 'subrev':
            x, y = y, x
        cin = '(%s & 1u)' % self.u(ops[4]) if m.group(2) else '0u'
        if m.group(1) == 'add':
            self.emit('a = %s; w = a + %s + %s;' % (x, y, cin))
            carry = '((%s != 0u ? w <= a : w < a) ? 1u : 0u)' % cin
        else:
            self.emit('a = %s; w = a - %s - %s;' % (x, y, cin))
            carry = '((%s != 0u ? w >= a : w > a) ? 1u : 0u)' % cin
        dst = ops[1]
        if dst.startswith('vcc'):
            dst = 'vcc'
        else:
            r = reg_range(dst)
            if not r or r[0] != 's':
                raise Unsupported('%s carry to %s' % (base, dst))
            dst = 's%d' % r[1]
        self.write_compare(False, dst, carry)
        self.set_v(reg_range(ops[0])[1], 'w')

    def op_v_mbcnt_lo_u32_b32(self, ops, mods):
        self.set_v(reg_range(ops[0])[1], 'x2w_mbcnt_lo(%s, lane) + %s' % (self.u(ops[1]), self.u(ops[2])))

    def op_v_mbcnt_hi_u32_b32(self, ops, mods):
        self.set_v(reg_range(ops[0])[1], 'x2w_mbcnt_hi(%s, lane) + %s' % (self.u(ops[1]), self.u(ops[2])))

    def op_v_perm_b32(self, ops, mods):
        if mods:
            raise Unsupported('v_perm_b32 with %s' % mods)
        self.set_v(reg_range(ops[0])[1], 'x2w_perm(%s, %s, %s)' % (self.u(ops[1]), self.u(ops[2]), self.u(ops[3])))

    def op_v_pack_b32_f16(self, ops, mods):
        sel = [0, 0, 0]
        for m in mods:
            if m.startswith('op_sel:['):
                sel = [int(x) for x in m[8:-1].split(',')]
            else:
                raise Unsupported('v_pack_b32_f16 with %s' % m)
        if any(t.startswith(('-', '|')) for t in ops[1:3]):
            raise Unsupported('v_pack_b32_f16 with source modifiers')
        lo = '(%s >> 16)' % self.h(ops[1]) if sel[0] else self.h(ops[1])
        hi = '(%s >> 16)' % self.h(ops[2]) if sel[1] else self.h(ops[2])
        self.set_v(reg_range(ops[0])[1], '(%s & 0xffffu) | ((%s & 0xffffu) << 16)' % (lo, hi))

    def op_v_readfirstlane_b32(self, ops, mods):
        src, dst = reg_range(ops[1]), reg_range(ops[0])
        if mods or not dst or dst[0] != 's' or not src or src[0] != 'v':
            raise Unsupported('v_readfirstlane_b32 form %s' % ops)
        self.masks.discard(dst[1])   # a number now, not a lane mask (s_or of it must not be masked to bit 0)
        if src[1] == 5 and 5 not in self.written_v:
            # v5 = group * 128 + thread (inferred fast-launch vertex id); the wave's first lane is its thread & ~63
            # (`wbase` = the wave's first thread: the same in every lane of a wave, so not lane-tainted)
            self.set_s(dst[1], '(gid * %du + wbase)' % GROUP)
            return None
        shadow = self.shadow.get(src[1])
        read_mask = ALL_LANES if self.exec_known else self.exec_static
        if shadow and read_mask:
            first = (read_mask & -read_mask).bit_length() - 1
            if (shadow[1] >> first) & 1:
                # the first active lane wrote this wave-uniform value: every thread takes it from the shadow
                self.set_s(dst[1], shadow[0])
                return None
        # A source that is the same in every active lane: each active lane's own value is the first active lane's
        # (the result is then active-uniform itself, through emit).
        if self.lane_dependent('v%d' % src[1]):
            # Lane-dependent source: the console wave reads its first active lane. With every lane active and
            # no per-lane region open (so every thread of the group reaches this statement, a group barrier is
            # legal), that lane is the wave's first thread: broadcast through groupshared memory.
            if not self.exec_known or self.pc < self.divergent_until:
                text = 'v_readfirstlane_b32 of a lane-dependent v%d with lanes off at %#x (%s)' % (
                    src[1], self.pc, 'in a per-lane range' if self.pc < self.divergent_until else 'partial exec')
                if not RFL_FIRST:
                    raise Unsupported(text)
                if any(t <= self.pc < o for o, t in self.backward_branches()):
                    # inside a loop: the trip count may differ between the waves of a group, so a group barrier
                    # there is not reached by every thread the same number of times (a waterfall loop over a
                    # per-lane index is mesh4's lowering, without a barrier)
                    raise Unsupported(text.replace('lanes off at', 'lanes off inside a loop at'))
                if self.pc < self.divergent_until:
                    # not every thread reaches the read: ask for the innermost execz range around it to be run by
                    # the whole group (once; a range already run so, or a per-lane range of another kind, refuses)
                    around = [o for o, t in self.execz_ranges() if o < self.pc < t]
                    if around and max(around) not in self.group_branches():
                        raise NeedGroupRange(max(around), text)
                    raise Unsupported(text)
                return self.first_active_broadcast(src, dst, ops)
            # The group barrier is a phase boundary of the body (see body()): the read starts the next block, so
            # that block must not also be entered by a branch.
            nxt = self.next_offset()
            if nxt in self.branch_targets():
                raise Unsupported('v_readfirstlane_b32 broadcast followed by a branch target at %#x' % self.pc)
            slot = self.broadcasts
            self.broadcasts += 1
            self.emit('if (lane == 0u) x2w_rfl[%d][wave] = %s;' % (slot, self.u(ops[1])))
            self.sym.pop(dst[1], None)
            self.used_s.add(dst[1])
            self.pending.append('s%d = x2w_rfl[%d][wave];' % (dst[1], slot))
            self.tainted.discard('s%d' % dst[1])
            self.au.pop('s%d' % dst[1], None)
            self.au_xs.pop('s%d' % dst[1], None)
            self.lds_written_in_phase = False
            return ('barrier',)
        self.set_s(dst[1], self.u(ops[1]))
        if self.is_tainted('v%d' % src[1]):
            self.au_scalars[dst[1]] = self.epoch

    def backward_branches(self):
        """(branch offset, target) of every branch whose target is not after it (the loops of the program)."""
        out = []
        for o, n, name, asm in self.ins:
            if name.startswith(('s_branch', 's_cbranch_')):
                args = parse(asm)[1]
                if args and args[0].startswith('label_'):
                    t = int(args[0].split('_')[1], 16)
                    if t <= o:
                        out.append((o, t))
        return out

    def execz_ranges(self):
        """(branch offset, target) of the forward s_cbranch_execz ranges of the program."""
        out = []
        for o, n, name, asm in self.ins:
            if name == 's_cbranch_execz':
                t = int(parse(asm)[1][0].split('_')[1], 16)
                if t > o:
                    out.append((o, t))
        return out

    def first_active_broadcast(self, src, dst, ops):
        """v_readfirstlane_b32 sD, vS of a lane-dependent vS under a partial exec, reached by every thread of the
        group (no per-lane range open). The console wave reads the lane with the lowest index whose EXEC bit is set
        (RDNA 2 ISA V_READFIRSTLANE_B32; with EXEC = 0 it reads lane 0 [ASSUMED from the ISA text, not measured]).
        Every thread stores its vS (an inactive lane's VGPR keeps its old value, as here) and its exec bit in
        groupshared memory, a group barrier follows, and the next block reads the first set bit's value of the
        thread's own wave (x2w_rfirst). Refused: more than RFL_FIRST_SLOTS such reads, and a branch target right
        after the read (the barrier is a phase boundary of the body, as for the lane-0 broadcast)."""
        nxt = self.next_offset()
        if nxt in self.branch_targets():
            raise Unsupported('v_readfirstlane_b32 broadcast followed by a branch target at %#x' % self.pc)
        if self.first_slots >= RFL_FIRST_SLOTS:
            raise Unsupported('more than %d first-active-lane v_readfirstlane_b32 reads' % RFL_FIRST_SLOTS)
        k = self.first_slots
        self.first_slots += 1
        self.emit('x2w_rfv[%d][tid] = %s;' % (k, self.u(ops[1])))
        self.emit('x2w_rfe[%d][tid] = exec & 1u;' % k)
        self.sym.pop(dst[1], None)
        self.used_s.add(dst[1])
        self.pending.append('s%d = x2w_rfirst(%du, wbase);' % (dst[1], k))
        self.tainted.discard('s%d' % dst[1])
        self.au.pop('s%d' % dst[1], None)
        self.au_xs.pop('s%d' % dst[1], None)
        self.lds_written_in_phase = False
        return ('barrier',)

    def op_s_buffer_load_dword(self, ops, mods, count=1):
        """As the pipeline translator, plus: the offset is read once before the destination is written (the
        title's `s_buffer_load_dwordx2 s[4:5], s[16:19], s4` overwrote s4 before the second word), the
        instruction's offset:N is added, and the address is dword aligned (the hardware ignores bits 1:0)."""
        G.smem_imm_refuse('s_buffer_load', ops, mods)
        desc = reg_range(ops[1])
        origin = self.sym.get(desc[1]) if desc else None
        if origin and origin[0] == 'code_lo':
            return self.scalar_literal_load(ops, mods, count, desc, origin[1])
        extra = 0
        for m in mods:
            if m.startswith('offset:'):
                extra += int(m[7:], 0)
            else:
                raise Unsupported('s_buffer_load with %s' % m)
        dst = reg_range(ops[0])
        buf = self.pipe.resource(self.descriptor(ops[1]), 'buf')
        fixed = re.fullmatch(r'0x[0-9a-f]+|\d+', ops[2])
        if fixed:
            base = '%du' % ((int(ops[2], 0) + extra) & ~3)
        else:
            self.emit('a = (%s + %du) & ~3u;' % (self.u(ops[2]), extra))
            base = 'a'
        for i in range(count):
            self.set_s(dst[1] + i, '%s.Load(%s + %d)' % (buf, base, 4 * i))
            if fixed:
                self.sym[dst[1] + i] = ('fixed_buffer_word',)

    # Typed buffer formats the pipeline translator's table lacks. SNORM: max(x / (2^(n-1) - 1), -1) (RDNA 2 ISA
    # data-format conversion). 2_10_10_10 lists fields from the most significant: x is bits 9:0 and w bits 31:30
    # (Mesa gfx10_format_table.py maps R10G10B10A2 to 2_10_10_10).
    MESH_FORMATS = {
        'BUF_FMT_16_16_SNORM': ('ldu({0}, a)', ['U(max(float(asint(w << 16) >> 16) / 32767.0, -1.0))',
                                                'U(max(float(asint(w) >> 16) / 32767.0, -1.0))', '0u', 'U(1.0)']),
        'BUF_FMT_16_16_16_16_SNORM': ('ldu({0}, a)', ['U(max(float(asint(w << 16) >> 16) / 32767.0, -1.0))',
                                                      'U(max(float(asint(w) >> 16) / 32767.0, -1.0))',
                                                      'U(max(float(asint(ldu({0}, a + 4) << 16) >> 16) / 32767.0, -1.0))',
                                                      'U(max(float(asint(ldu({0}, a + 4)) >> 16) / 32767.0, -1.0))']),
        'BUF_FMT_16_16_16_16_UNORM': ('ldu({0}, a)', ['U((w & 65535) / 65535.0)', 'U((w >> 16) / 65535.0)',
                                                      'U((ldu({0}, a + 4) & 65535) / 65535.0)',
                                                      'U((ldu({0}, a + 4) >> 16) / 65535.0)']),
        'BUF_FMT_8_8_8_8_SNORM': ('ldu({0}, a)', ['U(max(float(asint(w << 24) >> 24) / 127.0, -1.0))',
                                                  'U(max(float(asint(w << 16) >> 24) / 127.0, -1.0))',
                                                  'U(max(float(asint(w << 8) >> 24) / 127.0, -1.0))',
                                                  'U(max(float(asint(w) >> 24) / 127.0, -1.0))']),
        'BUF_FMT_2_10_10_10_UNORM': ('ldu({0}, a)', ['U((w & 1023) / 1023.0)', 'U(((w >> 10) & 1023) / 1023.0)',
                                                     'U(((w >> 20) & 1023) / 1023.0)', 'U((w >> 30) / 3.0)']),
        # integer formats: the register gets the integer; missing components read 0, w reads 1 (as the float
        # formats' 1.0)
        'BUF_FMT_8_8_8_8_UINT': ('ldu({0}, a)', ['w & 255u', '(w >> 8) & 255u', '(w >> 16) & 255u', 'w >> 24']),
        'BUF_FMT_8_8_UINT': ('ldu({0}, a)', ['w & 255u', '(w >> 8) & 255u', '0u', '1u']),
        'BUF_FMT_16_16_UINT': ('ldu({0}, a)', ['w & 65535u', 'w >> 16', '0u', '1u']),
        'BUF_FMT_16_UINT': ('ldu({0}, a)', ['w & 65535u', '0u', '0u', '1u']),
        'BUF_FMT_16_16_16_16_UINT': ('ldu({0}, a)', ['w & 65535u', 'w >> 16', 'ldu({0}, a + 4) & 65535u',
                                                     'ldu({0}, a + 4) >> 16']),
        'BUF_FMT_32_UINT': ('ldu({0}, a)', ['w', '0u', '0u', '1u']),
        'BUF_FMT_32_32_UINT': ('ldu({0}, a)', ['w', 'ldu({0}, a + 4)', '0u', '1u']),
        'BUF_FMT_32_32_32_32_UINT': ('ldu({0}, a)', ['w', 'ldu({0}, a + 4)', 'ldu({0}, a + 8)', 'ldu({0}, a + 12)']),
    }

    def tbuffer(self, ops, mods, count):
        formats = [m for m in mods if m.startswith('format:')]
        fmt = formats[0][8:-1] if formats else None
        if fmt not in self.MESH_FORMATS:
            return super().tbuffer(ops, mods, count)
        dst = reg_range(ops[0])
        buf, address = self.buffer_address(ops, [m for m in mods if not m.startswith('format:')])
        self.emit('a = %s;' % address)
        load, comps = self.MESH_FORMATS[fmt]
        self.emit('w = %s;' % load.format(buf))
        for i in range(count):
            self.set_v(dst[1] + i, comps[i].format(buf))

    def export(self, asm):
        m = re.fullmatch(r'exp (\w+) (.*?)((?: done| compr| vm)*)', asm)
        if not m:
            raise Unsupported('export form %r' % asm)
        target, vals = m.group(1), [v.strip() for v in m.group(2).split(',')]
        if target in ('prim', 'target_20'):
            if vals[0] == 'off' or any(v != 'off' for v in vals[1:]) or 'compr' in m.group(3):
                raise Unsupported('primitive export form %r' % asm)
            self.exported_prim = True
            self.emit('if (exec & 1) { x2w_prim = %s; }' % self.u(vals[0]))
            return None
        if target == 'pos1' and self.pipe.vs_out_cntl is None:
            # what pos1 carries (misc vector or clip/cull distances, and which channels) is pipeline state
            raise Unsupported('mesh export target pos1 without PA_CL_VS_OUT_CNTL (request .vsout)')
        if target not in ('pos0', 'pos1') and not target.startswith('param'):
            raise Unsupported('mesh export target %s' % target)
        before = len(self.lines)
        self.kind = 'vs'  # the base export only accepts pos/param outside a pixel stage
        try:
            super().export(asm)
        finally:
            self.kind = 'ms'
        body = self.lines[before:]
        del self.lines[before:]
        self.lines.append('if (exec & 1) { %s }' % ' '.join(body))
        self.exported_vertex = True
        return None


for _op in ('add', 'sub', 'subrev'):
    for _ci in ('', '_ci'):
        _name = 'v_%s_co%s_u32' % (_op, _ci)
        setattr(MeshStage, 'op_' + _name, lambda self, ops, mods, _n=_name: self.carry_op(_n, ops, mods))


def const_heap_sampler(stage, tok):
    """'qh[<entry>]' for a sampler operand whose four words are one sampler-heap entry loaded at a constant byte
    offset (`s_load_dwordx4 s[..], s[<sampler heap>], 0x410`: a fixed entry of the title's bound sampler heap, as
    the run programs read it), else None.

    The same table and manifest contract as the run-time offset form (gfx2hlsl heap_sampler; `smpheap 2048 1
    fallback 2047`: entry n = the title's heap entry at byte 16 * n, an unaligned or out-of-range offset takes the
    fallback entry), so the stand-in binds it as for a draw. Opt-in with X2W_SAMPLER_HEAP=1 as that form is.
    Refused: words edited in place after the load, or words from different loads."""
    r = reg_range(tok)
    if not r or r[0] != 's' or r[2] - r[1] != 3:
        return None
    first = stage.sym.get(r[1])
    if not first or first[0] != 'desc' or first[1] != ('smpheap',) or first[2] is None:
        return None
    if not stage.pipe.sampler_heap:
        raise Unsupported('sampler from the sampler heap at constant offset %#x (X2W_SAMPLER_HEAP=1 binds the heap)'
                          % (first[2] + 4 * first[3]))
    if any(stage.sym.get(r[1] + i) != ('desc', first[1], first[2], first[3] + i) for i in range(4)):
        raise Unsupported('sampler operand %s mixes sampler heap words' % tok)
    if (first[1], first[2]) in stage.pipe.modified:
        raise Unsupported('sampler heap entry at %#x edited in the shader' % (first[2] + 4 * first[3]))
    offset = first[2] + 4 * first[3]
    entry = offset >> 4 if offset % 16 == 0 and offset < 16 * G.SAMPLER_HEAP_FALLBACK else G.SAMPLER_HEAP_FALLBACK
    stage.pipe.uses_sampler_heap = True
    return 'qh[%du]' % entry


class MeshPixelStage(G.Stage):
    """The pixel stage of a mesh pipeline: the base pixel stage plus constant sampler-heap entries."""

    def heap_sampler(self, tok):
        return super().heap_sampler(tok) or const_heap_sampler(self, tok)


class MeshPipeline(G.Pipeline):
    """Mesh + pixel pipeline. ps_code None: the mesh stage alone (coverage runs; params = what it exports)."""

    def __init__(self, ms_code, ps_code, root_blob, sampler_heap=None, rtv_formats=(), user_sgpr=MESH_USER_SGPR,
                 ps_hw_inputs=None, vs_out_cntl=None):
        self.root = G.parse_root(root_blob)
        # First user SGPR: s16 for the run programs (measured, real roots). Archive programs (offline, synthesized
        # root) may start at s8: callers pass entry_sgpr(code) together with synth_root(code).
        self.user_sgpr = user_sgpr
        # PA_CL_VS_OUT_CNTL of the pipeline (request .vsout; regblob .vsregs u32 at +0x98, +0x54 in mesh blocks),
        # or None: what a pos1 export carries (G.clip_vector_width). None refuses pos1.
        self.vs_out_cntl = vs_out_cntl
        # Lane-mask SCCs to publish per wave: found by translating, refusing at the branch with the offsets of the
        # mask instructions the SCC comes from (LaneScc), and translating again with those published.
        self.scc_publish = frozenset()
        for _ in range(8):
            try:
                self.build(ms_code, ps_code, sampler_heap, rtv_formats, ps_hw_inputs)
                return
            except LaneScc as refusal:
                if not LANE_SCC or not refusal.origins or refusal.origins <= self.scc_publish:
                    raise Unsupported(str(refusal))
                if min(refusal.origins) < 0:
                    raise Unsupported('%s (through a non-mask op on a lane mask at %#x)'
                                      % (refusal, -min(refusal.origins) - 1))
                self.scc_publish |= refusal.origins
        raise Unsupported('lane-mask SCC: no fixed point after 8 translations')

    def build(self, ms_code, ps_code, sampler_heap, rtv_formats, ps_hw_inputs=None):
        # Fields of G.Pipeline.__init__ (which this does not call: it builds a vertex stage) that newer base
        # translators read. The draw lowerings (manifest `heap 1`, `udsrc bufword3`, `uav`) are named in
        # X2W_DRAW_FEATURES as for draws (pstrans gfx2hlsl.draw_features); a base without them keeps refusing.
        self.features = G.draw_features() if hasattr(G, 'draw_features') else frozenset()
        self.heapvars = []
        self.heapwords = False   # x2w_hword used (manifest `heapwords 1`)
        self.rw_keys = frozenset()
        # the pixel stage's SPI inputs (SPI_PS_INPUT_ENA/ADDR, request .psin), as for draws
        self.spi_inputs(ps_hw_inputs)
        self.quad_ops = True  # the pixel stage compiles with dxc ps_6_0: DPP quad reads (gfx2hlsl Stage.quad_dpp)
        self.rtv_formats = rtv_formats
        self.sampler_heap = os.environ.get('X2W_SAMPLER_HEAP') == '1' if sampler_heap is None else sampler_heap
        self.uses_sampler_heap = False
        self.resources, self.samplers, self.uds = {}, {}, []
        self.notes = set()
        self.modified = set()
        self.helper_text = None
        self.rect = self.ud(('rect',))
        self.dw_src, self.dw_used, self.dw_text = [], set(), {}   # descword placeholders (MeshStage.dw_word)
        # A lane-dependent v_readfirstlane_b32 inside a per-lane execz range asks for that range to be run by the
        # whole group (NeedGroupRange): translate again from the same state with it added (bounded: each range once).
        self.force_group = frozenset()
        fresh = copy.deepcopy(self.__dict__)
        while True:
            try:
                self.vs = self.mesh_stage(ms_code)
                break
            except NeedGroupRange as e:
                if e.off in self.force_group:
                    raise Unsupported(str(e))
                force = self.force_group | {e.off}
                self.__dict__.clear()
                self.__dict__.update(copy.deepcopy(fresh))
                self.force_group = force
        # descword placeholders: a word some numeric use reached is an entry value (`udsrc <n> descword <p> <byte>`),
        # any other is dead arithmetic and reads 0u
        self.dw_text = {k: ('ud%d' % self.ud(src) if k in self.dw_used else '0u') for k, src in enumerate(self.dw_src)}
        self.ps = MeshPixelStage(self, 'ps', ps_code) if ps_code is not None else None
        self.params = sorted(int(k[5:]) for k in self.vs.exports if k.startswith('param'))
        if self.ps is not None:
            self.spi_reads()
            self.flat = sorted(a for a, how in self.ps.attrs.items() if 'flat' in how)
            self.half = sorted(a for a, how in self.ps.attrs.items() if 'half' in how)
            self.check_half_reads()
            missing = [a for a in self.ps.attrs if a not in self.params]
            if missing:
                raise Unsupported('pixel stage reads attributes %s the mesh stage does not export' % missing)
        else:
            self.ps_entry_reads, self.flat, self.half = {}, [], []
        if 'pos' not in self.vs.exports:
            raise Unsupported('mesh stage exports no position')
        if not self.vs.exported_prim:
            raise Unsupported('mesh stage exports no primitive')

    def mesh_stage(self, ms_code):
        """The mesh stage, translated again (from the same pipeline state) until every loop head's state is the
        join of its edges (MeshStage.loop_settled); refused when that does not settle within LOOP_PASSES."""
        self.loop_seed = {}
        before = copy.deepcopy(self.__dict__)
        for _ in range(LOOP_PASSES):
            try:
                return MeshStage(self, ms_code)
            except LoopRetry as retry:
                seeds = retry.backs
            self.__dict__.clear()
            self.__dict__.update(copy.deepcopy(before))
            self.loop_seed = seeds
        raise Unsupported('loop state does not settle in %d passes' % LOOP_PASSES)

    def dw_sub(self, lines):
        """The mesh stage's lines with the descword placeholders replaced (build())."""
        dw = getattr(self, 'dw_text', {})
        if not dw:
            return lines
        out = [re.sub(r'\bX2W_DW(\d+)\b', lambda m: dw[int(m.group(1))], line) for line in lines]
        if any('X2W_DW' in line for line in out):
            raise Unsupported('descword placeholder left in the mesh stage')
        return out

    def common(self):
        if self.helper_text is None:
            self.helper_text = '\n'.join(self.dw_sub(self.vs.body()) + (self.ps.body() if self.ps else []))
        return super().common()

    def hlsl_ms(self, dump=False):
        """The mesh shader. dump=True (tests only): also store each thread's outputs and the group's allocation
        word in RWByteAddressBuffer u0 (per group: 16 bytes header [alloc], then per thread pos, param0..N-1,
        primitive word, then the clip distances when pos1 is exported) and give each primitive SV_PrimitiveID =
        group * 128 + index."""
        s = self.vs
        nparams = max(self.params) + 1 if self.params else 0
        nclip = s.exports.get('clip') or 0
        stride = 4 * (4 + 4 * nparams + 1 + nclip)
        out = self.common()
        body_text = self.helper_text
        if self.heapwords and not any(self.heapvars):
            # descriptor words read, but no heap view used as a resource: the base emitted no heap declarations
            out += ['ByteAddressBuffer heapmem : register(t0, space1);',
                    'uint x2w_vid(uint h) { return heapmem.Load(h) == 0x%08xu ? heapmem.Load(h + 4) : 0xffffffffu; }'
                    % G.HEAP_TOKEN]
        out += [text for name, text in MESH_HELPERS.items() if re.search(r'\b%s\(' % name, body_text)]
        out += ['struct MSOut {'] + self.interface(False) + ['};',
                'struct MSPrim { bool cull : SV_CullPrimitive;%s };' % (' uint pid : SV_PrimitiveID;' if dump else '')] + (
                ['RWByteAddressBuffer x2w_dump : register(u0);'] if dump else []) + [
                'groupshared uint x2w_alloc;'] + (
                ['groupshared uint x2w_rfl[%d][2];  // v_readfirstlane broadcasts, one slot per instruction' % s.broadcasts]
                if s.broadcasts else []) + (
                ['groupshared uint x2w_rfv[%d][%d];  // first-active-lane reads: each thread\'s value' % (s.first_slots, GROUP),
                 'groupshared uint x2w_rfe[%d][%d];  // ... and its exec bit' % (s.first_slots, GROUP),
                 '// v_readfirstlane_b32 under a partial exec: the lowest lane of the wave (first thread w0) with its',
                 '// exec bit set, lane 0 when none is (MeshStage.first_active_broadcast)',
                 'uint x2w_rfirst(uint k, uint w0) {',
                 '    for (uint i = 0u; i < 64u; i++) { if (x2w_rfe[k][w0 + i] != 0u) return x2w_rfv[k][w0 + i]; }',
                 '    return x2w_rfv[k][w0];',
                 '}'] if s.first_slots else []) + (
                ['groupshared uint x2w_lds[%d];  // LDS (ds_read / ds_write), dword addressed' % LDS_DWORDS]
                if s.lds_used else []) + (
                ['groupshared uint x2w_any[%d][2];  // wave-level execz branches: some lane of the wave active'
                 % len(s.wave_any)] if s.wave_any else []) + (
                ['groupshared uint x2w_sany[%d][2];  // lane-mask SCCs: some lane of the wave has its result bit set'
                 % len(s.scc_any)] if s.scc_any else []) + [
                '[outputtopology("triangle")]',
                '[numthreads(%d, 1, 1)]' % GROUP,
                'void main(uint tid : SV_GroupIndex, uint3 group : SV_GroupID,',
                '          out vertices MSOut verts[%d], out indices uint3 tris[%d], out primitives MSPrim prims[%d]) {'
                % (MAX_VERTICES, MAX_PRIMITIVES, MAX_PRIMITIVES),
                '    uint gid = group.x, lane = tid & 63u, wave = tid >> 6, wbase = tid & ~63u;',
                '    if (tid == 0) x2w_alloc = 0;'] + (
                ['    if (tid < %du) x2w_any[tid >> 1][tid & 1u] = 0u;' % (2 * len(s.wave_any))] if s.wave_any else []) + (
                ['    if (tid < %du) x2w_sany[tid >> 1][tid & 1u] = 0u;' % (2 * len(s.scc_any))] if s.scc_any else []) + [
                '    GroupMemoryBarrierWithGroupSync();',
                '    MSOut o = (MSOut)0;',
                '    uint x2w_prim = 0x80000000u;']
        s.used_v.add(5)
        flat = re.compile(r'.*o\.fparam(\d+) = ')
        body = []
        for line in self.dw_sub(s.body()):
            # attributes the pixel stage reads only interpolated: drop the flat copy (as hlsl_vs), and the
            # binary16 half copies unless it reads them (v_interp_p2_f16)
            line = re.sub(r'o\.(f|lo|hi)param(\d+) = [^;]*;', lambda mm: mm.group(0) if int(mm.group(2)) in (
                self.flat if mm.group(1) == 'f' else self.half) else '', line)
            body.append(line)
        if dump:
            # progress markers (tests only): header word 1 = 1 at entry, word 2 = phases completed
            marked, phase = [], 0
            for line in body:
                marked.append(line)
                if line.strip() == 'GroupMemoryBarrierWithGroupSync();' and line.startswith('    G'):
                    phase += 1
                    marked.append('    if (tid == 0) x2w_dump.Store(gid * %du + 8u, %du);' % (16 + GROUP * stride, phase))
            body = ['    if (tid == 0) x2w_dump.Store(gid * %du + 4u, 1u);' % (16 + GROUP * stride)] + marked
        out += s.declarations() + ['    v5 = gid * %du + tid;  // inferred fast-launch vertex id' % GROUP] + body
        out += ['    GroupMemoryBarrierWithGroupSync();',
                '    uint nv = min(x2w_alloc & 0x7ffu, %du), np = min((x2w_alloc >> 12) & 0x7ffu, %du);' % (MAX_VERTICES, MAX_PRIMITIVES),
                '    SetMeshOutputCounts(nv, np);',
                '    if (tid < nv) verts[tid] = o;',
                '    if (tid < np) {',
                '        uint3 t = uint3(x2w_prim & 0x1ffu, (x2w_prim >> 10) & 0x1ffu, (x2w_prim >> 20) & 0x1ffu);',
                '        bool cull = (x2w_prim >> 31) != 0u || t.x >= nv || t.y >= nv || t.z >= nv;',
                '        tris[tid] = cull ? uint3(0, 0, 0) : t;',
                '        prims[tid].cull = cull;'] + (
                ['        prims[tid].pid = gid * %du + tid;' % GROUP] if dump else []) + [
                '    }']
        if dump:
            out += ['    uint db = gid * %du, tb = db + 16u + tid * %du;' % (16 + GROUP * stride, stride),
                    '    if (tid == 0) x2w_dump.Store(db, x2w_alloc);',
                    '    x2w_dump.Store4(tb, asuint(o.pos));']
            out += ['    x2w_dump.Store4(tb + %du, asuint(o.param%d));' % (16 + 16 * p, p) for p in range(nparams)]
            out += ['    x2w_dump.Store(tb + %du, x2w_prim);' % (16 + 16 * nparams)]
            if nclip:
                out += ['    x2w_dump.Store%s(tb + %du, asuint(o.clip0));' % (nclip if nclip > 1 else '', 20 + 16 * nparams)]
        out += ['}']
        return '\n'.join(out) + '\n'

    def manifest(self):
        text = super().manifest() if self.ps else self.manifest_ms_only()
        if self.heapwords:
            if not re.search(r'^heap 1$', text, re.M):
                text += 'heap 1\n'
            text += 'heapwords 1\n'
        text += 'fma %s\n' % ('exact' if EXACT_FMA else 'precise')
        return text + 'stage mesh\nms %d %d %d\n' % (GROUP, MAX_VERTICES, MAX_PRIMITIVES)

    def manifest_ms_only(self):
        class _P:  # base manifest() reads self.ps.exports for the targets line
            exports = {}
        saved, self.ps = self.ps, _P()
        try:
            return G.Pipeline.manifest(self)
        finally:
            self.ps = saved


DXC = os.environ.get('X2W_DXC', r'C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe')


def compile_hlsl(path, profile, out):
    r = subprocess.run([DXC, '-nologo', '-T', profile, '-E', 'main', '-Fo', out, path], capture_output=True, text=True)
    return r.returncode, r.stderr


def translate_to(out_dir, key, ms_code, ps_code, root_blob, check=True, rtv_formats=(), rtv_request='',
                 user_sgpr=MESH_USER_SGPR, ps_hw_inputs=None, vs_out_cntl=None):
    """Writes <key>.ms.hlsl, <key>.ps.hlsl, the .cso files and <key>.txt (last), or <key>.fail.
    ps_hw_inputs: the pixel stage's (SPI_PS_INPUT_ENA, SPI_PS_INPUT_ADDR[, launch FLOAT_MODE]), or None (entry-VGPR
    reads refused). The FLOAT_MODE (SPI_SHADER_PGM_RSRC1_PS bits 19:12) is the optional third .psin field.
    vs_out_cntl: the pipeline's PA_CL_VS_OUT_CNTL (what a pos1 export carries), or None (pos1 refused)."""
    name = os.path.join(out_dir, '%016x' % key)
    try:
        pipe = MeshPipeline(ms_code, ps_code, root_blob, rtv_formats=rtv_formats, user_sgpr=user_sgpr,
                            ps_hw_inputs=ps_hw_inputs if ps_code is not None else None, vs_out_cntl=vs_out_cntl)
        if pipe.ps_entry_reads:
            reg, off = min(pipe.ps_entry_reads.items(), key=lambda item: item[1])
            raise Unsupported('pixel stage reads entry VGPR v%d (an SPI input; layout not in the request) at %#x' % (reg, off))
        ms = pipe.hlsl_ms()
        ps = pipe.hlsl_ps() if ps_code is not None else None
        manifest = pipe.manifest() + (rtv_request + '\n' if rtv_request else '')
    except Unsupported as reason:
        open(name + '.fail', 'w').write(str(reason) + '\n')
        return 'unsupported: %s' % reason
    open(name + '.ms.hlsl', 'w').write(ms)
    if ps is not None:
        open(name + '.ps.hlsl', 'w').write(ps)
    if check:
        # The pixel stage is compiled with dxc too: a pipeline with a DXIL mesh stage takes DXIL stages only.
        for stage, profile, present in (('ms', 'ms_6_5', True), ('ps', 'ps_6_0', ps is not None)):
            if not present:
                continue
            code, err = compile_hlsl(name + '.%s.hlsl' % stage, profile, name + '.%s.cso' % stage)
            if code:
                open(name + '.fail', 'w').write('dxc %s: %s\n' % (stage, err[-2000:]))
                return 'dxc %s failed: %s' % (stage, err.strip().splitlines()[0] if err.strip() else '?')
    temporary = '%s.txt.%d.tmp' % (name, os.getpid())
    with open(temporary, 'w') as result:
        result.write(manifest)
    os.replace(temporary, name + '.txt')
    return 'ok'


def is_mesh(code):
    """NGG mesh stage: it sends GS_ALLOC_REQ and exports a primitive."""
    text = [x[3] for x in G.decode(G.strip_header(code))]
    return any(t == 's_sendmsg 0x9' for t in text) and any(re.match(r'exp (prim|target_20)', t) for t in text)


def translate_request(out_dir, key):
    """A mesh pipeline request req_<key>.{a,b,root}: whichever blob is the mesh stage, the other the pixel stage.

    An RTV variant (the draw path's contract, gfx2hlsl.translate_request; the mesh runtime publishes it when a bound
    target is integer or 32-bit float, d3d12_mesh.inl): req_<key>.rtv holds `rtv1 <base key> <DXGI formats>` with
    key = fnv(that text); the blobs are the base request's, checked against the base key; the pixel exports take
    the target formats (Pipeline.mrt_kind) and the manifest carries the rtv1 line. A variant of a depth-only
    pipeline is refused (the runtime takes the base translation for any target there)."""
    base = os.path.join(out_dir, 'req_%016x' % key)
    request, formats, base_key = '', (), key
    if os.path.isfile(base + '.rtv'):
        try:
            with open(base + '.rtv', encoding='ascii') as file:
                request = file.read(256)
        except (OSError, UnicodeError) as error:
            return 'unreadable: %s' % error
        if not re.fullmatch(r'rtv1 [0-9]{1,20}(?: [0-9]{1,10}){1,8}', request) or G.fnv(request.encode('ascii')) != key:
            return 'unsupported: invalid RTV variant request'
        base_key, *formats = map(int, request.split()[1:])
        if base_key > 0xffffffffffffffff or any(fmt > 0xffffffff for fmt in formats):
            return 'unsupported: invalid RTV variant values'
        formats = tuple(formats)
        base = os.path.join(out_dir, 'req_%016x' % base_key)
    try:
        a, b, root = (open(base + ext, 'rb').read() for ext in ('.a', '.b', '.root'))
    except OSError as error:
        return 'unreadable: %s' % error
    if request:
        if G.pipeline_key(a, b, root) != base_key:
            return 'unsupported: RTV variant source key mismatch'
        if not (a and b):
            return 'unsupported: RTV variant of a mesh pipeline without a pixel stage'
    rtv = {'rtv_formats': formats, 'rtv_request': request} if request else {}
    # optional req_<key>.vsout `pa_cl_vs_out_cntl 1 <hex8>` (not part of the key): what a pos1 export carries
    try:
        cntl = G.read_vs_out(base + '.vsout')
    except Unsupported as reason:
        return 'unsupported: %s' % reason
    if not a and b and is_mesh(b):
        return translate_to(out_dir, key, b, None, root, vs_out_cntl=cntl)   # depth-only mesh pipeline (no pixel stage)
    try:
        # the pixel stage's SPI inputs: req_<key>.psin (gfx2hlsl.read_ps_inputs format), else none
        ps_hw_inputs = G.read_ps_inputs(base + '.psin')
    except Unsupported as reason:
        return 'unsupported: %s' % reason
    if a and b and is_mesh(b) and not is_mesh(a):
        return translate_to(out_dir, key, b, a, root, ps_hw_inputs=ps_hw_inputs, vs_out_cntl=cntl, **rtv)
    if a and b and is_mesh(a) and not is_mesh(b):
        return translate_to(out_dir, key, a, b, root, ps_hw_inputs=ps_hw_inputs, vs_out_cntl=cntl, **rtv)
    return 'not a mesh pipeline'


# ---- root signature synthesis for coverage runs over the archive (no real root blob there) ----------------------
def synth_root(*codes):
    """A root signature blob whose SGPR layout from s16 (MESH_USER_SGPR) matches how the programs (the stages of one pipeline, which
    share the layout) use their entry SGPRs: a register read as a 32-bit table pointer (with a zero high word)
    becomes a descriptor table, a pair used as a 64-bit base a root SRV, everything else 32-bit constants.
    COVERAGE ONLY: the title's real root signature comes with the request."""
    tables, pairs, used = set(), set(), set()
    for code in codes:
        if code:
            synth_scan(code, tables, pairs, used)
    return synth_layout(tables, pairs, used, min(entry_sgpr(c) for c in codes if c))


def entry_sgpr(code):
    """First user SGPR for a program without a real root: 8 if it reads any of s8..s15 before writing it (the
    archive programs), else MESH_USER_SGPR (16, the run programs: s8..s15 are written first). Straight-line scan
    up to the first branch; COVERAGE ONLY."""
    written = set()
    for off, n, name, asm in G.decode(G.strip_header(code)):
        _, ops, mods = parse(asm)
        base = re.sub(r'_e(32|64)$', '', name)
        for t in ops[1:]:
            if any(8 <= r < MESH_USER_SGPR and r not in written for r in scalar_regs(t)):
                return 8
        if base.startswith(('s_branch', 's_cbranch', 's_endpgm')):
            break
        dst = reg_range(ops[0]) if ops else None
        if dst and dst[0] == 's' and not base.startswith(NO_DEST):
            written.update(range(dst[1], dst[2] + 1))
    return MESH_USER_SGPR


def synth_scan(code, tables, pairs, used):
    ins = G.decode(G.strip_header(code))
    origin = {}          # current register -> entry register it holds
    zero = set()
    for r in range(8, 64):
        origin[r] = r
    for off, n, name, asm in ins:
        _, ops, mods = parse(asm)
        base = re.sub(r'_e(32|64)$', '', name)
        for t in ops[1:]:
            for r in scalar_regs(t):
                if r in origin:
                    used.add(origin[r])
        if base.startswith(('s_load', 's_buffer_load', 'buffer_', 'tbuffer_', 'image_')):
            sb = ops[1] if base.startswith(('s_load', 's_buffer_load')) else (ops[2] if len(ops) > 2 else '')
            rr = reg_range(sb)
            if rr and rr[0] == 's':
                lo = origin.get(rr[1])
                if lo is not None and (rr[1] + 1) in zero and base.startswith('s_load'):
                    tables.add(lo)
                elif lo is not None and origin.get(rr[1] + 1) == lo + 1:
                    pairs.add(lo)
        dst = reg_range(ops[0]) if ops else None
        if dst and dst[0] == 's' and base.startswith('s_') and not base.startswith(NO_DEST):
            src = reg_range(ops[1]) if len(ops) > 1 else None
            if base == 's_bitset1_b32':
                continue  # sets one bit in place (descriptor word 1 from a pointer's high word): same origin
            for r in range(dst[1], dst[2] + 1):
                origin.pop(r, None)
                zero.discard(r)
            if base == 's_mov_b32' and src and src[0] == 's' and src[1] in origin:
                origin[dst[1]] = origin[src[1]]
            elif base in ('s_or_b32', 's_bitset1_b32') and src and src[0] == 's' and src[1] in origin:
                origin[dst[1]] = origin[src[1]]   # descriptor word 1 built from a pointer's high word
            elif base == 's_mov_b32' and ops[1] == '0':
                zero.add(dst[1])


def synth_layout(tables, pairs, used, start=MESH_USER_SGPR):
    params, r = [], start
    used = {u for u in used if u >= start}
    if start == MESH_USER_SGPR and used & {23, 24}:
        # s23/s24 are the resource and sampler heaps after the dominant real layout (sampled real roots: 3 tables,
        # a root SRV, 2 constants, heap flags 0xc00 in ~87%); such a program cannot be expressed by the scan below
        return root_blob([(0, None), (0, None), (0, None), (3, None), (1, 2)], 0xc00)
    top = max(used | {start - 1}) + 1
    while r < top:
        if r in tables:
            params.append((0, None))
            r += 1
        elif r in pairs:
            params.append((3, None))
            r += 2
        else:
            n = 1
            while r + n < top and r + n not in tables and r + n not in pairs:
                n += 1
            params.append((1, n))
            r += n
    return root_blob(params)


def root_blob(params, flags=0):
    """DXBC container with an RTS0 part (version 2) for params [(type, count)]: type 0 table (no ranges), 1 constants,
    3 root SRV."""
    head = 24
    body = bytearray()
    extra = bytearray()
    poff = head
    eoff = head + 12 * len(params)
    for i, (t, count) in enumerate(params):
        if t == 1:
            body += struct.pack('<3I', t, 0, eoff + len(extra))
            extra += struct.pack('<3I', i, 0, count)
        elif t == 0:
            body += struct.pack('<3I', t, 0, eoff + len(extra))
            extra += struct.pack('<2I', 0, eoff + len(extra) + 8)
        else:
            body += struct.pack('<3I', t, 0, eoff + len(extra))
            extra += struct.pack('<3I', i, 0, 0)
    rts = struct.pack('<6I', 2, len(params), poff, 0, eoff + len(extra), flags) + bytes(body) + bytes(extra)
    part = b'RTS0' + struct.pack('<I', len(rts)) + rts
    total = 32 + 4 + len(part)
    return b'DXBC' + b'\0' * 16 + struct.pack('<III', 1, total, 1) + struct.pack('<I', 36) + part


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--requests':
        out_dir = sys.argv[2]
        for name in sorted(os.listdir(out_dir)):
            m = re.fullmatch(r'req_([0-9a-f]{16})\.a', name)
            if not m or os.path.exists(os.path.join(out_dir, '%s.txt' % m.group(1))):
                continue
            result = translate_request(out_dir, int(m.group(1), 16))
            if result != 'not a mesh pipeline':
                print('%s: %s' % (m.group(1), result))
        return
    if len(sys.argv) >= 4 and sys.argv[1] == '--ms-only':
        # --ms-only <out_dir> <ms.bin> [...]: the mesh stage alone with a synthesized root (coverage)
        out_dir = sys.argv[2]
        os.makedirs(out_dir, exist_ok=True)
        for i, path in enumerate(sys.argv[3:]):
            code = open(path, 'rb').read()
            print('%s: %s' % (os.path.basename(path), translate_to(out_dir, i, code, None, synth_root(code),
                                                                        user_sgpr=entry_sgpr(code))))
        return
    print(__doc__)


if __name__ == '__main__':
    main()
