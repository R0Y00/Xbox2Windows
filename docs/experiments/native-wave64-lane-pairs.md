# Guest wave64 groups whose waves branch apart in direct compute

Refs [#3237](https://github.com/R0Y00/Xbox2Windows/issues/3237).

Direct compute modules declare wave size 32. A guest wave64 normally runs as two host wave32s (split halves) that
exchange mask words through group memory, with group barriers around each exchange and before each LDS access.
D3D12 defines a group barrier only when every thread of the group reaches it. Guest waves that branch apart reach
a different number of these barriers, so a group of more than one guest wave was refused when it had a conditional
branch together with a mask-writing compare, a carry-out, a lane read, an LDS access or an `s_barrier`
("multi-wave64 exchange requires converged group control flow").

Such a group is now translated as lane pairs. Groups of one guest wave and converged groups keep split halves and
compile to the same module as before.

## Layout

- One host wave32 runs one guest wave64. Host lane j carries guest lanes j and j + 32, and the module declares
  `numthreads(guest threads / 2, 1, 1)`.
- Every instruction outside the scalar encodings is emitted once per lane set, with a second set of VGPR storage.
  Split halves already pick a half's EXEC and mask word through one value; each emission uses it as a constant.
- A scalar register written by a vector instruction is stored after both emissions, so the second one still reads
  the instruction's operands. Each emission writes its own mask word. A handler that writes the same scalar
  register in both emissions is refused ("lane-pair layout: one scalar register written for both lane sets").
- A compare ballots its own word, without exchange memory or a barrier. `v_readlane_b32` reads both sets and
  selects by lane. `v_readfirstlane_b32` takes the first active lane of the low set, else of the high set, and
  lane 0 for an empty EXEC.
- Guest thread ids are computed per lane set from the host wave index and lane, so the guest group shape only
  enters that arithmetic. TG_SIZE holds the guest wave index and count.

## LDS inside one host wave

A guest lane may read an LDS word that another lane of its wave wrote one instruction earlier. Split halves order
the two with the group barrier before each LDS access. A lane-pair module cannot place a group barrier in guest
control flow, and D3D12 does not say that a host runs the lanes of a wave in step. Each LDS access of a lane-pair
module therefore takes its lane predicate from a `WaveActiveBallot` of its own, because a wave operation needs
every lane of the wave at that point. Without that ballot, app-local WARP 1.0.22 read words that other lanes had
not written yet (see Result).

The wave32 path, where one host wave is one guest wave, has no such ballot and is not changed here. A wave32
program with the same cross-lane read differs on WARP 1.0.22 as well.

## `s_barrier`

A barrier that every guest wave passes stays one group barrier. If some path reaches `s_endpgm` without passing
an `s_barrier`, a wave could end or skip the barrier while the others wait at it. Such a program gets a single
barrier site in a loop:

- A wave runs to its next `s_barrier` or to `s_endpgm`, stores where to resume and goes to the site.
- There lane 0 publishes whether the wave has ended, the group meets, every thread reads all waves' words, and a
  second barrier closes those reads.
- A wave that has ended keeps returning until all of them have, so every thread passes the same barrier the same
  number of times.

An `s_barrier` inside a guest loop is refused ("lane-pair layout: s_barrier inside a guest loop").

## Fixture

`tests/native_wave64_lane_pairs_check.py` holds programs assembled with LLVM 15.0.7 (`clang -cc1as`, gfx1013,
`+wavefrontsize64`) and a CPU model for each. A program runs with 128 threads per group (two guest waves, lane
pairs) and with 64 (one guest wave, split halves), so each model is also compared with the layout that predates
lane pairs. All 2,064 output words are compared, guards included.

| Program | 128-thread group | What it checks |
|---|---|---|
| `mask_readlane` | 128×1×1 | only the last guest wave runs a compare into an SGPR pair, then `v_readlane_b32` of lane 40 |
| `readfirstlane_high` | 128×1×1 | `v_cmpx` leaves lanes 32–63 of the last wave, EXECZ skip, `v_readfirstlane_b32`; a compare that is true for every lane then gives low word 0 |
| `lds_divergent` | 128×1×1 | in-branch `ds_write_b32` and `ds_read_b32` across lanes j and 63 − j, then an `s_barrier` that every wave passes and a cross-wave read |
| `dpp_permlane_divergent` | 128×1×1 | `row_shr:1`, `row_mirror` and `v_permlanex16_b32` inside the branch |
| `group_shape_2d`, `group_shape_3d` | 8×16×1, 4×4×8 | `v0`, `v1`, `v2`, stored at an address computed from those ids |
| `readfirstlane_empty` | 128×1×1 | `v_readfirstlane_b32` with EXEC set to 0 returns lane 0 |
| `lds_same_address` | 128×1×1 | lanes j and j + 32 write one LDS word in one instruction |
| `atomic_count` | 128×1×1 | lanes 16–63 of the last wave each add 1 to one word with `buffer_atomic_add` |
| `exit_before_barrier` | 256×1×1 | wave 0 leaves through an EXECZ skip; waves 1–3 write LDS, meet and read wave 3's words |

`native_wave64_lane_pairs_controls` needs no GPU. It checks that the 128-thread modules declare 64 host threads
and no exchange memory, that `lds_same_address` has a ballot per lane set before each LDS access, that
`mask_readlane` with 64 threads keeps the exchange memory, that only `exit_before_barrier` gets the barrier loop,
and that a barrier inside a loop is refused.
`native_wave64_lane_pairs_hardware` returns 77 when replay reports that the adapter lacks the required
capabilities.

```powershell
cmake --build --preset dev --target compile_dxil native_store_replay
ctest --preset dev --no-tests=error -j 1 -R '^native_wave64_lane_pairs_' -V
```

Negative controls, each of which must report a mismatch: the model with the two 32-lane sets exchanged (three
programs), `v_readlane_b32` of lane 8 against the lane-40 expectation, a `v_cmpx` threshold of 31 against the
threshold-32 expectation, inactive mask bits left set, the atomic count without the high lane set, and for
`exit_before_barrier` the model in which wave 0 had not left and the model in which a thread reads its own LDS
word.

Two results are not asserted for the 64-thread split-halves runs:

- `lds_same_address`: the two halves race on the shared word.
- `readfirstlane_empty`: split halves return lane 32, not lane 0, for an empty EXEC. This is a separate defect of
  that lowering and is not changed here.

## Result

`main` `052abe6f` plus this change, Debug build. Windows 11 build 26300, NVIDIA GeForce RTX 4090 (driver
32.0.16.1088, 32-lane waves), DXC from Windows SDK 10.0.26100. The debug layer was not enabled.

- `native_wave64_lane_pairs_controls` passed.
- `native_wave64_lane_pairs_hardware` passed. The nine 128-thread programs and `exit_before_barrier` matched their
  models as lane pairs. Seven of the nine matched as split halves with 64 threads; the other two are the ones not
  asserted above. All 16 negative controls reported a mismatch, for example `mask_readlane` with the sets
  exchanged (word 264, got `0xffffffff`, expected `0x00001fff`) and `atomic_count` without the high lane set
  (word 8, got `0x69696999`, expected `0x69696979`).
- Against an unmodified compiler built from `052abe6f`, over 44 requests (the fixture programs with 64 and 128
  threads, the slot 67 wave-size program and three straight-line programs with 64, 128 and 256 threads):
  - all 31 modules that compiler produces are byte-identical in this build;
  - it refuses the other 13 with "multi-wave64 exchange requires converged group control flow". This build
    compiles 12 of them as lane pairs and refuses the barrier inside a loop.

  This comparison used a local script that is not in the repository.
- App-local WARP 1.0.22 (`d3d10warp.dll` 1.0.22.0.20261027.1 beside the replay executable, in-box D3D12 runtime).
  A local script replayed the fixture's requests with `--direct-warp` in place of `--direct`, 12 times each:
  - all 17 asserted readbacks matched in every run;
  - a build without the ballot before LDS accesses differed in every run of `lds_divergent` as lane pairs, in
    the read before the `s_barrier` (word 264, got `0x00000000`, expected `0x00007f05`);
  - a wave32 program in which each lane reads the word that lane 31 − l wrote one instruction earlier differed
    in every run (word 8, got `0x00000000`, expected `0x00001f05`). With a compare placed between the write and
    the read it matched in every run. Its module is byte-identical to the unmodified compiler's.
- An independent run of the first commit of this change on an RTX 4070 and on WARP 1.0.22 is recorded in
  [the pull request](https://github.com/R0Y00/Xbox2Windows/pull/3352#issuecomment-7239334348). There
  `dpp_permlane_divergent` differed on WARP under both layouts (88 words). It matched here.

## Limits

- Two NVIDIA adapters, the second in the independent run above, which predates the ballot. The fixture has no
  `--direct-warp` mode, so the WARP runs went through local scripts, and the in-box WARP has 4-lane waves (see
  [the WARP note](native-compute-warp.md)). The debug layer was not enabled, and no dispatch went through the
  stand-in.
- The ballot before an LDS access relies on a wave operation gathering every lane of the host wave. D3D12 does
  not state that. It held on WARP 1.0.22 in these runs; both GPUs matched without the ballot as well.
- On hardware, lane pairs ran integer VOP1, VOP2 and VOP3 instructions, compares, `v_cmpx`, DPP row operations,
  `v_permlanex16_b32`, both lane reads, LDS reads and writes, raw buffer stores and one buffer atomic. Buffer
  loads, typed stores, image operations (implicit derivatives included), 16-bit and packed operations go through
  the same per-lane handlers but were not run in a lane-pair module.
- `lds_same_address` fixes the host order: the high lane set is written last, so lane j + 32 wins. Which lane the
  guest hardware keeps is not established.
- The barrier loop was run with one `s_barrier`. `exit_before_barrier` shows that the loop works on this driver;
  it does not show that an early `ret` would have failed.
- A lane-pair module emits each vector instruction twice. Module size and timing against split halves were
  measured on an earlier base with a local layout switch that is not part of this change
  ([#3237 comment](https://github.com/R0Y00/Xbox2Windows/issues/3237#issuecomment-7232595595)). Those numbers
  predate the ballot before each LDS access, so their LDS rows no longer describe this code.
