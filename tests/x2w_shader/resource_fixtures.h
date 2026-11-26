// Fixtures of the upstream Rust shader tests in src/resource.rs that other test suites
// reuse (`pub(crate)` there): waterfall-loop programs, wave32 and wave64, as GFX10 words.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace x2w::vs::resource::fixtures {

namespace fixtures_detail {
// The index of `word` in `words` (Rust: tests::word_at; panics there when absent).
inline std::size_t word_at(const std::vector<std::uint32_t>& words, std::uint32_t word)
{
    const auto it = std::find(words.begin(), words.end(), word);
    if (it == words.end()) throw std::logic_error("word_at: word not in the program");
    return static_cast<std::size_t>(it - words.begin());
}
}  // namespace fixtures_detail

// A waterfall loop over an image descriptor each lane loads from the
// table at s0 (wave32, assembled with llvm-mc for gfx1030): lane i
// reads entry i with two `global_load_dwordx4`, the loop reads the
// first active lane's descriptor into s[12:19], samples with the lanes
// whose descriptor it is, and repeats for the others. The sampler comes
// from the table at s1; the result goes to LDS so the sample is needed.
inline const std::vector<std::uint32_t> WATERFALL_FETCHED_WAVE32 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xBE86'0301, // s_mov_b32 s6, s1
    0xBE87'0380, // s_mov_b32 s7, 0
    0x3402'0085, // v_lshlrev_b32 v1, 5, v0
    0xDC38'8000,
    0x0404'0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
    0xDC38'8010,
    0x0804'0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
    0xF408'0203,
    0xFA00'0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0x7E04'0D00, // v_cvt_f32_u32 v2, v0
    0x7E06'0302, // v_mov_b32 v3, v2
    0xBF8C'0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
    0xBE94'037E, // s_mov_b32 s20, exec_lo
    0x7E18'0504, // 0x3c: v_readfirstlane_b32 s12, v4
    0x7E1A'0505, // v_readfirstlane_b32 s13, v5
    0x7E1C'0506, // v_readfirstlane_b32 s14, v6
    0x7E1E'0507, // v_readfirstlane_b32 s15, v7
    0x7E20'0508, // v_readfirstlane_b32 s16, v8
    0x7E22'0509, // v_readfirstlane_b32 s17, v9
    0x7E24'050A, // v_readfirstlane_b32 s18, v10
    0x7E26'050B, // v_readfirstlane_b32 s19, v11
    0xD4E2'0015,
    0x0002'080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2'0016,
    0x0002'0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xBE95'3C15, // s_and_saveexec_b32 s21, s21
    0xBF80'0000, // s_nop 0
    0xF09C'0F08,
    0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x897E'157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89'FFE6, // s_cbranch_execnz -> 0x3c
    0xBEFE'0314, // s_mov_b32 exec_lo, s20
    0xD834'0000,
    0x0000'0C00, // ds_write_b32 v0, v12
    0xBF81'0000, // s_endpgm
};

// WATERFALL_FETCHED_WAVE32 in wave64: 64-bit lane masks.
inline const std::vector<std::uint32_t> WATERFALL_FETCHED_WAVE64 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xBE86'0301, // s_mov_b32 s6, s1
    0xBE87'0380, // s_mov_b32 s7, 0
    0x3402'0085, // v_lshlrev_b32 v1, 5, v0
    0xDC38'8000,
    0x0404'0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
    0xDC38'8010,
    0x0804'0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
    0xF408'0203,
    0xFA00'0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0x7E04'0D00, // v_cvt_f32_u32 v2, v0
    0x7E06'0302, // v_mov_b32 v3, v2
    0xBF8C'0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
    0xBE94'047E, // s_mov_b64 s[20:21], exec
    0x7E18'0504, // 0x3c: v_readfirstlane_b32 s12, v4
    0x7E1A'0505, // v_readfirstlane_b32 s13, v5
    0x7E1C'0506, // v_readfirstlane_b32 s14, v6
    0x7E1E'0507, // v_readfirstlane_b32 s15, v7
    0x7E20'0508, // v_readfirstlane_b32 s16, v8
    0x7E22'0509, // v_readfirstlane_b32 s17, v9
    0x7E24'050A, // v_readfirstlane_b32 s18, v10
    0x7E26'050B, // v_readfirstlane_b32 s19, v11
    0xD4E2'0016,
    0x0002'080C, // v_cmp_eq_u64_e64 s[22:23], s[12:13], v[4:5]
    0xD4E2'0018,
    0x0002'0C0E, // v_cmp_eq_u64_e64 s[24:25], s[14:15], v[6:7]
    0x8796'1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
    0xD4E2'0018,
    0x0002'1010, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[8:9]
    0x8796'1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
    0xD4E2'0018,
    0x0002'1412, // v_cmp_eq_u64_e64 s[24:25], s[18:19], v[10:11]
    0x8796'1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
    0xBE96'2416, // s_and_saveexec_b64 s[22:23], s[22:23]
    0xBF80'0000, // s_nop 0
    0xF09C'0F08,
    0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x89FE'167E, // s_xor_b64 exec, exec, s[22:23]
    0xBF89'FFE6, // s_cbranch_execnz -> 0x3c
    0xBEFE'0414, // s_mov_b64 exec, s[20:21]
    0xD834'0000,
    0x0000'0C00, // ds_write_b32 v0, v12
    0xBF81'0000, // s_endpgm
};

// The same waterfall over one of two image descriptors loaded from the
// table at s0, picked per lane by `v_cndmask_b32` (lanes below 16 take
// the first).
inline const std::vector<std::uint32_t> WATERFALL_PICKED_WAVE32 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xBE86'0301, // s_mov_b32 s6, s1
    0xBE87'0380, // s_mov_b32 s7, 0
    0xF40C'0602,
    0xFA00'0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
    0xF40C'0802,
    0xFA00'0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
    0xF408'0203,
    0xFA00'0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0x7E04'0D00, // v_cvt_f32_u32 v2, v0
    0x7E06'0302, // v_mov_b32 v3, v2
    0x7D88'0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
    0x7E08'0218, // v_mov_b32 v4, s24
    0x0208'0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
    0x7E0A'0219, // v_mov_b32 v5, s25
    0x020A'0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
    0x7E0C'021A, // v_mov_b32 v6, s26
    0x020C'0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
    0x7E0E'021B, // v_mov_b32 v7, s27
    0x020E'0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
    0x7E10'021C, // v_mov_b32 v8, s28
    0x0210'1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
    0x7E12'021D, // v_mov_b32 v9, s29
    0x0212'1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
    0x7E14'021E, // v_mov_b32 v10, s30
    0x0214'1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
    0x7E16'021F, // v_mov_b32 v11, s31
    0x0216'1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
    0xBE94'037E, // s_mov_b32 s20, exec_lo
    0x7E18'0504, // 0x7c: v_readfirstlane_b32 s12, v4
    0x7E1A'0505, // v_readfirstlane_b32 s13, v5
    0x7E1C'0506, // v_readfirstlane_b32 s14, v6
    0x7E1E'0507, // v_readfirstlane_b32 s15, v7
    0x7E20'0508, // v_readfirstlane_b32 s16, v8
    0x7E22'0509, // v_readfirstlane_b32 s17, v9
    0x7E24'050A, // v_readfirstlane_b32 s18, v10
    0x7E26'050B, // v_readfirstlane_b32 s19, v11
    0xD4E2'0015,
    0x0002'080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2'0016,
    0x0002'0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xBE95'3C15, // s_and_saveexec_b32 s21, s21
    0xBF80'0000, // s_nop 0
    0xF09C'0F08,
    0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x897E'157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89'FFE6, // s_cbranch_execnz -> 0x7c
    0xBEFE'0314, // s_mov_b32 exec_lo, s20
    0xD834'0000,
    0x0000'0C00, // ds_write_b32 v0, v12
    0xBF81'0000, // s_endpgm
};

// The waterfall loop of WATERFALL_PICKED_WAVE32 over one of two
// image descriptors loaded from the table at s0, chosen by control flow
// as the compiler lowers an if/else on a per-lane condition: lanes below
// 16 move the first into v[4:11], the others the second, each arm
// skipped when no lane takes it. The sample goes to the buffer at dword
// 16 of the table, at lane * 4.
inline const std::vector<std::uint32_t> WATERFALL_JOINED_WAVE32 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xBE86'0301, // s_mov_b32 s6, s1
    0xBE87'0380, // s_mov_b32 s7, 0
    0xF40C'0602,
    0xFA00'0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
    0xF40C'0802,
    0xFA00'0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
    0xF408'0203,
    0xFA00'0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408'0A02,
    0xFA00'0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x7E04'0D00, // v_cvt_f32_u32 v2, v0
    0x7E06'0302, // v_mov_b32 v3, v2
    0x3420'0082, // v_lshlrev_b32 v16, 2, v0
    0x7D88'0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
    0xBE94'3C6A, // s_and_saveexec_b32 s20, vcc_lo
    0x8914'147E, // s_xor_b32 s20, exec_lo, s20
    0xBF88'0008, // s_cbranch_execz -> 0x70
    0x7E08'0218, // v_mov_b32 v4, s24
    0x7E0A'0219, // v_mov_b32 v5, s25
    0x7E0C'021A, // v_mov_b32 v6, s26
    0x7E0E'021B, // v_mov_b32 v7, s27
    0x7E10'021C, // v_mov_b32 v8, s28
    0x7E12'021D, // v_mov_b32 v9, s29
    0x7E14'021E, // v_mov_b32 v10, s30
    0x7E16'021F, // v_mov_b32 v11, s31
    0xBE94'3D14, // 0x70: s_or_saveexec_b32 s20, s20
    0x897E'147E, // s_xor_b32 exec_lo, exec_lo, s20
    0xBF88'0008, // s_cbranch_execz -> 0x9c
    0x7E08'0220, // v_mov_b32 v4, s32
    0x7E0A'0221, // v_mov_b32 v5, s33
    0x7E0C'0222, // v_mov_b32 v6, s34
    0x7E0E'0223, // v_mov_b32 v7, s35
    0x7E10'0224, // v_mov_b32 v8, s36
    0x7E12'0225, // v_mov_b32 v9, s37
    0x7E14'0226, // v_mov_b32 v10, s38
    0x7E16'0227, // v_mov_b32 v11, s39
    0x887E'147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
    0xBE97'037E, // s_mov_b32 s23, exec_lo
    0x7E18'0504, // 0xa4: v_readfirstlane_b32 s12, v4
    0x7E1A'0505, // v_readfirstlane_b32 s13, v5
    0x7E1C'0506, // v_readfirstlane_b32 s14, v6
    0x7E1E'0507, // v_readfirstlane_b32 s15, v7
    0x7E20'0508, // v_readfirstlane_b32 s16, v8
    0x7E22'0509, // v_readfirstlane_b32 s17, v9
    0x7E24'050A, // v_readfirstlane_b32 s18, v10
    0x7E26'050B, // v_readfirstlane_b32 s19, v11
    0xD4E2'0015,
    0x0002'080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2'0016,
    0x0002'0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xBE95'3C15, // s_and_saveexec_b32 s21, s21
    0xBF80'0000, // s_nop 0
    0xF09C'0F08,
    0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x897E'157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89'FFE6, // s_cbranch_execnz -> 0xa4
    0xBEFE'0317, // s_mov_b32 exec_lo, s23
    0xE070'1000,
    0x800A'0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81'0000, // s_endpgm
};

// The console compiler's waterfall (the title's ray-tracing
// resolve and others): each lane holds its entry's byte offset in the
// table at s0 (v7 = (lane & 1) * 32); the loop reads the first active
// lane's offset, narrows EXEC to the lanes with the same one
// (`v_cmpx_eq_u32`), loads that entry with `s_load_dwordx8` and samples,
// then drops those lanes (`s_andn2_wrexec_b32`). The sample goes to the
// buffer at dword 16 of the table, at lane * 4.
inline const std::vector<std::uint32_t> WATERFALL_INDEXED_WAVE32 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xBE86'0301, // s_mov_b32 s6, s1
    0xBE87'0380, // s_mov_b32 s7, 0
    0xF408'0203,
    0xFA00'0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408'0A02,
    0xFA00'0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x7E04'0D00, // v_cvt_f32_u32 v2, v0
    0x7E06'0302, // v_mov_b32 v3, v2
    0x3420'0082, // v_lshlrev_b32 v16, 2, v0
    0x360E'0081, // v_and_b32 v7, 1, v0
    0x340E'0E85, // v_lshlrev_b32 v7, 5, v7
    0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
    0xBE97'037E, // s_mov_b32 s23, exec_lo
    0xBE95'037E, // s_mov_b32 s21, exec_lo
    0x7E28'0507, // 0x40: v_readfirstlane_b32 s20, v7
    0x7DA4'0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
    0xF40C'0302,
    0x2800'0000, // s_load_dwordx8 s[12:19], s[4:5], s20
    0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
    0xF09C'0F08,
    0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0xBE95'4715, // s_andn2_wrexec_b32 s21, s21
    0xBF89'FFF6, // s_cbranch_execnz -> 0x40
    0xBEFE'0317, // s_mov_b32 exec_lo, s23
    0xE070'1000,
    0x800A'0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81'0000, // s_endpgm
};

// WATERFALL_JOINED_WAVE32 with other sources on each path: lanes
// below 16 load table entry (lane & 1) with two `global_load_dwordx4`,
// the others pick per lane with `v_cndmask_b32` (lanes below 24 the
// first entry, the rest the second).
inline const std::vector<std::uint32_t> WATERFALL_JOINED_MIXED_WAVE32 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xBE86'0301, // s_mov_b32 s6, s1
    0xBE87'0380, // s_mov_b32 s7, 0
    0xF40C'0602,
    0xFA00'0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
    0xF40C'0802,
    0xFA00'0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
    0xF408'0203,
    0xFA00'0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408'0A02,
    0xFA00'0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x7E04'0D00, // v_cvt_f32_u32 v2, v0
    0x7E06'0302, // v_mov_b32 v3, v2
    0x3420'0082, // v_lshlrev_b32 v16, 2, v0
    0x3602'0081, // v_and_b32 v1, 1, v0
    0x3402'0285, // v_lshlrev_b32 v1, 5, v1
    0x7D88'0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
    0xBE94'3C6A, // s_and_saveexec_b32 s20, vcc_lo
    0x8914'147E, // s_xor_b32 s20, exec_lo, s20
    0xBF88'0004, // s_cbranch_execz -> 0x68
    0xDC38'8000,
    0x0404'0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
    0xDC38'8010,
    0x0804'0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
    0xBE94'3D14, // 0x68: s_or_saveexec_b32 s20, s20
    0x897E'147E, // s_xor_b32 exec_lo, exec_lo, s20
    0xBF88'0011, // s_cbranch_execz -> 0xb8
    0x7D88'0098, // v_cmp_gt_u32 vcc_lo, 24, v0
    0x7E08'0218, // v_mov_b32 v4, s24
    0x0208'0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
    0x7E0A'0219, // v_mov_b32 v5, s25
    0x020A'0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
    0x7E0C'021A, // v_mov_b32 v6, s26
    0x020C'0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
    0x7E0E'021B, // v_mov_b32 v7, s27
    0x020E'0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
    0x7E10'021C, // v_mov_b32 v8, s28
    0x0210'1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
    0x7E12'021D, // v_mov_b32 v9, s29
    0x0212'1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
    0x7E14'021E, // v_mov_b32 v10, s30
    0x0214'1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
    0x7E16'021F, // v_mov_b32 v11, s31
    0x0216'1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
    0x887E'147E, // 0xb8: s_or_b32 exec_lo, exec_lo, s20
    0xBE97'037E, // s_mov_b32 s23, exec_lo
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x7E18'0504, // 0xc4: v_readfirstlane_b32 s12, v4
    0x7E1A'0505, // v_readfirstlane_b32 s13, v5
    0x7E1C'0506, // v_readfirstlane_b32 s14, v6
    0x7E1E'0507, // v_readfirstlane_b32 s15, v7
    0x7E20'0508, // v_readfirstlane_b32 s16, v8
    0x7E22'0509, // v_readfirstlane_b32 s17, v9
    0x7E24'050A, // v_readfirstlane_b32 s18, v10
    0x7E26'050B, // v_readfirstlane_b32 s19, v11
    0xD4E2'0015,
    0x0002'080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2'0016,
    0x0002'0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xBE95'3C15, // s_and_saveexec_b32 s21, s21
    0xBF80'0000, // s_nop 0
    0xF09C'0F08,
    0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x897E'157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89'FFE6, // s_cbranch_execnz -> 0xc4
    0xBEFE'0317, // s_mov_b32 exec_lo, s23
    0xE070'1000,
    0x800A'0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81'0000, // s_endpgm
};

// A waterfall loop whose descriptor changes on each pass: v[4:11] holds
// the first table entry on entry and the second after every pass, so
// the first lane's dwords are phis of the loop header.
inline const std::vector<std::uint32_t> WATERFALL_LOOPED_WAVE32 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xBE86'0301, // s_mov_b32 s6, s1
    0xBE87'0380, // s_mov_b32 s7, 0
    0xF40C'0602,
    0xFA00'0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
    0xF40C'0802,
    0xFA00'0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
    0xF408'0203,
    0xFA00'0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
    0xF408'0A02,
    0xFA00'0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x7E04'0D00, // v_cvt_f32_u32 v2, v0
    0x7E06'0302, // v_mov_b32 v3, v2
    0x3420'0082, // v_lshlrev_b32 v16, 2, v0
    0x7E08'0218, // v_mov_b32 v4, s24
    0x7E0A'0219, // v_mov_b32 v5, s25
    0x7E0C'021A, // v_mov_b32 v6, s26
    0x7E0E'021B, // v_mov_b32 v7, s27
    0x7E10'021C, // v_mov_b32 v8, s28
    0x7E12'021D, // v_mov_b32 v9, s29
    0x7E14'021E, // v_mov_b32 v10, s30
    0x7E16'021F, // v_mov_b32 v11, s31
    0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
    0xBE97'037E, // s_mov_b32 s23, exec_lo
    0x7E18'0504, // 0x64: v_readfirstlane_b32 s12, v4
    0x7E1A'0505, // v_readfirstlane_b32 s13, v5
    0x7E1C'0506, // v_readfirstlane_b32 s14, v6
    0x7E1E'0507, // v_readfirstlane_b32 s15, v7
    0x7E20'0508, // v_readfirstlane_b32 s16, v8
    0x7E22'0509, // v_readfirstlane_b32 s17, v9
    0x7E24'050A, // v_readfirstlane_b32 s18, v10
    0x7E26'050B, // v_readfirstlane_b32 s19, v11
    0xD4E2'0015,
    0x0002'080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2'0016,
    0x0002'0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xD4E2'0016,
    0x0002'1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xBE95'3C15, // s_and_saveexec_b32 s21, s21
    0xBF80'0000, // s_nop 0
    0xF09C'0F08,
    0x0043'0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x897E'157E, // s_xor_b32 exec_lo, exec_lo, s21
    0x7E08'0220, // v_mov_b32 v4, s32
    0x7E0A'0221, // v_mov_b32 v5, s33
    0x7E0C'0222, // v_mov_b32 v6, s34
    0x7E0E'0223, // v_mov_b32 v7, s35
    0x7E10'0224, // v_mov_b32 v8, s36
    0x7E12'0225, // v_mov_b32 v9, s37
    0x7E14'0226, // v_mov_b32 v10, s38
    0x7E16'0227, // v_mov_b32 v11, s39
    0xBF89'FFDE, // s_cbranch_execnz -> 0x64
    0xBEFE'0317, // s_mov_b32 exec_lo, s23
    0xE070'1000,
    0x800A'0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81'0000, // s_endpgm
};

// A waterfall loop over one of two buffer descriptors (four dwords)
// loaded from the table at s0, chosen by nested if/else: lanes below 16
// move the first into v[4:7], lanes from 16 to 23 the second, and the
// others keep what v[4:7] held. Only lanes below 24 (the same mask)
// run the loop, which loads dword `lane` of the chosen buffer, and
// store it to the buffer at dword 16 of the table.
inline const std::vector<std::uint32_t> WATERFALL_JOINED_BUFFER_WAVE32 = {
    0xBE84'0300, // s_mov_b32 s4, s0
    0xBE85'0380, // s_mov_b32 s5, 0
    0xF408'0602,
    0xFA00'0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
    0xF408'0802,
    0xFA00'0020, // s_load_dwordx4 s[32:35], s[4:5], 0x20
    0xF408'0A02,
    0xFA00'0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0x3420'0082, // v_lshlrev_b32 v16, 2, v0
    0xD4C4'001E,
    0x0002'0098, // v_cmp_gt_u32_e64 s30, 24, v0
    0x7D88'0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBF8C'C07F, // s_waitcnt lgkmcnt(0)
    0xBE94'3C6A, // s_and_saveexec_b32 s20, vcc_lo
    0x8914'147E, // s_xor_b32 s20, exec_lo, s20
    0xBF88'0004, // s_cbranch_execz -> 0x50
    0x7E08'0218, // 0x40: v_mov_b32 v4, s24
    0x7E0A'0219, // v_mov_b32 v5, s25
    0x7E0C'021A, // v_mov_b32 v6, s26
    0x7E0E'021B, // v_mov_b32 v7, s27
    0xBE94'3D14, // 0x50: s_or_saveexec_b32 s20, s20
    0x897E'147E, // s_xor_b32 exec_lo, exec_lo, s20
    0xBF88'0007, // s_cbranch_execz -> 0x78
    0xBE9F'3C1E, // s_and_saveexec_b32 s31, s30
    0xBF88'0004, // s_cbranch_execz -> 0x74
    0x7E08'0220, // 0x64: v_mov_b32 v4, s32
    0x7E0A'0221, // v_mov_b32 v5, s33
    0x7E0C'0222, // v_mov_b32 v6, s34
    0x7E0E'0223, // v_mov_b32 v7, s35
    0x887E'1F7E, // 0x74: s_or_b32 exec_lo, exec_lo, s31
    0x887E'147E, // 0x78: s_or_b32 exec_lo, exec_lo, s20
    0xBE97'3C1E, // s_and_saveexec_b32 s23, s30
    0xBF88'0013, // s_cbranch_execz -> 0xd0
    0xBE9D'037E, // s_mov_b32 s29, exec_lo
    0x7E18'0504, // 0x88: v_readfirstlane_b32 s12, v4
    0x7E1A'0505, // v_readfirstlane_b32 s13, v5
    0x7E1C'0506, // v_readfirstlane_b32 s14, v6
    0x7E1E'0507, // v_readfirstlane_b32 s15, v7
    0xD4E2'0015,
    0x0002'080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
    0xD4E2'0016,
    0x0002'0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
    0x8715'1615, // s_and_b32 s21, s21, s22
    0xBE95'3C15, // s_and_saveexec_b32 s21, s21
    0xE030'1000,
    0x8003'0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
    0xBF8C'3F70, // s_waitcnt vmcnt(0)
    0x897E'157E, // s_xor_b32 exec_lo, exec_lo, s21
    0xBF89'FFF1, // s_cbranch_execnz -> 0x88
    0xBEFE'031D, // s_mov_b32 exec_lo, s29
    0xE070'1000,
    0x800A'0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0x887E'177E, // 0xd0: s_or_b32 exec_lo, exec_lo, s23
    0xBF81'0000, // s_endpgm
};
// The waterfall with `v_cmpx_eq` narrowing EXEC itself, and the retry as `todo & !EXEC` (`folded`:
// in one `s_andn2_wrexec`), as LLVM's AMDGPU backend writes it for GFX10 and later.
inline std::vector<std::uint32_t> cmpx_waterfall(bool wave64, bool folded)
{
    using fixtures_detail::word_at;
    const std::vector<std::uint32_t>& base = wave64 ? WATERFALL_FETCHED_WAVE64 : WATERFALL_JOINED_WAVE32;
    const std::uint32_t todo = wave64 ? 0xBE94'047E : 0xBE97'037E;
    const std::uint32_t restore = wave64 ? 0xBEFE'0414 : 0xBEFE'0317;
    const std::size_t reads = word_at(base, 0x7E18'0504);
    const std::size_t sample = word_at(base, 0xF09C'0F08);
    std::vector<std::uint32_t> w(base.begin(), base.begin() + static_cast<std::ptrdiff_t>(word_at(base, todo) + 1));
    // s_mov_b32 s21, exec_lo / s_mov_b64 s[22:23], exec
    w.push_back(wave64 ? 0xBE96'047E : 0xBE95'037E);
    w.insert(w.end(), base.begin() + static_cast<std::ptrdiff_t>(reads),
             base.begin() + static_cast<std::ptrdiff_t>(reads + 8));  // v_readfirstlane_b32 s12..s19
    w.insert(w.end(), {
                          0x7DE4'080C,  // v_cmpx_eq_u64 s[12:13], v[4:5]
                          0x7DE4'0C0E,  // v_cmpx_eq_u64 s[14:15], v[6:7]
                          0x7DE4'1010,  // v_cmpx_eq_u64 s[16:17], v[8:9]
                          0x7DE4'1412,  // v_cmpx_eq_u64 s[18:19], v[10:11]
                      });
    w.insert(w.end(), base.begin() + static_cast<std::ptrdiff_t>(sample),
             base.begin() + static_cast<std::ptrdiff_t>(sample + 2));  // image_sample_lz
    if (!wave64 && folded) {
        // s_andn2_wrexec_b32 s21, s21; s_cbranch_execnz
        w.insert(w.end(), {0xBE95'4715, 0xBF89'FFF0});
    } else if (!wave64 && !folded) {
        // s_andn2_b32 s21, s21, exec_lo; s_mov_b32 exec_lo, s21
        w.insert(w.end(), {0x8A15'7E15, 0xBEFE'0315, 0xBF89'FFEF});
    } else if (wave64 && folded) {
        // s_andn2_wrexec_b64 s[22:23], s[22:23]
        w.insert(w.end(), {0xBE96'3A16, 0xBF89'FFF0});
    } else {
        // s_andn2_b64 s[22:23], s[22:23], exec; s_mov_b64 exec, s[22:23]
        w.insert(w.end(), {0x8A96'7E16, 0xBEFE'0416, 0xBF89'FFEF});
    }
    w.insert(w.end(), base.begin() + static_cast<std::ptrdiff_t>(word_at(base, restore)), base.end());
    return w;
}

}  // namespace x2w::vs::resource::fixtures
