s_mov_b32 s20, s7
s_mov_b32 s21, 0
s_load_dwordx4 s[24:27], s[20:21], 0x30
v_lshl_add_u32 v0, s10, 5, v0
v_or_b32_e32 v4, 0x12340000, v0
v_mul_lo_u32 v5, v0, 4
v_add_nc_u32_e32 v5, 1, v5
v_mul_lo_u32 v5, v5, 0x9e3779b1
v_or_b32_e32 v5, 1, v5
v_and_b32_e32 v6, 1, v0
v_xor_b32_e32 v5, v6, v5
s_waitcnt lgkmcnt(0)
buffer_atomic_cmpswap v[4:5], v0, s[24:27], 0 idxen glc
s_waitcnt vmcnt(0)
buffer_store_dword v4, v0, s[24:27], 0 idxen offset:4
buffer_store_dword v5, v0, s[24:27], 0 idxen offset:8
s_endpgm
