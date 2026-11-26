s_mov_b32 s20, s7
s_mov_b32 s21, 0
s_load_dwordx4 s[24:27], s[20:21], 0x30
v_lshl_add_u32 v0, s10, 5, v0
v_lshlrev_b32_e32 v2, 3, v0
v_add_nc_u32_e32 v4, 1, v0
s_waitcnt lgkmcnt(0)
buffer_atomic_add v4, v2, s[24:27], 0 offen glc
s_waitcnt vmcnt(0)
buffer_store_dword v4, v2, s[24:27], 0 offen offset:4
s_endpgm
