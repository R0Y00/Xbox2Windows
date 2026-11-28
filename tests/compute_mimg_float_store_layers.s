s_mov_b32 s20, s7
s_mov_b32 s21, 0
s_load_dwordx8 s[24:31], s[20:21], 0x40
v_lshl_add_u32 v8, s10, 2, v0
v_lshl_add_u32 v9, s11, 2, v1
v_lshl_add_u32 v10, s12, 1, v2
v_or_b32_e32 v14, 0x3f810000, v8
v_or_b32_e32 v15, 0x40110000, v9
v_or_b32_e32 v16, 0x40410000, v10
v_mov_b32_e32 v17, 0xbf810001
s_waitcnt lgkmcnt(0)
image_store v[14:17], v[8:10], s[24:31] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY
s_endpgm
