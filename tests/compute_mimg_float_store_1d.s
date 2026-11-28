s_mov_b32 s12, s7
s_mov_b32 s13, 0
s_load_dwordx8 s[24:31], s[12:13], 0x40
s_load_dwordx8 s[32:39], s[12:13], 0x60
v_lshl_add_u32 v8, s10, 2, v0
v_and_b32_e32 v9, 1, v8
v_add_nc_u32_e32 v9, 1, v9
s_waitcnt lgkmcnt(0)
image_load_mip v[14:17], v[8:9], s[32:39] dmask:0xf dim:SQ_RSRC_IMG_1D
s_waitcnt vmcnt(0)
image_store v[14:17], v8, s[24:31] dmask:0xf dim:SQ_RSRC_IMG_1D
s_endpgm
