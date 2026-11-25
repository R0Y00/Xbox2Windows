.text
s_mov_b64 exec, 1
s_and_b32 s10, s3, 0xf000000
s_cbranch_scc1 consumer_wait
v_mov_b32_e32 v0, 0
v_mov_b32_e32 v1, 37
ds_write_b32 v0, v1 offset:8
s_waitcnt_depctr 0xffe3
v_mov_b32_e32 v1, 51
s_waitcnt lgkmcnt(0)
s_branch common_barrier
consumer_wait:
s_nop 0
common_barrier:
s_barrier
s_cbranch_scc0 done
v_mov_b32_e32 v0, 0
ds_read_b32 v2, v0 offset:8
s_waitcnt_depctr 0xffe3
s_waitcnt lgkmcnt(0)
v_readfirstlane_b32 s11, v2
done:
s_endpgm
