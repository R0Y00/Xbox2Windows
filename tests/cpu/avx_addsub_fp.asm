; coverage: source cpu_aot_avx_addsub_fp
; Source-owned differential fixture for 2 VEX AVX alternating FP add/subtract instructions (vaddsubps, vaddsubpd).
.code

vaddsubps_xmm_reg_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vaddsubps xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubps_xmm_reg_op ENDP

vaddsubpd_xmm_reg_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vaddsubpd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubpd_xmm_reg_op ENDP

vaddsubps_xmm_mem_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vaddsubps xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubps_xmm_mem_op ENDP

vaddsubpd_xmm_mem_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vaddsubpd xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubpd_xmm_mem_op ENDP

vaddsubps_ymm_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vaddsubps ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubps_ymm_reg_op ENDP

vaddsubpd_ymm_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vaddsubpd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubpd_ymm_reg_op ENDP

vaddsubps_ymm_mem_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vaddsubps ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubps_ymm_mem_op ENDP

vaddsubpd_ymm_mem_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vaddsubpd ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubpd_ymm_mem_op ENDP

vaddsubps_ymm_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vaddsubps ymm0, ymm0, ymm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubps_ymm_alias_dst_src1_op ENDP

vaddsubpd_ymm_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vaddsubpd ymm0, ymm1, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubpd_ymm_alias_dst_src2_op ENDP

vaddsubps_ymm_alias_src1_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vaddsubps ymm0, ymm1, ymm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubps_ymm_alias_src1_src2_op ENDP

vaddsubpd_ymm_alias_all_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vaddsubpd ymm0, ymm0, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubpd_ymm_alias_all_op ENDP

vaddsubps_xmm_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vaddsubps xmm0, xmm1, xmm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubps_xmm_alias_dst_src2_op ENDP

vaddsubpd_xmm_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vaddsubpd xmm0, xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsubpd_xmm_alias_dst_src1_op ENDP

END
