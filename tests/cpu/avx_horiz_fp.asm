; coverage: source cpu_aot_avx_horiz_fp
; Source-owned differential fixture for 4 VEX AVX horizontal FP instructions (vhaddps, vhaddpd, vhsubps, vhsubpd).
.code

vhaddps_xmm_reg_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhaddps xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddps_xmm_reg_op ENDP

vhaddpd_xmm_reg_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhaddpd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddpd_xmm_reg_op ENDP

vhsubps_xmm_reg_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhsubps xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubps_xmm_reg_op ENDP

vhsubpd_xmm_reg_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhsubpd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubpd_xmm_reg_op ENDP

vhaddps_xmm_mem_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vhaddps xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddps_xmm_mem_op ENDP

vhaddpd_xmm_mem_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vhaddpd xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddpd_xmm_mem_op ENDP

vhsubps_xmm_mem_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vhsubps xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubps_xmm_mem_op ENDP

vhsubpd_xmm_mem_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [r8]
    vmovups ymm1, ymmword ptr [rcx]
    vhsubpd xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubpd_xmm_mem_op ENDP

vhaddps_ymm_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhaddps ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddps_ymm_reg_op ENDP

vhaddpd_ymm_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhaddpd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddpd_ymm_reg_op ENDP

vhsubps_ymm_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhsubps ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubps_ymm_reg_op ENDP

vhsubpd_ymm_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vhsubpd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubpd_ymm_reg_op ENDP

vhaddps_ymm_mem_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vhaddps ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddps_ymm_mem_op ENDP

vhaddpd_ymm_mem_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vhaddpd ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddpd_ymm_mem_op ENDP

vhsubps_ymm_mem_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vhsubps ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubps_ymm_mem_op ENDP

vhsubpd_ymm_mem_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vhsubpd ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubpd_ymm_mem_op ENDP

vhaddps_ymm_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vhaddps ymm0, ymm0, ymm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddps_ymm_alias_dst_src1_op ENDP

vhaddpd_ymm_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vhaddpd ymm0, ymm1, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddpd_ymm_alias_dst_src2_op ENDP

vhsubps_ymm_alias_src1_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vhsubps ymm0, ymm1, ymm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubps_ymm_alias_src1_src2_op ENDP

vhsubpd_ymm_alias_all_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vhsubpd ymm0, ymm0, ymm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubpd_ymm_alias_all_op ENDP

vhaddps_xmm_alias_dst_src2_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm0, ymmword ptr [rdx]
    vhaddps xmm0, xmm1, xmm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhaddps_xmm_alias_dst_src2_op ENDP

vhsubps_xmm_alias_dst_src1_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vhsubps xmm0, xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vhsubps_xmm_alias_dst_src1_op ENDP

END
