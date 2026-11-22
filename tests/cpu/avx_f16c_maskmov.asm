; Source-owned differential fixture for 7 AVX F16C, masked FP load/store, and non-temporal store instructions.
.code

vcvtph2ps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcvtph2ps ymm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtph2ps_op ENDP

vcvtps2ph_rn_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcvtps2ph xmm0, ymm1, 0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtps2ph_rn_op ENDP

vcvtps2ph_floor_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vcvtps2ph xmm0, ymm1, 1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtps2ph_floor_op ENDP

vmaskmovps_load_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rdx]
    vmaskmovps ymm0, ymm1, ymmword ptr [rcx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmaskmovps_load_op ENDP

vmaskmovps_store_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vmaskmovps ymmword ptr [r8], ymm1, ymm0
    vzeroupper
    ret
vmaskmovps_store_op ENDP

vmaskmovpd_load_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rdx]
    vmaskmovpd ymm0, ymm1, ymmword ptr [rcx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmaskmovpd_load_op ENDP

vmaskmovpd_store_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    vmaskmovpd ymmword ptr [r8], ymm1, ymm0
    vzeroupper
    ret
vmaskmovpd_store_op ENDP

vmovntps_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovntps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovntps_op ENDP

vmovntpd_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovntpd ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovntpd_op ENDP

vmovntdq_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovntdq ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovntdq_op ENDP

END
