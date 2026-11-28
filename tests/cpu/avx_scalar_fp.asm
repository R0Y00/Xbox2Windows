; coverage: source cpu_aot_avx_scalar_fp
; Source-owned differential fixture for 28 VEX AVX scalar FP, half-register move, extract/insert, and compare instructions.
.code

vaddss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vaddss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddss_op ENDP

vsubss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vsubss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vsubss_op ENDP

vmulss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmulss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmulss_op ENDP

vdivss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vdivss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vdivss_op ENDP

vsqrtss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vsqrtss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vsqrtss_op ENDP

vminss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vminss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vminss_op ENDP

vmaxss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmaxss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmaxss_op ENDP

vroundss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vroundss xmm0, xmm1, xmm2, 08h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vroundss_op ENDP

vaddsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vaddsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vaddsd_op ENDP

vsubsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vsubsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vsubsd_op ENDP

vmulsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmulsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmulsd_op ENDP

vdivsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vdivsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vdivsd_op ENDP

vsqrtsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vsqrtsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vsqrtsd_op ENDP

vminsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vminsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vminsd_op ENDP

vmaxsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmaxsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmaxsd_op ENDP

vroundsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vroundsd xmm0, xmm1, xmm2, 09h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vroundsd_op ENDP

vmovss_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovss xmm2, dword ptr [rdx]
    vmovss xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovss_op ENDP

vmovhlps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmovhlps xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovhlps_op ENDP

vmovlhps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmovlhps xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovlhps_op ENDP

vmovlps_hps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovlps xmm0, xmm1, qword ptr [rdx]
    vmovhps xmm0, xmm0, qword ptr [rdx + 8]
    vmovlps qword ptr [r8], xmm0
    vmovhps qword ptr [r8 + 8], xmm0
    vzeroupper
    ret
vmovlps_hps_op ENDP

vmovlpd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovlpd xmm0, xmm1, qword ptr [rdx]
    vmovlpd qword ptr [r8], xmm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovlpd_op ENDP

vinsertps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vinsertps xmm0, xmm1, xmm2, 094h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vinsertps_op ENDP

vextractps_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vextractps eax, xmm0, 2
    vzeroupper
    ret
vextractps_op ENDP

vcomiss_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    xor eax, eax
    vcomiss xmm0, xmm1
    sete al
    jb vcomiss_cf
    vzeroupper
    ret
vcomiss_cf:
    or eax, 2
    vzeroupper
    ret
vcomiss_op ENDP

vucomiss_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    xor eax, eax
    vucomiss xmm0, xmm1
    sete al
    jb vucomiss_cf
    vzeroupper
    ret
vucomiss_cf:
    or eax, 2
    vzeroupper
    ret
vucomiss_op ENDP

vcomisd_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    xor eax, eax
    vcomisd xmm0, xmm1
    sete al
    jb vcomisd_cf
    vzeroupper
    ret
vcomisd_cf:
    or eax, 2
    vzeroupper
    ret
vcomisd_op ENDP

vucomisd_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]
    xor eax, eax
    vucomisd xmm0, xmm1
    sete al
    jb vucomisd_cf
    vzeroupper
    ret
vucomisd_cf:
    or eax, 2
    vzeroupper
    ret
vucomisd_op ENDP

END
