; coverage: source cpu_aot_avx_cvt_cmp_test
; Source-owned differential fixture for 28 AVX conversion, compare, flag-test, and aligned/streaming move instructions.
.code

vcvtdq2ps_op PROC FRAME
    .endprolog
    vmovdqa ymm1, ymmword ptr [rcx]
    vcvtdq2ps ymm0, ymm1
    vmovaps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtdq2ps_op ENDP

vcvtdq2pd_op PROC FRAME
    .endprolog
    vmovdqa xmm1, xmmword ptr [rcx]
    vcvtdq2pd ymm0, xmm1
    vmovapd ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtdq2pd_op ENDP

vcvtps2dq_op PROC FRAME
    .endprolog
    vmovaps ymm1, ymmword ptr [rcx]
    vcvtps2dq ymm0, ymm1
    vmovdqa ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtps2dq_op ENDP

vcvtpd2dq_op PROC FRAME
    .endprolog
    vmovapd ymm1, ymmword ptr [rcx]
    vcvtpd2dq xmm0, ymm1
    vmovdqa ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtpd2dq_op ENDP

vcvttps2dq_op PROC FRAME
    .endprolog
    vmovaps ymm1, ymmword ptr [rcx]
    vcvttps2dq ymm0, ymm1
    vmovdqa ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvttps2dq_op ENDP

vcvttpd2dq_op PROC FRAME
    .endprolog
    vmovapd ymm1, ymmword ptr [rcx]
    vcvttpd2dq xmm0, ymm1
    vmovdqa ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvttpd2dq_op ENDP

vcvtps2pd_op PROC FRAME
    .endprolog
    vmovaps xmm1, xmmword ptr [rcx]
    vcvtps2pd ymm0, xmm1
    vmovapd ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtps2pd_op ENDP

vcvtpd2ps_op PROC FRAME
    .endprolog
    vmovapd ymm1, ymmword ptr [rcx]
    vcvtpd2ps xmm0, ymm1
    vmovaps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtpd2ps_op ENDP

vcvtsi2ss_op PROC FRAME
    .endprolog
    vmovaps xmm1, xmmword ptr [rcx]
    mov eax, dword ptr [rdx]
    vcvtsi2ss xmm0, xmm1, eax
    vmovaps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsi2ss_op ENDP

vcvtsi2sd_op PROC FRAME
    .endprolog
    vmovapd xmm1, xmmword ptr [rcx]
    mov rax, qword ptr [rdx]
    vcvtsi2sd xmm0, xmm1, rax
    vmovapd ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsi2sd_op ENDP

vcvtss2sd_op PROC FRAME
    .endprolog
    vmovapd xmm1, xmmword ptr [rcx]
    vmovaps xmm2, xmmword ptr [rdx]
    vcvtss2sd xmm0, xmm1, xmm2
    vmovapd ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtss2sd_op ENDP

vcvtsd2ss_op PROC FRAME
    .endprolog
    vmovaps xmm1, xmmword ptr [rcx]
    vmovapd xmm2, xmmword ptr [rdx]
    vcvtsd2ss xmm0, xmm1, xmm2
    vmovaps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcvtsd2ss_op ENDP

vcvtss2si_op PROC FRAME
    .endprolog
    vmovaps xmm0, xmmword ptr [rcx]
    vcvtss2si eax, xmm0
    vzeroupper
    ret
vcvtss2si_op ENDP

vcvttss2si_op PROC FRAME
    .endprolog
    vmovaps xmm0, xmmword ptr [rcx]
    vcvttss2si rax, xmm0
    vzeroupper
    ret
vcvttss2si_op ENDP

vcvtsd2si_op PROC FRAME
    .endprolog
    vmovapd xmm0, xmmword ptr [rcx]
    vcvtsd2si eax, xmm0
    vzeroupper
    ret
vcvtsd2si_op ENDP

vcvttsd2si_op PROC FRAME
    .endprolog
    vmovapd xmm0, xmmword ptr [rcx]
    vcvttsd2si rax, xmm0
    vzeroupper
    ret
vcvttsd2si_op ENDP

vcmpps_op PROC FRAME
    .endprolog
    vmovaps ymm1, ymmword ptr [rcx]
    vmovaps ymm2, ymmword ptr [rdx]
    vcmpps ymm0, ymm1, ymm2, 1
    vmovaps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_op ENDP

vcmppd_op PROC FRAME
    .endprolog
    vmovapd ymm1, ymmword ptr [rcx]
    vmovapd ymm2, ymmword ptr [rdx]
    vcmppd ymm0, ymm1, ymm2, 2
    vmovapd ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmppd_op ENDP

vcmpss_op PROC FRAME
    .endprolog
    vmovaps xmm1, xmmword ptr [rcx]
    vmovaps xmm2, xmmword ptr [rdx]
    vcmpss xmm0, xmm1, xmm2, 4
    vmovaps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpss_op ENDP

vcmpsd_op PROC FRAME
    .endprolog
    vmovapd xmm1, xmmword ptr [rcx]
    vmovapd xmm2, xmmword ptr [rdx]
    vcmpsd xmm0, xmm1, xmm2, 7
    vmovapd ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpsd_op ENDP

vptest_op PROC FRAME
    .endprolog
    vmovdqa ymm0, ymmword ptr [rcx]
    vmovdqa ymm1, ymmword ptr [rdx]
    xor eax, eax
    vptest ymm0, ymm1
    sete al
    jb vptest_cf
    vzeroupper
    ret
vptest_cf:
    or eax, 2
    vzeroupper
    ret
vptest_op ENDP

vtestps_op PROC FRAME
    .endprolog
    vmovaps ymm0, ymmword ptr [rcx]
    vmovaps ymm1, ymmword ptr [rdx]
    xor eax, eax
    vtestps ymm0, ymm1
    sete al
    jb vtestps_cf
    vzeroupper
    ret
vtestps_cf:
    or eax, 2
    vzeroupper
    ret
vtestps_op ENDP

vtestpd_op PROC FRAME
    .endprolog
    vmovapd ymm0, ymmword ptr [rcx]
    vmovapd ymm1, ymmword ptr [rdx]
    xor eax, eax
    vtestpd ymm0, ymm1
    sete al
    jb vtestpd_cf
    vzeroupper
    ret
vtestpd_cf:
    or eax, 2
    vzeroupper
    ret
vtestpd_op ENDP

vlddqu_op PROC FRAME
    .endprolog
    vlddqu ymm0, ymmword ptr [rcx]
    vmovdqa ymmword ptr [r8], ymm0
    vzeroupper
    ret
vlddqu_op ENDP

vcmpps_true_signaling PROC FRAME
    .endprolog
    vmovaps ymm1, ymmword ptr [rcx]
    vmovaps ymm2, ymmword ptr [rdx]
    vcmpps ymm0, ymm1, ymm2, 31
    vmovaps ymmword ptr [r8], ymm0
    vzeroupper
    ret
vcmpps_true_signaling ENDP

END
