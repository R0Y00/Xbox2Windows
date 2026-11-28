; coverage: source cpu_aot_sse2_packed_cvt
; Source-owned differential fixture for 25 baseline 128-bit SSE/SSE2 packed conversions, compares, half moves, and scalar sqrt/min/max.
.code

cvtdq2ps_op PROC FRAME
    .endprolog
    movdqa xmm1, xmmword ptr [rcx]
    cvtdq2ps xmm0, xmm1
    movaps xmmword ptr [r8], xmm0
    ret
cvtdq2ps_op ENDP

cvtdq2pd_op PROC FRAME
    .endprolog
    movdqa xmm1, xmmword ptr [rcx]
    cvtdq2pd xmm0, xmm1
    movapd xmmword ptr [r8], xmm0
    ret
cvtdq2pd_op ENDP

cvtps2dq_op PROC FRAME
    .endprolog
    movaps xmm1, xmmword ptr [rcx]
    cvtps2dq xmm0, xmm1
    movdqa xmmword ptr [r8], xmm0
    ret
cvtps2dq_op ENDP

cvtpd2dq_op PROC FRAME
    .endprolog
    movapd xmm1, xmmword ptr [rcx]
    cvtpd2dq xmm0, xmm1
    movdqa xmmword ptr [r8], xmm0
    ret
cvtpd2dq_op ENDP

cvttps2dq_op PROC FRAME
    .endprolog
    movaps xmm1, xmmword ptr [rcx]
    cvttps2dq xmm0, xmm1
    movdqa xmmword ptr [r8], xmm0
    ret
cvttps2dq_op ENDP

cvttpd2dq_op PROC FRAME
    .endprolog
    movapd xmm1, xmmword ptr [rcx]
    cvttpd2dq xmm0, xmm1
    movdqa xmmword ptr [r8], xmm0
    ret
cvttpd2dq_op ENDP

cvtps2pd_op PROC FRAME
    .endprolog
    movaps xmm1, xmmword ptr [rcx]
    cvtps2pd xmm0, xmm1
    movapd xmmword ptr [r8], xmm0
    ret
cvtps2pd_op ENDP

cvtpd2ps_op PROC FRAME
    .endprolog
    movapd xmm1, xmmword ptr [rcx]
    cvtpd2ps xmm0, xmm1
    movaps xmmword ptr [r8], xmm0
    ret
cvtpd2ps_op ENDP

cmpps_lt_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movaps xmm1, xmmword ptr [rdx]
    cmpps xmm0, xmm1, 1
    movaps xmmword ptr [r8], xmm0
    ret
cmpps_lt_op ENDP

cmpps_unord_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movaps xmm1, xmmword ptr [rdx]
    cmpps xmm0, xmm1, 3
    movaps xmmword ptr [r8], xmm0
    ret
cmpps_unord_op ENDP

cmppd_le_op PROC FRAME
    .endprolog
    movapd xmm0, xmmword ptr [rcx]
    cmppd xmm0, xmmword ptr [rdx], 2
    movapd xmmword ptr [r8], xmm0
    ret
cmppd_le_op ENDP

cmppd_neq_op PROC FRAME
    .endprolog
    movapd xmm0, xmmword ptr [rcx]
    movapd xmm1, xmmword ptr [rdx]
    cmppd xmm0, xmm1, 4
    movapd xmmword ptr [r8], xmm0
    ret
cmppd_neq_op ENDP

movhlps_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movaps xmm1, xmmword ptr [rdx]
    movhlps xmm0, xmm1
    movaps xmmword ptr [r8], xmm0
    ret
movhlps_op ENDP

movlhps_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movaps xmm1, xmmword ptr [rdx]
    movlhps xmm0, xmm1
    movaps xmmword ptr [r8], xmm0
    ret
movlhps_op ENDP

movlps_hps_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movlps xmm0, qword ptr [rdx]
    movhps xmm0, qword ptr [rdx + 8]
    movlps qword ptr [r8], xmm0
    movhps qword ptr [r8 + 8], xmm0
    ret
movlps_hps_op ENDP

movlpd_hpd_op PROC FRAME
    .endprolog
    movapd xmm0, xmmword ptr [rcx]
    movlpd xmm0, qword ptr [rdx]
    movhpd xmm0, qword ptr [rdx + 8]
    movlpd qword ptr [r8], xmm0
    movhpd qword ptr [r8 + 8], xmm0
    ret
movlpd_hpd_op ENDP

sqrtss_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movaps xmm1, xmmword ptr [rdx]
    sqrtss xmm0, xmm1
    movaps xmmword ptr [r8], xmm0
    ret
sqrtss_op ENDP

minss_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movaps xmm1, xmmword ptr [rdx]
    minss xmm0, xmm1
    movaps xmmword ptr [r8], xmm0
    ret
minss_op ENDP

maxss_op PROC FRAME
    .endprolog
    movaps xmm0, xmmword ptr [rcx]
    movaps xmm1, xmmword ptr [rdx]
    maxss xmm0, xmm1
    movaps xmmword ptr [r8], xmm0
    ret
maxss_op ENDP

sqrtsd_op PROC FRAME
    .endprolog
    movapd xmm0, xmmword ptr [rcx]
    movapd xmm1, xmmword ptr [rdx]
    sqrtsd xmm0, xmm1
    movapd xmmword ptr [r8], xmm0
    ret
sqrtsd_op ENDP

minsd_op PROC FRAME
    .endprolog
    movapd xmm0, xmmword ptr [rcx]
    movapd xmm1, xmmword ptr [rdx]
    minsd xmm0, xmm1
    movapd xmmword ptr [r8], xmm0
    ret
minsd_op ENDP

maxsd_op PROC FRAME
    .endprolog
    movapd xmm0, xmmword ptr [rcx]
    movapd xmm1, xmmword ptr [rdx]
    maxsd xmm0, xmm1
    movapd xmmword ptr [r8], xmm0
    ret
maxsd_op ENDP

END
