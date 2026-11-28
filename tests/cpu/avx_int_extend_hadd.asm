; coverage: source cpu_aot_avx_int_extend_hadd
; Source-owned differential fixture for 19 VEX AVX sign/zero-extend and horizontal integer instructions.
.code

vpmovsxbw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovsxbw xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovsxbw_op ENDP

vpmovsxbd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovsxbd xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovsxbd_op ENDP

vpmovsxbq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovsxbq xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovsxbq_op ENDP

vpmovsxwd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovsxwd xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovsxwd_op ENDP

vpmovsxwq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovsxwq xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovsxwq_op ENDP

vpmovsxdq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovsxdq xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovsxdq_op ENDP

vpmovzxbw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovzxbw xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovzxbw_op ENDP

vpmovzxbd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovzxbd xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovzxbd_op ENDP

vpmovzxbq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovzxbq xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovzxbq_op ENDP

vpmovzxwd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovzxwd xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovzxwd_op ENDP

vpmovzxwq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovzxwq xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovzxwq_op ENDP

vpmovzxdq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpmovzxdq xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovzxdq_op ENDP

vphaddw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vphaddw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphaddw_op ENDP

vphaddd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vphaddd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphaddd_op ENDP

vphaddsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vphaddsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphaddsw_op ENDP

vphsubw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vphsubw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphsubw_op ENDP

vphsubd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vphsubd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphsubd_op ENDP

vphsubsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vphsubsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphsubsw_op ENDP

vphminposuw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vphminposuw xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphminposuw_op ENDP

vpmovsxbd_ymm_mem_op PROC FRAME
    .endprolog
    vpmovsxbd ymm0, qword ptr [rcx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmovsxbd_ymm_mem_op ENDP

vphaddd_ymm_mem_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vphaddd ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vphaddd_ymm_mem_op ENDP

END
