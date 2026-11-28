; coverage: source cpu_aot_avx_int_minmax_mul
; Source-owned differential fixture for 33 VEX AVX packed saturating, min/max, avg, multiply, and sad instructions.
.code

vpaddsb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpaddsb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpaddsb_op ENDP

vpaddsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpaddsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpaddsw_op ENDP

vpaddusb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpaddusb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpaddusb_op ENDP

vpaddusw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpaddusw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpaddusw_op ENDP

vpsubsb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubsb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubsb_op ENDP

vpsubsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubsw_op ENDP

vpsubusb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubusb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubusb_op ENDP

vpsubusw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubusw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubusw_op ENDP

vpminsb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpminsb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpminsb_op ENDP

vpmaxsb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaxsb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaxsb_op ENDP

vpminub_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpminub xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpminub_op ENDP

vpmaxub_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaxub xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaxub_op ENDP

vpminsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpminsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpminsw_op ENDP

vpmaxsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaxsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaxsw_op ENDP

vpminuw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpminuw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpminuw_op ENDP

vpmaxuw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaxuw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaxuw_op ENDP

vpminsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpminsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpminsd_op ENDP

vpmaxsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaxsd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaxsd_op ENDP

vpminud_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpminud xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpminud_op ENDP

vpmaxud_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaxud xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaxud_op ENDP

vpavgb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpavgb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpavgb_op ENDP

vpavgw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpavgw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpavgw_op ENDP

vpmulhw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmulhw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmulhw_op ENDP

vpmulhuw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmulhuw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmulhuw_op ENDP

vpmullw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmullw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmullw_op ENDP

vpmulld_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmulld xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmulld_op ENDP

vpmuldq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmuldq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmuldq_op ENDP

vpmuludq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmuludq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmuludq_op ENDP

vpmulhrsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmulhrsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmulhrsw_op ENDP

vpmaddwd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaddwd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaddwd_op ENDP

vpmaddubsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpmaddubsw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpmaddubsw_op ENDP

vpsadbw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsadbw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsadbw_op ENDP

vmpsadbw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmpsadbw xmm0, xmm1, xmm2, 5
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmpsadbw_op ENDP

vmpsadbw256_mem PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmpsadbw ymm0, ymm1, ymmword ptr [rdx], 15h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmpsadbw256_mem ENDP

END
