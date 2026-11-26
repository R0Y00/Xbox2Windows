; coverage: record cpu_aot_aes_pclmul
; Source-built MASM fixture for legacy SSE and VEX.128 AES-NI and PCLMULQDQ
; instructions from docs/coverage/x64_zen2.tsv: register and flat [rcx|rdx|r8]
; m128 forms, YMM[255:128] preservation vs VEX.128 zeroing, vzeroupper,
; aliased source-before-dst overwrite, and discarded faulting m128 read.
.code

; --- Legacy SSE register forms ---

op_aesenc_reg PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    aesenclast xmm0, xmm1
ELSE
    aesenc xmm0, xmm1
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesenc_reg ENDP

op_aesenclast_reg PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    aesenc xmm0, xmm1
ELSE
    aesenclast xmm0, xmm1
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesenclast_reg ENDP

op_aesdec_reg PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    aesdeclast xmm0, xmm1
ELSE
    aesdec xmm0, xmm1
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesdec_reg ENDP

op_aesdeclast_reg PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    aesdec xmm0, xmm1
ELSE
    aesdeclast xmm0, xmm1
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesdeclast_reg ENDP

op_aesimc_reg PROC FRAME
    .endprolog
    movdqa xmm1, xmmword ptr [rcx]
IFDEF ALTER_AES_PCLMUL
    movdqa xmm0, xmm1
ELSE
    aesimc xmm0, xmm1
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesimc_reg ENDP

op_aeskeygenassist_reg PROC FRAME
    .endprolog
    movdqa xmm1, xmmword ptr [rcx]
IFDEF ALTER_AES_PCLMUL
    aeskeygenassist xmm0, xmm1, 1Ch
ELSE
    aeskeygenassist xmm0, xmm1, 1Bh
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_aeskeygenassist_reg ENDP

op_pclmulqdq_00_reg PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    pclmulqdq xmm0, xmm1, 11h
ELSE
    pclmulqdq xmm0, xmm1, 00h
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_pclmulqdq_00_reg ENDP

op_pclmulqdq_11_reg PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    pclmulqdq xmm0, xmm1, 00h
ELSE
    pclmulqdq xmm0, xmm1, 11h
ENDIF
    movdqa xmmword ptr [r8], xmm0
    ret
op_pclmulqdq_11_reg ENDP

op_pclmulqdq_ignore_bits_reg PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
    pclmulqdq xmm0, xmm1, 0EFh
    movdqa xmmword ptr [r8], xmm0
    ret
op_pclmulqdq_ignore_bits_reg ENDP

; --- Legacy SSE flat [rcx|rdx] m128 forms (16-byte aligned) ---

op_aesenc_mem PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    aesenc xmm0, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesenc_mem ENDP

op_aesenclast_mem PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    aesenclast xmm0, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesenclast_mem ENDP

op_aesdec_mem PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    aesdec xmm0, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesdec_mem ENDP

op_aesdeclast_mem PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    aesdeclast xmm0, xmmword ptr [rdx]
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesdeclast_mem ENDP

op_aesimc_mem PROC FRAME
    .endprolog
    aesimc xmm0, xmmword ptr [rcx]
    movdqa xmmword ptr [r8], xmm0
    ret
op_aesimc_mem ENDP

op_aeskeygenassist_mem PROC FRAME
    .endprolog
    aeskeygenassist xmm0, xmmword ptr [rcx], 80h
    movdqa xmmword ptr [r8], xmm0
    ret
op_aeskeygenassist_mem ENDP

op_pclmulqdq_01_mem PROC FRAME
    .endprolog
    movdqa xmm0, xmmword ptr [rcx]
    pclmulqdq xmm0, xmmword ptr [rdx], 01h
    movdqa xmmword ptr [r8], xmm0
    ret
op_pclmulqdq_01_mem ENDP

; --- VEX.128 register forms ---

op_vaesenc_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vmovdqu xmm2, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    vaesdec xmm0, xmm1, xmm2
ELSE
    vaesenc xmm0, xmm1, xmm2
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesenc_reg ENDP

op_vaesenclast_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vmovdqu xmm2, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    vaesdeclast xmm0, xmm1, xmm2
ELSE
    vaesenclast xmm0, xmm1, xmm2
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesenclast_reg ENDP

op_vaesdec_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vmovdqu xmm2, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    vaesenc xmm0, xmm1, xmm2
ELSE
    vaesdec xmm0, xmm1, xmm2
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesdec_reg ENDP

op_vaesdeclast_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vmovdqu xmm2, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    vaesenclast xmm0, xmm1, xmm2
ELSE
    vaesdeclast xmm0, xmm1, xmm2
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesdeclast_reg ENDP

op_vaesimc_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
IFDEF ALTER_AES_PCLMUL
    vmovdqa xmm0, xmm1
ELSE
    vaesimc xmm0, xmm1
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesimc_reg ENDP

op_vaeskeygenassist_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
IFDEF ALTER_AES_PCLMUL
    vaeskeygenassist xmm0, xmm1, 37h
ELSE
    vaeskeygenassist xmm0, xmm1, 36h
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaeskeygenassist_reg ENDP

op_vpclmulqdq_10_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vmovdqu xmm2, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    vpclmulqdq xmm0, xmm1, xmm2, 01h
ELSE
    vpclmulqdq xmm0, xmm1, xmm2, 10h
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vpclmulqdq_10_reg ENDP

op_vpclmulqdq_ignore_bits_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vmovdqu xmm2, xmmword ptr [rdx]
    vpclmulqdq xmm0, xmm1, xmm2, 0FCh
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vpclmulqdq_ignore_bits_reg ENDP

; --- VEX.128 flat [rcx|rdx] m128 forms ---

op_vaesenc_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
IFDEF ALTER_AES_PCLMUL
    vaesdec xmm0, xmm1, xmmword ptr [rdx]
ELSE
    vaesenc xmm0, xmm1, xmmword ptr [rdx]
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesenc_mem ENDP

op_vaesenclast_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vaesenclast xmm0, xmm1, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesenclast_mem ENDP

op_vaesdec_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vaesdec xmm0, xmm1, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesdec_mem ENDP

op_vaesdeclast_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vaesdeclast xmm0, xmm1, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesdeclast_mem ENDP

op_vaesimc_mem PROC FRAME
    .endprolog
    vaesimc xmm0, xmmword ptr [rcx]
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaesimc_mem ENDP

op_vaeskeygenassist_mem PROC FRAME
    .endprolog
    vaeskeygenassist xmm0, xmmword ptr [rcx], 0FFh
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vaeskeygenassist_mem ENDP

op_vpclmulqdq_11_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
IFDEF ALTER_AES_PCLMUL
    vpclmulqdq xmm0, xmm1, xmmword ptr [rdx], 00h
ELSE
    vpclmulqdq xmm0, xmm1, xmmword ptr [rdx], 11h
ENDIF
    vmovdqu xmmword ptr [r8], xmm0
    ret
op_vpclmulqdq_11_mem ENDP

; --- YMM[255:128] preservation vs VEX.128 zeroing, vzeroupper, aliasing, [r8] src, discarded read ---

op_sse_preserves_ymm_hi PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    movdqa xmm1, xmmword ptr [rdx]
IFDEF ALTER_AES_PCLMUL
    vaesenc xmm0, xmm0, xmm1
ELSE
    aesenc xmm0, xmm1
ENDIF
    vmovdqu ymmword ptr [r8], ymm0
    ret
op_sse_preserves_ymm_hi ENDP

op_vex_zeroes_ymm_hi PROC FRAME
    .endprolog
    vmovdqu ymm0, ymmword ptr [rcx]
    vmovdqu xmm1, xmmword ptr [rdx]
    vaesenc xmm0, xmm0, xmm1
    vmovdqu ymmword ptr [r8], ymm0
    ret
op_vex_zeroes_ymm_hi ENDP

op_vzeroupper_clears_ymm_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, ymmword ptr [rcx]
    vzeroupper
    vmovdqu ymmword ptr [r8], ymm1
    ret
op_vzeroupper_clears_ymm_hi ENDP

op_r8_source_then_store PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vaesenc xmm0, xmm1, xmmword ptr [r8]
    vpclmulqdq xmm0, xmm0, xmmword ptr [rdx], 00h
    vmovdqa xmmword ptr [r8], xmm0
    ret
op_r8_source_then_store ENDP

op_dst_alias_src PROC FRAME
    .endprolog
    vmovdqu xmm0, xmmword ptr [rcx]
    vaesenc xmm0, xmm0, xmm0
    aesimc xmm0, xmm0
    aeskeygenassist xmm0, xmm0, 01h
    vpclmulqdq xmm0, xmm0, xmm0, 01h
    movdqu xmmword ptr [r8], xmm0
    ret
op_dst_alias_src ENDP

op_aes_pclmul_discarded_read PROC FRAME
    .endprolog
    vmovdqu xmm1, xmmword ptr [rcx]
    vaesenc xmm0, xmm1, xmmword ptr [rdx]
    vmovdqu xmmword ptr [r8], xmm1
    ret
op_aes_pclmul_discarded_read ENDP

END
