; coverage: record cpu_aot_sse41_fp
; Source-owned MASM fixture for legacy SSE4.1 floating-point instructions
; (roundps, roundpd, roundss, roundsd, dpps, dppd, insertps, extractps)
; across register and flat [rcx|rdx|r8] memory-source forms plus glue moves.
; ABI: void fn(const void* rcx, const void* rdx, void* r8).
.code

; --- ROUNDPS (reg imm 0,1,2,3,4,8 & flat [rcx|rdx|r8] mem) ---
op_roundps_reg_0 PROC FRAME
    .endprolog
    movaps xmm1, XMMWORD PTR [rcx]
    roundps xmm0, xmm1, 00h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_reg_0 ENDP

op_roundps_reg_1 PROC FRAME
    .endprolog
    movaps xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_SSE41_FP
    roundps xmm0, xmm1, 02h
ELSE
    roundps xmm0, xmm1, 01h
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_reg_1 ENDP

op_roundps_reg_2 PROC FRAME
    .endprolog
    movaps xmm1, XMMWORD PTR [rcx]
    roundps xmm0, xmm1, 02h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_reg_2 ENDP

op_roundps_reg_3 PROC FRAME
    .endprolog
    movaps xmm1, XMMWORD PTR [rcx]
    roundps xmm0, xmm1, 03h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_reg_3 ENDP

op_roundps_reg_4 PROC FRAME
    .endprolog
    movaps xmm1, XMMWORD PTR [rcx]
    roundps xmm0, xmm1, 04h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_reg_4 ENDP

op_roundps_reg_8 PROC FRAME
    .endprolog
    movaps xmm1, XMMWORD PTR [rcx]
    roundps xmm0, xmm1, 08h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_reg_8 ENDP

op_roundps_mem_rcx PROC FRAME
    .endprolog
    roundps xmm0, XMMWORD PTR [rcx], 01h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_mem_rcx ENDP

op_roundps_mem_rdx PROC FRAME
    .endprolog
    roundps xmm0, XMMWORD PTR [rdx], 02h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_mem_rdx ENDP

op_roundps_mem_r8 PROC FRAME
    .endprolog
    roundps xmm0, XMMWORD PTR [r8], 03h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_roundps_mem_r8 ENDP

; --- ROUNDPD (reg imm 0,1,2,3,4 & flat [rcx|rdx|r8] mem) ---
op_roundpd_reg_0 PROC FRAME
    .endprolog
    movapd xmm1, XMMWORD PTR [rcx]
    roundpd xmm0, xmm1, 00h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_reg_0 ENDP

op_roundpd_reg_1 PROC FRAME
    .endprolog
    movapd xmm1, XMMWORD PTR [rcx]
    roundpd xmm0, xmm1, 01h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_reg_1 ENDP

op_roundpd_reg_2 PROC FRAME
    .endprolog
    movapd xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_SSE41_FP
    roundpd xmm0, xmm1, 01h
ELSE
    roundpd xmm0, xmm1, 02h
ENDIF
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_reg_2 ENDP

op_roundpd_reg_3 PROC FRAME
    .endprolog
    movapd xmm1, XMMWORD PTR [rcx]
    roundpd xmm0, xmm1, 03h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_reg_3 ENDP

op_roundpd_reg_4 PROC FRAME
    .endprolog
    movapd xmm1, XMMWORD PTR [rcx]
    roundpd xmm0, xmm1, 04h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_reg_4 ENDP

op_roundpd_mem_rcx PROC FRAME
    .endprolog
    roundpd xmm0, XMMWORD PTR [rcx], 00h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_mem_rcx ENDP

op_roundpd_mem_rdx PROC FRAME
    .endprolog
    roundpd xmm0, XMMWORD PTR [rdx], 01h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_mem_rdx ENDP

op_roundpd_mem_r8 PROC FRAME
    .endprolog
    roundpd xmm0, XMMWORD PTR [r8], 02h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_roundpd_mem_r8 ENDP

; --- ROUNDSS (preserves dst[127:32], reg & flat [rcx|rdx|r8] m32) ---
op_roundss_reg_0 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_SSE41_FP
    roundss xmm0, xmm1, 03h
ELSE
    roundss xmm0, xmm1, 00h
ENDIF
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_reg_0 ENDP

op_roundss_reg_1 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
    roundss xmm0, xmm1, 01h
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_reg_1 ENDP

op_roundss_reg_2 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
    roundss xmm0, xmm1, 02h
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_reg_2 ENDP

op_roundss_reg_3 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
    roundss xmm0, xmm1, 03h
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_reg_3 ENDP

op_roundss_reg_4 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
    roundss xmm0, xmm1, 04h
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_reg_4 ENDP

op_roundss_mem_rcx PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rdx]
    roundss xmm0, DWORD PTR [rcx], 01h
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_mem_rcx ENDP

op_roundss_mem_rdx PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    roundss xmm0, DWORD PTR [rdx], 02h
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_mem_rdx ENDP

op_roundss_mem_r8 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    roundss xmm0, DWORD PTR [r8], 03h
    movups XMMWORD PTR [r8], xmm0
    ret
op_roundss_mem_r8 ENDP

; --- ROUNDSD (preserves dst[127:64], reg & flat [rcx|rdx|r8] m64) ---
op_roundsd_reg_0 PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rcx]
    movupd xmm1, XMMWORD PTR [rdx]
    roundsd xmm0, xmm1, 00h
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_reg_0 ENDP

op_roundsd_reg_1 PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rcx]
    movupd xmm1, XMMWORD PTR [rdx]
    roundsd xmm0, xmm1, 01h
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_reg_1 ENDP

op_roundsd_reg_2 PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rcx]
    movupd xmm1, XMMWORD PTR [rdx]
    roundsd xmm0, xmm1, 02h
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_reg_2 ENDP

op_roundsd_reg_3 PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rcx]
    movupd xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_SSE41_FP
    roundsd xmm0, xmm1, 02h
ELSE
    roundsd xmm0, xmm1, 03h
ENDIF
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_reg_3 ENDP

op_roundsd_reg_4 PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rcx]
    movupd xmm1, XMMWORD PTR [rdx]
    roundsd xmm0, xmm1, 04h
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_reg_4 ENDP

op_roundsd_mem_rcx PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rdx]
    roundsd xmm0, QWORD PTR [rcx], 01h
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_mem_rcx ENDP

op_roundsd_mem_rdx PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rcx]
    roundsd xmm0, QWORD PTR [rdx], 02h
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_mem_rdx ENDP

op_roundsd_mem_r8 PROC FRAME
    .endprolog
    movupd xmm0, XMMWORD PTR [rcx]
    roundsd xmm0, QWORD PTR [r8], 03h
    movupd XMMWORD PTR [r8], xmm0
    ret
op_roundsd_mem_r8 ENDP

; --- DPPS (reg & flat [rcx|rdx|r8] mem) ---
op_dpps_reg_f1 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
    dpps xmm0, xmm1, 0F1h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_dpps_reg_f1 ENDP

op_dpps_reg_ff PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_SSE41_FP
    dpps xmm0, xmm1, 055h
ELSE
    dpps xmm0, xmm1, 0FFh
ENDIF
    movaps XMMWORD PTR [r8], xmm0
    ret
op_dpps_reg_ff ENDP

op_dpps_reg_5a PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
    dpps xmm0, xmm1, 05Ah
    movaps XMMWORD PTR [r8], xmm0
    ret
op_dpps_reg_5a ENDP

op_dpps_reg_0f PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    movaps xmm1, XMMWORD PTR [rdx]
    dpps xmm0, xmm1, 00Fh
    movaps XMMWORD PTR [r8], xmm0
    ret
op_dpps_reg_0f ENDP

op_dpps_mem_rcx PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rdx]
    dpps xmm0, XMMWORD PTR [rcx], 0F3h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_dpps_mem_rcx ENDP

op_dpps_mem_rdx PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    dpps xmm0, XMMWORD PTR [rdx], 0C5h
    movaps XMMWORD PTR [r8], xmm0
    ret
op_dpps_mem_rdx ENDP

op_dpps_mem_r8 PROC FRAME
    .endprolog
    movaps xmm0, XMMWORD PTR [rcx]
    dpps xmm0, XMMWORD PTR [r8], 0AAh
    movaps XMMWORD PTR [r8], xmm0
    ret
op_dpps_mem_r8 ENDP

; --- DPPD (reg & flat [rcx|rdx|r8] mem) ---
op_dppd_reg_33 PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rcx]
    movapd xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_SSE41_FP
    dppd xmm0, xmm1, 011h
ELSE
    dppd xmm0, xmm1, 033h
ENDIF
    movapd XMMWORD PTR [r8], xmm0
    ret
op_dppd_reg_33 ENDP

op_dppd_reg_11 PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rcx]
    movapd xmm1, XMMWORD PTR [rdx]
    dppd xmm0, xmm1, 011h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_dppd_reg_11 ENDP

op_dppd_reg_22 PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rcx]
    movapd xmm1, XMMWORD PTR [rdx]
    dppd xmm0, xmm1, 022h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_dppd_reg_22 ENDP

op_dppd_reg_03 PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rcx]
    movapd xmm1, XMMWORD PTR [rdx]
    dppd xmm0, xmm1, 003h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_dppd_reg_03 ENDP

op_dppd_mem_rcx PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rdx]
    dppd xmm0, XMMWORD PTR [rcx], 031h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_dppd_mem_rcx ENDP

op_dppd_mem_rdx PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rcx]
    dppd xmm0, XMMWORD PTR [rdx], 032h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_dppd_mem_rdx ENDP

op_dppd_mem_r8 PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rcx]
    dppd xmm0, XMMWORD PTR [r8], 023h
    movapd XMMWORD PTR [r8], xmm0
    ret
op_dppd_mem_r8 ENDP

; --- INSERTPS (reg count_s/count_d/zmask & flat [rcx|rdx|r8] m32) ---
op_insertps_reg_s1_d2_z5 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_SSE41_FP
    insertps xmm0, xmm1, 060h
ELSE
    insertps xmm0, xmm1, 065h
ENDIF
    movups XMMWORD PTR [r8], xmm0
    ret
op_insertps_reg_s1_d2_z5 ENDP

op_insertps_reg_s3_d0_z0 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
    insertps xmm0, xmm1, 0C0h
    movups XMMWORD PTR [r8], xmm0
    ret
op_insertps_reg_s3_d0_z0 ENDP

op_insertps_reg_s0_d3_z8 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    movups xmm1, XMMWORD PTR [rdx]
    insertps xmm0, xmm1, 038h
    movups XMMWORD PTR [r8], xmm0
    ret
op_insertps_reg_s0_d3_z8 ENDP

op_insertps_mem_rcx PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rdx]
    insertps xmm0, DWORD PTR [rcx], 0D2h
    movups XMMWORD PTR [r8], xmm0
    ret
op_insertps_mem_rcx ENDP

op_insertps_mem_rdx PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    insertps xmm0, DWORD PTR [rdx], 0E4h
    movups XMMWORD PTR [r8], xmm0
    ret
op_insertps_mem_rdx ENDP

op_insertps_mem_r8 PROC FRAME
    .endprolog
    movups xmm0, XMMWORD PTR [rcx]
    insertps xmm0, DWORD PTR [r8], 091h
    movups XMMWORD PTR [r8], xmm0
    ret
op_insertps_mem_r8 ENDP

; --- EXTRACTPS (r32 zero-extend, r64 zero-extend, and m32 [r8]) ---
op_extractps_r32_lane2 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    mov rax, QWORD PTR [rdx]
IFDEF ALTER_SSE41_FP
    extractps eax, xmm0, 01h
ELSE
    extractps eax, xmm0, 02h
ENDIF
    mov QWORD PTR [r8], rax
    ret
op_extractps_r32_lane2 ENDP

op_extractps_r64_lane3 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    db 66h, 49h, 0Fh, 3Ah, 17h, 0C1h, 03h ; extractps r9, xmm0, 03h (REX.W)
    mov QWORD PTR [r8], r9
    ret
op_extractps_r64_lane3 ENDP

op_extractps_mem_lane1 PROC FRAME
    .endprolog
    movdqu xmm0, XMMWORD PTR [rcx]
    extractps DWORD PTR [r8], xmm0, 01h
    ret
op_extractps_mem_lane1 ENDP

; --- Glue transport pipeline ---
op_sse41_glue_pipeline PROC FRAME
    .endprolog
    movss xmm0, DWORD PTR [rcx]
    movsd xmm1, QWORD PTR [rdx]
    movdqa xmm2, XMMWORD PTR [rcx]
    movdqu xmm3, XMMWORD PTR [rdx]
    movss xmm2, xmm1
    movsd xmm3, xmm0
    roundps xmm4, xmm2, 01h
    insertps xmm4, xmm3, 052h
    movupd XMMWORD PTR [r8], xmm4
    ret
op_sse41_glue_pipeline ENDP

END
