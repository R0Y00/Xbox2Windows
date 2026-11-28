; coverage: record cpu_aot_avx_round
; Source-built MASM fixture for AVX rounding instructions from docs/coverage/x64_zen2.tsv
; group AVX: VROUNDPS (128/256-bit), VROUNDPD (128/256-bit), VROUNDSS, and VROUNDSD.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes)
;   RDX = const void* input_b (32 bytes)
;   R8  = void*       output  (32 bytes)
.code

; --- VROUNDPS XMM (128-bit) ---

op_vroundps_xmm_rn PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps xmm0, xmm1, 3
ELSE
    vroundps xmm0, xmm1, 0
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundps_xmm_rn ENDP

op_vroundps_xmm_rd PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps xmm0, xmm1, 2
ELSE
    vroundps xmm0, xmm1, 1
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundps_xmm_rd ENDP

op_vroundps_xmm_ru PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps xmm0, xmm1, 1
ELSE
    vroundps xmm0, xmm1, 2
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundps_xmm_ru ENDP

op_vroundps_xmm_rz PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps xmm0, xmm1, 0
ELSE
    vroundps xmm0, xmm1, 3
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundps_xmm_rz ENDP

op_vroundps_xmm_mxcsr PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps xmm0, xmm1, 3
ELSE
    vroundps xmm0, xmm1, 4
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundps_xmm_mxcsr ENDP

op_vroundps_xmm_suppress PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps xmm0, xmm1, 11
ELSE
    vroundps xmm0, xmm1, 8
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundps_xmm_suppress ENDP

op_vroundps_xmm_mem PROC FRAME
    .endprolog
IFDEF ALTER_ROUND
    vroundps xmm2, XMMWORD PTR [rdx], 1
ELSE
    vroundps xmm2, XMMWORD PTR [rdx], 1Ah
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundps_xmm_mem ENDP

; --- VROUNDPS YMM (256-bit) ---

op_vroundps_ymm_rn PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps ymm0, ymm1, 3
ELSE
    vroundps ymm0, ymm1, 0
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundps_ymm_rn ENDP

op_vroundps_ymm_rd PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps ymm0, ymm1, 2
ELSE
    vroundps ymm0, ymm1, 1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundps_ymm_rd ENDP

op_vroundps_ymm_ru PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps ymm0, ymm1, 1
ELSE
    vroundps ymm0, ymm1, 2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundps_ymm_ru ENDP

op_vroundps_ymm_rz PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps ymm0, ymm1, 0
ELSE
    vroundps ymm0, ymm1, 3
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundps_ymm_rz ENDP

op_vroundps_ymm_mxcsr PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundps ymm0, ymm1, 1
ELSE
    vroundps ymm0, ymm1, 4
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundps_ymm_mxcsr ENDP

op_vroundps_ymm_mem PROC FRAME
    .endprolog
IFDEF ALTER_ROUND
    vroundps ymm3, YMMWORD PTR [rdx], 2
ELSE
    vroundps ymm3, YMMWORD PTR [rdx], 9
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm3
    vzeroupper
    ret
op_vroundps_ymm_mem ENDP

; --- VROUNDPD XMM (128-bit) ---

op_vroundpd_xmm_rn PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd xmm0, xmm1, 3
ELSE
    vroundpd xmm0, xmm1, 0
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundpd_xmm_rn ENDP

op_vroundpd_xmm_rd PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd xmm0, xmm1, 2
ELSE
    vroundpd xmm0, xmm1, 1
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundpd_xmm_rd ENDP

op_vroundpd_xmm_ru PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd xmm0, xmm1, 1
ELSE
    vroundpd xmm0, xmm1, 2
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundpd_xmm_ru ENDP

op_vroundpd_xmm_rz PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd xmm0, xmm1, 0
ELSE
    vroundpd xmm0, xmm1, 3
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundpd_xmm_rz ENDP

op_vroundpd_xmm_mxcsr PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd xmm0, xmm1, 2
ELSE
    vroundpd xmm0, xmm1, 4
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vroundpd_xmm_mxcsr ENDP

op_vroundpd_xmm_mem PROC FRAME
    .endprolog
IFDEF ALTER_ROUND
    vroundpd xmm4, XMMWORD PTR [rdx], 1
ELSE
    vroundpd xmm4, XMMWORD PTR [rdx], 10
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm4
    ret
op_vroundpd_xmm_mem ENDP

; --- VROUNDPD YMM (256-bit) ---

op_vroundpd_ymm_rn PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd ymm0, ymm1, 3
ELSE
    vroundpd ymm0, ymm1, 0
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundpd_ymm_rn ENDP

op_vroundpd_ymm_rd PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd ymm0, ymm1, 2
ELSE
    vroundpd ymm0, ymm1, 1
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundpd_ymm_rd ENDP

op_vroundpd_ymm_ru PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd ymm0, ymm1, 1
ELSE
    vroundpd ymm0, ymm1, 2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundpd_ymm_ru ENDP

op_vroundpd_ymm_rz PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd ymm0, ymm1, 0
ELSE
    vroundpd ymm0, ymm1, 3
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundpd_ymm_rz ENDP

op_vroundpd_ymm_mxcsr PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundpd ymm0, ymm1, 3
ELSE
    vroundpd ymm0, ymm1, 12
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundpd_ymm_mxcsr ENDP

op_vroundpd_ymm_mem PROC FRAME
    .endprolog
IFDEF ALTER_ROUND
    vroundpd ymm5, YMMWORD PTR [rdx], 0
ELSE
    vroundpd ymm5, YMMWORD PTR [rdx], 11
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm5
    vzeroupper
    ret
op_vroundpd_ymm_mem ENDP

; --- VROUNDSS (scalar f32: preserves src1[127:32], zeroes [255:128]) ---

op_vroundss_rn PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundss xmm2, xmm0, xmm1, 3
ELSE
    vroundss xmm2, xmm0, xmm1, 0
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundss_rn ENDP

op_vroundss_rd PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundss xmm2, xmm0, xmm1, 2
ELSE
    vroundss xmm2, xmm0, xmm1, 1
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundss_rd ENDP

op_vroundss_ru PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundss xmm2, xmm0, xmm1, 1
ELSE
    vroundss xmm2, xmm0, xmm1, 2
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundss_ru ENDP

op_vroundss_rz PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundss xmm2, xmm0, xmm1, 0
ELSE
    vroundss xmm2, xmm0, xmm1, 3
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundss_rz ENDP

op_vroundss_mxcsr PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundss xmm2, xmm0, xmm1, 3
ELSE
    vroundss xmm2, xmm0, xmm1, 4
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundss_mxcsr ENDP

op_vroundss_mem PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundss xmm2, xmm0, DWORD PTR [rdx], 1
ELSE
    vroundss xmm2, xmm0, DWORD PTR [rdx], 10
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundss_mem ENDP

; --- VROUNDSD (scalar f64: preserves src1[127:64], zeroes [255:128]) ---

op_vroundsd_rn PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundsd xmm2, xmm0, xmm1, 3
ELSE
    vroundsd xmm2, xmm0, xmm1, 0
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundsd_rn ENDP

op_vroundsd_rd PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundsd xmm2, xmm0, xmm1, 2
ELSE
    vroundsd xmm2, xmm0, xmm1, 1
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundsd_rd ENDP

op_vroundsd_ru PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundsd xmm2, xmm0, xmm1, 1
ELSE
    vroundsd xmm2, xmm0, xmm1, 2
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundsd_ru ENDP

op_vroundsd_rz PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundsd xmm2, xmm0, xmm1, 0
ELSE
    vroundsd xmm2, xmm0, xmm1, 3
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundsd_rz ENDP

op_vroundsd_mxcsr PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_ROUND
    vroundsd xmm2, xmm0, xmm1, 2
ELSE
    vroundsd xmm2, xmm0, xmm1, 12
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundsd_mxcsr ENDP

op_vroundsd_mem PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
IFDEF ALTER_ROUND
    vroundsd xmm2, xmm0, QWORD PTR [rdx], 2
ELSE
    vroundsd xmm2, xmm0, QWORD PTR [rdx], 9
ENDIF
    vmovdqu XMMWORD PTR [r8], xmm2
    ret
op_vroundsd_mem ENDP

; --- VEX.128 upper-128 zeroing, [r8] reload, and discarded faulting m32 read ---

op_vroundps_xmm_zero_upper PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vroundps xmm0, xmm0, 0
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundps_xmm_zero_upper ENDP

op_vroundpd_xmm_zero_upper PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vroundpd xmm0, xmm0, 1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundpd_xmm_zero_upper ENDP

op_vroundss_zero_upper PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
    vroundss xmm0, xmm0, xmm1, 2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundss_zero_upper ENDP

op_vroundsd_zero_upper PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rdx]
    vroundsd xmm0, xmm0, xmm1, 3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vroundsd_zero_upper ENDP

op_vround_r8_reload PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vroundpd ymm1, YMMWORD PTR [r8], 0
    vmovdqu YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_vround_r8_reload ENDP

op_vround_discarded_read PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vroundss xmm5, xmm0, DWORD PTR [rdx], 0
    vmovdqu XMMWORD PTR [r8], xmm0
    ret
op_vround_discarded_read ENDP

END
