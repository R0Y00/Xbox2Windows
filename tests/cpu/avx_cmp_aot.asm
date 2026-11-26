; coverage: record cpu_aot_avx_cmp
; Source-owned MASM fixture for VEX.128 and VEX.256 AVX FP compare instructions
; (vcmpps, vcmppd, vcmpss, vcmpsd) across all 32 5-bit predicates (0x00..0x1F),
; flat [rcx|rdx|r8] memory sources, scalar upper-lane preservation, and glue moves.
; ABI: void fn(const void* rcx, const void* rdx, void* r8).
.code

MK_VCMPPS_YMM MACRO name, pred
name PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
    vcmpps ymm0, ymm1, ymm2, pred
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
name ENDP
ENDM

MK_VCMPPD_YMM MACRO name, pred
name PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
    vcmppd ymm0, ymm1, ymm2, pred
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
name ENDP
ENDM

MK_VCMPSS MACRO name, pred
name PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
    vcmpss xmm0, xmm1, xmm2, pred
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
name ENDP
ENDM

MK_VCMPSD MACRO name, pred
name PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
    vcmpsd xmm0, xmm1, xmm2, pred
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
name ENDP
ENDM

; --- VCMPPS YMM predicates 0x00..0x1F ---
op_vcmpps_ymm_p00 PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_AVX_CMP
    vcmpps ymm0, ymm1, ymm2, 04h
ELSE
    vcmpps ymm0, ymm1, ymm2, 00h
ENDIF
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_ymm_p00 ENDP

MK_VCMPPS_YMM op_vcmpps_ymm_p01, 01h
MK_VCMPPS_YMM op_vcmpps_ymm_p02, 02h
MK_VCMPPS_YMM op_vcmpps_ymm_p03, 03h
MK_VCMPPS_YMM op_vcmpps_ymm_p04, 04h
MK_VCMPPS_YMM op_vcmpps_ymm_p05, 05h
MK_VCMPPS_YMM op_vcmpps_ymm_p06, 06h
MK_VCMPPS_YMM op_vcmpps_ymm_p07, 07h
MK_VCMPPS_YMM op_vcmpps_ymm_p08, 08h
MK_VCMPPS_YMM op_vcmpps_ymm_p09, 09h
MK_VCMPPS_YMM op_vcmpps_ymm_p0a, 0Ah
MK_VCMPPS_YMM op_vcmpps_ymm_p0b, 0Bh
MK_VCMPPS_YMM op_vcmpps_ymm_p0c, 0Ch
MK_VCMPPS_YMM op_vcmpps_ymm_p0d, 0Dh
MK_VCMPPS_YMM op_vcmpps_ymm_p0e, 0Eh
MK_VCMPPS_YMM op_vcmpps_ymm_p0f, 0Fh
MK_VCMPPS_YMM op_vcmpps_ymm_p10, 10h
MK_VCMPPS_YMM op_vcmpps_ymm_p11, 11h
MK_VCMPPS_YMM op_vcmpps_ymm_p12, 12h
MK_VCMPPS_YMM op_vcmpps_ymm_p13, 13h
MK_VCMPPS_YMM op_vcmpps_ymm_p14, 14h
MK_VCMPPS_YMM op_vcmpps_ymm_p15, 15h
MK_VCMPPS_YMM op_vcmpps_ymm_p16, 16h
MK_VCMPPS_YMM op_vcmpps_ymm_p17, 17h
MK_VCMPPS_YMM op_vcmpps_ymm_p18, 18h
MK_VCMPPS_YMM op_vcmpps_ymm_p19, 19h
MK_VCMPPS_YMM op_vcmpps_ymm_p1a, 1Ah
MK_VCMPPS_YMM op_vcmpps_ymm_p1b, 1Bh
MK_VCMPPS_YMM op_vcmpps_ymm_p1c, 1Ch
MK_VCMPPS_YMM op_vcmpps_ymm_p1d, 1Dh
MK_VCMPPS_YMM op_vcmpps_ymm_p1e, 1Eh
MK_VCMPPS_YMM op_vcmpps_ymm_p1f, 1Fh

; --- VCMPPD YMM predicates 0x00..0x1F ---
MK_VCMPPD_YMM op_vcmppd_ymm_p00, 00h
op_vcmppd_ymm_p01 PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_AVX_CMP
    vcmppd ymm0, ymm1, ymm2, 05h
ELSE
    vcmppd ymm0, ymm1, ymm2, 01h
ENDIF
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmppd_ymm_p01 ENDP

MK_VCMPPD_YMM op_vcmppd_ymm_p02, 02h
MK_VCMPPD_YMM op_vcmppd_ymm_p03, 03h
MK_VCMPPD_YMM op_vcmppd_ymm_p04, 04h
MK_VCMPPD_YMM op_vcmppd_ymm_p05, 05h
MK_VCMPPD_YMM op_vcmppd_ymm_p06, 06h
MK_VCMPPD_YMM op_vcmppd_ymm_p07, 07h
MK_VCMPPD_YMM op_vcmppd_ymm_p08, 08h
MK_VCMPPD_YMM op_vcmppd_ymm_p09, 09h
MK_VCMPPD_YMM op_vcmppd_ymm_p0a, 0Ah
MK_VCMPPD_YMM op_vcmppd_ymm_p0b, 0Bh
MK_VCMPPD_YMM op_vcmppd_ymm_p0c, 0Ch
MK_VCMPPD_YMM op_vcmppd_ymm_p0d, 0Dh
MK_VCMPPD_YMM op_vcmppd_ymm_p0e, 0Eh
MK_VCMPPD_YMM op_vcmppd_ymm_p0f, 0Fh
MK_VCMPPD_YMM op_vcmppd_ymm_p10, 10h
MK_VCMPPD_YMM op_vcmppd_ymm_p11, 11h
MK_VCMPPD_YMM op_vcmppd_ymm_p12, 12h
MK_VCMPPD_YMM op_vcmppd_ymm_p13, 13h
MK_VCMPPD_YMM op_vcmppd_ymm_p14, 14h
MK_VCMPPD_YMM op_vcmppd_ymm_p15, 15h
MK_VCMPPD_YMM op_vcmppd_ymm_p16, 16h
MK_VCMPPD_YMM op_vcmppd_ymm_p17, 17h
MK_VCMPPD_YMM op_vcmppd_ymm_p18, 18h
MK_VCMPPD_YMM op_vcmppd_ymm_p19, 19h
MK_VCMPPD_YMM op_vcmppd_ymm_p1a, 1Ah
MK_VCMPPD_YMM op_vcmppd_ymm_p1b, 1Bh
MK_VCMPPD_YMM op_vcmppd_ymm_p1c, 1Ch
MK_VCMPPD_YMM op_vcmppd_ymm_p1d, 1Dh
MK_VCMPPD_YMM op_vcmppd_ymm_p1e, 1Eh
MK_VCMPPD_YMM op_vcmppd_ymm_p1f, 1Fh

; --- VCMPSS predicates 0x00..0x1F ---
MK_VCMPSS op_vcmpss_p00, 00h
MK_VCMPSS op_vcmpss_p01, 01h
op_vcmpss_p02 PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_AVX_CMP
    vcmpss xmm0, xmm2, xmm1, 02h
ELSE
    vcmpss xmm0, xmm1, xmm2, 02h
ENDIF
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpss_p02 ENDP

MK_VCMPSS op_vcmpss_p03, 03h
MK_VCMPSS op_vcmpss_p04, 04h
MK_VCMPSS op_vcmpss_p05, 05h
MK_VCMPSS op_vcmpss_p06, 06h
MK_VCMPSS op_vcmpss_p07, 07h
MK_VCMPSS op_vcmpss_p08, 08h
MK_VCMPSS op_vcmpss_p09, 09h
MK_VCMPSS op_vcmpss_p0a, 0Ah
MK_VCMPSS op_vcmpss_p0b, 0Bh
MK_VCMPSS op_vcmpss_p0c, 0Ch
MK_VCMPSS op_vcmpss_p0d, 0Dh
MK_VCMPSS op_vcmpss_p0e, 0Eh
MK_VCMPSS op_vcmpss_p0f, 0Fh
MK_VCMPSS op_vcmpss_p10, 10h
MK_VCMPSS op_vcmpss_p11, 11h
MK_VCMPSS op_vcmpss_p12, 12h
MK_VCMPSS op_vcmpss_p13, 13h
MK_VCMPSS op_vcmpss_p14, 14h
MK_VCMPSS op_vcmpss_p15, 15h
MK_VCMPSS op_vcmpss_p16, 16h
MK_VCMPSS op_vcmpss_p17, 17h
MK_VCMPSS op_vcmpss_p18, 18h
MK_VCMPSS op_vcmpss_p19, 19h
MK_VCMPSS op_vcmpss_p1a, 1Ah
MK_VCMPSS op_vcmpss_p1b, 1Bh
MK_VCMPSS op_vcmpss_p1c, 1Ch
MK_VCMPSS op_vcmpss_p1d, 1Dh
MK_VCMPSS op_vcmpss_p1e, 1Eh
MK_VCMPSS op_vcmpss_p1f, 1Fh

; --- VCMPSD predicates 0x00..0x1F ---
MK_VCMPSD op_vcmpsd_p00, 00h
MK_VCMPSD op_vcmpsd_p01, 01h
MK_VCMPSD op_vcmpsd_p02, 02h
MK_VCMPSD op_vcmpsd_p03, 03h
MK_VCMPSD op_vcmpsd_p04, 04h
MK_VCMPSD op_vcmpsd_p05, 05h
MK_VCMPSD op_vcmpsd_p06, 06h
MK_VCMPSD op_vcmpsd_p07, 07h
MK_VCMPSD op_vcmpsd_p08, 08h
MK_VCMPSD op_vcmpsd_p09, 09h
MK_VCMPSD op_vcmpsd_p0a, 0Ah
MK_VCMPSD op_vcmpsd_p0b, 0Bh
MK_VCMPSD op_vcmpsd_p0c, 0Ch
MK_VCMPSD op_vcmpsd_p0d, 0Dh
MK_VCMPSD op_vcmpsd_p0e, 0Eh
MK_VCMPSD op_vcmpsd_p0f, 0Fh
MK_VCMPSD op_vcmpsd_p10, 10h
MK_VCMPSD op_vcmpsd_p11, 11h
MK_VCMPSD op_vcmpsd_p12, 12h
MK_VCMPSD op_vcmpsd_p13, 13h
MK_VCMPSD op_vcmpsd_p14, 14h
MK_VCMPSD op_vcmpsd_p15, 15h
MK_VCMPSD op_vcmpsd_p16, 16h
MK_VCMPSD op_vcmpsd_p17, 17h
MK_VCMPSD op_vcmpsd_p18, 18h
MK_VCMPSD op_vcmpsd_p19, 19h
MK_VCMPSD op_vcmpsd_p1a, 1Ah
MK_VCMPSD op_vcmpsd_p1b, 1Bh
MK_VCMPSD op_vcmpsd_p1c, 1Ch
MK_VCMPSD op_vcmpsd_p1d, 1Dh
MK_VCMPSD op_vcmpsd_p1e, 1Eh
MK_VCMPSD op_vcmpsd_p1f, 1Fh

; --- VEX.128 packed & upper-zeroing checks ---
op_vcmpps_xmm_reg PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_AVX_CMP
    vcmpps ymm0, ymm1, ymm2, 00h
ELSE
    vcmpps xmm0, xmm1, xmm2, 00h
ENDIF
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_xmm_reg ENDP

op_vcmppd_xmm_reg PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm2, YMMWORD PTR [rdx]
    vcmppd xmm0, xmm1, xmm2, 0Ch
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmppd_xmm_reg ENDP

op_vcmpps_xmm_mem PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vcmpps xmm0, xmm1, XMMWORD PTR [rdx], 02h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_xmm_mem ENDP

op_vcmppd_xmm_mem PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
    vcmppd xmm0, xmm1, XMMWORD PTR [rdx], 11h
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmppd_xmm_mem ENDP

; --- Flat [rcx|rdx|r8] memory-source forms ---
op_vcmpps_ymm_mem_rcx PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rdx]
    vcmpps ymm0, ymm1, YMMWORD PTR [rcx], 01h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_ymm_mem_rcx ENDP

op_vcmpps_ymm_mem_rdx PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vcmpps ymm0, ymm1, YMMWORD PTR [rdx], 0Eh
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_ymm_mem_rdx ENDP

op_vcmpps_ymm_mem_r8 PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vcmpps ymm0, ymm1, YMMWORD PTR [r8], 12h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_ymm_mem_r8 ENDP

op_vcmppd_ymm_mem_rcx PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rdx]
    vcmppd ymm0, ymm1, YMMWORD PTR [rcx], 0Dh
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmppd_ymm_mem_rcx ENDP

op_vcmppd_ymm_mem_rdx PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vcmppd ymm0, ymm1, YMMWORD PTR [rdx], 15h
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmppd_ymm_mem_rdx ENDP

op_vcmppd_ymm_mem_r8 PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vcmppd ymm0, ymm1, YMMWORD PTR [r8], 1Eh
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmppd_ymm_mem_r8 ENDP

op_vcmpss_mem_rcx PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rdx]
    vcmpss xmm0, xmm1, DWORD PTR [rcx], 01h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpss_mem_rcx ENDP

op_vcmpss_mem_rdx PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [r8]
    vmovups ymm1, YMMWORD PTR [rcx]
    vcmpss xmm0, xmm1, DWORD PTR [rdx], 11h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpss_mem_rdx ENDP

op_vcmpss_mem_r8 PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vcmpss xmm0, xmm1, DWORD PTR [r8], 0Ch
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpss_mem_r8 ENDP

op_vcmpsd_mem_rcx PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rdx]
    vcmpsd xmm0, xmm1, QWORD PTR [rcx], 02h
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpsd_mem_rcx ENDP

op_vcmpsd_mem_rdx PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [r8]
    vmovupd ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_AVX_CMP
    vcmpsd xmm0, xmm1, QWORD PTR [rdx], 0Ah
ELSE
    vcmpsd xmm0, xmm1, QWORD PTR [rdx], 0Eh
ENDIF
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpsd_mem_rdx ENDP

op_vcmpsd_mem_r8 PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vcmpsd xmm0, xmm1, QWORD PTR [r8], 1Dh
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpsd_mem_r8 ENDP

; --- High imm8 bits [7:5] ignored (0xE5 & 0x1F == 0x05) ---
op_vcmpps_imm_highbits PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vmovups ymm2, YMMWORD PTR [rdx]
    vcmpps ymm0, ymm1, ymm2, 0E5h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_imm_highbits ENDP

; --- Register aliasing & scalar/packed glue moves ---
op_vcmpps_alias_dst_src1 PROC FRAME
    .endprolog
    vmovups ymm0, YMMWORD PTR [rcx]
    vmovups ymm1, YMMWORD PTR [rdx]
    vcmpps ymm0, ymm0, ymm1, 04h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_alias_dst_src1 ENDP

op_vcmppd_alias_dst_src2 PROC FRAME
    .endprolog
    vmovupd ymm1, YMMWORD PTR [rcx]
    vmovupd ymm0, YMMWORD PTR [rdx]
    vcmppd ymm0, ymm1, ymm0, 06h
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmppd_alias_dst_src2 ENDP

op_vcmpps_alias_src1_src2 PROC FRAME
    .endprolog
    vmovups ymm1, YMMWORD PTR [rcx]
    vcmpps ymm0, ymm1, ymm1, 03h
    vmovups YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpps_alias_src1_src2 ENDP

op_vcmpsd_alias_all PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [rcx]
    vcmpsd xmm0, xmm0, xmm0, 00h
    vmovupd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vcmpsd_alias_all ENDP

op_vmovss_mem_glue PROC FRAME
    .endprolog
    vmovss xmm0, DWORD PTR [rcx]
    vmovss xmm1, DWORD PTR [rdx]
    vcmpss xmm0, xmm0, xmm1, 01h
    vmovss DWORD PTR [r8], xmm0
    vzeroupper
    ret
op_vmovss_mem_glue ENDP

op_vmovsd_mem_glue PROC FRAME
    .endprolog
    vmovsd xmm0, QWORD PTR [rcx]
    vmovsd xmm1, QWORD PTR [rdx]
    vcmpsd xmm0, xmm0, xmm1, 01h
    vmovsd QWORD PTR [r8], xmm0
    vzeroupper
    ret
op_vmovsd_mem_glue ENDP

op_vmovss_reg_merge_glue PROC FRAME
    .endprolog
    vmovaps ymm0, YMMWORD PTR [r8]
    vmovaps xmm1, XMMWORD PTR [rcx]
    vmovaps xmm2, XMMWORD PTR [rdx]
    vmovss xmm0, xmm1, xmm2
    vmovaps YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovss_reg_merge_glue ENDP

op_vmovsd_reg_merge_glue PROC FRAME
    .endprolog
    vmovapd ymm0, YMMWORD PTR [r8]
    vmovapd xmm1, XMMWORD PTR [rcx]
    vmovapd xmm2, XMMWORD PTR [rdx]
    vmovsd xmm0, xmm1, xmm2
    vmovapd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmovsd_reg_merge_glue ENDP

op_vmov_aligned_glue PROC FRAME
    .endprolog
    vmovaps ymm1, YMMWORD PTR [rcx]
    vmovapd ymm2, YMMWORD PTR [rdx]
    vmovaps ymm3, ymm1
    vmovapd ymm4, ymm2
    vcmpps ymm0, ymm3, ymm4, 0Ch
    vmovaps YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vmov_aligned_glue ENDP

END
