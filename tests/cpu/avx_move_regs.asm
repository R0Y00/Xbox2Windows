; Source-built MASM fixture exercising 50 surviving AVX bit-move, shuffle,
; blend, broadcast, lane, permute, unpack, logical and masked load/store forms
; (excluding the 5 move forms carved out to cpu_vex_aot.py (PR #325),
; vmovdqu retained as transport glue) across XMM and YMM
; widths and encoding variants.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (buffer)
;   RDX = const void* input_b (buffer; also used to pre-seed dest on narrow writes)
;   R8  = void*       output  (buffer)
.code

; --- Logical bitwise ops (8 forms, XMM and YMM) ---

op_andpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vandpd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andpd_xmm ENDP

op_andpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vandpd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andpd_ymm ENDP

op_andps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vandps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andps_xmm ENDP

op_andps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vandps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andps_ymm ENDP

op_andnpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vandnpd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andnpd_xmm ENDP

op_andnpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vandnpd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andnpd_ymm ENDP

op_andnps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vandnps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andnps_xmm ENDP

op_andnps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vandnps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_andnps_ymm ENDP

op_orpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vorpd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_orpd_xmm ENDP

op_orpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vorpd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_orpd_ymm ENDP

op_orps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vorps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_orps_xmm ENDP

op_orps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vorps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_orps_ymm ENDP

op_xorpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vxorpd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_xorpd_xmm ENDP

op_xorpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vxorpd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_xorpd_ymm ENDP

op_xorps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vxorps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_xorps_xmm ENDP

op_xorps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vxorps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_xorps_ymm ENDP

; --- Blends (4 forms, immediate and sign-bit variable) ---

op_blendpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vblendpd xmm0, xmm1, xmm2, 0FDh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendpd_xmm ENDP

op_blendpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vblendpd ymm0, ymm1, ymm2, 0F5h
ELSE
    vblendpd ymm0, ymm1, ymm2, 0FAh
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendpd_ymm ENDP

op_blendpd_ymm_all PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vblendpd ymm0, ymm1, ymm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendpd_ymm_all ENDP

op_blendpd_ymm_none PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vblendpd ymm0, ymm1, ymm2, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendpd_ymm_none ENDP

op_blendps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vblendps xmm0, xmm1, xmm2, 0FAh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendps_xmm ENDP

op_blendps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vblendps ymm0, ymm1, ymm2, 55h
ELSE
    vblendps ymm0, ymm1, ymm2, 0AAh
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendps_ymm ENDP

op_blendps_ymm_all PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vblendps ymm0, ymm1, ymm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendps_ymm_all ENDP

op_blendps_ymm_none PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vblendps ymm0, ymm1, ymm2, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendps_ymm_none ENDP

; Variable blends: RCX = src1 (32B), RDX = [0..31] src2, [32..63] mask, R8 = out (32B).

op_blendvpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovdqu xmm3, XMMWORD PTR [rdx + 32]
    vblendvpd xmm0, xmm1, xmm2, xmm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendvpd_xmm ENDP

op_blendvpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vmovdqu ymm3, YMMWORD PTR [rdx + 32]
    vblendvpd ymm0, ymm1, ymm2, ymm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendvpd_ymm ENDP

op_blendvps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovdqu xmm3, XMMWORD PTR [rdx + 32]
    vblendvps xmm0, xmm1, xmm2, xmm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendvps_xmm ENDP

op_blendvps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vmovdqu ymm3, YMMWORD PTR [rdx + 32]
    vblendvps ymm0, ymm1, ymm2, ymm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_blendvps_ymm ENDP

; --- Broadcasts (3 forms: m128, m64/xmm, m32/xmm across lane positions) ---

op_broadcastf128 PROC FRAME
    .endprolog
IFDEF ALTER_IMM
    vbroadcastf128 ymm0, XMMWORD PTR [rcx + 16]
ELSE
    vbroadcastf128 ymm0, XMMWORD PTR [rcx]
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastf128 ENDP

op_broadcastsd_mem0 PROC FRAME
    .endprolog
    vbroadcastsd ymm0, QWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastsd_mem0 ENDP

op_broadcastsd_mem1 PROC FRAME
    .endprolog
    vbroadcastsd ymm0, QWORD PTR [rcx + 8]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastsd_mem1 ENDP

op_broadcastsd_mem2 PROC FRAME
    .endprolog
    vbroadcastsd ymm0, QWORD PTR [rcx + 16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastsd_mem2 ENDP

op_broadcastsd_mem3 PROC FRAME
    .endprolog
    vbroadcastsd ymm0, QWORD PTR [rcx + 24]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastsd_mem3 ENDP

op_broadcastsd_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vbroadcastsd ymm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastsd_reg ENDP

op_broadcastss_xmm_mem0 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vbroadcastss xmm0, DWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_xmm_mem0 ENDP

op_broadcastss_xmm_mem1 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vbroadcastss xmm0, DWORD PTR [rcx + 4]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_xmm_mem1 ENDP

op_broadcastss_xmm_mem2 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vbroadcastss xmm0, DWORD PTR [rcx + 8]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_xmm_mem2 ENDP

op_broadcastss_xmm_mem3 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vbroadcastss xmm0, DWORD PTR [rcx + 12]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_xmm_mem3 ENDP

op_broadcastss_ymm_mem4 PROC FRAME
    .endprolog
    vbroadcastss ymm0, DWORD PTR [rcx + 16]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_ymm_mem4 ENDP

op_broadcastss_ymm_mem5 PROC FRAME
    .endprolog
    vbroadcastss ymm0, DWORD PTR [rcx + 20]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_ymm_mem5 ENDP

op_broadcastss_ymm_mem6 PROC FRAME
    .endprolog
    vbroadcastss ymm0, DWORD PTR [rcx + 24]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_ymm_mem6 ENDP

op_broadcastss_ymm_mem7 PROC FRAME
    .endprolog
    vbroadcastss ymm0, DWORD PTR [rcx + 28]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_ymm_mem7 ENDP

op_broadcastss_xmm_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vbroadcastss xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_xmm_reg ENDP

op_broadcastss_ymm_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vbroadcastss ymm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_broadcastss_ymm_reg ENDP

; --- Lane ops (4 forms: vextractf128, vinsertf128, vextractps, vinsertps) ---

op_extractf128_lo PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vextractf128 xmm0, ymm1, 0FEh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_extractf128_lo ENDP

op_extractf128_hi PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vextractf128 xmm0, ymm1, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_extractf128_hi ENDP

op_insertf128_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vinsertf128 ymm0, ymm1, xmm2, 0FEh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_insertf128_lo ENDP

op_insertf128_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vinsertf128 ymm0, ymm1, xmm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_insertf128_hi ENDP

op_extractps_lane0 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vextractps eax, xmm1, 0FCh
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_extractps_lane0 ENDP

op_extractps_lane1 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vextractps eax, xmm1, 0FDh
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_extractps_lane1 ENDP

op_extractps_lane2 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vextractps eax, xmm1, 0FEh
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_extractps_lane2 ENDP

op_extractps_lane3 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vextractps eax, xmm1, 0FFh
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_extractps_lane3 ENDP

op_insertps_s0d1_z0 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vinsertps xmm0, xmm1, xmm2, 10h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_insertps_s0d1_z0 ENDP

op_insertps_s3d2_zmask PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vinsertps xmm0, xmm1, xmm2, 0EDh
ELSE
    vinsertps xmm0, xmm1, xmm2, 0E5h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_insertps_s3d2_zmask ENDP

op_insertps_zero_all PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vinsertps xmm0, xmm1, xmm2, 0BFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_insertps_zero_all ENDP

; --- Permutes (3 forms: vperm2f128, vpermilpd imm/var, vpermilps imm/var) ---

op_perm2f128_mix PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vperm2f128 ymm0, ymm1, ymm2, 31h
ELSE
    vperm2f128 ymm0, ymm1, ymm2, 20h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_perm2f128_mix ENDP

op_perm2f128_swap PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vperm2f128 ymm0, ymm1, ymm2, 03h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_perm2f128_swap ENDP

op_perm2f128_zero_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vperm2f128 ymm0, ymm1, ymm2, 28h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_perm2f128_zero_lo ENDP

op_perm2f128_zero_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vperm2f128 ymm0, ymm1, ymm2, 81h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_perm2f128_zero_hi ENDP

op_perm2f128_zero_both PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vperm2f128 ymm0, ymm1, ymm2, 88h
    ; 0CBh also zeroes both lanes, so the OR stays zero only if zeroing wins over lane selection.
    vperm2f128 ymm3, ymm1, ymm2, 0CBh
    vorps ymm0, ymm0, ymm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_perm2f128_zero_both ENDP

op_perm2f128_ignored PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vperm2f128 ymm0, ymm1, ymm2, 45h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_perm2f128_ignored ENDP

op_permilpd_imm_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpermilpd xmm0, xmm1, 0FDh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilpd_imm_xmm ENDP

op_permilpd_imm_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vpermilpd ymm0, ymm1, 0F5h
ELSE
    vpermilpd ymm0, ymm1, 0FAh
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilpd_imm_ymm ENDP

op_permilpd_imm_ymm_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermilpd ymm0, ymm1, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilpd_imm_ymm_lo ENDP

op_permilpd_imm_ymm_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermilpd ymm0, ymm1, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilpd_imm_ymm_hi ENDP

op_permilpd_var_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpermilpd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilpd_var_xmm ENDP

op_permilpd_var_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpermilpd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilpd_var_ymm ENDP

op_permilps_imm_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpermilps xmm0, xmm1, 1Bh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilps_imm_xmm ENDP

op_permilps_imm_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vpermilps ymm0, ymm1, 4Eh
ELSE
    vpermilps ymm0, ymm1, 0B1h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilps_imm_ymm ENDP

op_permilps_imm_ymm_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermilps ymm0, ymm1, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilps_imm_ymm_lo ENDP

op_permilps_imm_ymm_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpermilps ymm0, ymm1, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilps_imm_ymm_hi ENDP

op_permilps_var_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpermilps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilps_var_xmm ENDP

op_permilps_var_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vpermilps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_permilps_var_ymm ENDP

; --- Shuffles (2 forms: vshufpd, vshufps) ---

op_shufpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vshufpd xmm0, xmm1, xmm2, 0FEh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufpd_xmm ENDP

op_shufpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vshufpd ymm0, ymm1, ymm2, 0F5h
ELSE
    vshufpd ymm0, ymm1, ymm2, 0FAh
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufpd_ymm ENDP

op_shufpd_ymm_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vshufpd ymm0, ymm1, ymm2, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufpd_ymm_lo ENDP

op_shufpd_ymm_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vshufpd ymm0, ymm1, ymm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufpd_ymm_hi ENDP

op_shufps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vshufps xmm0, xmm1, xmm2, 0E4h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufps_xmm ENDP

op_shufps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vshufps ymm0, ymm1, ymm2, 1Bh
ELSE
    vshufps ymm0, ymm1, ymm2, 0B1h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufps_ymm ENDP

op_shufps_ymm_lo PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vshufps ymm0, ymm1, ymm2, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufps_ymm_lo ENDP

op_shufps_ymm_hi PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vshufps ymm0, ymm1, ymm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufps_ymm_hi ENDP

; --- Unpacks (4 forms: vunpcklpd, vunpckhpd, vunpcklps, vunpckhps) ---

op_unpcklpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vunpcklpd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpcklpd_xmm ENDP

op_unpcklpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vunpcklpd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpcklpd_ymm ENDP

op_unpckhpd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vunpckhpd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpckhpd_xmm ENDP

op_unpckhpd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vunpckhpd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpckhpd_ymm ENDP

op_unpcklps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vunpcklps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpcklps_xmm ENDP

op_unpcklps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vunpcklps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpcklps_ymm ENDP

op_unpckhps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vunpckhps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpckhps_xmm ENDP

op_unpckhps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vunpckhps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_unpckhps_ymm ENDP

; --- Full-width moves: aligned (vmovapd) and unaligned (vmovupd, vlddqu) ---

op_movapd_xmm PROC FRAME
    .endprolog
    vmovapd ymm0, YMMWORD PTR [rdx]
    vmovapd xmm0, XMMWORD PTR [rcx]
    vmovapd xmm1, xmm0
    vmovapd YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_movapd_xmm ENDP

op_movapd_ymm PROC FRAME
    .endprolog
    vmovapd ymm0, YMMWORD PTR [rcx]
    vmovapd ymm1, ymm0
    vmovapd YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_movapd_ymm ENDP

op_movapd_store128 PROC FRAME
    .endprolog
    vmovapd xmm0, XMMWORD PTR [rcx]
    vmovapd XMMWORD PTR [r8], xmm0
    vzeroupper
    ret
op_movapd_store128 ENDP

op_movupd_xmm PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [rdx]
    vmovupd xmm0, XMMWORD PTR [rcx]
    vmovupd xmm1, xmm0
    vmovupd YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_movupd_xmm ENDP

op_movupd_ymm PROC FRAME
    .endprolog
    vmovupd ymm0, YMMWORD PTR [rcx]
    vmovupd ymm1, ymm0
    vmovupd YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_movupd_ymm ENDP

op_movupd_store128 PROC FRAME
    .endprolog
    vmovupd xmm0, XMMWORD PTR [rcx]
    vmovupd XMMWORD PTR [r8], xmm0
    vzeroupper
    ret
op_movupd_store128 ENDP

op_lddqu_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vlddqu xmm0, XMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_lddqu_xmm ENDP

op_lddqu_ymm PROC FRAME
    .endprolog
    vlddqu ymm0, YMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_lddqu_ymm ENDP

; --- Non-temporal loads and stores (aligned 16/32-byte buffers) ---

op_movntdq_xmm PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovntdq XMMWORD PTR [r8], xmm0
    vzeroupper
    ret
op_movntdq_xmm ENDP

op_movntdq_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovntdq YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movntdq_ymm ENDP

op_movntdqa_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovntdqa xmm0, XMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movntdqa_xmm ENDP

op_movntdqa_ymm PROC FRAME
    .endprolog
    vmovntdqa ymm0, YMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movntdqa_ymm ENDP

op_movntpd_xmm PROC FRAME
    .endprolog
    vmovapd xmm0, XMMWORD PTR [rcx]
    vmovntpd XMMWORD PTR [r8], xmm0
    vzeroupper
    ret
op_movntpd_xmm ENDP

op_movntpd_ymm PROC FRAME
    .endprolog
    vmovapd ymm0, YMMWORD PTR [rcx]
    vmovntpd YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movntpd_ymm ENDP

op_movntps_xmm PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovntps XMMWORD PTR [r8], xmm0
    vzeroupper
    ret
op_movntps_xmm ENDP

op_movntps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovntps YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movntps_ymm ENDP

; --- Dup moves (3 forms: vmovddup, vmovshdup, vmovsldup) ---

op_movddup_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovddup xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movddup_xmm ENDP

op_movddup_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovddup ymm0, ymm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movddup_ymm ENDP

op_movshdup_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovshdup xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movshdup_xmm ENDP

op_movshdup_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovshdup ymm0, ymm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movshdup_ymm ENDP

op_movsldup_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovsldup xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movsldup_xmm ENDP

op_movsldup_ymm PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovsldup ymm0, ymm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movsldup_ymm ENDP

; --- Half-XMM register and memory moves (6 forms) ---

op_movhlps PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovhlps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movhlps ENDP

op_movlhps PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovlhps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movlhps ENDP

op_movhpd_load PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovhpd xmm0, xmm1, QWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movhpd_load ENDP

op_movhpd_store PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovhpd QWORD PTR [r8], xmm1
    vzeroupper
    ret
op_movhpd_store ENDP

op_movhps_load PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovhps xmm0, xmm1, QWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movhps_load ENDP

op_movhps_store PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovhps QWORD PTR [r8], xmm1
    vzeroupper
    ret
op_movhps_store ENDP

op_movlpd_load PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovlpd xmm0, xmm1, QWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movlpd_load ENDP

op_movlpd_store PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovlpd QWORD PTR [r8], xmm1
    vzeroupper
    ret
op_movlpd_store ENDP

op_movlps_load PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovlps xmm0, xmm1, QWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movlps_load ENDP

op_movlps_store PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovlps QWORD PTR [r8], xmm1
    vzeroupper
    ret
op_movlps_store ENDP

; --- Scalar moves (2 surviving forms: vmovsd, vmovss) ---

op_movsd_load PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovsd xmm0, QWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movsd_load ENDP

op_movsd_store PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovsd QWORD PTR [r8], xmm1
    vzeroupper
    ret
op_movsd_store ENDP

op_movsd_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovsd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movsd_reg ENDP

op_movss_load PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovss xmm0, DWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movss_load ENDP

op_movss_store PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovss DWORD PTR [r8], xmm1
    vzeroupper
    ret
op_movss_store ENDP

op_movss_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movss_reg ENDP

; --- Sign-bit mask to GPR (2 forms: vmovmskpd, vmovmskps) ---

op_movmskpd_xmm PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovmskpd eax, xmm1
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_movmskpd_xmm ENDP

op_movmskpd_ymm PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovmskpd eax, ymm1
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_movmskpd_ymm ENDP

op_movmskps_xmm PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovmskps eax, xmm1
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_movmskps_xmm ENDP

op_movmskps_ymm PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovmskps eax, ymm1
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_movmskps_ymm ENDP

; --- Masked loads and stores (2 forms: vmaskmovps, vmaskmovpd) ---
; Load ABI:  RCX = src_mem, RDX = [0..31] pre-seed ymm0 + [32..63] mask ymm1, R8 = [0..31] out ymm0 + [32..63] mask ymm1
; Store ABI: RCX = src_vec (32B), RDX = mask_vec (32B), R8 = dst_mem (pre-seeded)

op_maskmovps_load_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rdx + 32]
    vmaskmovps xmm0, xmm1, XMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu YMMWORD PTR [r8 + 32], ymm1
    vzeroupper
    ret
op_maskmovps_load_xmm ENDP

op_maskmovps_load_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rdx + 32]
IFDEF ALTER_IMM
    vmaskmovps ymm0, ymm1, YMMWORD PTR [rcx + 16]
ELSE
    vmaskmovps ymm0, ymm1, YMMWORD PTR [rcx]
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu YMMWORD PTR [r8 + 32], ymm1
    vzeroupper
    ret
op_maskmovps_load_ymm ENDP

op_maskmovpd_load_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rdx + 32]
    vmaskmovpd xmm0, xmm1, XMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu YMMWORD PTR [r8 + 32], ymm1
    vzeroupper
    ret
op_maskmovpd_load_xmm ENDP

op_maskmovpd_load_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rdx + 32]
    vmaskmovpd ymm0, ymm1, YMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu YMMWORD PTR [r8 + 32], ymm1
    vzeroupper
    ret
op_maskmovpd_load_ymm ENDP

op_maskmovps_store_xmm PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vmaskmovps XMMWORD PTR [r8], xmm1, xmm2
    vzeroupper
    ret
op_maskmovps_store_xmm ENDP

op_maskmovps_store_ymm PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vmaskmovps YMMWORD PTR [r8], ymm1, ymm2
    vzeroupper
    ret
op_maskmovps_store_ymm ENDP

op_maskmovpd_store_xmm PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vmaskmovpd XMMWORD PTR [r8], xmm1, xmm2
    vzeroupper
    ret
op_maskmovpd_store_xmm ENDP

op_maskmovpd_store_ymm PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vmaskmovpd YMMWORD PTR [r8], ymm1, ymm2
    vzeroupper
    ret
op_maskmovpd_store_ymm ENDP

; --- VZEROUPPER and multi-instruction pipeline ---

op_zeroupper PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vzeroupper
    vmovdqu YMMWORD PTR [r8], ymm0
    ret
op_zeroupper ENDP

; Chained multi-register sequence exercising mixed widths, register aliasing and GPR mask transfer.
op_pipeline PROC FRAME
    .endprolog
    vbroadcastf128 ymm0, XMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vxorps ymm2, ymm0, ymm1
    vandnps ymm3, ymm1, ymm2
    vorpd ymm4, ymm3, ymm0
    vperm2f128 ymm5, ymm4, ymm1, 21h
    vshufps ymm5, ymm5, ymm2, 0D8h
    vblendvps ymm4, ymm5, ymm3, ymm1
    vpermilps ymm4, ymm4, ymm0
    vunpcklps ymm3, ymm4, ymm5
    vmovshdup ymm2, ymm3
    vextractf128 xmm0, ymm2, 1
    vmovmskps eax, ymm2
    mov DWORD PTR [r8], eax
    vmovss xmm1, DWORD PTR [r8]
    vinsertps xmm0, xmm0, xmm1, 30h
    vinsertf128 ymm0, ymm2, xmm0, 0
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pipeline ENDP

op_mov_load32 PROC FRAME
    .endprolog
    mov eax, DWORD PTR [rdx]
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_mov_load32 ENDP

; --- Mutation-sensitivity exports ---

; vzeroupper must clear bits 128..255 of every live YMM register, not only ymm0.
op_zeroupper_ymm1 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vzeroupper
    vmovdqu YMMWORD PTR [r8], ymm1
    ret
op_zeroupper_ymm1 ENDP

; XMM register copy from a source whose upper 128 bits are non-zero.
op_movapd_xmm_dirty PROC FRAME
    .endprolog
    vmovapd ymm0, YMMWORD PTR [rcx]
    vmovapd xmm1, xmm0
    vmovapd YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_movapd_xmm_dirty ENDP

; 32-bit store of a GPR holding a full 64-bit value; bytes 4..31 of the output keep their fill.
op_mov_store32 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    mov DWORD PTR [r8], eax
    vzeroupper
    ret
op_mov_store32 ENDP

; Source lane 2 (imm8[7:6] = 10b), destination lane 3, zero mask lane 1.
op_insertps_s2d3_z1 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vinsertps xmm0, xmm1, xmm2, 0B2h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_insertps_s2d3_z1 ENDP

; VEX.W1 vextractps rax, xmm1, 0FEh (ML64 only emits the W0 r/m32 form).
op_extractps_w1_lane2 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    DB 0C4h, 0E3h, 0F9h, 17h, 0C8h, 0FEh
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_extractps_w1_lane2 ENDP

; XMM vshufpd with imm8 bit 0 set (q0 = src1.q1).
op_shufpd_xmm_sel1 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vshufpd xmm0, xmm1, xmm2, 01h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufpd_xmm_sel1 ENDP

; YMM vshufps with selector 0 = 2 (distinct selectors 2, 3, 0, 1).
op_shufps_ymm_sel2 PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vmovdqu ymm2, YMMWORD PTR [rdx]
    vshufps ymm0, ymm1, ymm2, 4Eh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_shufps_ymm_sel2 ENDP

; Negative displacements. RCX = input_a + 32, RDX = input_b + 32, R8 = output (32B).
op_negdisp_load PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx - 32]
    mov rax, QWORD PTR [rdx - 8]
    vmovdqu YMMWORD PTR [r8], ymm0
    mov QWORD PTR [r8 + 8], rax
    vzeroupper
    ret
op_negdisp_load ENDP

END
