; coverage: source cpu_aot_avx_shuffle_blend
; Source-owned differential fixture for 26 AVX broadcast, extract/insert, permute, shuffle, blend, dot, round, and zeroall instructions.
.code

vbroadcastss_op PROC FRAME
    .endprolog
    vbroadcastss ymm0, dword ptr [rcx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vbroadcastss_op ENDP

vbroadcastsd_op PROC FRAME
    .endprolog
    vbroadcastsd ymm0, qword ptr [rcx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vbroadcastsd_op ENDP

vbroadcastf128_op PROC FRAME
    .endprolog
    vbroadcastf128 ymm0, xmmword ptr [rcx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vbroadcastf128_op ENDP

vextractf128_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vextractf128 xmm0, ymm1, 1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vextractf128_op ENDP

vinsertf128_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vinsertf128 ymm0, ymm1, xmmword ptr [rdx], 1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vinsertf128_op ENDP

vperm2f128_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vperm2f128 ymm0, ymm1, ymm2, 31h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vperm2f128_op ENDP

vpermilps_imm_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpermilps ymm0, ymm1, 0B1h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpermilps_imm_op ENDP

vpermilpd_imm_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpermilpd ymm0, ymm1, 09h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpermilpd_imm_op ENDP

vshufps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vshufps ymm0, ymm1, ymm2, 0D2h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vshufps_op ENDP

vshufpd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vshufpd ymm0, ymm1, ymm2, 05h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vshufpd_op ENDP

vunpcklps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vunpcklps ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpcklps_op ENDP

vunpckhps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vunpckhps ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpckhps_op ENDP

vunpcklpd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vunpcklpd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpcklpd_op ENDP

vunpckhpd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vunpckhpd ymm0, ymm1, ymm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vunpckhpd_op ENDP

vmovddup_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovddup ymm0, ymm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovddup_op ENDP

; The XMM memory encoding reads exactly eight bytes and clears upper YMM.
vmovddup_m64_op PROC FRAME
    .endprolog
    vmovddup xmm0, qword ptr [rcx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovddup_m64_op ENDP

vmovsldup_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovsldup ymm0, ymm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovsldup_op ENDP

vmovshdup_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovshdup ymm0, ymm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vmovshdup_op ENDP

vblendps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vblendps ymm0, ymm1, ymm2, 0A5h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vblendps_op ENDP

vblendpd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vblendpd ymm0, ymm1, ymm2, 0Ah
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vblendpd_op ENDP

vblendvps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmovups ymm3, ymmword ptr [r9]
    vblendvps ymm0, ymm1, ymm2, ymm3
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vblendvps_op ENDP

vblendvpd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmovups ymm3, ymmword ptr [r9]
    vblendvpd ymm0, ymm1, ymm2, ymm3
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vblendvpd_op ENDP

vdpps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vdpps ymm0, ymm1, ymm2, 0D5h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vdpps_op ENDP

vdppd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vdppd xmm0, xmm1, xmm2, 031h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vdppd_op ENDP

vroundps_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vroundps ymm0, ymm1, 08h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vroundps_op ENDP

vroundpd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vroundpd ymm0, ymm1, 09h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vroundpd_op ENDP

vzeroall_lift_entry PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vzeroall
    vmovups ymmword ptr [r8], ymm0
    ret
vzeroall_lift_entry ENDP

; Native ABI wrapper. The raw lift entry clobbers the nonvolatile XMM halves.
vzeroall_op PROC FRAME
    sub rsp, 0C8h
    .allocstack 0C8h
    movdqa xmmword ptr [rsp+020h], xmm6
    .savexmm128 xmm6, 020h
    movdqa xmmword ptr [rsp+030h], xmm7
    .savexmm128 xmm7, 030h
    movdqa xmmword ptr [rsp+040h], xmm8
    .savexmm128 xmm8, 040h
    movdqa xmmword ptr [rsp+050h], xmm9
    .savexmm128 xmm9, 050h
    movdqa xmmword ptr [rsp+060h], xmm10
    .savexmm128 xmm10, 060h
    movdqa xmmword ptr [rsp+070h], xmm11
    .savexmm128 xmm11, 070h
    movdqa xmmword ptr [rsp+080h], xmm12
    .savexmm128 xmm12, 080h
    movdqa xmmword ptr [rsp+090h], xmm13
    .savexmm128 xmm13, 090h
    movdqa xmmword ptr [rsp+0A0h], xmm14
    .savexmm128 xmm14, 0A0h
    movdqa xmmword ptr [rsp+0B0h], xmm15
    .savexmm128 xmm15, 0B0h
    .endprolog
    call vzeroall_lift_entry
    movdqa xmm6, xmmword ptr [rsp+020h]
    movdqa xmm7, xmmword ptr [rsp+030h]
    movdqa xmm8, xmmword ptr [rsp+040h]
    movdqa xmm9, xmmword ptr [rsp+050h]
    movdqa xmm10, xmmword ptr [rsp+060h]
    movdqa xmm11, xmmword ptr [rsp+070h]
    movdqa xmm12, xmmword ptr [rsp+080h]
    movdqa xmm13, xmmword ptr [rsp+090h]
    movdqa xmm14, xmmword ptr [rsp+0A0h]
    movdqa xmm15, xmmword ptr [rsp+0B0h]
    add rsp, 0C8h
    ret
vzeroall_op ENDP

; Check the wrapper against a live caller sentinel and restore the caller's value.
vzeroall_abi PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    movdqa xmmword ptr [rsp+20h], xmm6
    .savexmm128 xmm6, 20h
    .endprolog
    pcmpeqd xmm6, xmm6
    call vzeroall_op
    pmovmskb eax, xmm6
    cmp eax, 0FFFFh
    sete al
    movzx eax, al
    movdqa xmm6, xmmword ptr [rsp+20h]
    add rsp, 38h
    ret
vzeroall_abi ENDP

END
