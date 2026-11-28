; coverage: source cpu_aot_avx_int_shifts_pack
; Source-owned differential fixture for 34 VEX AVX packed shifts, packs, unpacks, shuffles, blends, movemask, and insert/extract instructions.
.code

vpsllw_imm_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsllw xmm0, xmm1, 5
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsllw_imm_op ENDP

vpslld_imm_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpslld xmm0, xmm1, 11
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpslld_imm_op ENDP

vpsllq_imm_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsllq xmm0, xmm1, 19
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsllq_imm_op ENDP

vpsrlw_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrlw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsrlw_reg_op ENDP

vpsrlq_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrlq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsrlq_reg_op ENDP

vpsraw_imm_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsraw xmm0, xmm1, 5
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsraw_imm_op ENDP

vpsrad_reg_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsrad xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsrad_reg_op ENDP

vpslldq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpslldq xmm0, xmm1, 3
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpslldq_op ENDP

vpacksswb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpacksswb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpacksswb_op ENDP

vpackssdw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpackssdw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpackssdw_op ENDP

vpackuswb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpackuswb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpackuswb_op ENDP

vpackusdw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpackusdw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpackusdw_op ENDP

vpunpcklbw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpcklbw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpcklbw_op ENDP

vpunpcklwd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpcklwd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpcklwd_op ENDP

vpunpckldq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpckldq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpckldq_op ENDP

vpunpcklqdq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpcklqdq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpcklqdq_op ENDP

vpunpckhbw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpckhbw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpckhbw_op ENDP

vpunpckhwd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpckhwd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpckhwd_op ENDP

vpunpckhdq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpckhdq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpckhdq_op ENDP

vpunpckhqdq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpunpckhqdq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpunpckhqdq_op ENDP

vpshufd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpshufd xmm0, xmm1, 0B1h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshufd_op ENDP

vpshuflw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpshuflw xmm0, xmm1, 01Bh
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshuflw_op ENDP

vpshufhw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpshufhw xmm0, xmm1, 04Eh
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshufhw_op ENDP

vpblendw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpblendw xmm0, xmm1, xmm2, 0A5h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpblendw_op ENDP

vpblendvb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vmovups ymm0, ymmword ptr [r9]
    vpblendvb xmm0, xmm1, xmm2, xmm0
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpblendvb_op ENDP

vpinsrb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    mov eax, dword ptr [rdx]
    vpinsrb xmm0, xmm1, eax, 9
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpinsrb_op ENDP

vpinsrw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    mov eax, dword ptr [rdx]
    vpinsrw xmm0, xmm1, eax, 6
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpinsrw_op ENDP

vpinsrd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    mov eax, dword ptr [rdx]
    vpinsrd xmm0, xmm1, eax, 3
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpinsrd_op ENDP

vpinsrq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    mov rax, qword ptr [rdx]
    vpinsrq xmm0, xmm1, rax, 1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpinsrq_op ENDP

vpmovmskb_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpmovmskb eax, xmm0
    vzeroupper
    ret
vpmovmskb_op ENDP

vpextrb_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpextrb eax, xmm0, 11
    vzeroupper
    ret
vpextrb_op ENDP

vpextrw_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpextrw eax, xmm0, 5
    vzeroupper
    ret
vpextrw_op ENDP

vpextrd_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpextrd eax, xmm0, 2
    vzeroupper
    ret
vpextrd_op ENDP

vpextrq_op PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    vpextrq rax, xmm0, 1
    vzeroupper
    ret
vpextrq_op ENDP

vpsrlw_mem_count_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpsrlw xmm0, xmm1, xmmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsrlw_mem_count_op ENDP

vpshufd_ymm_mem_op PROC FRAME
    .endprolog
    vpshufd ymm0, ymmword ptr [rcx], 0B1h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshufd_ymm_mem_op ENDP

END
