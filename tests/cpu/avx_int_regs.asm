; Source-built MASM fixture exercising 90 surviving AVX packed-integer forms
; (including vzeroall and vzeroupper, excluding the 28 forms carved out to
; cpu_vex_aot.py (PR #325)) from docs/coverage/x64_zen2.tsv group AVX.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes)
;   RDX = const void* input_b (32 bytes; also used to pre-seed YMM0 on XMM writes)
;   R8  = void*       output  (32 bytes; 192 bytes for op_zeroall)
.code

; --- Packed absolute value ---

op_pabsb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpabsb xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pabsb ENDP

op_pabsw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpabsw xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pabsw ENDP

op_pabsd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpabsd xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pabsd ENDP

; --- Pack with signed/unsigned saturation ---

op_packsswb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpacksswb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_packsswb ENDP

op_packssdw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpackssdw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_packssdw ENDP

op_packusdw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpackusdw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_packusdw ENDP

; --- Packed add (wrapping and saturating) ---

op_paddb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpaddb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_paddb ENDP

op_paddq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpaddq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_paddq ENDP

op_paddsb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpaddsb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_paddsb ENDP

op_paddusb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpaddusb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_paddusb ENDP

op_paddusw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpaddusw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_paddusw ENDP

; --- Packed subtract (wrapping and saturating) ---

op_psubb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsubb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psubb ENDP

op_psubd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsubd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psubd ENDP

op_psubq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsubq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psubq ENDP

op_psubsb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsubsb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psubsb ENDP

op_psubsw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsubsw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psubsw ENDP

op_psubusb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsubusb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psubusb ENDP

op_psubusw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsubusw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psubusw ENDP

; --- Align and blend ---

op_palignr_0 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpalignr xmm0, xmm1, xmm2, 0
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_palignr_0 ENDP

op_palignr_5 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vpalignr xmm0, xmm1, xmm2, 11
ELSE
    vpalignr xmm0, xmm1, xmm2, 5
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_palignr_5 ENDP

op_palignr_16 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpalignr xmm0, xmm1, xmm2, 16
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_palignr_16 ENDP

op_palignr_23 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpalignr xmm0, xmm1, xmm2, 23
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_palignr_23 ENDP

op_palignr_32 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpalignr xmm0, xmm1, xmm2, 32
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_palignr_32 ENDP

op_palignr_69 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpalignr xmm0, xmm1, xmm2, 69
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_palignr_69 ENDP

op_palignr_255 PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpalignr xmm0, xmm1, xmm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_palignr_255 ENDP

op_pblendvb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmovdqu xmm3, XMMWORD PTR [rcx + 16]
    vpblendvb xmm0, xmm1, xmm2, xmm3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pblendvb ENDP

op_pblendw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vpblendw xmm0, xmm1, xmm2, 55h
ELSE
    vpblendw xmm0, xmm1, xmm2, 0AAh
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pblendw ENDP

op_pblendw_all PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpblendw xmm0, xmm1, xmm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pblendw_all ENDP

op_pblendw_none PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpblendw xmm0, xmm1, xmm2, 00h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pblendw_none ENDP

; --- Packed compare ---

op_pcmpeqb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpeqb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpeqb ENDP

op_pcmpeqw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpeqw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpeqw ENDP

op_pcmpeqd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpeqd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpeqd ENDP

op_pcmpeqq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpeqq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpeqq ENDP

op_pcmpgtb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpgtb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpgtb ENDP

op_pcmpgtw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpgtw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpgtw ENDP

op_pcmpgtd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpgtd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpgtd ENDP

op_pcmpgtq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpcmpgtq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pcmpgtq ENDP

; --- Packed min / max ---

op_pmaxsb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmaxsb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmaxsb ENDP

op_pmaxsd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmaxsd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmaxsd ENDP

op_pmaxub PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmaxub xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmaxub ENDP

op_pmaxuw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmaxuw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmaxuw ENDP

op_pmaxud PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmaxud xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmaxud ENDP

op_pminsb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpminsb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pminsb ENDP

op_pminsd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpminsd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pminsd ENDP

op_pminub PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpminub xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pminub ENDP

op_pminuw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpminuw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pminuw ENDP

op_pminud PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpminud xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pminud ENDP

; --- Horizontal add / sub and minpos ---

op_phaddw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vphaddw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_phaddw ENDP

op_phaddd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vphaddd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_phaddd ENDP

op_phaddsw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vphaddsw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_phaddsw ENDP

op_phsubw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vphsubw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_phsubw ENDP

op_phsubd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vphsubd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_phsubd ENDP

op_phsubsw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vphsubsw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_phsubsw ENDP

op_phminposuw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vphminposuw xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_phminposuw ENDP

; --- Multiply and multiply-add ---

op_pmaddwd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmaddwd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmaddwd ENDP

op_pmuldq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmuldq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmuldq ENDP

op_pmuludq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmuludq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmuludq ENDP

op_pmulhrsw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmulhrsw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmulhrsw ENDP

op_pmulhuw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmulhuw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmulhuw ENDP

op_pmulld PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpmulld xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmulld ENDP

; --- Sum of absolute differences ---

op_psadbw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsadbw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psadbw ENDP

op_mpsadbw_0 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    vmpsadbw xmm0, xmm1, xmm2, 05h
ELSE
    vmpsadbw xmm0, xmm1, xmm2, 00h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mpsadbw_0 ENDP

op_mpsadbw_3 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmpsadbw xmm0, xmm1, xmm2, 03h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mpsadbw_3 ENDP

op_mpsadbw_6 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmpsadbw xmm0, xmm1, xmm2, 06h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mpsadbw_6 ENDP

op_mpsadbw_7_ignored PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vmpsadbw xmm0, xmm1, xmm2, 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mpsadbw_7_ignored ENDP

; --- Shuffles ---

op_pshufhw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vpshufhw xmm0, xmm1, 1Bh
ELSE
    vpshufhw xmm0, xmm1, 93h
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pshufhw ENDP

op_pshuflw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpshuflw xmm0, xmm1, 4Eh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pshuflw ENDP

op_pshuflw_1b PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpshuflw xmm0, xmm1, 1Bh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pshuflw_1b ENDP

; --- Packed sign ---

op_psignb PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsignb xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psignb ENDP

op_psignw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsignw xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psignw ENDP

op_psignd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsignd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psignd ENDP

; --- Sign-extend and zero-extend moves ---

op_pmovsxbw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovsxbw xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovsxbw ENDP

op_pmovsxbd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovsxbd xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovsxbd ENDP

op_pmovsxbq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovsxbq xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovsxbq ENDP

op_pmovsxbq_wide_src PROC FRAME
    .endprolog
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vpmovsxbq xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovsxbq_wide_src ENDP

op_pmovsxwd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovsxwd xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovsxwd ENDP

op_pmovsxwq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovsxwq xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovsxwq ENDP

op_pmovsxdq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovsxdq xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovsxdq ENDP

op_pmovzxbw PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovzxbw xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovzxbw ENDP

op_pmovzxbd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovzxbd xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovzxbd ENDP

op_pmovzxbq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovzxbq xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovzxbq ENDP

op_pmovzxwd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovzxwd xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovzxwd ENDP

op_pmovzxwq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovzxwq xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovzxwq ENDP

op_pmovzxdq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovzxdq xmm0, xmm1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pmovzxdq ENDP

; --- Unpack / interleave ---

op_punpckldq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpunpckldq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_punpckldq ENDP

op_punpckhdq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpunpckhdq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_punpckhdq ENDP

op_punpcklqdq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpunpcklqdq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_punpcklqdq ENDP

op_punpckhqdq PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpunpckhqdq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_punpckhqdq ENDP

; --- Packed shifts: XMM-count encodings ---

op_pslld_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpslld xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pslld_xmm ENDP

op_psllq_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsllq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllq_xmm ENDP

op_psrld_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsrld xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrld_xmm ENDP

op_psrlq_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpsrlq xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrlq_xmm ENDP

; --- Packed shifts: imm8 encodings (in-range and >= lane width) ---

op_pslld_imm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpslld xmm0, xmm1, 7
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pslld_imm ENDP

op_pslld_imm_ovf PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpslld xmm0, xmm1, 32
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pslld_imm_ovf ENDP

op_psllq_imm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsllq xmm0, xmm1, 13
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllq_imm ENDP

op_psllq_imm_ovf PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsllq xmm0, xmm1, 64
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllq_imm_ovf ENDP

op_psllq_imm_129 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsllq xmm0, xmm1, 81h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psllq_imm_129 ENDP

op_psrld_imm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsrld xmm0, xmm1, 7
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrld_imm ENDP

op_psrld_imm_ovf PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsrld xmm0, xmm1, 32
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrld_imm_ovf ENDP

op_psrlq_imm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsrlq xmm0, xmm1, 13
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrlq_imm ENDP

op_psrlq_imm_ovf PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpsrlq xmm0, xmm1, 64
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_psrlq_imm_ovf ENDP

; --- Movemask, extract (reg & mem), insert (reg & mem) ---

op_pmovmskb PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpmovmskb eax, xmm1
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_pmovmskb ENDP

op_pextrb_reg PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
IFDEF ALTER_IMM
    vpextrb eax, xmm1, 3
ELSE
    vpextrb eax, xmm1, 13
ENDIF
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_pextrb_reg ENDP

op_pextrb_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrb BYTE PTR [r8 + 3], xmm1, 0F7h
    vzeroupper
    ret
op_pextrb_mem ENDP

op_pextrw_reg PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrw eax, xmm1, 5
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_pextrw_reg ENDP

op_pextrw_reg_2 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrw eax, xmm1, 2
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_pextrw_reg_2 ENDP

op_pextrw_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrw WORD PTR [r8 + 4], xmm1, 0FEh
    vzeroupper
    ret
op_pextrw_mem ENDP

op_pextrd_reg PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrd eax, xmm1, 3
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_pextrd_reg ENDP

op_pextrd_reg_2 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrd eax, xmm1, 2
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_pextrd_reg_2 ENDP

op_pextrd_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrd DWORD PTR [r8 + 8], xmm1, 0FDh
    vzeroupper
    ret
op_pextrd_mem ENDP

op_pextrq_reg PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrq rax, xmm1, 1
    mov QWORD PTR [r8], rax
    vzeroupper
    ret
op_pextrq_reg ENDP

op_pextrq_mem PROC FRAME
    .endprolog
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpextrq QWORD PTR [r8 + 8], xmm1, 0FEh
    vzeroupper
    ret
op_pextrq_mem ENDP

op_pinsrb_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    mov eax, DWORD PTR [rdx]
    vpinsrb xmm0, xmm1, eax, 11
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrb_reg ENDP

; src1 is loaded as a full YMM so its bits 255:128 are non-zero; VEX.128 vpinsrb must still zero them in the destination.
op_pinsrb_reg_upper PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    mov eax, DWORD PTR [rdx]
    vpinsrb xmm0, xmm1, eax, 4
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrb_reg_upper ENDP

op_pinsrb_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpinsrb xmm0, xmm1, BYTE PTR [rdx + 3], 0F5h
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrb_mem ENDP

op_pinsrw_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    mov eax, DWORD PTR [rdx]
    vpinsrw xmm0, xmm1, eax, 6
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrw_reg ENDP

op_pinsrw_reg_3 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    mov eax, DWORD PTR [rdx]
    vpinsrw xmm0, xmm1, eax, 3
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrw_reg_3 ENDP

op_pinsrw_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpinsrw xmm0, xmm1, WORD PTR [rdx + 4], 0FAh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrw_mem ENDP

op_pinsrd_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    mov eax, DWORD PTR [rdx]
    vpinsrd xmm0, xmm1, eax, 2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrd_reg ENDP

op_pinsrd_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpinsrd xmm0, xmm1, DWORD PTR [rdx + 8], 0FFh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrd_mem ENDP

op_pinsrq_reg PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    mov rax, QWORD PTR [rdx]
    vpinsrq xmm0, xmm1, rax, 1
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrq_reg ENDP

op_pinsrq_mem PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vpinsrq xmm0, xmm1, QWORD PTR [rdx + 8], 0FEh
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pinsrq_mem ENDP

; --- State management (vzeroupper, vzeroall) and multi-op pipeline ---

op_zeroupper PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vzeroupper
    vmovdqu YMMWORD PTR [r8], ymm0
    ret
op_zeroupper ENDP

; vzeroupper must clear bits 255:128 of every YMM, not only ymm0; observe ymm1.
op_zeroupper_ymm1 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu ymm1, YMMWORD PTR [rcx]
    vzeroupper
    vmovdqu YMMWORD PTR [r8], ymm1
    ret
op_zeroupper_ymm1 ENDP

op_zeroupper_ymm2 PROC FRAME
    .endprolog
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vzeroupper
    vmovdqu YMMWORD PTR [r8], ymm2
    ret
op_zeroupper_ymm2 ENDP

; Seeds all six volatile YMM registers, executes vzeroall, and stores all 192 bytes.
op_zeroall PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vmovdqu ymm2, YMMWORD PTR [rcx]
    vmovdqu ymm3, YMMWORD PTR [rdx]
    vmovdqu ymm4, YMMWORD PTR [rcx]
    vmovdqu ymm5, YMMWORD PTR [rdx]
    vzeroall
    vmovdqu YMMWORD PTR [r8], ymm0
    vmovdqu YMMWORD PTR [r8 + 32], ymm1
    vmovdqu YMMWORD PTR [r8 + 64], ymm2
    vmovdqu YMMWORD PTR [r8 + 96], ymm3
    vmovdqu YMMWORD PTR [r8 + 128], ymm4
    vmovdqu YMMWORD PTR [r8 + 160], ymm5
    ret
op_zeroall ENDP

; Chained multi-register sequence exercising AVX packed-integer ops, GPR bridge and upper-128 zeroing.
op_pipeline PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm1, XMMWORD PTR [rcx]
    vmovdqu xmm2, XMMWORD PTR [rdx]
    vpaddsb xmm3, xmm1, xmm2
    vpsignb xmm4, xmm3, xmm2
    vpshuflw xmm5, xmm4, 4Eh
    vpcmpgtb xmm3, xmm5, xmm2
    vpblendvb xmm4, xmm1, xmm5, xmm3
    vpmovmskb eax, xmm4
    vpinsrd xmm4, xmm4, eax, 3
    vphminposuw xmm5, xmm4
    vpalignr xmm0, xmm4, xmm5, 6
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_pipeline ENDP

op_movdqu_copy128 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rdx]
    vmovdqu xmm1, xmm0
    vmovdqu YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_movdqu_copy128 ENDP

op_mov_store32 PROC FRAME
    .endprolog
    mov eax, DWORD PTR [rdx]
    mov QWORD PTR [r8], rax
    mov DWORD PTR [r8+16], eax
    vzeroupper
    ret
op_mov_store32 ENDP

op_movdqu_regcopy PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, ymm0
    vmovdqu YMMWORD PTR [r8], ymm1
    vzeroupper
    ret
op_movdqu_regcopy ENDP

op_movdqu_store128 PROC FRAME
    .endprolog
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu XMMWORD PTR [r8], xmm0
    vzeroupper
    ret
op_movdqu_store128 ENDP

op_movdqu_load128 PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rdx]
    vmovdqu xmm0, XMMWORD PTR [rcx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_movdqu_load128 ENDP

END
