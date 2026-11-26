; coverage: record cpu_aot_movbe_dedicated
; Source-built MASM fixture for all 6 MOVBE forms (16/32/64-bit loads and stores)
; from docs/coverage/x64_zen2.tsv group MOVBE: [rcx|rdx|r8 (+small_disp)] loads,
; [r8 (+disp)] stores, 16-bit upper-bit preservation, 32-bit zero-extension,
; zero flag modification, [r8] source-then-store, and discarded faulting m64 read.
.code

PACK_CF_ZF_SF_OF_PF MACRO
    setb r9b
    sete r10b
    sets r11b
    seto byte ptr [r8+16]
    setp byte ptr [r8+20]
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    movzx r11d, r11b
    shl r11d, 2
    or r9d, r11d
    movzx r10d, byte ptr [r8+16]
    shl r10d, 3
    or r9d, r10d
    movzx r10d, byte ptr [r8+20]
    shl r10d, 4
    or r9d, r10d
    mov dword ptr [r8+16], r9d
ENDM

; --- Loads (16-bit preserves [63:16], 32-bit zero-extends [63:32], 64-bit full) ---

op_movbe_load16 PROC FRAME
    .endprolog
    mov rax, rdx
IFDEF ALTER_MOVBE
    mov ax, word ptr [rcx]
ELSE
    movbe ax, word ptr [rcx]
ENDIF
    ret
op_movbe_load16 ENDP

op_movbe_load16_disp PROC FRAME
    .endprolog
    mov r9, rdx
IFDEF ALTER_MOVBE
    movbe r9w, word ptr [rcx+1]
ELSE
    movbe r9w, word ptr [rcx+3]
ENDIF
    mov rax, r9
    ret
op_movbe_load16_disp ENDP

op_movbe_load32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_MOVBE
    mov eax, dword ptr [rcx]
ELSE
    movbe eax, dword ptr [rcx]
ENDIF
    ret
op_movbe_load32 ENDP

op_movbe_load32_disp PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_MOVBE
    movbe r10d, dword ptr [rdx+1]
ELSE
    movbe r10d, dword ptr [rdx+5]
ENDIF
    mov rax, r10
    ret
op_movbe_load32_disp ENDP

op_movbe_load64 PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    mov rax, qword ptr [rcx]
ELSE
    movbe rax, qword ptr [rcx]
ENDIF
    ret
op_movbe_load64 ENDP

op_movbe_load64_disp PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    movbe r11, qword ptr [rdx+4]
ELSE
    movbe r11, qword ptr [rdx+8]
ENDIF
    mov rax, r11
    ret
op_movbe_load64_disp ENDP

; --- Stores strictly to [r8] or [r8+disp] ---

op_movbe_store16 PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    mov word ptr [r8], dx
ELSE
    movbe word ptr [r8], dx
ENDIF
    mov rax, rdx
    ret
op_movbe_store16 ENDP

op_movbe_store16_disp PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    movbe word ptr [r8+4], dx
ELSE
    movbe word ptr [r8+2], dx
ENDIF
    mov rax, rdx
    ret
op_movbe_store16_disp ENDP

op_movbe_store32 PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    mov dword ptr [r8], edx
ELSE
    movbe dword ptr [r8], edx
ENDIF
    mov eax, edx
    ret
op_movbe_store32 ENDP

op_movbe_store32_disp PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    movbe dword ptr [r8+8], edx
ELSE
    movbe dword ptr [r8+4], edx
ENDIF
    mov eax, edx
    ret
op_movbe_store32_disp ENDP

op_movbe_store64 PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    mov qword ptr [r8], rdx
ELSE
    movbe qword ptr [r8], rdx
ENDIF
    mov rax, rdx
    ret
op_movbe_store64 ENDP

op_movbe_store64_disp PROC FRAME
    .endprolog
IFDEF ALTER_MOVBE
    movbe qword ptr [r8+16], rdx
ELSE
    movbe qword ptr [r8+8], rdx
ENDIF
    mov rax, rdx
    ret
op_movbe_store64_disp ENDP

; --- Flag preservation checks (PACK_CF_ZF_SF_OF_PF -> dword ptr [r8+16]) ---

op_movbe_flags_cf_zf_of_pf PROC FRAME
    .endprolog
    mov r9d, 80000000h
    shl r9d, 1
    movbe eax, dword ptr [rcx]
    movbe dword ptr [r8], eax
    mov r10d, 0
    mov r11d, 0
    PACK_CF_ZF_SF_OF_PF
    ret
op_movbe_flags_cf_zf_of_pf ENDP

op_movbe_flags_sf_of_pf PROC FRAME
    .endprolog
    mov r9d, 40000000h
    shl r9d, 1
    movbe rax, qword ptr [rcx]
    movbe qword ptr [r8+8], rax
    mov r10d, 0
    mov r11d, 0
    PACK_CF_ZF_SF_OF_PF
    ret
op_movbe_flags_sf_of_pf ENDP

op_movbe_flags_all_clear PROC FRAME
    .endprolog
    mov r9d, 0
    or r9d, 1
    mov rax, -1
    movbe ax, word ptr [rcx]
    movbe word ptr [r8], ax
    mov r10d, 0
    mov r11d, 0
    PACK_CF_ZF_SF_OF_PF
    ret
op_movbe_flags_all_clear ENDP

; --- [r8] source-then-store and discarded faulting m64 read ---

op_r8_source_then_store PROC FRAME
    .endprolog
    mov r11, rdx
    movbe r9, qword ptr [r8]
    movbe r10d, dword ptr [rcx]
    movbe r11w, word ptr [rdx]
    movbe qword ptr [r8+8], r9
    movbe dword ptr [r8], r10d
    mov rax, r11
    ret
op_r8_source_then_store ENDP

op_movbe_discarded_read PROC FRAME
    .endprolog
    movbe r9, qword ptr [rdx]
    mov rax, rcx
    ret
op_movbe_discarded_read ENDP

END
