; coverage: record cpu_aot_bmi1_logic
; Source-built MASM fixture for BMI1 ANDN, BEXTR, and TZCNT from
; docs/coverage/x64_zen2.tsv group BMI/ADX in 32-bit and 64-bit register and
; flat [rcx|rdx|r8] memory forms, plus SDM flag-packing, 32-bit zero-extension,
; destination aliasing, [r8] source-then-store, and discarded faulting m64 read.
.code

PACK_CF_ZF_SF_OF MACRO
    setb r9b
    sete r10b
    sets r11b
    seto byte ptr [r8]
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    movzx r11d, r11b
    shl r11d, 2
    or r9d, r11d
    movzx r10d, byte ptr [r8]
    shl r10d, 3
    or r9d, r10d
    mov dword ptr [r8], r9d
ENDM

PACK_CF_ZF_OF MACRO
    setb r9b
    sete r10b
    seto r11b
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    movzx r11d, r11b
    shl r11d, 3
    or r9d, r11d
    mov dword ptr [r8], r9d
ENDM

PACK_CF_ZF MACRO
    setb r9b
    sete r10b
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    mov dword ptr [r8], r9d
ENDM

; --- Register forms (32-bit & 64-bit; mov rax,-1 before 32-bit tests [63:32] zero-ext) ---

op_andn32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1_LOGIC
    andn eax, edx, ecx
ELSE
    andn eax, ecx, edx
ENDIF
    ret
op_andn32 ENDP

op_andn64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI1_LOGIC
    andn rax, rdx, rcx
ELSE
    andn rax, rcx, rdx
ENDIF
    ret
op_andn64 ENDP

op_bextr32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1_LOGIC
    bextr eax, edx, ecx
ELSE
    bextr eax, ecx, edx
ENDIF
    ret
op_bextr32 ENDP

op_bextr64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI1_LOGIC
    bextr rax, rdx, rcx
ELSE
    bextr rax, rcx, rdx
ENDIF
    ret
op_bextr64 ENDP

op_tzcnt32 PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1_LOGIC
    bextr eax, ecx, ecx
ELSE
    tzcnt eax, ecx
ENDIF
    ret
op_tzcnt32 ENDP

op_tzcnt64 PROC FRAME
    .endprolog
IFDEF ALTER_BMI1_LOGIC
    bextr rax, rcx, rcx
ELSE
    tzcnt rax, rcx
ENDIF
    ret
op_tzcnt64 ENDP

; --- Flat [rcx|rdx] m32 / m64 source forms ---

op_andn32_mem PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1_LOGIC
    bextr eax, dword ptr [rcx], edx
ELSE
    andn eax, edx, dword ptr [rcx]
ENDIF
    ret
op_andn32_mem ENDP

op_andn64_mem PROC FRAME
    .endprolog
IFDEF ALTER_BMI1_LOGIC
    bextr rax, qword ptr [rcx], rdx
ELSE
    andn rax, rdx, qword ptr [rcx]
ENDIF
    ret
op_andn64_mem ENDP

op_bextr32_mem PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1_LOGIC
    andn eax, edx, dword ptr [rcx]
ELSE
    bextr eax, dword ptr [rcx], edx
ENDIF
    ret
op_bextr32_mem ENDP

op_bextr64_mem PROC FRAME
    .endprolog
IFDEF ALTER_BMI1_LOGIC
    andn rax, rdx, qword ptr [rcx]
ELSE
    bextr rax, qword ptr [rcx], rdx
ENDIF
    ret
op_bextr64_mem ENDP

op_tzcnt32_mem PROC FRAME
    .endprolog
    mov rax, -1
IFDEF ALTER_BMI1_LOGIC
    bextr eax, dword ptr [rdx], ecx
ELSE
    tzcnt eax, dword ptr [rdx]
ENDIF
    ret
op_tzcnt32_mem ENDP

op_tzcnt64_mem PROC FRAME
    .endprolog
IFDEF ALTER_BMI1_LOGIC
    bextr rax, qword ptr [rdx], rcx
ELSE
    tzcnt rax, qword ptr [rdx]
ENDIF
    ret
op_tzcnt64_mem ENDP

; --- Flag-packing exports stored at [r8] ---

op_andn32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    mov rax, -1
    andn eax, ecx, edx
    PACK_CF_ZF_SF_OF
    ret
op_andn32_flags ENDP

op_andn64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    andn rax, rcx, rdx
    PACK_CF_ZF_SF_OF
    ret
op_andn64_flags ENDP

op_bextr32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    mov rax, -1
    bextr eax, ecx, edx
    PACK_CF_ZF_OF
    ret
op_bextr32_flags ENDP

op_bextr64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    bextr rax, rcx, rdx
    PACK_CF_ZF_OF
    ret
op_bextr64_flags ENDP

op_tzcnt32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    mov rax, -1
    tzcnt eax, ecx
    PACK_CF_ZF
    ret
op_tzcnt32_flags ENDP

op_tzcnt64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    tzcnt rax, rcx
    PACK_CF_ZF
    ret
op_tzcnt64_flags ENDP

op_bextr64_mem_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    xor r11d, r11d
    bextr rax, qword ptr [rcx], rdx
    PACK_CF_ZF_OF
    ret
op_bextr64_mem_flags ENDP

op_tzcnt64_mem_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r10d, r10d
    tzcnt rax, qword ptr [rcx]
    PACK_CF_ZF
    ret
op_tzcnt64_mem_flags ENDP

; --- [r8] source-then-store, dst aliasing src, and discarded faulting m64 read ---

op_r8_source_then_store PROC FRAME
    .endprolog
    mov r11, qword ptr [rdx]
    andn r9, r11, qword ptr [r8]
    bextr r10, qword ptr [rcx], r11
    tzcnt r11, qword ptr [rdx]
    or r9, r10
    xor r9, r11
    mov qword ptr [r8], r9
    mov rax, r9
    ret
op_r8_source_then_store ENDP

op_dst_alias_src PROC FRAME
    .endprolog
    mov r9, rcx
    mov r10, rdx
    andn r9, r9, r10
    bextr r10, r10, r9
    tzcnt r10, r10
    or r9, r10
    mov rax, r9
    ret
op_dst_alias_src ENDP

op_bmi1_logic_discarded_read PROC FRAME
    .endprolog
    tzcnt r9, qword ptr [rdx]
    mov rax, rcx
    ret
op_bmi1_logic_discarded_read ENDP

END
