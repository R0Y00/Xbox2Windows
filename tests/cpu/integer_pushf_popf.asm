; coverage: source cpu_aot_integer_pushf_popf
; Source-built MASM fixture exercising uncovered Integer forms `pushf`, `pushfq`,
; `popf`, and `popfq` from docs/coverage/x64_zen2.tsv after 8-, 16-, 32-, and
; 64-bit ADD, SUB, and CMP, with POPF/POPFQ feeding LAHF, SETcc, and CLD/STD/MOVSB,
; nested push/pop sequences, and ring-3 RFLAGS bit normalization.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const uint64_t* input_a (32 bytes)
;   RDX = const uint64_t* input_b (32 bytes)
;   R8  = uint64_t*       output  (32 bytes)

PUSHF_CAPTURE_AND_CONSUME MACRO
    pushf
    mov ax, WORD PTR [rsp]
    popf
    mov QWORD PTR [r8], rax
    setb BYTE PTR [r8+8]
    setl BYTE PTR [r8+9]
    sete BYTE PTR [r8+10]
    sets BYTE PTR [r8+11]
    seto BYTE PTR [r8+12]
    setp BYTE PTR [r8+13]
    ret
ENDM

PUSHFQ_CAPTURE_AND_CONSUME MACRO
    pushfq
    mov rax, QWORD PTR [rsp]
    popfq
    mov QWORD PTR [r8], rax
    setb BYTE PTR [r8+8]
    setl BYTE PTR [r8+9]
    sete BYTE PTR [r8+10]
    sets BYTE PTR [r8+11]
    seto BYTE PTR [r8+12]
    setp BYTE PTR [r8+13]
    ret
ENDM

.code

op_add8_pushf PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    add r9b, r10b
    PUSHF_CAPTURE_AND_CONSUME
op_add8_pushf ENDP

op_add16_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    add r9w, r10w
    PUSHFQ_CAPTURE_AND_CONSUME
op_add16_pushfq ENDP

op_add32_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    add r9d, r10d
    PUSHFQ_CAPTURE_AND_CONSUME
op_add32_pushfq ENDP

op_add64_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    add r9, r10
    PUSHFQ_CAPTURE_AND_CONSUME
op_add64_pushfq ENDP

op_sub8_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    sub r9b, r10b
    PUSHFQ_CAPTURE_AND_CONSUME
op_sub8_pushfq ENDP

op_sub16_pushf PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    sub r9w, r10w
    PUSHF_CAPTURE_AND_CONSUME
op_sub16_pushf ENDP

op_sub32_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    sub r9d, r10d
    PUSHFQ_CAPTURE_AND_CONSUME
op_sub32_pushfq ENDP

op_sub64_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    sub r9, r10
    PUSHFQ_CAPTURE_AND_CONSUME
op_sub64_pushfq ENDP

op_cmp8_pushf PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    cmp r9b, r10b
    PUSHF_CAPTURE_AND_CONSUME
op_cmp8_pushf ENDP

op_cmp16_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    cmp r9w, r10w
    PUSHFQ_CAPTURE_AND_CONSUME
op_cmp16_pushfq ENDP

op_cmp32_pushf PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    cmp r9d, r10d
    PUSHF_CAPTURE_AND_CONSUME
op_cmp32_pushf ENDP

op_cmp64_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    cmp r9, r10
    PUSHFQ_CAPTURE_AND_CONSUME
op_cmp64_pushfq ENDP

op_add8_al_pushf PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
IFDEF ALTER_IMM
    add al, 5
ELSE
    add al, r9b
ENDIF
    PUSHF_CAPTURE_AND_CONSUME
op_add8_al_pushf ENDP

op_add32_imm_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
IFDEF ALTER_IMM
    add r9d, 1
ELSE
    add r9d, 17
ENDIF
    PUSHFQ_CAPTURE_AND_CONSUME
op_add32_imm_pushfq ENDP

op_sub32_imm_pushf PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
IFDEF ALTER_IMM
    sub r9d, 1
ELSE
    sub r9d, 15
ENDIF
    PUSHF_CAPTURE_AND_CONSUME
op_sub32_imm_pushf ENDP

op_cmp64_imm_pushfq PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
IFDEF ALTER_IMM
    cmp r9, 1
ELSE
    cmp r9, 17
ENDIF
    PUSHFQ_CAPTURE_AND_CONSUME
op_cmp64_imm_pushfq ENDP

op_popf_lahf_setcc PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r11, QWORD PTR [rdx]
    and r11, -040101h
    mov r9, QWORD PTR [rdx+8]
    add r9, 0
    pushf
    mov WORD PTR [rsp], r11w
    popf
    lahf
    mov QWORD PTR [r8], rax
    setb BYTE PTR [r8+8]
    setl BYTE PTR [r8+9]
    sete BYTE PTR [r8+10]
    sets BYTE PTR [r8+11]
    seto BYTE PTR [r8+12]
    setp BYTE PTR [r8+13]
    cld
    ret
op_popf_lahf_setcc ENDP

op_popfq_df_movsb PROC FRAME
    .endprolog
    mov r10, rsi
    mov r11, rdi
    mov rsi, rdx
    add rsi, 16
    mov rdi, r8
    add rdi, 16
    mov r9, QWORD PTR [rcx]
    and r9, -040101h
    mov rax, QWORD PTR [rcx+8]
    add rax, 0
    pushfq
    mov QWORD PTR [rsp], r9
    popfq
    movsb
    movsb
    lahf
    setb BYTE PTR [r8+8]
    setp BYTE PTR [r8+9]
    cld
    sub rdi, r8
    mov QWORD PTR [r8], rdi
    mov QWORD PTR [r8+24], rax
    mov rax, rdi
    mov rsi, r10
    mov rdi, r11
    ret
op_popfq_df_movsb ENDP

op_pushfq_std_movsb_popfq PROC FRAME
    .endprolog
    mov r10, rsi
    mov r11, rdi
    mov rsi, rdx
    add rsi, 16
    mov rdi, r8
    add rdi, 16
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rcx+8]
    add r9, QWORD PTR [rcx+16]
    pushfq
    std
    movsb
    popfq
    lea rdi, [rdi+4]
    movsb
    lahf
    mov QWORD PTR [r8], rax
    setb BYTE PTR [r8+8]
    setl BYTE PTR [r8+9]
    sete BYTE PTR [r8+10]
    sets BYTE PTR [r8+11]
    seto BYTE PTR [r8+12]
    setp BYTE PTR [r8+13]
    sub rdi, r8
    mov QWORD PTR [r8+24], rdi
    mov rsi, r10
    mov rdi, r11
    ret
op_pushfq_std_movsb_popfq ENDP

op_nested_push_pop PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    mov r9, QWORD PTR [rdx]
    mov r10, QWORD PTR [rdx+8]
    add r9, r10
    pushfq
    mov r9, QWORD PTR [rdx+16]
    sub r9w, r10w
    pushf
    mov r11, QWORD PTR [rdx+24]
    cmp r11d, r10d
    pushfq
    popfq
    lahf
    mov QWORD PTR [r8], rax
    setb BYTE PTR [r8+24]
    seto BYTE PTR [r8+25]
    popf
    lahf
    mov QWORD PTR [r8+8], rax
    setb BYTE PTR [r8+26]
    seto BYTE PTR [r8+27]
    popfq
    lahf
    mov QWORD PTR [r8+16], rax
    setb BYTE PTR [r8+28]
    seto BYTE PTR [r8+29]
    setp BYTE PTR [r8+30]
    mov rax, QWORD PTR [r8]
    ret
op_nested_push_pop ENDP

op_popfq_normalize PROC FRAME
    .endprolog
    mov r9, QWORD PTR [rcx]
    mov r10, QWORD PTR [rcx+8]
    add r9, r10
    pushfq
    mov r11, QWORD PTR [rdx]
    pushfq
    mov QWORD PTR [rsp], r11
    and QWORD PTR [rsp], -040101h
    popfq
    pushfq
    mov rax, QWORD PTR [rsp]
    mov QWORD PTR [r8], rax
    setb BYTE PTR [r8+8]
    setl BYTE PTR [r8+9]
    sete BYTE PTR [r8+10]
    sets BYTE PTR [r8+11]
    seto BYTE PTR [r8+12]
    setp BYTE PTR [r8+13]
    lahf
    mov QWORD PTR [r8+16], rax
    popfq
    popfq
    mov rax, QWORD PTR [r8]
    ret
op_popfq_normalize ENDP

op_popf_normalize PROC FRAME
    .endprolog
    mov r9, QWORD PTR [rcx]
    mov r10, QWORD PTR [rcx+8]
    add r9, r10
    pushfq
    mov r11, QWORD PTR [rdx+8]
    pushfq
    mov QWORD PTR [rsp], r11
    and QWORD PTR [rsp], -040101h
    popfq
    mov r11, QWORD PTR [rdx]
    pushf
    mov WORD PTR [rsp], r11w
    and WORD PTR [rsp], 0FEFFh
    popf
    pushfq
    mov rax, QWORD PTR [rsp]
    mov QWORD PTR [r8], rax
    setb BYTE PTR [r8+8]
    setl BYTE PTR [r8+9]
    sete BYTE PTR [r8+10]
    sets BYTE PTR [r8+11]
    seto BYTE PTR [r8+12]
    setp BYTE PTR [r8+13]
    popfq
    popfq
    mov rax, QWORD PTR [r8]
    ret
op_popf_normalize ENDP

op_pushf_df_nt PROC FRAME
    .endprolog
    mov r9, QWORD PTR [rcx]
    and r9, -040101h
    mov rax, QWORD PTR [rcx+8]
    add rax, 0
    pushfq
    mov QWORD PTR [rsp], r9
    popfq
    pushf
    movzx eax, WORD PTR [rsp]
    popf
    mov QWORD PTR [r8], rax
    cld
    ret
op_pushf_df_nt ENDP

; The discarded read must still fault after the first store and before the second.
op_discarded_read PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rcx]
    add rax, 0
    pushfq
    mov QWORD PTR [r8], rax
    mov r9, QWORD PTR [rdx]
    mov QWORD PTR [r8+8], rax
    popfq
    ret
op_discarded_read ENDP

END
