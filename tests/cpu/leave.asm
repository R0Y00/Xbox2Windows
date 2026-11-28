; coverage: source cpu_aot_leave
; LEAVE as the epilogue of a frame with a proven RBP anchor, for the jump-table translator.
; Every export takes RCX, RDX and R8 = address of three writable QWORDs. The register probe seeds
; RBX and RSI before each call and samples them after it. leave_frame, leave_lea and leave_join are
; translated; the exports after them exist only to be rejected (tests/cpu/leave_check.py).
.code

; MOV RBP, RSP anchors the frame at the saved RBP. The flags of CMP survive LEAVE and reach SETB.
leave_frame PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    mov QWORD PTR [rbp-8], rcx
    mov QWORD PTR [rbp-10h], rdx
    mov rax, QWORD PTR [rbp-8]
    add rax, QWORD PTR [rbp-10h]
    mov QWORD PTR [r8], rax
    cmp rcx, rdx
    leave
    setb BYTE PTR [r8+8]
    ret
leave_frame ENDP

; LEA RBP, [RSP+20h] after SUB RSP, 20h also anchors at the saved RBP.
leave_lea PROC FRAME
    push rbp
    .pushreg rbp
    sub rsp, 20h
    .allocstack 20h
    lea rbp, [rsp+20h]
    .setframe rbp, 20h
    .endprolog
    mov QWORD PTR [rbp-10h], rcx
    xor rcx, rdx
    mov QWORD PTR [rbp-18h], rcx
    mov rax, QWORD PTR [rbp-10h]
    sub rax, QWORD PTR [rbp-18h]
    mov QWORD PTR [r8], rax
    leave
    ret
leave_lea ENDP

; Both paths keep the anchor and write the same local; LEAVE follows the join.
leave_join PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 10h
    .allocstack 10h
    .endprolog
    test dl, 1
    jz even_path
    lea rax, [rcx+rdx]
    mov QWORD PTR [rbp-8], rax
    jmp joined
even_path:
    mov rax, rcx
    xor rax, rdx
    mov QWORD PTR [rbp-8], rax
joined:
    mov rax, QWORD PTR [rbp-8]
    mov QWORD PTR [r8], rax
    leave
    ret
leave_join ENDP

; One path re-anchors RBP at RSP, so the anchors differ where the paths join before LEAVE.
leave_split PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 10h
    .allocstack 10h
    .endprolog
    test dl, 1
    jz split_join
    lea rbp, [rsp]
split_join:
    leave
    ret
leave_split ENDP

; The anchor slot was never written: LEAVE would load an undefined RBP.
leave_unsaved PROC FRAME
    sub rsp, 8
    .allocstack 8
    .endprolog
    mov rbp, rsp
    leave
    ret
leave_unsaved ENDP

; The saved RBP is overwritten with ordinary data before LEAVE: eight defined bytes, not the caller's RBP.
leave_overwritten PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    .endprolog
    mov QWORD PTR [rbp], rcx
    leave
    ret
leave_overwritten ENDP

; LEAVE releases the local at [RBP-8]; a later allocation must not read what it held.
leave_released PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    sub rsp, 10h
    .allocstack 10h
    .endprolog
    mov QWORD PTR [rbp-8], rcx
    leave
    sub rsp, 18h
    mov rax, QWORD PTR [rsp+8]
    add rsp, 18h
    ret
leave_released ENDP

; The anchor lies below RSP when LEAVE runs.
leave_lowers PROC FRAME
    sub rsp, 10h
    .allocstack 10h
    .endprolog
    mov rbp, rsp
    add rsp, 8
    leave
    ret
leave_lowers ENDP

; The anchor lies in the caller's home space, so LEAVE would raise RSP above entry RSP.
leave_above PROC FRAME
    .endprolog
    lea rbp, [rsp+10h]
    leave
    ret
leave_above ENDP

; A separate MOV RSP, RBP stays outside the frame model.
mov_rsp_frame PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    .endprolog
    mov rsp, rbp
    pop rbp
    ret
mov_rsp_frame ENDP

; ENTER stays outside the frame model.
enter_frame PROC FRAME
    .endprolog
    enter 10h, 0
    leave
    ret
enter_frame ENDP

; Static rejection control. The operand-size prefix selects a 2-byte pop.
leave_word PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    .endprolog
    db 66h, 0c9h
    ret
leave_word ENDP

END
