; All three functions take (uint64_t, uint64_t, uint64_t trace[3]).
; call_pair calls two different callees. Only RBX and RSI survive each call,
; so the trace pointer is parked in RSI and copied back to R8 for the second.
.code
mix_args PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 8
    .allocstack 8
    .endprolog
    mov rbx, rdx
    mov rsi, rcx
    mov [r8], rsi
    mov [r8+8], rbx
    mov rax, rsi
    sub rax, rbx
    ; sub rax, -11, forced to the signed imm8 encoding.
    DB 048h, 083h, 0E8h, 0F5h
    mov [r8+16], rax
IFDEF ALTER_FIRST
    add rax, 1
ENDIF
    add rsp, 8
    pop rsi
    pop rbx
    ret
mix_args ENDP

; XOR keeps the two callees from commuting, so swapping the calls is visible.
mix_other PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 8
    .allocstack 8
    .endprolog
    mov rbx, rdx
    mov rsi, rcx
    mov [r8], rsi
    mov [r8+8], rbx
    mov rax, rsi
    xor rax, rbx
    add rax, 29
    mov [r8+16], rax
IFDEF ALTER_SECOND
    add rax, 1
ENDIF
    add rsp, 8
    pop rsi
    pop rbx
    ret
mix_other ENDP

call_pair PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, rcx
    mov rsi, r8
    add rcx, 7
    sub rdx, 3
IFDEF SWAP_CALLS
    call mix_other
ELSE
    call mix_args
ENDIF
    ; ADD feeds the first result to the second call, so a callee that returns
    ; one more always changes the second call's first argument.
    mov rcx, rax
    add rcx, rbx
    mov rdx, rbx
    mov r8, rsi
IFDEF SWAP_CALLS
    call mix_args
ELSE
    call mix_other
ENDIF
    add rax, rbx
    add rsp, 28h
    pop rsi
    pop rbx
    ret
call_pair ENDP
END
