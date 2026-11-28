; Explicit .pdata/.xdata lets the test place another function in the gap.
OPTION DOTNAME
.code
PUBLIC choose_value, neighbor
choose_value PROC
    push rbx
    push rsi
IFDEF USE_LARGE_FRAME
    sub rsp, 136
    mov [rsp + 32], rcx
    mov rsi, [rsp + 32]
ELSE
    sub rsp, 8
    mov rsi, rcx
ENDIF
IFDEF USE_JNE
    test rsi, rsi
    jnz short load_inner
    jmp short empty_value
load_inner:
    mov rbx, [rsi]
    test rbx, rbx
    jnz short non_empty
    jmp short empty_value
non_empty:
    mov eax, 21
    jmp short finish_value
ELSE
    test rsi, rsi
    jz short empty_value
    mov rbx, [rsi]
    test rbx, rbx
    jz short empty_value
IFDEF USE_XOR_ZERO
    xor rdx, rdx
    lea rax, [rdx + 21]
ELSEIFDEF USE_IMM_REG
    mov r8d, 80000015h
    lea rax, [r8 - 2147483648]
ELSE
    mov eax, 21
ENDIF
    jmp short finish_value
ENDIF
choose_value ENDP
primary_end LABEL BYTE
neighbor PROC
    mov eax, 99
    ret
neighbor ENDP
neighbor_end LABEL BYTE
empty_value::
IFDEF USE_XOR_ZERO
    xor eax, eax
ELSEIFDEF USE_IMM_REG
    mov edx, 7
    lea rdx, [rdx - 2147483648]
    mov eax, edx
    lea rax, [rax - 2147483648]
ELSE
    mov eax, 7
ENDIF
finish_value::
IFDEF USE_LARGE_FRAME
    add rsp, 136
ELSE
    add rsp, 8
ENDIF
    pop rsi
    pop rbx
    ret
finish_end LABEL BYTE

.xdata SEGMENT READONLY
ALIGN 4
IFDEF USE_LARGE_FRAME
primary_info DB 1,9,4,0, 9,01h, 11h,00h, 2,60h, 1,30h
ELSE
primary_info DB 1,6,3,0, 6,02h, 2,60h, 1,30h, 0,0
ENDIF
neighbor_info DB 1,0,0,0
empty_info DB 21h,0,0,0
    DD IMAGEREL choose_value, IMAGEREL primary_end, IMAGEREL primary_info
finish_info DB 21h,0,0,0
    DD IMAGEREL empty_value, IMAGEREL finish_value, IMAGEREL empty_info
.xdata ENDS
.pdata SEGMENT READONLY
    DD IMAGEREL choose_value, IMAGEREL primary_end, IMAGEREL primary_info
    DD IMAGEREL neighbor, IMAGEREL neighbor_end, IMAGEREL neighbor_info
    DD IMAGEREL empty_value, IMAGEREL finish_value, IMAGEREL empty_info
    DD IMAGEREL finish_value, IMAGEREL finish_end, IMAGEREL finish_info
.pdata ENDS
END
