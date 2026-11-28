.data
callback_count QWORD 0
callback_arg QWORD 0
__security_cookie QWORD 02B992DDFA232h

.const
PUBLIC _load_config_used
_load_config_used LABEL BYTE
    DWORD 138h
    BYTE 54h DUP (0)
    QWORD __security_cookie
    QWORD 0
    QWORD 0
    QWORD 0
dispatch_slot LABEL QWORD
    IFDEF DISPATCH_ABSENT
    QWORD 0
    ELSEIFDEF DISPATCH_ALTERNATE
    QWORD alternate_callback
    ELSE
    QWORD dispatch_callback
    ENDIF
    QWORD 0
    QWORD 0
    DWORD 0
    BYTE 0A4h DUP (0)

.code
dispatch_callback PROC FRAME
    .endprolog
    inc qword ptr [callback_count]
    mov qword ptr [callback_arg], rcx
    mov eax, 5Ah
    ret
dispatch_callback ENDP

alternate_callback PROC FRAME
    .endprolog
    inc qword ptr [callback_count]
    mov qword ptr [callback_arg], rcx
    mov eax, 0A5h
    ret
alternate_callback ENDP

native_entry PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov rax, qword ptr [dispatch_slot]
    test rax, rax
    jz absent
    mov ecx, 1234h
    call rax
    mov r10d, eax
    mov eax, 0B100h
    IFDEF DISPATCH_ALTERNATE
    or eax, 40h
    ENDIF
    cmp qword ptr [callback_count], 1
    jne bad
    or eax, 10h
    cmp qword ptr [callback_arg], 1234h
    jne bad
    or eax, 1
    IFDEF DISPATCH_ALTERNATE
    cmp r10d, 0A5h
    ELSE
    cmp r10d, 5Ah
    ENDIF
    jne bad
    or eax, 100h
    jmp done
absent:
    mov eax, 0B200h
    cmp qword ptr [callback_count], 0
    jne bad
    or eax, 10h
    or eax, 1
    jmp done
bad:
    mov eax, 0E000h
done:
    add rsp, 38h
    ret
native_entry ENDP
END