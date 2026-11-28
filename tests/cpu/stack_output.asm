EXTERN __imp_write_result:QWORD
EXTERN __imp_write_constant:QWORD

.code

from_frame PROC FRAME
    push rbp
    .pushreg rbp
    sub rsp, 30h
    .allocstack 30h
    lea rbp, [rsp+20h]
    .setframe rbp, 20h
    .endprolog
    lea rcx, [rbp+8]
    xor rdx, 711h
    call qword ptr [__imp_write_result]
    mov eax, eax
    add rax, [rbp+8]
    add rsp, 30h
    pop rbp
    ret
from_frame ENDP

from_rsp PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call qword ptr [__imp_write_constant]
    mov rax, [rsp+20h]
    add rsp, 28h
    ret
from_rsp ENDP

overlaps_home PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+8]
    call qword ptr [__imp_write_constant]
    add rsp, 28h
    ret
overlaps_home ENDP

bypasses_lea PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test dl, 1
    jnz host_call
    lea rcx, [rsp+20h]
host_call:
    call qword ptr [__imp_write_constant]
    add rsp, 28h
    ret
bypasses_lea ENDP

misaligned PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    lea rcx, [rsp+21h]
    call qword ptr [__imp_write_constant]
    add rsp, 38h
    ret
misaligned ENDP

END
