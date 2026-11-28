; Source-built stand-in for the main executable's D3D12ConfigureVirtualMemoryX call (#874).
; Like the observed caller, it zero-fills a 32-byte block, sets only the DWORD at +4, passes the
; block's address, and then makes two XMemVirtualAlloc reservations of the same size whatever the
; HRESULT was. The DWORD/size pairs are input samples from that caller, not a defined contract.
; Canaries on both sides of the block catch writes by the callee.
;
; Exit codes: 0 success, 1 block changed, 2 canary changed, 3 null reservation,
; otherwise the failed HRESULT from D3D12ConfigureVirtualMemoryX.

IFNDEF SAMPLE
SAMPLE EQU 3
ENDIF
IF SAMPLE EQ 3
RESERVATION_HIGH EQU 4      ; 16 GiB
ELSEIF SAMPLE EQ 4
RESERVATION_HIGH EQU 6      ; 24 GiB
ELSE
.ERR <SAMPLE must be 3 or 4>
ENDIF

EXTERN __imp_D3D12ConfigureVirtualMemoryX:QWORD
EXTERN __imp_XMemVirtualAlloc:QWORD

.data
reservations dq 2 dup (0)

.code
native_entry PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 80h
    .allocstack 80h
    .endprolog
    ; [rsp+30h] canary, [rsp+40h] 32-byte block, [rsp+60h] canary
    mov rax, 0A5A5A5A5A5A5A5A5h
    mov qword ptr [rsp+30h], rax
    mov qword ptr [rsp+38h], rax
    mov qword ptr [rsp+60h], rax
    mov qword ptr [rsp+68h], rax
    xor eax, eax
    mov qword ptr [rsp+40h], rax
    mov qword ptr [rsp+48h], rax
    mov qword ptr [rsp+50h], rax
    mov qword ptr [rsp+58h], rax
    mov dword ptr [rsp+44h], SAMPLE
    lea rcx, [rsp+40h]
    call qword ptr [__imp_D3D12ConfigureVirtualMemoryX]
    mov ebx, eax

    ; The observed caller only breaks into a debugger on failure and continues.
    xor ecx, ecx
    mov rdx, RESERVATION_HIGH
    shl rdx, 32
    mov r8d, 20402000h
    mov r9d, 21h
    mov dword ptr [rsp+20h], 4
    call qword ptr [__imp_XMemVirtualAlloc]
    mov qword ptr [reservations], rax
    xor ecx, ecx
    mov rdx, RESERVATION_HIGH
    shl rdx, 32
    mov r8d, 20402000h
    mov r9d, 3
    mov dword ptr [rsp+20h], 801h
    call qword ptr [__imp_XMemVirtualAlloc]
    mov qword ptr [reservations+8], rax

    mov rax, 0A5A5A5A5A5A5A5A5h
    cmp qword ptr [rsp+30h], rax
    jne fail_canary
    cmp qword ptr [rsp+38h], rax
    jne fail_canary
    cmp qword ptr [rsp+60h], rax
    jne fail_canary
    cmp qword ptr [rsp+68h], rax
    jne fail_canary
    mov eax, SAMPLE
    shl rax, 32
    cmp qword ptr [rsp+40h], rax
    jne fail_block
    cmp qword ptr [rsp+48h], 0
    jne fail_block
    cmp qword ptr [rsp+50h], 0
    jne fail_block
    cmp qword ptr [rsp+58h], 0
    jne fail_block
    cmp qword ptr [reservations], 0
    je fail_reservation
    cmp qword ptr [reservations+8], 0
    je fail_reservation

    mov eax, ebx
    test eax, eax
    js done
    xor eax, eax
    jmp done
fail_block:
    mov eax, 1
    jmp done
fail_canary:
    mov eax, 2
    jmp done
fail_reservation:
    mov eax, 3
done:
    add rsp, 80h
    pop rbx
    ret
native_entry ENDP
END
