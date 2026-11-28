; Native-only harness for tests/cpu/cpuid.py; it is never translated.
;
; cpuid_raw(leaf, subleaf, out) runs CPUID with EAX = leaf and ECX = subleaf and stores EAX, EBX, ECX and EDX,
; zero-extended, to out[0..3]. has_osxsave returns CPUID.01H:ECX bit 27, the precondition for XGETBV.
; xgetbv_raw(index) returns EDX:EAX of XGETBV; call it only after has_osxsave returned 1.
.code

cpuid_raw PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    mov eax, ecx
    mov ecx, edx
    cpuid
    mov qword ptr [r8], rax
    mov qword ptr [r8+8], rbx
    mov qword ptr [r8+16], rcx
    mov qword ptr [r8+24], rdx
    pop rbx
    ret
cpuid_raw ENDP

has_osxsave PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    mov eax, 1
    xor ecx, ecx
    cpuid
    mov eax, ecx
    shr eax, 27
    and eax, 1
    pop rbx
    ret
has_osxsave ENDP

xgetbv_raw PROC FRAME
    .endprolog
    xgetbv
    shl rdx, 32
    or rax, rdx
    ret
xgetbv_raw ENDP

END
