; coverage: source cpu_aot_xgetbv
; Source-owned differential fixture for XGETBV in the opt-in host-CPU profile. xgetbv_op takes (index, b, preset,
; state) in RCX, RDX, R8 and R9, so ECX holds the index; RAX and RDX are all ones first, so a translation that does
; not zero-extend leaves bits behind. ADD R8, R8 sets ZF and CF from the preset; XGETBV leaves them alone, and
; (CF << 1) | ZF goes to state slot 4. RAX and RDX go to slots 0 and 1. xgetbv_unused discards the result and
; returns 7. Call either only after cpuid_harness.asm's has_osxsave returned 1.
.code

XCR_FLAGS MACRO slot
    LOCAL no_carry, no_zero
    mov r10d, 0
    jae short no_carry
    mov r10d, 2
no_carry:
    jne short no_zero
    or r10d, 1
no_zero:
    mov qword ptr [r9+slot], r10
ENDM

xgetbv_op PROC FRAME
    .endprolog
    mov rax, -1
    mov rdx, -1
    add r8, r8
    xgetbv
    mov qword ptr [r9], rax
    mov qword ptr [r9+8], rdx
    XCR_FLAGS 32
    ret
xgetbv_op ENDP

; An XGETBV whose result is overwritten: the translation must still execute it, also at /O2.
xgetbv_unused PROC FRAME
    .endprolog
    xgetbv
    mov eax, 7
    ret
xgetbv_unused ENDP

END
