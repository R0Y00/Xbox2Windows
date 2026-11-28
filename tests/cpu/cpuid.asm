; coverage: source cpu_aot_cpuid
; Source-owned differential fixture for CPUID in the opt-in host-CPU profile. cpuid_op takes (leaf, subleaf,
; preset, state) in RCX, RDX, R8 and R9. EAX and ECX get the leaf and subleaf with their upper halves all ones,
; and RBX and RDX are all ones, so a translation that does not zero-extend a result leaves bits behind. ADD R8,
; R8 sets ZF and CF from the preset; CPUID leaves them alone, and (CF << 1) | ZF goes to state slot 4. RAX, RBX,
; RCX and RDX go to slots 0-3. The lowering has no PUSH, so RBX is saved in slot 5 and restored from it.
; cpuid_unused discards the result and returns 7. CPUID for the expected values is in cpuid_harness.asm.
.code

CPU_FLAGS MACRO slot
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

cpuid_op PROC FRAME
    .endprolog
    mov qword ptr [r9+40], rbx
    mov r10, 0FFFFFFFF00000000h
    mov r11d, ecx
    or r11, r10
    mov rax, r11
    mov r11d, edx
    or r11, r10
    mov rcx, r11
    mov rbx, -1
    mov rdx, -1
    add r8, r8
    cpuid
    mov qword ptr [r9], rax
    mov qword ptr [r9+8], rbx
    mov qword ptr [r9+16], rcx
    mov qword ptr [r9+24], rdx
    CPU_FLAGS 32
    mov rbx, qword ptr [r9+40]
    ret
cpuid_op ENDP

; A CPUID whose results are overwritten: the translation must still execute it, also at /O2.
cpuid_unused PROC FRAME
    .endprolog
    mov qword ptr [r9+40], rbx
    mov eax, ecx
    mov ecx, edx
    cpuid
    mov eax, 7
    mov rbx, qword ptr [r9+40]
    ret
cpuid_unused ENDP

END
