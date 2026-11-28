; coverage: source cpu_aot_mxcsr
; Source-owned differential fixture for LDMXCSR and STMXCSR in the opt-in host-MXCSR profile. Each export takes
; (a, b, c, state) in RCX, RDX, R8 and R9; b is the requested MXCSR word. State layout (bytes):
;   00 caller word (STMXCSR at entry, LDMXCSR at exit)   04 guard
;   08 requested word                                     0C guard
;   10, 20 the ADDPS operands, 30 and 40 the CVTSS2SI operands (lane 0), all written by the test at run time
;   50 the ADDPS result                                   60 first CVTSS2SI result, 64 guard
;   68 second CVTSS2SI result, 6C guard                   70 STMXCSR read-back, 74 guard
;   78 the marker that mxcsr_after writes after its LDMXCSR, 80-FF guards
; mxcsr_round restores the caller's word itself. mxcsr_leak, mxcsr_after, mxcsr_badload and mxcsr_fault do not:
; there only the translated entry's boundary restores it, and the test calls them translated only.
.code

mxcsr_round PROC FRAME
    .endprolog
    stmxcsr dword ptr [r9]
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    movups xmm0, xmmword ptr [r9+10h]
    movups xmm1, xmmword ptr [r9+20h]
    addps xmm0, xmm1
    movups xmmword ptr [r9+50h], xmm0
    movups xmm2, xmmword ptr [r9+30h]
    cvtss2si eax, xmm2
    mov dword ptr [r9+60h], eax
    movups xmm2, xmmword ptr [r9+40h]
    cvtss2si eax, xmm2
    mov dword ptr [r9+68h], eax
    stmxcsr dword ptr [r9+70h]
    ldmxcsr dword ptr [r9]
    mov eax, 7
    ret
mxcsr_round ENDP

; Loads the requested word and returns without restoring the caller's.
mxcsr_leak PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    mov eax, 7
    ret
mxcsr_leak ENDP

; Guest work after the load: the marker store must not run when the word is refused.
mxcsr_after PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    mov qword ptr [r9+78h], 1
    mov eax, 7
    ret
mxcsr_after ENDP

; LDMXCSR of an operand the test places on an inaccessible page: the guest read faults before any load.
mxcsr_badload PROC FRAME
    .endprolog
    ldmxcsr dword ptr [rcx]
    mov eax, 7
    ret
mxcsr_badload ENDP

; A guest load through RCX after a valid word was loaded; the test points RCX at an inaccessible page.
mxcsr_fault PROC FRAME
    .endprolog
    mov dword ptr [r9+8], edx
    ldmxcsr dword ptr [r9+8]
    mov eax, dword ptr [rcx]
    ret
mxcsr_fault ENDP

END
