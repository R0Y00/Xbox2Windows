; coverage: record cpu_aot_adx_dedicated
; Source-built MASM fixture for ADX ADCX and ADOX from docs/coverage/x64_zen2.tsv
; group BMI/ADX in 32-bit and 64-bit register and flat [rcx|rdx|r8] memory forms,
; CF/OF-only isolation (preserving OF/CF/ZF/SF/PF), interleaved dual-carry chains,
; 32-bit zero-extension, destination aliasing, [r8] source-then-store, and discarded faulting m64 read.
.code

PACK_CF_OF_ZF_SF MACRO
    setb r9b
    seto r10b
    sete r11b
    sets byte ptr [r8]
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    movzx r11d, r11b
    shl r11d, 2
    or r9d, r11d
    movzx r10d, byte ptr [r8]
    shl r10d, 3
    or r9d, r10d
    mov dword ptr [r8], r9d
ENDM

; --- Register forms (32-bit & 64-bit; mov rax,rcx before 32-bit pins [63:32] zero-ext by ADCX/ADOX) ---

op_adcx32_c0 PROC FRAME
    .endprolog
    xor r9d, r9d
    mov rax, rcx
IFDEF ALTER_ADX
    adox eax, edx
ELSE
    adcx eax, edx
ENDIF
    ret
op_adcx32_c0 ENDP

op_adcx32_c1 PROC FRAME
    .endprolog
    xor r9d, r9d
    stc
    mov rax, -1
    mov eax, ecx
IFDEF ALTER_ADX
    adox eax, edx
ELSE
    adcx eax, edx
ENDIF
    ret
op_adcx32_c1 ENDP

op_adcx64_c0 PROC FRAME
    .endprolog
    xor r9d, r9d
    mov rax, rcx
IFDEF ALTER_ADX
    adox rax, rdx
ELSE
    adcx rax, rdx
ENDIF
    ret
op_adcx64_c0 ENDP

op_adcx64_c1 PROC FRAME
    .endprolog
    xor r9d, r9d
    stc
    mov rax, rcx
IFDEF ALTER_ADX
    adox rax, rdx
ELSE
    adcx rax, rdx
ENDIF
    ret
op_adcx64_c1 ENDP

op_adox32_o0 PROC FRAME
    .endprolog
    xor r9d, r9d
    mov rax, rcx
IFDEF ALTER_ADX
    adcx eax, edx
ELSE
    adox eax, edx
ENDIF
    ret
op_adox32_o0 ENDP

op_adox32_o1 PROC FRAME
    .endprolog
    mov r9d, 07fffffffh
    add r9d, 1          ; OF=1, CF=0
    mov rax, -1
    mov eax, ecx
IFDEF ALTER_ADX
    adcx eax, edx
ELSE
    adox eax, edx
ENDIF
    ret
op_adox32_o1 ENDP

op_adox64_o0 PROC FRAME
    .endprolog
    xor r9d, r9d
    mov rax, rcx
IFDEF ALTER_ADX
    adcx rax, rdx
ELSE
    adox rax, rdx
ENDIF
    ret
op_adox64_o0 ENDP

op_adox64_o1 PROC FRAME
    .endprolog
    mov r9d, 07fffffffh
    add r9d, 1          ; OF=1, CF=0
    mov rax, rcx
IFDEF ALTER_ADX
    adcx rax, rdx
ELSE
    adox rax, rdx
ENDIF
    ret
op_adox64_o1 ENDP

; --- Flat [rcx|rdx|r8] m32 / m64 source forms (r8 carries 0/1 carry-in) ---

op_adcx32_mem PROC FRAME
    .endprolog
    mov r9d, r8d
    add r9d, 0ffffffffh ; CF = (r8d != 0), OF = 0
    mov rax, -1
    mov eax, ecx
IFDEF ALTER_ADX
    adox eax, dword ptr [rdx]
ELSE
    adcx eax, dword ptr [rdx]
ENDIF
    ret
op_adcx32_mem ENDP

op_adcx64_mem PROC FRAME
    .endprolog
    mov r9d, r8d
    add r9d, 0ffffffffh
    mov rax, rcx
IFDEF ALTER_ADX
    adox rax, qword ptr [rdx]
ELSE
    adcx rax, qword ptr [rdx]
ENDIF
    ret
op_adcx64_mem ENDP

op_adox32_mem PROC FRAME
    .endprolog
    mov r9d, r8d
    add r9d, 07fffffffh ; OF = (r8d != 0), CF = 0
    mov rax, -1
    mov eax, ecx
IFDEF ALTER_ADX
    adcx eax, dword ptr [rdx]
ELSE
    adox eax, dword ptr [rdx]
ENDIF
    ret
op_adox32_mem ENDP

op_adox64_mem PROC FRAME
    .endprolog
    mov r9d, r8d
    add r9d, 07fffffffh
    mov rax, rcx
IFDEF ALTER_ADX
    adcx rax, qword ptr [rdx]
ELSE
    adox rax, qword ptr [rdx]
ENDIF
    ret
op_adox64_mem ENDP

; --- Flag isolation exports stored at [r8] ---

; Seeds OF=1, CF=0, ZF=0, SF=1 (0x7fffffff + 1 = 0x80000000); ADCX updates ONLY CF.
op_adcx32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r11d, r11d
    mov r10d, 07fffffffh
    add r10d, 1
    mov rax, rcx
    adcx eax, edx
    PACK_CF_OF_ZF_SF
    ret
op_adcx32_flags ENDP

; Seeds OF=1, ZF=0, SF=1, then STC sets CF=1 preserving OF/ZF/SF; ADCX updates ONLY CF.
op_adcx64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r11d, r11d
    mov r10d, 07fffffffh
    add r10d, 1
    stc
    mov rax, rcx
    adcx rax, rdx
    PACK_CF_OF_ZF_SF
    ret
op_adcx64_flags ENDP

; Seeds CF=1, OF=0, ZF=1, SF=0 (0xffffffff + 1 = 0); ADOX updates ONLY OF.
op_adox32_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r11d, r11d
    mov r10d, 0ffffffffh
    add r10d, 1
    mov rax, rcx
    adox eax, edx
    PACK_CF_OF_ZF_SF
    ret
op_adox32_flags ENDP

; Seeds CF=1, OF=1, ZF=1, SF=0 (0x80000000 + 0x80000000 = 0); ADOX updates ONLY OF.
op_adox64_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r11d, r11d
    mov r10d, 080000000h
    add r10d, r10d
    mov rax, rcx
    adox rax, rdx
    PACK_CF_OF_ZF_SF
    ret
op_adox64_flags ENDP

; Reads [rcx] for ADCX, packs flags to [r8].
op_adcx64_mem_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r11d, r11d
    mov r10d, 07fffffffh
    add r10d, 1
    mov rax, rdx
    adcx rax, qword ptr [rcx]
    PACK_CF_OF_ZF_SF
    ret
op_adcx64_mem_flags ENDP

; Reads [rcx] for ADOX, packs flags to [r8].
op_adox64_mem_flags PROC FRAME
    .endprolog
    xor r9d, r9d
    xor r11d, r11d
    mov r10d, 0ffffffffh
    add r10d, 1
    mov rax, rdx
    adox rax, qword ptr [rcx]
    PACK_CF_OF_ZF_SF
    ret
op_adox64_mem_flags ENDP

; --- Interleaved dual-carry chain, [r8] source-then-store, dst aliasing, discarded m64 read ---

; Reads c = [r8], runs interleaved adcx/adox chain, packs ((r11 << 2) | (OF << 1) | CF) at [r8], returns rax (adcx sum).
op_adx_dual64_adcx PROC FRAME
    .endprolog
    mov r9, qword ptr [r8]
    mov r10d, 080000000h
    add r10d, r10d      ; CF=1, OF=1
    mov rax, rcx
    mov r11, rdx
    adcx rax, r9
    adox r11, r9
    adcx rax, rdx
    adox r11, rcx
    setb r9b
    seto r10b
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    shl r11, 2
    or r9, r11
    mov qword ptr [r8], r9
    ret
op_adx_dual64_adcx ENDP

; Same interleaved chain returning full 64-bit adox sum in rax and ((OF << 1) | CF) in dword [r8].
op_adx_dual64_adox PROC FRAME
    .endprolog
    mov r9, qword ptr [r8]
    mov r10d, 080000000h
    add r10d, r10d      ; CF=1, OF=1
    mov r11, rcx
    mov rax, rdx
    adcx r11, r9
    adox rax, r9
    adcx r11, rdx
    adox rax, rcx
    setb r9b
    seto r10b
    movzx r9d, r9b
    movzx r10d, r10b
    shl r10d, 1
    or r9d, r10d
    mov dword ptr [r8], r9d
    ret
op_adx_dual64_adox ENDP

op_adx_dst_alias PROC FRAME
    .endprolog
    xor r10d, r10d
    stc                 ; CF=1, OF=0
    mov r9, rcx
    mov r10, rdx
    adcx r9, r9
    adox r10, r10
    mov rax, r9
    xor rax, r10
    ret
op_adx_dst_alias ENDP

op_adx_discarded_read PROC FRAME
    .endprolog
    xor r9d, r9d
    adcx r9, qword ptr [rdx]
    mov rax, rcx
    ret
op_adx_discarded_read ENDP

END
