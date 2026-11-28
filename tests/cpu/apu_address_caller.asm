; Source-built caller for acphal.dll ACP address and memory mapping exports:
;   acphal.dll!ApuAlloc
;   acphal.dll!ApuFree
;   acphal.dll!ApuMapVirtualAddress
;   acphal.dll!ApuMapApuAddress
;   acphal.dll!ApuIsVirtualAddressValid
;
; Public call shape from DirectXTK12 (SoundEffect.cpp, WaveBankReader.cpp, SoundStreamInstance.cpp).
; Validates bidirectional mapping identity, linear offset preservation, validity lifecycle,
; and unmapped/null/stack address refusals.
EXTERN __imp_ApuAlloc:QWORD
EXTERN __imp_ApuFree:QWORD
EXTERN __imp_ApuMapVirtualAddress:QWORD
EXTERN __imp_ApuMapApuAddress:QWORD
EXTERN __imp_ApuIsVirtualAddressValid:QWORD
EXTERN __imp_get_apu_unmapped_refusals:QWORD
PUBLIC native_entry

.data
first_block     dq 0
apu_addr        dq 0

.code
native_entry PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog

    ; Step 1: Allocate 4 KiB block at 2 KiB alignment via ApuAlloc
    lea rcx, first_block
    xor edx, edx
    mov r8d, 1000h
    mov r9d, 800h
    call qword ptr [__imp_ApuAlloc]
    test eax, eax
    jnz fail_1
    mov rax, qword ptr [first_block]
    test rax, rax
    jz fail_1
    test eax, 7FFh
    jnz fail_1

    ; Step 2: ApuIsVirtualAddressValid on live block must return 1 (TRUE)
    mov rcx, qword ptr [first_block]
    call qword ptr [__imp_ApuIsVirtualAddressValid]
    cmp eax, 1
    jne fail_2

    ; Step 3: ApuMapVirtualAddress on block must return non-zero APU address
    mov rcx, qword ptr [first_block]
    call qword ptr [__imp_ApuMapVirtualAddress]
    test rax, rax
    jz fail_3
    mov qword ptr [apu_addr], rax

    ; Step 4: Bidirectional mapping identity: ApuMapApuAddress(apu_addr) == first_block
    mov rcx, qword ptr [apu_addr]
    call qword ptr [__imp_ApuMapApuAddress]
    cmp rax, qword ptr [first_block]
    jne fail_4

    ; Step 5: Offset preservation: first_block + 0x400 <-> apu_addr + 0x400
    mov rcx, qword ptr [first_block]
    add rcx, 400h
    call qword ptr [__imp_ApuMapVirtualAddress]
    mov rdx, qword ptr [apu_addr]
    add rdx, 400h
    cmp rax, rdx
    jne fail_5

    mov rcx, rdx
    call qword ptr [__imp_ApuMapApuAddress]
    mov rdx, qword ptr [first_block]
    add rdx, 400h
    cmp rax, rdx
    jne fail_5

    ; Step 6: Data write & readback through pointer
    mov rax, qword ptr [first_block]
    mov dword ptr [rax], 5A17A0C1h
    mov dword ptr [rax + 400h], 0A0D10C0Dh
    cmp dword ptr [rax], 5A17A0C1h
    jne fail_6
    cmp dword ptr [rax + 400h], 0A0D10C0Dh
    jne fail_6

    ; Step 7: Refusal controls on invalid / unmapped addresses
    ; 7a: Null pointer
    xor ecx, ecx
    call qword ptr [__imp_ApuIsVirtualAddressValid]
    test eax, eax
    jnz fail_7
    xor ecx, ecx
    call qword ptr [__imp_ApuMapVirtualAddress]
    test rax, rax
    jnz fail_7
    xor ecx, ecx
    call qword ptr [__imp_ApuMapApuAddress]
    test rax, rax
    jnz fail_7

    ; 7b: Stack pointer
    lea rcx, [rsp]
    call qword ptr [__imp_ApuIsVirtualAddressValid]
    test eax, eax
    jnz fail_7
    lea rcx, [rsp]
    call qword ptr [__imp_ApuMapVirtualAddress]
    test rax, rax
    jnz fail_7

    ; 7c: Out-of-range APU address
    mov rcx, 0DEADBEEF00000000h
    call qword ptr [__imp_ApuMapApuAddress]
    test rax, rax
    jnz fail_7

    ; Step 8: Release the block via ApuFree
    mov rcx, qword ptr [first_block]
    call qword ptr [__imp_ApuFree]

    ; Step 9: Post-release verification: address is no longer valid or mapped
    mov rcx, qword ptr [first_block]
    call qword ptr [__imp_ApuIsVirtualAddressValid]
    test eax, eax
    jnz fail_9
    mov rcx, qword ptr [first_block]
    call qword ptr [__imp_ApuMapVirtualAddress]
    test rax, rax
    jnz fail_9
    mov rcx, qword ptr [apu_addr]
    call qword ptr [__imp_ApuMapApuAddress]
    test rax, rax
    jnz fail_9

    ; Step 10: Provider counters check (at least 6 refusals recorded)
    call qword ptr [__imp_get_apu_unmapped_refusals]
    cmp eax, 6
    jl fail_10

    mov eax, 8A1h
    jmp done

fail_1:
    mov eax, 0F01h
    jmp done
fail_2:
    mov eax, 0F02h
    jmp done
fail_3:
    mov eax, 0F03h
    jmp done
fail_4:
    mov eax, 0F04h
    jmp done
fail_5:
    mov eax, 0F05h
    jmp done
fail_6:
    mov eax, 0F06h
    jmp done
fail_7:
    mov eax, 0F07h
    jmp done
fail_9:
    mov eax, 0F09h
    jmp done
fail_10:
    mov eax, 0F0Ah

done:
    add rsp, 28h
    ret
native_entry ENDP
END
