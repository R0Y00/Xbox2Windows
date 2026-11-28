; Source-built caller for acphal.dll!ApuAlloc and acphal.dll!ApuFree.
; Call shape: HRESULT ApuAlloc(void**, nullptr, UINT32 size, alignment), ApuFree(void*),
; as in public DirectXTK12 call sites. The 64 KiB capacity is the test provider's policy.
EXTERN __imp_ApuAlloc:QWORD
EXTERN __imp_ApuFree:QWORD
EXTERN __imp_get_apu_alloc_call_count:QWORD
EXTERN __imp_get_apu_free_call_count:QWORD
EXTERN __imp_get_apu_live_block_count:QWORD
EXTERN __imp_get_apu_rejected_free_count:QWORD
PUBLIC native_entry

SENTINEL equ 0A9A9A9A9A9A9A9A9h
E_OUTOFMEMORY equ 08007000Eh

.data
first_block     dq 0
refused_block   dq 0
full_block      dq 0

.code
native_entry PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog

    ; Step 1: 4 KiB at 2 KiB alignment succeeds, aligned and writable.
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
    mov dword ptr [rax+0FFCh], 5A17A0C1h
    cmp dword ptr [rax+0FFCh], 5A17A0C1h
    jne fail_1

    ; Step 2: refusal. 128 KiB does not fit; E_OUTOFMEMORY and the output is untouched.
    mov rax, SENTINEL
    mov qword ptr [refused_block], rax
    lea rcx, refused_block
    xor edx, edx
    mov r8d, 20000h
    mov r9d, 800h
    call qword ptr [__imp_ApuAlloc]
    cmp eax, E_OUTOFMEMORY
    jne fail_2
    mov rax, SENTINEL
    cmp qword ptr [refused_block], rax
    jne fail_2

    ; Step 3: the whole 64 KiB does not fit while the first block is live.
    mov rax, SENTINEL
    mov qword ptr [full_block], rax
    lea rcx, full_block
    xor edx, edx
    mov r8d, 10000h
    mov r9d, 800h
    call qword ptr [__imp_ApuAlloc]
    cmp eax, E_OUTOFMEMORY
    jne fail_3
    mov rax, SENTINEL
    cmp qword ptr [full_block], rax
    jne fail_3

    ; Step 4: release the first block, then the same 64 KiB request succeeds.
    mov rcx, qword ptr [first_block]
    call qword ptr [__imp_ApuFree]
    lea rcx, full_block
    xor edx, edx
    mov r8d, 10000h
    mov r9d, 800h
    call qword ptr [__imp_ApuAlloc]
    test eax, eax
    jnz fail_4
    mov rax, qword ptr [full_block]
    mov rdx, SENTINEL
    cmp rax, rdx
    je fail_4
    test rax, rax
    jz fail_4
    mov dword ptr [rax+0FFFCh], 0A0D10C0Dh
    cmp dword ptr [rax+0FFFCh], 0A0D10C0Dh
    jne fail_4
    mov rcx, rax
    call qword ptr [__imp_ApuFree]

    ; Step 5: this process's provider saw 4 allocations, 2 releases, nothing live.
    call qword ptr [__imp_get_apu_alloc_call_count]
    cmp eax, 4
    jne fail_5
    call qword ptr [__imp_get_apu_free_call_count]
    cmp eax, 2
    jne fail_5
    call qword ptr [__imp_get_apu_live_block_count]
    test eax, eax
    jnz fail_5
    call qword ptr [__imp_get_apu_rejected_free_count]
    test eax, eax
    jnz fail_5

    mov eax, 8A0h
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

done:
    add rsp, 28h
    ret
native_entry ENDP
END
