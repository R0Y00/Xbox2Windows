EXTERN __imp_XMemMapPhysicalPages:QWORD
EXTERN __imp_XMemMapPhysicalPagesScatter:QWORD
EXTERN __imp_get_map_pages_call_count:QWORD
EXTERN __imp_get_map_pages_unmap_count:QWORD
EXTERN __imp_get_scatter_map_call_count:QWORD
EXTERN __imp_get_scatter_unmap_count:QWORD
EXTERN __imp_get_argument_mismatch_count:QWORD
PUBLIC native_entry

.data
aligned_addr        dq 010000000h
unaligned_addr      dq 010001234h
pages               dq 1, 2
wrong_pages         dq 999, 888
addresses           dq 010000000h, 010010000h
swapped_addresses   dq 010010000h, 010000000h

.code
native_entry PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog

    ; Step 1: Valid contiguous mapping
    mov rcx, qword ptr [aligned_addr]
    mov rdx, 2
    lea r8, pages
    call qword ptr [__imp_XMemMapPhysicalPages]
    cmp rax, qword ptr [aligned_addr]
    jne fail_step_1

    ; Step 2: Valid contiguous unmapping (NULL page array)
    mov rcx, qword ptr [aligned_addr]
    mov rdx, 2
    xor r8d, r8d
    call qword ptr [__imp_XMemMapPhysicalPages]
    cmp rax, qword ptr [aligned_addr]
    jne fail_step_2

    ; Step 3: Valid scatter mapping
    lea rcx, addresses
    mov edx, 2
    lea r8, pages
    call qword ptr [__imp_XMemMapPhysicalPagesScatter]
    cmp eax, 1
    jne fail_step_3

    ; Step 4: Valid scatter unmapping (NULL page array)
    lea rcx, addresses
    mov edx, 2
    xor r8d, r8d
    call qword ptr [__imp_XMemMapPhysicalPagesScatter]
    cmp eax, 1
    jne fail_step_4

    ; Step 5: Negative control - unaligned virtual address
    mov rcx, qword ptr [unaligned_addr]
    mov rdx, 2
    lea r8, pages
    call qword ptr [__imp_XMemMapPhysicalPages]
    test rax, rax
    jnz fail_step_5

    ; Step 6: Negative control - null virtual address
    xor ecx, ecx
    mov rdx, 2
    lea r8, pages
    call qword ptr [__imp_XMemMapPhysicalPages]
    test rax, rax
    jnz fail_step_6

    ; Step 7: Negative control - null scatter address array
    xor ecx, ecx
    mov edx, 2
    lea r8, pages
    call qword ptr [__imp_XMemMapPhysicalPagesScatter]
    test eax, eax
    jnz fail_step_7

    ; Step 8: Changed-argument control - wrong page IDs must be refused
    mov rcx, qword ptr [aligned_addr]
    mov rdx, 2
    lea r8, wrong_pages
    call qword ptr [__imp_XMemMapPhysicalPages]
    test rax, rax
    jnz fail_step_8

    ; Step 9: Changed-argument control - swapped scatter destinations must be refused
    lea rcx, swapped_addresses
    mov edx, 2
    lea r8, pages
    call qword ptr [__imp_XMemMapPhysicalPagesScatter]
    test eax, eax
    jnz fail_step_9

    ; Step 10: Verify provider call counters and mismatch counts directly in caller process
    call qword ptr [__imp_get_map_pages_call_count]
    cmp eax, 1
    jne fail_counters

    call qword ptr [__imp_get_map_pages_unmap_count]
    cmp eax, 1
    jne fail_counters

    call qword ptr [__imp_get_scatter_map_call_count]
    cmp eax, 1
    jne fail_counters

    call qword ptr [__imp_get_scatter_unmap_count]
    cmp eax, 1
    jne fail_counters

    call qword ptr [__imp_get_argument_mismatch_count]
    cmp eax, 2
    jne fail_counters

    ; All checks passed: return milestone status 0x870
    mov eax, 0870h
    jmp done

fail_step_1:
    mov eax, 0F01h
    jmp done
fail_step_2:
    mov eax, 0F02h
    jmp done
fail_step_3:
    mov eax, 0F03h
    jmp done
fail_step_4:
    mov eax, 0F04h
    jmp done
fail_step_5:
    mov eax, 0F05h
    jmp done
fail_step_6:
    mov eax, 0F06h
    jmp done
fail_step_7:
    mov eax, 0F07h
    jmp done
fail_step_8:
    mov eax, 0F08h
    jmp done
fail_step_9:
    mov eax, 0F09h
    jmp done
fail_counters:
    mov eax, 0F0Ah
    jmp done

done:
    add rsp, 28h
    ret
native_entry ENDP
END
