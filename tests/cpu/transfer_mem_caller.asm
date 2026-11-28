EXTERN __imp_XMemTransferMemory:QWORD
EXTERN __imp_get_total_call_count:QWORD
EXTERN __imp_get_transfer_call_count:QWORD
EXTERN __imp_get_argument_rejection_count:QWORD
EXTERN __imp_get_recorded_call_partition:QWORD
EXTERN __imp_get_recorded_call_bytes:QWORD
EXTERN __imp_get_recorded_call_result:QWORD
PUBLIC native_entry

.code
native_entry PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog

    ; Step 1: Valid 2 MiB transfer under test-provider policy partition 1
    mov ecx, 1
    mov rdx, 0200000h
    call qword ptr [__imp_XMemTransferMemory]
    cmp eax, 1
    jne fail_step_1

    ; Step 2: Valid 16 MiB transfer under test-provider policy partition 1
    mov ecx, 1
    mov rdx, 01000000h
    call qword ptr [__imp_XMemTransferMemory]
    cmp eax, 1
    jne fail_step_2

    ; Step 3: Refusal control - invalid partition (partition 0)
    xor ecx, ecx
    mov rdx, 0200000h
    call qword ptr [__imp_XMemTransferMemory]
    test eax, eax
    jnz fail_step_3

    ; Step 4: Refusal control - zero bytes
    mov ecx, 1
    xor edx, edx
    call qword ptr [__imp_XMemTransferMemory]
    test eax, eax
    jnz fail_step_4

    ; Step 5: Refusal control - unaligned bytes (not 2 MiB aligned)
    mov ecx, 1
    mov rdx, 0200100h
    call qword ptr [__imp_XMemTransferMemory]
    test eax, eax
    jnz fail_step_5

    ; Step 6: Refusal control - oversized bytes (exceeds 16 MiB policy maximum)
    mov ecx, 1
    mov rdx, 02000000h
    call qword ptr [__imp_XMemTransferMemory]
    test eax, eax
    jnz fail_step_6

    ; Step 7: Verify provider counters inside caller process
    call qword ptr [__imp_get_total_call_count]
    cmp eax, 6
    jne fail_counters

    call qword ptr [__imp_get_transfer_call_count]
    cmp eax, 2
    jne fail_counters

    call qword ptr [__imp_get_argument_rejection_count]
    cmp eax, 4
    jne fail_counters

    ; Verify call 0 ordered arguments: partition 1, 2 MiB (0x200000), result 1
    xor ecx, ecx
    call qword ptr [__imp_get_recorded_call_partition]
    cmp eax, 1
    jne fail_trace
    xor ecx, ecx
    call qword ptr [__imp_get_recorded_call_bytes]
    cmp rax, 0200000h
    jne fail_trace
    xor ecx, ecx
    call qword ptr [__imp_get_recorded_call_result]
    cmp eax, 1
    jne fail_trace

    ; Verify call 1 ordered arguments: partition 1, 16 MiB (0x1000000), result 1
    mov ecx, 1
    call qword ptr [__imp_get_recorded_call_partition]
    cmp eax, 1
    jne fail_trace
    mov ecx, 1
    call qword ptr [__imp_get_recorded_call_bytes]
    cmp rax, 01000000h
    jne fail_trace
    mov ecx, 1
    call qword ptr [__imp_get_recorded_call_result]
    cmp eax, 1
    jne fail_trace

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
fail_counters:
    mov eax, 0F0Ah
    jmp done
fail_trace:
    mov eax, 0F0Bh
    jmp done

done:
    add rsp, 28h
    ret
native_entry ENDP
END
