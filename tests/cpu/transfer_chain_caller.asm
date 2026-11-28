EXTERN __imp_XMemIsToolsMemAvailable:QWORD
EXTERN __imp_XMemGetWorkingSetStatistics:QWORD
EXTERN __imp_XMemTransferMemory:QWORD
PUBLIC native_entry

.data
; 128-byte statistics buffer for XMemGetWorkingSetStatistics
stats_buffer db 128 dup(0)

.code
native_entry PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog

    ; 1. Prerequisite query: XMemIsToolsMemAvailable
    call qword ptr [__imp_XMemIsToolsMemAvailable]
    test eax, eax
    jz blocked_by_tools_mem

    ; 2. Working-set statistics query with selector 0x2000000a
    mov ecx, 2000000Ah
    lea rdx, stats_buffer
    call qword ptr [__imp_XMemGetWorkingSetStatistics]
    test eax, eax
    jz working_set_failed

    ; 3. Evaluate available budget: total at +0x10, used at +0x18
    mov rax, qword ptr [stats_buffer + 10h]
    sub rax, qword ptr [stats_buffer + 18h]
    test rax, rax
    jle no_budget_available

    ; 4. Round up to 2 MiB boundary, clamp to 16 MiB maximum
    mov edx, 1000000h
    add rax, 1FFFFFh
    and rax, 0FFFFFFFFFFE00000h
    cmp rax, 1000000h
    cmovb rdx, rax

    ; 5. Transfer memory under partition 1
    mov ecx, 1
    call qword ptr [__imp_XMemTransferMemory]
    test eax, eax
    jz transfer_failed

    ; Success path reached
    mov eax, 0870h
    jmp done

blocked_by_tools_mem:
    mov eax, 0101h
    jmp done

working_set_failed:
    mov eax, 0102h
    jmp done

no_budget_available:
    mov eax, 0103h
    jmp done

transfer_failed:
    mov eax, 0104h
    jmp done

done:
    add rsp, 28h
    ret
native_entry ENDP
END
