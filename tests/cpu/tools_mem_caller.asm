EXTERN __imp_XMemIsToolsMemAvailable:QWORD
EXTERN __imp_get_tools_mem_call_count:QWORD

.code
native_entry PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog

    call qword ptr [__imp_XMemIsToolsMemAvailable]
    mov dword ptr [rsp+20h], eax
    call qword ptr [__imp_XMemIsToolsMemAvailable]
    cmp eax, dword ptr [rsp+20h]
    jne fail_mismatch

    ; Each process checks its own provider trace before choosing the path.
    call qword ptr [__imp_get_tools_mem_call_count]
    cmp eax, 2
    jne fail_call_count
    mov eax, dword ptr [rsp+20h]

    ; 4. Branch based on BOOL outcome:
    ; If 0 (FALSE): desktop without a tools partition fallback
    test eax, eax
    jz desktop_no_tools

    ; If 1 (TRUE): console devkit with a tools partition
    cmp eax, 1
    jne fail_invalid_bool

    ; Outcome: TRUE -> devkit mode, return 84
    mov eax, 84
    jmp done

desktop_no_tools:
    ; Outcome: FALSE -> desktop mode, return 42
    mov eax, 42
    jmp done

fail_mismatch:
    mov eax, 1
    jmp done

fail_invalid_bool:
    mov eax, 2
    jmp done

fail_call_count:
    mov eax, 3

done:
    add rsp, 28h
    ret
native_entry ENDP
END
