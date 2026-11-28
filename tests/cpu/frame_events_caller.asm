; Source-built caller for d3d12_xs.dll frame events (#874):
;   d3d12_xs.dll!SetFrameIntervalX
;   d3d12_xs.dll!ScheduleFrameEventX
;   d3d12_xs.dll!WaitFrameEventX
;
; Public call sequence from Microsoft Xbox-GDK-Samples (InGameStore/DeviceResources.cpp):
;   SetFrameIntervalX -> ScheduleFrameEventX(ORIGIN) once -> WaitFrameEventX(ORIGIN) each frame.
; Uses INFINITE wait and NULL event handle. (PresentX is a cited follow-up).
;
; The fixture exercises:
; 1. SetFrameIntervalX initialization
; 2. Unrepresented event refusal (fixture behavior: E_INVALIDARG 0x80070057, sentinel intact)
; 3. Timing check: bounded timeout before tick leaves sentinel intact (WAIT_TIMEOUT 0x00000102)
; 4. One ScheduleFrameEventX registration triggers delivery
; 5. First frame wait: blocks until Tick 1 is delivered, returning Token 1
; 6. Second frame wait: blocks until Tick 2 is delivered, returning advancing Token 2 (token > last_token)
; 7. ScheduleFrameEventX refusal for unrepresented event (E_INVALIDARG 0x80070057)
; 8. Clean exit with 0x874 (Refs #874)
EXTERN __imp_SetFrameIntervalX:QWORD
EXTERN __imp_ScheduleFrameEventX:QWORD
EXTERN __imp_WaitFrameEventX:QWORD
PUBLIC native_entry

.data
ALIGN 8
token           dq 0
last_token      dq 0

.code
native_entry PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog

    ; Step 1: SetFrameIntervalX(output=0, interval=60, period=1, flags=0)
    ; RCX = output (0, null output fixture choice)
    ; EDX = interval (60)
    ; R8D = period (1)
    ; R9D = flags (0)
    xor ecx, ecx
    mov edx, 3Ch
    mov r8d, 1
    xor r9d, r9d
    call qword ptr [__imp_SetFrameIntervalX]
    test eax, eax
    jnz fail_interval

    ; Step 2: Refusal control - unrepresented event rejection
    ; WaitFrameEventX(event_type=99, timeout_ms=1000, event_handle=0, flags=0, out_token=&token)
    ; Must return E_INVALIDARG (0x80070057) and preserve token sentinel.
    mov rax, 0DEADBEEFCAFEBABEh
    mov qword ptr [token], rax
    mov ecx, 63h        ; event_type = 99 (unrepresented event)
    mov edx, 3E8h       ; timeout = 1000 ms
    xor r8, r8          ; event_handle = NULL
    xor r9d, r9d        ; flags = 0
    lea rax, [token]
    mov qword ptr [rsp + 20h], rax
    call qword ptr [__imp_WaitFrameEventX]
    cmp eax, 80070057h  ; E_INVALIDARG
    jne fail_rejection_code
    mov rax, 0DEADBEEFCAFEBABEh
    cmp qword ptr [token], rax
    jne fail_rejection_clobber

    ; Step 3: Controlled tick timing check - bounded timeout before tick leaves sentinel intact
    ; WaitFrameEventX(event_type=0, timeout_ms=1, event_handle=0, flags=0, out_token=&token)
    ; Must return WAIT_TIMEOUT (0x00000102) and preserve token sentinel.
    mov rax, 1122334455667788h
    mov qword ptr [token], rax
    xor ecx, ecx        ; event_type = 0 (D3D12XBOX_FRAME_EVENT_ORIGIN)
    mov edx, 1          ; timeout = 1 ms (bounded wait before tick)
    xor r8, r8          ; event_handle = NULL
    xor r9d, r9d        ; flags = 0
    lea rax, [token]
    mov qword ptr [rsp + 20h], rax
    call qword ptr [__imp_WaitFrameEventX]
    cmp eax, 102h       ; WAIT_TIMEOUT
    jne fail_timeout_code
    mov rax, 1122334455667788h
    cmp qword ptr [token], rax
    jne fail_timeout_clobber

    ; Step 4: One ScheduleFrameEventX(ORIGIN) registration per Microsoft sample
    ; ScheduleFrameEventX(0, 0, NULL, 0)
    xor ecx, ecx        ; 0 (D3D12XBOX_FRAME_EVENT_ORIGIN)
    xor edx, edx        ; offset = 0
    xor r8, r8          ; event_handle = NULL
    xor r9d, r9d        ; flags = 0
    call qword ptr [__imp_ScheduleFrameEventX]
    test eax, eax
    jnz fail_sched

    ; Step 5: First frame wait - blocks until Tick 1 is delivered
    ; WaitFrameEventX(ORIGIN, 1000 ms, NULL, 0, &token)
    mov qword ptr [token], 0
    xor ecx, ecx        ; event_type = 0
    mov edx, 3E8h       ; timeout = 1000 ms (bounded wait covering tick delivery)
    xor r8, r8          ; event_handle = NULL
    xor r9d, r9d        ; flags = 0
    lea rax, [token]
    mov qword ptr [rsp + 20h], rax
    call qword ptr [__imp_WaitFrameEventX]
    test eax, eax
    jnz fail_origin1
    mov rax, qword ptr [token]
    test rax, rax       ; token must be delivered (> 0)
    jz fail_origin1_token
    mov qword ptr [last_token], rax

    ; Step 6: Second frame wait without re-scheduling - blocks until Tick 2 is delivered
    mov qword ptr [token], 0
    xor ecx, ecx
    mov edx, 3E8h       ; timeout = 1000 ms
    xor r8, r8
    xor r9d, r9d
    lea rax, [token]
    mov qword ptr [rsp + 20h], rax
    call qword ptr [__imp_WaitFrameEventX]
    test eax, eax
    jnz fail_origin2
    mov rax, qword ptr [token]
    cmp rax, qword ptr [last_token]
    jbe fail_monotonic  ; monotonic token advancement: token > last_token

    ; Step 7: ScheduleFrameEventX refusal for unrepresented event (77)
    mov ecx, 4Dh        ; 77
    xor edx, edx
    xor r8, r8
    xor r9d, r9d
    call qword ptr [__imp_ScheduleFrameEventX]
    cmp eax, 80070057h  ; E_INVALIDARG
    jne fail_sched_reject

    ; Step 8: Success exit code 0x874 (Refs #874)
    mov eax, 874h
    add rsp, 38h
    ret

fail_interval:
    mov eax, 0E01h
    add rsp, 38h
    ret

fail_rejection_code:
    mov eax, 0E02h
    add rsp, 38h
    ret

fail_rejection_clobber:
    mov eax, 0E03h      ; specific exit code for sentinel clobber fault
    add rsp, 38h
    ret

fail_timeout_code:
    mov eax, 0E04h      ; specific exit code for early release fault
    add rsp, 38h
    ret

fail_timeout_clobber:
    mov eax, 0E05h
    add rsp, 38h
    ret

fail_sched:
    mov eax, 0E06h
    add rsp, 38h
    ret

fail_origin1:
    mov eax, 0E07h
    add rsp, 38h
    ret

fail_origin1_token:
    mov eax, 0E08h
    add rsp, 38h
    ret

fail_origin2:
    mov eax, 0E09h
    add rsp, 38h
    ret

fail_monotonic:
    mov eax, 0E0Ah
    add rsp, 38h
    ret

fail_sched_reject:
    mov eax, 0E0Bh
    add rsp, 38h
    ret

native_entry ENDP
END
