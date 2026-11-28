; A source-built EXE that imports XMemAllocDefault and XMemFreeDefault by name from the desktop
; xmem.dll and starts through exe_startup.asm, so the same body runs in a normal Windows process and
; from pe_map_run. Synthetic attribute words only. The body returns 875h when every step passes,
; otherwise 0F00h + the number of the first failing step.
;   Heap block (05E50000h: heap, 32-byte alignment)
;    1  XMemAllocDefault(257, 05E50000h) returns a block
;    2  the block is 32-byte aligned
;    3  all 257 bytes read as zero
;    4  the block takes a 0A5h fill through its last byte
;    5  ownership: heap_block_busy reports a busy heap entry containing the block
;   Graphics block (0A8004000h: graphics type 5, 64 KiB pages, 256-byte alignment)
;    6  XMemAllocDefault(4096, 0A8004000h) returns a block
;    7  the block is 64 KiB aligned
;    8  all 4096 bytes read as zero
;    9  the block takes a write to its last byte
;   Rejected inputs (the attribute word or size is refused before any allocator runs)
;   10  reserved bit 33 returns null
;   11  a zero-byte request returns null
;   12  page size 3 (undefined) returns null
;   13  memory type 15 (undefined) returns null
;   14  the last-error value is still the 0 set before step 10 (observed desktop behavior)
;   Induced allocator failure (a valid word with a size the system allocator cannot satisfy)
;   15  XMemAllocDefault(2^46, 0A8004000h) returns null
;   16  ...and the last-error value is nonzero (VirtualAlloc's failure code)
;   17  XMemAllocDefault(2^46, 05E50000h) returns null
;   18  control: the graphics word with 4096 bytes succeeds again, and that block is freed
;   Release
;   19  after XMemFreeDefault on both blocks (and on null), VirtualQuery describes the graphics address
;   20  the graphics address is MEM_FREE
;   21  ownership: no busy heap entry contains the heap block any more
; Negative controls: SKIP_GPU_FREE (expected 0F14h), SKIP_HEAP_FREE (expected 0F15h) and UNINDUCED,
; whose step 15 asks for 4096 bytes (expected 0F0Fh).
EXTERN __imp_XMemAllocDefault:QWORD
EXTERN __imp_XMemFreeDefault:QWORD
EXTERN __imp_VirtualQuery:QWORD
EXTERN __imp_SetLastError:QWORD
EXTERN __imp_GetLastError:QWORD
EXTERN __imp_heap_block_busy:QWORD

BINK_WORD    EQU 05E50000h
AMD_WORD     EQU 0A8004000h
MEM_FREE     EQU 10000h
MBI_BYTES    EQU 48                 ; sizeof(MEMORY_BASIC_INFORMATION) on x64
MBI_STATE    EQU 32                 ; offsetof(MEMORY_BASIC_INFORMATION, State)

.code
native_entry PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
    push r12
    .pushreg r12
    sub rsp, 88                     ; 32 bytes of home space, then the MEMORY_BASIC_INFORMATION at rsp+32
    .allocstack 88
    .endprolog

    mov r12d, 1                     ; step 1
    mov ecx, 257
    mov edx, BINK_WORD
    call QWORD PTR __imp_XMemAllocDefault
    mov rbx, rax
    test rax, rax
    jz failed
    inc r12d                        ; step 2
    test bl, 31
    jnz failed
    inc r12d                        ; step 3
    mov rdi, rbx
    mov ecx, 257
    xor eax, eax
    repe scasb
    jnz failed
    inc r12d                        ; step 4
    mov rdi, rbx
    mov ecx, 257
    mov al, 0A5h
    rep stosb
    cmp BYTE PTR [rbx+256], 0A5h
    jne failed
    inc r12d                        ; step 5
    mov rcx, rbx
    call QWORD PTR __imp_heap_block_busy
    cmp eax, 1
    jne failed

    inc r12d                        ; step 6
    mov ecx, 4096
    mov edx, AMD_WORD
    call QWORD PTR __imp_XMemAllocDefault
    mov rsi, rax
    test rax, rax
    jz failed
    inc r12d                        ; step 7
    test si, 0FFFFh
    jnz failed
    inc r12d                        ; step 8
    mov rdi, rsi
    mov ecx, 512
    xor eax, eax
    repe scasq
    jnz failed
    inc r12d                        ; step 9
    mov BYTE PTR [rsi+4095], 5Ah
    cmp BYTE PTR [rsi+4095], 5Ah
    jne failed

    xor ecx, ecx
    call QWORD PTR __imp_SetLastError
    inc r12d                        ; step 10
    mov ecx, 64
    mov rdx, 205E50000h             ; BINK_WORD with reserved bit 33
    call QWORD PTR __imp_XMemAllocDefault
    test rax, rax
    jnz failed
    inc r12d                        ; step 11
    xor ecx, ecx
    mov edx, BINK_WORD
    call QWORD PTR __imp_XMemAllocDefault
    test rax, rax
    jnz failed
    inc r12d                        ; step 12
    mov ecx, 64
    mov edx, BINK_WORD OR 0C000h    ; page size 3
    call QWORD PTR __imp_XMemAllocDefault
    test rax, rax
    jnz failed
    inc r12d                        ; step 13
    mov ecx, 512
    mov rdx, 1E8004000h             ; AMD_WORD with memory type 15
    call QWORD PTR __imp_XMemAllocDefault
    test rax, rax
    jnz failed
    inc r12d                        ; step 14
    call QWORD PTR __imp_GetLastError
    test eax, eax
    jnz failed

    inc r12d                        ; step 15
IFDEF UNINDUCED
    mov ecx, 4096
ELSE
    mov rcx, 400000000000h          ; 2^46 bytes (64 TiB)
ENDIF
    mov edx, AMD_WORD
    call QWORD PTR __imp_XMemAllocDefault
    test rax, rax
    jnz failed
    inc r12d                        ; step 16
    call QWORD PTR __imp_GetLastError
    test eax, eax
    jz failed
    inc r12d                        ; step 17
    mov rcx, 400000000000h
    mov edx, BINK_WORD
    call QWORD PTR __imp_XMemAllocDefault
    test rax, rax
    jnz failed
    inc r12d                        ; step 18
    mov ecx, 4096
    mov edx, AMD_WORD
    call QWORD PTR __imp_XMemAllocDefault
    test rax, rax
    jz failed
    mov rcx, rax
    mov edx, AMD_WORD
    call QWORD PTR __imp_XMemFreeDefault

    inc r12d                        ; step 19
IFNDEF SKIP_GPU_FREE
    mov rcx, rsi
    mov edx, AMD_WORD
    call QWORD PTR __imp_XMemFreeDefault
ENDIF
IFNDEF SKIP_HEAP_FREE
    mov rcx, rbx
    mov edx, BINK_WORD
    call QWORD PTR __imp_XMemFreeDefault
ENDIF
    xor ecx, ecx
    mov edx, BINK_WORD
    call QWORD PTR __imp_XMemFreeDefault
    mov rcx, rsi
    lea rdx, [rsp+32]
    mov r8d, MBI_BYTES
    call QWORD PTR __imp_VirtualQuery
    cmp rax, MBI_BYTES
    jne failed
    inc r12d                        ; step 20
    cmp DWORD PTR [rsp+32+MBI_STATE], MEM_FREE
    jne failed
    inc r12d                        ; step 21
    mov rcx, rbx
    call QWORD PTR __imp_heap_block_busy
    test eax, eax
    jnz failed
    mov eax, 875h
    jmp finished
failed:
    lea eax, [r12+0F00h]
finished:
    add rsp, 88
    pop r12
    pop rdi
    pop rsi
    pop rbx
    ret
native_entry ENDP
END
