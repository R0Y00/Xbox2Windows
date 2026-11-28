; A source-built EXE that imports XMemAllocatePhysicalPages and XMemFreePhysicalPages by name from
; xmem.dll and starts through exe_startup.asm, so the same body runs in a normal Windows process and
; from pe_map_run. Synthetic inputs only. The body returns 880h when every step passes, otherwise
; 0F00h + the number of the first failing step:
;    1  XMemAllocatePhysicalPages(MEM_64K_PAGES, &count = 2, pages) returns TRUE
;    2  the count is still 2
;    3  both page numbers are nonzero and distinct, and the canary after them is untouched
;    4  refused release: count 1 for the two-page grant returns FALSE (proposed desktop policy)
;    5  the same grant still releases with count 2 (the refusal changed nothing)
;    6  double free: the same release again returns FALSE
;    7  refused allocation: MEM_LARGE_PAGES alone returns FALSE (proposed desktop policy)
;    8  ...and sets the count to 0 (proposed desktop policy)
;    9  ...and leaves all three canaries in the page array untouched (proposed desktop policy)
;   10  XMemFreePhysicalPages(1, NULL) returns FALSE
; Negative controls: FULL_COUNT releases with count 2 at step 4 (expected 0F04h); SKIP_RELEASE
; omits step 5, so the "double free" at step 6 releases the live grant (expected 0F06h).
EXTERN __imp_XMemAllocatePhysicalPages:QWORD
EXTERN __imp_XMemFreePhysicalPages:QWORD

MEM_64K_PAGES   EQU 20400000h
MEM_LARGE_PAGES EQU 20000000h
CANARY0         EQU 1111111111111111h
CANARY1         EQU 2222222222222222h
CANARY2         EQU 3333333333333333h

.data
page_count  DQ 0
pages       DQ 3 DUP (0)

.code
; Fills the three page slots with distinct canaries.
fill_canaries PROC
    mov rax, CANARY0
    mov pages, rax
    mov rax, CANARY1
    mov QWORD PTR pages+8, rax
    mov rax, CANARY2
    mov QWORD PTR pages+16, rax
    ret
fill_canaries ENDP

native_entry PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 32
    .allocstack 32
    .endprolog

    call fill_canaries
    mov ebx, 1                      ; step 1
    mov page_count, 2
    mov ecx, MEM_64K_PAGES
    lea rdx, page_count
    lea r8, pages
    call QWORD PTR __imp_XMemAllocatePhysicalPages
    test eax, eax
    jz failed
    inc ebx                         ; step 2
    cmp page_count, 2
    jne failed
    inc ebx                         ; step 3
    mov rax, pages
    mov rdx, QWORD PTR pages+8
    test rax, rax
    jz failed
    test rdx, rdx
    jz failed
    cmp rax, rdx
    je failed
    mov rax, CANARY2
    cmp QWORD PTR pages+16, rax
    jne failed

    inc ebx                         ; step 4
IFDEF FULL_COUNT
    mov ecx, 2
ELSE
    mov ecx, 1
ENDIF
    lea rdx, pages
    call QWORD PTR __imp_XMemFreePhysicalPages
    test eax, eax
    jnz failed
    inc ebx                         ; step 5
IFNDEF SKIP_RELEASE
    mov ecx, 2
    lea rdx, pages
    call QWORD PTR __imp_XMemFreePhysicalPages
    test eax, eax
    jz failed
ENDIF
    inc ebx                         ; step 6
    mov ecx, 2
    lea rdx, pages
    call QWORD PTR __imp_XMemFreePhysicalPages
    test eax, eax
    jnz failed

    inc ebx                         ; step 7
    call fill_canaries
    mov page_count, 3
    mov ecx, MEM_LARGE_PAGES
    lea rdx, page_count
    lea r8, pages
    call QWORD PTR __imp_XMemAllocatePhysicalPages
    test eax, eax
    jnz failed
    inc ebx                         ; step 8
    cmp page_count, 0
    jne failed
    inc ebx                         ; step 9
    mov rax, CANARY0
    cmp pages, rax
    jne failed
    mov rax, CANARY1
    cmp QWORD PTR pages+8, rax
    jne failed
    mov rax, CANARY2
    cmp QWORD PTR pages+16, rax
    jne failed

    inc ebx                         ; step 10
    mov ecx, 1
    xor edx, edx
    call QWORD PTR __imp_XMemFreePhysicalPages
    test eax, eax
    jnz failed
    mov eax, 880h
    jmp finished
failed:
    lea eax, [rbx+0F00h]
finished:
    add rsp, 32
    pop rbx
    ret
native_entry ENDP
END
