; Stack frames in functions that make no call, for the jump-table translator.
; Every export takes RCX, RDX and R8 = address of three writable QWORDs. The register probe seeds
; RBX and RSI before each call and samples them after it.
.code
; PUSH and POP of two nonvolatile registers used as scratch around a loop, as MSVC /O2 emits them.
; RBX = RCX, then (RDX & 31) times RBX = RBX * 3 + RDX. [R8] = RAX = RBX.
push_loop PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    .endprolog
    mov rbx, rcx
    mov esi, edx
    and esi, 31
    jz loop_done
loop_add:
    lea rbx, [rbx+rbx*2]
    add rbx, rdx
    dec esi
    jnz loop_add
loop_done:
    mov QWORD PTR [r8], rbx
    mov rax, rbx
    pop rsi
    pop rbx
    ret
push_loop ENDP

; SUB RSP with spills and reloads of 8, 4, 2 and 1 bytes, partial-overlap reads, a read-modify-write,
; SETcc into the frame and CMOVcc from it.
spill_widths PROC FRAME
    sub rsp, 18h
    .allocstack 18h
    .endprolog
    mov QWORD PTR [rsp], rcx
    mov DWORD PTR [rsp+8], edx
    mov WORD PTR [rsp+0Ch], dx
    mov BYTE PTR [rsp+0Eh], cl
    mov BYTE PTR [rsp+0Fh], 5Ah
    ; Stored as 8 bytes, read as the upper 4.
    mov eax, DWORD PTR [rsp+4]
    ; Two bytes across the 8-byte and the 4-byte store.
    movzx r9d, WORD PTR [rsp+7]
    ; Stored as 4 + 2 + 1 + 1 bytes, read as 8.
    mov r10, QWORD PTR [rsp+8]
    add QWORD PTR [rsp], rdx
    cmp rcx, rdx
    setb BYTE PTR [rsp+8]
    mov r11, QWORD PTR [rsp]
    cmp ecx, edx
    cmovl rax, QWORD PTR [rsp+8]
    mov QWORD PTR [r8], r10
    mov QWORD PTR [r8+8], r9
    mov QWORD PTR [r8+10h], r11
    add rsp, 18h
    ret
spill_widths ENDP

; The MSVC /O2 shape: RBX and RSI saved to home space before PUSH RDI and the allocation, restored
; with MOV before ADD RSP on each of two returns. The unwind codes record the saves at the end of
; the prologue with offsets from the final RSP, as MSVC writes them.
home_save PROC FRAME
    mov QWORD PTR [rsp+8], rbx
    mov QWORD PTR [rsp+10h], rsi
    push rdi
    .pushreg rdi
    sub rsp, 20h
    .allocstack 20h
    .savereg rbx, 30h
    .savereg rsi, 38h
    .endprolog
    mov rbx, rcx
    mov rsi, rdx
    lea rdi, [rcx+rdx]
    test rdx, rdx
    jz no_mix
    xor rbx, rsi
    mov QWORD PTR [rsp], rbx
    imul rdi, QWORD PTR [rsp]
    mov QWORD PTR [r8], rdi
    mov rax, rdi
    mov rbx, QWORD PTR [rsp+30h]
    mov rsi, QWORD PTR [rsp+38h]
    add rsp, 20h
    pop rdi
    ret
no_mix:
    mov QWORD PTR [r8+8], rdi
    mov rbx, QWORD PTR [rsp+30h]
    mov eax, edi
    mov rsi, QWORD PTR [rsp+38h]
    add rsp, 20h
    pop rdi
    ret
home_save ENDP

; The MSVC /Od shape, copied from cl /Od output for
;   u64 pick_max(u64 x, u64 y, u64 *trace) { u64 m; if (x < y) m = y; else m = x; trace[0] = m; return m + x; }
; The arguments are stored to home space and reloaded; the local m is written on both paths.
home_args PROC FRAME
    mov QWORD PTR [rsp+18h], r8
    mov QWORD PTR [rsp+10h], rdx
    mov QWORD PTR [rsp+8], rcx
    sub rsp, 18h
    .allocstack 18h
    .endprolog
    mov rax, QWORD PTR [rsp+28h]
    cmp QWORD PTR [rsp+20h], rax
    jae keep_x
    mov rax, QWORD PTR [rsp+28h]
    mov QWORD PTR [rsp], rax
    jmp stored
keep_x:
    mov rax, QWORD PTR [rsp+20h]
    mov QWORD PTR [rsp], rax
stored:
    mov eax, 8
    imul rax, rax, 0
    mov rcx, QWORD PTR [rsp+30h]
    mov rdx, QWORD PTR [rsp]
    mov QWORD PTR [rcx+rax], rdx
    mov rax, QWORD PTR [rsp+20h]
    mov rcx, QWORD PTR [rsp]
    add rcx, rax
    mov rax, rcx
    add rsp, 18h
    ret
home_args ENDP

; Two paths push different registers and rejoin at the same depth; a third path returns early
; and restores RBX on the way out. These pushes after the prologue are not described by unwind
; codes, which only matters to an unwinder.
two_paths PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    mov rbx, rdx
    test cl, 1
    jnz odd_path
    push rsi
    lea rsi, [rcx+rbx]
    mov QWORD PTR [r8], rsi
    pop rsi
    jmp joined
odd_path:
    push rdi
    mov rdi, rcx
    sub rdi, rbx
    mov QWORD PTR [r8+8], rdi
    pop rdi
joined:
    mov rax, rbx
    cmp rcx, rdx
    jb early
    xor rax, rcx
    mov QWORD PTR [r8+10h], rax
    pop rbx
    ret
early:
    pop rbx
    ret
two_paths ENDP

; RCX and RDX spilled to home space with no allocation of its own. RAX = [R8] = RCX + RDX, [R8+8] = RCX.
home_spill PROC FRAME
    .endprolog
    mov QWORD PTR [rsp+8], rcx
    mov QWORD PTR [rsp+10h], rdx
    mov rcx, r8
    mov rax, QWORD PTR [rsp+10h]
    add rax, QWORD PTR [rsp+8]
    mov QWORD PTR [rcx], rax
    mov rdx, QWORD PTR [rsp+8]
    mov QWORD PTR [rcx+8], rdx
    ret
home_spill ENDP

; A counter and an accumulator kept in the frame across a loop's back edge: (RDX & 15) times
; RAX = RAX * 3 + RCX from 0. [R8] = the final counter, [R8+8] = RBX = RDX & 15.
loop_counter PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 10h
    .allocstack 10h
    .endprolog
    mov DWORD PTR [rsp], 0
    mov QWORD PTR [rsp+8], 0
    mov ebx, edx
    and ebx, 15
count_again:
    cmp DWORD PTR [rsp], ebx
    jae count_done
    mov rax, QWORD PTR [rsp+8]
    lea rax, [rax+rax*2]
    add rax, rcx
    mov QWORD PTR [rsp+8], rax
    inc DWORD PTR [rsp]
    jmp count_again
count_done:
    mov rax, QWORD PTR [rsp+8]
    mov ecx, DWORD PTR [rsp]
    mov QWORD PTR [r8], rcx
    mov QWORD PTR [r8+8], rbx
    add rsp, 10h
    pop rbx
    ret
loop_counter ENDP

; Translated and inspected, never called: its frame array is exactly the 4096-byte bound.
frame_bound PROC FRAME
    sub rsp, 0FD8h
    .allocstack 0FD8h
    .endprolog
    mov QWORD PTR [rsp], rcx
    mov rax, QWORD PTR [rsp]
    add rsp, 0FD8h
    ret
frame_bound ENDP

; Releases its allocation and allocates the same bytes again, then writes the slot before reading it.
; RAX = RCX + RDX, stored to the first trace word.
realloc_slot PROC FRAME
    sub rsp, 18h
    .allocstack 18h
    .endprolog
    mov QWORD PTR [rsp], rcx
    mov rax, QWORD PTR [rsp]
    add rsp, 18h
    sub rsp, 18h
    mov QWORD PTR [rsp], rdx
    add rax, QWORD PTR [rsp]
    mov QWORD PTR [r8], rax
    add rsp, 18h
    ret
realloc_slot ENDP

; Rejected: the slot that holds the saved RBX is released by ADD RSP and allocated again, so POP RBX
; no longer restores a value the function may rely on.
stale_save PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    add rsp, 8
    sub rsp, 8
    pop rbx
    ret
stale_save ENDP

; Rejected: POP RSI reads the stored RCX and releases its slot; the slot is allocated again and read
; without a new store.
stale_pop PROC FRAME
    push rsi
    .pushreg rsi
    sub rsp, 8
    .allocstack 8
    .endprolog
    mov QWORD PTR [rsp], rcx
    pop rsi
    sub rsp, 8
    mov rax, QWORD PTR [rsp]
    add rsp, 8
    pop rsi
    ret
stale_pop ENDP

; Rejected: RAX holds a guest address (a RIP-relative LEA) that is spilled and reloaded.
guest_spill PROC FRAME
    sub rsp, 8
    .allocstack 8
    .endprolog
    lea rax, guest_spill
    mov QWORD PTR [rsp], rax
    mov rax, QWORD PTR [rsp]
    add rsp, 8
    ret
guest_spill ENDP

; Rejected: RBX saved twice, then reloaded from the 8 bytes that straddle the two copies, which hold
; its upper half followed by its lower half.
split_restore PROC FRAME
    mov QWORD PTR [rsp+8], rbx
    mov QWORD PTR [rsp+10h], rbx
    .endprolog
    mov rbx, rcx
    lea rax, [rbx+1]
    mov rbx, QWORD PTR [rsp+0Ch]
    ret
split_restore ENDP

; Rejected: after MOV BL, CL the upper seven bytes of RBX are still the caller's, so the second PUSH
; stores them as data and MOV RAX, [RSP] would return them.
partial_push PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    mov bl, cl
    push rbx
    mov rax, QWORD PTR [rsp]
    add rsp, 8
    pop rbx
    ret
partial_push ENDP

; Rejected: after POP RBX restores the caller's value, MOV RAX, RBX would return it.
reuse_restored PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    mov rbx, rcx
    mov QWORD PTR [r8], rbx
    pop rbx
    mov rax, rbx
    ret
reuse_restored ENDP

anchored_frame PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    sub rsp, 20h
    .allocstack 20h
    .setframe rbp, 20h
    .endprolog
    mov QWORD PTR [rbp-8], rcx
    test dl, 1
    jz anchored_join
    lea rbp, [rsp+20h]
    add QWORD PTR [rsp+18h], rdx
anchored_join:
    mov rax, QWORD PTR [rbp-8]
    mov QWORD PTR [r8], rax
    add rsp, 20h
    pop rbp
    ret
anchored_frame ENDP
END
