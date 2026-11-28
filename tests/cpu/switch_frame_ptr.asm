; coverage: source cpu_switch_frame_ptr
; Source-owned differential fixture and negative controls for bounded frame-slot pointers
; passed to non-recursive internal callees in tools/cpu_switch_aot.py.
; Every differential export takes RCX = x, RDX = y, and R8 = address of three writable QWORDs.
.code

; --- Internal callees (not exported) ---

; write_u32_out(uint32_t* out, uint64_t val): write-only 4-byte out-param in RCX.
write_u32_out PROC FRAME
    .endprolog
    mov eax, edx
    xor eax, 0A55A1234h
    mov dword ptr [rcx], eax
    and eax, 0FFh
    or eax, 1
    ret
write_u32_out ENDP

; rmw_u64(uint64_t* slot, uint64_t delta): read-modify-write 8-byte scalar via RCX.
rmw_u64 PROC FRAME
    .endprolog
    mov rax, qword ptr [rcx]
    lea r10, [rax+rdx]
    mov r11, 01122334455667788h
    xor r10, r11
    mov qword ptr [rcx], r10
    ret
rmw_u64 ENDP

; two_ptrs_u64_u32(uint64_t* out64, uint32_t* out32, uint64_t x, uint64_t y):
; copies RCX->R10 and RDX->R11, writes 8-byte and 4-byte non-overlapping out-params.
two_ptrs_u64_u32 PROC FRAME
    .endprolog
    mov r10, rcx
    mov r11, rdx
    lea rax, [r8+r9]
    mov qword ptr [r10], rax
    mov eax, r8d
    sub eax, r9d
    mov dword ptr [r11], eax
    mov rax, r8
    xor rax, r9
    ret
two_ptrs_u64_u32 ENDP

; rmw_u16_r8(uint64_t x, uint64_t y, uint16_t* slot_r8, uint8_t* out_r9):
; reads+writes 2-byte scalar via R8, writes 1-byte scalar via R9.
rmw_u16_r8 PROC FRAME
    .endprolog
    movzx eax, word ptr [r8]
    add ax, cx
    xor ax, dx
    mov word ptr [r8], ax
    mov byte ptr [r9], al
    movzx eax, ax
    ret
rmw_u16_r8 ENDP

; read_only_nv(uint64_t* slot, uint64_t delta): saves RCX in RBX across a non-pointer call,
; reads 8-byte scalar via RBX, restores RBX.
pure_mix PROC FRAME
    .endprolog
    lea rax, [rcx+rcx*2+7]
    ret
pure_mix ENDP

read_only_nv PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    mov rbx, rcx
    mov rcx, rdx
    call pure_mix
    add rax, qword ptr [rbx]
    add rsp, 20h
    pop rbx
    ret
read_only_nv ENDP

; fwd_u32(uint32_t* slot, uint32_t delta): forwards RCX to inner_add_u32.
inner_add_u32 PROC FRAME
    .endprolog
    add dword ptr [rcx], edx
    mov eax, dword ptr [rcx]
    ret
inner_add_u32 ENDP

fwd_u32 PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call inner_add_u32
    add eax, 13h
    add rsp, 28h
    ret
fwd_u32 ENDP

; cond_write_u32(uint32_t* slot, uint32_t val): writes [RCX] only when val is odd.
cond_write_u32 PROC FRAME
    .endprolog
    test edx, 1
    jz cond_skip
    mov dword ptr [rcx], edx
cond_skip:
    mov eax, edx
    and eax, 1
    ret
cond_write_u32 ENDP

; write_then_read_u32(uint32_t* slot, uint32_t val): initializes [RCX] before reading it.
write_then_read_u32 PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    mov eax, dword ptr [rcx]
    add eax, 5
    ret
write_then_read_u32 ENDP

; --- Differential exports ---

; Write-only out-param via LEA [rsp+20h] with uninitialized slot before call.
out_u32_rsp PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 30h
    .allocstack 30h
    .endprolog
    mov rbx, r8
    xor rdx, rcx
    lea rcx, [rsp+20h]
    call write_u32_out
    mov ecx, dword ptr [rsp+20h]
    mov qword ptr [rbx], rcx
    mov qword ptr [rbx+8], rax
    add rax, rcx
    add rsp, 30h
    pop rbx
    ret
out_u32_rsp ENDP

; Read-modify-write via proven RBP anchor (and caller home-space pointer check).
rmw_u64_rbp PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    push rbx
    .pushreg rbx
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov rbx, r8
    mov qword ptr [rbp-10h], rcx
    lea rcx, [rbp-10h]
    call rmw_u64
    mov qword ptr [rbx], rax
    mov rcx, qword ptr [rbp-10h]
    mov qword ptr [rbx+8], rcx
    add rax, rcx
    add rsp, 38h
    pop rbx
    pop rbp
    ret
rmw_u64_rbp ENDP

; Exercises 8/4/2/1 widths, RCX/RDX/R8/R9, nonvolatile pointer preservation, forwarding,
; conditional write on initialized slot, write-then-read callee, and caller home-space slot.
all_widths_and_regs PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 48h
    .allocstack 48h
    .endprolog
    mov rbx, r8
    mov rsi, rcx
    mov r8, rcx
    mov r9, rdx
    lea rcx, [rsp+20h]
    lea rdx, [rsp+28h]
    call two_ptrs_u64_u32
    mov qword ptr [rbx], rax

    mov word ptr [rsp+2Ch], si
    mov rcx, rsi
    mov rdx, qword ptr [rsp+20h]
    lea r8, [rsp+2Ch]
    lea r9, [rsp+2Eh]
    call rmw_u16_r8
    movzx eax, word ptr [rsp+2Ch]
    movzx ecx, byte ptr [rsp+2Eh]
    shl rcx, 16
    or rax, rcx
    mov qword ptr [rbx+8], rax

    mov qword ptr [rsp+70h], rsi       ; caller's own home space (entry RSP+8)
    lea rcx, [rsp+70h]
    mov edx, dword ptr [rsp+28h]
    call read_only_nv
    mov qword ptr [rbx+10h], rax

    mov dword ptr [rsp+30h], esi
    lea rcx, [rsp+30h]
    mov edx, 9
    call fwd_u32
    mov ecx, dword ptr [rsp+30h]
    add rax, rcx
    mov qword ptr [rsp+38h], rax

    mov dword ptr [rsp+34h], esi
    lea rcx, [rsp+34h]
    mov edx, dword ptr [rsp+28h]
    call cond_write_u32
    add eax, dword ptr [rsp+34h]
    add qword ptr [rsp+38h], rax

    lea rcx, [rsp+40h]
    mov edx, esi
    call write_then_read_u32
    mov ecx, dword ptr [rsp+40h]
    add rax, rcx
    add rax, qword ptr [rsp+38h]
    add rax, qword ptr [rbx+10h]

    add rsp, 48h
    pop rsi
    pop rbx
    ret
all_widths_and_regs ENDP

; Non-anchor RBP used as ordinary GPR arithmetic before an internal call (must remain accepted).
non_anchor_rbp_arith PROC FRAME
    push rbp
    .pushreg rbp
    push rbx
    .pushreg rbx
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, r8
    mov rbp, rcx
    lea rcx, [rbp+8]
    call pure_mix
    mov qword ptr [rbx], rax
    add rax, rbp
    add rsp, 28h
    pop rbx
    pop rbp
    ret
non_anchor_rbp_arith ENDP

; --- Refusal negative controls ---

ret_rax_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    mov rax, rcx
    ret
ret_rax_callee ENDP

neg_escape_ret_rax PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call ret_rax_callee
    add rsp, 28h
    ret
neg_escape_ret_rax ENDP

store_mem_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    mov qword ptr [rdx], rcx
    xor eax, eax
    ret
store_mem_callee ENDP

neg_escape_store_mem PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rdx, r8
    lea rcx, [rsp+20h]
    call store_mem_callee
    add rsp, 28h
    ret
neg_escape_store_mem ENDP

store_stack_callee PROC FRAME
    sub rsp, 18h
    .allocstack 18h
    .endprolog
    mov dword ptr [rcx], edx
    mov qword ptr [rsp], rcx
    mov rax, qword ptr [rsp]
    mov eax, dword ptr [rax]
    add rsp, 18h
    ret
store_stack_callee ENDP

neg_escape_store_stack PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov dword ptr [rsp+20h], ecx
    lea rcx, [rsp+20h]
    call store_stack_callee
    add rsp, 28h
    ret
neg_escape_store_stack ENDP

rec_self_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test edx, edx
    jz rec_self_done
    dec edx
    call rec_self_callee
    add rsp, 28h
    ret
rec_self_done:
    mov eax, dword ptr [rcx]
    add rsp, 28h
    ret
rec_self_callee ENDP

neg_recursive_cycle PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov dword ptr [rsp+20h], ecx
    lea rcx, [rsp+20h]
    call rec_self_callee
    add rsp, 28h
    ret
neg_recursive_cycle ENDP

cycle_a PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    test edx, edx
    jz cycle_a_done
    dec edx
    call cycle_b
    add rsp, 28h
    ret
cycle_a_done:
    mov eax, dword ptr [rcx]
    add rsp, 28h
    ret
cycle_a ENDP

cycle_b PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    call cycle_a
    add rsp, 28h
    ret
cycle_b ENDP

neg_mutual_cycle PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov dword ptr [rsp+20h], ecx
    lea rcx, [rsp+20h]
    call cycle_a
    add rsp, 28h
    ret
neg_mutual_cycle ENDP

neg_tail_jump PROC FRAME
    .endprolog
    lea rcx, [rsp+8]
    jmp write_u32_out
neg_tail_jump ENDP

neg_tail_rbp PROC FRAME
    push rbp
    .pushreg rbp
    mov rbp, rsp
    .setframe rbp, 0
    .endprolog
    lea rcx, [rbp-8]
    pop rbp
    jmp write_u32_out
neg_tail_rbp ENDP

neg_var_index_lea PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov eax, edx
    and eax, 8
    lea rcx, [rsp+rax+20h]
    call write_u32_out
    add rsp, 38h
    ret
neg_var_index_lea ENDP

neg_multi_arg_alias_same PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    xor r8d, r8d
    xor r9d, r9d
    lea rcx, [rsp+20h]
    lea rdx, [rsp+20h]
    call two_ptrs_u64_u32
    add rsp, 38h
    ret
neg_multi_arg_alias_same ENDP

neg_multi_arg_alias_overlap PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    xor r8d, r8d
    xor r9d, r9d
    lea rcx, [rsp+20h]
    lea rdx, [rsp+24h]
    call two_ptrs_u64_u32
    add rsp, 38h
    ret
neg_multi_arg_alias_overlap ENDP

neg_multi_arg_copy_alias PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    lea rcx, [rsp+20h]
    mov rdx, rcx
    call two_ptrs_u64_u32
    add rsp, 38h
    ret
neg_multi_arg_copy_alias ENDP

neg_home_space_overlap PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+18h]
    call write_u32_out
    add rsp, 28h
    ret
neg_home_space_overlap ENDP

write_u64_out PROC FRAME
    .endprolog
    mov qword ptr [rcx], rdx
    xor eax, eax
    ret
write_u64_out ENDP

neg_ret_addr_overlap PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+24h]
    call write_u64_out
    add rsp, 28h
    ret
neg_ret_addr_overlap ENDP

neg_past_home_end PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+50h]
    call write_u32_out
    add rsp, 28h
    ret
neg_past_home_end ENDP

neg_uninit_callee_read PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call rmw_u64
    add rsp, 28h
    ret
neg_uninit_callee_read ENDP

neg_uninit_caller_wider_read PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call write_u32_out
    mov rax, qword ptr [rsp+20h]
    add rsp, 28h
    ret
neg_uninit_caller_wider_read ENDP

neg_cond_write_uninit_read PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call cond_write_u32
    mov eax, dword ptr [rsp+20h]
    add rsp, 28h
    ret
neg_cond_write_uninit_read ENDP

mixed_widths_callee PROC FRAME
    .endprolog
    test edx, 1
    jz mixed_byte
    mov eax, dword ptr [rcx]
    ret
mixed_byte:
    mov byte ptr [rcx], dl
    xor eax, eax
    ret
mixed_widths_callee ENDP

neg_mixed_widths PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov dword ptr [rsp+20h], ecx
    lea rcx, [rsp+20h]
    call mixed_widths_callee
    add rsp, 28h
    ret
neg_mixed_widths ENDP

unaccessed_callee PROC FRAME
    .endprolog
    xor eax, eax
    ret
unaccessed_callee ENDP

neg_unaccessed_ptr PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call unaccessed_callee
    add rsp, 28h
    ret
neg_unaccessed_ptr ENDP

ptr_arith_callee PROC FRAME
    .endprolog
    mov eax, dword ptr [rcx]
    add rcx, 4
    ret
ptr_arith_callee ENDP

neg_ptr_arith PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov qword ptr [rsp+20h], rcx
    lea rcx, [rsp+20h]
    call ptr_arith_callee
    add rsp, 28h
    ret
neg_ptr_arith ENDP

neg_unproven_rbp PROC FRAME
    push rbp
    .pushreg rbp
    sub rsp, 30h
    .allocstack 30h
    .endprolog
    lea rcx, [rbp-8]
    call write_u32_out
    add rsp, 30h
    pop rbp
    ret
neg_unproven_rbp ENDP

neg_split_anchor_rbp PROC FRAME
    push rbp
    .pushreg rbp
    sub rsp, 30h
    .allocstack 30h
    .endprolog
    test dl, 1
    jz split_no_anchor
    mov rbp, rsp
    jmp split_join
split_no_anchor:
    mov rbp, rcx
split_join:
    lea rcx, [rbp+20h]
    call write_u32_out
    add rsp, 30h
    pop rbp
    ret
neg_split_anchor_rbp ENDP

tail_partial_write PROC FRAME
    .endprolog
    test edx, 1
    jz tail_no_write
    mov dword ptr [rcx], edx
    mov eax, 1
    ret
tail_no_write:
    xor ecx, ecx
    jmp pure_mix
tail_partial_write ENDP

neg_tail_exit_partial_write PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call tail_partial_write
    mov eax, dword ptr [rsp+20h]
    add rsp, 28h
    ret
neg_tail_exit_partial_write ENDP

cmov32_trunc_callee PROC FRAME
    .endprolog
    mov r9, rcx
    test edx, edx
    cmove ecx, r9d
    mov dword ptr [rcx], edx
    xor eax, eax
    ret
cmov32_trunc_callee ENDP

neg_cmov32_ptr_trunc PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call cmov32_trunc_callee
    add rsp, 28h
    ret
neg_cmov32_ptr_trunc ENDP

bsf_escape_callee PROC FRAME
    .endprolog
    mov rax, rcx
    mov dword ptr [rcx], edx
    bsf rax, rdx
    ret
bsf_escape_callee ENDP

neg_bsf_cond_write_escape PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call bsf_escape_callee
    add rsp, 28h
    ret
neg_bsf_cond_write_escape ENDP

bsr_escape_callee PROC FRAME
    .endprolog
    mov rax, rcx
    mov dword ptr [rcx], edx
    bsr eax, edx
    ret
bsr_escape_callee ENDP

neg_bsr_cond_write_escape PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call bsr_escape_callee
    add rsp, 28h
    ret
neg_bsr_cond_write_escape ENDP

tail_arg_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    jmp write_u32_out
tail_arg_callee ENDP

neg_callee_tail_ptr_live PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call tail_arg_callee
    add rsp, 28h
    ret
neg_callee_tail_ptr_live ENDP

nested_int_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov dword ptr [rcx], edx
    call pure_mix
    add rsp, 28h
    ret
nested_int_callee ENDP

neg_nested_int_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call nested_int_callee
    add rsp, 28h
    ret
neg_nested_int_callee ENDP

nested_multi_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov qword ptr [rcx], rdx
    mov rdx, rcx
    xor r8d, r8d
    xor r9d, r9d
    call two_ptrs_u64_u32
    add rsp, 28h
    ret
nested_multi_callee ENDP

neg_nested_multi_live PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call nested_multi_callee
    add rsp, 28h
    ret
neg_nested_multi_live ENDP

nested_cmov_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov eax, 1
    test edx, edx
    cmove rcx, rax
    call write_u32_out
    add rsp, 28h
    ret
nested_cmov_callee ENDP

neg_nested_cmov_ptr PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call nested_cmov_callee
    add rsp, 28h
    ret
neg_nested_cmov_ptr ENDP

lea_disp_callee PROC FRAME
    .endprolog
    lea r9, [rcx+4]
    mov dword ptr [r9], edx
    xor eax, eax
    ret
lea_disp_callee ENDP

neg_lea_disp_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call lea_disp_callee
    add rsp, 28h
    ret
neg_lea_disp_callee ENDP

lea_index_callee PROC FRAME
    .endprolog
    xor eax, eax
    lea r9, [rcx+rax]
    mov dword ptr [r9], edx
    xor eax, eax
    ret
lea_index_callee ENDP

neg_lea_index_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call lea_index_callee
    add rsp, 28h
    ret
neg_lea_index_callee ENDP

mem_disp_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx+4], edx
    xor eax, eax
    ret
mem_disp_callee ENDP

neg_mem_disp_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call mem_disp_callee
    add rsp, 28h
    ret
neg_mem_disp_callee ENDP

mem_index_callee PROC FRAME
    .endprolog
    xor eax, eax
    mov dword ptr [rcx+rax], edx
    ret
mem_index_callee ENDP

neg_mem_index_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call mem_index_callee
    add rsp, 28h
    ret
neg_mem_index_callee ENDP

mem_ecx_callee PROC FRAME
    .endprolog
    mov dword ptr [ecx], edx
    xor eax, eax
    ret
mem_ecx_callee ENDP

neg_mem_ecx_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call mem_ecx_callee
    add rsp, 28h
    ret
neg_mem_ecx_callee ENDP

stos_ptr_callee PROC FRAME
    push rdi
    .pushreg rdi
    .endprolog
    mov dword ptr [rcx], edx
    mov rdi, r8
    mov rax, rcx
    stosq
    xor eax, eax
    pop rdi
    ret
stos_ptr_callee ENDP

neg_stos_ptr_rax PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call stos_ptr_callee
    add rsp, 28h
    ret
neg_stos_ptr_rax ENDP

movs_ptr_callee PROC FRAME
    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
    .endprolog
    mov dword ptr [rcx], edx
    mov rdi, r8
    mov rsi, rcx
    movsq
    xor eax, eax
    pop rdi
    pop rsi
    ret
movs_ptr_callee ENDP

neg_movs_ptr_rsi PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call movs_ptr_callee
    add rsp, 28h
    ret
neg_movs_ptr_rsi ENDP

sign_ext_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    mov rax, rcx
    cqo
    xor eax, eax
    ret
sign_ext_callee ENDP

neg_sign_ext_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call sign_ext_callee
    add rsp, 28h
    ret
neg_sign_ext_callee ENDP

mul_ptr_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    mov rax, rcx
    mul rdx
    xor eax, eax
    ret
mul_ptr_callee ENDP

neg_mul_ptr_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call mul_ptr_callee
    add rsp, 28h
    ret
neg_mul_ptr_callee ENDP

partial_write_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    mov cl, dl
    xor eax, eax
    ret
partial_write_callee ENDP

neg_partial_write_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call partial_write_callee
    add rsp, 28h
    ret
neg_partial_write_callee ENDP

xor8_partial_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    xor cl, cl
    xor eax, eax
    ret
xor8_partial_callee ENDP

neg_xor8_partial_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call xor8_partial_callee
    add rsp, 28h
    ret
neg_xor8_partial_callee ENDP

bit_test_reg_callee PROC FRAME
    .endprolog
    bt dword ptr [rcx], edx
    setc al
    movzx eax, al
    ret
bit_test_reg_callee ENDP

neg_bit_test_reg_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov dword ptr [rsp+20h], ecx
    lea rcx, [rsp+20h]
    call bit_test_reg_callee
    add rsp, 28h
    ret
neg_bit_test_reg_callee ENDP

cmov32_mem_callee PROC FRAME
    .endprolog
    mov dword ptr [rcx], edx
    test edx, edx
    cmove ecx, dword ptr [r8]
    xor eax, eax
    ret
cmov32_mem_callee ENDP

neg_cmov32_mem_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+20h]
    call cmov32_mem_callee
    add rsp, 28h
    ret
neg_cmov32_mem_callee ENDP

neg_may_write_saved_reg PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov qword ptr [rsp+20h], rbx
    mov ebx, ecx
    lea rcx, [rsp+20h]
    mov edx, 1
    call cond_write_u32
    mov rbx, qword ptr [rsp+20h]
    add rsp, 28h
    ret
neg_may_write_saved_reg ENDP

neg_read_saved_reg PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov qword ptr [rsp+20h], rbx
    lea rcx, [rsp+20h]
    mov edx, 1
    call rmw_u64
    mov rbx, qword ptr [rsp+20h]
    add rsp, 28h
    ret
neg_read_saved_reg ENDP

five_arg_callee PROC FRAME
    .endprolog
    mov rax, qword ptr [rsp+28h]
    mov dword ptr [rcx], eax
    xor eax, eax
    ret
five_arg_callee ENDP

neg_callee_stack_arg_overlap PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog
    mov qword ptr [rsp+20h], rdx
    lea rcx, [rsp+20h]
    call five_arg_callee
    add rsp, 38h
    ret
neg_callee_stack_arg_overlap ENDP

incoming_slot_ptr_callee PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    lea rcx, [rsp+50h]
    call rmw_u64
    add rsp, 28h
    ret
incoming_slot_ptr_callee ENDP

neg_incoming_stack_slot_uninit PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    xor edx, edx
    call incoming_slot_ptr_callee
    add rsp, 28h
    ret
neg_incoming_stack_slot_uninit ENDP

END
