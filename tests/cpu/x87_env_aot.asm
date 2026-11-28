; coverage: record cpu_aot_x87_env
; Source-owned differential fixture for x87 environment and state instructions:
; FDECSTP, FNSTENV, FSTENV, FLDENV, FNSAVE, FSAVE, FRSTOR plus bounded
; stack/control/status glue over flat [rcx|rdx|r8] memory.
;   RCX = const void* input_a (m28 env, m108 state, m16 cw, or m32/64/80 operand)
;   RDX = const void* input_b (m16 cw or m32/64/80 operand)
;   R8  = void*       output  (m28 env, m108 state, or m16/m32/m64/m80 scalar)
.code

op_fdecstp_sw PROC FRAME
    .endprolog
    fninit
IFDEF ALTER_X87_ENV
    fdecstp
    fdecstp
ELSE
    fdecstp
ENDIF
    fnstsw ax
    mov word ptr [r8], ax
    ret
op_fdecstp_sw ENDP

op_fdecstp_wrap_env PROC FRAME
    .endprolog
    fninit
    fldenv byte ptr [rcx]
IFDEF ALTER_X87_ENV
    fdecstp
ENDIF
    fdecstp
    fnstenv byte ptr [r8]
    fninit
    ret
op_fdecstp_wrap_env ENDP

op_fnstenv_pushed PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
    fld1
    fldz
    fnstenv byte ptr [r8]
    fstp st(0)
    fstp st(0)
    fstp st(0)
    ret
op_fnstenv_pushed ENDP

op_fstenv_pushed PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld qword ptr [rcx]
    fld1
IFDEF ALTER_X87_ENV
    fstp st(0)
    fldz
ENDIF
    fstenv byte ptr [r8]
    fstp st(0)
    fstp st(0)
    ret
op_fstenv_pushed ENDP

op_fnstenv_post_cw PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rcx]
    fnstenv byte ptr [r8]
    fnstcw word ptr [r8]
    ret
op_fnstenv_post_cw ENDP

op_fldenv_fnstenv PROC FRAME
    .endprolog
    fldenv byte ptr [rcx]
IFDEF ALTER_X87_ENV
    fninit
ENDIF
    fnstenv byte ptr [r8]
    fninit
    ret
op_fldenv_fnstenv ENDP

op_fldenv_wait_fnstenv PROC FRAME
    .endprolog
    fldenv byte ptr [rcx]
    wait
    fnstenv byte ptr [r8]
    fninit
    ret
op_fldenv_wait_fnstenv ENDP

op_fnsave_stack PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
    fld1
    fldz
IFDEF ALTER_X87_ENV
    fstp st(0)
ENDIF
    fnsave byte ptr [r8]
    ret
op_fnsave_stack ENDP

op_fnsave_post_cw PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
    fnsave byte ptr [r8]
    fnstcw word ptr [r8]
    ret
op_fnsave_post_cw ENDP

op_fnsave_post_sw PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
    fnsave byte ptr [r8]
    fnstsw ax
    mov word ptr [r8], ax
    ret
op_fnsave_post_sw ENDP

op_fsave_stack PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
    fld1
IFDEF ALTER_X87_ENV
    fdecstp
ENDIF
    fsave byte ptr [r8]
    ret
op_fsave_stack ENDP

op_frstor_fnsave PROC FRAME
    .endprolog
    frstor byte ptr [rcx]
IFDEF ALTER_X87_ENV
    fdecstp
ENDIF
    fnsave byte ptr [r8]
    ret
op_frstor_fnsave ENDP

op_frstor_fdecstp_fnsave PROC FRAME
    .endprolog
    frstor byte ptr [rcx]
    fdecstp
    fnsave byte ptr [r8]
    ret
op_frstor_fdecstp_fnsave ENDP

op_fld_m32_fstp_m64 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld dword ptr [rcx]
    fstp qword ptr [r8]
    ret
op_fld_m32_fstp_m64 ENDP

op_fld_m64_fstp_m32 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld qword ptr [rcx]
    fstp dword ptr [r8]
    ret
op_fld_m64_fstp_m32 ENDP

op_fld_m80_fstp_m80 PROC FRAME
    .endprolog
    fninit
    fld tbyte ptr [rcx]
    fstp tbyte ptr [r8]
    ret
op_fld_m80_fstp_m80 ENDP

op_fstp_c1_fdecstp PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
    fstp dword ptr [r8]
    fdecstp
    fnstsw ax
    mov word ptr [r8], ax
    ret
op_fstp_c1_fdecstp ENDP

END
