; coverage: record cpu_aot_x87_exp_log
; Source-owned differential fixture for x87 F2XM1, FYL2X, and FYL2XP1 plus
; bounded x87 stack/control/status transport over flat [rcx|rdx|r8] memory.
;   RCX = const void* input_a (m32/m64/m80 operand or m16 control word)
;   RDX = const void* input_b (m32/m64/m80 operand or m16 control word)
;   R8  = void*       output  (m32/m64/m80/m16 output; also carries input m16
;                              control word on binary FYL2X/FYL2XP1 exports)
.code

op_f2xm1_m80 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
IFDEF ALTER_X87_EXP_LOG
    fldz
    fyl2xp1
ELSE
    f2xm1
ENDIF
    fstp tbyte ptr [r8]
    ret
op_f2xm1_m80 ENDP

op_f2xm1_m64 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld qword ptr [rcx]
    f2xm1
    fstp qword ptr [r8]
    ret
op_f2xm1_m64 ENDP

op_f2xm1_m32 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld dword ptr [rcx]
    f2xm1
    fstp dword ptr [r8]
    ret
op_f2xm1_m32 ENDP

op_f2xm1_fst_m64 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld qword ptr [rcx]
    f2xm1
    fst qword ptr [r8]
    fstp st(0)
    ret
op_f2xm1_fst_m64 ENDP

op_f2xm1_fst_m32 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld dword ptr [rcx]
    f2xm1
    fst dword ptr [r8]
    fstp st(0)
    ret
op_f2xm1_fst_m32 ENDP

op_f2xm1_sw PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld tbyte ptr [rcx]
    f2xm1
    fnstsw ax
    fstp st(0)
    mov word ptr [r8], ax
    ret
op_f2xm1_sw ENDP

op_fyl2x_m80 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld tbyte ptr [rcx]
    fld tbyte ptr [rdx]
IFDEF ALTER_X87_EXP_LOG
    fyl2xp1
ELSE
    fyl2x
ENDIF
    fstp tbyte ptr [r8]
    ret
op_fyl2x_m80 ENDP

op_fyl2x_m64 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld qword ptr [rcx]
    fld qword ptr [rdx]
    fyl2x
    fstp qword ptr [r8]
    ret
op_fyl2x_m64 ENDP

op_fyl2x_m32 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld dword ptr [rcx]
    fld dword ptr [rdx]
    fyl2x
    fstp dword ptr [r8]
    ret
op_fyl2x_m32 ENDP

op_fyl2x_sw PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld tbyte ptr [rcx]
    fld tbyte ptr [rdx]
    fyl2x
    fnstsw ax
    fstp st(0)
    mov word ptr [r8], ax
    ret
op_fyl2x_sw ENDP

; Verifies FYL2X pops ST(0) and preserves the underlying stack slot (+1.0 sentinel).
op_fyl2x_pop_depth PROC FRAME
    .endprolog
    fninit
    fld1
    fld tbyte ptr [rcx]
    fld tbyte ptr [rdx]
    fyl2x
    fstp st(0)
    fstp tbyte ptr [r8]
    ret
op_fyl2x_pop_depth ENDP

; Exercises FYL2X at maximum x87 stack depth (8 entries) and unwinds cleanly.
op_fyl2x_depth8 PROC FRAME
    .endprolog
    fninit
    fldz
    fldz
    fldz
    fldz
    fldz
    fldz
    fld tbyte ptr [rcx]
    fld tbyte ptr [rdx]
    fyl2x
    fstp tbyte ptr [r8]
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp st(0)
    fstp st(0)
    ret
op_fyl2x_depth8 ENDP

op_fyl2xp1_m80 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld tbyte ptr [rcx]
    fld tbyte ptr [rdx]
IFDEF ALTER_X87_EXP_LOG
    fyl2x
ELSE
    fyl2xp1
ENDIF
    fstp tbyte ptr [r8]
    ret
op_fyl2xp1_m80 ENDP

op_fyl2xp1_m64 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld qword ptr [rcx]
    fld qword ptr [rdx]
    fyl2xp1
    fstp qword ptr [r8]
    ret
op_fyl2xp1_m64 ENDP

op_fyl2xp1_m32 PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld dword ptr [rcx]
    fld dword ptr [rdx]
    fyl2xp1
    fstp dword ptr [r8]
    ret
op_fyl2xp1_m32 ENDP

op_fyl2xp1_sw PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [r8]
    fld tbyte ptr [rcx]
    fld tbyte ptr [rdx]
    fyl2xp1
    fnstsw ax
    fstp st(0)
    mov word ptr [r8], ax
    ret
op_fyl2xp1_sw ENDP

; Verifies FYL2XP1 pops ST(0) and preserves the underlying stack slot (+0.0 sentinel).
op_fyl2xp1_pop_depth PROC FRAME
    .endprolog
    fninit
    fldz
    fld tbyte ptr [rcx]
    fld tbyte ptr [rdx]
    fyl2xp1
    fstp st(0)
    fstp tbyte ptr [r8]
    ret
op_fyl2xp1_pop_depth ENDP

; Composition: 2^(1 * log2(x + 1)) - 1 == x for x in [-(1-sqrt(2)/2), 1-sqrt(2)/2].
op_exp_log_roundtrip PROC FRAME
    .endprolog
    fninit
    fldcw word ptr [rdx]
    fld1
    fld tbyte ptr [rcx]
    fyl2xp1
IFDEF ALTER_X87_EXP_LOG
    fstp st(0)
    fldz
ELSE
    f2xm1
ENDIF
    fstp tbyte ptr [r8]
    ret
op_exp_log_roundtrip ENDP

; Control word transport: fldcw -> fnstcw preserves rounding and precision control bits.
op_fcw_roundtrip PROC FRAME
    .endprolog
    fninit
IFDEF ALTER_X87_EXP_LOG
    fld1
    fstp st(0)
ELSE
    fldcw word ptr [rcx]
ENDIF
    fnstcw word ptr [r8]
    ret
op_fcw_roundtrip ENDP

; FNINIT resets FCW to 0x037F and empties any live x87 stack entries.
op_fninit_cw_sw PROC FRAME
    .endprolog
    fldcw word ptr [rcx]
    fld tbyte ptr [rdx]
    f2xm1
    fninit
    fnstcw word ptr [r8]
    ret
op_fninit_cw_sw ENDP

; FNINIT clears sticky status flags and resets TOP to 0.
op_fninit_sw PROC FRAME
    .endprolog
    fld tbyte ptr [rcx]
    f2xm1
    fninit
    fnstsw ax
    mov word ptr [r8], ax
    ret
op_fninit_sw ENDP

END
