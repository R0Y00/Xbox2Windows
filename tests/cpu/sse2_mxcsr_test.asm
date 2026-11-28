; Host-only controls for the native and lifted differential.
.code

read_mxcsr PROC
    stmxcsr dword ptr [rcx]
    ret
read_mxcsr ENDP

write_mxcsr PROC
    ldmxcsr dword ptr [rcx]
    ret
write_mxcsr ENDP

END
