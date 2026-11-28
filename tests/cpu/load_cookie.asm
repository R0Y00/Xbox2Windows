; Record the loader's cookie at the first DLL or EXE entry, with no CRT initialization.
.data
__security_cookie QWORD 02B992DDFA232h
observed_cookie QWORD 0

.const
PUBLIC _load_config_used
_load_config_used LABEL BYTE
    DWORD 138h
    BYTE 54h DUP (0)
    QWORD __security_cookie
    BYTE 0D8h DUP (0)

.code
fixture_entry PROC
    cmp edx, 1
    jne return_ok
    mov rax, __security_cookie
    mov observed_cookie, rax
return_ok:
    mov eax, 1
    ret
fixture_entry ENDP

read_cookie PROC
    mov rax, observed_cookie
    ret
read_cookie ENDP

native_entry PROC
    mov rax, __security_cookie
    mov rcx, 02B992DDFA232h
    cmp rax, rcx
    je short default_cookie
    test rax, rax
    je short bad_cookie
    shr rax, 48
    jne short bad_cookie
    mov eax, 42
    ret
default_cookie:
    mov eax, 41
    ret
bad_cookie:
    mov eax, 43
    ret
native_entry ENDP
END
