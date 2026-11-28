EXTERN __imp_DStorageGetFactory:QWORD
PUBLIC native_entry

.data
; Fresh fixture-owned interface identifiers
; IID_FixtureFactory: {ca891497-6f8f-4d03-b1a2-f239b4ce41af}
IID_FixtureFactory    db 97h, 14h, 89h, 0cah, 8fh, 6fh, 03h, 4dh, 0b1h, 0a2h, 0f2h, 39h, 0b4h, 0ceh, 41h, 0afh

; IID_FixtureQueue: {82af1362-6445-4b96-9fbf-5d2d416b605e}
IID_FixtureQueue      db 62h, 13h, 0afh, 82h, 45h, 64h, 96h, 4bh, 9fh, 0bfh, 5dh, 2dh, 41h, 6bh, 60h, 5eh

; Public MIT DirectStorage PC IDStorageFactory: {6924ea0c-c3cd-4826-b10a-f64f4ed927c1}
; Source: https://github.com/microsoft/DirectStorage (include/dstorage.h, MIT license)
IID_IDStorageFactory_PC db 0ch, 0eah, 24h, 69h, 0cdh, 0c3h, 26h, 48h, 0b1h, 0ah, 0f6h, 4fh, 4eh, 0d9h, 27h, 0c1h

pFactory dq 0
pQueue   dq 0

queue_name db "TestQueue", 0
ALIGN 8
queue_desc:
dd 0                ; SourceType = DSTORAGE_REQUEST_SOURCE_FILE (0)
dd 0                ; padding
dw 16               ; Capacity = 16
db 0                ; Priority = NORMAL (0)
db 5 dup(0)         ; padding
pQueueName dq 0     ; Name pointer (initialized at runtime)

src_payload db "DSTORAGE_READBACK_TEST_DATA_32B!" ; exactly 32 bytes
dest_buffer db 32 dup(0AAh)

ALIGN 8
test_request:
; Options (8 bytes): SourceType=Memory(1) << 8, DestinationType=Memory(0)
dd 00000100h        ; offset 0
dd 0                ; offset 4
; Source (16 bytes)
pSource dq 0        ; offset 8: SourceBuffer (initialized at runtime)
dd 32               ; offset 16: SourceSize
dd 0                ; offset 20: SourcePad
; Destination (16 bytes)
pDest dq 0          ; offset 24: DestBuffer (initialized at runtime)
dd 32               ; offset 32: DestSize
dd 0                ; offset 36: DestPad
; UncompressedSize (4 bytes)
dd 32               ; offset 40: UncompressedSize
dd 0                ; offset 44: Pad
; CancellationTag (8 bytes)
dq 12345678h        ; offset 48: CancellationTag
; Name (8 bytes)
dq 0                ; offset 56: Name

.code
native_entry PROC FRAME
    sub rsp, 38h
    .allocstack 38h
    .endprolog

    ; Initialize runtime pointers to avoid 32-bit static relocations in 64-bit image
    lea rax, queue_name
    mov qword ptr [pQueueName], rax
    lea rax, src_payload
    mov qword ptr [pSource], rax
    lea rax, dest_buffer
    mov qword ptr [pDest], rax

    ; Step 0: Refusal control - request PC IDStorageFactory IID {6924ea0c-c3cd-4826-b10a-f64f4ed927c1}
    ; Must return E_NOINTERFACE (0x80004002) and leave *ppv untouched.
    mov qword ptr [pFactory], 0DEADBEEFh
    lea rcx, IID_IDStorageFactory_PC
    lea rdx, pFactory
    call qword ptr [__imp_DStorageGetFactory]
    cmp eax, 80004002h ; E_NOINTERFACE
    jne fail_pc_refusal
    cmp qword ptr [pFactory], 0DEADBEEFh
    jne fail_pc_clobber

    ; Step 1: DStorageGetFactory(&IID_FixtureFactory, &pFactory)
    mov qword ptr [pFactory], 0
    lea rcx, IID_FixtureFactory
    lea rdx, pFactory
    call qword ptr [__imp_DStorageGetFactory]
    test eax, eax
    jnz fail_factory
    mov rcx, qword ptr [pFactory]
    test rcx, rcx
    jz fail_factory

    ; Step 2: factory->SetStagingBufferSize(0x2000000) (slot 8 = +0x40 per recorder convention)
    mov rax, qword ptr [rcx]
    mov edx, 2000000h
    call qword ptr [rax + 40h]
    test eax, eax
    js fail_staging_buffer

    ; Step 3: factory->CreateQueue(&queue_desc, &IID_FixtureQueue, &pQueue) (slot 3 = +0x18 per recorder convention)
    mov rcx, qword ptr [pFactory]
    mov rax, qword ptr [rcx]
    lea rdx, queue_desc
    lea r8, IID_FixtureQueue
    lea r9, pQueue
    call qword ptr [rax + 18h]
    test eax, eax
    js fail_create_queue
    mov rcx, qword ptr [pQueue]
    test rcx, rcx
    jz fail_create_queue

    ; Step 4: queue->EnqueueRequest(&test_request) (slot 3 = +0x18 per recorder convention)
    mov rcx, qword ptr [pQueue]
    mov rax, qword ptr [rcx]
    lea rdx, test_request
    call qword ptr [rax + 18h]

    ; Step 5: Delayed-submit negative control:
    ; Staged request (< half capacity) must remain pending before Submit
    mov rcx, qword ptr [pQueue]
    cmp dword ptr [rcx + 14h], 0
    jne fail_delayed_submit

    ; Check destination buffer remains unmodified (0xAA) before Submit
    cmp byte ptr [dest_buffer], 0AAh
    jne fail_delayed_submit

    ; Step 6: queue->Submit() (slot 6 = +0x30 per recorder convention)
    mov rcx, qword ptr [pQueue]
    mov rax, qword ptr [rcx]
    call qword ptr [rax + 30h]

    ; Step 7: Verification: completion flag must now be set and readback verified
    mov rcx, qword ptr [pQueue]
    cmp dword ptr [rcx + 14h], 1
    jne fail_submit_flag

    ; Exact readback verification: dest_buffer must match src_payload exactly
    lea r8, src_payload
    lea r9, dest_buffer
    xor eax, eax
verify_loop:
    mov dl, byte ptr [r8 + rax]
    cmp byte ptr [r9 + rax], dl
    jne fail_submit_readback
    inc eax
    cmp eax, 32
    jl verify_loop

    ; Step 8: Cleanup and release objects (slot 2 = +0x10 in IUnknown)
    mov rcx, qword ptr [pQueue]
    mov rax, qword ptr [rcx]
    call qword ptr [rax + 10h] ; queue->Release()

    mov rcx, qword ptr [pFactory]
    mov rax, qword ptr [rcx]
    call qword ptr [rax + 10h] ; factory->Release()

    ; Success exit milestone
    mov eax, 0870h
    jmp done

fail_factory:
    mov eax, 0301h
    jmp done

fail_staging_buffer:
    mov eax, 0302h
    jmp done

fail_create_queue:
    mov eax, 0303h
    jmp done

fail_delayed_submit:
    mov eax, 0304h
    jmp done

fail_submit_flag:
    mov eax, 0305h
    jmp done

fail_pc_refusal:
    mov eax, 0306h
    jmp done

fail_pc_clobber:
    mov eax, 0307h
    jmp done

fail_submit_readback:
    mov eax, 0308h
    jmp done

done:
    add rsp, 38h
    ret
native_entry ENDP
END
