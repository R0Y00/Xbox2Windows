// DirectStorage Queue::CancelRequestsWithTag (slot 7, #1843 slice 3): cancels previously enqueued
// requests whose CancellationTag (+0x38) matches (tag & mask) == value across both pre-Submit
// (`waiting`) and post-Submit (`submitted`) queues, without counting cancelled requests as failures.
//
// Synthetic file and memory inputs only; no game data.
// Usage: startup_storage_cancel XCRDAPI.dll [--control-expect-uncancelled]
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

namespace {

using u64 = std::uint64_t;
constexpr DWORD timeout_ms = 15000;
constexpr unsigned piece = 256;
constexpr unsigned char poison = 0xA5;

struct Object { void** vtable; };
using GetFactoryFn = long(__stdcall*)(const void*, void**);
using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);
using CloseFileFn = void(__stdcall*)(void*);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, u64);
using SubmitFn = void(__stdcall*)(void*);
using CancelFn = void(__stdcall*)(void*, u64, u64);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);

struct SimpleFence {
    void** vtable;
    HANDLE event;
    u64 value;
    static long __stdcall signal(void* self, u64 value)
    {
        auto* fence = static_cast<SimpleFence*>(self);
        fence->value = value;
        SetEvent(fence->event);
        return 0;
    }
};

struct GateFence {
    void** vtable;
    HANDLE entered;
    HANDLE release;
    static long __stdcall signal(void* self, u64)
    {
        auto* fence = static_cast<GateFence*>(self);
        SetEvent(fence->entered);
        WaitForSingleObject(fence->release, timeout_ms);
        return 0;
    }
};

bool expect(bool condition, const char* what)
{
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", what);
    return condition;
}

bool all_equal(const std::vector<unsigned char>& bytes, unsigned char value)
{
    for (unsigned char b : bytes)
        if (b != value) return false;
    return true;
}

std::string read_text(const wchar_t* path)
{
    std::string text;
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return text;
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(file, buf, sizeof buf, &got, nullptr) && got) text.append(buf, got);
    CloseHandle(file);
    return text;
}

void fill_file_request(unsigned char (&req)[0x48], void* dest, unsigned dest_size, void* file_obj,
                       u64 offset, unsigned src_size, u64 tag)
{
    std::memset(req, 0, sizeof req);
    *reinterpret_cast<u64*>(req + 0x00) = 0;
    *reinterpret_cast<void**>(req + 0x08) = dest;
    *reinterpret_cast<unsigned*>(req + 0x18) = dest_size;
    *reinterpret_cast<void**>(req + 0x20) = file_obj;
    *reinterpret_cast<u64*>(req + 0x28) = offset;
    *reinterpret_cast<unsigned*>(req + 0x30) = src_size;
    *reinterpret_cast<u64*>(req + 0x38) = tag;
}

void fill_memory_request(unsigned char (&req)[0x48], void* dest, unsigned dest_size, const void* src,
                         unsigned src_size, u64 tag)
{
    std::memset(req, 0, sizeof req);
    *reinterpret_cast<u64*>(req + 0x00) = 0x400000;  // memory source
    *reinterpret_cast<void**>(req + 0x08) = dest;
    *reinterpret_cast<unsigned*>(req + 0x18) = dest_size;
    *reinterpret_cast<const void**>(req + 0x20) = src;
    *reinterpret_cast<unsigned*>(req + 0x30) = src_size;
    *reinterpret_cast<u64*>(req + 0x38) = tag;
}

bool wait_submit(Object* queue, Object* status, unsigned status_index, void* simple_vtable[11], u64 fence_val)
{
    SimpleFence fence{simple_vtable, CreateEventW(nullptr, TRUE, FALSE, nullptr), 0};
    reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, status_index);
    reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &fence, fence_val);
    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    const DWORD waited = WaitForSingleObject(fence.event, timeout_ms);
    CloseHandle(fence.event);
    return waited == WAIT_OBJECT_0 && fence.value == fence_val;
}

}  // namespace

int main(int argc, char** argv)
{
    bool control_expect_uncancelled = false;
    if (argc == 3 && std::strcmp(argv[2], "--control-expect-uncancelled") == 0) {
        control_expect_uncancelled = true;
    } else if (argc != 2) {
        std::fprintf(stderr, "usage: startup_storage_cancel XCRDAPI.dll [--control-expect-uncancelled]\n");
        return 2;
    }

    SetEnvironmentVariableW(L"X2W_XMEM_PERMISSIVE", L"1");
    wchar_t temp_dir[MAX_PATH] = {}, log_path[MAX_PATH] = {}, data_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_cancel_%lu_%d.log", temp_dir, GetCurrentProcessId(),
                  control_expect_uncancelled ? 1 : 0);
    std::swprintf(data_path, MAX_PATH, L"%lsx2w_cancel_%lu_%d.bin", temp_dir, GetCurrentProcessId(),
                  control_expect_uncancelled ? 1 : 0);
    DeleteFileW(log_path);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);

    // Four patterned 256-byte blocks in a synthetic file.
    std::vector<unsigned char> file_bytes(4 * piece);
    for (std::size_t i = 0; i < file_bytes.size(); ++i)
        file_bytes[i] = static_cast<unsigned char>(((i * 13) ^ (i >> 5) ^ 0x5A) & 0xFF);
    {
        HANDLE out = CreateFileW(data_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD written = 0;
        if (out == INVALID_HANDLE_VALUE ||
            !WriteFile(out, file_bytes.data(), static_cast<DWORD>(file_bytes.size()), &written, nullptr) ||
            written != file_bytes.size()) {
            std::fprintf(stderr, "cannot write test fixture file\n");
            return 1;
        }
        CloseHandle(out);
    }

    HMODULE library = LoadLibraryA(argv[1]);
    auto* get_factory = library ? reinterpret_cast<GetFactoryFn>(GetProcAddress(library, "DStorageGetFactory")) : nullptr;
    void* factory_ptr = nullptr;
    if (!get_factory || get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) {
        std::fprintf(stderr, "cannot load XCRDAPI stand-in\n");
        DeleteFileW(data_path);
        return 1;
    }
    auto* factory = static_cast<Object*>(factory_ptr);

    void* file_ptr = nullptr;
    void* status_ptr = nullptr;
    void* queue_ptr = nullptr;
    unsigned char qdesc[0x20] = {};
    *reinterpret_cast<std::uint16_t*>(qdesc + 8) = 64;
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &file_ptr) != 0 || !file_ptr ||
        reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 16, "Cancel", nullptr, &status_ptr) != 0 ||
        reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, qdesc, nullptr, &queue_ptr) != 0) {
        std::fprintf(stderr, "cannot create DirectStorage test objects\n");
        DeleteFileW(data_path);
        return 1;
    }

    auto* status = static_cast<Object*>(status_ptr);
    auto* queue = static_cast<Object*>(queue_ptr);
    auto enqueue_req = reinterpret_cast<EnqueueRequestFn>(queue->vtable[3]);
    auto enqueue_status = reinterpret_cast<EnqueueStatusFn>(queue->vtable[4]);
    auto enqueue_signal = reinterpret_cast<EnqueueSignalFn>(queue->vtable[5]);
    auto submit = reinterpret_cast<SubmitFn>(queue->vtable[6]);
    auto cancel = reinterpret_cast<CancelFn>(queue->vtable[7]);
    auto is_complete = reinterpret_cast<IsCompleteFn>(status->vtable[3]);
    auto get_hr = reinterpret_cast<GetHResultFn>(status->vtable[4]);

    void* simple_vtable[11] = {};
    simple_vtable[10] = reinterpret_cast<void*>(&SimpleFence::signal);
    void* gate_vtable[11] = {};
    gate_vtable[10] = reinterpret_cast<void*>(&GateFence::signal);

    bool ok = true;

    // 0. Cancelling on a fresh queue before any EnqueueRequest must be a safe no-op.
    cancel(queue, 0, 0);

    // 1. Exact tag match before Submit: middle request cancelled, surrounding requests complete.
    {
        std::vector<unsigned char> d0(piece, poison), d1(piece, poison), d2(piece, poison);
        alignas(8) unsigned char r0[0x48], r1[0x48], r2[0x48];
        fill_file_request(r0, d0.data(), piece, file_ptr, 0 * piece, piece, 0x1001ull);
        fill_file_request(r1, d1.data(), piece, file_ptr, 1 * piece, piece, 0x2002ull);
        fill_memory_request(r2, d2.data(), piece, file_bytes.data() + 2 * piece, piece, 0x3003ull);
        enqueue_req(queue, r0);
        enqueue_req(queue, r1);
        enqueue_req(queue, r2);
        cancel(queue, ~0ull, 0x2002ull);
        ok = expect(wait_submit(queue, status, 0, simple_vtable, 1), "case 1 completed") && ok;
        ok = expect(is_complete(status, 0) && get_hr(status, 0) == S_OK, "case 1 status S_OK") && ok;
        ok = expect(std::memcmp(d0.data(), file_bytes.data() + 0 * piece, piece) == 0, "case 1 r0 bytes") && ok;
        if (control_expect_uncancelled) {
            // Negative control: must FAIL because r1 was cancelled.
            ok = expect(std::memcmp(d1.data(), file_bytes.data() + 1 * piece, piece) == 0,
                        "control expected cancelled r1 to have executed") && ok;
            reinterpret_cast<CloseFileFn>(static_cast<Object*>(file_ptr)->vtable[3])(file_ptr);
            DeleteFileW(data_path);
            DeleteFileW(log_path);
            return ok ? 0 : 1;
        }
        ok = expect(all_equal(d1, poison), "case 1 cancelled r1 destination untouched") && ok;
        ok = expect(std::memcmp(d2.data(), file_bytes.data() + 2 * piece, piece) == 0, "case 1 r2 bytes") && ok;
    }

    // 2. Partial bitmask group cancel: upper 32 bits select group 2, keeping groups 1 and 3 intact.
    {
        std::vector<unsigned char> g1(piece, poison), g2a(piece, poison), g2b(piece, poison), g3(piece, poison);
        alignas(8) unsigned char r1[0x48], r2a[0x48], r2b[0x48], r3[0x48];
        fill_file_request(r1, g1.data(), piece, file_ptr, 0 * piece, piece, 0x0000000100000001ull);
        fill_file_request(r2a, g2a.data(), piece, file_ptr, 1 * piece, piece, 0x0000000200000001ull);
        fill_file_request(r2b, g2b.data(), piece, file_ptr, 2 * piece, piece, 0x0000000200000099ull);
        fill_file_request(r3, g3.data(), piece, file_ptr, 3 * piece, piece, 0x0000000300000001ull);
        enqueue_req(queue, r1);
        enqueue_req(queue, r2a);
        enqueue_req(queue, r2b);
        enqueue_req(queue, r3);
        cancel(queue, 0xFFFFFFFF00000000ull, 0x0000000200000000ull);
        ok = expect(wait_submit(queue, status, 1, simple_vtable, 2), "case 2 completed") && ok;
        ok = expect(is_complete(status, 1) && get_hr(status, 1) == S_OK, "case 2 status S_OK") && ok;
        ok = expect(std::memcmp(g1.data(), file_bytes.data() + 0 * piece, piece) == 0, "case 2 group 1 kept") && ok;
        ok = expect(all_equal(g2a, poison) && all_equal(g2b, poison), "case 2 group 2 cancelled") && ok;
        ok = expect(std::memcmp(g3.data(), file_bytes.data() + 3 * piece, piece) == 0, "case 2 group 3 kept") && ok;
    }

    // 3. Adversarial mask/value mismatch: (value & ~mask) != 0 must match nothing.
    {
        std::vector<unsigned char> dst(piece, poison);
        alignas(8) unsigned char req[0x48];
        fill_file_request(req, dst.data(), piece, file_ptr, 0 * piece, piece, 0xFFull);
        enqueue_req(queue, req);
        cancel(queue, 0x0Full, 0x1Full);  // bit 4 set in value where mask is 0 -> (tag & mask) != value
        ok = expect(wait_submit(queue, status, 2, simple_vtable, 3), "case 3 completed") && ok;
        ok = expect(is_complete(status, 2) && get_hr(status, 2) == S_OK, "case 3 status S_OK") && ok;
        ok = expect(std::memcmp(dst.data(), file_bytes.data(), piece) == 0, "case 3 unmasked bit did not cancel") && ok;
    }

    // 4. Wildcard cancel-all (mask=0, value=0): cancels a valid request AND an EOF-failing request;
    //    the cancelled failing request must NOT run and must NOT fail the status entry.
    {
        std::vector<unsigned char> d_ok(piece, poison), d_eof(piece, poison);
        alignas(8) unsigned char r_ok[0x48], r_eof[0x48];
        fill_file_request(r_ok, d_ok.data(), piece, file_ptr, 0, piece, 0xABCDull);
        fill_file_request(r_eof, d_eof.data(), piece, file_ptr, file_bytes.size(), piece, 0xDEADull);
        enqueue_req(queue, r_ok);
        enqueue_req(queue, r_eof);
        cancel(queue, 0, 0);
        ok = expect(wait_submit(queue, status, 3, simple_vtable, 4), "case 4 completed") && ok;
        ok = expect(is_complete(status, 3) && get_hr(status, 3) == S_OK,
                    "case 4 cancelled EOF request did not fail status") && ok;
        ok = expect(all_equal(d_ok, poison) && all_equal(d_eof, poison), "case 4 all destinations untouched") && ok;
        const std::string log_so_far = read_text(log_path);
        ok = expect(log_so_far.find("read failed") == std::string::npos,
                    "case 4 cancelled EOF request never executed read") && ok;
    }

    // 5. Temporal boundary: a request enqueued AFTER CancelRequestsWithTag with the SAME tag must run.
    {
        std::vector<unsigned char> before_dst(piece, poison), after_dst(piece, poison);
        alignas(8) unsigned char r_before[0x48], r_after[0x48];
        fill_file_request(r_before, before_dst.data(), piece, file_ptr, 1 * piece, piece, 0xCAFEull);
        fill_file_request(r_after, after_dst.data(), piece, file_ptr, 2 * piece, piece, 0xCAFEull);
        enqueue_req(queue, r_before);
        cancel(queue, ~0ull, 0xCAFEull);
        enqueue_req(queue, r_after);
        ok = expect(wait_submit(queue, status, 4, simple_vtable, 5), "case 5 completed") && ok;
        ok = expect(is_complete(status, 4) && get_hr(status, 4) == S_OK, "case 5 status S_OK") && ok;
        ok = expect(all_equal(before_dst, poison), "case 5 pre-cancel request cancelled") && ok;
        ok = expect(std::memcmp(after_dst.data(), file_bytes.data() + 2 * piece, piece) == 0,
                    "case 5 post-cancel request with same tag still executed") && ok;
    }

    // 6. Post-Submit cancellation: block worker on a gate fence so requests are deterministically in
    //    `work->submitted`, cancel the matching one live, release gate, and verify clean drain & S_OK.
    {
        GateFence gate{gate_vtable, CreateEventW(nullptr, TRUE, FALSE, nullptr),
                       CreateEventW(nullptr, TRUE, FALSE, nullptr)};
        SimpleFence done{simple_vtable, CreateEventW(nullptr, TRUE, FALSE, nullptr), 0};
        std::vector<unsigned char> d_cancel(piece, poison), d_keep(piece, poison);
        alignas(8) unsigned char r_cancel[0x48], r_keep[0x48];
        // Make r_cancel an EOF-failing request too, proving post-Submit cancel skips execution & failure.
        fill_file_request(r_cancel, d_cancel.data(), piece, file_ptr, file_bytes.size(), piece, 0xBEEFull);
        fill_file_request(r_keep, d_keep.data(), piece, file_ptr, 3 * piece, piece, 0xFEEDull);

        enqueue_signal(queue, &gate, 10);
        enqueue_req(queue, r_cancel);
        enqueue_req(queue, r_keep);
        enqueue_status(queue, status, 5);
        enqueue_signal(queue, &done, 11);
        submit(queue);

        ok = expect(WaitForSingleObject(gate.entered, timeout_ms) == WAIT_OBJECT_0, "case 6 worker entered gate") && ok;
        cancel(queue, ~0ull, 0xBEEFull);
        SetEvent(gate.release);
        ok = expect(WaitForSingleObject(done.event, timeout_ms) == WAIT_OBJECT_0 && done.value == 11,
                    "case 6 worker drained cleanly after post-submit cancel") && ok;
        CloseHandle(gate.entered);
        CloseHandle(gate.release);
        CloseHandle(done.event);

        ok = expect(is_complete(status, 5) && get_hr(status, 5) == S_OK, "case 6 status S_OK") && ok;
        ok = expect(all_equal(d_cancel, poison), "case 6 post-submit cancelled request untouched") && ok;
        ok = expect(std::memcmp(d_keep.data(), file_bytes.data() + 3 * piece, piece) == 0,
                    "case 6 post-submit uncancelled request executed") && ok;
    }

    // 7. Mixed batch: cancelled valid request + uncancelled EOF-failing request -> cancelled stays
    //    untouched while uncancelled failure still surfaces as E_FAIL on status.
    {
        std::vector<unsigned char> d_cancel(piece, poison), d_fail(piece, poison), d_ok(piece, poison);
        alignas(8) unsigned char r_cancel[0x48], r_fail[0x48], r_ok[0x48];
        fill_file_request(r_cancel, d_cancel.data(), piece, file_ptr, 0, piece, 0x7777ull);
        fill_file_request(r_fail, d_fail.data(), piece, file_ptr, file_bytes.size(), piece, 0x8888ull);
        fill_file_request(r_ok, d_ok.data(), piece, file_ptr, 1 * piece, piece, 0x9999ull);
        enqueue_req(queue, r_cancel);
        enqueue_req(queue, r_fail);
        enqueue_req(queue, r_ok);
        cancel(queue, ~0ull, 0x7777ull);
        ok = expect(wait_submit(queue, status, 6, simple_vtable, 7), "case 7 completed") && ok;
        ok = expect(is_complete(status, 6) && get_hr(status, 6) == E_FAIL,
                    "case 7 uncancelled failure still reports E_FAIL") && ok;
        ok = expect(all_equal(d_cancel, poison), "case 7 cancelled request untouched") && ok;
        ok = expect(all_equal(d_fail, poison), "case 7 failed request untouched") && ok;
        ok = expect(std::memcmp(d_ok.data(), file_bytes.data() + 1 * piece, piece) == 0,
                    "case 7 valid sibling still completed") && ok;
    }

    reinterpret_cast<CloseFileFn>(static_cast<Object*>(file_ptr)->vtable[3])(file_ptr);
    DeleteFileW(data_path);
    DeleteFileW(log_path);

    if (ok) {
        std::puts("CancelRequestsWithTag: exact, masked, mismatch, wildcard-eof, post-cancel, post-submit and mixed cases passed");
    }
    return ok ? 0 : 1;
}
