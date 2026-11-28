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
#include <psapi.h>

#include "startup_stubs/dstorage_error_record.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

namespace {

using u64 = std::uint64_t;
using x2w_startup::StorageErrorRecord;
using x2w_startup::StorageRequest;
constexpr DWORD timeout_ms = 15000;
constexpr unsigned piece = 256;
constexpr unsigned char poison = 0xA5;
constexpr unsigned file_kind = 0x40;

struct Object { void** vtable; };
using ReleaseFn = unsigned long(__stdcall*)(void*);
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
using GetErrorEventFn = HANDLE(__stdcall*)(void*);
using RetrieveErrorRecordFn = void(__stdcall*)(void*, StorageErrorRecord*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);
using LiveObjectsFn = unsigned long(__cdecl*)(unsigned);

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

std::uint32_t adler32(const unsigned char* data, std::size_t size)
{
    std::uint32_t a = 1, b = 0;
    for (std::size_t i = 0; i < size; ++i) {
        a = (a + data[i]) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

std::vector<unsigned char> stored_zlib(const std::vector<unsigned char>& plain)
{
    const auto size = static_cast<unsigned>(plain.size());
    std::vector<unsigned char> out{0x78, 0x01, 0x01, static_cast<unsigned char>(size), static_cast<unsigned char>(size >> 8),
                                   static_cast<unsigned char>(~size), static_cast<unsigned char>((~size) >> 8)};
    out.insert(out.end(), plain.begin(), plain.end());
    const std::uint32_t sum = adler32(plain.data(), plain.size());
    for (int shift = 24; shift >= 0; shift -= 8) out.push_back(static_cast<unsigned char>(sum >> shift));
    return out;
}

bool current_commit_bytes(SIZE_T& out)
{
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)))
        return false;
    out = counters.PagefileUsage;
    return true;
}

StorageRequest make_file_request(void* dest, unsigned dest_size, void* file_obj,
                                 u64 offset, unsigned src_size, u64 tag)
{
    StorageRequest req{};
    req.options = 0;
    req.destination = dest;
    req.destination_size = dest_size;
    req.source = file_obj;
    req.file_offset = offset;
    req.source_size = src_size;
    req.cancellation_tag = tag;
    return req;
}

StorageRequest make_memory_request(void* dest, unsigned dest_size, const void* src,
                                   unsigned src_size, u64 tag, u64 extra_options = 0)
{
    StorageRequest req{};
    req.options = 0x400000ull | extra_options;
    req.destination = dest;
    req.destination_size = dest_size;
    req.source = const_cast<void*>(src);
    req.source_size = src_size;
    req.cancellation_tag = tag;
    return req;
}

bool expect_no_error_record(Object* queue, HANDLE error_event, const char* label)
{
    auto retrieve = reinterpret_cast<RetrieveErrorRecordFn>(queue->vtable[10]);
    StorageErrorRecord record{};
    retrieve(queue, &record);
    bool ok = true;
    ok = expect(WaitForSingleObject(error_event, 0) == WAIT_TIMEOUT, label) && ok;
    ok = expect(record.failure_count == 0 && record.first_failure.command_type == x2w_startup::command_type_none,
                label) && ok;
    return ok;
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
    auto* live_objects = library ? reinterpret_cast<LiveObjectsFn>(GetProcAddress(library, "x2w_xcrdapi_live_objects")) : nullptr;
    void* factory_ptr = nullptr;
    if (!get_factory || !live_objects || get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) {
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
    auto get_error_event = reinterpret_cast<GetErrorEventFn>(queue->vtable[9]);
    auto retrieve_error = reinterpret_cast<RetrieveErrorRecordFn>(queue->vtable[10]);
    auto is_complete = reinterpret_cast<IsCompleteFn>(status->vtable[3]);
    auto get_hr = reinterpret_cast<GetHResultFn>(status->vtable[4]);
    HANDLE error_event = get_error_event(queue);
    if (!error_event) {
        std::fprintf(stderr, "cannot acquire DirectStorage error event\n");
        DeleteFileW(data_path);
        return 1;
    }

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
        const StorageRequest r0 = make_file_request(d0.data(), piece, file_ptr, 0 * piece, piece, 0x1001ull);
        const StorageRequest r1 = make_file_request(d1.data(), piece, file_ptr, 1 * piece, piece, 0x2002ull);
        const StorageRequest r2 = make_memory_request(d2.data(), piece, file_bytes.data() + 2 * piece, piece, 0x3003ull);
        enqueue_req(queue, &r0);
        enqueue_req(queue, &r1);
        enqueue_req(queue, &r2);
        cancel(queue, ~0ull, 0x2002ull);
        ok = expect(wait_submit(queue, status, 0, simple_vtable, 1), "case 1 completed") && ok;
        ok = expect(is_complete(status, 0) && get_hr(status, 0) == S_OK, "case 1 status S_OK") && ok;
        ok = expect(std::memcmp(d0.data(), file_bytes.data() + 0 * piece, piece) == 0, "case 1 r0 bytes") && ok;
        if (control_expect_uncancelled) {
            // Negative control: must FAIL because r1 was cancelled.
            ok = expect(std::memcmp(d1.data(), file_bytes.data() + 1 * piece, piece) == 0,
                        "control expected cancelled r1 to have executed") && ok;
            CloseHandle(error_event);
            reinterpret_cast<CloseFileFn>(static_cast<Object*>(file_ptr)->vtable[3])(file_ptr);
            reinterpret_cast<ReleaseFn>(static_cast<Object*>(file_ptr)->vtable[2])(file_ptr);
            reinterpret_cast<ReleaseFn>(status->vtable[2])(status);
            reinterpret_cast<ReleaseFn>(queue->vtable[2])(queue);
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
        const StorageRequest r1 = make_file_request(g1.data(), piece, file_ptr, 0 * piece, piece, 0x0000000100000001ull);
        const StorageRequest r2a = make_file_request(g2a.data(), piece, file_ptr, 1 * piece, piece, 0x0000000200000001ull);
        const StorageRequest r2b = make_file_request(g2b.data(), piece, file_ptr, 2 * piece, piece, 0x0000000200000099ull);
        const StorageRequest r3 = make_file_request(g3.data(), piece, file_ptr, 3 * piece, piece, 0x0000000300000001ull);
        enqueue_req(queue, &r1);
        enqueue_req(queue, &r2a);
        enqueue_req(queue, &r2b);
        enqueue_req(queue, &r3);
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
        const StorageRequest req = make_file_request(dst.data(), piece, file_ptr, 0 * piece, piece, 0xFFull);
        enqueue_req(queue, &req);
        cancel(queue, 0x0Full, 0x1Full);  // bit 4 set in value where mask is 0 -> (tag & mask) != value
        ok = expect(wait_submit(queue, status, 2, simple_vtable, 3), "case 3 completed") && ok;
        ok = expect(is_complete(status, 2) && get_hr(status, 2) == S_OK, "case 3 status S_OK") && ok;
        ok = expect(std::memcmp(dst.data(), file_bytes.data(), piece) == 0, "case 3 unmasked bit did not cancel") && ok;
    }

    // 4. Wildcard cancel-all (mask=0, value=0): cancels a valid request AND an EOF-failing request;
    //    the cancelled failing request must NOT run, must NOT fail status, and must NOT record an error.
    {
        std::vector<unsigned char> d_ok(piece, poison), d_eof(piece, poison);
        const StorageRequest r_ok = make_file_request(d_ok.data(), piece, file_ptr, 0, piece, 0xABCDull);
        const StorageRequest r_eof = make_file_request(d_eof.data(), piece, file_ptr, file_bytes.size(), piece, 0xDEADull);
        enqueue_req(queue, &r_ok);
        enqueue_req(queue, &r_eof);
        cancel(queue, 0, 0);
        ok = expect(wait_submit(queue, status, 3, simple_vtable, 4), "case 4 completed") && ok;
        ok = expect(is_complete(status, 3) && get_hr(status, 3) == S_OK,
                    "case 4 cancelled EOF request did not fail status") && ok;
        ok = expect(all_equal(d_ok, poison) && all_equal(d_eof, poison), "case 4 all destinations untouched") && ok;
        ok = expect_no_error_record(queue, error_event, "case 4 cancelled EOF did not record error or signal event") && ok;
        const std::string log_so_far = read_text(log_path);
        ok = expect(log_so_far.find("read failed") == std::string::npos,
                    "case 4 cancelled EOF request never executed read") && ok;
    }

    // 5. Temporal boundary: a request enqueued AFTER CancelRequestsWithTag with the SAME tag must run.
    {
        std::vector<unsigned char> before_dst(piece, poison), after_dst(piece, poison);
        const StorageRequest r_before = make_file_request(before_dst.data(), piece, file_ptr, 1 * piece, piece, 0xCAFEull);
        const StorageRequest r_after = make_file_request(after_dst.data(), piece, file_ptr, 2 * piece, piece, 0xCAFEull);
        enqueue_req(queue, &r_before);
        cancel(queue, ~0ull, 0xCAFEull);
        enqueue_req(queue, &r_after);
        ok = expect(wait_submit(queue, status, 4, simple_vtable, 5), "case 5 completed") && ok;
        ok = expect(is_complete(status, 4) && get_hr(status, 4) == S_OK, "case 5 status S_OK") && ok;
        ok = expect(all_equal(before_dst, poison), "case 5 pre-cancel request cancelled") && ok;
        ok = expect(std::memcmp(after_dst.data(), file_bytes.data() + 2 * piece, piece) == 0,
                    "case 5 post-cancel request with same tag still executed") && ok;
    }

    // 6. Post-Submit cancellation: block worker on a gate fence so requests are deterministically in
    //    `work->submitted`, cancel the matching one live, release gate, and verify clean drain, S_OK,
    //    and no error record or error event.
    {
        GateFence gate{gate_vtable, CreateEventW(nullptr, TRUE, FALSE, nullptr),
                       CreateEventW(nullptr, TRUE, FALSE, nullptr)};
        SimpleFence done{simple_vtable, CreateEventW(nullptr, TRUE, FALSE, nullptr), 0};
        std::vector<unsigned char> d_cancel(piece, poison), d_keep(piece, poison);
        // Make r_cancel an EOF-failing request too, proving post-Submit cancel skips execution & failure.
        const StorageRequest r_cancel = make_file_request(d_cancel.data(), piece, file_ptr, file_bytes.size(), piece, 0xBEEFull);
        const StorageRequest r_keep = make_file_request(d_keep.data(), piece, file_ptr, 3 * piece, piece, 0xFEEDull);

        enqueue_signal(queue, &gate, 10);
        enqueue_req(queue, &r_cancel);
        enqueue_req(queue, &r_keep);
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
        ok = expect_no_error_record(queue, error_event, "case 6 post-submit cancelled EOF did not record error or signal event") && ok;
    }

    // 7. Mixed batch: cancelled valid request + uncancelled EOF-failing request -> cancelled stays
    //    untouched while uncancelled failure surfaces as ERROR_HANDLE_EOF on status and DSTORAGE_ERROR_RECORD.
    {
        std::vector<unsigned char> d_cancel(piece, poison), d_fail(piece, poison), d_ok(piece, poison);
        const StorageRequest r_cancel = make_file_request(d_cancel.data(), piece, file_ptr, 0, piece, 0x7777ull);
        const StorageRequest r_fail = make_file_request(d_fail.data(), piece, file_ptr, file_bytes.size(), piece, 0x8888ull);
        const StorageRequest r_ok = make_file_request(d_ok.data(), piece, file_ptr, 1 * piece, piece, 0x9999ull);
        enqueue_req(queue, &r_cancel);
        enqueue_req(queue, &r_fail);
        enqueue_req(queue, &r_ok);
        cancel(queue, ~0ull, 0x7777ull);
        ok = expect(wait_submit(queue, status, 6, simple_vtable, 7), "case 7 completed") && ok;
        const HRESULT expected_eof = HRESULT_FROM_WIN32(ERROR_HANDLE_EOF);
        ok = expect(is_complete(status, 6) && get_hr(status, 6) == expected_eof,
                    "case 7 uncancelled EOF failure reports ERROR_HANDLE_EOF") && ok;
        ok = expect(all_equal(d_cancel, poison), "case 7 cancelled request untouched") && ok;
        ok = expect(all_equal(d_fail, poison), "case 7 failed request untouched") && ok;
        ok = expect(std::memcmp(d_ok.data(), file_bytes.data() + 1 * piece, piece) == 0,
                    "case 7 valid sibling still completed") && ok;
        ok = expect(WaitForSingleObject(error_event, 0) == WAIT_OBJECT_0,
                    "case 7 uncancelled EOF failure signaled error event") && ok;
        StorageErrorRecord record{};
        retrieve_error(queue, &record);
        ok = expect(record.failure_count == 1 &&
                    record.first_failure.result == expected_eof &&
                    record.first_failure.command_type == x2w_startup::command_type_request &&
                    record.first_failure.cancellation_tag == 0x8888ull,
                    "case 7 error record captured only uncancelled failure with its tag") && ok;
        ok = expect(WaitForSingleObject(error_event, 0) == WAIT_TIMEOUT,
                    "case 7 RetrieveErrorRecord reset error event") && ok;
    }

    // 8. Source-built zlib payload with two distinct nonzero CancellationTags (one carrying a ~2 GiB
    //    low 32-bit word): accepted pair must complete identically with bounded decode commit, and
    //    its cancellation counterpart must leave the destination untouched and preserve status/fence order.
    {
        constexpr std::size_t zlib_plain_size = 4096;
        std::vector<unsigned char> zlib_plain(zlib_plain_size);
        for (std::size_t i = 0; i < zlib_plain_size; ++i)
            zlib_plain[i] = static_cast<unsigned char>(((i * 37) ^ (i >> 4) ^ 0x3C) & 0xFF);
        const std::vector<unsigned char> zlib_stream = stored_zlib(zlib_plain);

        std::vector<unsigned char> z_a(zlib_plain_size, poison), z_b(zlib_plain_size, poison);
        const StorageRequest req_a = make_memory_request(
            z_a.data(), static_cast<unsigned>(z_a.size()), zlib_stream.data(),
            static_cast<unsigned>(zlib_stream.size()), 0x1111111100000001ull, 1ull);
        const StorageRequest req_b = make_memory_request(
            z_b.data(), static_cast<unsigned>(z_b.size()), zlib_stream.data(),
            static_cast<unsigned>(zlib_stream.size()), 0x222222227FFFFFFFull, 1ull);

        SIZE_T commit_before = 0, commit_after = 0;
        const bool have_before = current_commit_bytes(commit_before);
        enqueue_req(queue, &req_a);
        enqueue_req(queue, &req_b);
        ok = expect(wait_submit(queue, status, 7, simple_vtable, 8), "case 8a zlib accepted pair completed") && ok;
        const bool have_after = current_commit_bytes(commit_after);
        ok = expect(is_complete(status, 7) && get_hr(status, 7) == S_OK, "case 8a zlib accepted pair S_OK") && ok;
        ok = expect(z_a == zlib_plain && z_b == zlib_plain,
                    "case 8a zlib decoded bytes identical across distinct nonzero tags") && ok;
        if (have_before && have_after) {
            const SIZE_T growth = commit_after > commit_before ? commit_after - commit_before : 0;
            ok = expect(growth < (64u << 20), "case 8a zlib decode sizing stayed bounded despite 2 GiB low tag word") && ok;
        }
        ok = expect_no_error_record(queue, error_event, "case 8a zlib accepted pair produced no error record") && ok;

        std::vector<unsigned char> z_cancel(zlib_plain_size, poison), z_keep(zlib_plain_size, poison);
        const StorageRequest req_cancel = make_memory_request(
            z_cancel.data(), static_cast<unsigned>(z_cancel.size()), zlib_stream.data(),
            static_cast<unsigned>(zlib_stream.size()), 0x333333337FFFFFFFull, 1ull);
        const StorageRequest req_keep = make_memory_request(
            z_keep.data(), static_cast<unsigned>(z_keep.size()), zlib_stream.data(),
            static_cast<unsigned>(zlib_stream.size()), 0x4444444400000001ull, 1ull);
        enqueue_req(queue, &req_cancel);
        enqueue_req(queue, &req_keep);
        cancel(queue, ~0ull, 0x333333337FFFFFFFull);
        ok = expect(wait_submit(queue, status, 8, simple_vtable, 9), "case 8b zlib cancel counterpart completed") && ok;
        ok = expect(is_complete(status, 8) && get_hr(status, 8) == S_OK, "case 8b zlib cancel counterpart S_OK") && ok;
        ok = expect(all_equal(z_cancel, poison), "case 8b cancelled zlib destination untouched") && ok;
        ok = expect(z_keep == zlib_plain, "case 8b surviving zlib sibling completed") && ok;
        ok = expect_no_error_record(queue, error_event, "case 8b cancelled zlib produced no error record") && ok;
    }

    // 9. Source-file lifetime (#2788): cancelled pre-Submit (`waiting`) and post-Submit (`submitted`)
    //    requests must release the file references retained at EnqueueRequest.
    {
        const unsigned long files_before = live_objects(file_kind);
        void* temp_file = nullptr;
        ok = expect(reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &temp_file) == S_OK &&
                    temp_file != nullptr, "case 9 opened temporary file") && ok;
        if (temp_file) {
            GateFence gate{gate_vtable, CreateEventW(nullptr, TRUE, FALSE, nullptr),
                           CreateEventW(nullptr, TRUE, FALSE, nullptr)};
            SimpleFence done{simple_vtable, CreateEventW(nullptr, TRUE, FALSE, nullptr), 0};
            std::vector<unsigned char> d_wait(piece, poison), d_sub(piece, poison);
            const StorageRequest r_wait = make_file_request(d_wait.data(), piece, temp_file, 0, piece, 0x55550001ull);
            const StorageRequest r_sub = make_file_request(d_sub.data(), piece, temp_file, piece, piece, 0x55550002ull);

            enqueue_req(queue, &r_wait);
            enqueue_signal(queue, &gate, 20);
            enqueue_req(queue, &r_sub);
            enqueue_status(queue, status, 9);
            enqueue_signal(queue, &done, 21);

            // Drop guest ownership while both requests still hold internal references.
            reinterpret_cast<ReleaseFn>(static_cast<Object*>(temp_file)->vtable[2])(temp_file);
            ok = expect(live_objects(file_kind) == files_before + 1,
                        "case 9 pending requests retained temp_file after guest Release") && ok;

            cancel(queue, ~0ull, 0x55550001ull);
            submit(queue);
            ok = expect(WaitForSingleObject(gate.entered, timeout_ms) == WAIT_OBJECT_0, "case 9 worker entered gate") && ok;
            cancel(queue, ~0ull, 0x55550002ull);
            SetEvent(gate.release);
            ok = expect(WaitForSingleObject(done.event, timeout_ms) == WAIT_OBJECT_0 && done.value == 21,
                        "case 9 drained after cancelling both requests") && ok;
            CloseHandle(gate.entered);
            CloseHandle(gate.release);
            CloseHandle(done.event);

            ok = expect(is_complete(status, 9) && get_hr(status, 9) == S_OK, "case 9 status S_OK") && ok;
            ok = expect(all_equal(d_wait, poison) && all_equal(d_sub, poison), "case 9 destinations untouched") && ok;
            ok = expect(live_objects(file_kind) == files_before,
                        "case 9 cancelled waiting and submitted requests released file references") && ok;
        }
    }

    CloseHandle(error_event);
    reinterpret_cast<CloseFileFn>(static_cast<Object*>(file_ptr)->vtable[3])(file_ptr);
    reinterpret_cast<ReleaseFn>(static_cast<Object*>(file_ptr)->vtable[2])(file_ptr);
    reinterpret_cast<ReleaseFn>(status->vtable[2])(status);
    reinterpret_cast<ReleaseFn>(queue->vtable[2])(queue);
    DeleteFileW(data_path);
    DeleteFileW(log_path);

    if (ok) {
        std::puts("CancelRequestsWithTag: exact, masked, mismatch, wildcard-eof, post-cancel, post-submit, mixed, zlib-tag and lifetime cases passed");
    }
    return ok ? 0 : 1;
}
