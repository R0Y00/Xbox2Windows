// #2788: the DirectStorage stand-in frees its COM objects at final Release, the queue's worker leaves
// before the queue object is freed, and a pending entry keeps the status array and source file it uses
// alive after the guest releases them.
//
// Usage: startup_xcrdapi_lifetime XCRDAPI.dll [control]
//   control runs the same scenario with X2W_TEST_XCRDAPI_LIFETIME_MUTANT=naive, which deletes at final
//   Release without stopping the worker or retaining what an entry uses: the fixture must print FAIL.
//
// The live-object counters are the stand-in's fixture-only exports x2w_xcrdapi_live_objects(kind) and
// x2w_xcrdapi_live_workers(). No GPU or game data is used.
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
#include <vector>

namespace {

using u64 = std::uint64_t;
constexpr unsigned piece = 4096;
constexpr DWORD completion_timeout_ms = 15000;
constexpr unsigned factory_kind = 0x20, file_kind = 0x40, status_kind = 0x60, queue_kind = 0x80;

struct Object { void** vtable; };
using GetFactoryFn = long(__stdcall*)(const void*, void**);
using OpenFileFn = long(__stdcall*)(void*, const wchar_t*, const void*, void**);
using CreateStatusArrayFn = long(__stdcall*)(void*, unsigned, const char*, const void*, void**);
using CreateQueueFn = long(__stdcall*)(void*, const void*, const void*, void**);
using EnqueueRequestFn = void(__stdcall*)(void*, const void*);
using EnqueueStatusFn = void(__stdcall*)(void*, void*, unsigned);
using EnqueueSignalFn = void(__stdcall*)(void*, void*, u64);
using SubmitFn = void(__stdcall*)(void*);
using ReleaseFn = unsigned long(__stdcall*)(void*);
using IsCompleteFn = bool(__stdcall*)(void*, unsigned);
using GetHResultFn = long(__stdcall*)(void*, unsigned);
using LiveObjectsFn = unsigned long(__cdecl*)(unsigned);
using LiveWorkersFn = unsigned long(__cdecl*)();

bool expect(bool condition, const char* what)
{
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", what);
    return condition;
}

unsigned long live(LiveObjectsFn fn, unsigned kind) { return fn(kind); }

struct Fence {
    void** vtable;
    volatile LONG signaled;
    u64 value;
    static long __stdcall signal(void* self, u64 value)
    {
        auto* fence = static_cast<Fence*>(self);
        fence->value = value;
        InterlockedExchange(&fence->signaled, 1);
        return 0;
    }
};

// Releases the queue from inside the worker's signal callback. The queue's final Release then runs on
// the worker itself, which must free the object as it exits instead of trying to join itself.
struct ReleasingFence {
    void** vtable;
    Object* queue;
    volatile LONG signaled;
    static long __stdcall signal(void* self, u64)
    {
        auto* fence = static_cast<ReleasingFence*>(self);
        reinterpret_cast<ReleaseFn>(fence->queue->vtable[2])(fence->queue);
        InterlockedExchange(&fence->signaled, 1);
        return 0;
    }
};

bool wait_fence(Fence& fence, DWORD timeout_ms)
{
    const DWORD start = GetTickCount();
    while (!fence.signaled && GetTickCount() - start < timeout_ms) Sleep(2);
    return fence.signaled != 0;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: startup_xcrdapi_lifetime XCRDAPI.dll [control]\n");
        return 2;
    }
    const bool control = argc == 3 && std::strcmp(argv[2], "control") == 0;
    if (control) {
        SetEnvironmentVariableA("X2W_TEST_XCRDAPI_LIFETIME_MUTANT", "naive");
    } else {
        SetEnvironmentVariableA("X2W_TEST_XCRDAPI_LIFETIME_MUTANT", nullptr);
    }

    wchar_t temp_dir[MAX_PATH] = {}, log_path[MAX_PATH] = {}, data_path[MAX_PATH] = {};
    GetTempPathW(MAX_PATH, temp_dir);
    std::swprintf(log_path, MAX_PATH, L"%lsx2w_lifetime_%lu.log", temp_dir, GetCurrentProcessId());
    std::swprintf(data_path, MAX_PATH, L"%lsx2w_lifetime_%lu.bin", temp_dir, GetCurrentProcessId());
    DeleteFileW(log_path);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path);

    std::vector<unsigned char> payload(piece);
    for (unsigned i = 0; i < piece; ++i) payload[i] = static_cast<unsigned char>((i * 13 + (i >> 5)) ^ 0x5a);
    {
        HANDLE file = CreateFileW(data_path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        DWORD written = 0;
        if (file == INVALID_HANDLE_VALUE ||
            !WriteFile(file, payload.data(), piece, &written, nullptr) || written != piece) {
            std::fprintf(stderr, "cannot write the test file\n");
            return 2;
        }
        CloseHandle(file);
    }

    HMODULE library = LoadLibraryA(argv[1]);
    auto* get_factory = library ? reinterpret_cast<GetFactoryFn>(GetProcAddress(library, "DStorageGetFactory")) : nullptr;
    auto* live_objects = library ? reinterpret_cast<LiveObjectsFn>(GetProcAddress(library, "x2w_xcrdapi_live_objects")) : nullptr;
    auto* live_workers = library ? reinterpret_cast<LiveWorkersFn>(GetProcAddress(library, "x2w_xcrdapi_live_workers")) : nullptr;
    void* factory_ptr = nullptr;
    if (!get_factory || !live_objects || !live_workers || get_factory(nullptr, &factory_ptr) != 0 || !factory_ptr) {
        std::fprintf(stderr, "cannot load the DirectStorage stand-in or its lifetime probes\n");
        return 2;
    }
    auto* factory = static_cast<Object*>(factory_ptr);
    unsigned char description[0x28] = {};
    *reinterpret_cast<std::uint16_t*>(description + 8) = 16;
    bool ok = true;

    ok = expect(live(live_objects, factory_kind) == 1, "the factory singleton is not reported alive") && ok;
    ok = expect(live(live_objects, 0) == 1, "objects are alive before any creation") && ok;

    // A. A file and a status array released by their only owner are freed.
    void* file0 = nullptr;
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &file0) != 0 || !file0) {
        std::fprintf(stderr, "cannot open the test file\n");
        return 2;
    }
    void* status0 = nullptr;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 4, "Lifetime", nullptr, &status0) != 0 || !status0)
        return 2;
    ok = expect(live(live_objects, file_kind) == 1, "the open file is not counted") && ok;
    ok = expect(live(live_objects, status_kind) == 1, "the status array is not counted") && ok;
    reinterpret_cast<ReleaseFn>(static_cast<Object*>(file0)->vtable[2])(file0);
    ok = expect(live(live_objects, file_kind) == 0, "a file released by its only owner was not freed") && ok;
    reinterpret_cast<ReleaseFn>(static_cast<Object*>(status0)->vtable[2])(status0);
    ok = expect(live(live_objects, status_kind) == 0, "a status array released by its only owner was not freed") && ok;
    if (!ok) return 1;

    // B. A submitted entry keeps its source file alive after the guest releases it, the status entry
    // completes, and the queue's worker has left by the time the queue's final Release returns. The
    // status array keeps the guest's own reference for the checks below, so it can be read after
    // completion; section C proves the entry retains it when the guest has released it.
    void* file1 = nullptr;
    void* status1 = nullptr;
    void* queue1 = nullptr;
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &file1) != 0 || !file1) return 2;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 4, "Pending", nullptr, &status1) != 0 || !status1) return 2;
    if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue1) != 0 || !queue1) return 2;
    auto* queue = static_cast<Object*>(queue1);
    auto* status = static_cast<Object*>(status1);
    void* fence_vtable[11] = {};
    fence_vtable[10] = reinterpret_cast<void*>(&Fence::signal);
    Fence first{}, barrier{};
    first.vtable = fence_vtable;
    barrier.vtable = fence_vtable;

    std::vector<unsigned char> destination(piece, 0xA5);
    alignas(8) unsigned char request[0x48] = {};
    *reinterpret_cast<unsigned char**>(request + 0x08) = destination.data();
    *reinterpret_cast<unsigned*>(request + 0x18) = piece;
    *reinterpret_cast<void**>(request + 0x20) = file1;
    *reinterpret_cast<u64*>(request + 0x30) = piece;
    reinterpret_cast<EnqueueRequestFn>(queue->vtable[3])(queue, request);
    reinterpret_cast<EnqueueStatusFn>(queue->vtable[4])(queue, status, 0);
    reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &first, 0x100);
    // A barrier behind the request: when it runs, the request's entry has released its references.
    reinterpret_cast<EnqueueSignalFn>(queue->vtable[5])(queue, &barrier, 0x101);

    // The guest drops the file while the request is pending.
    reinterpret_cast<ReleaseFn>(static_cast<Object*>(file1)->vtable[2])(file1);
    ok = expect(live(live_objects, file_kind) == 1,
                "a file referenced by a pending entry was freed by the guest's Release") && ok;
    if (!ok) return 1;

    reinterpret_cast<SubmitFn>(queue->vtable[6])(queue);
    ok = expect(wait_fence(first, completion_timeout_ms), "the request's fence was never signaled") && ok;
    ok = expect(wait_fence(barrier, completion_timeout_ms), "the barrier fence was never signaled") && ok;
    if (!ok) return 1;
    ok = expect(reinterpret_cast<IsCompleteFn>(status->vtable[3])(status, 0), "the status entry did not complete") && ok;
    ok = expect(reinterpret_cast<GetHResultFn>(status->vtable[4])(status, 0) == S_OK, "the request status is not S_OK") && ok;
    ok = expect(destination == payload, "the request did not write the file bytes") && ok;
    ok = expect(live(live_objects, file_kind) == 0, "the entry did not drop its file reference") && ok;
    ok = expect(live(live_objects, status_kind) == 1, "the status array lost the guest's reference before the checks") && ok;
    reinterpret_cast<ReleaseFn>(status->vtable[2])(status);
    ok = expect(live(live_objects, status_kind) == 0, "the status array was not freed at its last reference") && ok;
    ok = expect(live_workers() == 1, "the queue's worker is not reported running before the queue is released") && ok;
    if (!ok) return 1;

    reinterpret_cast<ReleaseFn>(queue->vtable[2])(queue);
    ok = expect(live(live_objects, queue_kind) == 0, "the queue was not freed at final Release") && ok;
    ok = expect(live_workers() == 0, "the queue's worker was still running after the queue was freed") && ok;
    if (!ok) return 1;

    // C. A queue released with entries that were never submitted drops their references too.
    void* file2 = nullptr;
    void* status2 = nullptr;
    void* queue2 = nullptr;
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &file2) != 0 || !file2) return 2;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 4, "Unsubmitted", nullptr, &status2) != 0 || !status2) return 2;
    if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue2) != 0 || !queue2) return 2;
    auto* queue_2 = static_cast<Object*>(queue2);
    alignas(8) unsigned char request2[0x48] = {};
    *reinterpret_cast<unsigned char**>(request2 + 0x08) = destination.data();
    *reinterpret_cast<unsigned*>(request2 + 0x18) = piece;
    *reinterpret_cast<void**>(request2 + 0x20) = file2;
    *reinterpret_cast<u64*>(request2 + 0x30) = piece;
    reinterpret_cast<EnqueueRequestFn>(queue_2->vtable[3])(queue_2, request2);
    reinterpret_cast<EnqueueStatusFn>(queue_2->vtable[4])(queue_2, static_cast<Object*>(status2), 1);
    reinterpret_cast<ReleaseFn>(static_cast<Object*>(file2)->vtable[2])(file2);
    reinterpret_cast<ReleaseFn>(static_cast<Object*>(status2)->vtable[2])(status2);
    ok = expect(live(live_objects, file_kind) == 1, "an unsubmitted entry did not retain its file") && ok;
    ok = expect(live(live_objects, status_kind) == 1, "an unsubmitted entry did not retain its status array") && ok;
    reinterpret_cast<ReleaseFn>(queue_2->vtable[2])(queue_2);
    ok = expect(live(live_objects, file_kind) == 0, "releasing the queue left its file alive") && ok;
    ok = expect(live(live_objects, status_kind) == 0, "releasing the queue left its status array alive") && ok;
    ok = expect(live_workers() == 0, "an unused queue's worker outlived the queue") && ok;
    ok = expect(live(live_objects, 0) == 1, "objects other than the factory singleton outlived every release") && ok;

    // D. The last external reference can be released from the queue's own worker (a fence callback).
    // The worker cannot join itself: it must stop, drop the entries that never ran, and free the queue.
    void* file3 = nullptr;
    void* status3 = nullptr;
    void* queue3 = nullptr;
    if (reinterpret_cast<OpenFileFn>(factory->vtable[4])(factory, data_path, nullptr, &file3) != 0 || !file3) return 2;
    if (reinterpret_cast<CreateStatusArrayFn>(factory->vtable[5])(factory, 4, "WorkerRelease", nullptr, &status3) != 0 || !status3) return 2;
    if (reinterpret_cast<CreateQueueFn>(factory->vtable[10])(factory, description, nullptr, &queue3) != 0 || !queue3) return 2;
    auto* queue_3 = static_cast<Object*>(queue3);
    auto* status_3 = static_cast<Object*>(status3);
    void* release_vtable[11] = {};
    release_vtable[10] = reinterpret_cast<void*>(&ReleasingFence::signal);
    ReleasingFence releasing{};
    releasing.vtable = release_vtable;
    releasing.queue = queue_3;
    alignas(8) unsigned char request3[0x48] = {};
    *reinterpret_cast<unsigned char**>(request3 + 0x08) = destination.data();
    *reinterpret_cast<unsigned*>(request3 + 0x18) = piece;
    *reinterpret_cast<void**>(request3 + 0x20) = file3;
    *reinterpret_cast<u64*>(request3 + 0x30) = piece;
    reinterpret_cast<EnqueueRequestFn>(queue_3->vtable[3])(queue_3, request3);
    reinterpret_cast<EnqueueStatusFn>(queue_3->vtable[4])(queue_3, status_3, 0);
    reinterpret_cast<EnqueueSignalFn>(queue_3->vtable[5])(queue_3, &releasing, 0x200);
    reinterpret_cast<ReleaseFn>(static_cast<Object*>(file3)->vtable[2])(file3);
    reinterpret_cast<ReleaseFn>(status_3->vtable[2])(status3);
    reinterpret_cast<SubmitFn>(queue_3->vtable[6])(queue_3);
    const DWORD release_start = GetTickCount();
    while ((!releasing.signaled || live(live_objects, queue_kind) || live_workers()) &&
           GetTickCount() - release_start < completion_timeout_ms)
        Sleep(2);
    ok = expect(releasing.signaled != 0, "the releasing fence never ran") && ok;
    ok = expect(live(live_objects, queue_kind) == 0, "a queue released from its own worker was not freed") && ok;
    ok = expect(live_workers() == 0, "a queue's worker outlived a release from that worker") && ok;
    ok = expect(live(live_objects, file_kind) == 0, "the worker-release queue left its file alive") && ok;
    ok = expect(live(live_objects, status_kind) == 0, "the worker-release queue left its status array alive") && ok;

    reinterpret_cast<ReleaseFn>(factory->vtable[2])(factory);
    DeleteFileW(data_path);
    DeleteFileW(log_path);
    if (!ok) return 1;
    std::puts("xcrdapi objects are freed at final Release; pending entries kept their status array and file alive; "
              "the queue worker left before the queue was freed");
    return 0;
}
