// Controlled DirectStorage factory and queue adapter provider for startup and staging tests.
//
// Public sources:
// - Microsoft DirectStorage (include/dstorage.h, MIT License):
//   https://github.com/microsoft/DirectStorage
//   Defines public PC IDStorageFactory {6924ea0c-c3cd-4826-b10a-f64f4ed927c1},
//   IDStorageQueue {cf90e079-90c1-4320-be50-b01ec69e0637}, DSTORAGE_QUEUE_DESC,
//   and public COM vtable layouts.
// - Microsoft Learn GDK DirectStorage Overview:
//   https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview
//   Describes factory acquisition, staging buffer configuration, queue dispatch,
//   delayed submit, and auto-submit thresholds.
//
// Convention vs Evidence:
// - Windows runtime evidence: export binding of XCRDAPI!DStorageGetFactory and
//   cpu_exe_io!finish_exe; execution under Windows native loader and pe_map.
// - Public DirectStorage evidence: PC IDStorageFactory IID, explicit submission, and the
//   documented half-capacity threshold. Refusal of that IID and preservation of *ppv
//   are recorder behavior.
// - Recorder convention: fresh fixture-owned interface identifiers (IID_FixtureFactory and
//   IID_FixtureQueue) and the adapter vtable layout (slots 0..3, 6, 8 matching the recorder
//   convention and public dstorage.h slots). A public source would be needed before a fixture
//   or production adapter presents other IDs or slots as a platform contract.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <unknwn.h>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {
// Fresh fixture-owned interface identifiers
// IID_FixtureFactory: {ca891497-6f8f-4d03-b1a2-f239b4ce41af}
const GUID IID_FixtureFactory = {
    0xca891497, 0x6f8f, 0x4d03, {0xb1, 0xa2, 0xf2, 0x39, 0xb4, 0xce, 0x41, 0xaf}
};

// IID_FixtureQueue: {82af1362-6445-4b96-9fbf-5d2d416b605e}
const GUID IID_FixtureQueue = {
    0x82af1362, 0x6445, 0x4b96, {0x9f, 0xbf, 0x5d, 0x2d, 0x41, 0x6b, 0x60, 0x5e}
};

// Public MIT DirectStorage PC IDStorageFactory: {6924ea0c-c3cd-4826-b10a-f64f4ed927c1}
// Source: https://github.com/microsoft/DirectStorage (include/dstorage.h, MIT license)
const GUID IID_IDStorageFactory_PC = {
    0x6924ea0c, 0xc3cd, 0x4826, {0xb1, 0x0a, 0xf6, 0x4f, 0x4e, 0xd9, 0x27, 0xc1}
};

// Public MIT DirectStorage PC IDStorageQueue: {cf90e079-90c1-4320-be50-b01ec69e0637}
const GUID IID_IDStorageQueue_PC = {
    0xcf90e079, 0x90c1, 0x4320, {0xbe, 0x50, 0xb0, 0x1e, 0xc6, 0x9e, 0x06, 0x37}
};

// Test request structure matching DSTORAGE_REQUEST in dstorage.h
#pragma pack(push, 8)
struct TestRequest {
    uint32_t OptionsLow;
    uint32_t OptionsHigh;
    const void* SourceBuffer;
    uint32_t SourceSize;
    uint32_t SourcePad;
    void* DestBuffer;
    uint32_t DestSize;
    uint32_t DestPad;
    uint32_t UncompressedSize;
    uint32_t Pad;
    uint64_t CancellationTag;
    const char* Name;
};
#pragma pack(pop)

static_assert(offsetof(TestRequest, SourceBuffer) == 8, "SourceBuffer offset mismatch");
static_assert(offsetof(TestRequest, DestBuffer) == 24, "DestBuffer offset mismatch");

// Test provider counters and records
volatile LONG cfg_factory_success = 1;
volatile LONG cfg_queue_success = 1;
volatile LONG cfg_eager_completion = 0;

volatile LONG factory_calls = 0;
volatile LONG queue_calls = 0;
volatile LONG staged_requests = 0;
volatile LONG submitted_batches = 0;
volatile uint32_t current_staging_buffer_size = 0;
volatile LONG active_queues = 0;

// Forward declarations
struct MockQueue;
struct MockFactory;

void* factory_vtable[16] = {};
void* queue_vtable[16] = {};

struct MockQueue {
    void** vtable;
    volatile LONG ref_count;
    uint16_t capacity;
    uint32_t pending_requests;
    volatile LONG is_completed;
    const TestRequest* last_request;

    MockQueue(uint16_t cap)
        : vtable(queue_vtable), ref_count(1), capacity(cap), pending_requests(0),
          is_completed(0), last_request(nullptr)
    {
        InterlockedIncrement(&active_queues);
        InterlockedIncrement(&queue_calls);
    }

    ~MockQueue()
    {
        InterlockedDecrement(&active_queues);
    }

    static HRESULT WINAPI QueryInterface(MockQueue* self, const GUID& riid, void** ppv)
    {
        if (!ppv) return E_POINTER;
        if (IsEqualGUID(riid, IID_IUnknown) ||
            IsEqualGUID(riid, IID_FixtureQueue) ||
            IsEqualGUID(riid, IID_IDStorageQueue_PC))
        {
            AddRef(self);
            *ppv = self;
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    static ULONG WINAPI AddRef(MockQueue* self)
    {
        return InterlockedIncrement(&self->ref_count);
    }

    static ULONG WINAPI Release(MockQueue* self)
    {
        ULONG r = InterlockedDecrement(&self->ref_count);
        if (r == 0) {
            delete self;
        }
        return r;
    }

    static void WINAPI EnqueueRequest(MockQueue* self, const void* request)
    {
        InterlockedIncrement(&staged_requests);
        self->pending_requests++;
        self->last_request = reinterpret_cast<const TestRequest*>(request);

        // If eager-provider control is active, simulate a faulty provider completing before Submit:
        if (cfg_eager_completion) {
            Submit(self);
            return;
        }

        // DirectStorage auto-submission triggers when reaching half capacity
        if (self->capacity > 0 && self->pending_requests >= (self->capacity / 2)) {
            Submit(self);
        }
    }

    static void WINAPI EnqueueStatus(MockQueue* self, void* statusArray, uint32_t index)
    {
        (void)self;
        (void)statusArray;
        (void)index;
    }

    static void WINAPI EnqueueSignal(MockQueue* self, void* fence, uint64_t value)
    {
        (void)self;
        (void)fence;
        (void)value;
    }

    static void WINAPI Submit(MockQueue* self)
    {
        InterlockedIncrement(&submitted_batches);
        self->pending_requests = 0;

        // Perform memory readback transfer if request is present
        if (self->last_request &&
            self->last_request->SourceBuffer &&
            self->last_request->DestBuffer &&
            self->last_request->SourceSize > 0)
        {
            uint32_t copy_size = self->last_request->SourceSize;
            if (self->last_request->DestSize > 0 &&
                self->last_request->DestSize < copy_size)
            {
                copy_size = self->last_request->DestSize;
            }
            std::memcpy(self->last_request->DestBuffer,
                        self->last_request->SourceBuffer,
                        copy_size);
        }

        InterlockedExchange(&self->is_completed, 1);
    }

    static void WINAPI CancelRequestsWithTag(MockQueue* self, uint64_t mask, uint64_t tag)
    {
        (void)self;
        (void)mask;
        (void)tag;
    }
};

static_assert(offsetof(MockQueue, is_completed) == 0x14, "MockQueue is_completed must be at offset 0x14");

struct QueueDesc {
    uint32_t SourceType;
    uint16_t Capacity;
    int8_t Priority;
    const char* Name;
};

struct MockFactory {
    void** vtable;
    volatile LONG ref_count;

    MockFactory() : vtable(factory_vtable), ref_count(1) {}

    static HRESULT WINAPI DummyMethod()
    {
        return S_OK;
    }

    static HRESULT WINAPI QueryInterface(MockFactory* self, const GUID& riid, void** ppv)
    {
        if (!ppv) return E_POINTER;
        if (IsEqualGUID(riid, IID_IUnknown) || IsEqualGUID(riid, IID_FixtureFactory)) {
            AddRef(self);
            *ppv = self;
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    static ULONG WINAPI AddRef(MockFactory* self)
    {
        return InterlockedIncrement(&self->ref_count);
    }

    static ULONG WINAPI Release(MockFactory* self)
    {
        ULONG r = InterlockedDecrement(&self->ref_count);
        if (r == 0) {
            delete self;
        }
        return r;
    }

    static HRESULT WINAPI OpenFile(MockFactory* self, const wchar_t* path, const GUID& riid, void** ppv)
    {
        (void)self;
        (void)path;
        (void)riid;
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        return S_OK;
    }

    static HRESULT WINAPI SetStagingBufferSize(MockFactory* self, uint32_t size)
    {
        (void)self;
        // DirectStorage contract: staging buffer cannot be resized while queues or files are active
        if (active_queues > 0) {
            return E_FAIL;
        }
        current_staging_buffer_size = size;
        return S_OK;
    }

    static HRESULT WINAPI CreateQueue(MockFactory* self, const QueueDesc* desc, const GUID& riid, void** ppv)
    {
        (void)self;
        if (!ppv) return E_POINTER;
        if (!cfg_queue_success) {
            *ppv = nullptr;
            return E_FAIL;
        }
        if (!IsEqualGUID(riid, IID_FixtureQueue) && !IsEqualGUID(riid, IID_IDStorageQueue_PC)) {
            *ppv = nullptr;
            return E_NOINTERFACE;
        }

        uint16_t cap = desc ? desc->Capacity : 128;
        *ppv = new MockQueue(cap);
        return S_OK;
    }
};

void init_vtables()
{
    static bool initialized = false;
    if (initialized) return;

    for (int i = 0; i < 16; ++i) {
        factory_vtable[i] = reinterpret_cast<void*>(&MockFactory::DummyMethod);
        queue_vtable[i] = reinterpret_cast<void*>(&MockFactory::DummyMethod);
    }

    // Recorder convention vtable slot ordering:
    // IDStorageFactory:
    // Slot 0..2: IUnknown
    // Slot 3: CreateQueue
    // Slot 4: OpenFile
    // Slot 8: SetStagingBufferSize
    factory_vtable[0] = reinterpret_cast<void*>(&MockFactory::QueryInterface);
    factory_vtable[1] = reinterpret_cast<void*>(&MockFactory::AddRef);
    factory_vtable[2] = reinterpret_cast<void*>(&MockFactory::Release);
    factory_vtable[3] = reinterpret_cast<void*>(&MockFactory::CreateQueue);
    factory_vtable[4] = reinterpret_cast<void*>(&MockFactory::OpenFile);
    factory_vtable[8] = reinterpret_cast<void*>(&MockFactory::SetStagingBufferSize);
    factory_vtable[10] = reinterpret_cast<void*>(&MockFactory::CreateQueue); // alias

    // IDStorageQueue:
    // Slot 0..2: IUnknown
    // Slot 3: EnqueueRequest
    // Slot 4: EnqueueStatus
    // Slot 5: EnqueueSignal
    // Slot 6: Submit
    // Slot 7: CancelRequestsWithTag
    queue_vtable[0] = reinterpret_cast<void*>(&MockQueue::QueryInterface);
    queue_vtable[1] = reinterpret_cast<void*>(&MockQueue::AddRef);
    queue_vtable[2] = reinterpret_cast<void*>(&MockQueue::Release);
    queue_vtable[3] = reinterpret_cast<void*>(&MockQueue::EnqueueRequest);
    queue_vtable[4] = reinterpret_cast<void*>(&MockQueue::EnqueueStatus);
    queue_vtable[5] = reinterpret_cast<void*>(&MockQueue::EnqueueSignal);
    queue_vtable[6] = reinterpret_cast<void*>(&MockQueue::Submit);
    queue_vtable[7] = reinterpret_cast<void*>(&MockQueue::CancelRequestsWithTag);

    initialized = true;
}
} // namespace

extern "C" __declspec(dllexport) HRESULT WINAPI DStorageGetFactory(const GUID& riid, void** ppv)
{
    init_vtables();
    InterlockedIncrement(&factory_calls);
    if (!ppv) return E_POINTER;

    if (!IsEqualGUID(riid, IID_FixtureFactory)) {
        // DirectStorage contract: on refusal (such as PC IID or unknown interface),
        // *ppv MUST be left untouched and not clobbered.
        return E_NOINTERFACE;
    }

    if (!cfg_factory_success) {
        *ppv = nullptr;
        return E_FAIL;
    }

    *ppv = new MockFactory();
    return S_OK;
}

// Verification and configuration exports
extern "C" __declspec(dllexport) uint32_t WINAPI get_dstorage_factory_calls(void)
{
    return static_cast<uint32_t>(factory_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_dstorage_queue_calls(void)
{
    return static_cast<uint32_t>(queue_calls);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_dstorage_staging_buffer_size(void)
{
    return current_staging_buffer_size;
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_dstorage_staged_requests_count(void)
{
    return static_cast<uint32_t>(staged_requests);
}

extern "C" __declspec(dllexport) uint32_t WINAPI get_dstorage_submitted_count(void)
{
    return static_cast<uint32_t>(submitted_batches);
}

extern "C" __declspec(dllexport) void WINAPI set_dstorage_factory_success(int success)
{
    InterlockedExchange(&cfg_factory_success, success ? 1 : 0);
}

extern "C" __declspec(dllexport) void WINAPI set_dstorage_queue_success(int success)
{
    InterlockedExchange(&cfg_queue_success, success ? 1 : 0);
}

extern "C" __declspec(dllexport) void WINAPI set_dstorage_eager_completion(int eager)
{
    InterlockedExchange(&cfg_eager_completion, eager ? 1 : 0);
}

extern "C" __declspec(dllexport) void WINAPI reset_dstorage_records(void)
{
    InterlockedExchange(&cfg_factory_success, 1);
    InterlockedExchange(&cfg_queue_success, 1);
    InterlockedExchange(&cfg_eager_completion, 0);
    InterlockedExchange(&factory_calls, 0);
    InterlockedExchange(&queue_calls, 0);
    InterlockedExchange(&staged_requests, 0);
    InterlockedExchange(&submitted_batches, 0);
    current_staging_buffer_size = 0;
    InterlockedExchange(&active_queues, 0);
}
