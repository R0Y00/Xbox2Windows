// A sampler-heap-indexed root signature set on a console list (debug layer message ID 1419).
//
//   startup_root_sampler_heap <warp|hardware>   (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// The desktop list never receives the title's descriptor heaps, so a title root signature created with
// D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED used to be set with no sampler heap bound. Every
// debug layer message is counted through a message callback, because the stand-in drains the stored ones:
//
//   control   the flagged root signature set on a plain desktop list, no heap   >= 1 message with ID 1419
//   flagged   the same root signature through the stand-in's list               0 messages with ID 1419
//   plain     a root signature without the flag through the stand-in's list     0 messages with ID 1419
//
// The list is closed, submitted and waited for each time, and the device must not be removed.
//
// Exit codes: 0 pass, 77 no physical adapter or no debug layer, other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
constexpr int sampler_heap_message = 1419;
std::atomic<int> messages_1419{0}, messages_other{0};

template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

void WINAPI on_message(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id, LPCSTR text, void*)
{
    if (severity > D3D12_MESSAGE_SEVERITY_ERROR) return;
    if (static_cast<int>(id) == sampler_heap_message) ++messages_1419;
    else ++messages_other;
    std::printf("debug layer id %d: %s\n", static_cast<int>(id), text);
}

// A root signature with no parameters and the given D3D12_ROOT_SIGNATURE_FLAGS.
ComPtr<ID3DBlob> root_blob(UINT flags)
{
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1.Flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags);
    ComPtr<ID3DBlob> blob;
    return SUCCEEDED(D3D12SerializeVersionedRootSignature(&desc, &blob, nullptr)) ? blob : nullptr;
}

void wait_for(ID3D12Device* desktop, ID3D12CommandQueue* queue)
{
    ComPtr<ID3D12Fence> fence;
    if (FAILED(desktop->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) return;
    queue->Signal(fence.Get(), 1);
    while (fence->GetCompletedValue() < 1) Sleep(1);
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            const HRESULT hr = factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate));
            if (hr == DXGI_ERROR_NOT_FOUND) return 77;
            if (FAILED(hr)) return 3;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(candidate->GetDesc1(&desc))) return 3;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }

    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 4;
    ComPtr<ID3D12InfoQueue1> queue1;
    DWORD cookie = 0;
    if (FAILED(desktop.As(&queue1)) || FAILED(queue1->RegisterMessageCallback(on_message, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie))) {
        std::printf("no debug layer message callback: skipped\n");
        return 77;
    }

    const auto flagged = root_blob(0x800), plain = root_blob(0);
    if (!flagged || !plain) {
        std::printf("this runtime cannot serialize the sampler heap flag: skipped\n");
        return 77;
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    ComPtr<ID3D12CommandQueue> desktop_queue;
    ComPtr<ID3D12CommandAllocator> desktop_allocator;
    ComPtr<ID3D12GraphicsCommandList> desktop_list;
    ComPtr<ID3D12RootSignature> desktop_root;
    if (FAILED(desktop->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&desktop_queue))) ||
        FAILED(desktop->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&desktop_allocator))) ||
        FAILED(desktop->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, desktop_allocator.Get(), nullptr, IID_PPV_ARGS(&desktop_list))) ||
        FAILED(desktop->CreateRootSignature(0, flagged->GetBufferPointer(), flagged->GetBufferSize(), IID_PPV_ARGS(&desktop_root))))
        return 5;

    // Control: the debug layer reports the flagged root signature set with no sampler heap bound.
    desktop_list->SetGraphicsRootSignature(desktop_root.Get());
    desktop_list->Close();
    ID3D12CommandList* lists[] = {desktop_list.Get()};
    desktop_queue->ExecuteCommandLists(1, lists);
    wait_for(desktop.Get(), desktop_queue.Get());
    std::printf("control: %d messages with ID %d\n", messages_1419.load(), sampler_heap_message);
    if (!messages_1419) {
        std::printf("the debug layer does not report the sampler heap message here: skipped\n");
        return 77;
    }
    const int control = messages_1419.exchange(0);
    messages_other = 0;

    // The same two root signatures through the stand-in's list.
    void *console_queue = nullptr, *console_fence = nullptr, *allocator = nullptr, *list = nullptr, *flagged_root = nullptr, *plain_root = nullptr;
    const unsigned list_desc[] = {D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    if (!ok(invoke(device, 8, arg(&queue_desc), arg(&__uuidof(ID3D12CommandQueue)), arg(&console_queue))) ||
        !ok(invoke(device, 36, 0, 0, arg(&__uuidof(ID3D12Fence)), arg(&console_fence))) ||
        !ok(invoke(device, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, arg(&__uuidof(ID3D12CommandAllocator)), arg(&allocator))) ||
        !ok(invoke(device, 73, arg(list_desc), arg(allocator), 0, arg(&__uuidof(ID3D12GraphicsCommandList)), arg(&list))) ||
        !ok(invoke(device, 16, 0, arg(flagged->GetBufferPointer()), flagged->GetBufferSize(), arg(&__uuidof(ID3D12RootSignature)), arg(&flagged_root))) ||
        !ok(invoke(device, 16, 0, arg(plain->GetBufferPointer()), plain->GetBufferSize(), arg(&__uuidof(ID3D12RootSignature)), arg(&plain_root))))
        return 6;
    Word fence_value = 0;
    const auto record_and_submit = [&](void* root) {
        invoke(list, 34, arg(root));  // SetGraphicsRootSignature
        if (!ok(invoke(list, 9))) return false;
        void* submit[] = {list};
        invoke(console_queue, 10, 1, arg(submit));
        if (!ok(invoke(console_queue, 14, arg(console_fence), ++fence_value))) return false;
        const ULONGLONG deadline = GetTickCount64() + 10000;
        while (invoke(console_fence, 8) < fence_value) {
            if (GetTickCount64() > deadline) return false;
            Sleep(1);
        }
        if (!ok(invoke(allocator, 8)) || !ok(invoke(list, 139, arg(allocator)))) return false;
        return true;
    };
    if (!record_and_submit(flagged_root)) return 10;
    const int flagged_messages = messages_1419.exchange(0);
    std::printf("flagged root signature: %d messages with ID %d\n", flagged_messages, sampler_heap_message);
    if (!record_and_submit(plain_root)) return 11;
    const int plain_messages = messages_1419.exchange(0);
    std::printf("plain root signature: %d messages with ID %d\n", plain_messages, sampler_heap_message);
    if (!record_and_submit(flagged_root)) return 12;
    const int second_flagged_messages = messages_1419.exchange(0);
    std::printf("flagged root signature after reset: %d messages with ID %d\n", second_flagged_messages, sampler_heap_message);

    const HRESULT removed = desktop->GetDeviceRemovedReason();
    std::printf("control %d, device removed reason 0x%08lx, other debug layer errors %d\n", control,
                static_cast<unsigned long>(removed), messages_other.load());
    queue1->UnregisterMessageCallback(cookie);
    if (flagged_messages) return 20;
    if (plain_messages) return 21;
    if (second_flagged_messages) return 22;
    return removed == S_OK && messages_other == 0 ? 0 : 70;
}
