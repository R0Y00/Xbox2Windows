// A depth texture of console format 118 (D16_UNORM_S8_UINT) created with a clear value, through device slots
// 133 and 27, must be accepted by desktop D3D12 on the first attempt: the clear value carries the mapped format,
// so the debug layer stores no message for an invalid clear value (ids 816 and 817).
//
//   startup_clear_format <warp|hardware>     (X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 X2W_D3D12_DEBUG=1)
//
// Both creations must return success and an object. The debug layer's stored messages are read right after
// each creation (slot 27 clears the shared queue): none may be an error, and none may carry id 816 or 817.
// A negative control first sends the unmapped clear value to desktop D3D12 and requires it to be rejected with
// a message (exit 14 otherwise), so the checks can fail on the defect. When the debug layer is not installed the creations are still
// checked and the test reports a skip (77) for the message part.
//
// Exit codes: 0 pass, 77 no physical adapter or no debug layer, other values name the failing step.
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Word = std::uint64_t;
template <class T> Word arg(T* value) { return reinterpret_cast<Word>(value); }
Word invoke(void* object, unsigned index, Word a = 0, Word b = 0, Word c = 0, Word d = 0, Word e = 0, Word f = 0, Word g = 0)
{
    using Slot = Word (WINAPI*)(void*, Word, Word, Word, Word, Word, Word, Word);
    return reinterpret_cast<Slot>((*static_cast<void***>(object))[index])(object, a, b, c, d, e, f, g);
}
bool ok(Word hr) { return SUCCEEDED(static_cast<HRESULT>(hr)); }

constexpr UINT console_d16_s8 = 118;
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
    DXGI_ADAPTER_DESC adapter_desc{};
    if (FAILED(adapter->GetDesc(&adapter_desc))) return 3;
    std::printf("adapter=%ls\n", adapter_desc.Description);

    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 4;
    ComPtr<ID3D12Device> desktop;
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&desktop)))) return 4;

    ComPtr<ID3D12InfoQueue> info;
    const bool has_layer = SUCCEEDED(desktop.As(&info));
    if (has_layer) info->ClearStoredMessages();

    const D3D12_RESOURCE_DESC depth_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 16, 16, 1, 1, static_cast<DXGI_FORMAT>(console_d16_s8),
                                         {1, 0}, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL};
    D3D12_CLEAR_VALUE clear{};
    clear.Format = static_cast<DXGI_FORMAT>(console_d16_s8);
    clear.DepthStencil.Depth = 1.0f;

    // Reads and clears the stored debug layer messages. Every call site runs right after the creation it
    // checks: a later device call clears the shared queue, which would hide the messages of an earlier one.
    struct Messages { UINT64 errors = 0, clear_value = 0; };
    const auto drain = [&](const char* step) {
        Messages seen;
        for (UINT64 i = 0, n = info->GetNumStoredMessages(); i < n; ++i) {
            SIZE_T bytes = 0;
            if (FAILED(info->GetMessage(i, nullptr, &bytes))) continue;
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(info->GetMessage(i, message, &bytes))) continue;
            if (static_cast<int>(message->ID) == 816 || static_cast<int>(message->ID) == 817) ++seen.clear_value;
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                ++seen.errors;
                std::printf("debug layer %d (%s): %.*s\n", static_cast<int>(message->ID), step,
                            static_cast<int>(message->DescriptionByteLength), message->pDescription);
            }
        }
        info->ClearStoredMessages();
        std::printf("%s: clear value messages (816, 817): %llu, debug layer errors: %llu\n", step,
                    static_cast<unsigned long long>(seen.clear_value), static_cast<unsigned long long>(seen.errors));
        return seen;
    };

    // Negative control: the same clear value, unmapped (console format 118), sent straight to desktop D3D12 must
    // be refused and leave debug layer messages, so the checks below can fail on exactly this defect. Without
    // it, a mapping that is removed could still pass through the creation retry without a clear value.
    if (has_layer) {
        D3D12_RESOURCE_DESC mapped_desc = depth_desc;
        mapped_desc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        const D3D12_HEAP_PROPERTIES control_props{D3D12_HEAP_TYPE_DEFAULT};
        ComPtr<ID3D12Resource> control;
        const HRESULT control_hr = desktop->CreateCommittedResource(&control_props, D3D12_HEAP_FLAG_NONE, &mapped_desc,
                                                                    D3D12_RESOURCE_STATE_COMMON, &clear, IID_PPV_ARGS(&control));
        const Messages control_seen = drain("negative control (unmapped clear value)");
        if (SUCCEEDED(control_hr) || (control_seen.clear_value == 0 && control_seen.errors == 0)) {
            std::printf("negative control: an unmapped clear value was not rejected with a message\n");
            return 14;
        }
    }

    // Slot 133, placed on title memory: a texture is a committed stand-in.
    static std::array<std::uint8_t, 65536 + 4096> memory{};
    const Word address = (reinterpret_cast<Word>(memory.data()) + 0xffff) & ~Word{0xffff};
    void* placed = nullptr;
    if (!ok(invoke(device, 133, address, arg(&depth_desc), D3D12_RESOURCE_STATE_COMMON, arg(&clear), arg(&__uuidof(ID3D12Resource)),
                   arg(&placed))) || !placed)
        return 10;
    std::printf("slot 133: depth texture with a clear value created\n");
    // Inspect right now: the slot 27 call below clears the shared queue before it could be read.
    Messages slot133;
    if (has_layer) slot133 = drain("slot 133");

    // Slot 27, desktop order (props, flags, desc, state, clear, riid, ppv).
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    void* committed = nullptr;
    if (!ok(invoke(device, 27, arg(&props), D3D12_HEAP_FLAG_NONE, arg(&depth_desc), D3D12_RESOURCE_STATE_COMMON, arg(&clear),
                   arg(&__uuidof(ID3D12Resource)), arg(&committed))) || !committed)
        return 11;
    std::printf("slot 27: depth texture with a clear value created\n");

    if (desktop->GetDeviceRemovedReason() != S_OK) return 12;
    if (!has_layer) {
        std::printf("debug layer not installed: message check skipped\n");
        return 77;
    }
    const Messages slot27 = drain("slot 27");
    if (slot133.clear_value || slot27.clear_value) return 13;
    return slot133.errors + slot27.errors == 0 ? 0 : 70;
}
