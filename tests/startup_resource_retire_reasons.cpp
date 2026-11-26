// X2W_D3D12_RETIRE_REASONS=1 (docs/startup-runtime.md#d3d12-resource-budget): with X2W_D3D12_RELEASE_RETIRED=1 and
// X2W_D3D12_BUDGET=1 the heartbeat names why resources that the guest released for good still hold their desktop
// object. The fixture creates one resource per main reason, releases it, and compares the whole line the export builds
// (the same text the heartbeat logs) with the expected counts and bytes. Bytes come from the desktop allocation size,
// which is what the budget ticket records.
//
//   startup_resource_retire_reasons <warp|hardware> [off|no-retire]
//     (X2W_D3D12_BUDGET=1 X2W_D3D12_PERMISSIVE=1 X2W_D3D12_LIST_NOOP=1 X2W_D3D12_RENDER=1 and
//      X2W_D3D12_RELEASE_RETIRED=1 X2W_D3D12_RETIRE_REASONS=1; "off" runs without RETIRE_REASONS, "no-retire" without
//      RELEASE_RETIRED: the export must then report nothing)
//
// Expected reasons, each with its own resource:
//   view-refs (two, one of them in a shader-visible heap), never-class:rt-ds-uav, never-class:msaa, never-class:buffer,
//   left-class:view, left-class:residency, left-class:present, list-hold (an unsubmitted closed list), flight (a
//   submission held behind a gate), queued-waiting-fence (a release behind the same gate).
// Then the gate opens, the flight and fence entries leave, and the list hold leaves with the Reset of its list.
#include "startup_stubs/resource_budget.h"

#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <string>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) std::size_t x2w_d3d12_retire_reasons_line(char*, std::size_t);

namespace {

constexpr DWORD wait_ms = 10000;
constexpr int settle_limit = 200;
constexpr UINT side = 64;

template<class Result = HRESULT, class... Args>
Result invoke(IUnknown* object, unsigned slot, Args... args)
{
    using Call = Result (WINAPI*)(IUnknown*, Args...);
    return reinterpret_cast<Call>((*reinterpret_cast<void***>(object))[slot])(object, args...);
}

template<class T>
ComPtr<T> desktop_object(IUnknown* guest, ID3D12Device* device, IUnknown* object)
{
    constexpr GUID key{0x0e34d52f, 0xd5a5, 0x4790, {0x91, 0xd4, 0x11, 0xa6, 0x7c, 0xbc, 0x2b, 0xc1}};
    ComPtr<T> output;
    if (FAILED(invoke(guest, 5, &key, object))) return output;
    ComPtr<IUnknown> reference;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT hr = device->GetPrivateData(key, &bytes, reference.GetAddressOf());
    invoke(guest, 5, &key, static_cast<IUnknown*>(nullptr));
    if (SUCCEEDED(hr)) reference.As(&output);
    return output;
}

bool require(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) std::printf("FAIL: %s hr=0x%08lx\n", stage, static_cast<unsigned long>(hr));
    return SUCCEEDED(hr);
}

ComPtr<IDXGIAdapter1> pick_adapter(IDXGIFactory4* factory, bool warp)
{
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter));
        return adapter;
    }
    wchar_t name[128]{};
    GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", name, static_cast<DWORD>(std::size(name)));
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i, adapter.Reset()) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
            (!name[0] || std::wcsstr(desc.Description, name)) &&
            SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)))
            return adapter;
    }
    return nullptr;
}

struct Res {
    ComPtr<IUnknown> guest;
    UINT64 bytes = 0;
};
struct GuestQueue {
    ComPtr<IUnknown> queue, done;
    ComPtr<ID3D12Fence> done_real;
    UINT64 value = 0;
};
struct GuestList {
    ComPtr<IUnknown> allocator, list;
};
using Tokens = std::map<std::string, std::pair<long long, long long>>;

Tokens parse(const char* line)
{
    Tokens tokens;
    for (const char* p = line; *p;) {
        while (*p == ' ') ++p;
        const char* end = p;
        while (*end && *end != ' ') ++end;
        char word[128]{};
        std::memcpy(word, p, (std::min<std::size_t>)(static_cast<std::size_t>(end - p), sizeof(word) - 1));
        char name[96]{};
        long long count = 0, bytes = 0;
        if (std::sscanf(word, "%95[^=]=%lld/%lld", name, &count, &bytes) == 3) tokens[name] = {count, bytes};
        p = end;
    }
    return tokens;
}

struct Context {
    IUnknown* guest;
    ID3D12Device* device;
    HANDLE event = nullptr;
    std::deque<Res> pool;
    GuestQueue direct, other, gated;
    GuestList drain_list;
    D3D12_RESOURCE_DESC desc{};
    std::array<GUID, 1> iid_resource{__uuidof(ID3D12Resource)};

    bool wait(ID3D12Fence* fence, UINT64 value)
    {
        if (fence->GetCompletedValue() >= value) return true;
        if (FAILED(fence->SetEventOnCompletion(value, event))) return false;
        return WaitForSingleObject(event, wait_ms) == WAIT_OBJECT_0;
    }
    Res& make(const D3D12_RESOURCE_DESC& d, bool through_slot_27 = false)
    {
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        Res& r = pool.emplace_back();
        r.bytes = device->GetResourceAllocationInfo(0, 1, &d).SizeInBytes;
        const HRESULT hr = through_slot_27
            ? invoke(guest, 27, &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, &iid_resource[0],
                     r.guest.GetAddressOf())
            : invoke(guest, 127, &heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr, 0u, nullptr,
                     &iid_resource[0], r.guest.GetAddressOf());
        if (!require(hr, "create resource")) throw 4;
        return r;
    }
    Res& texture(D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, UINT samples = 1)
    {
        D3D12_RESOURCE_DESC d = desc;
        d.Flags = flags;
        d.SampleDesc = {samples, 0};
        return make(d);
    }
    Res& buffer()
    {
        const D3D12_RESOURCE_DESC d{D3D12_RESOURCE_DIMENSION_BUFFER, 0, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE};
        return make(d, true);
    }
    bool make_queue(GuestQueue& q)
    {
        const D3D12_COMMAND_QUEUE_DESC d{};
        return require(invoke(guest, 8, &d, &__uuidof(ID3D12CommandQueue), q.queue.GetAddressOf()), "queue") &&
               require(invoke(guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), q.done.GetAddressOf()), "done fence") &&
               (q.done_real = desktop_object<ID3D12Fence>(guest, device, q.done.Get())) != nullptr;
    }
    bool sync(GuestQueue& q)
    {
        if (!require(invoke(q.queue.Get(), 14, q.done.Get(), ++q.value), "queue signal")) return false;
        if (wait(q.done_real.Get(), q.value)) return true;
        std::puts("FAIL: a queue did not reach its completion fence");
        return false;
    }
    bool sync_all() { return sync(direct) && sync(other) && sync(gated); }
    bool make_list(GuestList& l)
    {
        const unsigned list_desc[] = {static_cast<unsigned>(D3D12_COMMAND_LIST_TYPE_DIRECT), 2, 1, 0};
        return require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator), l.allocator.GetAddressOf()), "allocator") &&
               require(invoke(guest, 73, list_desc, l.allocator.Get(), nullptr, &__uuidof(ID3D12GraphicsCommandList), l.list.GetAddressOf()), "list") &&
               require(invoke(l.list.Get(), 139, l.allocator.Get(), nullptr), "list reset");
    }
    bool reset_list(GuestList& l)
    {
        ComPtr<IUnknown> next;
        if (!require(invoke(guest, 9, D3D12_COMMAND_LIST_TYPE_DIRECT, &__uuidof(ID3D12CommandAllocator), next.GetAddressOf()), "allocator")) return false;
        if (!require(invoke(l.list.Get(), 139, next.Get(), nullptr), "list reset")) return false;
        l.allocator = next;
        return true;
    }
    bool close(GuestList& l) { return require(invoke(l.list.Get(), 9), "list close"); }
    void execute(GuestQueue& q, GuestList& l)
    {
        IUnknown* lists[] = {l.list.Get()};
        invoke<void>(q.queue.Get(), 10, 1u, lists);
    }
    void drain() { execute(other, drain_list); }
    // A closed list that copies source into destination.
    bool record_copy(GuestList& l, Res& destination, Res& source)
    {
        if (!make_list(l)) return false;
        invoke<void>(l.list.Get(), 17, destination.guest.Get(), source.guest.Get());
        return close(l);
    }
    bool line(std::string& text) const
    {
        char buffer[2048]{};
        if (!x2w_d3d12_retire_reasons_line(buffer, sizeof(buffer))) return false;
        text = buffer;
        return true;
    }
};

struct Want {
    const char* name;
    long long count, bytes;
};

// The line carries exactly the wanted tokens, plus total and unattributed.
bool expect(Context& c, const char* stage, std::initializer_list<Want> wants, long long unattributed_count = 0)
{
    std::string text;
    if (!c.line(text)) {
        std::printf("FAIL: %s: the diagnostics line is empty although the switches are on\n", stage);
        return false;
    }
    Tokens want;
    long long total_count = 0, total_bytes = 0;
    for (const Want& w : wants) {
        want[w.name] = {w.count, w.bytes};
        if (std::strcmp(w.name, "view-refs-in-visible-heap")) total_count += w.count, total_bytes += w.bytes;
    }
    want["total"] = {total_count, total_bytes};
    const Tokens got = parse(text.c_str());
    Tokens compare = got;
    const auto un = compare.find("unattributed");
    const bool unattributed_ok = un != compare.end() && un->second.first == unattributed_count;
    compare.erase("unattributed");
    if (compare == want && unattributed_ok) return true;
    std::printf("FAIL: %s\n  got:  %s\n  want:", stage, text.c_str());
    for (const auto& [name, value] : want) std::printf(" %s=%lld/%lld", name.c_str(), value.first, value.second);
    std::printf(" unattributed=%lld/*\n", unattributed_count);
    return false;
}

// Drains until the line matches, without sleeping.
bool settle(Context& c, const char* stage, std::initializer_list<Want> wants, long long unattributed_count = 0)
{
    for (int i = 0; i < settle_limit; ++i) {
        c.drain();
        std::string text;
        Tokens want;
        long long total_count = 0, total_bytes = 0;
        for (const Want& w : wants) {
            want[w.name] = {w.count, w.bytes};
            if (std::strcmp(w.name, "view-refs-in-visible-heap")) total_count += w.count, total_bytes += w.bytes;
        }
        want["total"] = {total_count, total_bytes};
        if (c.line(text)) {
            Tokens got = parse(text.c_str());
            const auto un = got.find("unattributed");
            const bool unattributed_ok = un != got.end() && un->second.first == unattributed_count;
            got.erase("unattributed");
            if (got == want && unattributed_ok) return true;
        }
    }
    return expect(c, stage, wants, unattributed_count);
}

bool scenario(Context& c)
{
    const struct {
        D3D12_DESCRIPTOR_HEAP_DESC desc;
        UINT64 backing, bytes;
    } visible_heap{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0},
      hidden_heap{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1}, 0, 0};
    ComPtr<IUnknown> visible, hidden;
    if (!require(invoke(c.guest, 14, &visible_heap, &__uuidof(ID3D12DescriptorHeap), visible.GetAddressOf()), "visible heap") ||
        !require(invoke(c.guest, 14, &hidden_heap, &__uuidof(ID3D12DescriptorHeap), hidden.GetAddressOf()), "hidden heap"))
        return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;

    if (!c.sync_all()) return false;  // every queue has a completion point before anything is released
    Res& A = c.texture();
    Res& B = c.texture();
    invoke<void>(c.guest, 18, A.guest.Get(), &srv, invoke<UINT64>(visible.Get(), 9));
    invoke<void>(c.guest, 18, B.guest.Get(), &srv, invoke<UINT64>(hidden.Get(), 9));
    const UINT64 tb = A.bytes;
    A.guest.Reset();
    B.guest.Reset();

    Res& rt = c.texture(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    Res& msaa = c.texture(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 4);
    Res& buf = c.buffer();
    const UINT64 rt_bytes = rt.bytes, msaa_bytes = msaa.bytes, buf_bytes = buf.bytes;
    rt.guest.Reset();
    msaa.guest.Reset();
    buf.guest.Reset();

    // A view whose descriptor handle the shim cannot translate takes the resource out of the class.
    Res& viewed = c.texture();
    invoke<void>(c.guest, 18, viewed.guest.Get(), &srv, UINT64{0});
    viewed.guest.Reset();
    Res& resident = c.texture();
    IUnknown* residents[] = {resident.guest.Get()};
    if (!require(invoke(c.guest, 34, 1u, residents), "MakeResident")) return false;
    resident.guest.Reset();
    Res& presented = c.texture();
    void* surface = presented.guest.Get();
    const UINT64 plane[10]{0, 1, reinterpret_cast<UINT64>(&surface)};
    const UINT64 present_parameters[8]{};
    if (!require(invoke(c.direct.queue.Get(), 32, 1u, plane, present_parameters), "Present")) return false;
    presented.guest.Reset();

    // A released source that a closed, unsubmitted list names.
    GuestList listed;
    Res& list_destination = c.texture();
    Res& list_source = c.texture();
    if (!c.record_copy(listed, list_destination, list_source)) return false;
    list_source.guest.Reset();
    if (!c.sync_all()) return false;
    for (int i = 0; i < 4; ++i) c.drain();

    // A submission behind a gate, its list reset, then two releases behind the same gate.
    ComPtr<IUnknown> gate;
    if (!require(invoke(c.guest, 36, UINT64{0}, D3D12_FENCE_FLAG_NONE, &__uuidof(ID3D12Fence), gate.GetAddressOf()), "gate fence") ||
        !require(invoke(c.gated.queue.Get(), 17, gate.Get(), UINT64{1}), "queue wait"))
        return false;
    GuestList flying;
    Res& fly_destination = c.texture();
    Res& fly_source = c.texture();
    if (!c.record_copy(flying, fly_destination, fly_source)) return false;
    c.execute(c.gated, flying);
    fly_source.guest.Reset();
    if (!c.reset_list(flying)) return false;
    Res& waiting = c.texture();
    waiting.guest.Reset();
    if (!c.sync(c.direct) || !c.sync(c.other)) return false;
    for (int i = 0; i < 4; ++i) c.drain();

    if (!expect(c, "every reason once",
                {{"view-refs", 2, static_cast<long long>(2 * tb)}, {"view-refs-in-visible-heap", 1, static_cast<long long>(tb)},
                 {"never-class:rt-ds-uav", 1, static_cast<long long>(rt_bytes)},
                 {"never-class:msaa", 1, static_cast<long long>(msaa_bytes)},
                 {"never-class:buffer", 1, static_cast<long long>(buf_bytes)},
                 {"left-class:view", 1, static_cast<long long>(tb)}, {"left-class:residency", 1, static_cast<long long>(tb)},
                 {"left-class:present", 1, static_cast<long long>(tb)}, {"list-hold", 1, static_cast<long long>(tb)},
                 {"flight", 1, static_cast<long long>(tb)}, {"queued-waiting-fence", 1, static_cast<long long>(tb)}}))
        return false;
    std::puts("1 one resource per reason: view-refs, rt-ds-uav, msaa, buffer, left view/residency/present, list-hold, flight, fence");

    if (!require(invoke(gate.Get(), 10, UINT64{1}), "gate signal") || !c.sync_all()) return false;
    if (!settle(c, "gate opened",
                {{"view-refs", 2, static_cast<long long>(2 * tb)}, {"view-refs-in-visible-heap", 1, static_cast<long long>(tb)},
                 {"never-class:rt-ds-uav", 1, static_cast<long long>(rt_bytes)},
                 {"never-class:msaa", 1, static_cast<long long>(msaa_bytes)},
                 {"never-class:buffer", 1, static_cast<long long>(buf_bytes)},
                 {"left-class:view", 1, static_cast<long long>(tb)}, {"left-class:residency", 1, static_cast<long long>(tb)},
                 {"left-class:present", 1, static_cast<long long>(tb)}, {"list-hold", 1, static_cast<long long>(tb)}}))
        return false;
    std::puts("2 gate opened: flight and queued-waiting-fence leave, the rest stays");

    if (!c.reset_list(listed)) return false;
    if (!settle(c, "list reset",
                {{"view-refs", 2, static_cast<long long>(2 * tb)}, {"view-refs-in-visible-heap", 1, static_cast<long long>(tb)},
                 {"never-class:rt-ds-uav", 1, static_cast<long long>(rt_bytes)},
                 {"never-class:msaa", 1, static_cast<long long>(msaa_bytes)},
                 {"never-class:buffer", 1, static_cast<long long>(buf_bytes)},
                 {"left-class:view", 1, static_cast<long long>(tb)}, {"left-class:residency", 1, static_cast<long long>(tb)},
                 {"left-class:present", 1, static_cast<long long>(tb)}}))
        return false;
    std::puts("3 list reset: the list hold leaves");

    // Overwriting the shader-visible descriptor drops one view reference; the resource then retires.
    Res& replacement = c.texture();
    invoke<void>(c.guest, 18, replacement.guest.Get(), &srv, invoke<UINT64>(visible.Get(), 9));
    if (!settle(c, "descriptor overwritten",
                {{"view-refs", 1, static_cast<long long>(tb)},
                 {"never-class:rt-ds-uav", 1, static_cast<long long>(rt_bytes)},
                 {"never-class:msaa", 1, static_cast<long long>(msaa_bytes)},
                 {"never-class:buffer", 1, static_cast<long long>(buf_bytes)},
                 {"left-class:view", 1, static_cast<long long>(tb)}, {"left-class:residency", 1, static_cast<long long>(tb)},
                 {"left-class:present", 1, static_cast<long long>(tb)}}))
        return false;
    std::puts("4 descriptor overwritten: the visible-heap view-refs entry retires, the hidden-heap one stays");
    return true;
}

// Negative control: the same kind of release with the switches off reports nothing.
bool nothing_reported(Context& c)
{
    if (!c.sync_all()) return false;
    Res& r = c.texture();
    r.guest.Reset();
    Res& g = c.texture(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    g.guest.Reset();
    for (int i = 0; i < 4; ++i) c.drain();
    char buffer[256];
    std::memset(buffer, 'x', sizeof(buffer));
    const std::size_t length = x2w_d3d12_retire_reasons_line(buffer, sizeof(buffer));
    if (length || buffer[0]) {
        std::printf("FAIL: the diagnostics reported %zu bytes without their switches: %s\n", length, buffer);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "hardware")) ||
        (argc == 3 && std::strcmp(argv[2], "off") && std::strcmp(argv[2], "no-retire")))
        return 2;
    const auto env_on = [](const char* name) {
        char value[2]{};
        return GetEnvironmentVariableA(name, value, sizeof(value)) == 1 && value[0] == '1';
    };
    const bool off = argc == 3 && !std::strcmp(argv[2], "off");
    const bool no_retire = argc == 3 && !std::strcmp(argv[2], "no-retire");
    if (env_on("X2W_D3D12_RETIRE_REASONS") != !off || env_on("X2W_D3D12_RELEASE_RETIRED") != !no_retire ||
        !env_on("X2W_D3D12_BUDGET")) {
        std::puts("FAIL: the environment does not match the requested mode");
        return 2;
    }
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    const auto adapter = pick_adapter(factory.Get(), std::strcmp(argv[1], "warp") == 0);
    if (!adapter) return 77;
    ComPtr<IUnknown> guest;
    ComPtr<ID3D12Device> device;
    std::array<unsigned char, 64> parameters{};
    if (!require(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device),
                                      reinterpret_cast<void**>(guest.GetAddressOf())), "Xbox device") ||
        !require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "desktop device"))
        return 3;
    std::printf("mode=%s\n", off ? "off" : no_retire ? "no-retire" : "on");
    Context c{guest.Get(), device.Get()};
    try {
        c.desc = {D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, side, side, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1, 0},
                  D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
        c.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!c.event || !c.make_queue(c.direct) || !c.make_queue(c.other) || !c.make_queue(c.gated)) return 4;
        if (!c.make_list(c.drain_list) || !c.close(c.drain_list)) return 4;
        if (!(off || no_retire ? nothing_reported(c) : scenario(c))) return 5;
    } catch (int code) {
        return code;
    }
    if (FAILED(device->GetDeviceRemovedReason())) return 4;
    std::puts(off || no_retire ? "retire-reasons off checks passed" : "retire-reasons checks passed");
    return 0;
}
