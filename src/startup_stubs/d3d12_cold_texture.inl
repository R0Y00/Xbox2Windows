// Residency hints only. The committed object and its descriptors keep their normal lifetime.
// A title that samples a persistent descriptor without a shim call can still use a low-priority texture.
enum { priority_untracked, priority_normal, priority_low };
SRWLOCK cold_texture_lock = SRWLOCK_INIT;
unsigned long long cold_texture_count, cold_texture_bytes, warmed_texture_count, warmed_texture_bytes, cold_texture_failures;

bool cold_textures_enabled()
{
    static const bool on = x2w_local::env_on("X2W_D3D12_COLD_TEXTURE_PRIORITY") &&
        !x2w_local::env_on("X2W_D3D12_RELEASE_RETIRED") && !x2w_local::env_on("X2W_MEMCUT_RELEASE");
    return on;
}

void update_texture_priority(Wrapper* wrapper, bool cool)
{
    AcquireSRWLockExclusive(&cold_texture_lock);
    auto& w = wrapper->own();
    const LONG previous = cool ? priority_normal : priority_low;
    const LONG state = ReadNoFence(&w.cold_texture_state);
    if (!cool && state == priority_normal) {
        // A bind at zero guest refs, or the AddRef from zero, reached this lock before the final
        // Release could lower the priority. Keep the resource warm when that Release arrives.
        InterlockedExchange(&w.cold_texture_state, priority_untracked);
    } else if (state == previous &&
        (!cool || InterlockedCompareExchange(&w.refs, 0, 0) == 0)) {
        auto* resource = static_cast<ID3D12Resource*>(w.real);
        ID3D12Device1* device = nullptr;
        HRESULT hr = resource->GetDevice(IID_PPV_ARGS(&device));
        unsigned long long bytes = 0;
        if (SUCCEEDED(hr)) {
            const auto desc = resource->GetDesc();
            bytes = device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
            const auto priority = cool ? D3D12_RESIDENCY_PRIORITY_LOW : D3D12_RESIDENCY_PRIORITY_NORMAL;
            ID3D12Pageable* object = resource;
            hr = device->SetResidencyPriority(1, &object, &priority);
            device->Release();
        }
        if (SUCCEEDED(hr)) {
            // Keep a texture warm permanently once we see its reuse, including a bind at zero guest refs.
            InterlockedExchange(&w.cold_texture_state, cool ? priority_low : priority_untracked);
            if (cool) {
                ++cold_texture_count;
                cold_texture_bytes += bytes;
            } else {
                ++warmed_texture_count;
                warmed_texture_bytes += bytes;
            }
        } else {
            ++cold_texture_failures;
            if (cool) InterlockedExchange(&w.cold_texture_state, priority_untracked);
            if (cold_texture_failures <= 3) log("x2w-cold texture: %s failed 0x%08lx", cool ? "lowering priority" : "restoring priority", hr);
        }
    }
    ReleaseSRWLockExclusive(&cold_texture_lock);
}

void warm_texture(Wrapper* wrapper, bool reacquired = false)
{
    if (!wrapper) return;
    const auto& w = wrapper->own();
    const LONG state = ReadNoFence(&w.cold_texture_state);
    if (state == priority_low || (state == priority_normal && (reacquired || !ReadNoFence(&w.refs))))
        update_texture_priority(wrapper, false);
}

void cool_texture(Wrapper* wrapper)
{
    if (ReadNoFence(&wrapper->own().cold_texture_state) == priority_normal)
        update_texture_priority(wrapper, true);
}
