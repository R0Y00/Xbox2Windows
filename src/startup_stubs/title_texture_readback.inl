// A placed title buffer may have a separate desktop allocation or a tight, unaligned
// footprint. Stage the copied texels and publish the actual rows at the guest pitch.
struct TitleTextureReadback {
    Microsoft::WRL::ComPtr<ID3D12Resource> staging;
    Microsoft::WRL::ComPtr<ID3D12Fence> last;
    unsigned char* dest{};
    UINT width{}, rows{}, host_pitch{}, title_pitch{};
};

using TextureReadbackPtr = std::shared_ptr<TitleTextureReadback>;
std::mutex texture_readback_mutex;
std::unordered_map<ID3D12CommandList*, std::vector<TextureReadbackPtr>> texture_readback_lists;

void discard_texture_readbacks(ID3D12CommandList* list)
{
    std::lock_guard lock(texture_readback_mutex);
    texture_readback_lists.erase(list);
}

struct TextureReadbackFlight {
    std::vector<TextureReadbackPtr> copies;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    HANDLE event{};
    ~TextureReadbackFlight() { if (event) CloseHandle(event); }
};

void CALLBACK land_texture_readbacks(PTP_CALLBACK_INSTANCE, void* data, PTP_WAIT wait, TP_WAIT_RESULT)
{
    std::unique_ptr<TextureReadbackFlight> flight(static_cast<TextureReadbackFlight*>(data));
    if (flight->fence->GetCompletedValue() == 1) {
        for (const auto& copy : flight->copies) {
            void* mapped{};
            const SIZE_T size = SIZE_T(copy->rows - 1) * copy->host_pitch + copy->width;
            const D3D12_RANGE read{0, size};
            if (FAILED(copy->staging->Map(0, &read, &mapped))) continue;
            bool landed = true;
            for (UINT y = 0; y < copy->rows && landed; ++y) {
                auto* to = copy->dest + SIZE_T(y) * copy->title_pitch;
                auto* from = static_cast<unsigned char*>(mapped) + SIZE_T(y) * copy->host_pitch;
                SIZE_T wrote{};
                landed = x2w_local::writable(to, copy->width) &&
                    WriteProcessMemory(GetCurrentProcess(), to, from, copy->width, &wrote) && wrote == copy->width;
            }
            static LONG reports, failures;
            if ((!landed && first_calls(failures, 8)) || (landed && first_calls(reports, 16)))
                log("x2w-gpu placed texture readback landed=%u address=%p row_bytes=%u rows=%u first_float=%g",
                    unsigned(landed), copy->dest, copy->width, copy->rows,
                    copy->width >= sizeof(float) ? double(*static_cast<const float*>(mapped)) : 0.);
            const D3D12_RANGE none{};
            copy->staging->Unmap(0, &none);
        }
    }
    flight->fence->Signal(2);
    if (wait) CloseThreadpoolWait(wait);
}

// Preserve staging until the queue reaches the copies. The queue then waits for CPU
// publication, so a later guest fence can be observed without any subsequent present.
std::unique_ptr<TextureReadbackFlight> prepare_texture_readbacks(ID3D12CommandQueue* queue,
                                                                const std::vector<ID3D12CommandList*>& lists)
{
    std::lock_guard lock(texture_readback_mutex);
    std::unique_ptr<TextureReadbackFlight> flight;
    for (auto* list : lists) {
        const auto found = texture_readback_lists.find(list);
        if (found == texture_readback_lists.end()) continue;
        if (!flight) flight = std::make_unique<TextureReadbackFlight>();
        for (const auto& copy : found->second) {
            if (copy->last && copy->last->GetCompletedValue() < 2) queue->Wait(copy->last.Get(), 2);
            flight->copies.push_back(copy);
        }
    }
    if (!flight) return {};
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (FAILED(queue->GetDevice(IID_PPV_ARGS(&device))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&flight->fence)))) {
        log("x2w-gpu placed texture readback could not create submission fence");
        return {};
    }
    for (const auto& copy : flight->copies) copy->last = flight->fence;
    return flight;
}

void submit_texture_readbacks(ID3D12CommandQueue* queue, std::unique_ptr<TextureReadbackFlight> flight)
{
    if (!flight) return;
    if (FAILED(queue->Signal(flight->fence.Get(), 1))) {
        flight->fence->Signal(2);
        log("x2w-gpu placed texture readback queue signal failed");
        return;
    }
    // A pending thread-pool callback can outlive the last application reference to this DLL.
    static const bool pinned = [] {
        HMODULE module{};
        return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                                  reinterpret_cast<LPCWSTR>(&land_texture_readbacks), &module) != FALSE;
    }();
    if (pinned) flight->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const auto wait = flight->event ? CreateThreadpoolWait(land_texture_readbacks, flight.get(), nullptr) : nullptr;
    if (wait && SUCCEEDED(flight->fence->SetEventOnCompletion(1, flight->event))) {
        queue->Wait(flight->fence.Get(), 2);
        auto* pending = flight.release();
        SetThreadpoolWait(wait, pending->event, nullptr);
        return;
    }
    if (wait) CloseThreadpoolWait(wait);
    // Losing an event must not expose stale bytes or release an active readback buffer.
    if (SUCCEEDED(flight->fence->SetEventOnCompletion(1, nullptr))) {
        land_texture_readbacks(nullptr, flight.release(), nullptr, 0);
    } else {
        flight->fence->Signal(2);
        log("x2w-gpu placed texture readback completion failed");
    }
}
