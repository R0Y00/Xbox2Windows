
ID3D12Resource* vdesc_buffer;
std::uint32_t* vdesc_words;
LONG vdesc_image, vdesc_buffer_views, vdesc_none;  // ids filled, by kind (totals)

// fill_view_kinds (d3d12_compute.inl), for every view record it fills; draw_lock held.
void vdesc_fill(const ViewRecord& record)
{
    if (!vdesc_words || record.id >= view_ids_max) return;
    std::uint32_t out[8];
    bool image = false, unmapped = false;
    const bool made = view_record_words(record, out, &image, &unmapped);  // out is zero when !made
    if (made && image && out[1] == 0) out[1] = 1u << 8;
    std::memcpy(vdesc_words + u64{record.id} * 8, out, 32);
    InterlockedIncrement(!made ? &vdesc_none : image ? &vdesc_image : &vdesc_buffer_views);
    static LONG shown;
    if (made && first_calls(shown, 4))
        log("x2w-vdesc view %u (slot %u): %s words %08x %08x %08x %08x %08x %08x %08x %08x%s", record.id, record.slot, image ? "T#" : "V#",
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], unmapped ? " (format has no gfx10 code: FORMAT 0)" : "");
}

// draw_translated (d3d12_draw.inl), a `heapwords 1` translation, after fill_views; draw_lock held. Creates the
// buffer on first use and fills the views filled so far. False: no buffer (the draw / mesh call is skipped).
bool ensure_vdesc()
{
    static bool failed;
    if (vdesc_words || failed) return vdesc_words != nullptr;
    D3D12_HEAP_PROPERTIES up{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = u64{view_ids_max} * 32;
    rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const HRESULT hr = device_of(the_device)->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                                       IID_PPV_ARGS(&vdesc_buffer));
    std::uint32_t* words = nullptr;
    if (SUCCEEDED(hr) && FAILED(vdesc_buffer->Map(0, nullptr, reinterpret_cast<void**>(&words)))) words = nullptr;
    if (!words) {
        failed = true;
        log("x2w-vdesc buffer (%llu bytes) -> 0x%08lx: heapwords translations are skipped", static_cast<unsigned long long>(rd.Width),
            static_cast<unsigned long>(hr));
        return false;
    }
    std::memset(words, 0, static_cast<std::size_t>(rd.Width));
    vdesc_words = words;
    for (UINT i = 0; i < views_filled; ++i) {
        AcquireSRWLockShared(&note_lock);
        const ViewRecord record = view_records()[i];
        ReleaseSRWLockShared(&note_lock);
        vdesc_fill(record);
    }
    log("x2w-vdesc buffer %u ids x 32 bytes at desktop 0x%llx; %u views filled so far (T# %ld, V# %ld, none %ld)", view_ids_max,
        static_cast<unsigned long long>(vdesc_buffer->GetGPUVirtualAddress()), views_filled, vdesc_image, vdesc_buffer_views, vdesc_none);
    return true;
}
