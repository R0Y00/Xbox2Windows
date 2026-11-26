// Translated shader ABI memory adapter (PLAN.md section 1.5, WP3, owner vbind): what lies at a title GPU address,
// bounded by the title buffer or the contiguous run of its window. Included by d3d12_xs_device.cpp inside its
// anonymous namespace, before gpu_bind.inl (it uses desktop_range and the address indexes). Contract:
// gpu_hooks.h. The bounding itself is x2w_vbind::find_bounded (gpu_roots.h, header-tested); this is X2W's
// side of it.
#pragma once

namespace vbind {
bool buffer_extent(std::uint64_t address, std::uint64_t* base, std::uint64_t* end, const Wrapper** identity,
                   std::uint64_t length = 1)
{
    x2w_vbind::TitleBufferExtent extent;
    AcquireSRWLockShared(&note_lock);
    resource_address_index().for_each_reaching(address, 1, [&](const Wrapper* w) {
        const auto it = resource_infos().find(w);
        if (it == resource_infos().end()) return false;
        const ResourceInfo& info = it->second;
        if (info.console.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) return false;
        if (InterlockedCompareExchange(&const_cast<Wrapper*>(w)->own().refs, 0, 0) <= 0) return false;
        const std::uint64_t b = reinterpret_cast<std::uint64_t>(w->own().title_memory), width = info.console.Width;
        extent.consider(address, length, b, width, reinterpret_cast<std::uintptr_t>(w));
        return false;
    });
    ReleaseSRWLockShared(&note_lock);
    if (!extent.base) return false;
    *base = extent.base;
    *end = extent.end;
    if (identity) *identity = reinterpret_cast<const Wrapper*>(extent.identity);
    return true;
}
// find_bounded over X2W: windows from desktop_range (cached per frame), extents from the resource records.
bool find(std::uint64_t address, std::uint64_t len, std::uint64_t most, std::uint64_t unknown_most, x2w_vbind::Found* out)
{
    // Title buffers can be created or released mid-frame. Recheck the live extent,
    // then reuse the prefix previously validated against this frame's desktop windows.
    std::uint64_t base = 0, end = 0;
    const bool known = address && len && buffer_extent(address, &base, &end, nullptr, len) && end > address;
    const auto limit = (std::min)(most, known ? end - address : unknown_most);
    struct Prefix {
        std::uint64_t address{}, len{}, limit{}, generation{};
        LONG frame{-1};
        x2w_vbind::Found found{};
    };
    static thread_local std::array<Prefix, 256> prefixes{};
    const LONG frame = frame_number;
    const auto generation = gpu_mapping_generation();
    const auto mix = (address ^ (len * 1099511628211ull) ^ limit) * 0x9e3779b97f4a7c15ull;
    auto& slot = prefixes[(mix >> 32) % prefixes.size()];
    if (slot.frame == frame && slot.generation == generation && slot.address == address && slot.len == len && slot.limit == limit &&
        slot.found.known_extent == known) {
        *out = slot.found;
        return true;
    }
    const bool found = x2w_vbind::find_bounded(address, len, most, unknown_most,
        [&](std::uint64_t, std::uint64_t* b, std::uint64_t* e) { *b = base; *e = end; return known; },
        [](std::uint64_t a, std::uint64_t n, void** buffer, std::uint64_t* offset) {
            ID3D12Resource* b = nullptr;
            const bool ok = desktop_range(a, n, &b, offset);
            *buffer = b;
            return ok && b;
        }, out);
    if (found && gpu_mapping_generation() == generation) slot = {address, len, limit, generation, frame, *out};
    return found;
}
}  // namespace vbind

bool gpu_find(std::uint64_t address, std::uint64_t bytes, GpuFound* out)
{
    x2w_vbind::Found f;
    if (!out) return false;
    *out = {};
    const std::uint64_t len = bytes ? bytes : 1;
    if (!vbind::find(address, len, ~0ull, len, &f)) return false;
    out->resource = static_cast<ID3D12Resource*>(f.buffer);
    out->offset = f.offset;
    out->bytes_left = f.left;
    return true;
}
