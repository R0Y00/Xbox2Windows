// Included from d3d12_xs_device.cpp after the command-list state helpers.
//
// Whole-resource copies on a recording list (render mode): CopyBufferRegion (list slot 15), CopyResource
// (17) and ResolveSubresource (19). The title's resources are mapped to desktop ones, the pair is checked
// by src/startup_stubs/list_copy.h, and a pair the desktop would reject is counted and skipped instead of
// forwarded. Desktop resources rest in COMMON between lists, so each is moved to the copy or resolve state
// around the call and back at Close; upload and readback buffers keep the state their heap requires.
// CopyTextureRegion (16) and CopyTiles (18) carry footprints and addresses and are not handled here.

LONG list_buffer_copies, list_resource_copies, list_resolves, list_copies_refused;

// The desktop resource behind a copy argument; null for a null value, or a wrapper that is not a resource.
// A value that is no wrapper is taken as a desktop resource, as the view slots do.
ID3D12Resource* copy_endpoint(u64 value)
{
    if (!value) return nullptr;
    if (const Wrapper* w = as_wrapper(reinterpret_cast<const void*>(value))) {
        if (w->own().kind != resource_kind) return nullptr;
        return static_cast<ID3D12Resource*>(w->own().real);
    }
    return x2w_local::readable(reinterpret_cast<const void*>(value), sizeof(void*)) ? reinterpret_cast<ID3D12Resource*>(value) : nullptr;
}

void refuse_copy(const char* what, x2w_startup::CopyRefusal reason)
{
    InterlockedIncrement(&list_copies_refused);
    static LONG logged;
    if (first_calls(logged, 6)) log("x2w-d3d12 %s skipped: %s", what, x2w_startup::copy_refusal_name(reason));
}

// The state a copy needs from `resource`: an upload buffer stays readable and a readback buffer stays a
// copy destination, because desktop allows no other state for them.
void require_copy_state(Wrapper* self, ID3D12Resource* resource, D3D12_RESOURCE_STATES state)
{
    D3D12_HEAP_PROPERTIES heap{};
    D3D12_HEAP_FLAGS flags{};
    if (SUCCEEDED(resource->GetHeapProperties(&heap, &flags)) &&
        (heap.Type == D3D12_HEAP_TYPE_UPLOAD || heap.Type == D3D12_HEAP_TYPE_READBACK))
        return;
    require_state(self, resource, state);
}

// List slot 15, CopyBufferRegion(destination, destination offset, source, source offset, bytes).
void __stdcall list_copy_buffer_region(Wrapper* self, u64 destination, u64 destination_offset, u64 source, u64 source_offset,
                                       u64 bytes)
{
    if (!self->own().list_open) return;
    ID3D12Resource* const to = copy_endpoint(destination);
    ID3D12Resource* const from = copy_endpoint(source);
    D3D12_RESOURCE_DESC to_desc{}, from_desc{};
    if (to) to_desc = to->GetDesc();
    if (from) from_desc = from->GetDesc();
    const auto refusal = x2w_startup::check_copy_buffer_region(to ? &to_desc : nullptr, destination_offset,
                                                               from ? &from_desc : nullptr, source_offset, bytes, to == from);
    if (refusal != x2w_startup::CopyRefusal::none) return refuse_copy("CopyBufferRegion", refusal);
    require_copy_state(self, from, D3D12_RESOURCE_STATE_COPY_SOURCE);
    require_copy_state(self, to, D3D12_RESOURCE_STATE_COPY_DEST);
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)->CopyBufferRegion(to, destination_offset, from, source_offset, bytes);
    InterlockedIncrement(&list_buffer_copies);
}

// List slot 17, CopyResource(destination, source).
void __stdcall list_copy_resource(Wrapper* self, u64 destination, u64 source)
{
    if (!self->own().list_open) return;
    ID3D12Resource* const to = copy_endpoint(destination);
    ID3D12Resource* const from = copy_endpoint(source);
    D3D12_RESOURCE_DESC to_desc{}, from_desc{};
    if (to) to_desc = to->GetDesc();
    if (from) from_desc = from->GetDesc();
    const auto refusal = x2w_startup::check_copy_resource(to ? &to_desc : nullptr, from ? &from_desc : nullptr, to == from);
    if (refusal != x2w_startup::CopyRefusal::none) return refuse_copy("CopyResource", refusal);
    require_copy_state(self, from, D3D12_RESOURCE_STATE_COPY_SOURCE);
    require_copy_state(self, to, D3D12_RESOURCE_STATE_COPY_DEST);
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)->CopyResource(to, from);
    InterlockedIncrement(&list_resource_copies);
}

// List slot 19, ResolveSubresource(destination, destination subresource, source, source subresource, format).
// The format is a console value, mapped like a resource format; a typeless one resolves as its typed choice.
void __stdcall list_resolve_subresource(Wrapper* self, u64 destination, UINT destination_subresource, u64 source,
                                        UINT source_subresource, UINT format)
{
    if (!self->own().list_open) return;
    ID3D12Resource* const to = copy_endpoint(destination);
    ID3D12Resource* const from = copy_endpoint(source);
    D3D12_RESOURCE_DESC to_desc{}, from_desc{};
    if (to) to_desc = to->GetDesc();
    if (from) from_desc = from->GetDesc();
    const auto plan = x2w_startup::plan_resolve(to ? &to_desc : nullptr, destination_subresource, from ? &from_desc : nullptr,
                                                source_subresource, desktop_format(static_cast<DXGI_FORMAT>(format)), to == from);
    if (plan.refusal != x2w_startup::CopyRefusal::none) return refuse_copy("ResolveSubresource", plan.refusal);
    require_copy_state(self, from, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    require_copy_state(self, to, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)
        ->ResolveSubresource(to, destination_subresource, from, source_subresource, plan.format);
    InterlockedIncrement(&list_resolves);
}

// Fixtures: how many buffer copies, resource copies and resolves were forwarded, and how many were refused.
extern "C" __declspec(dllexport) void x2w_d3d12_list_copy_stats(unsigned* buffer_copies, unsigned* resource_copies,
                                                                unsigned* resolves, unsigned* refused)
{
    if (buffer_copies) *buffer_copies = static_cast<unsigned>(InterlockedCompareExchange(&list_buffer_copies, 0, 0));
    if (resource_copies) *resource_copies = static_cast<unsigned>(InterlockedCompareExchange(&list_resource_copies, 0, 0));
    if (resolves) *resolves = static_cast<unsigned>(InterlockedCompareExchange(&list_resolves, 0, 0));
    if (refused) *refused = static_cast<unsigned>(InterlockedCompareExchange(&list_copies_refused, 0, 0));
}
