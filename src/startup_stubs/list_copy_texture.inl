// Included by d3d12_xs_device.cpp after d3d12_draw.inl. Command list slot 16, CopyTextureRegion (#870).
// Each location's resource is mapped to its desktop resource, the call is checked by
// list_copy_texture.h, the two resources move to COPY_SOURCE / COPY_DEST for the call (the list's Close
// puts them back to COMMON), and a call the desktop would reject is counted and logged, not forwarded.
#include "list_copy_texture.h"
#include "title_texture_readback.inl"
#include "bc_bind_trace.inl"

volatile LONG list_copy_texture_forwarded, list_copy_texture_refused;

namespace {

// Memory planes of a format on the resource's device (1 when the device cannot say).
UINT copy_texture_planes(ID3D12Resource* resource, DXGI_FORMAT format)
{
    ID3D12Device* device = nullptr;
    if (FAILED(resource->GetDevice(IID_PPV_ARGS(&device))) || !device) return 1;
    D3D12_FEATURE_DATA_FORMAT_INFO info{format, 1};
    const bool known = SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &info, sizeof(info)));
    device->Release();
    return known && info.PlaneCount ? info.PlaneCount : 1;
}

// One side of the copy, read from the desktop resource it names. False when the location is unusable.
bool copy_texture_side(const D3D12_TEXTURE_COPY_LOCATION* location, ID3D12Resource*& real, x2w_list_copy_texture::Side& side,
                       DXGI_FORMAT& format, bool& bad_subresource)
{
    Wrapper* w = location && x2w_local::readable(location, sizeof(*location)) ? as_wrapper(location->pResource) : nullptr;
    if (!w || w->own().kind != resource_kind || !w->own().real) return false;
    real = static_cast<ID3D12Resource*>(w->own().real);
    const D3D12_RESOURCE_DESC desc = real->GetDesc();
    side = {};
    side.block_w = side.block_h = 1;
    side.samples = desc.SampleDesc.Count ? desc.SampleDesc.Count : 1;
    side.depth_stencil = (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL) != 0 ||
        desc.Format == DXGI_FORMAT_D16_UNORM || desc.Format == DXGI_FORMAT_D32_FLOAT;
    side.mips = side.array_size = side.planes = 1;
    if (location->Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
        if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) return false;
        const auto& fp = location->PlacedFootprint;
        side.kind = x2w_list_copy_texture::Kind::placed_footprint;
        side.width = fp.Footprint.Width;
        side.height = fp.Footprint.Height;
        side.depth = fp.Footprint.Depth;
        side.offset = fp.Offset;
        side.row_pitch = fp.Footprint.RowPitch;
        side.resource_bytes = desc.Width;
        side.format = fp.Footprint.Format;
        format = fp.Footprint.Format;
        return true;
    }
    if (location->Type != D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX || desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return false;
    side.mips = desc.MipLevels ? desc.MipLevels : 1;
    side.array_size = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1 : desc.DepthOrArraySize;
    side.planes = copy_texture_planes(real, desc.Format);
    side.subresource = location->SubresourceIndex;
    side.format = desc.Format;
    // The index must lie inside mips * array slices * planes; the mip it names sizes this side.
    x2w_list_copy_texture::Subresource parts{};
    if (!x2w_list_copy_texture::decode_subresource(side.subresource, side.mips, side.array_size, side.planes, parts)) {
        bad_subresource = true;
        return false;
    }
    const UINT mip = parts.mip;
    side.kind = x2w_list_copy_texture::Kind::subresource;
    side.width = static_cast<UINT>((std::max<UINT64>)(1, desc.Width >> mip));
    side.height = (std::max<UINT>)(1, desc.Height >> mip);
    side.depth = desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? (std::max<UINT>)(1, desc.DepthOrArraySize >> mip) : 1;
    format = desc.Format;
    return true;
}

// Texels per block and bytes per block, from the desktop's own footprint for a one-subresource copy.
void copy_texture_block(ID3D12Resource* texture, x2w_list_copy_texture::Side& side)
{
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    ID3D12Device* device = nullptr;
    if (FAILED(texture->GetDevice(IID_PPV_ARGS(&device))) || !device) return;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &rows, &row_bytes, &total);
    device->Release();
    const bool bc = (desc.Format >= DXGI_FORMAT_BC1_TYPELESS && desc.Format <= DXGI_FORMAT_BC5_SNORM) ||
                    (desc.Format >= DXGI_FORMAT_BC6H_TYPELESS && desc.Format <= DXGI_FORMAT_BC7_UNORM_SRGB);
    side.block_w = side.block_h = bc ? 4 : 1;
    const UINT64 blocks = (std::max<UINT64>)(1, (desc.Width + side.block_w - 1) / side.block_w);
    side.bytes_per_block = static_cast<std::uint32_t>(row_bytes / blocks);
}

// Upload buffers stay in GENERIC_READ and readback buffers in COPY_DEST: the desktop refuses a barrier on either.
bool fixed_heap_state(ID3D12Resource* resource, D3D12_HEAP_TYPE& type)
{
    D3D12_HEAP_PROPERTIES props{};
    D3D12_HEAP_FLAGS flags{};
    if (FAILED(resource->GetHeapProperties(&props, &flags))) return false;
    type = props.Type;
    return type == D3D12_HEAP_TYPE_UPLOAD || type == D3D12_HEAP_TYPE_READBACK;
}

}  // namespace

// List slot 16 (unshifted, like slots 12 to 25), CopyTextureRegion(dst, x, y, z, src, box).
bool encoded_block_pair(DXGI_FORMAT raw, DXGI_FORMAT bc)
{
    switch (bc) {
    case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_UNORM: case DXGI_FORMAT_BC4_SNORM:
        return raw == DXGI_FORMAT_R32G32_UINT || raw == DXGI_FORMAT_R32G32_SINT ||
               raw == DXGI_FORMAT_R16G16B16A16_UINT || raw == DXGI_FORMAT_R16G16B16A16_SINT;
    case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_UF16: case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_UNORM: case DXGI_FORMAT_BC7_UNORM_SRGB:
        return raw == DXGI_FORMAT_R32G32B32A32_UINT || raw == DXGI_FORMAT_R32G32B32A32_SINT;
    default: return false;
    }
}

x2w_list_copy_texture::Verdict check_block_copy(x2w_list_copy_texture::Side dst, UINT x, UINT y, UINT z,
    x2w_list_copy_texture::Side src, const x2w_list_copy_texture::Box* box, bool compressed_source)
{
    using namespace x2w_list_copy_texture;
    auto blocks = [](UINT pixels) { return pixels / 4 + (pixels % 4 != 0); };
    auto& compressed = compressed_source ? src : dst;
    Box area{};
    if (compressed_source && box) {
        if (box->right > src.width || box->bottom > src.height) return Verdict::source_overflow;
        if (box->left % 4 || box->top % 4 || (box->right % 4 && box->right != src.width) ||
            (box->bottom % 4 && box->bottom != src.height)) return Verdict::bad_alignment;
        area = {box->left/4, box->top/4, box->front, blocks(box->right), blocks(box->bottom), box->back};
        box = &area;
    } else if (!compressed_source) {
        if (x % 4 || y % 4) return Verdict::bad_alignment;
        x /= 4;
        y /= 4;
    }
    compressed.width = blocks(compressed.width);
    compressed.height = blocks(compressed.height);
    dst.block_w = dst.block_h = src.block_w = src.block_h = 1;
    return check(dst, x, y, z, src, box, true);
}

void mark_copied_texture(Wrapper*, Wrapper*, ID3D12Resource*);

ID3D12Resource* recording_plane_scratch(Wrapper* self, UINT64 bytes)
{
    auto& state = self->own();
    auto*& scratch = state.plane_copy_scratch;
    if (scratch && scratch->GetDesc().Width >= bytes) return scratch;
    Microsoft::WRL::ComPtr<ID3D12Resource> created;
    if (!x2w_vmemory::buffer(device_of(state.list_device), bytes, D3D12_HEAP_TYPE_DEFAULT, false, &created)) return nullptr;
    ID3D12Resource* buffer = nullptr;
    AcquireSRWLockExclusive(&allocator_lock);
    const auto found = list_recordings.find(self);
    if (found != list_recordings.end() && found->second) {
        try {
            found->second->scratches.push_back(std::move(created));
            buffer = scratch = found->second->scratches.back().Get();
        } catch (const std::bad_alloc&) {}
    }
    ReleaseSRWLockExclusive(&allocator_lock);
    return buffer;
}

bool copy_depth_plane(Wrapper* self, ID3D12Resource* src, const x2w_list_copy_texture::Side& s,
                     ID3D12Resource* dst, const x2w_list_copy_texture::Side& d)
{
    using x2w_list_copy_texture::Kind;
    if (s.kind != Kind::subresource || d.kind != Kind::subresource ||
        s.format != DXGI_FORMAT_R32G8X24_TYPELESS || d.format != DXGI_FORMAT_R32_TYPELESS ||
        !s.depth_stencil || !d.depth_stencil || s.planes != 2 || d.planes != 1 ||
        s.subresource >= s.mips * s.array_size || s.samples != 1 || d.samples != 1 ||
        s.width != d.width || s.height != d.height || s.depth != 1 || d.depth != 1) return false;
    auto* device = device_of(self->own().list_device);
    const auto from_desc = src->GetDesc(), to_desc = dst->GetDesc();
    if (from_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || to_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT from{}, to{};
    UINT64 bytes{}, dest_bytes{};
    device->GetCopyableFootprints(&from_desc, s.subresource, 1, 0, &from, nullptr, nullptr, &bytes);
    device->GetCopyableFootprints(&to_desc, d.subresource, 1, 0, &to, nullptr, nullptr, &dest_bytes);
    if (!bytes || bytes == UINT64_MAX || bytes != dest_bytes || from.Offset != to.Offset ||
        from.Footprint.Format != DXGI_FORMAT_R32_TYPELESS || to.Footprint.Format != DXGI_FORMAT_R32_TYPELESS ||
        from.Footprint.RowPitch != to.Footprint.RowPitch || from.Footprint.Width != s.width ||
        to.Footprint.Width != d.width || from.Footprint.Height != s.height || to.Footprint.Height != d.height) return false;
    auto* scratch = recording_plane_scratch(self, bytes);
    if (!scratch) return false;
    auto* list = recording(self);
    const auto before = current_planes(self, src, D3D12_RESOURCE_STATE_COMMON);
    require_planes(self, src, D3D12_RESOURCE_STATE_COPY_SOURCE, before.stencil);
    require_state(self, scratch, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION image{src, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    image.SubresourceIndex = s.subresource;
    D3D12_TEXTURE_COPY_LOCATION buffer{scratch, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    buffer.PlacedFootprint = from;
    list->CopyTextureRegion(&buffer, 0, 0, 0, &image, nullptr);
    require_state(self, scratch, D3D12_RESOURCE_STATE_COPY_SOURCE);
    require_state(self, dst, D3D12_RESOURCE_STATE_COPY_DEST);
    buffer.PlacedFootprint = to;
    image.pResource = dst;
    image.SubresourceIndex = d.subresource;
    list->CopyTextureRegion(&image, 0, 0, 0, &buffer, nullptr);
    return true;
}

void report_planar_copy(UINT slot, const x2w_list_copy_texture::Side& src, ID3D12Resource* src_resource,
                        const x2w_list_copy_texture::Side& dst, ID3D12Resource* dst_resource,
                        Wrapper* src_wrapper, Wrapper* dst_wrapper, UINT x, UINT y, UINT z, const D3D12_BOX* box)
{
    using namespace x2w_list_copy_texture;
    static LONG64 refused;
    const auto n = InterlockedIncrement64(&refused);
    if ((n & (n - 1)) == 0) log("x2w-d3d12 planar copies refused %lld", n);
    Subresource s{}, d{};
    decode_subresource(src.subresource, src.mips, src.array_size, src.planes, s);
    decode_subresource(dst.subresource, dst.mips, dst.array_size, dst.planes, d);
    using Key = std::array<UINT, 10>;
    const Key key{slot, UINT(src.kind), src.format, s.plane, src.samples, UINT(dst.kind), dst.format, d.plane, dst.samples, UINT(box != nullptr)};
    static std::mutex mutex;
    static std::array<Key, 24> seen{};
    static size_t used = 0;
    {
        std::lock_guard guard(mutex);
        if (used == seen.size() || std::find(seen.begin(), seen.begin() + used, key) != seen.begin() + used) return;
        seen[used++] = key;
    }
    D3D12_BOX area{};
    SIZE_T copied = 0;
    const bool valid = box && ReadProcessMemory(GetCurrentProcess(), box, &area, sizeof(area), &copied) && copied == sizeof(area);
    log("x2w-d3d12 planar copy sample=%lld slot=%u dst_origin=%u,%u,%u box=%s %u,%u,%u..%u,%u,%u",
        n, slot, x, y, z, !box ? "full" : valid ? "region" : "unreadable",
        area.left, area.top, area.front, area.right, area.bottom, area.back);
    const auto side = [&](const char* name, const Side& value, Subresource parts, ID3D12Resource* resource, Wrapper* wrapper) {
        const auto desc = resource->GetDesc();
        const auto* title = resource_info(wrapper);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        if (value.kind == x2w_list_copy_texture::Kind::placed_footprint)
            placed = {value.offset, {DXGI_FORMAT(value.format), value.width, value.height, value.depth, value.row_pitch}};
        else if (value.samples == 1) {
            Microsoft::WRL::ComPtr<ID3D12Device> device;
            if (SUCCEEDED(resource->GetDevice(IID_PPV_ARGS(&device))))
                device->GetCopyableFootprints(&desc, value.subresource, 1, 0, &placed, nullptr, nullptr, nullptr);
        }
        const auto& f = placed.Footprint;
        log("x2w-d3d12 planar copy sample=%lld %s kind=%u title_fmt=%u title_layout=0x%x resource_fmt=%u dim=%u flags=0x%x samples=%u "
            "sub=%u plane=%u/%u mip=%u/%u slice=%u/%u mip_size=%ux%ux%u footprint_fmt=%u footprint=%ux%ux%u pitch=%u offset=%llu",
            n, name, UINT(value.kind), title ? UINT(title->console.Format) : UINT_MAX, title ? UINT(title->console.Layout) : UINT_MAX,
            UINT(desc.Format), UINT(desc.Dimension), UINT(desc.Flags), value.samples,
            value.subresource, parts.plane, value.planes, parts.mip, value.mips, parts.slice, value.array_size,
            value.width, value.height, value.depth, UINT(f.Format), f.Width, f.Height, f.Depth, f.RowPitch, placed.Offset);
    };
    side("src", src, s, src_resource, src_wrapper);
    side("dst", dst, d, dst_resource, dst_wrapper);
}

void record_texture_copy(UINT slot, Wrapper* self, const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
                                 const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box)
{
    if (!self->own().list_open) return;
    static LONG logged;
    const auto refuse = [&](const char* why) {
        InterlockedIncrement(&list_copy_texture_refused);
        if (first_calls(logged, 6)) log("x2w-d3d12 CopyTextureRegion refused: %s", why);
    };
    ID3D12Resource *real_dst = nullptr, *real_src = nullptr;
    x2w_list_copy_texture::Side side_dst{}, side_src{};
    DXGI_FORMAT format_dst = DXGI_FORMAT_UNKNOWN, format_src = DXGI_FORMAT_UNKNOWN;
    bool bad_subresource = false;
    if (!copy_texture_side(dst, real_dst, side_dst, format_dst, bad_subresource) ||
        !copy_texture_side(src, real_src, side_src, format_src, bad_subresource))
        return refuse(bad_subresource ? "bad_subresource" : "unknown_resource");
    // Planar (depth-stencil, video) copies name a plane footprint format; only single-plane copies are forwarded.
    if (side_dst.planes > 1 || side_src.planes > 1) {
        if (!box && !x && !y && !z && copy_depth_plane(self, real_src, side_src, real_dst, side_dst)) {
            auto* dst_wrapper = as_wrapper(dst->pResource);
            mark_copied_texture(self, dst_wrapper, real_dst);
            retire_hold(self, dst_wrapper);
            retire_hold(self, as_wrapper(src->pResource));
            InterlockedIncrement(&list_copy_texture_forwarded);
            static LONG64 copies;
            const auto n = InterlockedIncrement64(&copies);
            if ((n & (n - 1)) == 0) log("x2w-d3d12 depth-plane copies forwarded %lld slot=%u %ux%u", n, slot, side_src.width, side_src.height);
            return;
        }
        report_planar_copy(slot, side_src, real_src, side_dst, real_dst, as_wrapper(src->pResource), as_wrapper(dst->pResource), x, y, z, box);
        return refuse("planar_unsupported");
    }
    ID3D12Resource* texture = side_dst.kind == x2w_list_copy_texture::Kind::subresource ? real_dst : real_src;
    if (side_dst.kind == x2w_list_copy_texture::Kind::subresource || side_src.kind == x2w_list_copy_texture::Kind::subresource) {
        x2w_list_copy_texture::Side blocks{};
        copy_texture_block(texture, blocks);
        for (auto* s : {&side_dst, &side_src}) {
            s->block_w = blocks.block_w ? blocks.block_w : 1;
            s->block_h = blocks.block_h ? blocks.block_h : 1;
            if (s->kind == x2w_list_copy_texture::Kind::placed_footprint) s->bytes_per_block = blocks.bytes_per_block;
        }
    }
    x2w_list_copy_texture::Box copy_box{};
    if (box && x2w_local::readable(box, sizeof(*box))) copy_box = {box->left, box->top, box->front, box->right, box->bottom, box->back};
    const bool same_family = x2w_startup::compatible_formats(format_src, format_dst);
    const bool textures = side_src.kind == x2w_list_copy_texture::Kind::subresource &&
        side_dst.kind == x2w_list_copy_texture::Kind::subresource && side_src.samples == 1 && side_dst.samples == 1;
    const bool compressed_source = textures && encoded_block_pair(format_dst, format_src);
    const bool encoded = compressed_source || (textures && encoded_block_pair(format_src, format_dst));
    const auto verdict = encoded ? check_block_copy(side_dst, x, y, z, side_src, box ? &copy_box : nullptr, compressed_source)
        : x2w_list_copy_texture::check(side_dst, x, y, z, side_src, box ? &copy_box : nullptr, same_family);
    if (verdict == x2w_list_copy_texture::Verdict::format_mismatch) {
        // Report each distinct pair once: the first calls can all be copies of one atlas level.
        using Shape = std::array<UINT, 5>;
        static std::mutex mutex;
        static std::array<Shape, 16> seen{};
        static size_t used{};
        const Shape shape{slot, UINT(side_src.kind), UINT(format_src), UINT(side_dst.kind), UINT(format_dst)};
        std::lock_guard guard(mutex);
        if (used < seen.size() && std::find(seen.begin(), seen.begin() + used, shape) == seen.begin() + used) {
            seen[used++] = shape;
            log("x2w-d3d12 CopyTextureRegion slot%u format mismatch src kind=%u fmt=%u %ux%ux%u sub=%u "
                "dst kind=%u fmt=%u %ux%ux%u sub=%u corner=%u,%u,%u box=%u",
                slot, shape[1], shape[2], side_src.width, side_src.height, side_src.depth, side_src.subresource,
                shape[3], shape[4], side_dst.width, side_dst.height, side_dst.depth, side_dst.subresource, x, y, z, unsigned(box != nullptr));
        }
    }
    Wrapper* const src_wrapper = as_wrapper(src->pResource);
    Wrapper* const dst_wrapper = as_wrapper(dst->pResource);
    const auto* dest_info = resource_info(dst_wrapper);
    if (side_dst.kind == x2w_list_copy_texture::Kind::placed_footprint &&
        side_src.kind == x2w_list_copy_texture::Kind::subresource && dst_wrapper->own().title_memory && dest_info &&
        (dest_info->placement != Placement::aliased_buffer ||
         verdict != x2w_list_copy_texture::Verdict::ok)) {
        // Stage only the source region, leaving the uncopied part of the CPU footprint intact.
        if (side_dst.depth != 1 || side_src.samples != 1 || !side_dst.bytes_per_block || !side_dst.row_pitch)
            return refuse("unsupported_title_footprint");
        const auto width = box ? copy_box.right - copy_box.left : side_src.width;
        const auto height = box ? copy_box.bottom - copy_box.top : side_src.height;
        if (!width || !height || z || UINT64(x) + width > side_dst.width || UINT64(y) + height > side_dst.height ||
            x % side_dst.block_w || y % side_dst.block_h) return refuse("bad_title_region");
        const UINT64 row = ((UINT64(width) + side_dst.block_w - 1) / side_dst.block_w) * side_dst.bytes_per_block;
        const UINT64 host_pitch = (row + 255) & ~UINT64(255);
        const UINT rows = UINT((UINT64(height) + side_dst.block_h - 1) / side_dst.block_h);
        const UINT64 start_x = UINT64(x / side_dst.block_w) * side_dst.bytes_per_block;
        const UINT64 within = UINT64(y / side_dst.block_h) * side_dst.row_pitch + start_x;
        UINT64 start{}, end{};
        if (row + start_x > side_dst.row_pitch || host_pitch > UINT_MAX ||
            !x2w_list_copy_texture::checked_add(side_dst.offset, within, start) ||
            !x2w_list_copy_texture::checked_add(start, UINT64(rows - 1) * side_dst.row_pitch + row, end) ||
            end > dest_info->console.Width || end > UINTPTR_MAX - reinterpret_cast<UINT_PTR>(dst_wrapper->own().title_memory))
            return refuse("title_footprint_overflow");
        auto host = side_dst;
        host.width = width; host.height = height; host.offset = 0; host.row_pitch = UINT(host_pitch);
        host.resource_bytes = host_pitch * rows;
        const auto valid = x2w_list_copy_texture::check(host, 0, 0, 0, side_src, box ? &copy_box : nullptr, same_family);
        if (valid != x2w_list_copy_texture::Verdict::ok) return refuse(x2w_list_copy_texture::name(valid));
        auto copy = std::make_shared<TitleTextureReadback>();
        copy->dest = static_cast<unsigned char*>(dst_wrapper->own().title_memory) + start;
        copy->width = UINT(row); copy->rows = rows; copy->host_pitch = UINT(host_pitch); copy->title_pitch = side_dst.row_pitch;
        if (!x2w_local::writable(copy->dest, SIZE_T(end-start))) return refuse("title_destination_unwritable");
        const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC buffer{D3D12_RESOURCE_DIMENSION_BUFFER, 0, host.resource_bytes, 1, 1, 1,
                                         DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR};
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        if (FAILED(real_src->GetDevice(IID_PPV_ARGS(&device))) || FAILED(device->CreateCommittedResource(
            &props, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&copy->staging))))
            return refuse("title_readback_allocation");
        auto* list = recording(self);
        {
            std::lock_guard lock(texture_readback_mutex);
            texture_readback_lists[list].push_back(copy);
        }
        AcquireSRWLockExclusive(&draw_lock);
        upload_texture(self, src_wrapper, real_src);
        ReleaseSRWLockExclusive(&draw_lock);
        require_state(self, real_src, D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto to = *dst, from = *src;
        to.pResource = copy->staging.Get();
        to.PlacedFootprint = {0, {format_dst, width, height, 1, UINT(host_pitch)}};
        from.pResource = real_src;
        list->CopyTextureRegion(&to, 0, 0, 0, &from, box);
        retire_hold(self, src_wrapper);
        retire_hold(self, dst_wrapper);
        // Preserve a valid desktop buffer copy as well as the separate CPU backing.
        D3D12_HEAP_TYPE heap{};
        if (verdict != x2w_list_copy_texture::Verdict::ok || (fixed_heap_state(real_dst, heap) && heap == D3D12_HEAP_TYPE_UPLOAD)) {
            InterlockedIncrement(&list_copy_texture_forwarded);
            return;
        }
    }
    if (verdict != x2w_list_copy_texture::Verdict::ok) return refuse(x2w_list_copy_texture::name(verdict));
    // One resource tracks one state, so two subresources of it cannot be COPY_SOURCE and COPY_DEST at once.
    if (real_src == real_dst) return refuse("same_resource");
    D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT;
    const bool src_fixed = fixed_heap_state(real_src, heap);
    if (src_fixed && heap == D3D12_HEAP_TYPE_READBACK) return refuse("bad_heap");
    const bool dst_fixed = fixed_heap_state(real_dst, heap);
    if (dst_fixed && heap == D3D12_HEAP_TYPE_UPLOAD) return refuse("bad_heap");
    // A copied (non-aliased) buffer has no refresh path here, so its desktop bytes may be stale.
    if (side_src.kind == x2w_list_copy_texture::Kind::placed_footprint) {
        const auto* info = resource_info(src_wrapper);
        if (info && info->placement == Placement::copied_buffer) return refuse("stale_source");
    }
    // Bring a texture's desktop copy up to date with its guest backing before the copy reads or partly overwrites it,
    // and record that upload so a later sample does not replace the copied texels.
    // A resource of the retire class that has no title memory has nothing to upload and leaves no texture_uploads()
    // entry behind; every other resource uploads exactly as it did before the switch (switch off: nothing changes).
    const auto needs_upload = [](const Wrapper* wrapper) {
        return !wrapper->own().retire_when_idle || wrapper->own().title_memory;
    };
    if (side_src.kind == x2w_list_copy_texture::Kind::subresource && needs_upload(src_wrapper)) {
        AcquireSRWLockExclusive(&draw_lock);
        upload_texture(self, src_wrapper, real_src);
        ReleaseSRWLockExclusive(&draw_lock);
    }
    if (side_dst.kind == x2w_list_copy_texture::Kind::subresource && needs_upload(dst_wrapper)) {
        AcquireSRWLockExclusive(&draw_lock);
        upload_texture(self, dst_wrapper, real_dst);
        ReleaseSRWLockExclusive(&draw_lock);
    }
    if (!src_fixed) require_state(self, real_src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (!dst_fixed) require_state(self, real_dst, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION to = *dst, from = *src;
    to.pResource = real_dst;
    from.pResource = real_src;
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)->CopyTextureRegion(&to, x, y, z, &from, box ? box : nullptr);
    if (encoded && !compressed_source && bc_bind_trace::enabled()) bc_bind_trace::copied(self, dst_wrapper, real_dst, side_dst.subresource);
    if (side_dst.kind == x2w_list_copy_texture::Kind::subresource) mark_copied_texture(self, dst_wrapper, real_dst);
    retire_hold(self, dst_wrapper);
    retire_hold(self, src_wrapper);
    InterlockedIncrement(&list_copy_texture_forwarded);
}

extern "C" __declspec(dllexport) void x2w_d3d12_list_copy_texture_stats(LONG* forwarded, LONG* refused)
{
    if (forwarded) *forwarded = list_copy_texture_forwarded;
    if (refused) *refused = list_copy_texture_refused;
}

void __stdcall list_copy_texture(Wrapper* self, const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
                                 const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box)
{
    record_texture_copy(16, self, dst, x, y, z, src, box);
}

void __stdcall list_copy_texture_x(Wrapper* self, const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
                                   const D3D12_TEXTURE_COPY_LOCATION* src, const D3D12_BOX* box, UINT)
{
    record_texture_copy(85, self, dst, x, y, z, src, box);
}
