// Included by d3d12_xs_device.cpp after d3d12_draw.inl. Command list slot 16, CopyTextureRegion (#870).
// Each location's resource is mapped to its desktop resource, the call is checked by
// list_copy_texture.h, the two resources move to COPY_SOURCE / COPY_DEST for the call (the list's Close
// puts them back to COMMON), and a call the desktop would reject is counted and logged, not forwarded.
#include "list_copy_texture.h"

volatile LONG list_copy_texture_forwarded, list_copy_texture_refused;

namespace {

// One side of the copy, read from the desktop resource it names. False when the location is unusable.
bool copy_texture_side(const D3D12_TEXTURE_COPY_LOCATION* location, ID3D12Resource*& real, x2w_list_copy_texture::Side& side,
                       DXGI_FORMAT& format)
{
    Wrapper* w = location && x2w_local::readable(location, sizeof(*location)) ? as_wrapper(location->pResource) : nullptr;
    if (!w || w->own().kind != resource_kind || !w->own().real) return false;
    real = static_cast<ID3D12Resource*>(w->own().real);
    const D3D12_RESOURCE_DESC desc = real->GetDesc();
    side = {};
    side.block_w = side.block_h = 1;
    side.multisampled = desc.SampleDesc.Count > 1;
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
        format = fp.Footprint.Format;
        return true;
    }
    if (location->Type != D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX || desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) return false;
    const UINT mips = desc.MipLevels ? desc.MipLevels : 1;
    const UINT mip = location->SubresourceIndex % mips;
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

}  // namespace

// List slot 16 (unshifted, like slots 12 to 25), CopyTextureRegion(dst, x, y, z, src, box).
void __stdcall list_copy_texture(Wrapper* self, const D3D12_TEXTURE_COPY_LOCATION* dst, UINT x, UINT y, UINT z,
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
    if (!copy_texture_side(dst, real_dst, side_dst, format_dst) || !copy_texture_side(src, real_src, side_src, format_src))
        return refuse("unknown_resource");
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
    const auto verdict = x2w_list_copy_texture::check(side_dst, x, y, z, side_src, box ? &copy_box : nullptr);
    if (verdict != x2w_list_copy_texture::Verdict::ok) return refuse(x2w_list_copy_texture::name(verdict));
    require_state(self, real_src, D3D12_RESOURCE_STATE_COPY_SOURCE);
    require_state(self, real_dst, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION to = *dst, from = *src;
    to.pResource = real_dst;
    from.pResource = real_src;
    static_cast<ID3D12GraphicsCommandList*>(self->own().real)->CopyTextureRegion(&to, x, y, z, &from, box ? box : nullptr);
    InterlockedIncrement(&list_copy_texture_forwarded);
}

extern "C" __declspec(dllexport) void x2w_d3d12_list_copy_texture_stats(LONG* forwarded, LONG* refused)
{
    if (forwarded) *forwarded = list_copy_texture_forwarded;
    if (refused) *refused = list_copy_texture_refused;
}
