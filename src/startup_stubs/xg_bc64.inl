u64 __stdcall gfx10_add_ref(Computer* self) { return InterlockedIncrement(&self->refs); }

long __stdcall gfx10_layout(Computer* self, unsigned char* out)
{
    if (!out) return E_POINTER;
    std::memset(out, 0, layout_header + 4 * plane_bytes + 8);
    const x2w_startup::Gfx10Texture tex(self->desc);
    put64(out, tex.bytes());
    put64(out + 8, tex.alignment());
    put32(out + 0x10, tex.levels);
    put32(out + 0x14, 1);
    put32(out + 0x1b38, D3D12_RESOURCE_DIMENSION_TEXTURE2D);
    auto* plane = out + layout_header;
    put32(plane, 1);
    put64(plane + 8, tex.bytes());
    put64(plane + 0x10, tex.slice_bytes());
    put64(plane + 0x20, tex.alignment());
    put32(plane + 0x28, tex.bpe);
    put32(plane + 0x2c, tex.first_tail);
    put32(plane + 0x30, tex.levels - tex.first_tail);
    for (UINT level = 0; level < tex.levels; ++level) {
        auto* mip = plane + mip_base + level * mip_bytes;
        const UINT columns = tex.columns(level), rows = tex.rows(level);
        const bool tail = level >= tex.first_tail;
        const UINT padded_columns = tex.padded_columns(level);
        const UINT padded_rows = tex.padded_rows(level);
        const UINT64 span = UINT64{tex.bpe} * padded_columns * padded_rows;
        put64(mip, span);
        put64(mip + 8, tex.start(level));
        put64(mip + 0x10, span);
        put32(mip + 0x18, tex.block_dim * padded_columns);
        put32(mip + 0x1c, tex.bpe * padded_columns);
        put32(mip + 0x20, tail ? 256 : static_cast<UINT>(tex.alignment()));
        put32(mip + 0x24, padded_columns);
        put32(mip + 0x28, padded_rows);
        put32(mip + 0x2c, tex.slices);
        put32(mip + 0x30, columns);
        put32(mip + 0x34, rows);
        put32(mip + 0x38, (std::max)(1u, static_cast<UINT>(self->desc.Width) >> level));
        put32(mip + 0x3c, (std::max)(1u, self->desc.Height >> level));
        put32(mip + 0x40, tex.slices);
        put32(mip + 0x44, 1);
        put32(mip + 0x48, tail ? 1 : tex.swizzle_mode);
        put32(mip + 0x60, 1);
        put32(mip + 0x64, tail);
    }
    return S_OK;
}

template<bool into>
long __stdcall gfx10_copy(Computer* self, unsigned char* target, UINT plane, UINT subresource, const unsigned char* source,
                          UINT pitch, UINT slice_pitch)
{
    const x2w_startup::Gfx10Texture tex(self->desc);
    if (plane || subresource >= tex.levels * tex.slices || !target || !source) return E_INVALIDARG;
    const UINT level = subresource % tex.levels, slice = subresource / tex.levels;
    const u64 row_bytes = u64{tex.bpe} * tex.columns(level);
    const u64 linear_bytes = u64{tex.rows(level) - 1} * pitch + row_bytes;
    if (pitch < row_bytes || (slice_pitch && slice_pitch < linear_bytes)) return E_INVALIDARG;
    const auto dest = reinterpret_cast<std::uintptr_t>(target);
    const auto src = reinterpret_cast<std::uintptr_t>(source);
    const auto dest_bytes = into ? tex.bytes() : linear_bytes;
    const auto src_bytes = into ? linear_bytes : tex.bytes();
    if (dest <= src ? src - dest < dest_bytes : dest - src < src_bytes) return E_INVALIDARG;
    if (!x2w_local::writable(target, dest_bytes) || !x2w_local::readable(source, src_bytes)) return E_FAIL;
    tex.copy_subresource<into>(level, slice, target, source, pitch);
    if constexpr (into) notify_texture_write(target, tex.bytes());
    return S_OK;
}
u64 __stdcall gfx10_size(Computer* self) { return x2w_startup::Gfx10Texture(self->desc).bytes(); }
u64 __stdcall gfx10_alignment(Computer* self) { return x2w_startup::Gfx10Texture(self->desc).alignment(); }
u64 __stdcall gfx10_offset(Computer* self, UINT plane, UINT level)
{
    const x2w_startup::Gfx10Texture tex(self->desc);
    return !plane && level < tex.levels ? tex.start(level) : 0;
}
u64 __stdcall gfx10_element(Computer* self, UINT plane, UINT level, u64 x, UINT y, UINT slice, UINT sample, UINT* bit)
{
    if (bit) *bit = 0;
    const x2w_startup::Gfx10Texture tex(self->desc);
    if (plane || level >= tex.levels || slice >= tex.slices || sample ||
        x >= tex.columns(level) || y >= tex.rows(level)) return 0;
    return tex.address(level, static_cast<UINT>(x), y, slice);
}

long __stdcall gfx10_tiling(Computer* self, UINT* tiles, D3D12_PACKED_MIP_INFO* packed, D3D12_TILE_SHAPE* shape,
                            UINT* count, UINT first, D3D12_SUBRESOURCE_TILING* tilings)
{
    const auto& desc = self->desc;
    const x2w_startup::Gfx10Texture tex(desc);
    const UINT64 slice_tiles = tex.slice_bytes() / tex.tile_bytes;
    if (tiles) *tiles = static_cast<UINT>(slice_tiles * desc.DepthOrArraySize);
    if (packed) *packed = {static_cast<UINT8>(tex.first_tail),
                           static_cast<UINT8>(desc.MipLevels - tex.first_tail),
                           static_cast<UINT>(tex.first_tail < desc.MipLevels ? 1 : 0),
                           0};
    if (shape) *shape = {tex.tile_width * tex.block_dim, tex.tile_height * tex.block_dim, 1};
    if (count) {
        const UINT subresources = UINT{desc.MipLevels} * desc.DepthOrArraySize;
        const UINT available = first < subresources ? subresources - first : 0;
        const UINT requested = (std::min)(*count, available);
        UINT written = 0;
        for (UINT i = 0; i < requested; ++i) {
            const UINT subresource = first + i;
            const UINT level = subresource % desc.MipLevels;
            if (level >= tex.first_tail) continue;
            if (tilings) {
                const UINT start = static_cast<UINT>((subresource / desc.MipLevels) * slice_tiles +
                                                     tex.start(level) / tex.tile_bytes);
                tilings[i] = {tex.tiles_wide(level), static_cast<UINT16>(tex.tiles_high(level)), 1, start};
            }
            ++written;
        }
        *count = written;
    }
    return S_OK;
}

// Backward-compatible aliases for existing bc64 table references
inline u64 __stdcall bc64_add_ref(Computer* self) { return gfx10_add_ref(self); }
inline long __stdcall bc64_layout(Computer* self, unsigned char* out) { return gfx10_layout(self, out); }
template<bool into>
long __stdcall bc64_copy(Computer* self, unsigned char* target, UINT plane, UINT subresource, const unsigned char* source,
                         UINT pitch, UINT slice_pitch)
{
    return gfx10_copy<into>(self, target, plane, subresource, source, pitch, slice_pitch);
}
long __stdcall bc64_copy_from(Computer* self, unsigned char* target, UINT plane, UINT subresource,
                              UINT scarlett_option, const unsigned char* source, UINT pitch, UINT slice_pitch)
{
    // DirectXTex's Scarlett copy-out call passes zero between the subresource and the tiled base.
    // Reject unknown options before interpreting the remaining arguments or touching either buffer.
    if (scarlett_option != 0) return E_NOTIMPL;
    return bc64_copy<false>(self, target, plane, subresource, source, pitch, slice_pitch);
}
inline u64 __stdcall bc64_size(Computer* self) { return gfx10_size(self); }
inline u64 __stdcall bc64_alignment(Computer* self) { return gfx10_alignment(self); }
inline u64 __stdcall bc64_offset(Computer* self, UINT plane, UINT level) { return gfx10_offset(self, plane, level); }
inline u64 __stdcall bc64_element(Computer* self, UINT plane, UINT level, u64 x, UINT y, UINT slice, UINT sample, UINT* bit)
{
    return gfx10_element(self, plane, level, x, y, slice, sample, bit);
}
inline long __stdcall bc64_tiling(Computer* self, UINT* tiles, D3D12_PACKED_MIP_INFO* packed, D3D12_TILE_SHAPE* shape,
                                  UINT* count, UINT first, D3D12_SUBRESOURCE_TILING* tilings)
{
    return gfx10_tiling(self, tiles, packed, shape, count, first, tilings);
}
