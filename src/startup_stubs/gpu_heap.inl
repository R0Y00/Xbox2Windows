#include "startup_stubs/gpu_material_trace.h"
// Host mirror heaps for title descriptors. The shim supplies resources through Env, which lets WARP fixtures
// exercise the same device code. Include gpu_heap.h at global scope before including this file in the shim's
// anonymous namespace.
//
// Title slot i maps to host slot i. Dynamic slots follow title slots, the null slot is at 2^21 - 1, and raw twins
// start at 2^21. The stride table tracks buffer element sizes. Submission mirrors byte-copied records and queues
// indexed textures for upload.
//
// An X2WV token names a captured CBV (17), SRV (18), UAV (19) or sampler (22). Raw GFX10 V# words describe a placed
// buffer range, and S# words describe samplers. Buffer bounds come from the title allocation. An unrepresentable
// read-only view gets a copy refreshed at submission. The equivalent UAV is null because writes to a copy would be
// lost.
#pragma once

#include "gpu_heap.h"
#include "gpu_view_trace.h"
#include "gpu_srgb.h"
#include "gfx10_view_words.h"
#include "translated_sampler_bindings.h"

// (Its own namespace: in the shim this file is inside the anonymous namespace, where gpu_heap, from the header at
// the top of the file, cannot be reopened.)
namespace gpu_mirror {
using namespace ::gpu_heap;

// What the mirror needs of X2W. The shim fills it from view_records, unwrap, gpu_find/desktop_range and
// upload_texture; fixtures fill it with their own.
struct Record {
    u32 id = 0, slot = 0;          // slot: 17 CBV, 18 SRV, 19 UAV, 22 sampler
    u64 resource = 0, counter = 0;  // title wrappers (0: none)
    bool materialized = true;
    u32 desc_bytes = 0;
    unsigned char desc[64]{};       // as the desktop takes it (formats mapped, as view_slot passes it)
};
struct ResourceFacts {
    ID3D12Resource* desktop = nullptr;  // the desktop resource (no reference held)
    bool buffer = false;
    bool depth_and_uav = false;
    u64 title_memory = 0;               // a buffer: the title address of its first byte
    u64 bytes = 0;                      // a buffer: its width
};
// Shared by all heaps warmed on the same device. A try-lock leaves a concurrently recording
// list free to continue without spending the same frame budget twice.
struct LazyBudget {
    std::mutex mutex;
    u64 frame = ~u64{0}, bytes = 0, room = 0;
    u32 visits = 0;
    double elapsed_ms = 0;
};
struct Env {
    ID3D12Device* device = nullptr;
    u64 (*mapping_epoch)() = nullptr;
    u64 (*copied_bytes)() = nullptr;
    u64 (*local_room)(ID3D12Device*) = nullptr;
    u64 (*allocation_size)(ID3D12Device*, const u32*) = nullptr;
    u64 (*allocated_bytes)() = nullptr;
    void* ctx = nullptr;
    // [address, address + bytes) of title memory on the desktop: the resource and the offset of `address` in it, and
    // the bytes from `address` to the end of the title buffer or contiguous run there.
    // A fixed, coherent window can be bound directly for the lifetime of the descriptor.
    // A CPU shadow or a remappable chunk needs its submit-time refresh.
    bool (*coherent_window)(u64 address, u64 bytes, ID3D12Resource*, u64 offset) = nullptr;
    bool (*find)(void* ctx, u64 address, u64 bytes, ID3D12Resource** resource, u64* offset, u64* left) = nullptr;
    // The view record a token's id names; false if none (or its resource is gone).
    bool (*record)(void* ctx, u32 id, Record* out) = nullptr;
    // A title resource wrapper.
    bool (*resource)(void* ctx, u64 wrapper, ResourceFacts* out) = nullptr;
    std::shared_ptr<x2w_depth_twins::Twins> (*depth_uav)(u64 wrapper) = nullptr;
    // Uploads the texture of title resource `wrapper` (desktop `texture`) on list `list` and leaves it readable by
    // shaders (SRV | non-pixel SRV); false if it could not.
    bool (*upload)(void* ctx, void* list, u64 wrapper, ID3D12Resource* texture) = nullptr;
    // Reads title memory into `out`, false if any of it cannot be read (a page the title unmapped).
    bool (*read)(void* ctx, u64 address, void* out, std::size_t bytes) = nullptr;
    bool (*placement)(u64 wrapper, const D3D12_SHADER_RESOURCE_VIEW_DESC& view, u32 (&words)[8]) = nullptr;
    u64 (*late_texture)(const u32* words, D3D12_SHADER_RESOURCE_VIEW_DESC& view, bool& pending, bool allow_create) = nullptr;
};

// Reads title memory through env.read, or through the kernel (a page the title unmapped fails the read where a read
// in place would fault).
inline bool read_memory(const Env& env, u64 address, void* out, std::size_t bytes)
{
    if (env.read) return env.read(env.ctx, address, out, bytes);
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), out, bytes, &got) && got == bytes;
}

constexpr u32 token_magic = 0x56573258;  // "X2WV" (d3d12_capture.inl view_token_magic)

// Counters (since the start), for the "x2w-gpu heap" lines and the tests.
enum Count : int {
    c_made, c_retried, c_lazy_warmed, c_lazy_staged, c_retained_pending, c_null, c_twins, c_twins_moved, c_twins_unmoved, c_twins_null, c_refused, c_cbv_unaligned,
    c_cross_uav, c_exact, c_exact_failed, c_exact_invalid, c_exact_capacity, c_exact_create, c_exact_map,
    c_exact_unreadable, c_direct_offset, c_vsharp, c_ssharp, c_textures_due, c_uploads, c_mirrored, c_changed,
    c_unfit, c_fit_cut, c_fit_refused, c_count
};
inline std::atomic<u64>* counts()
{
    static std::atomic<u64> c[c_count];
    return c;
}
inline void tally(Count c, u64 n = 1) { counts()[c].fetch_add(n, std::memory_order_relaxed); }
inline u64 counted(Count c) { return counts()[c].load(std::memory_order_relaxed); }

// Decoded title descriptor.
enum class Kind : std::uint8_t { none, constant, shader, unordered, sampler, placed };
struct View {
    Kind kind = Kind::none;
    u64 key = 0;                       // what the slot was made from (0: nothing): token id + 1 with its resource, or
                                       // a hash of the V#/S# words
    u64 wrapper = 0;                   // the title resource (textures: what is uploaded)
    ID3D12Resource* resource = nullptr;  // the desktop resource of a shader/unordered view (no reference held)
    ID3D12Resource* counter = nullptr;
    std::shared_ptr<x2w_depth_twins::Twins> depth_twins;
    bool buffer = false;               // a view of buffer memory
    bool unordered = false;            // placed: an unordered view
    u64 address = 0;                   // constant: its location; buffer: the title address of the buffer's start
    u64 bytes = 0;                     // constant: its size; buffer: the buffer's width (the view's extent)
    std::array<u32, 8> placement{};
    bool pending = false;
    Microsoft::WRL::ComPtr<ID3D12Resource> retained;
    bool explicit_null = false;
    bool has_desc = false;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    D3D12_SAMPLER_DESC sampler{};

    bool is_texture() const { return (kind == Kind::shader || kind == Kind::unordered) && !buffer && resource; }
    // Whether this view addresses a buffer.
    bool is_buffer() const { return kind == Kind::constant || kind == Kind::placed || buffer; }
};

// A buffer view's structure stride, whether it is raw, and its format.
inline std::optional<Elements> view_elements(const View& v)
{
    if (!v.buffer || !v.has_desc) return std::nullopt;
    const bool uav = v.kind == Kind::unordered || (v.kind == Kind::placed && v.unordered);
    if (uav) {
        if (v.uav.ViewDimension != D3D12_UAV_DIMENSION_BUFFER) return std::nullopt;
        return Elements{v.uav.Buffer.StructureByteStride, (v.uav.Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) != 0,
                        static_cast<u32>(v.uav.Format)};
    }
    if (v.srv.ViewDimension != D3D12_SRV_DIMENSION_BUFFER) return std::nullopt;
    return Elements{v.srv.Buffer.StructureByteStride, (v.srv.Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) != 0,
                    static_cast<u32>(v.srv.Format)};
}

// Whether the view has a raw twin when its slot is made.
inline bool twinned(const View& v) { return v.kind == Kind::constant || view_elements(v).has_value(); }

// ---- Reading records -------------------------------------------------------------------------------------------

inline u64 mix_words(const std::uint32_t* w, std::size_t n, u64 seed)
{
    u64 h = seed;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ w[i]) * 0x100000001b3ull;
    return h | 1;  // never 0
}

// The DXGI format of a GFX10 buffer format code with `bytes` per element (the reverse of gfx10_format), 0 if none.
inline u32 dxgi_of_gfx10(u32 code, u32 bytes)
{
    static const u32* table = [] {
        static u32 t[128]{};
        static u32 b[128]{};
        for (u32 f = 1; f < 130; ++f) {
            const auto g = x2w_startup::gfx10_format(static_cast<DXGI_FORMAT>(f));
            if (g.code && g.code < 128 && g.bytes && !t[g.code]) {
                t[g.code] = f;
                b[g.code] = g.bytes;
            }
        }
        return t;
    }();
    if (code >= 128 || !table[code]) return 0;
    return element_bytes(table[code]) == bytes ? table[code] : 0;
}

// A V# (GFX10 buffer resource words) as a placed buffer SRV: raw when it has no stride, structured otherwise (a
// typed one with a format the host has: typed).
inline bool view_of_vsharp(const std::uint32_t w[4], View& v)
{
    const u64 base = w[0] | (u64{w[1] & 0xffff} << 32);
    const u32 stride = (w[1] >> 16) & 0x3fff;
    const u32 records = w[2];
    const u32 format = (w[3] >> 12) & 127;
    const u32 oob = (w[3] >> 28) & 3;
    const u32 type = w[3] >> 30;
    if (!base || !records || type != 0 || (w[3] & 0xfff) == 0) return false;
    if (stride == 0 && oob != 3) return false;
    v = View{};
    v.kind = Kind::placed;
    v.buffer = true;
    v.has_desc = true;
    v.address = base;
    v.srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    v.srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (stride == 0) {
        v.srv.Format = DXGI_FORMAT_R32_TYPELESS;
        v.srv.Buffer.NumElements = records / 4;
        v.srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        v.bytes = records & ~3u;
        if (!v.srv.Buffer.NumElements) return false;
    } else {
        const u32 typed = dxgi_of_gfx10(format, stride);
        // (A typed V# of a format of `stride` bytes reads as that format; any other strided one is structured.)
        if (typed && format != 22 /* 32_FLOAT, the raw/structured descriptors' */) {
            v.srv.Format = static_cast<DXGI_FORMAT>(typed);
        } else {
            if (stride % 4 != 0) return false;
            v.srv.Format = DXGI_FORMAT_UNKNOWN;
            v.srv.Buffer.StructureByteStride = stride;
        }
        v.srv.Buffer.NumElements = records;
        v.bytes = u64{records} * stride;
    }
    v.key = mix_words(w, 4, 0x56ull);
    tally(c_vsharp);
    return true;
}

inline bool title_as_address(const Env& env, const std::uint32_t words[8], u64* address)
{
    if (words[0] != token_magic || words[2] != 18) return false;
    Record record{};
    if (!env.record || !env.record(env.ctx, words[1], &record) || record.slot != 18 ||
        record.desc_bytes < sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
    std::memcpy(&desc, record.desc, sizeof(desc));
    if (desc.ViewDimension != D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE) return false;
    *address = x2w_as_address::address(words, desc.RaytracingAccelerationStructure.Location);
    return true;
}

// The view a slot's record names, from `words`, the record's first 8 dwords. `sampler_heap`: a sampler heap's
// record (S# words where it is no token).
inline bool view_of_record(const Env& env, const std::uint32_t words[8], bool sampler_heap, View& v, bool allow_create = true)
{
    v = View{};
    if (!sampler_heap && x2w_late_texture::marker(words[0]) && env.late_texture) {
        D3D12_RESOURCE_DESC desc{};
        if (!x2w_late_texture::decode(words, desc, v.srv)) return false;
        v.kind = Kind::shader;
        v.has_desc = true;
        v.key = mix_words(words, 8, 0x54ull);
        std::copy_n(words, 8, v.placement.begin());
        v.placement[0] = x2w_late_texture::magic;
        v.wrapper = env.late_texture(words, v.srv, v.pending, allow_create);
        ResourceFacts f{};
        if (v.wrapper && env.resource && env.resource(env.ctx, v.wrapper, &f)) {
            v.resource = f.desktop;
            u32 fields[sizeof(v.srv) / sizeof(u32)];
            std::memcpy(fields, &v.srv, sizeof(fields));
            v.key = mix_words(fields, std::size(fields), v.key);
        }
        return true;
    }
    if (words[0] == token_magic) {
        Record r;
        if (!env.record || !env.record(env.ctx, words[1], &r)) return false;
        v.key = (u64{r.id} + 1) ^ (r.resource * 0x9e3779b97f4a7c15ull);
        if (!v.key) v.key = 1;
        v.wrapper = r.resource;
        switch (r.slot) {
        case 17: {
            if (r.desc_bytes < sizeof(D3D12_CONSTANT_BUFFER_VIEW_DESC)) return false;
            D3D12_CONSTANT_BUFFER_VIEW_DESC c;
            std::memcpy(&c, r.desc, sizeof(c));
            v.kind = Kind::constant;
            v.address = c.BufferLocation;
            v.bytes = c.SizeInBytes;
            return v.address != 0;
        }
        case 18:
        case 19: {
            const bool uav = r.slot == 19;
            if (uav && r.desc_bytes >= sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC)) {
                std::memcpy(&v.uav, r.desc, sizeof(v.uav));
                v.has_desc = true;
            } else if (!uav && r.desc_bytes >= sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC)) {
                std::memcpy(&v.srv, r.desc, sizeof(v.srv));
                v.has_desc = true;
            }

            v.kind = uav ? Kind::unordered : Kind::shader;
            if (!r.resource) {
                if (!r.materialized || !v.has_desc || r.counter) return false;
                if (!uav && v.srv.ViewDimension == D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE) return false;
                v.buffer = uav ? v.uav.ViewDimension == D3D12_UAV_DIMENSION_BUFFER : v.srv.ViewDimension == D3D12_SRV_DIMENSION_BUFFER;
                v.explicit_null = true;
                return true;
            }
            ResourceFacts f;
            if (!env.resource || !env.resource(env.ctx, r.resource, &f) || !f.desktop) {
                return false;
            }
            v.resource = f.desktop;
            v.buffer = f.buffer;
            v.address = f.title_memory;
            v.bytes = f.bytes;
            if (uav && r.counter && env.resource) {
                ResourceFacts c;
                if (env.resource(env.ctx, r.counter, &c)) v.counter = c.desktop;
            }
            if (uav && f.depth_and_uav && env.depth_uav) {
                auto twins = env.depth_uav(r.resource);
                D3D12_UNORDERED_ACCESS_VIEW_DESC adapted{};
                auto* backing = twins ? twins->view(v.has_desc ? &v.uav : nullptr, adapted) : nullptr;
                // A packed depth plane needs a graphics pass to return its values. Expose only
                // twins whose writes can already be carried back on this path.
                if (backing && backing == twins->colour.Get()) {
                    v.resource = backing;
                    v.uav = adapted;
                    v.has_desc = true;
                    v.depth_twins = std::move(twins);
                }
            }
            if (!uav && !v.buffer && v.has_desc && env.placement) {
                u32 words[8]{};
                if (env.placement(r.resource, v.srv, words)) std::copy_n(words, 8, v.placement.begin());
            }
            // A buffer view of a buffer object needs its description.
            return !v.buffer || v.has_desc;
        }
        case 22:
            if (r.desc_bytes < sizeof(D3D12_SAMPLER_DESC)) return false;
            std::memcpy(&v.sampler, r.desc, sizeof(v.sampler));
            v.kind = Kind::sampler;
            return true;
        default:
            return false;
        }
    }
    if (PlacedRecord p; placed_of(words, p)) {
        v.kind = Kind::placed;
        v.buffer = true;
        v.unordered = p.unordered;
        v.has_desc = true;
        v.address = p.address;
        v.bytes = p.width;
        if (p.unordered) {
            v.uav.Format = static_cast<DXGI_FORMAT>(p.format);
            v.uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            v.uav.Buffer.FirstElement = p.first;
            v.uav.Buffer.NumElements = p.elements;
            v.uav.Buffer.StructureByteStride = p.stride;
            v.uav.Buffer.Flags = p.raw ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE;
        } else {
            v.srv.Format = static_cast<DXGI_FORMAT>(p.format);
            v.srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            v.srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            v.srv.Buffer.FirstElement = p.first;
            v.srv.Buffer.NumElements = p.elements;
            v.srv.Buffer.StructureByteStride = p.stride;
            v.srv.Buffer.Flags = p.raw ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
        }
        v.key = mix_words(words, 8, 0x50ull);
        return true;
    }
    if (sampler_heap) {
        // The title's own S# words (translated_sampler_bindings.h decode_sampler).
        const bool any = words[0] | words[1] | words[2] | words[3];
        if (!any) return false;
        x2w_startup::SamplerWords s{words[0], words[1], words[2], words[3]};
        const D3D12_STATIC_SAMPLER_DESC d = x2w_startup::decode_sampler(s, 0, false, true);
        v.kind = Kind::sampler;
        v.sampler.Filter = d.Filter;
        v.sampler.AddressU = d.AddressU;
        v.sampler.AddressV = d.AddressV;
        v.sampler.AddressW = d.AddressW;
        v.sampler.MipLODBias = d.MipLODBias;
        v.sampler.MaxAnisotropy = d.MaxAnisotropy;
        v.sampler.ComparisonFunc = d.ComparisonFunc;
        const float border = d.BorderColor == D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK ? 0.0f : d.BorderColor == D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE ? 1.0f : 0.0f;
        v.sampler.BorderColor[0] = v.sampler.BorderColor[1] = v.sampler.BorderColor[2] = border;
        v.sampler.BorderColor[3] = d.BorderColor == D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK ? 0.0f : 1.0f;
        v.sampler.MinLOD = d.MinLOD;
        v.sampler.MaxLOD = d.MaxLOD;
        v.key = mix_words(words, 4, 0x55ull);
        tally(c_ssharp);
        return true;
    }
    return view_of_vsharp(words, v);
}

// ---- Null views ---------

inline void null_view(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Texture2D.MipLevels = 1;
    gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(nullptr, &d, at);
}
inline void null_buffer_view(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_R32_TYPELESS;
    d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.NumElements = 1;
    d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(nullptr, &d, at);
}
// A texture view that reads zeros with the title's dimension (one mip, one slice, plane 0).
inline void null_like(ID3D12Device* device, const D3D12_SHADER_RESOURCE_VIEW_DESC& title, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    const u32 dim = static_cast<u32>(title.ViewDimension);
    if (dim < 2 || dim > 10) return null_view(device, at);
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.ViewDimension = title.ViewDimension;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // Words 5 (and 7) of the union are MipLevels (and ArraySize / NumCubes).
    std::uint32_t w[10]{};
    w[1] = dim;
    switch (dim) {
    case 6: break;                         // 2DMS
    case 7: w[5] = 1; break;               // 2DMS array: ArraySize at word 5
    case 3: case 5: case 10: w[5] = 1; w[7] = 1; break;  // 1D array, 2D array, cube array
    default: w[5] = 1; break;
    }
    std::memcpy(reinterpret_cast<unsigned char*>(&d) + 16, &w[4], 24);
    gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(nullptr, &d, at);
}
inline D3D12_UNORDERED_ACCESS_VIEW_DESC null_uav_desc(std::optional<u32> dimension)
{
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format = DXGI_FORMAT_R32_UINT;
    d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    if (!dimension) return d;
    switch (*dimension) {
    case D3D12_UAV_DIMENSION_TEXTURE1D: d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D; break;
    case D3D12_UAV_DIMENSION_TEXTURE1DARRAY:
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
        d.Texture1DArray.ArraySize = 1;
        break;
    case D3D12_UAV_DIMENSION_TEXTURE2DARRAY:
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        d.Texture2DArray.ArraySize = 1;
        break;
    case D3D12_UAV_DIMENSION_TEXTURE3D:
        d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
        d.Texture3D.WSize = ~0u;
        break;
    default: break;
    }
    return d;
}
// What a slot of `view` reads when the host cannot make it.
inline void null_for(ID3D12Device* device, const View& v, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    bool buffer = false;
    std::optional<u32> dimension;
    switch (v.kind) {
    case Kind::unordered:
        buffer = v.buffer || (v.has_desc && v.uav.ViewDimension == D3D12_UAV_DIMENSION_BUFFER);
        if (v.has_desc) dimension = static_cast<u32>(v.uav.ViewDimension);
        break;
    case Kind::placed:
        if (!v.unordered) return null_buffer_view(device, at);
        buffer = true;
        break;
    case Kind::shader:
        if (v.buffer || (v.has_desc && v.srv.ViewDimension == D3D12_SRV_DIMENSION_BUFFER)) return null_buffer_view(device, at);
        if (v.has_desc) return null_like(device, v.srv, at);
        return null_view(device, at);
    case Kind::constant: {
        D3D12_CONSTANT_BUFFER_VIEW_DESC c{};
        return gpu_view_trace::MirrorCall{device, __FUNCTION__}.cbv(&c, at);
    }
    default:
        return null_view(device, at);
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    if (buffer) {
        d.Format = DXGI_FORMAT_R32_UINT;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    } else {
        d = null_uav_desc(dimension);
    }
    gpu_view_trace::MirrorCall{device, __FUNCTION__}.uav(nullptr, nullptr, &d, at);
}
inline void null_sampler(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE at)
{
    D3D12_SAMPLER_DESC d{};
    d.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    d.AddressU = d.AddressV = d.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    d.MaxAnisotropy = 1;
    d.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    gpu_view_trace::MirrorCall{device, __FUNCTION__}.sampler(&d, at);
}

// ---- Copies of a view's own bytes -------------------------------------------

// A buffer of `bytes` of title memory from `start`, in upload memory the GPU reads, refreshed from title memory at
// every submit (Exacts::refresh). Kept while a slot shows it, then for a few submits.
struct Exact {
    u64 start = 0, bytes = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
    void* data = nullptr;
    u64 used = 0;                      // the submit it was last made for
};
class Exacts {
public:
    // The copy of [start, start + bytes), made if new and filled now. Null if it cannot be.
    Microsoft::WRL::ComPtr<ID3D12Resource> get(const Env& env, u64 start, u64 bytes, u64 submit)
    {
        constexpr u64 most = 64ull << 20;  // bytes of copies in all
        std::lock_guard<std::mutex> lock(mutex_);
        const auto key = std::make_pair(start, bytes);
        auto found = by_key_.find(key);
        if (found != by_key_.end()) {
            found->second.used = submit;
            return found->second.buffer;
        }
        if (!start || !bytes || bytes > most || start > UINT64_MAX - bytes) {
            tally(c_exact_invalid);
            return nullptr;
        }
        const u64 width = (bytes + 255) & ~u64{255};
        if (width > most - total_) {
            tally(c_exact_capacity);
            return nullptr;
        }
        D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = width;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;
        if (FAILED(env.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                       nullptr, IID_PPV_ARGS(&buffer)))) {
            tally(c_exact_create);
            return nullptr;
        }
        void* data = nullptr;
        const D3D12_RANGE none{0, 0};
        if (FAILED(buffer->Map(0, &none, &data))) {
            tally(c_exact_map);
            return nullptr;
        }
        if (!read_memory(env, start, data, bytes)) {
            tally(c_exact_unreadable);
            return nullptr;
        }
        total_ += width;
        Exact& e = by_key_[key];
        e = Exact{start, bytes, buffer, data, submit};
        return buffer;
    }
    // Every copy gets title memory's bytes again; those no slot has shown for `keep` submits go. A slot that shows a
    // copy (and the retired list, for a few submits after) holds a reference to its buffer: a copy whose only
    // reference is ours is shown by no slot.
    void refresh(const Env& env, u64 submit, u64 keep)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = by_key_.begin(); it != by_key_.end();) {
            Exact& e = it->second;
            e.buffer->AddRef();
            if (e.buffer->Release() > 1) e.used = submit;
            if (submit > e.used + keep) {
                total_ -= (e.bytes + 255) & ~u64{255};
                it = by_key_.erase(it);
                continue;
            }
            read_memory(env, e.start, e.data, e.bytes);
            ++it;
        }
    }
    std::size_t size()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return by_key_.size();
    }

private:
    std::mutex mutex_;
    std::map<std::pair<u64, u64>, Exact> by_key_;
    u64 total_ = 0;
};

// ---- Making views -----------------------------------------------------

// What a made view shows of the host's: the resource the slot holds (null: none, the view needs nothing kept).
struct Made {
    bool ok = false;
    ID3D12Resource* kept = nullptr;  // no reference: the caller takes one
    u64 raw_start{}, raw_bytes{};
    Microsoft::WRL::ComPtr<ID3D12Resource> pin; // keeps an exact copy alive until the slot owns it
};

// Where a buffer view (title memory from `base`, `width` bytes of buffer) is on the desktop: the resource, the
// offset of `base` in it, and the width of what may be viewed there from the resource's start.
struct Home {
    ID3D12Resource* resource = nullptr;
    u64 offset = 0;   // of `base`
    u64 width = 0;    // bytes from the resource start that are this buffer's
};

// Moves the raw view at `at` (a CPU-only slot) `by` bytes on, where AMD's descriptor is its address (48 bits) and
// bytes; false if the descriptor is not that.
inline bool move_raw(D3D12_CPU_DESCRIPTOR_HANDLE at, u64 start, u32 bytes, u32 by)
{
    if (by >= bytes) return false;
    std::uint32_t words[4];
    SIZE_T got = 0;
    if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at.ptr), words, sizeof(words), &got) ||
        got != sizeof(words))
        return false;
    const u64 address = words[0] | (u64{words[1]} << 32);
    if (address != start || words[2] != bytes || (start >> 48) != 0) return false;
    const u64 moved = start + by;
    auto* w = reinterpret_cast<std::uint32_t*>(at.ptr);
    w[0] = static_cast<std::uint32_t>(moved);
    w[1] = static_cast<std::uint32_t>(moved >> 32);
    w[2] = bytes - by;
    return true;
}

// The bytes per element of a buffer SRV description.
inline u64 element_size(const D3D12_SHADER_RESOURCE_VIEW_DESC& d)
{
    if (d.Buffer.StructureByteStride > 0) return d.Buffer.StructureByteStride;
    if (d.Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) return 4;
    return element_bytes(static_cast<u32>(d.Format));
}
inline u64 element_size(const D3D12_UNORDERED_ACCESS_VIEW_DESC& d)
{
    if (d.Buffer.StructureByteStride > 0) return d.Buffer.StructureByteStride;
    if (d.Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) return 4;
    return element_bytes(static_cast<u32>(d.Format));
}

inline void (*&fit_log())(const char*)
{
    static void (*sink)(const char*) = nullptr;
    return sink;
}
inline void fit_note(const char* site, const char* what, ID3D12Resource* r, u64 a, u64 b, u64 c)
{
    static std::atomic<int> lines{0};
    if (!fit_log() || lines.fetch_add(1) >= 24) return;
    char text[400];
    const D3D12_RESOURCE_DESC d = r ? r->GetDesc() : D3D12_RESOURCE_DESC{};
    std::snprintf(text, sizeof(text), "x2w-gpu heap fit: %s: %s (resource %p dim %d width %llu height %u depth %u mips %u; "
                  "%llu %llu %llu)", site, what, static_cast<void*>(r), static_cast<int>(d.Dimension),
                  static_cast<unsigned long long>(d.Width), d.Height, d.DepthOrArraySize, d.MipLevels,
                  static_cast<unsigned long long>(a), static_cast<unsigned long long>(b), static_cast<unsigned long long>(c));
    fit_log()(text);
}
inline bool fit_refuse(const char* site, const char* what, ID3D12Resource* r, u64 a = 0, u64 b = 0, u64 c = 0)
{
    tally(c_fit_refused);
    fit_note(site, what, r, a, b, c);
    return false;
}
// A buffer view (SRV or UAV description, ViewDimension BUFFER) on `r`: cut at its end; false when none of it is there.
template <class Desc>
bool fit_buffer(ID3D12Resource* r, Desc& d, const char* site)
{
    if (!r) return true;
    const D3D12_RESOURCE_DESC rd = r->GetDesc();
    if (rd.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) return fit_refuse(site, "buffer view of a texture", r);
    const u64 e = element_size(d);
    if (!e) return fit_refuse(site, "buffer view without an element size (format)", r, static_cast<u64>(d.Format));
    const u64 total = rd.Width / e;
    const u64 first = d.Buffer.FirstElement, n = d.Buffer.NumElements;
    if (!n || first >= total) return fit_refuse(site, "buffer view past the buffer (first, count, element bytes)", r, first, n, e);
    if (first + n > total) {
        d.Buffer.NumElements = static_cast<UINT>(total - first);
        tally(c_fit_cut);
        fit_note(site, "buffer view cut at the buffer's end (first, count, element bytes)", r, first, n, e);
    }
    return true;
}
inline bool fit_cbv(const D3D12_CONSTANT_BUFFER_VIEW_DESC& c, const char* site)
{
    if (!c.BufferLocation && !c.SizeInBytes) return true;
    if (c.BufferLocation % 256 || c.SizeInBytes % 256 || c.SizeInBytes > (64u << 10) || !c.SizeInBytes)
        return fit_refuse(site, "constant view (location, size)", nullptr, c.BufferLocation, c.SizeInBytes);
    return true;
}
// Mips [most, most + count) of `mips` (count ~0: all from most): the count cut; false when `most` is past them.
inline bool fit_mips(UINT most, UINT& count, UINT mips)
{
    if (most >= mips) return false;
    if (count != ~0u && (count == 0 || most + u64{count} > mips)) count = mips - most;
    return true;
}
inline bool fit_slices(UINT first, UINT& size, UINT slices)
{
    if (first >= slices) return false;
    if (size == 0 || first + u64{size} > slices) size = slices - first;
    return true;
}
inline bool fit_srv(ID3D12Resource* r, D3D12_SHADER_RESOURCE_VIEW_DESC* d, const char* site)
{
    if (!r || !d) return true;
    if (d->ViewDimension == D3D12_SRV_DIMENSION_BUFFER) return fit_buffer(r, *d, site);
    const D3D12_RESOURCE_DESC rd = r->GetDesc();
    const UINT mips = rd.MipLevels ? rd.MipLevels : 1;
    const UINT slices = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : rd.DepthOrArraySize;
    const auto want = [&](D3D12_RESOURCE_DIMENSION dim) { return rd.Dimension == dim; };
    const auto t1 = D3D12_RESOURCE_DIMENSION_TEXTURE1D, t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D, t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    bool ok = true;
    switch (d->ViewDimension) {
    case D3D12_SRV_DIMENSION_TEXTURE1D: ok = want(t1) && fit_mips(d->Texture1D.MostDetailedMip, d->Texture1D.MipLevels, mips); break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
        ok = want(t1) && fit_mips(d->Texture1DArray.MostDetailedMip, d->Texture1DArray.MipLevels, mips) &&
             fit_slices(d->Texture1DArray.FirstArraySlice, d->Texture1DArray.ArraySize, slices);
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2D: ok = want(t2) && fit_mips(d->Texture2D.MostDetailedMip, d->Texture2D.MipLevels, mips); break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
        ok = want(t2) && fit_mips(d->Texture2DArray.MostDetailedMip, d->Texture2DArray.MipLevels, mips) &&
             fit_slices(d->Texture2DArray.FirstArraySlice, d->Texture2DArray.ArraySize, slices);
        break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMS: ok = want(t2); break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
        ok = want(t2) && fit_slices(d->Texture2DMSArray.FirstArraySlice, d->Texture2DMSArray.ArraySize, slices);
        break;
    case D3D12_SRV_DIMENSION_TEXTURE3D: ok = want(t3) && fit_mips(d->Texture3D.MostDetailedMip, d->Texture3D.MipLevels, mips); break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
        ok = want(t2) && slices >= 6 && fit_mips(d->TextureCube.MostDetailedMip, d->TextureCube.MipLevels, mips);
        break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY: {
        auto& c = d->TextureCubeArray;
        ok = want(t2) && fit_mips(c.MostDetailedMip, c.MipLevels, mips) && c.First2DArrayFace + 6ull <= slices;
        if (ok && (c.NumCubes == 0 || c.First2DArrayFace + u64{c.NumCubes} * 6 > slices)) c.NumCubes = (slices - c.First2DArrayFace) / 6;
        break;
    }
    default: return true;  // raytracing: not made by the mirror
    }
    if (ok) return true;
    return fit_refuse(site, "texture SRV the texture cannot take (view dimension, words 4 and 6 of the description)", r,
                      static_cast<u64>(d->ViewDimension), d->Texture2DArray.MostDetailedMip, d->Texture2DArray.FirstArraySlice);
}
inline bool fit_uav(ID3D12Resource* r, D3D12_UNORDERED_ACCESS_VIEW_DESC* d, const char* site)
{
    if (!r || !d) return true;
    if (d->ViewDimension == D3D12_UAV_DIMENSION_BUFFER) return fit_buffer(r, *d, site);
    const D3D12_RESOURCE_DESC rd = r->GetDesc();
    const UINT mips = rd.MipLevels ? rd.MipLevels : 1;
    const UINT slices = rd.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D ? 1u : rd.DepthOrArraySize;
    const auto want = [&](D3D12_RESOURCE_DIMENSION dim) { return rd.Dimension == dim; };
    const auto t1 = D3D12_RESOURCE_DIMENSION_TEXTURE1D, t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D, t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    bool ok = true;
    switch (d->ViewDimension) {
    case D3D12_UAV_DIMENSION_TEXTURE1D: ok = want(t1) && d->Texture1D.MipSlice < mips; break;
    case D3D12_UAV_DIMENSION_TEXTURE1DARRAY:
        ok = want(t1) && d->Texture1DArray.MipSlice < mips && fit_slices(d->Texture1DArray.FirstArraySlice, d->Texture1DArray.ArraySize, slices);
        break;
    case D3D12_UAV_DIMENSION_TEXTURE2D: ok = want(t2) && d->Texture2D.MipSlice < mips; break;
    case D3D12_UAV_DIMENSION_TEXTURE2DARRAY:
        ok = want(t2) && d->Texture2DArray.MipSlice < mips && fit_slices(d->Texture2DArray.FirstArraySlice, d->Texture2DArray.ArraySize, slices);
        break;
    case D3D12_UAV_DIMENSION_TEXTURE3D: {
        auto& w = d->Texture3D;
        ok = want(t3) && w.MipSlice < mips;
        const UINT depth = ok ? (std::max)(1u, static_cast<UINT>(rd.DepthOrArraySize) >> w.MipSlice) : 0u;
        ok = ok && w.FirstWSlice < depth;
        if (ok && w.WSize != ~0u && (w.WSize == 0 || w.FirstWSlice + u64{w.WSize} > depth)) w.WSize = depth - w.FirstWSlice;
        break;
    }
    default: return true;
    }
    if (ok) return true;
    return fit_refuse(site, "texture UAV the texture cannot take (view dimension, mip, first slice)", r,
                      static_cast<u64>(d->ViewDimension), d->Texture2DArray.MipSlice, d->Texture2DArray.FirstArraySlice);
}

// The raw twin of a buffer view made at `at` on `resource`: the same bytes from the 16 below, moved on where the host
// lets it (staged slots only), else a null buffer view.
template <class Desc>
bool make_twin(ID3D12Device* device, ID3D12Resource* resource, Desc desc, u64 first, u32 n, u32 stride, bool raw,
               u64 width, D3D12_CPU_DESCRIPTOR_HANDLE twin, bool staged)
{
    const auto w = twin_window(static_cast<u32>(desc.Format), first, n, stride, raw, width);
    if (!w) {
        null_buffer_view(device, twin);
        tally(c_twins_null);
        return false;
    }
    desc.Format = DXGI_FORMAT_R32_TYPELESS;
    desc.Buffer.FirstElement = w->first;
    desc.Buffer.NumElements = w->dwords;
    desc.Buffer.StructureByteStride = 0;
    if constexpr (std::is_same_v<Desc, D3D12_UNORDERED_ACCESS_VIEW_DESC>) {
        desc.Buffer.CounterOffsetInBytes = 0;
        desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        if (!fit_buffer(resource, desc, "twin")) {
            null_buffer_view(device, twin);
            tally(c_twins_null);
            return false;
        }
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.uav(resource, nullptr, &desc, twin);
    } else {
        desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (!fit_buffer(resource, desc, "twin")) {
            null_buffer_view(device, twin);
            tally(c_twins_null);
            return false;
        }
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(resource, &desc, twin);
    }
    tally(c_twins);
    if (w->by == 0) return true;
    // Only a staged slot of ours, which the card does not see, is moved.
    const u64 start = resource->GetGPUVirtualAddress() + w->first * 4;
    if (staged && move_raw(twin, start, w->dwords * 4u, w->by)) {
        tally(c_twins_moved);
        return true;
    }
    tally(c_twins_unmoved);
    tally(c_twins_null);
    null_buffer_view(device, twin);
    return false;
}

// Use the same tested raw descriptor layout as a moved twin, but only in a private CPU
// staging slot. The read stays on the authoritative GPU resource when the driver accepts it.
inline bool offset_raw_srv(ID3D12Device* device, const Home& home, u64 offset, u64 bytes,
                           D3D12_CPU_DESCRIPTOR_HANDLE at, std::optional<D3D12_CPU_DESCRIPTOR_HANDLE> twin)
{
    const u64 aligned = offset & ~u64{15};
    const UINT delta = UINT(offset - aligned);
    const auto resource = home.resource->GetDesc();
    const u64 limit = (std::min<u64>)(resource.Width, home.width);
    if (!delta || ((offset | bytes) & 3) || !bytes || bytes > UINT32_MAX - delta ||
        resource.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER || offset > limit || bytes > limit - offset)
        return false;
    const UINT span = static_cast<UINT>(bytes) + delta;
    D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw.Buffer.FirstElement = aligned / 4;
    raw.Buffer.NumElements = span / 4;
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(home.resource, &raw, at);
    if (!move_raw(at, home.resource->GetGPUVirtualAddress() + aligned, span, delta)) return false;
    if (twin) {
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.copy(1, *twin, at, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tally(c_twins);
        tally(c_twins_moved);
    }
    tally(c_direct_offset);
    return true;
}

// Create a buffer SRV or UAV on the window containing the title range. Base and width describe the buffer or placed
// memory; desc describes the title view.
template <class Desc>
Made buffer_view(const Env& env, Exacts& exacts, u64 submit, u64 base, u64 width, Desc desc, ID3D12Resource* own,
                 D3D12_CPU_DESCRIPTOR_HANDLE at, std::optional<D3D12_CPU_DESCRIPTOR_HANDLE> twin, bool staged, bool primary_staged)
{
    constexpr bool uav = std::is_same_v<Desc, D3D12_UNORDERED_ACCESS_VIEW_DESC>;
    ID3D12Device* const device = env.device;
    const u64 unit = element_size(desc);
    if (!unit || desc.Buffer.FirstElement > (UINT64_MAX - base) / unit) return {};
    const u64 start = base + desc.Buffer.FirstElement * unit;
    const u64 len = u64{desc.Buffer.NumElements} * unit;
    // The window of the whole buffer, else of the view's own range, else the buffer object's own desktop resource.
    Home home;
    {
        ID3D12Resource* r = nullptr;
        u64 off = 0, left = 0;
        if (env.find && base && width && env.find(env.ctx, base, width, &r, &off, &left)) {
            home = {r, off, off + (std::min)(width, left)};
        } else if (env.find && start && len && env.find(env.ctx, start, len, &r, &off, &left)) {
            home = {r, off, off + (std::min)(len, left)};
            desc.Buffer.FirstElement = 0;
            base = start;
        } else if (own) {
            home = {own, 0, (std::min)(width, own->GetDesc().Width)};
        }
    }
    if (!home.resource || desc.Buffer.FirstElement > (UINT64_MAX - home.offset) / unit) return {};
    Microsoft::WRL::ComPtr<ID3D12Resource> pin;
    bool raw_flag;
    if constexpr (uav) raw_flag = (desc.Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) != 0;
    else raw_flag = (desc.Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) != 0;
    if (off_elements(home.offset + desc.Buffer.FirstElement * unit, desc.Buffer.StructureByteStride, raw_flag,
                     element_bytes(static_cast<u32>(desc.Format)))) {
        if constexpr (uav) {
            // What a shader writes through a copy is lost: null, counted (PLAN.md 1.3).
            tally(c_cross_uav);
            return {};
        } else {
            const u64 offset = home.offset + desc.Buffer.FirstElement * unit;
            if (primary_staged && (raw_flag || (desc.Format == DXGI_FORMAT_UNKNOWN && desc.Buffer.StructureByteStride)) &&
                env.coherent_window && env.coherent_window(start, len, home.resource, offset) &&
                offset_raw_srv(device, home, offset, len, at, twin))
                return {true, home.resource, twin ? offset : 0, twin ? len : 0};
            pin = exacts.get(env, start, len, submit);
            if (!pin) {
                tally(c_exact_failed);
                return {};
            }
            tally(c_exact);
            home = {pin.Get(), 0, len};
            desc.Buffer.FirstElement = 0;
        }
    }
    const u64 offset = home.offset;
    // A structured view as a raw view of the same bytes.
    std::optional<RawWindow> window;
    if (desc.Format == DXGI_FORMAT_UNKNOWN)
        window = raw_window(offset, desc.Buffer.FirstElement, desc.Buffer.NumElements, desc.Buffer.StructureByteStride);
    if (window) {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Buffer.FirstElement = window->first;
        desc.Buffer.NumElements = window->dwords;
        desc.Buffer.StructureByteStride = 0;
        if constexpr (uav) desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        else desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    } else {
        const u64 element = element_size(desc);
        if (element == 0 || offset % element != 0) {
            tally(c_refused);
            return {};
        }
        desc.Buffer.FirstElement += offset / element;
    }
    if constexpr (uav) desc.Buffer.CounterOffsetInBytes = 0;
    const u64 first = desc.Buffer.FirstElement;
    const u32 n = desc.Buffer.NumElements, stride = desc.Buffer.StructureByteStride;
    bool raw;
    if constexpr (uav) raw = (desc.Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW) != 0;
    else raw = (desc.Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW) != 0;
    const u64 view_width = home.width;
    if constexpr (uav) {
        if (!(home.resource->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) {
            tally(c_refused);
            return {};
        }
    }
    if (!buffer_view_ok(static_cast<u32>(desc.Format), first, n, stride, raw, view_width)) {
        tally(c_refused);
        return {};
    }
    if (!fit_buffer(home.resource, desc, uav ? "buffer UAV" : "buffer SRV")) return {};
    if constexpr (uav) {
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.uav(home.resource, nullptr, &desc, at);
    } else {
        // The console has no component mapping.
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(home.resource, &desc, at);
    }
    Made made{true, home.resource};
    made.pin = std::move(pin);
    if (twin && make_twin(device, home.resource, desc, first, n, stride, raw, view_width, *twin, staged) && !uav) {
        const auto w = twin_window(static_cast<u32>(desc.Format), first, n, stride, raw, view_width);
        made.raw_start = w->first * 4 + w->by;
        made.raw_bytes = u64{w->dwords} * 4 - w->by;
    }
    return made;
}

// Whether the host can make a shader (or unordered) view of `texture` in `format` (0: its own); a view of a typed
// texture in another format of its family takes the texture's.
inline bool fits(ID3D12Resource* texture, DXGI_FORMAT* format, bool unordered)
{
    const D3D12_RESOURCE_DESC d = texture->GetDesc();
    const bool allowed = unordered ? (d.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) != 0
                                   : (d.Flags & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE) == 0;
    const u32 own = static_cast<u32>(d.Format);
    // The typed depth formats.
    const bool depth = own == 20 || own == 40 || own == 45 || own == 55;
    if (!allowed || depth) {
        tally(c_unfit);
        return false;
    }
    const bool typeless = family(own) == own && family(own + 1) == own;
    // An unordered view in an sRGB format: the linear one.
    if (unordered && x2w_srgb::linear(d.Format) != d.Format) {
        // An externally supplied typed sRGB texture cannot provide our typed UAV.
        tally(c_unfit);
        return false;
    }
    if (format && unordered) *format = x2w_srgb::linear(*format);
    const u32 view = format ? static_cast<u32>(*format) : 0;
    if (view == 0) {
        if (typeless) tally(c_unfit);
        return !typeless;
    }
    // For an unsupported target format, use its rendering stand-in: formats 86 and 115 map to 28, and 67 maps to 10.
    const u32 stand_in = view == 86 || view == 115 ? 28u : view == 67 ? 10u : view;
    if (view != own && stand_in == own) {
        *format = d.Format;
        return true;
    }
    if (family(view) != family(own)) {
        tally(c_unfit);
        return false;
    }
    if (!typeless) *format = d.Format;
    return true;
}

// The host descriptor for `view` at `at`, and its raw twin at `twin` (a buffer view's; a constant buffer's is the
// same view again). `staged`: `twin` is a CPU-only slot (move_raw may change it).
inline Made make_twinned(const Env& env, Exacts& exacts, u64 submit, const View& view, D3D12_CPU_DESCRIPTOR_HANDLE at,
                         std::optional<D3D12_CPU_DESCRIPTOR_HANDLE> twin, bool staged, bool primary_staged)
{
    ID3D12Device* const device = env.device;
    if (view.explicit_null) {
        const bool uav = view.kind == Kind::unordered;
        if (uav) gpu_view_trace::MirrorCall{device, __FUNCTION__}.uav(nullptr, nullptr, &view.uav, at);
        else gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(nullptr, &view.srv, at);
        if (twin && view.buffer) {
            if (uav) {
                D3D12_UNORDERED_ACCESS_VIEW_DESC raw{};
                raw.Format = DXGI_FORMAT_R32_TYPELESS;
                raw.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                raw.Buffer.NumElements = 1;
                raw.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
                gpu_view_trace::MirrorCall{device, __FUNCTION__}.uav(nullptr, nullptr, &raw, *twin);
            } else null_buffer_view(device, *twin);
        }
        return {true, nullptr};
    }
    switch (view.kind) {
    case Kind::constant: {
        ID3D12Resource* r = nullptr;
        u64 off = 0, left = 0;
        if (!env.find || !env.find(env.ctx, view.address, (std::max<u64>)(view.bytes, 1), &r, &off, &left)) return {};
        const u64 gpu = r->GetGPUVirtualAddress() + off;
        if (gpu % 256 != 0) {
            // WP3 copies such constants per draw (PLAN.md 1.3).
            tally(c_cbv_unaligned);
            return {};
        }
        u32 size = constant_bytes(static_cast<u32>((std::min<u64>)(view.bytes, 0xffffffffu)));
        // Not past the window.
        const u64 room = r->GetDesc().Width - off;
        if (size > room) size = static_cast<u32>(room & ~u64{255});
        if (!size) return {};
        D3D12_CONSTANT_BUFFER_VIEW_DESC c{gpu, size};
        if (!fit_cbv(c, "constant")) return {};
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.cbv(&c, at);
        if (twin) gpu_view_trace::MirrorCall{device, __FUNCTION__}.cbv(&c, *twin);
        return {true, r};
    }
    case Kind::shader: {
        if (view.buffer) {
            if (!view.has_desc) return {};
            return buffer_view(env, exacts, submit, view.address, view.bytes, view.srv, view.resource, at, twin, staged, primary_staged);
        }
        D3D12_SHADER_RESOURCE_VIEW_DESC desc = view.srv;
        if (!view.resource || !fits(view.resource, view.has_desc ? &desc.Format : nullptr, false)) return {};
        if (view.has_desc && !fit_srv(view.resource, &desc, "texture SRV")) return {};
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.srv(view.resource, view.has_desc ? &desc : nullptr, at);
        return {true, view.resource};
    }
    case Kind::unordered: {
        if (view.buffer) {
            if (!view.has_desc) return {};
            return buffer_view(env, exacts, submit, view.address, view.bytes, view.uav, view.resource, at, twin, staged, primary_staged);
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc = view.uav;
        if (!fits(view.resource, view.has_desc ? &desc.Format : nullptr, true)) return {};
        if (view.has_desc && !fit_uav(view.resource, &desc, "texture UAV")) return {};
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.uav(view.resource, view.counter, view.has_desc ? &desc : nullptr, at);
        return {true, view.resource};
    }
    case Kind::sampler:
        gpu_view_trace::MirrorCall{device, __FUNCTION__}.sampler(&view.sampler, at);
        return {true, nullptr};
    case Kind::placed:
        if (view.unordered) return buffer_view(env, exacts, submit, view.address, view.bytes, view.uav, nullptr, at, twin, staged, primary_staged);
        return buffer_view(env, exacts, submit, view.address, view.bytes, view.srv, nullptr, at, twin, staged, primary_staged);
    default:
        return {};
    }
}

// ---- The mirror heap ----------------------------------------------


// Slots of this thread's own in heaps the card does not see: two of views and two of samplers. Descriptors are made
// there and copied whole into the heap the card sees, so no submission reads one half made.
struct Stage {
    ID3D12Device* device = nullptr;
    ID3D12DescriptorHeap* heaps[2]{};
    u64 starts[2]{};
    u32 steps[2]{};
    ~Stage()
    {
        for (auto* h : heaps)
            if (h) h->Release();
    }
};
inline bool view_copy_on()
{
    static const bool on = [] {
        char v[4]{};
        return !(GetEnvironmentVariableA("X2W_GPU_VIEW_COPY", v, sizeof(v)) && v[0] == '0');
    }();
    return on;
}
inline std::optional<D3D12_CPU_DESCRIPTOR_HANDLE> staged(ID3D12Device* device, bool sampler, u32 which)
{
    if (!view_copy_on()) return std::nullopt;
    thread_local Stage stage;
    if (stage.device != device) {
        for (auto*& h : stage.heaps)
            if (h) h->Release(), h = nullptr;
        stage.device = device;
        for (int k = 0; k < 2; ++k) {
            D3D12_DESCRIPTOR_HEAP_DESC d{k ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            if (FAILED(device->CreateDescriptorHeap(&d, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&stage.heaps[k])))) {
                stage.heaps[k] = nullptr;
                continue;
            }
            stage.starts[k] = stage.heaps[k]->GetCPUDescriptorHandleForHeapStart().ptr;
            stage.steps[k] = device->GetDescriptorHandleIncrementSize(d.Type);
        }
    }
    const int k = sampler ? 1 : 0;
    if (!stage.heaps[k]) return std::nullopt;
    return D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(stage.starts[k] + u64{which} * stage.steps[k])};
}

// What is known of a record a list's tables hold.
struct Listed {
    Kind kind = Kind::none;
    bool unordered = false;
    bool texture = false;
};

class HostHeap {
    struct MaterialRetry {
        std::array<u32, 8> words;
        const char* reason;
    };
public:
    // The layout for `count` title slots under `limits` (probe_view_limit), every slot null. `title_cpu`, `title_step`:
    // the title's descriptor memory and its bytes a slot (records are read there).
    static std::unique_ptr<HostHeap> create(const Env& env, bool sampler, u32 count, const Limits& limits, u64 title_cpu,
                                            u32 title_step, u32 dynamic_most = dynamic_most_default)
    {
        std::unique_ptr<HostHeap> h(new HostHeap);
        h->env_ = env;
        h->layout_ = layout_for(sampler, count, limits, dynamic_most);
        h->title_cpu_ = title_cpu;
        h->title_step_ = title_step;
        const auto kind = sampler ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        D3D12_DESCRIPTOR_HEAP_DESC d{kind, h->layout_.size, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        if (FAILED(env.device->CreateDescriptorHeap(&d, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&h->heap_))))
            return nullptr;
        h->cpu_ = h->heap_->GetCPUDescriptorHandleForHeapStart().ptr;
        h->gpu_ = h->heap_->GetGPUDescriptorHandleForHeapStart().ptr;
        h->step_ = env.device->GetDescriptorHandleIncrementSize(kind);
        const Layout& l = h->layout_;
        h->made_.assign(l.count, 0);
        h->twin_shown_.assign(l.twins, 0);
        h->kept_.assign(u64{l.count} + l.dynamic, nullptr);
        h->runs_.reset(l.count, l.dynamic);
        h->buffers_.resize(l.count);
        h->changed_.resize(l.count);
        if (title_cpu) h->last_.assign(std::size_t{l.count} * title_step, 0);
        for (auto& lock : h->locks_) InitializeSRWLock(&lock);
        if (!sampler) {
            const u64 bytes = u64{slot_strides_count} * 4;
            D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC b{};
            b.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            b.Width = bytes;
            b.Height = 1;
            b.DepthOrArraySize = 1;
            b.MipLevels = 1;
            b.SampleDesc.Count = 1;
            b.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            const D3D12_RANGE none{0, 0};
            if (SUCCEEDED(env.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &b, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                              nullptr, __uuidof(ID3D12Resource),
                                                              reinterpret_cast<void**>(&h->strides_))) &&
                SUCCEEDED(h->strides_->Map(0, &none, reinterpret_cast<void**>(&h->stride_words_)))) {
                std::memset(h->stride_words_, 0, bytes);
                h->strides_->SetName(L"x2w-gpu element strides by view heap slot");
            }
        }
        if (!h->fill_null(kind)) return nullptr;
        return h;
    }
    ~HostHeap()
    {
        for (auto* r : kept_)
            if (r) r->Release();
        for (auto& [_, r] : retired_)
            if (r) r->Release();
        if (strides_) strides_->Release();
        if (heap_) heap_->Release();
    }
    HostHeap(const HostHeap&) = delete;
    HostHeap& operator=(const HostHeap&) = delete;

    const Layout& layout() const { return layout_; }
    ID3D12DescriptorHeap* heap() const { return heap_; }
    u32 count() const { return layout_.count; }
    u32 step() const { return step_; }
    bool sampler() const { return layout_.sampler; }
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(u32 index) const { return {static_cast<SIZE_T>(cpu_ + u64{index} * step_)}; }
    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(u32 index) const { return {gpu_ + u64{index} * step_}; }
    // GPU address and resource for the stride table.
    u64 strides_gpu() const { return strides_ ? strides_->GetGPUVirtualAddress() : 0; }
    ID3D12Resource* strides_buffer() const { return strides_; }
    u32 stride_word(u32 index) const { return stride_words_ && index < slot_strides_count ? stride_words_[index] : 0; }

    // Makes host slot `index` from the record the title has at its slot if it changed since; true if the slot shows a
    // view. A list's dynamic table owns these slot references until its submission retires. A rejected view keeps
    // nullptr even when the original title record still names a texture.
    ID3D12Resource* kept_at(u32 index)
    {
        if (index >= kept_.size()) return nullptr;
        SRWLOCK& lock = lock_of(index);
        AcquireSRWLockShared(&lock);
        ID3D12Resource* resource = kept_[index];
        ReleaseSRWLockShared(&lock);
        return resource;
    }
    x2w_material_trace::Record material_record(u32 slot, u32 base, x2w_material::Mask fields)
    {
        x2w_material_trace::Record record;
        if (!fields || slot >= kept_.size() || base % 4) return record;
        auto& lock = lock_of(slot);
        AcquireSRWLockShared(&lock);
        const auto& ranges = raw_ranges_[slot % 512];
        const auto it = ranges.find(slot);
        if (it != ranges.end() && base <= it->second.second && it->second.second - base >= x2w_material::byte_count(fields)) {
            record.source = kept_[slot];
            record.offset = it->second.first + base;
            record.bytes = it->second.second - base;
            record.fields = fields;
        }
        ReleaseSRWLockShared(&lock);
        return record;
    }
    void material_slots(std::vector<x2w_material_trace::Slot>& slots)
    {
        for (UINT stripe = 0; stripe < 512; ++stripe) {
            auto& lock = locks_[stripe];
            AcquireSRWLockShared(&lock);
            struct Unlock { SRWLOCK* lock; ~Unlock() { ReleaseSRWLockShared(lock); } } unlock{&lock};
            std::lock_guard<std::mutex> guard(mutex_);
            for (UINT i = stripe; i < layout_.count; i += 512) {
                x2w_material_trace::Slot item;
                const size_t pos = size_t(i) * title_step_;
                if (title_step_ <= 32 && pos + title_step_ <= last_.size())
                    std::memcpy(item.words.data(), last_.data() + pos, title_step_);
                auto* resource = i < kept_.size() ? kept_[i] : nullptr;
                if (!resource && !item.words[0] && !item.words[1]) continue;
                item.index = i;
                if (resource) {
                    const auto desc = resource->GetDesc();
                    item.host = reinterpret_cast<UINT_PTR>(resource);
                    item.format = desc.Format;
                    item.dimension = desc.Dimension;
                    item.width = static_cast<UINT>(desc.Width);
                    item.height = desc.Height;
                }
                const auto& group = textures_[i % textures_.size()];
                const auto tex = group.find(i);
                if (tex != group.end()) {
                    const auto& t = tex->second;
                    item.view_format = t.srv.Format;
                    item.texture_address = u64(t.placement[2]) << 32 | t.placement[1];
                }
                if (item.words[0] == x2w_late_texture::lazy || item.words[0] == x2w_late_texture::magic) {
                    item.lazy = lazy_.count(i) || lazy_waiting_.count(i);
                    for (const auto* retries : {&material_due_, &material_waiting_}) {
                        auto it = retries->find(i);
                        if (it != retries->end() && it->second.words == item.words)
                            item.material_retry = it->second.reason;
                    }
                    item.late = late_.count(i) != 0;
                    if (!item.texture_address) item.texture_address = u64(item.words[2]) << 32 | item.words[1];
                    if (!item.view_format) item.view_format = (item.words[4] >> 8) & 255;
                }
                slots.push_back(item);
            }
        }
    }
    bool mirror(u32 index, bool use = true)
    {
        if (index >= layout_.count) return false;
        std::uint32_t words[8]{};
        const u32 bytes = (std::min)(title_step_, 32u);
        if (!title_cpu_ || !read_title(title_cpu_ + u64{index} * title_step_, words, bytes)) return false;
        // Remember the mirrored words so submit detects a title copy that clears this slot.
        if (title_step_ <= 32) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!last_.empty()) std::memcpy(last_.data() + std::size_t{index} * title_step_, words, title_step_);
        }
        return mirror_words(index, words, use);
    }
    // As mirror, from the record's words as already read.
    bool mirror_words(u32 index, const std::uint32_t words[8], bool use = true)
    {
        if (index >= layout_.count) return false;
        View v;
        if (!use && words[0] == x2w_late_texture::lazy && !layout_.sampler) {
            tally(c_lazy_staged);
            v.kind = Kind::shader;
            v.has_desc = true;
            D3D12_RESOURCE_DESC desc{};
            if (!x2w_late_texture::decode(words, desc, v.srv)) {
                make_null(index);
                return false;
            }
            std::copy_n(words, 8, v.placement.begin());
            v.placement[0] = x2w_late_texture::magic;
            SRWLOCK& lock = lock_of(index);
            AcquireSRWLockExclusive(&lock);
            const bool retained = restore_pending(index, v);
            if (!retained) {
                make_locked(index, v);
                made_[index] = 0;
                note_buffer(index, nullptr);
            }
            {
                std::lock_guard<std::mutex> guard(mutex_);
                late_.erase(index);
                unmade_.erase(index);
                lazy_waiting_.erase(index);
                lazy_.insert(index);
            }
            ReleaseSRWLockExclusive(&lock);
            return retained;
        }
        const bool has = view_of_record(env_, words, layout_.sampler, v);
        SRWLOCK& lock = lock_of(index);
        AcquireSRWLockExclusive(&lock);
        const u64 shown = made_[index];
        if (!has) {
            // Not a view of ours (never written, or of a resource the title released): nothing to read.
            if (shown != 0) {
                made_[index] = 0;
                make_null_locked(index);
                note_buffer(index, nullptr);
            }
            ReleaseSRWLockExclusive(&lock);
            return false;
        }
        if (v.pending && !v.resource && restore_pending(index, v)) {
            tally(c_retained_pending);
            {
                std::lock_guard<std::mutex> guard(mutex_);
                late_.insert(index);
            }
            ReleaseSRWLockExclusive(&lock);
            return true;
        }
        if (shown == v.key && (!x2w_late_texture::marker(words[0]) || (v.resource && kept_[index] == v.resource))) {
            if (v.pending) {
                std::lock_guard<std::mutex> guard(mutex_);
                late_.insert(index);
            }
            ReleaseSRWLockExclusive(&lock);
            return true;
        }
        tally(c_mirrored);
        const bool made = make_locked(index, v);
        // A view that could not be made (its window not yet known, say) shows null and is not taken as made: it is
        // tried again at the next submits (retry_unmade) until it is, or its record changes.
        made_[index] = made ? v.key : 0;
        note_buffer(index, made ? &v : nullptr);
        ReleaseSRWLockExclusive(&lock);
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (words[0] == x2w_late_texture::lazy && made && !v.pending) {
                lazy_.erase(index);
                lazy_waiting_.erase(index);
                material_due_.erase(index);
                material_waiting_.erase(index);
            }
            if ((!made || v.pending) && x2w_late_texture::marker(words[0])) late_.insert(index);
            else if (made) unmade_.erase(index);
            else ++unmade_[index];
        }
        return made;
    }
    // The slots whose views could not be made, tried again (at most `most` of them a submit; a slot is given up after
    // 64 tries, until its record changes); how many were made.
    u64 retry_unmade(std::size_t most)
    {
        constexpr u32 tries = 64;
        std::vector<u32> trying;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            for (auto it = unmade_.begin(); it != unmade_.end() && trying.size() < most;) {
                if (it->second >= tries) {
                    it = unmade_.erase(it);
                    continue;
                }
                trying.push_back(it->first);
                ++it;
            }
        }
        u64 made = 0;
        for (u32 index : trying)
            if (made_[index] == 0 && mirror(index) && made_[index] != 0) ++made;
        return made;
    }
    // Mirrors `n` slots from `first` (a range the title wrote or copied by a call).
    void mirror_range(u32 first, u32 n)
    {
        for (u64 i = first; i < u64{first} + n && i < layout_.count; ++i) mirror(static_cast<u32>(i), false);
    }
    bool view_at(u32 source, const u32 words[8], View& view)
    {
        if (!view_of_record(env_, words, layout_.sampler, view)) return false;
        if (view.pending && !view.resource && source < layout_.count) {
            auto& lock = lock_of(source);
            AcquireSRWLockShared(&lock);
            restore_pending(source, view);
            ReleaseSRWLockShared(&lock);
        }
        return true;
    }
    // Data-dependent GPU indices have no CPU table lookup. Resolve a bounded rotating set on an open
    // list, recording the upload before exposing the real binding. Re-read the title after the upload.
    void warm_lazy(void* list, u64 frame, LazyBudget& budget, u32 visit_limit = 4096,
                   u64 byte_limit = 96ull << 20, double time_limit_ms = 32)
    {
        if (!visit_limit || !env_.upload) return;
        std::unique_lock budget_lock(budget.mutex, std::try_to_lock);
        if (!budget_lock) return;
        {
            std::lock_guard guard(mutex_);
            if (lazy_.empty() && lazy_waiting_.empty()) return;
        }
        if (budget.frame != frame) {
            budget.frame = frame;
            budget.bytes = budget.visits = 0;
            budget.elapsed_ms = 0;
            budget.room = env_.local_room ? env_.local_room(env_.device) : UINT64_MAX;
        }
        if (budget.visits >= visit_limit || budget.bytes >= byte_limit ||
            budget.elapsed_ms >= time_limit_ms) return;
        const auto begin = now();
        const u64 generation = env_.mapping_epoch ? env_.mapping_epoch() : 0;
        {
            std::lock_guard guard(mutex_);
            if (lazy_frame_ != frame || lazy_generation_ != generation) {
                lazy_.merge(lazy_waiting_);
                material_due_.merge(material_waiting_);
                lazy_frame_ = frame;
                lazy_generation_ = generation;
            }
            if (lazy_.empty()) return;
        }
        for (u32 i = 0; i < 64 && budget.visits < visit_limit && budget.bytes < byte_limit; ++i) {
            if (budget.elapsed_ms + ms(now() - begin) >= time_limit_ms) break;
            if (env_.local_room && budget.visits && (budget.visits % 32 == 0))
                budget.room = env_.local_room(env_.device);
            u32 slot;
            std::optional<MaterialRetry> selected;
            {
                std::lock_guard guard(mutex_);
                if (lazy_.empty() || (!budget.room && material_due_.empty())) break;
                auto it = lazy_.lower_bound(lazy_cursor_);
                if (it == lazy_.end()) it = lazy_.begin();
                slot = *it;
                // Leave every fourth visit to the general queue while allocations are possible.
                if (!material_due_.empty() && (!budget.room || budget.visits % 4 != 3)) {
                    auto next = material_due_.lower_bound(material_cursor_);
                    if (next == material_due_.end()) next = material_due_.begin();
                    slot = next->first;
                    material_cursor_ = slot + 1;
                } else lazy_cursor_ = slot + 1;
                if (auto entry = material_due_.extract(slot); !entry.empty()) selected = entry.mapped();
                lazy_.erase(slot);
            }
            ++budget.visits;
            const u64 before = env_.copied_bytes ? env_.copied_bytes() : 0;
            const u64 allocated = env_.allocated_bytes ? env_.allocated_bytes() : 0;
            warm_slot(list, slot, false, budget.room, selected ? &*selected : nullptr);
            if (env_.copied_bytes) budget.bytes += env_.copied_bytes() - before;
            if (env_.allocated_bytes) {
                const u64 delta = env_.allocated_bytes() - allocated;
                budget.room -= (std::min)(budget.room, delta);
            }
        }
        budget.elapsed_ms += ms(now() - begin);
    }
    bool warm_slot(void* list, u32 slot, bool targeted = false, u64 room = UINT64_MAX, const MaterialRetry* retry = nullptr)
    {
        if (!env_.upload || slot >= layout_.count || title_step_ != 32) return false;
        const u64 at = title_cpu_ + u64{slot} * title_step_;
        u32 words[8]{}, latest[8];
        const auto park_locked = [&](const char* reason) {
            if ((targeted || retry) && words[0] == x2w_late_texture::lazy &&
                (material_waiting_.count(slot) || material_due_.count(slot) ||
                 material_waiting_.size() + material_due_.size() < 512)) {
                MaterialRetry entry{};
                std::copy_n(words, 8, entry.words.begin());
                entry.reason = reason;
                material_due_.erase(slot);
                material_waiting_[slot] = entry;
                lazy_.erase(slot);
            }
            if (!lazy_.count(slot)) lazy_waiting_.insert(slot);
        };
        const auto park = [&](const char* reason) {
            std::lock_guard guard(mutex_);
            park_locked(reason);
            return false;
        };
        if (!read_title(at, words, sizeof(words))) {
            if (retry) std::copy_n(retry->words.begin(), 8, words);
            return park("descriptor_unreadable");
        }
        if (retry && !std::equal(retry->words.begin(), retry->words.end(), words)) {
            mirror_words(slot, words, false);
            return false;
        }
        if (words[0] != x2w_late_texture::lazy) return false;
        D3D12_RESOURCE_DESC desc{};
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        if (!x2w_late_texture::decode(words, desc, srv)) return false;
        if (targeted) {
            std::lock_guard<std::mutex> guard(mutex_);
            const auto& bucket = textures_[slot % textures_.size()];
            const auto it = bucket.find(slot);
            if (it != bucket.end() && it->second.texture && !lazy_.count(slot) && !lazy_waiting_.count(slot) &&
                it->second.placement[0] == x2w_late_texture::magic &&
                std::equal(words + 1, words + 8, it->second.placement.begin() + 1)) return false;
        }
        if (targeted && room == UINT64_MAX && env_.local_room) room = env_.local_room(env_.device);
        // Selected materials may already have a compatible placement or streamed mip suffix.
        // Charge an allocation only when the publisher actually creates a host texture.
        const bool allow_create = room && (room == UINT64_MAX || !env_.allocation_size ||
                                          env_.allocation_size(env_.device, words) <= room);
        if (!allow_create && !targeted && !retry) return park("local_budget");
        const u64 generation = !allow_create && env_.mapping_epoch ? env_.mapping_epoch() : 0;
        View view;
        const bool ready = view_of_record(env_, words, false, view, allow_create) && view.resource &&
            env_.upload(env_.ctx, list, view.wrapper, view.resource);
        if (!read_title(at, latest, sizeof(latest))) return park("descriptor_unreadable");
        if (!std::equal(std::begin(words), std::end(words), latest)) {
            mirror_words(slot, latest, false);
            return false;
        }
        if (!allow_create && env_.mapping_epoch && env_.mapping_epoch() != generation) return park("mapping_changed");
        if (!ready) return park(view.resource ? "upload_failed" : !allow_create ? "local_budget" :
                               view.pending ? "backing_unavailable" : "unresolved");
        SRWLOCK& lock = lock_of(slot);
        AcquireSRWLockExclusive(&lock);
        bool made = false;
        if (read_title(at, latest, sizeof(latest)) && std::equal(std::begin(words), std::end(words), latest)) {
            made = make_locked(slot, view);
            made_[slot] = made ? view.key : 0;
            note_buffer(slot, nullptr);
            if (made) tally(c_lazy_warmed);
            std::lock_guard<std::mutex> guard(mutex_);
            if (!made || view.pending) {
                park_locked(made ? "partial_mips" : "view_failed");
            } else {
                material_due_.erase(slot);
                material_waiting_.erase(slot);
                lazy_.erase(slot);
                lazy_waiting_.erase(slot);
            }
        } else {
            park("descriptor_changed");
        }
        ReleaseSRWLockExclusive(&lock);
        return made;
    }
    unsigned materialize(void* list, const x2w_material_trace::Words& words, x2w_material::Mask fields)
    {
        unsigned made = 0;
        for (size_t i = 0; fields; fields >>= 1, ++i)
            if (fields & 1) made += warm_slot(list, x2w_material_trace::index(words[i]), true);
        return made;
    }
    // Whether the slot currently exposes a null descriptor.
    bool shows_none(u32 index) const { return index >= layout_.count || made_[index] == 0; }

    // Makes slot `index` (a dynamic one) from `view`; a null view if the host cannot.
    void make_at(u32 index, const View& view)
    {
        if (index >= layout_.size) return;
        SRWLOCK& lock = lock_of(index);
        AcquireSRWLockExclusive(&lock);
        make_locked(index, view);
        ReleaseSRWLockExclusive(&lock);
    }
    // Makes slot `index` (a dynamic one) the view the record `words` names. Constant buffers are not made so (their
    // bytes are copied for each draw, WP3): nullopt, as for a record that names no view.
    std::optional<Listed> make_listed(u32 index, const std::uint32_t words[8], u32 source = UINT32_MAX, View* result = nullptr)
    {
        if (layout_.sampler) return std::nullopt;
        View v;
        if (!view_at(source, words, v) || v.kind == Kind::constant) return std::nullopt;
        make_at(index, v);
        if (result) *result = v;
        return Listed{v.kind, v.kind == Kind::unordered || (v.kind == Kind::placed && v.unordered), v.is_texture()};
    }
    // Reset a dynamic slot to a null descriptor.
    void make_null(u32 index)
    {
        if (index >= layout_.size) return;
        SRWLOCK& lock = lock_of(index);
        AcquireSRWLockExclusive(&lock);
        make_null_locked(index);
        if (index < layout_.count) {
            made_[index] = 0;
            note_buffer(index, nullptr);
        }
        ReleaseSRWLockExclusive(&lock);
    }

    // Slot `index` (a dynamic one) a raw view of `len` bytes of `buffer` from byte `offset` (a multiple of 16): a root
    // SRV with a size.
    void raw_at(u32 index, ID3D12Resource* buffer, u64 offset, u64 len)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.FirstElement = offset / 4;
        d.Buffer.NumElements = static_cast<UINT>((std::min<u64>)((std::max<u64>)(len / 4, 1), 0xffffffffu));
        d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        if (!fit_buffer(buffer, d, "root SRV")) return null_raw_at(index);
        made_by(index, buffer, [&](D3D12_CPU_DESCRIPTOR_HANDLE at) { gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.srv(buffer, &d, at); });
    }
    // A root UAV with a size.
    void raw_uav_at(u32 index, ID3D12Resource* buffer, u64 offset, u64 len)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.FirstElement = offset / 4;
        d.Buffer.NumElements = static_cast<UINT>((std::min<u64>)((std::max<u64>)(len / 4, 1), 0xffffffffu));
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        if (!fit_buffer(buffer, d, "root UAV")) return null_raw_uav_at(index);
        made_by(index, buffer, [&](D3D12_CPU_DESCRIPTOR_HANDLE at) { gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.uav(buffer, nullptr, &d, at); });
    }
    // Create a null raw SRV or UAV.
    void null_raw_at(u32 index)
    {
        made_by(index, nullptr, [&](D3D12_CPU_DESCRIPTOR_HANDLE at) { null_buffer_view(env_.device, at); });
    }
    void null_raw_uav_at(u32 index)
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_R32_TYPELESS;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.NumElements = 1;
        d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        made_by(index, nullptr, [&](D3D12_CPU_DESCRIPTOR_HANDLE at) { gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.uav(nullptr, nullptr, &d, at); });
    }
    // Slot `index` (a dynamic one) a constant buffer view of `size` bytes at `gpu` of `buffer`, which the slot holds;
    // its twin the raw view of the same bytes.
    void constant_at(u32 index, ID3D12Resource* buffer, u64 gpu, u32 size)
    {
        if (index >= layout_.size) return;
        D3D12_CONSTANT_BUFFER_VIEW_DESC c{gpu, constant_bytes(size)};
        if (!fit_cbv(c, "constant_at")) c = {};
        SRWLOCK& lock = lock_of(index);
        AcquireSRWLockExclusive(&lock);
        const auto at = make_place(0);
        gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.cbv(&c, at.first);
        put(index, at);
        bool twin = false;
        if (layout_.has_twin(index) && c.SizeInBytes > 0 && buffer) {
            const auto t = make_place(1);
            D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
            raw.Format = DXGI_FORMAT_R32_TYPELESS;
            raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            raw.Buffer.FirstElement = (gpu - buffer->GetGPUVirtualAddress()) / 4;
            raw.Buffer.NumElements = c.SizeInBytes / 4;
            raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            if (fit_buffer(buffer, raw, "constant twin")) gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.srv(buffer, &raw, t.first);
            else null_buffer_view(env_.device, t.first);
            put(layout_.twin(index), t);
            twin = true;
        }
        twin_made(index, twin);
        keep(index, buffer);
        ReleaseSRWLockExclusive(&lock);
    }

    // Allocate and release dynamic descriptor slots.
    std::optional<u32> dynamic(u32 n)
    {
        if (layout_.sampler) return std::nullopt;
        std::lock_guard<std::mutex> lock(mutex_);
        const auto first = runs_.take(n);
        if (!first) {
            static std::atomic<bool> said{false};
            if (!said.exchange(true)) log("x2w-gpu heap: no free dynamic descriptor slots: a table is not made (%u asked)", n);
        }
        return first;
    }
    void release(u32 first, u32 n)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        runs_.release(first, n);
    }

    // Track occupied buffer slots and changes to their descriptors.
    std::vector<u32> buffer_slots()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<u32> out;
        buffers_.each([&](u32 i) { out.push_back(i); });
        return out;
    }
    std::vector<u32> buffer_changes()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return changed_.take();
    }

    // At submission, compare title records with their last mirrored bytes to catch descriptors copied without API
    // calls. Refresh private buffer-view copies, retire old resources and schedule a portion of visible textures for
    // upload. Return the slots that changed.
    u64 submit()
    {
        const LONGLONG began = now();
        const u64 at = ++submits_;
        u64 changed = 0;
        // Retry the previous submit's unresolved slots before scanning new writes. mirror() updates
        // last_ from the live record, so the diff does not resolve the same unchanged slot twice.
        // Revisit a bounded rotating subset each submit. Read the live record in mirror()
        // so a descriptor overwritten while streaming cannot restore the earlier texture.
        std::vector<u32> late;
        if (env_.late_texture) {
            std::lock_guard<std::mutex> guard(mutex_);
            auto it = late_.lower_bound(late_cursor_);
            if (it == late_.end()) it = late_.begin();
            while (it != late_.end() && late.size() < 32) {
                late.push_back(*it);
                it = late_.erase(it);
            }
            late_cursor_ = it == late_.end() ? 0 : *it;
        }
        for (u32 index : late) mirror(index);
        if (title_cpu_) {
            const std::size_t step = title_step_;
            const std::size_t total = std::size_t{layout_.count} * step;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (last_.size() != total) last_.assign(total, 0);
            }
            constexpr std::size_t piece = 64u << 10;
            std::vector<unsigned char> now_bytes(piece);
            std::vector<u32> slots;
            for (std::size_t from = 0; from < total; from += piece) {
                const std::size_t n = (std::min)(piece, total - from) / step * step;
                if (!n) break;
                if (!read_title(title_cpu_ + from, now_bytes.data(), n)) continue;
                slots.clear();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    changed_records(last_.data() + from, now_bytes.data(), n, step, static_cast<u32>(from / step), slots);
                }
                for (u32 index : slots) {
                    std::uint32_t words[8]{};
                    std::memcpy(words, now_bytes.data() + (std::size_t{index} * step - from), (std::min<std::size_t>)(step, 32));
                    mirror_words(index, words, false);
                }
                changed += slots.size();
            }
        }
        tally(c_changed, changed);
        tally(c_retried, retry_unmade(4096));
        exacts_.refresh(env_, at, 3);
        retire(at);
        sweep_textures(at);
        const LONGLONG spent = now() - began;
        submit_ticks_ += spent;
        if ((at & (at - 1)) == 0)
            log("x2w-gpu heap submit %llu: %llu records changed (%llu in all), %.2f ms (%.2f ms a submit); made %llu, null %llu, "
                "twins %llu (moved %llu, null %llu), refused %llu, unaligned CBVs %llu, crossing UAVs %llu, own-byte copies %llu "
                "(%zu live, %llu failed: invalid %llu, cache full %llu, create %llu, map %llu, unreadable %llu), direct unaligned SRVs %llu, retried and made %llu, V# %llu, S# %llu, unfit %llu, textures due %llu, uploaded %llu, views cut to fit %llu, refused as unfit for the host %llu, matching pending texture views retained %llu, lazy visible slots staged %llu, warmed on bind %llu",
                at, changed, counted(c_changed), ms(spent), ms(submit_ticks_) / static_cast<double>(at), counted(c_made),
                counted(c_null), counted(c_twins), counted(c_twins_moved), counted(c_twins_null), counted(c_refused),
                counted(c_cbv_unaligned), counted(c_cross_uav), counted(c_exact), exacts_.size(), counted(c_exact_failed),
                counted(c_exact_invalid), counted(c_exact_capacity), counted(c_exact_create), counted(c_exact_map),
                counted(c_exact_unreadable), counted(c_direct_offset), counted(c_retried),
                counted(c_vsharp), counted(c_ssharp), counted(c_unfit), counted(c_textures_due), counted(c_uploads), counted(c_fit_cut),
                counted(c_fit_refused), counted(c_retained_pending), counted(c_lazy_staged), counted(c_lazy_warmed));
        return changed;
    }

    // PLAN.md 4b: up to `max` textures due (made into a slot, or swept again) are uploaded on `list`; how many.
    std::size_t upload_due(void* list, std::size_t max)
    {
        if (!env_.upload) return 0;
        std::size_t done = 0;
        while (done < max) {
            Texture t;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (due_.empty()) break;
                const u32 index = due_.front();
                due_.pop_front();
                due_set_[index] = 0;
                const auto& group = textures_[index % textures_.size()];
                const auto found = group.find(index);
                if (found == group.end()) continue;
                t = found->second;
                t.texture->AddRef();
            }
            if (env_.upload(env_.ctx, list, t.wrapper, t.texture)) tally(c_uploads);
            t.texture->Release();
            ++done;
        }
        return done;
    }
    std::size_t textures_due()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return due_.size();
    }

private:
    HostHeap() = default;

    struct Texture {
        u64 wrapper = 0;
        ID3D12Resource* texture = nullptr;  // held by kept_ while the slot shows it
        std::array<u32, 8> placement{};
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        u64 view_key = 0;
    };

    static LONGLONG now()
    {
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        return t.QuadPart;
    }
    static double ms(LONGLONG ticks)
    {
        static const double per = [] {
            LARGE_INTEGER f;
            QueryPerformanceFrequency(&f);
            return 1000.0 / static_cast<double>(f.QuadPart);
        }();
        return static_cast<double>(ticks) * per;
    }
    void log(const char* format, ...) const
    {
        if (!env_log()) return;
        char text[1024];
        va_list args;
        va_start(args, format);
        std::vsnprintf(text, sizeof(text), format, args);
        va_end(args);
        env_log()(text);
    }

public:
    // Where log lines go (the shim's log; fixtures: stdout). Null: nowhere.
    static void (*&env_log())(const char*)
    {
        static void (*sink)(const char*) = nullptr;
        return sink;
    }

private:
    bool read_title(u64 address, void* out, std::size_t bytes) const
    {
        return read_memory(env_, address, out, bytes);
    }
    SRWLOCK& lock_of(u32 index) { return locks_[index % 512]; }

    // Create the descriptor in this thread's staged slot (or in place), then copy it.
    std::pair<D3D12_CPU_DESCRIPTOR_HANDLE, bool> make_place(u32 which)
    {
        if (const auto s = staged(env_.device, layout_.sampler, which)) return {*s, true};
        return {D3D12_CPU_DESCRIPTOR_HANDLE{0}, false};
    }
    D3D12_CPU_DESCRIPTOR_HANDLE where(std::pair<D3D12_CPU_DESCRIPTOR_HANDLE, bool> place, u32 index) const
    {
        return place.second ? place.first : cpu(index);
    }
    void put(u32 index, std::pair<D3D12_CPU_DESCRIPTOR_HANDLE, bool> place)
    {
        if (!place.second) return;  // made in place
        gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.copy(1, cpu(index), place.first,
                                           layout_.sampler ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }

    // The body of mirror and make_at, under the slot's lock.
    bool make_locked(u32 index, const View& v)
    {
        auto at = make_place(0);
        at.first = where(at, index);
        std::optional<D3D12_CPU_DESCRIPTOR_HANDLE> twin;
        auto tw = make_place(1);
        const bool twins = layout_.has_twin(index);
        if (twins) {
            tw.first = where(tw, layout_.twin(index));
            twin = tw.first;
        }
        const Made m = make_twinned(env_, exacts_, submits_, v, at.first, twin, tw.second, at.second);
        if (!m.ok && !layout_.sampler) {
            null_for(env_.device, v, at.first);
            tally(c_null);
        }
        if (m.ok) tally(c_made);
        if (m.ok || !layout_.sampler) put(index, at);
        const bool twin_shown = m.ok && twinned(v) && twins;
        if (twin_shown) put(layout_.twin(index), tw);
        twin_made(index, twin_shown);
        keep(index, m.ok ? m.kept : nullptr);
        if (m.raw_bytes) raw_ranges_[index % 512][index] = {m.raw_start, m.raw_bytes};
        note_texture(index, m.ok && v.is_texture() && !v.depth_twins ? &v : nullptr);
        return m.ok;
    }
    void make_null_locked(u32 index)
    {
        auto at = make_place(0);
        at.first = where(at, index);
        if (layout_.sampler) null_sampler(env_.device, at.first);
        else null_view(env_.device, at.first);
        put(index, at);
        twin_made(index, false);
        keep(index, nullptr);
        note_texture(index, nullptr);
    }
    // A dynamic slot made by `make` (raw_at and its kin): bound by its handle, not read by index, so no twin.
    template <class F> void made_by(u32 index, ID3D12Resource* kept, F&& make)
    {
        if (index >= layout_.size) return;
        SRWLOCK& lock = lock_of(index);
        AcquireSRWLockExclusive(&lock);
        auto at = make_place(0);
        at.first = where(at, index);
        make(at.first);
        put(index, at);
        twin_made(index, false);
        keep(index, kept);
        ReleaseSRWLockExclusive(&lock);
    }
    // Copy the staged descriptor into the twin of slot `index`, or clear it to a null buffer view.
    void twin_made(u32 index, bool made)
    {
        if (!layout_.has_twin(index)) return;
        if (made) {
            twin_shown_[index] = 1;
            return;
        }
        if (twin_shown_[index]) {
            twin_shown_[index] = 0;
            auto at = make_place(1);
            at.first = where(at, layout_.twin(index));
            null_buffer_view(env_.device, at.first);
            put(layout_.twin(index), at);
        }
    }
    // Slot `index` now views `resource` (one reference held); the old one is released some submits on (a running
    // submission may still read it through the slot).
    void keep(u32 index, ID3D12Resource* resource)
    {
        raw_ranges_[index % 512].erase(index);
        if (index >= kept_.size()) return;
        ID3D12Resource*& slot = kept_[index];
        if (slot == resource) return;
        if (resource) resource->AddRef();
        ID3D12Resource* old = slot;
        slot = resource;
        if (old) {
            std::lock_guard<std::mutex> lock(mutex_);
            retired_.push_back({submits_, old});
        }
    }
    void retire(u64 at)
    {
        constexpr u64 keep_submits = 3;
        std::vector<ID3D12Resource*> going;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            while (!retired_.empty() && retired_.front().first + keep_submits < at) {
                going.push_back(retired_.front().second);
                retired_.pop_front();
            }
        }
        for (auto* r : going) r->Release();
    }
    // The table of strides, and the set of buffer slots.
    void note_buffer(u32 index, const View* v)
    {
        if (layout_.sampler) return;
        const auto e = v ? view_elements(*v) : std::nullopt;
        if (stride_words_ && index < slot_strides_count) {
            volatile std::uint32_t* w = stride_words_ + index;
            *w = ::gpu_heap::stride_word(e);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        const bool was = e ? buffers_.put(index) || true : buffers_.remove(index);
        if (was) changed_.put(index);
    }
    // PLAN.md 4b: a texture view made into a slot is due for upload.
    void note_texture(u32 index, const View* v)
    {
        if (layout_.sampler) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!v || !v->wrapper) {
            textures_[index % textures_.size()].erase(index);
            return;
        }
        textures_[index % textures_.size()][index] = Texture{v->wrapper, v->resource, v->placement, v->srv, v->key};
        queue_texture(index);
    }
    bool restore_pending(u32 index, View& v)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        const auto& group = textures_[index % textures_.size()];
        const auto it = group.find(index);
        if (it == group.end() || it->second.placement != v.placement || !it->second.texture) return false;
        v.retained = it->second.texture;
        v.resource = v.retained.Get();
        v.wrapper = it->second.wrapper;
        v.srv = it->second.srv;
        v.key = it->second.view_key;
        return true;
    }
    void queue_texture(u32 index)
    {
        if (due_set_.size() < layout_.count) due_set_.resize(layout_.count, 0);
        if (index >= due_set_.size() || due_set_[index]) return;
        due_set_[index] = 1;
        due_.push_back(index);
        tally(c_textures_due);
    }
    // A sixteenth of the slots that show a texture are due again at each submit: the title may have written its
    // memory since (upload_texture uploads only what changed).
    void sweep_textures(u64 at)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [index, texture] : textures_[at % textures_.size()]) queue_texture(index);
    }

    // The first `size` slots null: one null slot in a CPU-side heap, doubled to a block, the block copied over the
    // heap; the twins null buffer views.
    bool fill_null(D3D12_DESCRIPTOR_HEAP_TYPE kind)
    {
        constexpr u32 block = 1u << 16;
        ID3D12DescriptorHeap* cpu_heap = nullptr;
        D3D12_DESCRIPTOR_HEAP_DESC d{kind, block, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        if (FAILED(env_.device->CreateDescriptorHeap(&d, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&cpu_heap)))) return false;
        const u64 start = cpu_heap->GetCPUDescriptorHandleForHeapStart().ptr;
        const auto at = [&](u32 i) { return D3D12_CPU_DESCRIPTOR_HANDLE{static_cast<SIZE_T>(start + u64{i} * step_)}; };
        const u32 total = layout_.size;
        const u32 ends[2] = {layout_.null_views_end(), total};
        u32 done = 0;
        for (int twin = 0; twin < 2; ++twin) {
            if (done >= ends[twin]) continue;
            if (layout_.sampler) null_sampler(env_.device, at(0));
            else if (twin) null_buffer_view(env_.device, at(0));
            else null_view(env_.device, at(0));
            for (u32 filled = 1; filled < block;) {
                const u32 n = (std::min)(filled, block - filled);
                gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.copy(n, at(filled), at(0), kind);
                filled += n;
            }
            while (done < ends[twin]) {
                const u32 n = (std::min)(block, ends[twin] - done);
                gpu_view_trace::MirrorCall{env_.device, __FUNCTION__}.copy(n, cpu(done), at(0), kind);
                done += n;
            }
        }
        cpu_heap->Release();
        return true;
    }

    Env env_;
    Layout layout_;
    ID3D12DescriptorHeap* heap_ = nullptr;
    u64 cpu_ = 0, gpu_ = 0;
    u32 step_ = 0;
    u64 title_cpu_ = 0;
    u32 title_step_ = 32;
    SRWLOCK locks_[512];
    std::unordered_map<u32, std::pair<u64, u64>> raw_ranges_[512];
    std::vector<u64> made_;                 // per title slot: the key of the record it was made from (0: none)
    std::vector<std::uint8_t> twin_shown_;  // per slot with a twin: the twin is not a null view
    std::unordered_map<u32, u32> unmade_;   // slots whose views could not be made: tries (mutex_)
    std::set<u32> late_;                    // unmapped placed textures (mutex_)
    u32 late_cursor_ = 0;
    std::set<u32> lazy_, lazy_waiting_;
    std::map<u32, MaterialRetry> material_due_, material_waiting_;
    u32 material_cursor_ = 0;
    u32 lazy_cursor_ = 0;
    u64 lazy_generation_ = 0;
    u64 lazy_frame_ = ~u64{0};
    std::vector<ID3D12Resource*> kept_;     // per slot: what it views (one reference)
    std::mutex mutex_;                      // runs_, retired_, buffers_, changed_, last_, textures_, due_
    DynamicRuns runs_;
    std::deque<std::pair<u64, ID3D12Resource*>> retired_;
    SlotSet buffers_, changed_;
    std::vector<unsigned char> last_;       // the records last seen of the title's heap
    std::array<std::unordered_map<u32, Texture>, 16> textures_;
    std::deque<u32> due_;
    std::vector<std::uint8_t> due_set_;
    Exacts exacts_;
    ID3D12Resource* strides_ = nullptr;
    std::uint32_t* stride_words_ = nullptr;
    std::atomic<u64> submits_{0};
    LONGLONG submit_ticks_ = 0;
};

// Probe once per device for 2^22 views including twins, then 2^21 views, then 1,000,000.
inline Limits probe_view_limit(ID3D12Device* device, bool twins_wanted)
{
    return ::gpu_heap::probe_view_limit(twins_wanted, [device](u32 n) {
        D3D12_DESCRIPTOR_HEAP_DESC d{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, n, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        ID3D12DescriptorHeap* heap = nullptr;
        if (FAILED(device->CreateDescriptorHeap(&d, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&heap)))) return false;
        heap->Release();
        return true;
    });
}

}  // namespace gpu_mirror
