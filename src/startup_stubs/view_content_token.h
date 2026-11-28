// Content view tokens (opt-in X2W_D3D12_VIEW_CONTENT_TOKENS=1, #1199): the 32-byte token note_view writes into title
// descriptor memory carries the view itself instead of a view record id, for the two kinds that fit: a constant
// buffer view and a buffer shader-resource view. A copy the title makes with its own memory copy then still names
// the view after the record behind the shadow slot is gone, so such records can be freed once no shadow slot names
// them (docs/startup-runtime.md#d3d12-view-content-tokens).
//
// Layout (32-bit words): [0] view token magic, [1] content_bit | flags, [2] slot (bits 0-7), [3] always 0, then
//   slot 17: [4..5] BufferLocation, [6] SizeInBytes
//   slot 18: [4..5] resource wrapper (where an id token keeps it), [6] FirstElement, [7] NumElements, and
//            StructureByteStride in word 2 bits 8-31
// with word 1 holding the created bit, and for slot 18 the format (bits 0-9), the raw flag (bit 10) and the component
// mapping (bits 11-23). An id token's word 1 is a record index below 2^31, so the content bit never appears there.
// Word 3 stays 0 as in an id token: the formatted-buffer readers put the word they build there in their copy.
#pragma once

#include <d3d12.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace x2w_startup {

constexpr std::uint32_t view_content_bit = 0x80000000u;
constexpr std::uint32_t view_content_created = 0x40000000u;
constexpr std::size_t cbv_desc_bytes = 16;
constexpr std::size_t buffer_srv_desc_bytes = 40;
static_assert(sizeof(D3D12_CONSTANT_BUFFER_VIEW_DESC) == cbv_desc_bytes, "desktop CBV description layout");
static_assert(sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC) == buffer_srv_desc_bytes, "desktop SRV description layout");

struct ContentView {
    std::uint32_t slot = 0;
    std::uint64_t resource = 0;
    std::uint32_t desc_bytes = 0;
    unsigned char desc[buffer_srv_desc_bytes] = {};
    bool created = false;
};

inline bool is_view_content(const std::uint32_t* words)
{
    return (words[1] & view_content_bit) != 0;
}

// Fills token and returns true when the view fits a content token; otherwise leaves token alone.
inline bool encode_view_content(std::uint32_t magic, std::uint32_t slot, std::uint64_t resource, std::uint64_t counter,
                                const unsigned char* desc, std::size_t desc_bytes, bool created, std::uint32_t (&token)[8])
{
    if (counter || !desc) return false;
    std::uint32_t out[8] = {magic, view_content_bit | (created ? view_content_created : 0u), slot, 0, 0, 0, 0, 0};
    if (slot == 17) {
        if (resource || desc_bytes != cbv_desc_bytes) return false;
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv;
        std::memcpy(&cbv, desc, sizeof(cbv));
        out[4] = static_cast<std::uint32_t>(cbv.BufferLocation);
        out[5] = static_cast<std::uint32_t>(cbv.BufferLocation >> 32);
        out[6] = cbv.SizeInBytes;
    } else if (slot == 18) {
        if (!resource || desc_bytes != buffer_srv_desc_bytes) return false;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv;
        std::memcpy(&srv, desc, sizeof(srv));
        if (srv.ViewDimension != D3D12_SRV_DIMENSION_BUFFER || srv.Buffer.FirstElement > 0xffffffffull ||
            static_cast<std::uint32_t>(srv.Format) >= (1u << 10) || srv.Shader4ComponentMapping >= (1u << 13) ||
            srv.Buffer.StructureByteStride >= (1u << 24) ||
            (srv.Buffer.Flags != D3D12_BUFFER_SRV_FLAG_NONE && srv.Buffer.Flags != D3D12_BUFFER_SRV_FLAG_RAW))
            return false;
        out[1] |= static_cast<std::uint32_t>(srv.Format) | (srv.Buffer.Flags == D3D12_BUFFER_SRV_FLAG_RAW ? 1u << 10 : 0u) |
                  srv.Shader4ComponentMapping << 11;
        out[2] |= srv.Buffer.StructureByteStride << 8;
        out[4] = static_cast<std::uint32_t>(resource);
        out[5] = static_cast<std::uint32_t>(resource >> 32);
        out[6] = static_cast<std::uint32_t>(srv.Buffer.FirstElement);
        out[7] = srv.Buffer.NumElements;
    } else {
        return false;
    }
    std::memcpy(token, out, sizeof(out));
    return true;
}

// The view a content token names, with the description's padding bytes zero. Word 3 is not read. False for an id token
// or a word 2 that names no content kind.
inline bool decode_view_content(const std::uint32_t* words, ContentView& view)
{
    if (!is_view_content(words)) return false;
    view = ContentView{};
    view.slot = words[2] & 0xffu;
    view.created = (words[1] & view_content_created) != 0;
    const std::uint64_t pointer = words[4] | static_cast<std::uint64_t>(words[5]) << 32;
    if (view.slot == 17) {
        if (words[2] != 17) return false;
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{};
        cbv.BufferLocation = pointer;
        cbv.SizeInBytes = words[6];
        view.desc_bytes = static_cast<std::uint32_t>(cbv_desc_bytes);
        std::memcpy(view.desc, &cbv, sizeof(cbv));
        return true;
    }
    if (view.slot != 18) return false;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = static_cast<DXGI_FORMAT>(words[1] & 0x3ffu);
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = (words[1] >> 11) & 0x1fffu;
    srv.Buffer.FirstElement = words[6];
    srv.Buffer.NumElements = words[7];
    srv.Buffer.StructureByteStride = words[2] >> 8;
    srv.Buffer.Flags = (words[1] & (1u << 10)) ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
    view.resource = pointer;
    view.desc_bytes = static_cast<std::uint32_t>(buffer_srv_desc_bytes);
    std::memcpy(view.desc, &srv, sizeof(srv));
    return true;
}

}  // namespace x2w_startup
