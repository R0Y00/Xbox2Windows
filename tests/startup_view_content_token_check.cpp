// Checks src/startup_stubs/view_content_token.h without a device: which views fit a content token, and that the
// decoded view equals the created one with its padding bytes zero.
#include "startup_stubs/view_content_token.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
using x2w_startup::ContentView;
constexpr std::uint32_t magic = 0x56573258;
int failures;

void fail(const char* name, const char* what)
{
    std::printf("FAIL %s: %s\n", name, what);
    ++failures;
}

template<class Desc>
void round_trip(const char* name, std::uint32_t slot, std::uint64_t resource, const Desc& desc, bool created)
{
    std::uint32_t token[8] = {};
    if (!x2w_startup::encode_view_content(magic, slot, resource, 0, reinterpret_cast<const unsigned char*>(&desc),
                                          sizeof(desc), created, token))
        return fail(name, "not encoded");
    if (token[0] != magic || !x2w_startup::is_view_content(token) || (token[2] & 0xffu) != slot || token[3])
        return fail(name, "token header");
    ContentView view;
    if (!x2w_startup::decode_view_content(token, view)) return fail(name, "not decoded");
    // The formatted-buffer readers write their own word 3 into the copy they decode.
    std::uint32_t scribbled[8];
    std::memcpy(scribbled, token, sizeof(scribbled));
    scribbled[3] = 0xdeadbeefu;
    ContentView again;
    if (!x2w_startup::decode_view_content(scribbled, again) || again.slot != view.slot || again.resource != view.resource ||
        again.created != view.created || again.desc_bytes != view.desc_bytes || std::memcmp(again.desc, view.desc, sizeof(view.desc)))
        return fail(name, "word 3 changes the decoded view");
    if (view.slot != slot || view.resource != resource || view.created != created || view.desc_bytes != sizeof(desc))
        return fail(name, "decoded fields");
    if (std::memcmp(view.desc, &desc, sizeof(desc))) return fail(name, "decoded description");
}

template<class Desc>
void refused(const char* name, std::uint32_t slot, std::uint64_t resource, std::uint64_t counter, const Desc& desc)
{
    std::uint32_t token[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    if (x2w_startup::encode_view_content(magic, slot, resource, counter, reinterpret_cast<const unsigned char*>(&desc),
                                         sizeof(desc), true, token))
        return fail(name, "encoded");
    if (token[0] != 1 || token[7] != 8) fail(name, "token changed");
}

D3D12_SHADER_RESOURCE_VIEW_DESC raw_srv(std::uint64_t first, UINT count)
{
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = first;
    srv.Buffer.NumElements = count;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    return srv;
}
}  // namespace

int main()
{
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{};
    cbv.BufferLocation = 0x00000123456789a0ull;
    cbv.SizeInBytes = 0x1100;
    round_trip("cbv", 17, 0, cbv, true);
    round_trip("cbv not created", 17, 0, cbv, false);

    round_trip("raw srv", 18, 0x000001f5249d7fe0ull, raw_srv(0xfffffff0u, 72), true);
    D3D12_SHADER_RESOURCE_VIEW_DESC structured{};
    structured.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    structured.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(3, 2, 1, 0);
    structured.Buffer.FirstElement = 17;
    structured.Buffer.NumElements = 4096;
    structured.Buffer.StructureByteStride = 48;
    round_trip("structured srv", 18, 0x7ff612340000ull, structured, true);
    D3D12_SHADER_RESOURCE_VIEW_DESC typed = structured;
    typed.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    typed.Buffer.StructureByteStride = 0;
    round_trip("typed srv", 18, 0x7ff612340000ull, typed, true);

    // Padding bytes do not survive: two views that differ only there get the same token.
    D3D12_SHADER_RESOURCE_VIEW_DESC padded = raw_srv(64, 16), clean = raw_srv(64, 16);
    std::memset(reinterpret_cast<unsigned char*>(&padded) + 12, 0xcd, 4);
    std::memset(reinterpret_cast<unsigned char*>(&padded) + 36, 0xcd, 4);
    std::uint32_t a[8], b[8];
    if (!x2w_startup::encode_view_content(magic, 18, 0x1000, 0, reinterpret_cast<const unsigned char*>(&padded), 40, true, a) ||
        !x2w_startup::encode_view_content(magic, 18, 0x1000, 0, reinterpret_cast<const unsigned char*>(&clean), 40, true, b) ||
        std::memcmp(a, b, sizeof(a)))
        fail("padding", "padded and clean views differ");
    // A view that differs in any kept field gets another token.
    const D3D12_SHADER_RESOURCE_VIEW_DESC other = raw_srv(65, 16);
    if (!x2w_startup::encode_view_content(magic, 18, 0x1000, 0, reinterpret_cast<const unsigned char*>(&other), 40, true, b) ||
        !std::memcmp(a, b, sizeof(a)))
        fail("first element", "different views share a token");

    refused("first element above 32 bits", 18, 0x1000, 0, raw_srv(0x100000000ull, 4));
    D3D12_SHADER_RESOURCE_VIEW_DESC texture{};
    texture.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    texture.Texture2D.MipLevels = 1;
    refused("texture srv", 18, 0x1000, 0, texture);
    refused("srv without resource", 18, 0, 0, raw_srv(0, 4));
    refused("srv with counter", 18, 0x1000, 0x2000, raw_srv(0, 4));
    D3D12_SHADER_RESOURCE_VIEW_DESC wide_format = typed;
    wide_format.Format = static_cast<DXGI_FORMAT>(1024);
    refused("format above 10 bits", 18, 0x1000, 0, wide_format);
    D3D12_SHADER_RESOURCE_VIEW_DESC wide_stride = structured;
    wide_stride.Buffer.StructureByteStride = 1u << 24;
    refused("stride above 24 bits", 18, 0x1000, 0, wide_stride);
    refused("cbv with resource", 17, 0x1000, 0, cbv);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    refused("uav", 19, 0x1000, 0, uav);
    D3D12_SAMPLER_DESC sampler{};
    refused("sampler", 22, 0, 0, sampler);
    std::uint32_t short_desc[8] = {};
    if (x2w_startup::encode_view_content(magic, 17, 0, 0, reinterpret_cast<const unsigned char*>(&cbv), 12, true, short_desc))
        fail("short cbv description", "encoded");

    // An id token is not a content token.
    const std::uint32_t id_token[8] = {magic, 0x7fffffffu, 18, 0, 0x1000, 0, 0, 0};
    ContentView view;
    if (x2w_startup::is_view_content(id_token) || x2w_startup::decode_view_content(id_token, view))
        fail("id token", "read as content");

    if (failures) return 1;
    std::printf("view content tokens: all checks passed\n");
    return 0;
}
