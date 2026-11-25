#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "desktop_gpu.h"
#include "startup_indirect_mesh_mesh.h"
#include "startup_indirect_mesh_pixel.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
using Microsoft::WRL::ComPtr;
using Words = std::array<std::uint32_t, 7>;
constexpr UINT width = 4, height = 2;
constexpr auto format = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr std::uint32_t black = 0xff000000, red = 0xff0000ff, green = 0xff00ff00;
constexpr Words initial{0xa5a5a5a5, 0, 0, 0, 0, 0, 0x5a5a5a5a};
constexpr std::array<std::uint16_t, 10> indices{99, 99, 99, 0, 1, 2, 3, 4, 5, 99};
constexpr std::array<std::uint32_t, 10> wide_indices{99, 99, 99, 0, 1, 2, 3, 4, 5, 99};
constexpr UINT argument_offset = sizeof(std::uint32_t);
constexpr UINT64 replay_copy_offset = 0xbfffa0, replay_copy_bytes = 0x44;

constexpr char shader[] = R"(
RWByteAddressBuffer data : register(u0);

[numthreads(1, 1, 1)]
void write_left() {
    data.Store4(4, uint4(3, 1, 1, 0));
    data.Store(20, 0);
}

[numthreads(1, 1, 1)]
void write_right() {
    data.Store4(4, uint4(3, 1, 4, 0));
    data.Store(20, 0);
}

[numthreads(1, 1, 1)]
void write_one_triangle() {
    data.Store4(4, uint4(3, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_two_triangles() {
    data.Store4(4, uint4(6, 1, 0, 0));
}

[numthreads(1, 1, 1)]
void write_buffer_colour() {
    data.Store(16, 0xff00ff00u);
    data.Store(32, 0xff0000ffu);
}

[numthreads(1, 1, 1)]
void write_replay_colour() {
    data.Store(0, 0xff00ff00u);
    data.Store(4, 0xff00ff00u);
    data.Store4(0xbfffa0, uint4(0xff00ff00u, 0xff0000ffu, 0, 0));
    data.Store4(0xbfffb0, uint4(0, 0, 0, 0));
    data.Store4(0xbfffc0, uint4(0, 0, 0, 0));
    data.Store4(0xbfffd0, uint4(0, 0, 0, 0));
    data.Store(0xbfffe0, 0);
}

[numthreads(1, 1, 1)]
void write_one_mesh_group() {
    data.Store3(4, uint3(1, 1, 1));
}

[numthreads(1, 1, 1)]
void write_two_mesh_groups() {
    data.Store3(4, uint3(2, 1, 1));
}

struct Varying {
    float4 position : SV_Position;
    nointerpolation uint side : SIDE;
};

Varying vertex(uint id : SV_VertexID) {
    Varying result;
    uint corner = id % 3;
    result.position = float4(corner == 2 ? 3 : -1, corner == 1 ? 3 : -1, 0, 1);
    result.side = id / 3;
    return result;
}

float4 pixel(Varying input) : SV_Target {
    if (input.side == 0 && input.position.x < 2) return float4(1, 0, 0, 1);
    if (input.side == 1 && input.position.x >= 2) return float4(0, 1, 0, 1);
    discard;
    return float4(0, 0, 0, 1);
}
cbuffer entry : register(b0) { uint ud0; uint ud1; uint ud2; };
Texture2D<float4> translated_texture : register(t0);
ByteAddressBuffer translated_buffer : register(t0);

float4 translated_vertex(uint id : SV_VertexID) : SV_Position {
    return float4(id == 2 ? 3 : -1, id == 1 ? 3 : -1, 0, 1);
}

float4 translated_vertex_with_start(uint id : SV_VertexID) : SV_Position {
    return translated_vertex(id + ud1);
}

float4 translated_pixel() : SV_Target {
    return translated_texture.Load(int3(0, 0, 0));
}

float4 translated_buffer_pixel() : SV_Target {
    uint value = translated_buffer.Load(ud2);
    return float4(value & 255, (value >> 8) & 255, (value >> 16) & 255, 255) / 255.0;
}

)";

template <typename Method>
Method slot(IUnknown* object, unsigned index)
{
    return reinterpret_cast<Method>((*reinterpret_cast<void***>(object))[index]);
}

template <typename T>
ComPtr<T> native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* object)
{
    // The guest forwards SetPrivateDataInterface with the underlying desktop argument.
    constexpr GUID key{0x18097577, 0x0a4a, 0x427c, {0xac, 0xd5, 0xac, 0x1d, 0x98, 0x90, 0x65, 0x85}};
    check(guest->SetPrivateDataInterface(key, object), "publish underlying object");
    ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "clear underlying object");
    check(read, "retrieve underlying object");
    ComPtr<T> result;
    check(stored.As(&result), "query underlying object");
    return result;
}

ComPtr<ID3DBlob> compile(const char* entry, const char* profile)
{
    ComPtr<ID3DBlob> code, errors;
    const HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, nullptr, nullptr, nullptr, entry, profile, 0, 0,
                                  &code, &errors);
    if (FAILED(hr) && errors) std::fprintf(stderr, "%s\n", static_cast<const char*>(errors->GetBufferPointer()));
    check(hr, entry);
    return code;
}

ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device, const D3D12_ROOT_SIGNATURE_DESC& desc,
                                       ComPtr<ID3DBlob>* serialized = nullptr)
{
    ComPtr<ID3DBlob> code;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &code, nullptr), "serialize root");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, code->GetBufferPointer(), code->GetBufferSize(), IID_PPV_ARGS(&root)),
          "create root");
    if (serialized) *serialized = code;
    return root;
}

ComPtr<IUnknown> translated_pipeline(ID3D12Device* guest, ID3D12RootSignature* root, ID3DBlob* root_blob,
                                    bool needs_start, bool table_buffer = false)
{
    // The opaque guest programs identify synthetic cache entries. The actual bound shaders are source-built DXBC.
    constexpr char revision[] = "x2w-startup-shader-8";
    static std::uint32_t vertex_ids[]{0xe0000101, 0xe0000102};
    static std::uint32_t pixel_ids[]{0xe0000202, 0xe0000203};
    auto& vertex_id = vertex_ids[needs_start];
    auto& pixel_id = pixel_ids[table_buffer];
    std::uint64_t key = 1469598103934665603ull;
    const auto add = [&key](const void* data, std::size_t bytes) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < bytes; ++i) key = (key ^ p[i]) * 1099511628211ull;
    };
    add(revision, sizeof(revision) - 1);
    add(&vertex_id, sizeof(vertex_id));
    add(&pixel_id, sizeof(pixel_id));
    add(root_blob->GetBufferPointer(), root_blob->GetBufferSize());
    char stem[32];
    std::snprintf(stem, sizeof(stem), "%016llx", static_cast<unsigned long long>(key));
    char directory[MAX_PATH];
    const DWORD size = GetEnvironmentVariableA("X2W_SHADER_CACHE", directory, MAX_PATH);
    check(size && size < MAX_PATH ? S_OK : E_FAIL, "find fixture cache");
    const std::string prefix = std::string(directory) + "\\" + stem;
    const auto write = [&](const char* suffix, const void* bytes, std::size_t count) {
        std::ofstream file(prefix + suffix, std::ios::binary | std::ios::trunc);
        file.write(static_cast<const char*>(bytes), count);
        check(file.good() ? S_OK : E_FAIL, "write fixture cache");
    };
    const auto vs = compile(needs_start ? "translated_vertex_with_start" : "translated_vertex", "vs_5_1");
    const auto ps = compile(table_buffer ? "translated_buffer_pixel" : "translated_pixel", "ps_5_1");
    write(".vs.cso", vs->GetBufferPointer(), vs->GetBufferSize());
    write(".ps.cso", ps->GetBufferPointer(), ps->GetBufferSize());
    constexpr char manifest[] = "x2w-manifest 1\nud 2\nudsrc 0 rect\nudsrc 1 basevertex\nsrv 0 tex2d table 0 0\nparams 0\ntargets 0\n";
    constexpr char buffer_manifest[] = "x2w-manifest 1\nud 3\nudsrc 0 rect\nudsrc 1 basevertex\nudsrc 2 stride 0\nsrv 0 buf table 0 0\nparams 0\ntargets 0\n";
    if (table_buffer) write(".txt", buffer_manifest, sizeof(buffer_manifest) - 1);
    else write(".txt", manifest, sizeof(manifest) - 1);
    const std::array<UINT64, 0x300 / sizeof(UINT64)> desc{
        reinterpret_cast<UINT64>(root), 0, 0, 0, 0,
        reinterpret_cast<UINT64>(&vertex_id), sizeof(vertex_id), 0, 0,
        reinterpret_cast<UINT64>(&pixel_id), sizeof(pixel_id), 0};
    using Create = HRESULT (WINAPI*)(IUnknown*, const void*, REFIID, void**);
    ComPtr<IUnknown> result;
    check(slot<Create>(guest, 57)(guest, desc.data(), __uuidof(ID3D12PipelineState),
        reinterpret_cast<void**>(result.GetAddressOf())), "create translated guest pipeline");
    return result;
}

ComPtr<ID3D12PipelineState> writer_pipeline(ID3D12Device* device, ID3D12RootSignature* root, const char* entry)
{
    const auto cs = compile(entry, "cs_5_0");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root;
    desc.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
    ComPtr<ID3D12PipelineState> result;
    check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&result)), "create writer pipeline");
    return result;
}

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Kind;
    T value{};
};

template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Stage>
ComPtr<IUnknown> guest_pipeline(ID3D12Device* guest, ID3D12RootSignature* root,
                               D3D12_SHADER_BYTECODE source, D3D12_SHADER_BYTECODE pixel)
{
    struct {
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
        Subobject<Stage, D3D12_SHADER_BYTECODE> source;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> targets;
        Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, DXGI_SAMPLE_DESC> samples;
    } stream;
    static_assert(sizeof(stream) == 16 + 24 + 24 + 48 + 56 + 8 + 40 + 16);
    stream.root.value = root;
    stream.source.value = source;
    stream.ps.value = pixel;
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.depth.value.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    stream.depth.value.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    stream.targets.value.RTFormats[0] = format;
    stream.targets.value.NumRenderTargets = 1;
    stream.samples.value.Count = 1;
    const D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    using Create = HRESULT (WINAPI*)(IUnknown*, const D3D12_PIPELINE_STATE_STREAM_DESC*, REFIID, void**);
    ComPtr<IUnknown> result;
    check(slot<Create>(guest, 86)(guest, &desc, __uuidof(ID3D12PipelineState),
                                 reinterpret_cast<void**>(result.GetAddressOf())), "create guest graphics pipeline");
    return result;
}

struct FreeMemory {
    void operator()(void* pointer) const { VirtualFree(pointer, 0, MEM_RELEASE); }
};

struct PlacedBuffer {
    PlacedBuffer(ID3D12Device* device, ID3D12Device* guest, UINT width, D3D12_RESOURCE_FLAGS flags)
        : backing(VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
    {
        if (!backing) throw std::bad_alloc();
        // A deliberately unaligned placed address selects the runtime's committed stand-in.
        // Its GPU data is independent of the zero-filled guest virtual address.
        address = static_cast<char*>(backing.get()) + 4;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = width;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = flags;
        using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
        check(slot<Place>(guest, 133)(guest, address, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, __uuidof(ID3D12Resource),
                                      reinterpret_cast<void**>(wrapped.GetAddressOf())), "place guest buffer");
        native = native_object<ID3D12Resource>(device, guest, wrapped.Get());
    }

    std::unique_ptr<void, FreeMemory> backing;
    void* address{};
    ComPtr<IUnknown> wrapped;
    ComPtr<ID3D12Resource> native;
};

ComPtr<IUnknown> begin_guest_list(ID3D12Device* guest, ID3D12CommandAllocator* allocator)
{
    using CreateList = HRESULT (WINAPI*)(IUnknown*, const void*, IUnknown*, IUnknown*, REFIID, void**);
    const UINT list_desc[]{D3D12_COMMAND_LIST_TYPE_DIRECT, 2, 1, 0};
    ComPtr<IUnknown> result;
    check(slot<CreateList>(guest, 73)(guest, list_desc, allocator, nullptr, __uuidof(ID3D12GraphicsCommandList),
                                     reinterpret_cast<void**>(result.GetAddressOf())), "create guest command list");
    using Reset = HRESULT (WINAPI*)(IUnknown*, IUnknown*, void*, void*);
    check(slot<Reset>(result.Get(), 139)(result.Get(), allocator, nullptr, nullptr), "reset guest command list");
    return result;
}

void submit(ID3D12Device* device, ID3D12GraphicsCommandList* list)
{
    const D3D12_COMMAND_QUEUE_DESC desc{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12Fence> fence;
    check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "create queue");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "create completion fence");
    ID3D12CommandList* submitted[]{list};
    queue->ExecuteCommandLists(1, submitted);
    check(queue->Signal(fence.Get(), 1), "signal completion fence");
    check(fence->SetEventOnCompletion(1, nullptr), "wait for completion");
    check(device->GetDeviceRemovedReason(), "GPU completion");
}

enum class Refusal { none, invalid_index, short_argument, missing_viewport, wrong_root, missing_texture, start_vertex,
                     insufficient_table_range, copied_table_buffer, upload_range_crossing, unproven_copy };
enum class DrawKind { indexed, vertices, mesh };

ComPtr<ID3D12Resource> make_gpu_table_buffer(ID3D12Device* guest, UINT64 bytes)
{
    const D3D12_HEAP_PROPERTIES props{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc{D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};
    using Create = HRESULT (WINAPI*)(IUnknown*, const D3D12_HEAP_PROPERTIES*, UINT, const D3D12_RESOURCE_DESC*,
        UINT, const void*, void*, UINT, const void*, REFIID, void**);
    ComPtr<ID3D12Resource> result;
    check(slot<Create>(guest, 127)(guest, &props, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, nullptr,
        0, nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(result.GetAddressOf())), "create GPU table buffer");
    return result;
}

struct TranslatedDrawFixture {
    TranslatedDrawFixture(ID3D12Device* guest, std::uint32_t colour, ID3D12Device* desktop = nullptr,
                          bool short_view = false, bool copied = false, bool replay = false)
        : wrong_root(create_root(guest, {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE})),
          backing(VirtualAlloc(nullptr, 65536, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE))
    {
        if (!backing) throw std::bad_alloc();
        if (desktop && !copied) {
            sampled = make_gpu_table_buffer(guest, 64);
        } else {
            const D3D12_RESOURCE_DESC desc = copied ?
                D3D12_RESOURCE_DESC{D3D12_RESOURCE_DIMENSION_BUFFER, 0, replay ? replay_copy_bytes : 64u, 1, 1, 1, DXGI_FORMAT_UNKNOWN, {1, 0},
                                    D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS} :
                D3D12_RESOURCE_DESC{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, 1, 1, 1, 1, format, {1, 0},
                                    D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE};
            auto* address = static_cast<unsigned char*>(backing.get()) + (copied ? 16 : 0);
            std::memcpy(address + (replay ? 4 : copied ? 32 : 0), &colour, sizeof(colour));
            using Place = HRESULT (WINAPI*)(IUnknown*, void*, const D3D12_RESOURCE_DESC*, UINT, const void*, REFIID, void**);
            check(slot<Place>(guest, 133)(guest, address, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                __uuidof(ID3D12Resource), reinterpret_cast<void**>(sampled.GetAddressOf())), "place sampled resource");
        }
        if (desktop) gpu_buffer = native_object<ID3D12Resource>(desktop, guest, sampled.Get());
        const struct {
            D3D12_DESCRIPTOR_HEAP_DESC desc;
            UINT64 backing, bytes;
        } heap_desc{{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE}, 0, 0};
        using CreateHeap = HRESULT (WINAPI*)(IUnknown*, const void*, REFIID, void**);
        check(slot<CreateHeap>(guest, 14)(guest, &heap_desc, __uuidof(ID3D12DescriptorHeap),
            reinterpret_cast<void**>(heap.GetAddressOf())), "create sampled texture heap");
        struct {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            UINT64 address = 0, tail[2]{};
        } storage;
        auto& srv = storage.view;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (desktop) {
            srv.Format = DXGI_FORMAT_UNKNOWN;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Buffer.FirstElement = replay ? 0 : 1;
            srv.Buffer.NumElements = replay ? static_cast<UINT>(replay_copy_bytes / 4) : short_view ? 1 : 2;
            srv.Buffer.StructureByteStride = replay ? 4 : 16;
        } else {
            srv.Format = format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MipLevels = 1;
        }
        using Handle = UINT64 (WINAPI*)(IUnknown*, void*);
        cpu = slot<Handle>(heap.Get(), 9)(heap.Get(), nullptr);
        gpu = slot<Handle>(heap.Get(), 10)(heap.Get(), nullptr);
        check(cpu && gpu ? S_OK : E_FAIL, "get guest texture handles");
        using View = void (WINAPI*)(IUnknown*, ID3D12Resource*, const D3D12_SHADER_RESOURCE_VIEW_DESC*, UINT64, UINT64);
        slot<View>(guest, 18)(guest, sampled.Get(), &srv, cpu, 0);
    }

    ComPtr<ID3D12RootSignature> wrong_root;
    std::unique_ptr<void, FreeMemory> backing;
    ComPtr<ID3D12Resource> sampled;
    ComPtr<ID3D12Resource> gpu_buffer;
    ComPtr<ID3D12PipelineState> buffer_writer;
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT64 cpu = 0;
    UINT64 gpu = 0;
};

struct ReplayCopy {
    ID3D12Resource* producer;
    ID3D12Resource* native;
    ID3D12PipelineState* writer;
    UINT64 bytes;
    bool uncertified_overwrite = false;
};

constexpr UINT command_stride(DrawKind kind)
{
    switch (kind) {
    case DrawKind::indexed: return sizeof(D3D12_DRAW_INDEXED_ARGUMENTS);
    case DrawKind::vertices: return sizeof(D3D12_DRAW_ARGUMENTS);
    case DrawKind::mesh: return sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
    }
    return 0;
}

bool supports_mesh(ID3D12Device* device)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options))) &&
        options.MeshShaderTier >= D3D12_MESH_SHADER_TIER_1;
}

bool run(ID3D12Device* device, ID3D12Device* guest, ID3D12RootSignature* writer_root, ID3D12PipelineState* writer,
         ID3D12RootSignature* graphics_root, IUnknown* graphics, ID3D12CommandSignature* signature,
         DrawKind kind, bool draw_right, Refusal refusal, const char* label,
         const TranslatedDrawFixture* translated = nullptr, bool upload_table = false, const ReplayCopy* replay = nullptr)
{
    const bool indexed = kind == DrawKind::indexed;
    const bool mesh = kind == DrawKind::mesh;
    const UINT index_bytes = indexed ? (draw_right ? sizeof(wide_indices) : sizeof(indices)) : 0;
    const UINT index_offset = 2 * (draw_right ? sizeof(std::uint32_t) : sizeof(std::uint16_t));
    const void* index_data = draw_right ? static_cast<const void*>(wide_indices.data()) : indices.data();
    Words initial_words = initial;
    if (!indexed) initial_words[1 + command_stride(kind) / sizeof(std::uint32_t)] = 0x3c3c3c3c;
    const PlacedBuffer arguments(device, guest, sizeof(initial_words), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    std::optional<PlacedBuffer> index_buffer;
    if (indexed) index_buffer.emplace(device, guest, index_bytes, D3D12_RESOURCE_FLAG_NONE);
    const auto upload = make_buffer(device, sizeof(initial_words) + index_bytes, D3D12_HEAP_TYPE_UPLOAD,
                                    D3D12_RESOURCE_STATE_GENERIC_READ);
    const auto arg_readback = make_buffer(device, sizeof(initial), D3D12_HEAP_TYPE_READBACK,
                                          D3D12_RESOURCE_STATE_COPY_DEST);
    void* memory = nullptr;
    const D3D12_RANGE untouched{0, 0};
    check(upload->Map(0, &untouched, &memory), "map upload");
    std::memcpy(memory, initial_words.data(), sizeof(initial_words));
    if (indexed) std::memcpy(static_cast<char*>(memory) + sizeof(initial_words), index_data, index_bytes);
    upload->Unmap(0, nullptr);

    const D3D12_RESOURCE_DESC target_desc{D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1, format, {1, 0},
                                          D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    constexpr float clear[]{0, 0, 0, 1};
    const D3D12_CLEAR_VALUE clear_value{format, {0, 0, 0, 1}};
    const auto target = resource(device, target_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &clear_value);
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
                                              D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "create RTV heap");
    const auto rtv = heap->GetCPUDescriptorHandleForHeapStart();
    guest->CreateRenderTargetView(target.Get(), nullptr, rtv);

    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    const auto wrapped_list = begin_guest_list(guest, allocator.Get());
    auto* wrapped = wrapped_list.Get();
    const auto native_list = native_object<ID3D12GraphicsCommandList>(device, guest, wrapped);
    auto* list = native_list.Get();
    if (upload_table) {
        using Allocate = void* (WINAPI*)(IUnknown*, UINT64, UINT64);
        void* allocation = slot<Allocate>(wrapped, 148)(wrapped, 32, 16);
        if (!allocation) return false;
        const std::array<std::uint32_t, 8> words{green, 0, 0, 0, red, 0, 0, 0};
        std::memcpy(allocation, words.data(), sizeof(words));
        const UINT64 address = reinterpret_cast<UINT64>(allocation);
        const std::array<std::uint32_t, 4> view{static_cast<std::uint32_t>(address),
            static_cast<std::uint32_t>(address >> 32) | (16u << 16),
            refusal == Refusal::upload_range_crossing ? 3u : 2u, 0};
        std::memcpy(reinterpret_cast<void*>(translated->cpu), view.data(), sizeof(view));
    }
    if (index_buffer) {
        transition(list, index_buffer->native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(index_buffer->native.Get(), 0, upload.Get(), sizeof(initial_words), index_bytes);
        transition(list, index_buffer->native.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    }
    transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(arguments.native.Get(), 0, upload.Get(), 0, sizeof(initial));
    auto argument_state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (writer) {
        transition(list, arguments.native.Get(), argument_state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        argument_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        list->SetPipelineState(writer);
        list->SetComputeRootSignature(writer_root);
        list->SetComputeRootUnorderedAccessView(0, arguments.native->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
    }
    transition(list, arguments.native.Get(), argument_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(arg_readback.Get(), 0, arguments.native.Get(), 0, sizeof(initial));
    // The wrapper owns the subsequent COMMON -> INDIRECT_ARGUMENT transition.
    transition(list, arguments.native.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    if (replay) {
        transition(list, replay->native, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(replay->writer);
        list->SetComputeRootSignature(writer_root);
        list->SetComputeRootUnorderedAccessView(0, replay->native->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        transition(list, replay->native, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        using Copy = void (WINAPI*)(IUnknown*, IUnknown*, UINT64, IUnknown*, UINT64, UINT64);
        slot<Copy>(wrapped, 15)(wrapped, translated->sampled.Get(), 0, replay->producer, replay_copy_offset, replay->bytes);
        if (replay->uncertified_overwrite)
            slot<Copy>(wrapped, 15)(wrapped, translated->sampled.Get(), 0, replay->native, 0, 4);
    }
    if (translated && translated->gpu_buffer && translated->buffer_writer) {
        auto* table_resource = translated->gpu_buffer.Get();
        transition(list, table_resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetPipelineState(translated->buffer_writer.Get());
        list->SetComputeRootSignature(writer_root);
        list->SetComputeRootUnorderedAccessView(0, table_resource->GetGPUVirtualAddress());
        list->Dispatch(1, 1, 1);
        transition(list, table_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    }

    using Bind = void (WINAPI*)(IUnknown*, IUnknown*);
    using Clear = void (WINAPI*)(IUnknown*, SIZE_T, const float*, UINT, const D3D12_RECT*);
    using Targets = void (WINAPI*)(IUnknown*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, BOOL,
                                  const D3D12_CPU_DESCRIPTOR_HANDLE*);
    using Topology = void (WINAPI*)(IUnknown*, D3D12_PRIMITIVE_TOPOLOGY);
    using Viewport = void (WINAPI*)(IUnknown*, UINT, const D3D12_VIEWPORT*);
    using Scissor = void (WINAPI*)(IUnknown*, UINT, const D3D12_RECT*);
    using Index = void (WINAPI*)(IUnknown*, const D3D12_INDEX_BUFFER_VIEW*);
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    const D3D12_VIEWPORT viewport{0, 0, width, height, 0, 1};
    const D3D12_RECT scissor{0, 0, width, height};
    slot<Clear>(wrapped, 52)(wrapped, rtv.ptr, clear, 0, nullptr);
    slot<Targets>(wrapped, 50)(wrapped, 1, &rtv, FALSE, nullptr);
    slot<Bind>(wrapped, 25)(wrapped, graphics);
    slot<Bind>(wrapped, 34)(wrapped, refusal == Refusal::wrong_root ? translated->wrong_root.Get() : graphics_root);
    if (translated && refusal != Refusal::missing_texture) {
        ID3D12DescriptorHeap* heaps[]{translated->heap.Get()};
        using Heaps = void (WINAPI*)(IUnknown*, UINT, ID3D12DescriptorHeap* const*);
        using Table = void (WINAPI*)(IUnknown*, UINT, UINT64);
        slot<Heaps>(wrapped, 32)(wrapped, 1, heaps);
        slot<Table>(wrapped, 36)(wrapped, 0, translated->gpu);
    }
    if (!mesh) slot<Topology>(wrapped, 20)(wrapped, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    if (refusal != Refusal::missing_viewport) slot<Viewport>(wrapped, 21)(wrapped, 1, &viewport);
    slot<Scissor>(wrapped, 22)(wrapped, 1, &scissor);
    if (index_buffer) {
        const auto address = reinterpret_cast<UINT64>(index_buffer->address);
        const auto start = refusal == Refusal::invalid_index ? index_bytes : index_offset;
        const D3D12_INDEX_BUFFER_VIEW view{address + start, index_bytes - index_offset,
                                         draw_right ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT};
        slot<Index>(wrapped, 47)(wrapped, &view);
    }
    const UINT64 offset = refusal == Refusal::short_argument
        ? sizeof(initial_words) - command_stride(kind) + sizeof(std::uint32_t) : argument_offset;
    slot<Execute>(wrapped, 61)(wrapped, signature, 1, arguments.wrapped.Get(), offset, nullptr, 0);
    check(slot<HRESULT (WINAPI*)(IUnknown*)>(wrapped, 9)(wrapped), "close guest command list");
    submit(device, list);

    if (translated) {
        constexpr Words empty{};
        if (std::memcmp(arguments.address, empty.data(), sizeof(empty))) {
            std::fprintf(stderr, "%s: CPU argument backing changed\n", label);
            return false;
        }
    }

    const D3D12_RANGE arg_range{0, sizeof(initial)};
    check(arg_readback->Map(0, &arg_range, &memory), "map argument readback");
    Words observed{};
    std::memcpy(observed.data(), memory, sizeof(observed));
    arg_readback->Unmap(0, &untouched);
    Words expected = initial_words;
    if (writer) {
        expected[1] = !translated && !indexed && draw_right ? 6 : 3;
        expected[2] = 1;
        if (indexed) expected[3] = draw_right ? 4 : 1;
        if (mesh) {
            expected[1] = draw_right ? 2 : 1;
            expected[3] = 1;
        }
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (observed[i] == expected[i]) continue;
        std::fprintf(stderr, "%s: argument word %zu: got 0x%08x, expected 0x%08x\n", label, i, observed[i], expected[i]);
        return false;
    }

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
    const auto pixel_readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    ComputeCommands copy(device, nullptr, D3D12_COMMAND_LIST_TYPE_DIRECT);
    transition(copy.list.Get(), target.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION source{target.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION destination{pixel_readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    destination.PlacedFootprint = footprint;
    copy.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    copy.submit_and_wait(device);

    const D3D12_RANGE pixel_range{0, static_cast<SIZE_T>(bytes)};
    check(pixel_readback->Map(0, &pixel_range, &memory), "map pixel readback");
    UINT painted = 0;
    bool pixels_match = true;
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x) {
            std::uint32_t actual = 0;
            std::memcpy(&actual, static_cast<char*>(memory) + y * footprint.Footprint.RowPitch + x * sizeof(actual),
                        sizeof(actual));
            const bool right_half = x >= width / 2;
            const bool covered = translated || (indexed ? right_half == draw_right : !right_half || draw_right);
            const bool drawn = writer && refusal == Refusal::none && covered;
            const auto wanted = drawn ? (translated ? (draw_right ? green : red) : (right_half ? green : red)) : black;
            if (actual != black) ++painted;
            if (actual == wanted) continue;
            std::fprintf(stderr, "%s: pixel (%u,%u): got 0x%08x, expected 0x%08x\n", label, x, y, actual, wanted);
            pixels_match = false;
        }
    pixel_readback->Unmap(0, &untouched);
    std::printf("%s: %u/%u non-clear pixels, argument guards intact\n", label, painted, width * height);
    return pixels_match;
}
// Calls refused before any argument is read: one per counted reason, and one with two commands, which is
// counted as other under its signature's kind. The list is closed and dropped.
void refuse_each_reason(ID3D12Device* guest, ID3D12CommandSignature* draw_indexed)
{
    D3D12_INDIRECT_ARGUMENT_DESC dispatch{};
    dispatch.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    const D3D12_COMMAND_SIGNATURE_DESC dispatch_desc{sizeof(D3D12_DISPATCH_ARGUMENTS), 1, &dispatch, 0};
    ComPtr<ID3D12CommandSignature> dispatch_signature;
    check(guest->CreateCommandSignature(&dispatch_desc, nullptr, IID_PPV_ARGS(&dispatch_signature)), "create guest dispatch signature");
    ComPtr<ID3D12CommandAllocator> allocator;
    check(guest->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "create guest allocator");
    const auto list = begin_guest_list(guest, allocator.Get());
    using Execute = void (WINAPI*)(IUnknown*, IUnknown*, UINT, IUnknown*, UINT64, IUnknown*, UINT64);
    const auto execute = slot<Execute>(list.Get(), 61);
    execute(list.Get(), nullptr, 1, nullptr, 0, nullptr, 0);                    // placeholder signature
    execute(list.Get(), draw_indexed, 1, nullptr, 0, nullptr, 0);               // no desktop graphics pipeline bound
    execute(list.Get(), dispatch_signature.Get(), 1, nullptr, 0, nullptr, 0);   // no desktop compute pipeline bound
    execute(list.Get(), draw_indexed, 2, nullptr, 0, nullptr, 0);               // two commands
    check(slot<HRESULT (WINAPI*)(IUnknown*)>(list.Get(), 9)(list.Get()), "close refusal list");
}

// 300 empty console presents (queue slot 32) reach the render-totals heartbeat; nothing is shown.
std::string indirect_line(ID3D12Device* guest, const std::wstring& log_path)
{
    const D3D12_COMMAND_QUEUE_DESC desc{};
    ComPtr<IUnknown> queue;
    check(guest->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(queue.GetAddressOf())),
          "create guest queue");
    using Present = HRESULT (WINAPI*)(IUnknown*, UINT, const void*, const void*);
    const std::array<unsigned char, 0x50> planes{};
    const std::array<unsigned char, 0x40> present{};
    for (int frame = 0; frame < 300; ++frame) slot<Present>(queue.Get(), 32)(queue.Get(), 0, planes.data(), present.data());
    std::ifstream file(log_path, std::ios::binary);
    std::string line, found;
    while (std::getline(file, line))
        if (line.rfind("x2w-render indirect:", 0) == 0) found = line;
    while (!found.empty() && (found.back() == '\r' || found.back() == '\n')) found.pop_back();
    return found;
}
}  // namespace

int main(int argc, char** argv) try
{
    DrawKind kind = DrawKind::indexed;
    const bool copy_on = argc == 3 && std::strcmp(argv[2], "copy-provenance") == 0;
    const bool copy_off = argc == 3 && std::strcmp(argv[2], "copy-provenance-off") == 0;
    const bool translated = copy_on || copy_off || (argc == 3 && std::strcmp(argv[2], "translated") == 0);
    // "count-control" is the negative control: it expects one forwarded call too many.
    const bool count_control = argc == 3 && std::strcmp(argv[2], "count-control") == 0;
    if (translated || (argc == 3 && std::strcmp(argv[2], "draw") == 0)) kind = DrawKind::vertices;
    else if (argc == 3 && std::strcmp(argv[2], "mesh") == 0) kind = DrawKind::mesh;
    else if (argc != 2 && !count_control) return 2;
    const bool indexed = kind == DrawKind::indexed;
    const bool mesh = kind == DrawKind::mesh;
    const bool warp = std::strcmp(argv[1], "warp") == 0;
    if (!warp && std::strcmp(argv[1], "hardware") != 0) return 2;
    ComPtr<ID3D12Debug> debug;
    const HRESULT debug_hr = D3D12GetDebugInterface(IID_PPV_ARGS(&debug));
    if (debug_hr != DXGI_ERROR_SDK_COMPONENT_MISSING) check(debug_hr, "get debug layer");
    if (debug) debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
    ComPtr<IDXGIAdapter1> adapter;
    if (warp) {
        check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "find WARP");
    } else {
        for (UINT index = 0;; ++index) {
            const HRESULT hr = factory->EnumAdapters1(index, adapter.ReleaseAndGetAddressOf());
            if (hr == DXGI_ERROR_NOT_FOUND) {
                std::puts("indirect graphics capability skip: no compatible physical adapter");
                return 77;
            }
            check(hr, "enumerate physical adapters");
            DXGI_ADAPTER_DESC1 desc{};
            check(adapter->GetDesc1(&desc), "get physical adapter description");
            const bool basic_render_driver = desc.VendorId == 0x1414 && desc.DeviceId == 0x008c;
            if ((desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) || basic_render_driver) continue;
            ComPtr<ID3D12Device> probe;
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&probe))) &&
                (!mesh || supports_mesh(probe.Get()))) break;
        }
    }
    wchar_t directory[MAX_PATH];
    if (!GetTempPathW(MAX_PATH, directory)) return 4;
    const std::wstring log_path = std::wstring(directory) + L"x2w_indirect_counts_" + std::to_wstring(GetCurrentProcessId()) + L".log";
    if (indexed) {
        DeleteFileW(log_path.c_str());
        SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
        SetEnvironmentVariableA("X2W_PRESENT_SIGNATURE_EVERY", "0");
    }
    DXGI_ADAPTER_DESC1 desc{};
    check(adapter->GetDesc1(&desc), "get selected adapter description");
    std::printf("adapter: %ls, debug layer %s\n", desc.Description, debug ? "enabled" : "unavailable");
    LARGE_INTEGER driver{};
    if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver)))
        std::printf("driver: %u.%u.%u.%u\n", HIWORD(driver.HighPart), LOWORD(driver.HighPart),
                    HIWORD(driver.LowPart), LOWORD(driver.LowPart));

    ComPtr<ID3D12Device> device, guest;
    check(D3D12XboxCreateDevice(adapter.Get(), nullptr, IID_PPV_ARGS(&device)), "create desktop device through d3d12_xs");
    if (mesh && !supports_mesh(device.Get())) {
        std::puts("indirect mesh capability skip: mesh shaders unavailable");
        return 77;
    }
    check(SetEnvironmentVariableA("X2W_D3D12_PERMISSIVE", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable guest");
    check(SetEnvironmentVariableA("X2W_D3D12_LIST_NOOP", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable guest list");
    check(SetEnvironmentVariableA("X2W_D3D12_RENDER", "1") ? S_OK : HRESULT_FROM_WIN32(GetLastError()), "enable recording");
    const std::array<unsigned char, 64> parameters{};
    check(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), IID_PPV_ARGS(&guest)), "create guest device");
    ComPtr<ID3D12InfoQueue> info;
    if (debug) check(device.As(&info), "get debug message queue");
    const D3D12_ROOT_PARAMETER uav{D3D12_ROOT_PARAMETER_TYPE_UAV, {}, D3D12_SHADER_VISIBILITY_ALL};
    const D3D12_ROOT_SIGNATURE_DESC writer_desc{1, &uav, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto root_flags = mesh ? D3D12_ROOT_SIGNATURE_FLAG_NONE : D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    const D3D12_ROOT_SIGNATURE_DESC graphics_desc{0, nullptr, 0, nullptr, root_flags};
    const auto writer = create_root(device.Get(), writer_desc);
    if (translated) {
        const auto cache = std::filesystem::current_path() / ("indirect-translated-" + std::to_string(GetCurrentProcessId()));
        std::filesystem::create_directory(cache);
        check(SetEnvironmentVariableW(L"X2W_SHADER_CACHE", cache.c_str()) ? S_OK : E_FAIL, "set fixture cache");
        check(SetEnvironmentVariableA("X2W_TRANSLATED_INDIRECT_DRAW", "1") ? S_OK : E_FAIL, "enable fixture path");
        check(SetEnvironmentVariableA("X2W_TRANSLATED_INDIRECT_COPY_RANGES", copy_on ? "1" : "0") ? S_OK : E_FAIL,
              "set copy-range provenance control");
        const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER table{};
        table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        table.DescriptorTable = {1, &range};
        const D3D12_ROOT_SIGNATURE_DESC title_desc{1, &table, 0, nullptr, root_flags};
        ComPtr<ID3DBlob> root_blob;
        const auto title_root = create_root(guest.Get(), title_desc, &root_blob);
        const auto pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), false);
        const auto start_pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), true);
        const auto buffer_pso = translated_pipeline(guest.Get(), title_root.Get(), root_blob.Get(), false, true);
        const auto producer = writer_pipeline(device.Get(), writer.Get(), "write_one_triangle");
        const auto buffer_writer = writer_pipeline(device.Get(), writer.Get(), "write_buffer_colour");
        D3D12_INDIRECT_ARGUMENT_DESC argument{};
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
        const D3D12_COMMAND_SIGNATURE_DESC signature_desc{sizeof(D3D12_DRAW_ARGUMENTS), 1, &argument, 0};
        ComPtr<ID3D12CommandSignature> signature;
        check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create translated draw signature");
        const TranslatedDrawFixture red_binding(guest.Get(), red), green_binding(guest.Get(), green);
        TranslatedDrawFixture full_buffer(guest.Get(), 0, device.Get()), short_buffer(guest.Get(), 0, device.Get(), true),
            copied_buffer(guest.Get(), green, device.Get(), false, true);
        const TranslatedDrawFixture raw_upload(guest.Get(), 0);
        full_buffer.buffer_writer = short_buffer.buffer_writer = copied_buffer.buffer_writer = buffer_writer;
        const auto verify = [&](ID3D12PipelineState* gpu_writer, bool use_green, Refusal control, const char* label) {
            return run(device.Get(), guest.Get(), writer.Get(), gpu_writer, title_root.Get(),
                control == Refusal::start_vertex ? start_pso.Get() : pso.Get(), signature.Get(),
                DrawKind::vertices, use_green, control, label, use_green ? &green_binding : &red_binding);
        };
        bool ok = verify(producer.Get(), false, Refusal::none, "translated red texture") &&
            verify(producer.Get(), true, Refusal::none, "translated green texture") &&
            verify(nullptr, false, Refusal::none, "translated omitted GPU writer") &&
            verify(producer.Get(), false, Refusal::wrong_root, "translated wrong guest root") &&
            verify(producer.Get(), false, Refusal::short_argument, "translated short argument range") &&
            verify(producer.Get(), false, Refusal::missing_texture, "translated missing texture") &&
            verify(producer.Get(), false, Refusal::start_vertex, "translated uses start vertex") &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::none, "translated GPU buffer range with nonzero offset and stride", &full_buffer) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::insufficient_table_range, "translated table exposes only its own range", &short_buffer) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::copied_table_buffer, "translated refuses copied GPU buffer and CPU shadow", &copied_buffer) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::none, "translated raw table uses its recording upload page", &raw_upload, true) &&
            run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::upload_range_crossing, "translated raw table refuses crossing its upload allocation", &raw_upload, true);
        auto deferred_desc = title_desc;
        deferred_desc.Flags |= D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED;
        ComPtr<ID3DBlob> deferred_blob;
        ComPtr<ID3D12RootSignature> deferred_root;
        if (SUCCEEDED(D3D12SerializeRootSignature(&deferred_desc, D3D_ROOT_SIGNATURE_VERSION_1, &deferred_blob, nullptr)) &&
            SUCCEEDED(guest->CreateRootSignature(0, deferred_blob->GetBufferPointer(), deferred_blob->GetBufferSize(),
                                                 IID_PPV_ARGS(&deferred_root)))) {
            const auto deferred_pso = translated_pipeline(guest.Get(), deferred_root.Get(), deferred_blob.Get(), false, true);
            if (ok) ok = run(device.Get(), guest.Get(), writer.Get(), producer.Get(), deferred_root.Get(), deferred_pso.Get(), signature.Get(),
                DrawKind::vertices, false, Refusal::none, "translated table with deferred guest root", &full_buffer);
        } else {
            std::puts("deferred guest root capability unavailable");
        }
        if (ok && (copy_on || copy_off)) {
            const TranslatedDrawFixture copied(guest.Get(), green, device.Get(), false, true, true);
            const auto source = make_gpu_table_buffer(guest.Get(), replay_copy_offset + replay_copy_bytes);
            const auto native = native_object<ID3D12Resource>(device.Get(), guest.Get(), source.Get());
            const auto replay_writer = writer_pipeline(device.Get(), writer.Get(), "write_replay_colour");
            const ReplayCopy complete{source.Get(), native.Get(), replay_writer.Get(), replay_copy_bytes};
            const ReplayCopy partial{source.Get(), native.Get(), replay_writer.Get(), replay_copy_bytes - 4};
            const ReplayCopy overwritten{source.Get(), native.Get(), replay_writer.Get(), replay_copy_bytes, true};
            const auto draw = [&](const ReplayCopy* copy, Refusal refusal, const char* label) {
                return run(device.Get(), guest.Get(), writer.Get(), producer.Get(), title_root.Get(), buffer_pso.Get(), signature.Get(),
                    DrawKind::vertices, false, refusal, label, &copied, false, copy);
            };
            ok = draw(&complete, copy_on ? Refusal::none : Refusal::unproven_copy, "complete 68-byte GPU copy") &&
                 draw(nullptr, Refusal::unproven_copy, "new recording cannot reuse the previous GPU copy proof") &&
                 draw(&partial, Refusal::unproven_copy, "64-byte copy cannot authorize the 68-byte view") &&
                 draw(&overwritten, Refusal::unproven_copy, "untracked overlap invalidates the prior copy range");
            std::uint32_t title_word = 0;
            std::memcpy(&title_word, static_cast<unsigned char*>(copied.backing.get()) + 20, sizeof(title_word));
            if (title_word != green) {
                std::fprintf(stderr, "title-side green sentinel changed to %08x\n", title_word);
                return 1;
            }
        }
        if (ok) std::filesystem::remove_all(cache);
        return ok && (!info || !info->GetNumStoredMessages()) ? 0 : 1;
    }
    const auto graphics = create_root(guest.Get(), graphics_desc);
    const char* left_entry = indexed ? "write_left" : "write_one_triangle";
    const char* right_entry = indexed ? "write_right" : "write_two_triangles";
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = indexed ? D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    ComPtr<IUnknown> pso;
    if (mesh) {
        left_entry = "write_one_mesh_group";
        right_entry = "write_two_mesh_groups";
        argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS>(guest.Get(), graphics.Get(),
            {startup_indirect_mesh_mesh, sizeof(startup_indirect_mesh_mesh)},
            {startup_indirect_mesh_pixel, sizeof(startup_indirect_mesh_pixel)});
    } else {
        const auto vs = compile("vertex", "vs_5_0");
        const auto ps = compile("pixel", "ps_5_0");
        pso = guest_pipeline<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS>(guest.Get(), graphics.Get(),
            {vs->GetBufferPointer(), vs->GetBufferSize()}, {ps->GetBufferPointer(), ps->GetBufferSize()});
    }
    const auto left = writer_pipeline(device.Get(), writer.Get(), left_entry);
    const auto right = writer_pipeline(device.Get(), writer.Get(), right_entry);
    const D3D12_COMMAND_SIGNATURE_DESC signature_desc{command_stride(kind), 1, &argument, 0};
    ComPtr<ID3D12CommandSignature> signature;
    check(guest->CreateCommandSignature(&signature_desc, nullptr, IID_PPV_ARGS(&signature)), "create guest draw signature");

    const auto verify = [&](ID3D12PipelineState* producer, bool draw_right, Refusal refusal, const char* label) {
        return run(device.Get(), guest.Get(), writer.Get(), producer, graphics.Get(), pso.Get(), signature.Get(),
                   kind, draw_right, refusal, label);
    };
    const char* left_label = indexed ? "left writer" : "one triangle";
    const char* right_label = indexed ? "right writer" : "two triangles";
    if (mesh) {
        left_label = "one mesh group";
        right_label = "two mesh groups";
    }
    if (!verify(left.Get(), false, Refusal::none, left_label) ||
        !verify(right.Get(), true, Refusal::none, right_label) ||
        !verify(nullptr, false, Refusal::none, "omitted writer") ||
        !verify(left.Get(), false, Refusal::short_argument, "short argument range")) return 1;
    if (indexed ? !verify(left.Get(), false, Refusal::invalid_index, "invalid index range")
                : !verify(left.Get(), false, Refusal::missing_viewport, "missing viewport")) return 1;
    if (indexed) {
        // Forwarded: the three unrefused runs. Other reasons: the short argument range, the invalid index range and two commands.
        refuse_each_reason(guest.Get(), signature.Get());
        const std::string line = indirect_line(guest.Get(), log_path);
        DeleteFileW(log_path.c_str());
        const std::string expected = std::string("x2w-render indirect: forwarded: drawindexed ") + (count_control ? "4" : "3") +
            "; refused graphics-pipeline: drawindexed 1; compute-pipeline: dispatch 1; placeholder-signature: other 1; "
            "other: drawindexed 3";
        if (line != expected) {
            std::fprintf(stderr, "indirect line mismatch:\n  got      '%s'\n  expected '%s'\n", line.c_str(), expected.c_str());
            return 1;
        }
        std::printf("%s\n", line.c_str());
    }
    if (info) {
        for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
            SIZE_T bytes = 0;
            check(info->GetMessage(i, nullptr, &bytes), "get debug message size");
            std::vector<char> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            check(info->GetMessage(i, message, &bytes), "get debug message");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
                std::fprintf(stderr, "D3D12 debug message %u: %s\n", static_cast<unsigned>(message->ID), message->pDescription);
                return 1;
            }
        }
    }
    return 0;
}
catch (const std::exception&) {
    return 3;
}
