#pragma once

// Desktop graphics-PSO cache for translated startup pipelines.
//
// The cache is deliberately optional. It opens an ID3D12PipelineLibrary from a
// blob under X2W_SHADER_CACHE, loads a pipeline by its translation key, and
// falls back to CreateGraphicsPipelineState for every miss, corrupt blob,
// unsupported interface or failed store. A cache failure never fails a draw;
// the caller still receives the pipeline that D3D12 created directly.

#include <d3d12.h>
#include <dxgi.h>
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <new>

namespace x2w_startup {

namespace graphics_pipeline_cache_detail {

inline std::uint64_t fnv_bytes(const void* data, std::size_t bytes,
                               std::uint64_t hash = 1469598103934665603ull)
{
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ p[i]) * 1099511628211ull;
    return hash;
}

// Hash the semantic fields rather than the raw structures: D3D12_BLEND_DESC and
// D3D12_DEPTH_STENCIL_DESC contain ABI padding that a state block may leave
// uninitialized. Raw-byte hashing would turn identical states into different
// cache names across runs.
inline std::uint64_t hash_blend_key(const D3D12_BLEND_DESC& blend, std::uint64_t key)
{
    key = fnv_bytes(&blend.AlphaToCoverageEnable, sizeof(blend.AlphaToCoverageEnable), key);
    key = fnv_bytes(&blend.IndependentBlendEnable, sizeof(blend.IndependentBlendEnable), key);
    for (const auto& target : blend.RenderTarget) {
        key = fnv_bytes(&target.BlendEnable, sizeof(target.BlendEnable), key);
        key = fnv_bytes(&target.LogicOpEnable, sizeof(target.LogicOpEnable), key);
        key = fnv_bytes(&target.SrcBlend, sizeof(target.SrcBlend), key);
        key = fnv_bytes(&target.DestBlend, sizeof(target.DestBlend), key);
        key = fnv_bytes(&target.BlendOp, sizeof(target.BlendOp), key);
        key = fnv_bytes(&target.SrcBlendAlpha, sizeof(target.SrcBlendAlpha), key);
        key = fnv_bytes(&target.DestBlendAlpha, sizeof(target.DestBlendAlpha), key);
        key = fnv_bytes(&target.BlendOpAlpha, sizeof(target.BlendOpAlpha), key);
        key = fnv_bytes(&target.LogicOp, sizeof(target.LogicOp), key);
        key = fnv_bytes(&target.RenderTargetWriteMask, sizeof(target.RenderTargetWriteMask), key);
    }
    return key;
}

inline std::uint64_t hash_stencil_ops_key(const D3D12_DEPTH_STENCILOP_DESC& ops, std::uint64_t key)
{
    key = fnv_bytes(&ops.StencilFailOp, sizeof(ops.StencilFailOp), key);
    key = fnv_bytes(&ops.StencilDepthFailOp, sizeof(ops.StencilDepthFailOp), key);
    key = fnv_bytes(&ops.StencilPassOp, sizeof(ops.StencilPassOp), key);
    key = fnv_bytes(&ops.StencilFunc, sizeof(ops.StencilFunc), key);
    return key;
}

inline std::uint64_t hash_depth_stencil_key(const D3D12_DEPTH_STENCIL_DESC& depth, std::uint64_t key)
{
    key = fnv_bytes(&depth.DepthEnable, sizeof(depth.DepthEnable), key);
    key = fnv_bytes(&depth.DepthWriteMask, sizeof(depth.DepthWriteMask), key);
    key = fnv_bytes(&depth.DepthFunc, sizeof(depth.DepthFunc), key);
    key = fnv_bytes(&depth.StencilEnable, sizeof(depth.StencilEnable), key);
    key = fnv_bytes(&depth.StencilReadMask, sizeof(depth.StencilReadMask), key);
    key = fnv_bytes(&depth.StencilWriteMask, sizeof(depth.StencilWriteMask), key);
    key = hash_stencil_ops_key(depth.FrontFace, key);
    return hash_stencil_ops_key(depth.BackFace, key);
}

// The translated shader bytes are part of the cache name. Two translations with the same title
// words but different desktop VS/PS output must not share a driver pipeline.
inline std::uint64_t translated_code_key(std::uint64_t translation_key, const void* vs,
                                         std::size_t vs_bytes, const void* ps,
                                         std::size_t ps_bytes)
{
    return fnv_bytes(ps, ps_bytes, fnv_bytes(vs, vs_bytes, translation_key));
}

// Shared key for the title state and the translated code. Kept here so the device-free test can
// exercise the same formula as the draw path, including the code_key fallback.
inline std::uint64_t pipeline_key(std::uint64_t translation_key, std::uint64_t code_key,
                                  const DXGI_FORMAT* target_formats, std::uint32_t target_count,
                                  DXGI_FORMAT dsv_format, bool has_blend,
                                  const D3D12_BLEND_DESC* blend, bool has_depth_stencil,
                                  const D3D12_DEPTH_STENCIL_DESC* depth_stencil)
{
    const unsigned char schema = 3;
    std::uint64_t key = fnv_bytes(&schema, sizeof(schema), code_key ? code_key : translation_key);
    key = fnv_bytes(&target_count, sizeof(target_count), key);
    for (std::uint32_t i = 0; i < target_count; ++i)
        key = fnv_bytes(&target_formats[i], sizeof(target_formats[i]), key);
    key = fnv_bytes(&dsv_format, sizeof(dsv_format), key);
    const unsigned char states = (has_blend ? 1u : 0u) | (has_depth_stencil ? 2u : 0u);
    key = fnv_bytes(&states, sizeof(states), key);
    if (has_blend) key = hash_blend_key(*blend, key);
    if (has_depth_stencil) key = hash_depth_stencil_key(*depth_stencil, key);
    return key;
}

// Stable adapter/driver identity. The LUID locates the DXGI adapter but is not
// hashed: it can change across boots. Vendor, device, subsystem, revision and
// the UMD version identify a pipeline cache the driver can still consume.
inline std::uint64_t adapter_identity(ID3D12Device* device)
{
    std::uint64_t hash = fnv_bytes("x2w-pso-cache-v2", sizeof("x2w-pso-cache-v2") - 1);
    if (!device) return hash;

    const LUID luid = device->GetAdapterLuid();
    bool found = false;
    IDXGIFactory4* factory = nullptr;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && factory) {
        IDXGIAdapter* adapter = nullptr;
        if (SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))) && adapter) {
            DXGI_ADAPTER_DESC desc{};
            if (SUCCEEDED(adapter->GetDesc(&desc))) {
                found = true;
                hash = fnv_bytes(&desc.VendorId, sizeof(desc.VendorId), hash);
                hash = fnv_bytes(&desc.DeviceId, sizeof(desc.DeviceId), hash);
                hash = fnv_bytes(&desc.SubSysId, sizeof(desc.SubSysId), hash);
                hash = fnv_bytes(&desc.Revision, sizeof(desc.Revision), hash);
            }
            LARGE_INTEGER umd{};
            if (SUCCEEDED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd)))
                hash = fnv_bytes(&umd, sizeof(umd), hash);
            adapter->Release();
        }
        factory->Release();
    }
    // A device without a DXGI adapter still gets a distinct cache file.
    if (!found) hash = fnv_bytes(&luid, sizeof(luid), hash);
    return hash;
}

struct BlobHeader {
    std::uint32_t magic;
    std::uint16_t version;
    std::uint16_t header_size;
    std::uint64_t adapter;
    std::uint64_t payload_size;
    std::uint64_t checksum;
};

constexpr std::uint32_t blob_magic = 0x31505358u;  // "XSP1"
constexpr std::uint16_t blob_version = 1;
constexpr std::uint64_t max_blob_size = 256ull * 1024 * 1024;

inline bool read_file(const wchar_t* path, std::uint64_t max_size, void** data, std::size_t* size)
{
    *data = nullptr;
    *size = 0;
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER length{};
    if (!GetFileSizeEx(file, &length) || length.QuadPart < 0 ||
        static_cast<std::uint64_t>(length.QuadPart) > max_size) {
        CloseHandle(file);
        return false;
    }
    const auto bytes = static_cast<std::size_t>(length.QuadPart);
    auto* buffer = new (std::nothrow) unsigned char[bytes ? bytes : 1];
    if (!buffer) {
        CloseHandle(file);
        return false;
    }
    DWORD got = 0;
    const bool ok = bytes == 0 || (ReadFile(file, buffer, static_cast<DWORD>(bytes), &got, nullptr) && got == bytes);
    CloseHandle(file);
    if (!ok) {
        delete[] buffer;
        return false;
    }
    *data = buffer;
    *size = bytes;
    return true;
}

inline bool write_file_atomic(const wchar_t* path, const void* data, std::size_t size,
                              const wchar_t* temporary)
{
    HANDLE file = CreateFileW(temporary, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    DWORD written = 0;
    const bool complete = size == 0 || (WriteFile(file, data, static_cast<DWORD>(size), &written, nullptr) &&
                                        written == size);
    if (complete) FlushFileBuffers(file);
    CloseHandle(file);
    if (complete && MoveFileExW(temporary, path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) return true;
    DeleteFileW(temporary);
    return false;
}

}  // namespace graphics_pipeline_cache_detail

class GraphicsPipelineCache {
public:
    struct Stats {
        long hits;
        long misses;
        long stores;
        long duplicates;
        long failures;
    };

    explicit GraphicsPipelineCache(ID3D12Device* device, bool allow_library = true)
        : device_(device), allow_library_(allow_library)
    {
        if (device_) device_->AddRef();
        adapter_ = graphics_pipeline_cache_detail::adapter_identity(device_);
        const DWORD n = GetEnvironmentVariableW(L"X2W_SHADER_CACHE", directory_, MAX_PATH);
        if (n && n < MAX_PATH) {
            const std::size_t length = std::wcslen(directory_);
            if (length + 1 < sizeof(directory_) / sizeof(directory_[0]) && length && directory_[length - 1] != L'\\' &&
                directory_[length - 1] != L'/')
                std::wcscat(directory_, L"\\");
            swprintf(path_, sizeof(path_) / sizeof(path_[0]), L"%spso_%016llx.lib", directory_,
                     static_cast<unsigned long long>(adapter_));
        }
        if (configured() && allow_library_) OpenLibrary();
    }

    ~GraphicsPipelineCache() { Shutdown(); }

    GraphicsPipelineCache(const GraphicsPipelineCache&) = delete;
    GraphicsPipelineCache& operator=(const GraphicsPipelineCache&) = delete;

    bool configured() const { return directory_[0] != 0; }
    bool enabled() const { return library_ != nullptr; }
    const wchar_t* path() const { return path_; }

    Stats stats() const
    {
        return {hits_, misses_, stores_, duplicates_, failures_};
    }

    // Coalesce the serialized library: a batch of stores marks the cache dirty and writes the
    // blob once. The draw path must not serialize and flush the whole library on every new PSO.
    void Flush()
    {
        AcquireSRWLockExclusive(&lock_);
        if (dirty_) {
            PersistLibrary();
            dirty_ = false;
        }
        ReleaseSRWLockExclusive(&lock_);
    }

    void Shutdown()
    {
        AcquireSRWLockExclusive(&lock_);
        if (dirty_) {
            PersistLibrary();
            dirty_ = false;
        }
        DropLibrary();
        if (device_) {
            device_->Release();
            device_ = nullptr;
        }
        ReleaseSRWLockExclusive(&lock_);
    }

    // Creates the PSO directly on every cache miss or failure. `cache_hit` is
    // set only when the library supplied a usable pipeline state.
    HRESULT GetOrCreateGraphicsPipeline(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc, std::uint64_t key,
                                        REFIID iid, void** out, bool* cache_hit = nullptr)
    {
        if (cache_hit) *cache_hit = false;
        if (!out) return E_POINTER;
        *out = nullptr;
        AcquireSRWLockExclusive(&lock_);
        if (!device_) {
            ReleaseSRWLockExclusive(&lock_);
            return E_FAIL;
        }

        const std::uint64_t effective = graphics_pipeline_cache_detail::fnv_bytes(&key, sizeof(key), adapter_);
        wchar_t name[64]{};
        swprintf(name, sizeof(name) / sizeof(name[0]), L"pso_%016llx",
                 static_cast<unsigned long long>(effective));

        if (library_) {
            HRESULT loaded = library_->LoadGraphicsPipeline(name, &desc, iid, out);
            if (SUCCEEDED(loaded) && *out) {
                InterlockedIncrement(&hits_);
                if (cache_hit) *cache_hit = true;
                ReleaseSRWLockExclusive(&lock_);
                return S_OK;
            }
            // A driver or adapter change invalidates the whole library. Drop
            // the blob so the next run starts from an empty library.
            if (loaded == D3D12_ERROR_DRIVER_VERSION_MISMATCH || loaded == D3D12_ERROR_ADAPTER_NOT_FOUND) {
                DeleteFileW(path_);
                DropLibrary();
                OpenEmptyLibrary();
            }
        }

        const HRESULT hr = device_->CreateGraphicsPipelineState(&desc, iid, out);
        if (FAILED(hr) || !*out) {
            InterlockedIncrement(&failures_);
            ReleaseSRWLockExclusive(&lock_);
            return hr;
        }

        InterlockedIncrement(&misses_);
        if (library_ && *out) {
            const HRESULT stored = library_->StorePipeline(name, static_cast<ID3D12PipelineState*>(*out));
            if (SUCCEEDED(stored)) {
                InterlockedIncrement(&stores_);
                dirty_ = true;
            } else if (stored == E_INVALIDARG) {
                // The name is already in the library: the existing entry is kept, but this is not
                // a new store and must not mask a same-name pipeline with different bytes.
                InterlockedIncrement(&duplicates_);
            } else {
                InterlockedIncrement(&failures_);
            }
        }
        ReleaseSRWLockExclusive(&lock_);
        return hr;
    }

private:
    bool OpenEmptyLibrary()
    {
        ID3D12Device1* device1 = nullptr;
        if (FAILED(device_->QueryInterface(IID_PPV_ARGS(&device1))) || !device1) return false;
        ID3D12PipelineLibrary* library = nullptr;
        const HRESULT hr = device1->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&library));
        device1->Release();
        if (FAILED(hr) || !library) return false;
        library_ = library;
        return true;
    }

    void DropLibrary()
    {
        if (library_) {
            library_->Release();
            library_ = nullptr;
        }
        delete[] static_cast<unsigned char*>(blob_);
        blob_ = nullptr;
    }

    void OpenLibrary()
    {
        if (!configured()) return;
        CreateDirectoryW(directory_, nullptr);

        void* file = nullptr;
        std::size_t file_size = 0;
        if (graphics_pipeline_cache_detail::read_file(path_, graphics_pipeline_cache_detail::max_blob_size + 4096,
                                                      &file, &file_size)) {
            bool valid = file_size >= sizeof(graphics_pipeline_cache_detail::BlobHeader);
            graphics_pipeline_cache_detail::BlobHeader header{};
            if (valid) std::memcpy(&header, file, sizeof(header));
            valid = valid && header.magic == graphics_pipeline_cache_detail::blob_magic &&
                    header.version == graphics_pipeline_cache_detail::blob_version &&
                    header.header_size == sizeof(header) && header.adapter == adapter_ &&
                    header.payload_size <= graphics_pipeline_cache_detail::max_blob_size &&
                    header.payload_size == file_size - sizeof(header);
            if (valid) {
                const auto* payload = static_cast<const unsigned char*>(file) + sizeof(header);
                valid = graphics_pipeline_cache_detail::fnv_bytes(payload, static_cast<std::size_t>(header.payload_size)) ==
                        header.checksum;
            }
            if (valid) {
                ID3D12Device1* device1 = nullptr;
                if (SUCCEEDED(device_->QueryInterface(IID_PPV_ARGS(&device1))) && device1) {
                    ID3D12PipelineLibrary* library = nullptr;
                    const auto* payload = static_cast<const unsigned char*>(file) + sizeof(header);
                    const HRESULT hr = device1->CreatePipelineLibrary(payload, static_cast<SIZE_T>(header.payload_size),
                                                                      IID_PPV_ARGS(&library));
                    device1->Release();
                    if (SUCCEEDED(hr) && library) {
                        library_ = library;
                        blob_ = file;
                        file = nullptr;
                    }
                }
            }
            if (!library_) DeleteFileW(path_);
            delete[] static_cast<unsigned char*>(file);
        }

        if (!library_) OpenEmptyLibrary();
        if (!library_) InterlockedIncrement(&failures_);
    }

    void PersistLibrary()
    {
        if (!library_ || !configured()) return;
        const SIZE_T payload_size = library_->GetSerializedSize();
        if (!payload_size || payload_size > graphics_pipeline_cache_detail::max_blob_size) {
            InterlockedIncrement(&failures_);
            return;
        }
        auto* payload = new (std::nothrow) unsigned char[payload_size];
        if (!payload) {
            InterlockedIncrement(&failures_);
            return;
        }
        if (FAILED(library_->Serialize(payload, payload_size))) {
            delete[] payload;
            InterlockedIncrement(&failures_);
            return;
        }
        graphics_pipeline_cache_detail::BlobHeader header{};
        header.magic = graphics_pipeline_cache_detail::blob_magic;
        header.version = graphics_pipeline_cache_detail::blob_version;
        header.header_size = sizeof(header);
        header.adapter = adapter_;
        header.payload_size = static_cast<std::uint64_t>(payload_size);
        header.checksum = graphics_pipeline_cache_detail::fnv_bytes(payload, payload_size);

        const std::size_t total = sizeof(header) + payload_size;
        auto* bytes = new (std::nothrow) unsigned char[total];
        if (!bytes) {
            delete[] payload;
            InterlockedIncrement(&failures_);
            return;
        }
        std::memcpy(bytes, &header, sizeof(header));
        std::memcpy(bytes + sizeof(header), payload, payload_size);
        delete[] payload;

        wchar_t temporary[MAX_PATH + 48]{};
        swprintf(temporary, sizeof(temporary) / sizeof(temporary[0]), L"%s.%lu.tmp", path_, GetCurrentProcessId());
        if (!graphics_pipeline_cache_detail::write_file_atomic(path_, bytes, total, temporary))
            InterlockedIncrement(&failures_);
        delete[] bytes;
    }

    ID3D12Device* device_ = nullptr;
    ID3D12PipelineLibrary* library_ = nullptr;
    void* blob_ = nullptr;  // CreatePipelineLibrary keeps this pointer for the library's lifetime.
    bool allow_library_ = true;
    std::uint64_t adapter_ = 0;
    wchar_t directory_[MAX_PATH]{};
    wchar_t path_[MAX_PATH + 32]{};
    SRWLOCK lock_ = SRWLOCK_INIT;
    volatile LONG hits_ = 0, misses_ = 0, stores_ = 0, duplicates_ = 0, failures_ = 0;
    bool dirty_ = false;
};

}  // namespace x2w_startup
