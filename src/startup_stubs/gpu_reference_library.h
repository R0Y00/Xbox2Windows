#pragma once
#include "startup_stubs/blake3/blake3.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace x2w_gpu {
// Compute library key from the stage name, serialized root signature, DXIL and empty graphics suffix.
inline std::wstring vp_compute_library_name(const std::vector<unsigned char>& root, D3D12_SHADER_BYTECODE code)
{
    blake3_hasher hash;
    blake3_hasher_init(&hash);
    auto part = [&](const void* data, std::uint64_t bytes) {
        unsigned char length[8];
        for (unsigned i = 0; i < 8; ++i) length[i] = static_cast<unsigned char>(bytes >> (8 * i));
        blake3_hasher_update(&hash, length, sizeof(length));
        if (bytes) blake3_hasher_update(&hash, data, static_cast<std::size_t>(bytes));
    };
    part("compute", 7);
    part(root.data(), root.size());
    part(code.pShaderBytecode, code.BytecodeLength);
    part(nullptr, 0);
    unsigned char digest[16];
    blake3_hasher_finalize(&hash, digest, sizeof(digest));
    constexpr wchar_t hex[] = L"0123456789abcdef";
    std::wstring name(32, L'0');
    for (std::size_t i = 0; i < sizeof(digest); ++i) {
        name[2 * i] = hex[digest[i] >> 4];
        name[2 * i + 1] = hex[digest[i] & 15];
    }
    return name;
}

class VpReferenceEntryLocks {
public:
    std::mutex& for_name(const std::wstring& name) {
        std::lock_guard<std::mutex> lock(index_mutex_);
        // References survive rehash. Entries live as long as their library.
        return entries_.try_emplace(name).first->second;
    }
private:
    std::mutex index_mutex_;
    std::unordered_map<std::wstring, std::mutex> entries_;
};

// D3D12 borrows the library bytes for the lifetime of the object. Keep both the immutable mapping and the
// underlying file handle open across any atomic replacement of the pipeline cache.
class VpReferenceLibrary {
public:
    VpReferenceLibrary() = default;
    VpReferenceLibrary(const VpReferenceLibrary&) = delete;
    VpReferenceLibrary& operator=(const VpReferenceLibrary&) = delete;
    ~VpReferenceLibrary() {
        library_.Reset();
        if (bytes_) UnmapViewOfFile(bytes_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }

    HRESULT open(ID3D12Device* device, const wchar_t* path) {
        if (file_ != INVALID_HANDLE_VALUE) return E_UNEXPECTED;
        Microsoft::WRL::ComPtr<ID3D12Device1> device1;
        HRESULT hr = device->QueryInterface(IID_PPV_ARGS(&device1));
        if (FAILED(hr)) return hr;
        file_ = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) return HRESULT_FROM_WIN32(GetLastError());
        LARGE_INTEGER length{};
        if (!GetFileSizeEx(file_, &length)) return HRESULT_FROM_WIN32(GetLastError());
        if (length.QuadPart <= 0 || static_cast<ULONGLONG>(length.QuadPart) > SIZE_MAX) return E_INVALIDARG;
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) return HRESULT_FROM_WIN32(GetLastError());
        bytes_ = MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
        if (!bytes_) return HRESULT_FROM_WIN32(GetLastError());
        return device1->CreatePipelineLibrary(bytes_, static_cast<SIZE_T>(length.QuadPart), IID_PPV_ARGS(&library_));
    }

    HRESULT load(const std::wstring& name, const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc, ID3D12PipelineState** result) {
        *result = nullptr;
        if (!library_) return E_NOINTERFACE;
        // D3D12 permits concurrent accesses to the library but requires serialization by entry name.
        std::lock_guard<std::mutex> lock(loads_.for_name(name));
        return library_->LoadComputePipeline(name.c_str(), &desc, IID_PPV_ARGS(result));
    }
private:
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    const void* bytes_ = nullptr;
    Microsoft::WRL::ComPtr<ID3D12PipelineLibrary> library_;
    VpReferenceEntryLocks loads_;
};
} // namespace x2w_gpu
