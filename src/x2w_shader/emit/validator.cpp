// the DXIL validator in dxil.dll (IDxcValidator).

#include "x2w_shader/emit/validator.h"

// Not WIN32_LEAN_AND_MEAN: dxcapi.h needs the OLE types (BSTR, IUnknown).
#include <windows.h>

#include <dxcapi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <system_error>
#include <vector>

namespace x2w::vs::validator {

namespace {

// Validator and blob interface IDs, supplied explicitly to avoid a dependency on the UUID declarations in
// dxcapi.h.
constexpr GUID CLSID_DXC_VALIDATOR = {0x8ca3e215, 0xf728, 0x4cf3, {0x8c, 0xdd, 0x88, 0xaf, 0x91, 0x75, 0x87, 0xa1}};
constexpr GUID IID_IDXC_VALIDATOR = {0xa6e82bd2, 0x1fd7, 0x4826, {0x98, 0x11, 0x28, 0x57, 0xe7, 0x97, 0xf4, 0x9a}};
constexpr GUID IID_IDXC_BLOB = {0x8ba5fb08, 0x5195, 0x40e2, {0xac, 0x58, 0x0d, 0x98, 0x9c, 0x3a, 0x01, 0x02}};
// DxcValidatorFlags_InPlaceEdit: sign the container we pass.
constexpr UINT32 IN_PLACE_EDIT = 1;

using CreateInstance = HRESULT(WINAPI*)(REFCLSID, REFIID, LPVOID*);

std::string hex(HRESULT hr) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "0x%x", static_cast<unsigned>(hr));
    return buf;
}

// An IDxcBlob over our bytes: QueryInterface, AddRef, Release, GetBufferPointer,
// GetBufferSize. Lives on the caller's stack (AddRef/Release do nothing).
class Blob final : public IDxcBlob {
public:
    Blob(std::uint8_t* data, std::size_t len) : data_(data), len_(len) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** out) override {
        if (iid == __uuidof(IUnknown) || iid == IID_IDXC_BLOB) {
            *out = static_cast<IDxcBlob*>(this);
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return data_; }
    SIZE_T STDMETHODCALLTYPE GetBufferSize() override { return len_; }

private:
    std::uint8_t* data_;
    std::size_t len_;
};

bool is_trimmed(char c) { return c == '\0' || c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

}  // namespace

std::optional<std::filesystem::path> Validator::default_path() {
    if (const wchar_t* p = _wgetenv(L"X2W_SHADERCFG_DXIL")) return std::filesystem::path(p);
    const std::filesystem::path kits = LR"(C:\Program Files (x86)\Windows Kits\10\bin)";
    std::error_code ec;
    std::filesystem::directory_iterator it(kits, ec);
    if (ec) return std::nullopt;
    std::vector<std::filesystem::path> versions;
    for (; it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (ec) break;
        std::filesystem::path p = it->path() / LR"(x64\dxil.dll)";
        std::error_code e2;
        if (std::filesystem::exists(p, e2)) versions.push_back(std::move(p));
    }
    std::sort(versions.begin(), versions.end());
    if (versions.empty()) return std::nullopt;
    return versions.back();
}

std::optional<Validator> Validator::load(const std::optional<std::filesystem::path>& path, std::string* error) {
    auto fail = [&](std::string text) -> std::optional<Validator> {
        if (error) *error = std::move(text);
        return std::nullopt;
    };
    const std::wstring name = path ? path->wstring() : std::wstring(L"dxil.dll");
    HMODULE module = LoadLibraryW(name.c_str());
    if (!module) return fail("cannot load " + (path ? path->string() : std::string("dxil.dll")));
    auto proc = GetProcAddress(module, "DxcCreateInstance");
    if (!proc) return fail("dxil.dll lacks DxcCreateInstance");
    auto create = reinterpret_cast<CreateInstance>(reinterpret_cast<void*>(proc));
    void* object = nullptr;
    HRESULT hr = create(CLSID_DXC_VALIDATOR, IID_IDXC_VALIDATOR, &object);
    if (FAILED(hr) || !object) return fail("DxcCreateInstance(validator) failed: " + hex(hr));
    return Validator(object);
}

std::optional<std::string> Validator::validate(std::span<std::uint8_t> container) {
    Blob blob(container.data(), container.size());
    auto* v = static_cast<IDxcValidator*>(object_);
    IDxcOperationResult* result = nullptr;
    HRESULT hr = v->Validate(&blob, IN_PLACE_EDIT, &result);
    if (FAILED(hr) || !result) return "Validate failed: " + hex(hr);
    HRESULT status = 0;
    result->GetStatus(&status);
    std::string text;
    if (status < 0) {
        IDxcBlobEncoding* errors = nullptr;
        if (result->GetErrorBuffer(&errors) >= 0 && errors) {
            const char* p = static_cast<const char*>(errors->GetBufferPointer());
            std::size_t n = errors->GetBufferSize();
            text.assign(p, p + n);
            // trim_end_matches('\0').trim()
            while (!text.empty() && is_trimmed(text.back())) text.pop_back();
            std::size_t start = 0;
            while (start < text.size() && is_trimmed(text[start]) && text[start] != '\0') ++start;
            text.erase(0, start);
            errors->Release();
        }
        if (text.empty()) text = "rejected (" + hex(status) + ")";
    }
    result->Release();
    if (status < 0) return text;
    return std::nullopt;
}

Validator& Validator::operator=(Validator&& o) noexcept {
    if (this != &o) {
        if (object_) static_cast<IUnknown*>(object_)->Release();
        object_ = o.object_;
        o.object_ = nullptr;
    }
    return *this;
}

Validator::~Validator() {
    // We own one reference.
    if (object_) static_cast<IUnknown*>(object_)->Release();
}

}  // namespace x2w::vs::validator
