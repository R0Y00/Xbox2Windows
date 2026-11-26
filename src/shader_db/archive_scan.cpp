// See archive_scan.h.
#include "archive_scan.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "../startup_stubs/inflate.h"

namespace shader_db {
namespace {

uint32_t u32(const uint8_t* p)
{
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

const uint8_t* find(const uint8_t* from, const uint8_t* end, const char (&needle)[5])
{
    const uint8_t* at = std::search(from, end, needle, needle + 4);
    return at == end ? nullptr : at;
}

const uint8_t* find(const uint8_t* from, const uint8_t* end, const uint8_t (&needle)[4])
{
    const uint8_t* at = std::search(from, end, needle, needle + 4);
    return at == end ? nullptr : at;
}

constexpr std::size_t max_unpacked = 64u << 20;

// The length-prefixed printable name ends at the NUL before the record.
std::string name_before(const uint8_t* file, std::size_t nul)
{
    std::size_t run = nul;
    while (run > 0 && file[run - 1] >= 0x21 && file[run - 1] <= 0x7e) --run;
    for (std::size_t l = run ? run - 1 : 0; l < nul; ++l)
        if (static_cast<std::size_t>(file[l]) == nul - l) return std::string(reinterpret_cast<const char*>(file + l + 1), nul - l - 1);
    return {};
}

// The stream must end exactly at the declared unpacked size.
bool unpack(const uint8_t* packed, std::size_t packed_bytes, std::size_t unpacked_bytes, std::vector<uint8_t>& out)
{
    out.assign(unpacked_bytes + 1, 0);  // one spare byte: a stream that runs longer must not fit
    x2w_inflate::Stream s{packed, packed_bytes, 0, out.data(), out.size()};
    if (!x2w_inflate::inflate_raw(s) || s.out_pos != unpacked_bytes) return false;
    out.resize(unpacked_bytes);
    return true;
}

}  // namespace

std::size_t dxbc_size(const uint8_t* data, std::size_t available)
{
    if (available < 32 || std::memcmp(data, "DXBC", 4)) return 0;
    const std::size_t total = u32(data + 24), count = u32(data + 28);
    if (total < 32 || total > available || count > 256 || 32 + 4 * count > total) return 0;
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t offset = u32(data + 32 + 4 * i);
        if (offset > total || total - offset < 8) return 0;
        const std::size_t size = u32(data + offset + 4);
        if (size > total - offset - 8) return 0;
    }
    return total;
}

std::size_t scan_bytes(const uint8_t* data, std::size_t n, const ContainerSink& sink)
{
    const uint8_t* end = data + n;
    std::size_t corrupt = 0;
    char label[64];
    if (find(data, end, "SGD5")) {
        static const uint8_t marker[4] = {0xe7, 0xdc, 0x0d, 0x00};
        std::vector<uint8_t> unpacked;
        for (const uint8_t* at = find(data, end, marker); at; at = find(at + 1, end, marker)) {
            const std::size_t o = static_cast<std::size_t>(at - data);
            if (o < 6 || n - o < 12) continue;
            const std::size_t size = u32(at - 4), unpacked_bytes = u32(at + 8);
            if (u32(at + 4) != size || size < 12 || unpacked_bytes > max_unpacked || size > n - o) continue;
            const std::string name = data[o - 6] == 0 ? name_before(data, o - 6) : std::string();
            if (!unpack(at + 12, size - 12, unpacked_bytes, unpacked)) {
                ++corrupt;
                continue;
            }
            std::snprintf(label, sizeof(label), "@0x%zx ", o);
            // Limit a container embedded in the unpacked record to its declared size.
            const std::size_t bounded = dxbc_size(unpacked.data(), unpacked.size());
            sink(unpacked.data(), bounded ? bounded : unpacked.size(), label + (name.empty() ? std::string("?") : name));
        }
    }
    for (const uint8_t* at = find(data, end, "DXBC"); at; at = find(at + 1, end, "DXBC")) {
        const std::size_t size = dxbc_size(at, static_cast<std::size_t>(end - at));
        if (!size) continue;
        std::snprintf(label, sizeof(label), "@0x%zx", static_cast<std::size_t>(at - data));
        sink(at, size, label);
    }
    return corrupt;
}

namespace {

void walk(const std::wstring& path, uint64_t max_file, const std::function<void(const std::wstring&)>& file_sink,
          const ContainerSink& sink)
{
    const DWORD attributes = GetFileAttributesW(path.c_str());
    if (attributes == INVALID_FILE_ATTRIBUTES) return;
    if (attributes & FILE_ATTRIBUTE_DIRECTORY) {
        WIN32_FIND_DATAW found;
        std::vector<std::wstring> names;
        HANDLE h = FindFirstFileW((path + L"\\*").c_str(), &found);
        if (h == INVALID_HANDLE_VALUE) return;
        do {
            if (std::wcscmp(found.cFileName, L".") && std::wcscmp(found.cFileName, L"..")) names.push_back(found.cFileName);
        } while (FindNextFileW(h, &found));
        FindClose(h);
        std::sort(names.begin(), names.end());  // a fixed order, so the first source of a container is stable
        for (const auto& name : names) walk(path + L"\\" + name, max_file, file_sink, sink);
        return;
    }
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) return;
    LARGE_INTEGER size{};
    if (GetFileSizeEx(file, &size) && size.QuadPart > 0 && static_cast<uint64_t>(size.QuadPart) <= max_file) {
        HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        const void* view = mapping ? MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
        if (view) {
            file_sink(path);
            scan_bytes(static_cast<const uint8_t*>(view), static_cast<std::size_t>(size.QuadPart), sink);
            UnmapViewOfFile(view);
        }
        if (mapping) CloseHandle(mapping);
    }
    CloseHandle(file);
}

}  // namespace

void scan_paths(const std::vector<std::wstring>& paths, uint64_t max_file, const std::function<void(const std::wstring&)>& file_sink,
                const ContainerSink& sink)
{
    for (const auto& p : paths) walk(p, max_file, file_sink, sink);
}

}  // namespace shader_db
