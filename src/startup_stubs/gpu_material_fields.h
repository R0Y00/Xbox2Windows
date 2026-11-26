#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace x2w_material {
using Mask = std::uint16_t;
constexpr unsigned byte_count(Mask mask)
{
    unsigned bytes = 0;
    for (; mask; mask >>= 1) bytes += 4;
    return bytes;
}
struct Entry {
    std::uint64_t key, hash_low, hash_high;
    Mask fields;
};
// Each local profile row contains the shader key, both DXBC hash halves and the field mask, in hexadecimal.
inline std::vector<Entry> read_profile(std::istream& input)
{
    std::vector<Entry> entries;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        Entry entry{};
        unsigned fields;
        std::string trailing;
        if ((row >> std::hex >> entry.key >> entry.hash_low >> entry.hash_high >> fields) &&
            fields <= 0xffff && !(row >> trailing)) {
            entry.fields = static_cast<Mask>(fields);
            entries.push_back(entry);
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.key < b.key; });
    return entries;
}
inline Mask lookup(const std::vector<Entry>& entries, std::uint64_t key, const void* dxil, std::size_t bytes)
{
    const auto end = entries.end();
    const auto it = std::lower_bound(entries.begin(), end, key, [](const Entry& r, std::uint64_t k) { return r.key < k; });
    if (it == end || it->key != key || !dxil || bytes < 20 || std::memcmp(dxil, "DXBC", 4)) return 0;
    std::uint64_t hash[2];
    std::memcpy(hash, static_cast<const char*>(dxil) + 4, sizeof(hash));
    for (auto row = it; row != end && row->key == key; ++row)
        if (hash[0] == row->hash_low && hash[1] == row->hash_high) return row->fields;
    return 0;
}
inline Mask lookup(std::uint64_t key, const void* dxil, std::size_t bytes)
{
    static const auto profile = [] {
        const char* path = std::getenv("X2W_MATERIAL_FIELDS_PROFILE");
        if (!path || !*path) return std::vector<Entry>{};
        std::ifstream input(path);
        return read_profile(input);
    }();
    return lookup(profile, key, dxil, bytes);
}
}
