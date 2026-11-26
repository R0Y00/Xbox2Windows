#pragma once
#include "startup_stubs/blake3/blake3.h"
#include "startup_stubs/gpu_abi.h"

#include <istream>
#include <string>
#include <string_view>
#include <unordered_set>

namespace x2w_gpu {

// Hash the stage name, lowercase ASCII shader key and graphics description when present. A single compute stage has
// an empty graphics suffix.
inline std::string gpu_compute_pending_key(std::string_view shader)
{
    blake3_hasher hash;
    blake3_hasher_init(&hash);
    constexpr std::string_view stage = "compute";
    blake3_hasher_update(&hash, stage.data(), stage.size());
    blake3_hasher_update(&hash, shader.data(), shader.size());
    unsigned char digest[16]{};
    blake3_hasher_finalize(&hash, digest, sizeof(digest));
    constexpr char hex[] = "0123456789abcdef";
    std::string key(32, '0');
    for (size_t i = 0; i < sizeof(digest); ++i) {
        key[2 * i] = hex[digest[i] >> 4];
        key[2 * i + 1] = hex[digest[i] & 15];
    }
    return key;
}

struct GpuWarmBad {
    std::unordered_set<std::string> keys;

    void read(std::istream& input)
    {
        std::string line;
        constexpr auto space = " \t\r\n\f\v";
        while (std::getline(input, line)) {
            const auto first = line.find_first_not_of(space);
            if (first == std::string::npos || line[first] == '#') continue;
            const auto end = line.find_first_of(space, first);
            auto token = line.substr(first, end - first);
            if (token.size() == 32 && token.find_first_not_of("0123456789abcdef") == std::string::npos)
                keys.insert(std::move(token));
        }
    }

    bool contains(const GpuCreateDesc& desc, bool has_graphics_state) const
    {
        // Graphics keys include the complete pipeline state description. A shader-only match could refuse unrelated
        // pipelines using the same stages.
        if (keys.empty() || has_graphics_state || desc.stage_count != 1 || desc.stages[0].stage != GpuStage::compute) return false;
        return keys.count(gpu_compute_pending_key(std::string_view(desc.stages[0].key, 16))) != 0;
    }
};

} // namespace x2w_gpu
