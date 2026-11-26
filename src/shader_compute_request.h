#pragma once
#include <cstdint>
#include <filesystem>
#include <string>

namespace x2w {
struct ComputeRequest {
    std::string code, data, root, description, sa, sb;
    uint64_t key;
};
struct ComputeContainer {
    std::string bytes, root;
};
struct GraphicsRequest {
    std::string a, b, root, input_layout, pixel_entry, pixel_input_description;
    uint64_t key{};
};
GraphicsRequest read_graphics_request(const std::filesystem::path &marker);
std::string read_shader_container_bytes(const std::filesystem::path &file);
ComputeContainer read_compute_container(const std::filesystem::path &file);
// Verify the content key over all six immutable capture parts before exposing any bytes.
ComputeRequest read_compute_request(const std::filesystem::path &marker);
} // namespace x2w
