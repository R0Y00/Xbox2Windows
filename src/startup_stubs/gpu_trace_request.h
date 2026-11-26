#pragma once

#include <charconv>
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace x2w_gpu {

using CapturePoint = std::optional<std::array<unsigned, 2>>;
struct SceneRequest {
    long id;
    std::string_view stage_key;
    std::string_view second_key;
    CapturePoint point{};
};

inline std::optional<std::uint64_t> stage_key_number(std::string_view key)
{
    if (key.size() != 16) return {};
    std::uint64_t value{};
    const auto result = std::from_chars(key.data(), key.data() + key.size(), value, 16);
    if (result.ec != std::errc{} || result.ptr != key.data() + key.size()) return {};
    return value;
}

// Accessed under the capture mutex. A bind from an earlier request cannot claim a new slot.
struct CaptureSelection {
    long id{};
    std::array<std::optional<std::uint64_t>, 2> keys;
    bool paired{};

    bool matches(long request, std::string_view key) const
    {
        const auto value = stage_key_number(key);
        return request && request == id && value && (value == keys[0] || value == keys[1]);
    }
    void complete(std::string_view key)
    {
        const auto value = stage_key_number(key);
        for (auto& pending : keys) if (pending == value) pending.reset();
        if (!keys[0] && !keys[1]) id = 0;
    }
};

// A positive sequence number followed by zero, one or two distinct 16-digit shader keys.
inline std::optional<SceneRequest> parse_scene_request(std::string_view text)
{
    constexpr auto spaces = " \t\r\n";
    const auto start = text.find_first_not_of(spaces);
    if (start == text.npos) return {};
    text = text.substr(start, text.find_last_not_of(spaces) - start + 1);
    CapturePoint point;
    const auto marker = text.find('@');
    if (marker != text.npos) {
        if (!marker || text.find_last_of(spaces, marker) != marker - 1) return {};
        const auto coords = text.substr(marker + 1);
        const auto comma = coords.find(',');
        if (comma == coords.npos) return {};
        const auto coordinate = [](std::string_view value, unsigned& out) {
            if (value.empty()) return false;
            const auto end = value.data() + value.size();
            const auto result = std::from_chars(value.data(), end, out);
            return result.ec == std::errc{} && result.ptr == end && out <= 1000;
        };
        std::array<unsigned, 2> xy{};
        if (!coordinate(coords.substr(0, comma), xy[0]) || !coordinate(coords.substr(comma + 1), xy[1])) return {};
        point = xy;
        text = text.substr(0, text.find_last_not_of(spaces, marker - 1) + 1);
    }
    if (text.front() == '+') text.remove_prefix(1);
    long id = 0;
    const char* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, id);
    if (parsed.ec != std::errc{} || id <= 0) return {};
    if (parsed.ptr == end) return SceneRequest{id, {}, {}, point};
    std::string_view tail(parsed.ptr, static_cast<std::size_t>(end - parsed.ptr));
    const auto key_start = tail.find_first_not_of(spaces);
    if (!key_start || key_start == tail.npos) return {};
    tail.remove_prefix(key_start);
    const auto separator = tail.find_first_of(spaces);
    const auto first = tail.substr(0, separator);
    const auto number = stage_key_number(first);
    if (!number) return {};
    if (separator == tail.npos) return SceneRequest{id, first, {}, point};
    const auto second = tail.substr(tail.find_first_not_of(spaces, separator));
    const auto other = stage_key_number(second);
    if (!other || other == number) return {};
    return SceneRequest{id, first, second, point};
}

}
