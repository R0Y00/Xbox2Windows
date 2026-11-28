// Experimental (docs/startup-runtime.md, #870): the 32-bit compute root arguments a recorded command list keeps
// (list slots 37 and 39; the graphics slot 40 keeps its own words). Plain standard C++ with no Windows types, so it can be tested on its own.
//
// A list keeps up to 16 root parameters of up to 32 words each (native slot 86 roots declare up to 32 words;
// translated compute defaults to 8 words). A write that starts past the parameter's capacity is dropped and one
// that runs past it is cut off at the end of the parameter, so the caller can log either case.
#pragma once

#include <cstddef>
#include <cstdint>

namespace x2w_root_constants {

inline constexpr unsigned parameter_count = 16;
inline constexpr unsigned words_per_parameter = 8;
inline constexpr unsigned max_words_per_parameter = 32;

enum class Result { stored, truncated, dropped };

struct Store {
    std::uint32_t words[parameter_count][max_words_per_parameter] = {};
    // One bit per word that a write stored, like the graphics path's `constant_words_bound`, so a later
    // dispatch can tell "set to 0" from "never set".
    std::uint32_t bound[parameter_count] = {};

    void clear() noexcept
    {
        for (auto& row : words)
            for (auto& word : row) word = 0;
        for (auto& mask : bound) mask = 0;
    }

    // True when every word in [offset, offset + count) of `parameter` was written since the last clear.
    bool is_bound(unsigned parameter, unsigned offset, unsigned count, unsigned capacity = words_per_parameter) const noexcept
    {
        if (!count || parameter >= parameter_count || offset >= capacity || count > capacity - offset || capacity > max_words_per_parameter)
            return false;
        const std::uint32_t mask = count >= 32 ? ~0u : (((1u << count) - 1) << offset);
        return (bound[parameter] & mask) == mask;
    }

    // SetXxxRoot32BitConstants(parameter, count, data, offset). `data` must hold `count` words.
    Result set(unsigned parameter, unsigned count, const std::uint32_t* data, unsigned offset,
               unsigned capacity = words_per_parameter) noexcept
    {
        if (!count || !data || parameter >= parameter_count || offset >= capacity || capacity > max_words_per_parameter)
            return Result::dropped;
        const bool cut = count > capacity - offset;
        const unsigned n = cut ? capacity - offset : count;
        for (unsigned i = 0; i < n; ++i) words[parameter][offset + i] = data[i];
        const std::uint32_t mask = n >= 32 ? ~0u : (((1u << n) - 1) << offset);
        bound[parameter] |= mask;
        return cut ? Result::truncated : Result::stored;
    }

    // SetXxxRoot32BitConstant(parameter, value, offset).
    Result set_one(unsigned parameter, std::uint32_t value, unsigned offset,
                   unsigned capacity = words_per_parameter) noexcept
    {
        return set(parameter, 1, &value, offset, capacity);
    }

    // The word a shader reads; 0 outside the kept range.
    std::uint32_t word(int parameter, int offset) const noexcept
    {
        return parameter >= 0 && parameter < static_cast<int>(parameter_count) && offset >= 0 &&
                       offset < static_cast<int>(max_words_per_parameter)
                   ? words[parameter][offset]
                   : 0;
    }
};

}  // namespace x2w_root_constants
