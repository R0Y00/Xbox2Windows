// Shared ShaderError declaration for the shader module headers included by lib.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <utility>

namespace x2w::vs {

// The reason field stores static text for malformed input, ISA decoding errors and unsupported
// instructions. Compare its contents.
struct ShaderError : std::exception {
    enum class Kind : std::uint8_t { NotDxbc, Malformed, Isa, Unsupported };

    Kind kind = Kind::NotDxbc;
    std::size_t offset = 0;      // Isa, Unsupported
    const char* reason = "";     // Malformed, Isa, Unsupported

    static ShaderError not_dxbc();
    static ShaderError malformed(const char* why);
    static ShaderError isa(std::size_t offset, const char* reason);
    static ShaderError unsupported(std::size_t offset, const char* what);

    // User-facing diagnostics: "not a DXBC container" / "malformed shader: {0}" / "GFX10 code at {offset:#x}:
    // {reason}" / "GFX10 code at {offset:#x}: {what} is not supported"
    std::string to_string() const;
    // Diagnostic form, for example Isa { offset: 4, reason: "..." }.
    std::string debug() const;
    const char* what() const noexcept override;

    bool operator==(const ShaderError& o) const;

private:
    std::string text_;  // to_string(), for what()
    void fill();
};

// Call f and catch ShaderError.
template <class F>
auto try_shader(F&& f) -> std::optional<decltype(f())> {
    try {
        return f();
    } catch (const ShaderError&) {
        return std::nullopt;
    }
}

// The ShaderError `f()` throws, if any (`f` returns void or anything).
template <class F>
std::optional<ShaderError> catch_shader_error(F&& f) {
    try {
        f();
    } catch (const ShaderError& e) {
        return e;
    }
    return std::nullopt;
}

// `{:#x}` of a usize: "0x1f", "0x0".
std::string hex_alt(std::uint64_t v);

}  // namespace x2w::vs
