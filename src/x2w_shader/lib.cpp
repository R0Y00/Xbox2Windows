// Shader error formatting.
#include "x2w_shader/lib.h"

#include <cstdio>
#include <cstring>

namespace x2w::vs {

std::string hex_alt(std::uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "0x%llx", static_cast<unsigned long long>(v));
    return buf;
}

ShaderError ShaderError::not_dxbc() {
    ShaderError e;
    e.kind = Kind::NotDxbc;
    e.fill();
    return e;
}

ShaderError ShaderError::malformed(const char* why) {
    ShaderError e;
    e.kind = Kind::Malformed;
    e.reason = why;
    e.fill();
    return e;
}

ShaderError ShaderError::isa(std::size_t offset, const char* reason) {
    ShaderError e;
    e.kind = Kind::Isa;
    e.offset = offset;
    e.reason = reason;
    e.fill();
    return e;
}

ShaderError ShaderError::unsupported(std::size_t offset, const char* what) {
    ShaderError e;
    e.kind = Kind::Unsupported;
    e.offset = offset;
    e.reason = what;
    e.fill();
    return e;
}

void ShaderError::fill() {
    switch (kind) {
    case Kind::NotDxbc:
        text_ = "not a DXBC container";
        break;
    case Kind::Malformed:
        text_ = std::string("malformed shader: ") + reason;
        break;
    case Kind::Isa:
        text_ = "GFX10 code at " + hex_alt(offset) + ": " + reason;
        break;
    case Kind::Unsupported:
        text_ = "GFX10 code at " + hex_alt(offset) + ": " + reason + " is not supported";
        break;
    }
}

std::string ShaderError::to_string() const { return text_; }

namespace {
// Format diagnostic strings as quoted, with `"` and `\` escaped (the static texts hold nothing else).
std::string quoted(const char* s) {
    std::string out = "\"";
    for (; *s; ++s) {
        if (*s == '"' || *s == '\\') out += '\\';
        out += *s;
    }
    return out + "\"";
}
}  // namespace

std::string ShaderError::debug() const {
    switch (kind) {
    case Kind::NotDxbc:
        return "NotDxbc";
    case Kind::Malformed:
        return "Malformed(" + quoted(reason) + ")";
    case Kind::Isa:
        return "Isa { offset: " + std::to_string(offset) + ", reason: " + quoted(reason) + " }";
    case Kind::Unsupported:
        return "Unsupported { offset: " + std::to_string(offset) + ", what: " + quoted(reason) + " }";
    }
    return {};
}

const char* ShaderError::what() const noexcept { return text_.c_str(); }

bool ShaderError::operator==(const ShaderError& o) const {
    return kind == o.kind && offset == o.offset && std::strcmp(reason, o.reason) == 0;
}

}  // namespace x2w::vs
