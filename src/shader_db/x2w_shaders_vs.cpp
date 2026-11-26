// Adapts x2w::vs::translate to shader_db::Translator. Each container uses the same translation and JSON encoding as
// the shader dump command, then record_from_json reads its metadata. Probe the host before translating because wave
// sizes, RDNA capabilities and view-heap layout affect output.
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include "tools/x2w/scan.h"
#include "tools/x2w/shaders_dump.h"
#include "gpu_export.h"
#include "x2w_shaders_translate.h"

namespace shader_db {
namespace {

std::once_flag probed;

void probe() {
    std::call_once(probed, [] { x2w::tool::probe_host(); });
}

bool translate(const uint8_t* container, std::size_t n, ShaderRecord& out, std::string& why) {
    probe();
    x2w::tool::Found f;
    f.bytes = x2w::tool::bounded(x2w::tool::Bytes(container, n));
    f.full = x2w::tool::blake3(f.bytes);
    f.key = x2w::tool::key_of(f.full);
    f.name = x2w::tool::hex16(f.key);
    f.fnv = x2w::tool::fnv64(f.bytes);
    x2w::tool::Made m;
    try {
        m = x2w::tool::make(f);
    } catch (const std::exception& e) {
        // Keep translating the remaining containers if this one fails.
        why = std::string("panic: ") + e.what();
        return false;
    }
    if (m.failed) {
        why = *m.failed;
        return false;
    }
    return record_from_json(m.json, std::move(m.dxil), out, why);
}

// The manifest lines of `x2w shaders` that name the translator and the host settings.
std::string settings() {
    probe();
    const std::string all = x2w::tool::manifest(0);
    std::string out;
    std::size_t at = 0;
    while (at < all.size()) {
        std::size_t end = all.find('\n', at);
        if (end == std::string::npos) end = all.size();
        const std::string line = all.substr(at, end - at);
        if (!line.empty() && line.rfind("format ", 0) && line.rfind("key ", 0) && line.rfind("records ", 0) &&
            line.rfind("producer ", 0) && line.rfind("cache ", 0))
            out += line + "\n";
        at = end + 1;
    }
    return out;
}

const Translator x2w_vs{"x2w-shader-port", translate, settings};

}  // namespace

const Translator* builtin_translator() { return &x2w_vs; }

}  // namespace shader_db
