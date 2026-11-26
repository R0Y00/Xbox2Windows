
#include "x2w_shader/emit/census.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace x2w::vs::bitcode::census {

namespace {

// The modules this thread wrote since the last take(): the length and a hash of each one's
// bitcode, with its counts (a program may be emitted in several forms before one is kept).
thread_local std::vector<std::tuple<std::size_t, std::uint64_t, Census>> written;

std::uint64_t fnv(std::span<const std::uint8_t> bytes) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::uint8_t b : bytes) h = (h ^ b) * 0x00000100000001b3ull;
    return h;
}

std::size_t k(std::string_view s) {
    for (std::size_t i = 0; i < KINDS.size(); ++i)
        if (s == KINDS[i]) return i;
    bug("kind");
}

bool starts(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

}  // namespace

std::size_t kind(std::string_view name) {
    std::string_view n = starts(name, "dx.op.") ? name.substr(6) : name;
    n = n.substr(0, n.find('.'));
    if (n == "unary" || n == "binary" || n == "tertiary" || n == "unaryBits" || n == "legacyF16ToF32" ||
        n == "legacyF32ToF16" || n == "isSpecialFloat" || n == "dot2" || n == "dot3" || n == "dot4" ||
        n == "quaternary")
        return k("math_call");
    if (n == "bufferLoad" || n == "rawBufferLoad") return k("buffer_load");
    if (n == "cbufferLoadLegacy" || n == "cbufferLoad") return k("cbuffer_load");
    if (n == "bufferStore" || n == "rawBufferStore" || n == "textureStore") return k("buffer_store");
    if (n == "atomicBinOp" || n == "atomicCompareExchange") return k("atomic");
    if (n == "createHandle" || n == "createHandleFromBinding" || n == "createHandleFromHeap") return k("handle");
    if (n == "annotateHandle") return k("annotate");
    if (n == "getDimensions") return k("dimensions");
    if (n == "barrier") return k("barrier");
    if (starts(n, "wave") || starts(n, "quad")) return k("wave_op");
    if (starts(n, "textureLoad") || starts(n, "sample") || starts(n, "textureGather") || starts(n, "calculateLOD"))
        return k("texture_read");
    return k("io_other");
}

bool enabled() {
    static const bool on = std::getenv("X2W_SHADERCFG_SHADER_CENSUS") != nullptr;
    return on;
}

std::optional<Census> take(std::span<const std::uint8_t> dxil) {
    auto list = std::exchange(written, {});
    const std::uint8_t magic[4] = {'B', 'C', 0xC0, 0xDE};
    auto it = std::search(dxil.begin(), dxil.end(), magic, magic + 4);
    if (it == dxil.end()) return std::nullopt;
    auto at = std::size_t(it - dxil.begin());
    for (auto r = list.rbegin(); r != list.rend(); ++r) {
        auto& [len, hash, c] = *r;
        if (at + len <= dxil.size() && fnv(dxil.subspan(at, len)) == hash) return c;
    }
    return std::nullopt;
}

void note(const Module& m, std::span<const std::uint8_t> bitcode) {
    if (!enabled()) return;
    Census c = m.census();
    written.emplace_back(bitcode.size(), fnv(bitcode), c);
}

}  // namespace x2w::vs::bitcode::census

namespace x2w::vs::bitcode {

census::Census Module::census() const {
    auto k = [](std::string_view s) {
        for (std::size_t i = 0; i < census::KINDS.size(); ++i)
            if (s == census::KINDS[i]) return i;
        bug("kind");
    };
    std::vector<std::size_t> call_kind;
    call_kind.reserve(funcs.size());
    for (const Func& f : funcs) call_kind.push_back(census::kind(f.name));
    const std::vector<bool> lv = live();
    // The blocks write() lays out: those reached from the entry.
    std::vector<bool> seen(blocks.size(), false);
    std::vector<std::size_t> stack = {0};
    if (!blocks.empty()) seen[0] = true;
    while (!stack.empty()) {
        std::size_t b = stack.back();
        stack.pop_back();
        for (std::size_t s : successors(blocks[b].term)) {
            if (!seen[s]) {
                seen[s] = true;
                stack.push_back(s);
            }
        }
    }
    census::Census c;
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        if (!seen[b]) continue;
        const Block& blk = blocks[b];
        c.counts[k("blocks")] += 1;
        auto count = [&](std::uint32_t i) {
            c.counts[0] += 1;
            const Op& op = insts[i].op;
            std::size_t at = 0;
            switch (op.kind) {
            case Op::Bin:
            case Op::Cmp:
                at = k("arith");
                break;
            case Op::Cast:
                at = op.code == std::uint8_t(Cast::Bitcast) ? k("bitcast") : k("cast");
                break;
            case Op::Select:
                at = k("select");
                break;
            case Op::Extract:
                at = k("extract");
                break;
            case Op::Phi:
                at = k("phi");
                break;
            case Op::Gep:
                at = k("gep");
                break;
            case Op::Load:
                at = k("lds_load");
                break;
            case Op::Store:
                at = k("lds_store");
                break;
            case Op::AtomicRmw:
            case Op::CmpXchg:
                at = k("atomic");
                break;
            case Op::Call:
                at = call_kind[op.index];
                break;
            }
            c.counts[at] += 1;
            auto t = tags.find(i);
            if (t != tags.end()) {
                switch (t->second) {
                case Tag::Guard:
                    c.counts[k("guard_select")] += 1;
                    break;
                case Tag::Clamp:
                    c.counts[k("clamp")] += 1;
                    break;
                case Tag::Counter:
                    c.counts[k("loop_counter")] += 1;
                    break;
                }
            }
        };
        for (std::uint32_t i : blk.phis)
            if (lv[i]) count(i);
        for (std::uint32_t i : blk.body) count(i);
    }
    return c;
}

}  // namespace x2w::vs::bitcode
