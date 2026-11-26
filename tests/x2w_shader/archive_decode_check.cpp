// Real-program check of the vs-core port: decodes the GFX10 code of every native program listed in
// a reference file made by the upstream Rust implementation (a Rust helper that prints hashes only), and
// compares the decode listing, the SSA form and two IR passes hash for hash.
//
//   archive_decode_check                 env X2W_VS_CORE_ARCHIVE (read only) and
//                                        X2W_VS_CORE_REFERENCE (the helper's lines); without them the
//                                        check is skipped (exit 77), so a plain ctest does not sweep.
//   archive_decode_check --dump <label> <k>   the full dumps of one program, to diff with the helper's.
//
// Locating the programs (SGD5 records, XDAT/XBCH/FSSM, FSPX) is a minimal stand-in for the upstream
// sgd5.rs and native.rs written for this test only; the vs-front port replaces it in the library.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "startup_stubs/inflate.h"
#include "x2w_shader/gfx10/mod.h"
#include "x2w_shader/ir.h"
#include "x2w_shader/lib.h"

using namespace x2w::vs;

namespace {

using Bytes = std::span<const std::uint8_t>;

std::uint64_t fnv(const std::string& s) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char b : s) {
        h ^= b;
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string hex(std::uint64_t v, int width = 0) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%0*llx", width, static_cast<unsigned long long>(v));
    return buf;
}

std::uint32_t le32(Bytes d, std::size_t at) { return dxbc::u32_at(d, at); }
std::uint64_t le64(Bytes d, std::size_t at) { return dxbc::u64_at(d, at); }

// sgd5.rs `record`: the container packed at the marker `at`, or nullopt.
std::optional<std::vector<std::uint8_t>> sgd5_unpack(Bytes file, std::size_t at) {
    auto u32 = [&](std::size_t o) -> std::optional<std::size_t> {
        if (o + 4 > file.size()) return std::nullopt;
        return le32(file, o);
    };
    if (at < 6) return std::nullopt;
    auto n = u32(at - 4), unpacked = u32(at + 8), n2 = u32(at + 4);
    if (!n || !unpacked || !n2 || *n2 != *n || *n < 12 || *unpacked > (64u << 20)) return std::nullopt;
    if (at + *n > file.size()) return std::nullopt;
    std::vector<std::uint8_t> out(*unpacked);
    x2w_inflate::Stream s{file.data() + at + 12, *n - 12, 0, out.data(), out.size()};
    if (!x2w_inflate::inflate_raw(s) || s.out_pos != out.size()) return std::nullopt;
    return out;
}

struct Native {
    std::string stage;
    bool wave64;
    std::uint16_t vgprs;
    std::vector<std::uint8_t> code;
};

std::optional<Bytes> sub(Bytes d, std::uint64_t offset, std::uint64_t len) {
    if (offset > d.size() || len > d.size() - offset) return std::nullopt;
    return d.subspan(std::size_t(offset), std::size_t(len));
}

// native.rs `native_programs`, the code-locating part only. Throws on what it cannot read.
std::vector<Native> native_programs(Bytes part) {
    std::vector<Native> out;
    if (part.size() >= 4 && std::memcmp(part.data(), "FSPX", 4) == 0) {
        std::size_t count = le32(part, 8);
        auto entry = [&](std::uint32_t kind) -> Bytes {
            for (std::size_t i = 0; i < count; ++i) {
                std::size_t at = 0x10 + 16 * i;
                if (le32(part, at) == kind) {
                    auto b = sub(part, le32(part, at + 4), le32(part, at + 8));
                    if (!b) throw ShaderError::malformed("FSPX entry out of bounds");
                    return *b;
                }
            }
            throw ShaderError::malformed("FSPX entry missing");
        };
        Bytes packet = entry(0), code = entry(6);
        bool wave64 = (dxbc::u16_at(packet, 0x12) & (1 << 4)) == 0;
        std::uint32_t granules = (le32(packet, 0x14) & 0x3F) + 1;
        std::uint32_t vgprs = std::min<std::uint32_t>(granules * (wave64 ? 4 : 8), 256);
        out.push_back({"Cs", wave64, std::uint16_t(vgprs), {code.begin(), code.end()}});
        return out;
    }
    if (part.size() < 0x1C || std::memcmp(part.data() + 0x18, "XDAT", 4) != 0) return out;
    std::size_t count = le32(part, 0x1C);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t at = 0x18 + 12 + 12 * i;
        if (le32(part, at) != 0x0d) continue;
        auto blob = sub(part, le32(part, at + 4), le32(part, at + 8));
        if (!blob || blob->size() < 0x18 || std::memcmp(blob->data(), "XBCH", 4) != 0) {
            throw ShaderError::malformed("XBCH");
        }
        std::uint64_t packed = le64(*blob, 8), unpacked = le64(*blob, 16);
        auto stream = sub(*blob, 0x18, packed);
        if (!stream || unpacked > (16u << 20)) throw ShaderError::malformed("XBCH stream");
        std::vector<std::uint8_t> fssm(static_cast<std::size_t>(unpacked));
        std::size_t produced = 0;
        if (!x2w_inflate::inflate_zlib(stream->data(), stream->size(), fssm.data(), fssm.size(), &produced) ||
            produced != fssm.size()) {
            throw ShaderError::malformed("XBCH inflate");
        }
        Bytes f(fssm);
        if (f.size() < 0x10 || std::memcmp(f.data(), "FSSM", 4) != 0) throw ShaderError::malformed("FSSM");
        Bytes payload = f.subspan(0x10);
        static const std::pair<const char*, std::size_t> SLOTS[] = {
            {"Ls", 0x08}, {"Hs", 0x10}, {"Es", 0x18}, {"Gs", 0x20}, {"Vs", 0x28}, {"Ps", 0x30}, {"Ms", 0x48}, {"Cs", 0x60},
        };
        for (auto [stage, slot] : SLOTS) {
            std::uint64_t start = le64(payload, slot);
            if (start == 0xDEADBEEFDEADBEEFull) continue;
            auto record = sub(payload, start, 0x54);
            if (!record) throw ShaderError::malformed("FSSM record out of bounds");
            auto code = sub(payload, le64(*record, 0), le64(*record, 8));
            if (!code) throw ShaderError::malformed("native code out of bounds");
            Native n;
            n.stage = stage;
            n.vgprs = std::uint16_t(le32(*record, 0x50) & 0x3FF);
            n.wave64 = (le32(*record, 0x14) & (1u << 5)) == 0;
            n.code.assign(code->begin(), code->end());
            out.push_back(std::move(n));
        }
    }
    return out;
}

// --- the dumps, line for line as the helper prints them

std::string decode_dump(const std::vector<std::uint8_t>& code, bool wave64, std::size_t& ninst) {
    ninst = 0;
    std::vector<gfx10::Instruction> list;
    try {
        list = gfx10::decode(code);
    } catch (const ShaderError& e) {
        return "ERR " + e.to_string() + "\n";
    }
    ninst = list.size();
    std::string out;
    for (const auto& i : list) {
        out += hex(i.offset) + " " + gfx10::debug(i.encoding) + " " + std::to_string(i.opcode);
        for (std::uint32_t w : i.words) out += " " + hex(w, 8);
        out += "|" + i.text(wave64) + "\n";
    }
    return out;
}

std::string ir_dump(const ir::Program& p) {
    std::string out = "inputs";
    for (const auto& [r, v] : p.inputs) out += " " + r.debug() + "=" + std::to_string(v.n);
    out += "\n";
    for (std::size_t b = 0; b < p.blocks.size(); ++b) {
        const ir::Block& block = p.blocks[b];
        std::string preds;
        for (std::size_t i = 0; i < block.preds.size(); ++i) preds += (i ? "," : "") + std::to_string(block.preds[i]);
        out += "B" + std::to_string(b) + " @" + hex(block.offset) + " preds " + preds + " exit " + block.exit.debug() + "\n";
        for (const ir::Phi& phi : block.phis) {
            std::string args;
            for (std::size_t i = 0; i < phi.args.size(); ++i) args += (i ? "," : "") + std::to_string(phi.args[i].n);
            out += " phi " + phi.reg.debug() + "=" + std::to_string(phi.value.n) + " <- " + args + "\n";
        }
        for (const ir::Inst& inst : block.insts) {
            out += " " + std::string(inst.name());
            for (std::uint32_t w : inst.machine.words) out += " " + hex(w, 8);
            out += " r";
            for (const auto& [r, v] : inst.reads) out += " " + r.debug() + "=" + std::to_string(v.n);
            out += " w";
            for (const auto& [r, v] : inst.writes) out += " " + r.debug() + "=" + std::to_string(v.n);
            out += "\n";
        }
    }
    return out;
}

struct Dumps {
    std::string decode, ir, early, skips, status;
    std::size_t ninst = 0;
};

Dumps dumps(const Native& n) {
    Dumps d;
    d.decode = decode_dump(n.code, n.wave64, d.ninst);
    // Where Rust panics (an index out of range), the port throws a std::exception other than
    // ShaderError: both sides record "PANIC" for that stage.
    std::optional<ir::Program> p;
    try {
        p = ir::build(n.code, n.wave64, n.vgprs);
    } catch (const ShaderError& e) {
        d.ir = "ERR " + e.to_string() + "\n";
        d.status = e.to_string();
        return d;
    } catch (const std::exception&) {
        d.ir = "PANIC\n";
        d.status = "PANIC";
        return d;
    }
    d.ir = ir_dump(*p);
    d.status = "ok";
    try {
        auto early = ir::without_early_outs(*p);
        d.early = early ? ir_dump(*early) : "none\n";
    } catch (const std::exception&) {
        d.early = "PANIC\n";
    }
    try {
        auto skips = ir::without_skips(*p, true);
        d.skips = skips ? ir_dump(*skips) : "none\n";
    } catch (const std::exception&) {
        d.skips = "PANIC\n";
    }
    return d;
}

std::optional<std::vector<std::uint8_t>> container_at(Bytes file, const std::string& label) {
    std::size_t at = std::stoull(label.substr(1), nullptr, 16);
    if (label[0] == 'r') return sgd5_unpack(file, at);
    if (at >= file.size()) return std::nullopt;
    return std::vector<std::uint8_t>(file.begin() + at, file.end());
}

std::vector<Native> programs_of(Bytes file, const std::string& label) {
    auto bytes = container_at(file, label);
    if (!bytes) throw ShaderError::malformed("corrupt record");
    // The container borrows `bytes`; the programs are copies.
    Container c = Container::parse(*bytes);
    const Part* part = c.part(FourCC::DXIL);
    if (!part) part = c.part(FourCC::ACCC);
    if (!part) throw ShaderError::malformed("no DXIL part");
    return native_programs(part->data);
}

std::vector<std::uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, '\t')) out.push_back(field);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    const char* archive = std::getenv("X2W_VS_CORE_ARCHIVE");
    const char* reference = std::getenv("X2W_VS_CORE_REFERENCE");
    if (!archive || !*archive) {
        std::printf("SKIP: X2W_VS_CORE_ARCHIVE is not set\n");
        return 77;
    }
    std::vector<std::uint8_t> file = read_file(archive);
    if (file.empty()) {
        std::printf("SKIP: archive unreadable\n");
        return 77;
    }
    if (argc == 4 && std::strcmp(argv[1], "--dump") == 0) {
        auto progs = programs_of(file, argv[2]);
        const Native& n = progs.at(std::stoul(argv[3]));
        Dumps d = dumps(n);
        std::printf("== decode\n%s== ir\n%s== early\n%s== skips\n%s", d.decode.c_str(), d.ir.c_str(), d.early.c_str(),
                    d.skips.c_str());
        return 0;
    }
    if (!reference || !*reference) {
        std::printf("SKIP: X2W_VS_CORE_REFERENCE is not set\n");
        return 77;
    }
    std::ifstream ref(reference);
    std::string line;
    std::size_t programs = 0, instructions = 0, skipped = 0, mismatches = 0, ir_ok = 0;
    std::map<std::string, std::size_t> statuses;  // IR failures by text
    std::string cached_label;
    std::vector<Native> cached;
    std::string cached_error;
    while (std::getline(ref, line)) {
        auto f = split_tabs(line);
        if (f.size() < 3) continue;
        if (f[1] == "-") {
            ++skipped;
            continue;
        }
        if (f.size() != 12) {
            std::printf("bad reference line: %s\n", line.c_str());
            return 2;
        }
        const std::string& label = f[0];
        std::size_t k = std::stoul(f[1]);
        if (label != cached_label) {
            cached_label = label;
            cached.clear();
            cached_error.clear();
            try {
                cached = programs_of(file, label);
            } catch (const std::exception& e) {
                cached_error = e.what();
            }
        }
        ++programs;
        auto report = [&](const std::string& what) {
            if (++mismatches <= 30) std::printf("MISMATCH %s %zu: %s\n", label.c_str(), k, what.c_str());
        };
        if (k >= cached.size()) {
            report("program not found (" + cached_error + ")");
            continue;
        }
        const Native& n = cached[k];
        std::string got_meta = n.stage + "\t" + (n.wave64 ? "1" : "0") + "\t" + std::to_string(n.vgprs) + "\t" +
                               std::to_string(n.code.size());
        std::string want_meta = f[2] + "\t" + f[3] + "\t" + f[4] + "\t" + f[5];
        if (got_meta != want_meta) {
            report("program differs: " + got_meta + " vs " + want_meta);
            continue;
        }
        Dumps d = dumps(n);
        instructions += d.ninst;
        if (std::to_string(d.ninst) != f[6]) report("instruction count " + std::to_string(d.ninst) + " vs " + f[6]);
        if (hex(fnv(d.decode), 16) != f[7]) report("decode listing");
        if (hex(fnv(d.ir), 16) != f[8]) report("ir");
        if (hex(fnv(d.early), 16) != f[9]) report("without_early_outs");
        if (hex(fnv(d.skips), 16) != f[10]) report("without_skips");
        if (d.status != f[11]) report("ir status \"" + d.status + "\" vs \"" + f[11] + "\"");
        if (d.status == "ok") {
            ++ir_ok;
        } else {
            ++statuses[d.status.substr(d.status.find(": ") + 2)];
        }
    }
    std::printf("%zu programs (%zu instructions), %zu with IR, %zu container lines skipped, %zu mismatches\n", programs,
                instructions, ir_ok, skipped, mismatches);
    std::size_t shown = 0;
    for (const auto& [why, count] : statuses) {
        if (++shown > 15) break;
        std::printf("  %6zu  %s\n", count, why.c_str());
    }
    return programs > 0 && mismatches == 0 ? 0 : 1;
}
