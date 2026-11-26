// Shader discovery, naming, serialization and export scheduling. See scan.h.
#include "scan.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "x2w_shader/dxbc.h"

namespace x2w::tool {

namespace fs = std::filesystem;

std::uint64_t fnv64(Bytes bytes) {
    std::uint64_t h = 1469598103934665603ull;
    for (std::uint8_t b : bytes) h = (h ^ std::uint64_t{b}) * 0x00000100000001b3ull;
    return h;
}

Bytes bounded(Bytes bytes) {
    if (bytes.size() < 28) return bytes;
    const std::size_t total = std::size_t{bytes[24]} | std::size_t{bytes[25]} << 8 | std::size_t{bytes[26]} << 16 |
                              std::size_t{bytes[27]} << 24;
    return total >= 32 && total <= bytes.size() ? bytes.first(total) : bytes;
}

std::uint64_t key_of(const Blake3Hash& full) {
    std::uint64_t k = 0;
    for (std::size_t i = 0; i < 8; ++i) k = k << 8 | full[i];
    return k;
}

std::string hex16(std::uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

namespace {

// `{:#x}` of a usize.
std::string hex_alt(std::uint64_t v) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "0x%llx", static_cast<unsigned long long>(v));
    return buf;
}

}  // namespace

std::optional<Bytes> Source::bytes(std::vector<std::uint8_t>& storage) const {
    if (!packed) return data;
    auto unpacked = record.unpack();
    if (!unpacked) return std::nullopt;
    storage = std::move(*unpacked);
    return Bytes(storage);
}

std::string Source::label() const {
    if (!packed) return "@" + hex_alt(offset);
    return "@" + hex_alt(record.offset) + " " + std::string(record.name ? *record.name : std::string_view("?"));
}

std::vector<Source> sources(Bytes data) {
    std::vector<Source> out;
    if (sgd5::is_sgd5(data)) {
        for (const auto& r : sgd5::records(data)) {
            Source s;
            s.packed = true;
            s.record = r;
            out.push_back(s);
        }
    }
    static constexpr std::uint8_t DXBC[4] = {'D', 'X', 'B', 'C'};
    for (std::size_t offset : sgd5::find_all(data, DXBC)) {
        const Bytes rest = data.subspan(offset);
        try {
            (void)vs::dxbc::Container::parse(rest);
        } catch (const std::exception&) {
            continue;
        }
        Source s;
        s.offset = offset;
        s.data = rest;
        out.push_back(s);
    }
    return out;
}

namespace {

void push_utf8(std::string& o, std::uint32_t c) {
    if (c < 0x80) {
        o.push_back(static_cast<char>(c));
    } else if (c < 0x800) {
        o.push_back(static_cast<char>(0xC0 | c >> 6));
        o.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        o.push_back(static_cast<char>(0xE0 | c >> 12));
        o.push_back(static_cast<char>(0x80 | (c >> 6 & 0x3F)));
        o.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    } else {
        o.push_back(static_cast<char>(0xF0 | c >> 18));
        o.push_back(static_cast<char>(0x80 | (c >> 12 & 0x3F)));
        o.push_back(static_cast<char>(0x80 | (c >> 6 & 0x3F)));
        o.push_back(static_cast<char>(0x80 | (c & 0x3F)));
    }
}

// Encode the UTF-16 path for display, replacing unpaired surrogates with U+FFFD.
std::string display(const fs::path& p) {
#ifdef _WIN32
    const std::wstring& w = p.native();
    std::string o;
    for (std::size_t i = 0; i < w.size(); ++i) {
        std::uint32_t c = static_cast<std::uint16_t>(w[i]);
        if (c >= 0xD800 && c < 0xDC00 && i + 1 < w.size()) {
            const std::uint32_t d = static_cast<std::uint16_t>(w[i + 1]);
            if (d >= 0xDC00 && d < 0xE000) {
                push_utf8(o, 0x10000 + ((c - 0xD800) << 10) + (d - 0xDC00));
                ++i;
                continue;
            }
        }
        if (c >= 0xD800 && c < 0xE000) c = 0xFFFD;
        push_utf8(o, c);
    }
    return o;
#else
    const auto& n = p.native();
    return utf8_lossy(Bytes(reinterpret_cast<const std::uint8_t*>(n.data()), n.size()));
#endif
}

bool read_file(const fs::path& p, std::vector<std::uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff size = f.tellg();
    if (size < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));
    if (size > 0 && !f.read(reinterpret_cast<char*>(out.data()), size)) return false;
    return true;
}

void visit(const fs::path& path, std::uintmax_t size,
           const std::function<void(const std::string& file, Bytes data)>& f) {
    if (size > MAX_FILE) return;
    std::vector<std::uint8_t> data;
    if (!read_file(path, data)) return;
    f(display(path), Bytes(data));
}

}  // namespace

void walk_files(const std::vector<fs::path>& roots, const std::function<void(const std::string&, Bytes)>& f) {
    for (const auto& root : roots) {
        std::error_code ec;
        // walkdir follows a root that is a link; it yields a root that is a file as itself.
        const fs::file_status st = fs::status(root, ec);
        if (ec) continue;
        if (fs::is_regular_file(st)) {
            const std::uintmax_t size = fs::file_size(root, ec);
            visit(root, ec ? 0 : size, f);
            continue;
        }
        if (!fs::is_directory(st)) continue;
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec);
        if (ec) continue;
        for (; it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            std::error_code e2;
            if (it->symlink_status(e2).type() != fs::file_type::regular || e2) continue;
            const std::uintmax_t size = it->file_size(e2);
            visit(it->path(), e2 ? 0 : size, f);
        }
    }
}

std::string json_float(float v) {
    if (!std::isfinite(v)) return "null";
    char buf[64];
    // Generate the shortest round-trip digits in d[.ddd]e[+-]XX form.
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    std::string sign;
    if (!s.empty() && s[0] == '-') {
        sign = "-";
        s.erase(0, 1);
    }
    const std::size_t e = s.find('e');
    std::string digits;
    for (std::size_t i = 0; i < e; ++i)
        if (s[i] != '.') digits.push_back(s[i]);
    const int exp = std::stoi(s.substr(e + 1));
    const float a = std::fabs(v);
    if ((a != 0.0f && a < 1e-4f) || a >= 1e16f) {
        std::string o = sign + digits.substr(0, 1);
        if (digits.size() > 1) o += "." + digits.substr(1);
        return o + "e" + std::to_string(exp);
    }
    if (a == 0.0f) return sign + "0.0";
    std::string o = sign;
    if (exp >= 0) {
        const std::size_t whole = static_cast<std::size_t>(exp) + 1;
        if (digits.size() <= whole) {
            o += digits + std::string(whole - digits.size(), '0') + ".0";
        } else {
            o += digits.substr(0, whole) + "." + digits.substr(whole);
        }
    } else {
        o += "0." + std::string(static_cast<std::size_t>(-exp - 1), '0') + digits;
    }
    return o;
}

std::string json_text(std::string_view s) {
    std::string o = "\"";
    for (char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"') {
            o += "\\\"";
        } else if (c == '\\') {
            o += "\\\\";
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            o += buf;
        } else {
            o.push_back(ch);
        }
    }
    o += "\"";
    return o;
}

std::string utf8_lossy(Bytes b) {
    std::string o;
    std::size_t i = 0;
    while (i < b.size()) {
        const std::uint8_t c = b[i];
        if (c < 0x80) {
            o.push_back(static_cast<char>(c));
            ++i;
            continue;
        }
        // The sequence's length and the range of its second byte (Unicode table 3-7).
        std::size_t need = 0;
        std::uint8_t lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) {
            need = 2;
        } else if (c >= 0xE0 && c <= 0xEF) {
            need = 3;
            if (c == 0xE0) lo = 0xA0;
            if (c == 0xED) hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 4;
            if (c == 0xF0) lo = 0x90;
            if (c == 0xF4) hi = 0x8F;
        }
        // Invalid: the lead byte and the continuation bytes that fit so far become one U+FFFD.
        std::size_t got = 1;
        bool ok = need != 0;
        while (ok && got < need) {
            if (i + got >= b.size()) {
                ok = false;
                break;
            }
            const std::uint8_t n = b[i + got];
            const std::uint8_t l = got == 1 ? lo : 0x80, h = got == 1 ? hi : 0xBF;
            if (n < l || n > h) {
                ok = false;
                break;
            }
            ++got;
        }
        if (ok) {
            o.append(reinterpret_cast<const char*>(b.data() + i), need);
        } else {
            o += "\xEF\xBF\xBD";
        }
        i += got;
    }
    return o;
}

namespace {

std::string sanitize(std::string why) {
    for (char& c : why)
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return why;
}

bool write_bytes(const fs::path& p, const void* data, std::size_t n) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    if (n) f.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

void write_tsv(const fs::path& p, const char* header, std::vector<std::string> lines) {
    std::sort(lines.begin(), lines.end());
    std::string text = header;
    text += "\n";
    for (const auto& l : lines) text += l + "\n";
    if (!write_bytes(p, text.data(), text.size())) throw std::runtime_error("writing " + display(p));
}

struct Shared {
    // Key -> BLAKE3 of the first container with it (collisions).
    std::map<std::uint64_t, Blake3Hash> keys;
    std::set<Blake3Hash> seen;
    std::vector<std::string> index, programs, sources, failed, collisions;
};

}  // namespace

DumpCounts dump(const DumpOptions& o, const std::function<Made(const Found&)>& make) {
    std::error_code ec;
    fs::create_directories(o.out, ec);
    if (ec || !fs::is_directory(o.out)) throw std::runtime_error("creating " + display(o.out));
    const std::size_t threads = std::max<std::size_t>(o.jobs, 1);
    const auto started = std::chrono::steady_clock::now();
    std::mutex mutex;
    Shared sh;
    std::atomic<std::size_t> translated{0}, failed{0};
    walk_files(o.paths, [&](const std::string& file, Bytes data) {
        const std::vector<Source> jobs = sources(data);
        std::atomic<std::size_t> next{0};
        auto work = [&] {
            for (std::size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < jobs.size();) {
                const Source& job = jobs[i];
                std::vector<std::uint8_t> storage;
                const auto raw = job.bytes(storage);
                if (!raw) continue;
                Found found;
                found.bytes = bounded(*raw);
                found.full = blake3(found.bytes);
                found.key = key_of(found.full);
                found.name = hex16(found.key);
                found.fnv = fnv64(found.bytes);
                found.file = file;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    sh.sources.push_back(found.name + "\t" + hex16(found.fnv) + "\t" + file + "\t" + job.label());
                    if (!sh.seen.insert(found.full).second) continue;
                    const auto it = sh.keys.find(found.key);
                    if (it != sh.keys.end() && it->second != found.full) {
                        sh.collisions.push_back(found.name + "\t" + to_hex(it->second) + "\t" + to_hex(found.full));
                        continue;
                    }
                    sh.keys[found.key] = found.full;
                }
                Made made;
                try {
                    made = make(found);
                } catch (const std::exception& e) {
                    // Record a failed container and continue exporting.
                    made = Made{};
                    made.failed = std::string("panic: ") + e.what();
                }
                if (made.failed) {
                    failed.fetch_add(1, std::memory_order_relaxed);
                    const std::string line = found.name + "\t" + hex16(found.fnv) + "\t" +
                                             std::to_string(found.bytes.size()) + "\t" + file + "\t" +
                                             sanitize(*made.failed);
                    std::lock_guard<std::mutex> lock(mutex);
                    sh.failed.push_back(line);
                    continue;
                }
                translated.fetch_add(1, std::memory_order_relaxed);
                const bool ok = write_bytes(o.out / (found.name + ".dxil"), made.dxil.data(), made.dxil.size()) &&
                                write_bytes(o.out / (found.name + ".json"), made.json.data(), made.json.size());
                if (!ok) continue;
                std::lock_guard<std::mutex> lock(mutex);
                sh.index.push_back(made.index_line);
                for (auto& l : made.program_lines) sh.programs.push_back(std::move(l));
            }
        };
        const std::size_t n = std::min(threads, jobs.size());
        std::vector<std::thread> pool;
        pool.reserve(n);
        for (std::size_t t = 0; t < n; ++t) pool.emplace_back(work);
        for (auto& t : pool) t.join();
    });
    DumpCounts c;
    c.containers = sh.seen.size();
    c.translated = translated.load();
    c.failed = failed.load();
    c.collisions = sh.collisions.size();
    write_tsv(o.out / "index.tsv", INDEX_HEADER, std::move(sh.index));
    write_tsv(o.out / "programs.tsv", PROGRAMS_HEADER, std::move(sh.programs));
    write_tsv(o.out / "sources.tsv", SOURCES_HEADER, std::move(sh.sources));
    write_tsv(o.out / "failed.tsv", FAILED_HEADER, std::move(sh.failed));
    write_tsv(o.out / "collisions.tsv", COLLISIONS_HEADER, std::move(sh.collisions));
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    char secs[64];
    std::snprintf(secs, sizeof secs, "%.0f", seconds);
    // No cache: nothing is cached or missing, and there are no orphan entries (no orphans/ and
    // orphans.tsv, and no orphan_entries line).
    c.summary = "build " + o.build + "\ncontainers " + std::to_string(c.containers) + "\ncached 0\ntranslated " +
                std::to_string(c.translated) + "\nmissing 0\nfailed " + std::to_string(c.failed) + "\nseconds " +
                secs + "\n";
    if (!write_bytes(o.out / "summary.txt", c.summary.data(), c.summary.size()))
        throw std::runtime_error("writing summary.txt");
    if (o.manifest) {
        const std::string m = o.manifest(c.translated);
        if (!write_bytes(o.out / "manifest.txt", m.data(), m.size())) throw std::runtime_error("writing manifest.txt");
    }
    return c;
}

}  // namespace x2w::tool
