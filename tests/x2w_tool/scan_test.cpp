// scan.h on synthetic files: fnv64 and bounded (the tests in the upstream export_x2w.rs), the export's
// text helpers against Rust's output, sources() over raw and SGD5-packed containers, the walk, and
// dump() with a stand-in translator (files, rows, sorting, deduplication, failures, threads).
#include <algorithm>
#include <bit>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <process.h>
#include <sstream>
#include <string>
#include <vector>

#include "scan.h"
#include "test_util.h"
#include "x2w_shader/dxbc.h"

namespace fs = std::filesystem;
using namespace x2w::tool;
using x2w::vs::dxbc::FourCC;

namespace {

using Buf = std::vector<std::uint8_t>;

Buf container(std::uint8_t fill, std::size_t n) {
    const Buf data(n, fill);
    return x2w::vs::dxbc::build({{FourCC::DXIL, x2w::vs::dxbc::Bytes(data)}});
}

void put32(Buf& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(std::uint8_t(v >> (8 * i)));
}

// An SGD5 record holding `payload` in one stored deflate block.
Buf record(const std::string& name, std::uint8_t kind, const Buf& payload) {
    Buf packed = {0x01, std::uint8_t(payload.size()), std::uint8_t(payload.size() >> 8),
                  std::uint8_t(~payload.size()), std::uint8_t(~payload.size() >> 8)};
    packed.insert(packed.end(), payload.begin(), payload.end());
    const auto n = std::uint32_t(packed.size() + 12);
    Buf out = {std::uint8_t(name.size() + 1)};
    out.insert(out.end(), name.begin(), name.end());
    out.push_back(0);
    out.push_back(kind);
    for (std::uint32_t v : {n, 0x000DDCE7u, n, std::uint32_t(payload.size())}) put32(out, v);
    out.insert(out.end(), packed.begin(), packed.end());
    return out;
}

void append(Buf& a, const Buf& b) { a.insert(a.end(), b.begin(), b.end()); }

std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

void write(const fs::path& p, const Buf& b) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
}

std::vector<std::string> lines(const std::string& text) {
    std::vector<std::string> out;
    std::size_t at = 0;
    while (at < text.size()) {
        const std::size_t nl = text.find('\n', at);
        out.push_back(text.substr(at, nl - at));
        at = nl == std::string::npos ? text.size() : nl + 1;
    }
    return out;
}

void fnv_and_bounded() {
    // X2W's basis.
    CHECK(fnv64({}) == 0x14650fb0739d0383ull);
    const std::uint8_t a = 'a';
    CHECK(fnv64(Bytes(&a, 1)) == 0x44bd8ad473cd9906ull);

    // Containers are cut to their size.
    Buf c = {'D', 'X', 'B', 'C'};
    c.resize(24, 0);
    put32(c, 40);
    c.resize(48, 7);
    CHECK(bounded(c).size() == 40);
    // A size below the header or past the bytes leaves them as they are.
    Buf small = c;
    small[24] = 31;
    CHECK(bounded(small).size() == 48);
    Buf big = c;
    big[24] = 49;
    CHECK(bounded(big).size() == 48);
    CHECK(bounded(Bytes(c).first(27)).size() == 27);

    Blake3Hash h{};
    h[0] = 0x01;
    h[7] = 0xef;
    h[8] = 0xff;
    CHECK(key_of(h) == 0x01000000000000efull);
    CHECK(hex16(key_of(h)) == "01000000000000ef");
}

void text_helpers() {
    // Rust `{:?}` of these f32 values (printed by a scratch Rust program).
    const struct {
        std::uint32_t bits;
        const char* text;
    } floats[] = {
        {0x00000000u, "0.0"},
        {0x80000000u, "-0.0"},
        {0x3f800000u, "1.0"},
        {0xbf800000u, "-1.0"},
        {0x3dcccccdu, "0.1"},
        {0x3f000000u, "0.5"},
        {0x447a0000u, "1000.0"},
        {0x41800000u, "16.0"},
        {0x7f7fffffu, "3.4028235e38"},
        {0xff7fffffu, "-3.4028235e38"},
        {0x00800000u, "1.1754944e-38"},
        {0x00000001u, "1e-45"},
        {0x38d1b717u, "0.0001"},
        {0x38d1b68eu, "9.9999e-5"},
        {0x3727c5acu, "1e-5"},
        {0x377ba882u, "1.5e-5"},
        {0x5a0e1bcau, "1e16"},
        {0x5a0e1bc9u, "9999999000000000.0"},
        {0x58635fa9u, "1000000000000000.0"},
        {0x4ceb79a3u, "123456790.0"},
        {0x3b449ba6u, "0.003"},
        {0xc1540000u, "-13.25"},
    };
    for (const auto& f : floats) {
        const std::string got = json_float(std::bit_cast<float>(f.bits));
        if (got != f.text) std::fprintf(stderr, "%08x: %s, Rust %s\n", f.bits, got.c_str(), f.text);
        CHECK(got == f.text);
    }
    CHECK(json_float(std::bit_cast<float>(0x7f800000u)) == "null");
    CHECK(json_float(std::bit_cast<float>(0xff800000u)) == "null");
    CHECK(json_float(std::bit_cast<float>(0x7fc00000u)) == "null");

    CHECK(json_text("a\"b\\c") == "\"a\\\"b\\\\c\"");
    CHECK(json_text(std::string("x\ty\n\x01", 5)) == "\"x\\u0009y\\u000a\\u0001\"");
    CHECK(json_text("\xc3\xa9") == "\"\xc3\xa9\"");

    auto lossy = [](std::initializer_list<std::uint8_t> b) { return utf8_lossy(Buf(b)); };
    CHECK(lossy({'D', 'X', 'I', 'L'}) == "DXIL");
    CHECK(lossy({0xff, 'a'}) == "\xEF\xBF\xBD" "a");
    CHECK(lossy({0xe2, 0x82, 'a'}) == "\xEF\xBF\xBD" "a");                 // a cut sequence: one U+FFFD
    CHECK(lossy({0xe0, 0x80, 'a'}) == "\xEF\xBF\xBD\xEF\xBF\xBD" "a");     // overlong: lead, then 0x80
    CHECK(lossy({0xed, 0xa0, 0x80}) == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD");  // a surrogate
    CHECK(lossy({0xf0, 0x9f, 0x98, 0x80}) == "\xF0\x9F\x98\x80");
    CHECK(lossy({0xf0, 0x9f, 0x98}) == "\xEF\xBF\xBD");

    const std::vector<int> v = {1, 2, 3};
    CHECK(json_list(v, [](int x) { return std::to_string(x); }) == "[1,2,3]");
    CHECK(json_list(std::vector<int>{}, [](int x) { return std::to_string(x); }) == "[]");
}

void finds_sources() {
    const Buf a = container(0xA1, 8), b = container(0xB2, 12), packed = container(0xC3, 20);
    Buf file = {'x', 'y'};
    append(file, a);
    // "DXBC" that does not parse is not a container.
    append(file, {'D', 'X', 'B', 'C', 0, 0});
    append(file, {'S', 'G', 'D', '5'});
    append(file, record("PS_one_Wrapped", 0x32, packed));
    append(file, b);

    const auto found = sources(file);
    // (The record's stored block holds its container as is: the scan finds that copy too, as
    // the reference scan does.)
    CHECK(found.size() == 4);
    if (found.size() != 4) return;
    // The records first, then the containers stored as is, in file order.
    CHECK(found[0].packed);
    Buf storage;
    auto bytes = found[0].bytes(storage);
    CHECK(bytes && Buf(bytes->begin(), bytes->end()) == packed);
    CHECK(!found[1].packed && found[1].offset == 2);
    bytes = found[1].bytes(storage);
    CHECK(bytes && bounded(*bytes).size() == a.size());
    CHECK(Buf(bounded(*bytes).begin(), bounded(*bytes).end()) == a);
    // Raw bytes run to the end of the file.
    CHECK(bytes->size() == file.size() - 2);
    const std::size_t b_at = file.size() - b.size();
    CHECK(!found[2].packed && found[2].offset == found[0].record.offset + 12 + 5);
    CHECK(!found[3].packed && found[3].offset == b_at);

    char label[64];
    std::snprintf(label, sizeof label, "@%#zx PS_one_Wrapped", found[0].record.offset);
    CHECK(found[0].label() == label);
    CHECK(found[1].label() == "@0x2");
    std::snprintf(label, sizeof label, "@%#zx", b_at);
    CHECK(found[3].label() == label);
    // The SGD5 container that holds a raw container too: the packed one's bytes are not raw.
    // A record without a name is labelled "?"; a corrupt record has no bytes.
    Buf unnamed = {'S', 'G', 'D', '5'};
    Buf r = record("PS_x", 0x32, packed);
    r[0] = 9;  // the length byte disagrees
    append(unnamed, r);
    auto u = sources(unnamed);
    CHECK(u.size() == 2 && u[0].packed && u[0].label().ends_with(" ?"));
    Buf corrupt = {'S', 'G', 'D', '5'};
    Buf rc = record("PS_x", 0x32, packed);
    rc[1 + 4 + 2 + 16] = 0x07;  // first packed byte: BFINAL 1, reserved block type
    append(corrupt, rc);
    u = sources(corrupt);
    CHECK(u.size() == 2 && u[0].packed && !u[0].bytes(storage) && !u[1].packed);
    // Without "SGD5" the records are not looked for (only the stored copy is found).
    CHECK(sources(r).size() == 1 && !sources(r)[0].packed);
    // "0x0" for a container at the start.
    CHECK(sources(a).size() == 1 && sources(a)[0].label() == "@0x0");
}

fs::path scratch(const char* name) {
    const fs::path p = fs::temp_directory_path() / ("x2w_tool_scan_test_" + std::to_string(_getpid()) + name);
    std::error_code ec;
    fs::remove_all(p, ec);
    fs::create_directories(p);
    return p;
}

void walks() {
    const fs::path root = scratch("_walk");
    write(root / "a.bin", {1});
    write(root / "sub" / "b.bin", {2, 3});
    write(root / "sub" / "deeper" / "c.bin", {4, 5, 6});
    write(root / "z.bin", {});
    std::vector<std::pair<std::string, std::size_t>> seen;
    walk_files({root}, [&](const std::string& file, Bytes data) { seen.emplace_back(file, data.size()); });
    CHECK(seen.size() == 4);
    const std::string r = root.u8string().size() ? reinterpret_cast<const char*>(root.u8string().c_str()) : "";
    auto has = [&](const std::string& name, std::size_t n) {
        return std::find(seen.begin(), seen.end(), std::make_pair(r + name, n)) != seen.end();
    };
    CHECK(has("\\a.bin", 1));
    CHECK(has("\\sub\\b.bin", 2));
    CHECK(has("\\sub\\deeper\\c.bin", 3));
    CHECK(has("\\z.bin", 0));
    // A root that is a file is itself; a missing root is nothing.
    seen.clear();
    walk_files({root / "sub" / "b.bin", root / "missing"}, [&](const std::string& file, Bytes data) {
        seen.emplace_back(file, data.size());
    });
    CHECK(seen.size() == 1 && seen[0].first == r + "\\sub\\b.bin");
    std::error_code ec;
    fs::remove_all(root, ec);
}

void dumps() {
    const fs::path in = scratch("_in"), out = scratch("_out") / "export";
    // Twenty containers in one file (threads share it), one of them twice; the same container in a
    // second file; one the stand-in translator refuses.
    Buf one = {'S', 'G', 'D', '5'};
    std::vector<Buf> cs;
    for (std::uint8_t i = 0; i < 20; ++i) cs.push_back(container(i, 4 + i));
    for (const auto& c : cs) append(one, c);
    append(one, record("PS_again", 0x32, cs[3]));
    write(in / "one.bin", one);
    Buf two = cs[5];
    // Bytes after the container's size do not make it another container.
    append(two, {9, 9, 9});
    write(in / "two" / "two.bin", two);
    const Buf refused = container(0xEE, 6);
    write(in / "two" / "three.bin", refused);
    const Bytes refused_bytes(refused);
    const std::uint64_t refused_key = key_of(blake3(refused_bytes));

    DumpOptions o;
    o.out = out;
    o.paths = {in};
    o.jobs = 3;
    o.build = "b";
    o.manifest = [](std::size_t records) { return "records " + std::to_string(records) + "\n"; };
    std::mutex m;
    std::size_t calls = 0;
    const DumpCounts c = dump(o, [&](const Found& f) {
        {
            std::lock_guard<std::mutex> lock(m);
            ++calls;
        }
        Made made;
        if (f.key == refused_key) {
            made.failed = "no\tgood\nat all";
            return made;
        }
        made.dxil.assign(f.bytes.begin(), f.bytes.end());
        made.dxil.push_back(0x5A);
        made.json = "{\"key\": " + json_text(f.name) + "}\n";
        made.index_line = f.name + "\t" + hex16(f.fnv) + "\t" + to_hex(f.full) + "\t" + f.file;
        made.program_lines = {hex16(f.fnv) + "\tcode\t" + f.name};
        return made;
    });
    CHECK(calls == 21);
    CHECK(c.containers == 21 && c.translated == 20 && c.failed == 1 && c.collisions == 0);

    const auto index = lines(read(out / "index.tsv"));
    CHECK(index.size() == 21 && index[0] == INDEX_HEADER);
    CHECK(std::is_sorted(index.begin() + 1, index.end()));
    for (const auto& cb : cs) {
        const Blake3Hash full = blake3(cb);
        const std::string name = hex16(key_of(full));
        const std::string dxil = read(out / (name + ".dxil"));
        CHECK(dxil.size() == cb.size() + 1 && std::equal(cb.begin(), cb.end(), dxil.begin()));
        CHECK(read(out / (name + ".json")) == "{\"key\": \"" + name + "\"}\n");
        const auto row = std::find_if(index.begin(), index.end(), [&](const std::string& l) { return l.starts_with(name); });
        CHECK(row != index.end());
        // cs[5] is first found in one.bin (walked first: "one.bin" before "two").
        if (row != index.end()) CHECK(row->ends_with("one.bin"));
        if (row != index.end()) CHECK(row->find(to_hex(full)) != std::string::npos);
    }
    const auto sources_rows = lines(read(out / "sources.tsv"));
    // 20 raw + 1 packed (and its stored copy, raw) in one.bin, 1 in two.bin, 1 in three.bin.
    CHECK(sources_rows.size() == 1 + 24 && sources_rows[0] == SOURCES_HEADER);
    CHECK(std::is_sorted(sources_rows.begin() + 1, sources_rows.end()));
    const std::string again = hex16(key_of(blake3(cs[3])));
    CHECK(std::count_if(sources_rows.begin(), sources_rows.end(), [&](const std::string& l) {
              return l.starts_with(again) && l.ends_with(" PS_again");
          }) == 1);
    const auto failed = lines(read(out / "failed.tsv"));
    CHECK(failed.size() == 2 && failed[0] == FAILED_HEADER);
    if (failed.size() == 2) {
        CHECK(failed[1].starts_with(hex16(refused_key) + "\t" + hex16(fnv64(refused)) + "\t" +
                                    std::to_string(refused.size()) + "\t"));
        CHECK(failed[1].ends_with("three.bin\tno good at all"));
    }
    CHECK(read(out / "collisions.tsv") == std::string(COLLISIONS_HEADER) + "\n");
    const auto programs = lines(read(out / "programs.tsv"));
    CHECK(programs.size() == 21 && programs[0] == PROGRAMS_HEADER);
    CHECK(read(out / "manifest.txt") == "records 20\n");
    const std::string summary = read(out / "summary.txt");
    CHECK(summary.starts_with("build b\ncontainers 21\ncached 0\ntranslated 20\nmissing 0\nfailed 1\nseconds "));
    CHECK(summary == c.summary);
    // No carriage returns anywhere.
    CHECK(read(out / "index.tsv").find('\r') == std::string::npos);
    CHECK(!fs::exists(out / "orphans.tsv") && !fs::exists(out / "orphans"));
    std::error_code ec;
    fs::remove_all(in, ec);
    fs::remove_all(out.parent_path(), ec);
}

}  // namespace

int main() {
    fnv_and_bounded();
    text_helpers();
    finds_sources();
    walks();
    dumps();
    return test_util::finish("x2w_tool_scan_test");
}
