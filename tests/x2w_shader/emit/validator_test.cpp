// validator.cpp (port of emit/validator.rs): the validator accepts and signs a container dxc.exe
// compiled at test time, and refuses a corrupted one. Skips (and says so) without dxil.dll or
// dxc.exe.

#include "x2w_shader/emit/validator.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "test_util.h"

namespace fs = std::filesystem;
using x2w::vs::validator::Validator;

namespace {

// Compiles hlsl with dxc.exe (next to dxil.dll, as diff_tests.rs finds it); nullopt when it
// fails.
std::optional<std::vector<std::uint8_t>> dxc(const fs::path& dxc_exe, const char* hlsl, const char* profile) {
    const fs::path dir = fs::temp_directory_path() / ("x2w_vs_validator_test_" + std::to_string(GetCurrentProcessId()));
    fs::create_directories(dir);
    const fs::path src = dir / "shader.hlsl";
    const fs::path out = dir / "shader.dxil";
    std::ofstream(src, std::ios::binary) << hlsl;
    // The whole command quoted once more: cmd.exe strips the outer pair.
    const std::string cmd = "\"\"" + dxc_exe.string() + "\" -nologo -T " + profile + " -E main -Fo \"" +
                            out.string() + "\" \"" + src.string() + "\"\"";
    const int rc = std::system(cmd.c_str());
    std::ifstream in(out, std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::error_code ec;
    fs::remove_all(dir, ec);
    if (rc != 0 || bytes.empty()) return std::nullopt;
    return bytes;
}

// The offset of the data of the container's first part with this tag, or 0.
std::size_t part_data(const std::vector<std::uint8_t>& c, const char (&tag)[5]) {
    const std::uint32_t count = test_util::u32le(c, 28);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t at = test_util::u32le(c, 32 + 4 * i);
        if (std::equal(tag, tag + 4, c.begin() + at)) return at + 8;
    }
    return 0;
}

const char* CS = R"(
RWByteAddressBuffer o : register(u0);
[numthreads(4, 1, 1)]
void main(uint i : SV_DispatchThreadID) { o.Store(i * 4, i * 3 + 1); }
)";

}  // namespace

int main() {
    const std::optional<fs::path> path = Validator::default_path();
    if (!path) {
        std::printf("validator_test: skipped (no dxil.dll)\n");
        return 0;
    }
    std::printf("validator_test: dxil.dll at %s\n", path->string().c_str());

    // A missing DLL is an error, not a crash.
    {
        std::string error;
        auto none = Validator::load(fs::path("C:/no/such/dir/dxil.dll"), &error);
        CHECK(!none);
        CHECK(error.find("cannot load") == 0);
    }

    std::string error;
    std::optional<Validator> v = Validator::load(path, &error);
    CHECK(v.has_value());
    if (!v) {
        std::fprintf(stderr, "load: %s\n", error.c_str());
        return test_util::finish("validator_test");
    }

    fs::path dxc_exe = *path;
    dxc_exe.replace_filename("dxc.exe");
    std::optional<std::vector<std::uint8_t>> compiled = dxc(dxc_exe, CS, "cs_6_6");
    if (!compiled) {
        std::printf("validator_test: dxc.exe unavailable, container checks skipped\n");
        return test_util::finish("validator_test");
    }
    CHECK(compiled->size() > 32 && std::equal(compiled->begin(), compiled->begin() + 4, "DXBC"));
    const std::vector<std::uint8_t> original_digest(compiled->begin() + 4, compiled->begin() + 20);
    CHECK(std::any_of(original_digest.begin(), original_digest.end(), [](std::uint8_t b) { return b != 0; }));

    // Accepted as dxc wrote it.
    {
        std::vector<std::uint8_t> c = *compiled;
        auto r = v->validate(c);
        CHECK(!r);
        if (r) std::fprintf(stderr, "rejected: %s\n", r->c_str());
        CHECK(c == *compiled);
    }

    // With the digest cleared, validation signs it in place: dxc's digest comes back.
    {
        std::vector<std::uint8_t> c = *compiled;
        std::fill(c.begin() + 4, c.begin() + 20, std::uint8_t{0});
        auto r = v->validate(c);
        CHECK(!r);
        if (r) std::fprintf(stderr, "rejected: %s\n", r->c_str());
        CHECK(std::vector<std::uint8_t>(c.begin() + 4, c.begin() + 20) == original_digest);
        CHECK(c == *compiled);
    }

    // A broken bitcode magic in the DXIL part is refused, with the validator's text.
    {
        std::vector<std::uint8_t> c = *compiled;
        const std::size_t dxil = part_data(c, "DXIL");
        CHECK(dxil != 0);
        // Program header: version, size in dwords, "DXIL", DXIL version, bitcode offset, size.
        const std::size_t bitcode = dxil + 8 + test_util::u32le(c, dxil + 16);
        CHECK(c[bitcode] == 'B' && c[bitcode + 1] == 'C');
        c[bitcode] = 'X';
        c[bitcode + 1] = 'Y';
        auto r = v->validate(c);
        CHECK(r.has_value());
        if (r) std::printf("validator_test: corrupted bitcode refused: %.120s\n", r->c_str());
    }

    // A PSV0 that disagrees with the module is refused: the validator rebuilds the part from the
    // module and compares. (PSV0: the runtime-info size, then the info; byte 4 is in the info.)
    {
        std::vector<std::uint8_t> c = *compiled;
        const std::size_t psv = part_data(c, "PSV0");
        CHECK(psv != 0);
        c[psv + 4] ^= 0x40;
        auto r = v->validate(c);
        CHECK(r.has_value());
        if (r) std::printf("validator_test: corrupted PSV0 refused: %.120s\n", r->c_str());
    }

    return test_util::finish("validator_test");
}
