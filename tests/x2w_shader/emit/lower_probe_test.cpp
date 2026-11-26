// Port of the upstream Rust module emit/probe_tests.rs.
//
// Not a test of the translator: what the host's compiler makes of DXIL. X2W_SHADERCFG_PROBE=<folder>
// and `--ignored` (Rust `#[ignore]`, `--ignored probe`): each `.dxil` there (a compute program
// with a constant buffer b0 of four dwords and a raw buffer, slot 0 of the view heap) is run on the
// card. `sem*` programs are given the bits of 0.0, infinity, -0.0 and a NaN and what they store is
// printed; of the others the time is (several runs of 4096 groups).

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "lower_card_util.h"

using namespace card;

namespace {

// Rust `{}` of an f32 (shortest round trip; "inf", "NaN").
std::string display(float f) {
    if (std::isnan(f)) return "NaN";
    char b[64];
    auto r = std::to_chars(b, b + sizeof b, f);
    return std::string(b, r.ptr);
}

// ---- rs:11 --------------------------------------------------------------------------------------
void probe() {
    x2w::vs::lower::test::set_dump_name("emit__probe_tests__probe");
    const char* folder = std::getenv("X2W_SHADERCFG_PROBE");
    if (!folder) return;
    std::string why;
    auto host = new_host(false, &why);
    if (!host) fail("a card: " + why);
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    std::filesystem::directory_iterator dir(folder, ec);
    if (ec) fail("the folder: " + ec.message());
    for (const auto& e : dir)
        if (e.path().extension() == ".dxil") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
        std::string name = file.stem().string();
        HostShader shader;
        shader.stage = HwStage::Cs;
        std::ifstream in(file, std::ios::binary);
        if (!in) fail("the file");
        shader.dxil.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        for (std::uint32_t n = 0; n < 4; ++n) {
            UserData u;
            u.index = n;
            u.slot = 0;
            u.root = RootType::Constants();
            u.dword = n;
            shader.user_data.push_back(u);
        }
        auto pipe = compute_pipeline(*host, shader, &why);
        if (!pipe) {
            std::printf("%s: %s\n", name.c_str(), why.c_str());
            continue;
        }
        auto run_once = [&](std::array<std::uint32_t, 4> constants, std::uint32_t groups) {
            auto began = std::chrono::steady_clock::now();
            const View views[] = {raw_uav(64)};
            auto out = dispatch(*host, *pipe, std::vector<std::uint32_t>(constants.begin(), constants.end()), views,
                                {}, {groups, 1, 1}, "a dispatch");
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
            return std::pair{out[0], ms};
        };
        if (name.starts_with("sem")) {
            auto [out, ms] = run_once({0, 0x7F80'0000, 0x8000'0000, 0x7FC0'0000}, 1);
            (void)ms;
            for (std::size_t n = 0; n < out.size() && n < 40; ++n)
                std::printf("%s %2zu: %08x %s\n", name.c_str(), n, out[n], display(f32_from_bits(out[n])).c_str());
            continue;
        }
        const std::array<std::uint32_t, 4> constants = {f32_bits(0.37f), f32_bits(0.11f), 16384, 0};
        run_once(constants, 64);
        std::vector<double> times;
        for (int k = 0; k < 9; ++k) times.push_back(run_once(constants, 4096).second);
        // Rust sort_by(f64::total_cmp): the times are finite and non-negative here.
        std::sort(times.begin(), times.end());
        std::printf("%s: %.2f ms (least %.2f)\n", name.c_str(), times[times.size() / 2], times[0]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Rust #[ignore]: runs only with --ignored.
    bool ignored = false;
    for (int i = 1; i < argc; ++i) ignored |= std::string_view(argv[i]) == "--ignored";
    if (ignored)
        guarded("probe", probe);
    else
        std::printf("probe: ignored (a tool: X2W_SHADERCFG_PROBE=<folder>, --ignored)\n");
    return test_util::finish("lower_probe_test");
}
