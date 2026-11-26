#include "startup_stubs/gpu_warm_bad.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace x2w_gpu;

namespace {
GpuCreateDesc compute(std::string_view shader)
{
    GpuCreateDesc desc{};
    desc.stage_count = 1;
    desc.stages[0].stage = GpuStage::compute;
    std::memcpy(desc.stages[0].key, shader.data(), 16);
    return desc;
}
}

int main(int argc, char** argv)
{
    const auto shader = "0123456789abcdef";
    const auto id = gpu_compute_pending_key(shader);
    const auto good = compute(shader);
    GpuWarmBad bad;
    if (bad.contains(good, false)) return 1;
    std::istringstream rows("   # comment\n\n\t" + id + "\tcomment\r\n" + id + " duplicate\n" +
                            "truncated " + gpu_compute_pending_key("0123456789abcdee") + "\n");
    bad.read(rows);
    if (bad.keys.size() != 1 || !bad.contains(good, false) || bad.contains(good, true)) return 2;
    if (bad.contains(compute("0123456789abcdee"), false)) return 3;
    auto other = good;
    other.stages[0].stage = GpuStage::pixel;
    if (bad.contains(other, false)) return 4;
    other = good;
    other.stage_count = 2;
    if (bad.contains(other, false)) return 5;

    if (argc == 2) {
        std::ifstream file(argv[1]);
        if (!file) return 6;
        int count = 0;
        for (std::string line; std::getline(file, line); ) {
            // The human-readable comments serve only as an independent reference fixture here.
            constexpr std::string_view marker = "[compute ";
            const auto pos = line.find(marker);
            if (pos == std::string::npos) continue;
            const auto end = line.find(']', pos);
            if (end != pos + marker.size() + 16) return 7;
            const auto desc = compute(std::string_view(line).substr(pos + marker.size(), 16));
            GpuWarmBad one;
            std::istringstream input(line);
            one.read(input);
            if (!one.contains(desc, false) || one.contains(desc, true)) return 8;
            ++count;
        }
        if (!count) return 9;
        std::printf("%d Rust compute annotations matched their independently recorded keys\n", count);
    }
    std::puts("warm-bad key filtering passed");
    return 0;
}
