#include <algorithm>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

// Extract the runtime's explicit translation misses. Other skipped work (bad
// heap, missing PSO, no targets, etc.) cannot be fixed by warming a shader.
static std::uint64_t missed_key(std::string_view line)
{
    const bool draw = line.rfind("x2w-frame draw ", 0) == 0 || line.rfind("x2w-frame draw-indexed ", 0) == 0;
    const bool compute = line.rfind("x2w-frame cx-uav #", 0) == 0;
    if (!draw && !compute) return 0;
    auto pos = line.find(" key ");
    if (pos == line.npos) return 0;
    if (draw && line.substr(0, pos).find(" NOT drawn target ") == line.npos) return 0;
    auto field = line.substr(pos + 5);
    if (field.size() < 17 || field[16] != ' ') return 0;
    auto tail = field.substr(17);
    if (draw && tail != "no translation" && tail != "translation failed") return 0;
    if (compute && tail.find(": skipped: no translation,") == tail.npos) return 0;
    std::uint64_t key = 0;
    auto end = field.data() + 16;
    auto result = std::from_chars(field.data(), end, key, 16);
    return result.ec == std::errc{} && result.ptr == end ? key : 0;
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: shader_trace_misses <trap.log>\n";
        return 2;
    }
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::cerr << "cannot open trace: " << argv[1] << '\n';
        return 1;
    }
    std::map<std::uint64_t, std::uint64_t> counts;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (auto key = missed_key(line)) ++counts[key];
    }
    if (input.bad() || !input.eof()) {
        std::cerr << "error reading trace\n";
        return 1;
    }
    using Entry = std::pair<std::uint64_t, std::uint64_t>;
    std::vector<Entry> ranked(counts.begin(), counts.end());
    std::sort(ranked.begin(), ranked.end(), [](const Entry& a, const Entry& b) {
        return a.second != b.second ? a.second > b.second : a.first < b.first;
    });
    for (auto [key, count] : ranked) {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << key
                  << '\t' << std::dec << count << '\n';
    }
    return std::cout ? 0 : 1;
}
