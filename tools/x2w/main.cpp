// x2w: X2W's command-line tool.
//
// x2w shaders <paths...> --out <dir> [--jobs N] translates every shader container under the input paths,
// including SGD5 records and standalone containers. Output files are described in shaders_dump.h.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <iostream>
#include <string>
#include <vector>

#include "shaders_dump.h"

namespace {

std::string utf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(static_cast<std::size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    s.pop_back();  // the NUL
    return s;
}

}  // namespace

// Wide arguments, as UTF-8 (paths outside the ANSI code page).
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.push_back(utf8(argv[i]));
    if (!args.empty() && args[0] == "shaders") {
        args.erase(args.begin());
        return x2w::tool::shaders_main(args);
    }
    std::cerr << "usage: x2w shaders <paths...> --out <dir> [--jobs N]\n";
    return 2;
}
