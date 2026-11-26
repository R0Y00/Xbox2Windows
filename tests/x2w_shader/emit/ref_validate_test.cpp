// The DXIL containers of the reference byte comparison (ref_test: the same bytes as the C++ port
// writes) pass dxc's validator. X2W_REF_DIR: the reference folder. Skipped without dxil.dll.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "test_util.h"
#include "x2w_shader/emit/validator.h"

int main() {
    const char* dir = std::getenv("X2W_REF_DIR");
    if (!dir) {
        std::fprintf(stderr, "X2W_REF_DIR not set\n");
        return 1;
    }
    std::string why;
    auto v = x2w::vs::validator::Validator::load(x2w::vs::validator::Validator::default_path(), &why);
    if (!v) {
        std::printf("ref_validate_test: no validator (%s); skipped\n", why.c_str());
        return 0;
    }
    for (const char* name : {"cs_basic", "ps_basic", "rect_prim", "rect_plain"}) {
        std::ifstream f(std::string(dir) + "/" + name + ".bin", std::ios::binary);
        std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (bytes.empty()) {
            std::fprintf(stderr, "%s: missing\n", name);
            ++test_util::failures();
            continue;
        }
        auto err = v->validate(bytes);
        if (err) {
            std::fprintf(stderr, "%s: refused: %s\n", name, err->c_str());
            ++test_util::failures();
        } else {
            std::printf("%s: valid (%zu bytes)\n", name, bytes.size());
        }
    }
    return test_util::finish("ref_validate_test");
}
