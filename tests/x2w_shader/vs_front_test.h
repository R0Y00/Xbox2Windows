// A minimal test harness for the vs-front ports' ctest executables: each
// TEST registers a case; main runs every case (or those named on the command
// line) and fails if any check failed.
#pragma once

#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

namespace vs_front_test {

struct Case {
    const char* name;
    void (*run)();
};

inline std::vector<Case>& cases()
{
    static std::vector<Case> all;
    return all;
}

inline int& failures()
{
    static int n = 0;
    return n;
}

struct Register {
    Register(const char* name, void (*run)()) { cases().push_back({name, run}); }
};

inline void fail(const char* file, int line, const std::string& what)
{
    std::fprintf(stderr, "%s:%d: check failed: %s\n", file, line, what.c_str());
    ++failures();
}

struct Abort {};

inline int run_all(int argc, char** argv)
{
    int ran = 0;
    for (const Case& c : cases()) {
        bool wanted = argc <= 1;
        for (int i = 1; i < argc; ++i) wanted |= std::strcmp(argv[i], c.name) == 0;
        if (!wanted) continue;
        ++ran;
        const int before = failures();
        try {
            c.run();
        } catch (const Abort&) {
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s: exception: %s\n", c.name, e.what());
            ++failures();
        }
        std::fprintf(stderr, "%s %s\n", failures() == before ? "ok  " : "FAIL", c.name);
    }
    if (ran == 0) {
        std::fprintf(stderr, "no test case matched\n");
        return 1;
    }
    return failures() == 0 ? 0 : 1;
}

}  // namespace vs_front_test

#define VS_TEST(name)                                                       \
    static void name();                                                     \
    static ::vs_front_test::Register name##_registered(#name, &name);       \
    static void name()

#define VS_CHECK(cond)                                                      \
    do {                                                                    \
        if (!(cond)) ::vs_front_test::fail(__FILE__, __LINE__, #cond);      \
    } while (0)

// A check the rest of the case depends on.
#define VS_REQUIRE(cond)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            ::vs_front_test::fail(__FILE__, __LINE__, #cond);               \
            throw ::vs_front_test::Abort{};                                 \
        }                                                                   \
    } while (0)

#define VS_TEST_MAIN()                                                      \
    int main(int argc, char** argv) { return ::vs_front_test::run_all(argc, argv); }
