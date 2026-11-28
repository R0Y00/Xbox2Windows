#include <cstdint>

extern "C" __declspec(dllexport) std::uint32_t write_result(std::uint64_t* output, std::uint64_t value)
{
    *output = value ^ 0x1981ee11d960cef1ULL;
    return 0x98231417;
}

extern "C" __declspec(dllexport) void write_constant(std::uint64_t* output)
{
    *output = 0x867594414c474131ULL;
}
