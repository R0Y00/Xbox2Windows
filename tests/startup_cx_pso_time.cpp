#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

using Microsoft::WRL::ComPtr;
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);

namespace {
std::string read_log(const std::wstring& path)
{
    std::ifstream file(path, std::ios::binary);
    return std::string{std::istreambuf_iterator<char>(file), {}};
}
void print_lines(const std::string& log, const char* what)
{
    for (std::size_t p = log.find("x2w-cx "); p != std::string::npos; p = log.find("x2w-cx ", p + 1)) {
        const std::size_t end = log.find('\n', p);
        const std::string line = log.substr(p, end == std::string::npos ? std::string::npos : end - p);
        if (line.find(what) != std::string::npos) std::printf("%s\n", line.c_str());
    }
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) return 2;
    const int seconds = argc > 3 ? std::atoi(argv[3]) : 300;
    ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 3;
    ComPtr<IDXGIAdapter> adapter;
    if (std::strcmp(argv[1], "hardware") == 0) {
        for (UINT n = 0;; ++n) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapterByGpuPreference(n, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&candidate)) == DXGI_ERROR_NOT_FOUND) return 77;
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(candidate->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                adapter = candidate;
                break;
            }
        }
    } else if (std::strcmp(argv[1], "warp") != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
        return 3;
    }
    DXGI_ADAPTER_DESC ad{};
    LARGE_INTEGER driver{};
    adapter->GetDesc(&ad);
    adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driver);
    std::printf("adapter=%ls driver=%u.%u.%u.%u\n", ad.Description, HIWORD(driver.HighPart), LOWORD(driver.HighPart), HIWORD(driver.LowPart),
                LOWORD(driver.LowPart));
    wchar_t dir[MAX_PATH];
    if (!MultiByteToWideChar(CP_UTF8, 0, argv[2], -1, dir, MAX_PATH)) return 4;
    const std::wstring log_path = std::wstring(dir) + L"\\trap.log";
    DeleteFileW(log_path.c_str());
    SetEnvironmentVariableW(L"X2W_SHADER_CACHE", dir);
    SetEnvironmentVariableW(L"X2W_TRAP_LOG", log_path.c_str());
    SetEnvironmentVariableW(L"X2W_CX_PREWARM_ALL", L"1");
    SetEnvironmentVariableW(L"X2W_CX_PREWARM_THREADS", L"1");
    SetEnvironmentVariableW(L"X2W_CX_SLOW_LOG_MS", L"5000");
    SetEnvironmentVariableW(L"X2W_D3D12_RENDER", L"1");
    SetEnvironmentVariableW(L"X2W_D3D12_PERMISSIVE", L"1");
    SetEnvironmentVariableW(L"X2W_D3D12_LIST_NOOP", L"1");
    std::array<unsigned char, 64> parameters{};
    void* device = nullptr;
    const ULONGLONG start = GetTickCount64();
    if (FAILED(D3D12XboxCreateDevice(adapter.Get(), parameters.data(), __uuidof(ID3D12Device), &device))) return 6;
    std::string log;
    bool done = false;
    while (!done && GetTickCount64() - start < static_cast<ULONGLONG>(seconds) * 1000) {
        Sleep(200);
        log = read_log(log_path);
        done = log.find("x2w-cx prewarm: all ") != std::string::npos;
    }
    print_lines(log, "known pipelines queued");
    print_lines(log, " translated: ");
    print_lines(log, ": load running for ");
    print_lines(log, "prewarm: all ");
    std::printf("%s after %.1f s\n", done ? "ALL LOADED" : "TIMEOUT", (GetTickCount64() - start) / 1000.0);
    std::fflush(stdout);
    TerminateProcess(GetCurrentProcess(), done ? 0 : 1);  // a driver compile still running cannot be joined
    return done ? 0 : 1;
}
