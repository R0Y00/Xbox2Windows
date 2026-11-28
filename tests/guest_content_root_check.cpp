#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {
namespace fs = std::filesystem;

constexpr wchar_t archive[] = L"sample-content.rpf";
constexpr wchar_t missing[] = L"sample-missing.rpf";
constexpr wchar_t package_folder[] = L"package root";
constexpr wchar_t save_folder[] = L"save-config";
constexpr std::string_view package_bytes = "generated package archive";
constexpr std::string_view save_bytes = "generated save/config decoy";

struct File {
    File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
    HANDLE handle = INVALID_HANDLE_VALUE;
};

DWORD drive_bit(wchar_t letter)
{
    return DWORD{1} << (letter - L'A');
}

// A single process-local mapping for root-level synthetic archives.
struct ContentRoot {
    DWORD host_drives;
    wchar_t letter = 0;
    fs::path directory;

    DWORD open(std::wstring_view guest, File& file) const
    {
        if (!letter || guest.empty() || guest.front() != letter) return ERROR_INVALID_DRIVE;
        if (guest.size() < 4 || guest.substr(1, 2) != L":\\") return ERROR_INVALID_NAME;
        const auto name = guest.substr(3);
        if (name.size() < 12 || name.substr(0, 7) != L"sample-" ||
            name.substr(name.size() - 4) != L".rpf" ||
            name.find_first_not_of(L"abcdefghijklmnopqrstuvwxyz0123456789-.") != std::wstring_view::npos)
            return ERROR_INVALID_NAME;
        const auto path = directory / fs::path(name);
        file.handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                  FILE_ATTRIBUTE_NORMAL, nullptr);
        return file.handle == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
    }
};

struct ScopedContentRoot {
    ContentRoot& root;
    DWORD status = ERROR_ALREADY_ASSIGNED;

    ScopedContentRoot(ContentRoot& target, wchar_t letter, const fs::path& directory) : root(target)
    {
        if (root.letter || (root.host_drives & drive_bit(letter))) return;
        root.directory = directory;
        root.letter = letter;
        status = ERROR_SUCCESS;
    }
    ScopedContentRoot(const ScopedContentRoot&) = delete;
    ScopedContentRoot& operator=(const ScopedContentRoot&) = delete;

    ~ScopedContentRoot()
    {
        if (status == ERROR_SUCCESS) {
            root.letter = 0;
            root.directory.clear();
        }
    }
};

bool expect_error(const char* description, DWORD actual, DWORD expected)
{
    if (actual == expected) return true;
    std::fprintf(stderr, "%s: got Win32 error %lu, expected %lu\n", description, actual, expected);
    return false;
}

bool expect_bytes(const char* description, const File& file, std::string_view expected)
{
    std::array<char, 64> data{};
    DWORD read = 0;
    if (ReadFile(file.handle, data.data(), static_cast<DWORD>(data.size()), &read, nullptr) &&
        std::string_view(data.data(), read) == expected) return true;
    std::fprintf(stderr, "%s: archive byte check failed (read %lu bytes)\n", description, read);
    return false;
}

bool write_bytes(const fs::path& path, std::string_view data)
{
    std::ofstream stream(path, std::ios::binary);
    stream.write(data.data(), static_cast<std::streamsize>(data.size()));
    stream.close();
    return !!stream;
}

struct Scratch {
    const fs::path directory;

    ~Scratch()
    {
        const auto package = directory / package_folder;
        const auto saves = directory / save_folder;
        std::error_code error;
        fs::remove(package / archive, error);
        fs::remove(saves / archive, error);
        fs::remove(saves / missing, error);
        fs::remove(package, error);
        fs::remove(saves, error);
        fs::remove(directory, error);
    }
};

bool exercise(ContentRoot& root, wchar_t drive, wchar_t occupied, const Scratch& files)
{
    const auto package = files.directory / package_folder;
    const auto saves = files.directory / save_folder;
    if (!fs::create_directory(package) || !fs::create_directory(saves) ||
        !write_bytes(package / archive, package_bytes) || !write_bytes(saves / archive, save_bytes) ||
        !write_bytes(saves / missing, save_bytes)) {
        std::fputs("could not create synthetic fixture files\n", stderr);
        return false;
    }
    const std::wstring prefix = std::wstring(1, drive) + L":\\";
    const auto guest_archive = prefix + archive;
    {
        const ScopedContentRoot collision(root, occupied, saves);
        if (!expect_error("occupied host drive", collision.status, ERROR_ALREADY_ASSIGNED)) return false;
    }
    {
        const ScopedContentRoot binding(root, drive, package);
        File file;
        if (!expect_error("bind content", binding.status, ERROR_SUCCESS) ||
            !expect_error("open guest archive", root.open(guest_archive, file), ERROR_SUCCESS) ||
            !expect_bytes("read package archive", file, package_bytes)) return false;
        DWORD written = 0;
        const char byte = '!';
        if (WriteFile(file.handle, &byte, 1, &written, nullptr) ||
            !expect_error("content write", GetLastError(), ERROR_ACCESS_DENIED)) return false;
    }
    {
        File file;
        if (!expect_error("after successful scope", root.open(guest_archive, file), ERROR_INVALID_DRIVE))
            return false;
    }
    {
        const ScopedContentRoot binding(root, drive, package);
        File file;
        if (!expect_error("rebind content", binding.status, ERROR_SUCCESS) ||
            !expect_error("missing guest archive", root.open(prefix + missing, file), ERROR_FILE_NOT_FOUND))
            return false;
    }
    {
        File file;
        if (!expect_error("after missing-file scope", root.open(guest_archive, file), ERROR_INVALID_DRIVE))
            return false;
    }
    if (GetLogicalDrives() != root.host_drives) {
        std::fputs("host drive assignments changed\n", stderr);
        return false;
    }
    std::puts("package bytes matched; save/config decoy stayed outside content; occupied drive refused");
    std::puts("content was read-only; both scopes released their mapping; host drives unchanged");
    return true;
}
}

int wmain(int argc, wchar_t** argv) try
{
    if (argc != 2) {
        std::fputs("expected the build output directory\n", stderr);
        return 1;
    }
    const DWORD host = GetLogicalDrives();
    if (!host) {
        std::fprintf(stderr, "GetLogicalDrives failed (%lu)\n", GetLastError());
        return 1;
    }
    wchar_t occupied = 0;
    wchar_t available = 0;
    for (wchar_t letter = L'A'; letter <= L'Z'; ++letter) {
        if (host & drive_bit(letter)) occupied = letter;
        else available = letter;
    }
    if (!available) {
        std::puts("skipped: all drive letters are occupied");
        return 77;
    }
    const auto scratch = fs::canonical(argv[1]) / (L"guest-content-root-" + std::to_wstring(GetCurrentProcessId()));
    if (!fs::create_directory(scratch)) {
        std::fputs("fixture output directory already exists\n", stderr);
        return 1;
    }
    bool passed;
    {
        const Scratch files{scratch};
        ContentRoot root{host};
        passed = exercise(root, available, occupied, files);
    }
    if (fs::exists(scratch)) {
        std::fputs("fixture output directory was not cleaned up\n", stderr);
        return 1;
    }
    return passed ? 0 : 1;
}
catch (const std::exception& error) {
    std::fprintf(stderr, "content-root fixture failed: %s\n", error.what());
    return 1;
}
