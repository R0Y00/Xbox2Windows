// Synthetic title-level check for XGameSaveInitializeProviderAsync and
// XGameSaveInitializeProviderResult in the fixture-only xgameruntime.dll. The DLL
// is loaded from its explicit fixture path. The block mirrors the three leading
// public XAsyncBlock fields and pads the rest with a sentinel pattern, which
// the shim must leave untouched because it keeps its own state.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace {
namespace fs = std::filesystem;

constexpr HRESULT not_implemented = static_cast<HRESULT>(0x80004001L);
constexpr std::uint64_t synthetic_user_id = 0x0009000000000001ull;
constexpr HRESULT e_illegal_state_change = static_cast<HRESULT>(0x8000000DL);
constexpr std::uint8_t sentinel = 0xA5;

namespace xgameruntime_shim {
struct XUserHandleStorage;
struct XGameSaveProviderStorage;
struct XGameSaveContainerStorage;
}
using XUserHandle = xgameruntime_shim::XUserHandleStorage*;
using XGameSaveProviderHandle = xgameruntime_shim::XGameSaveProviderStorage*;
using XGameSaveContainerHandle = xgameruntime_shim::XGameSaveContainerStorage*;

struct XAsyncBlock;
using CompletionRoutine = void CALLBACK(XAsyncBlock*);
struct XAsyncBlock {
    void* queue;
    void* context;
    CompletionRoutine* callback;
    std::uint8_t opaque[sizeof(void*) * 4];
};

struct XGameRuntimeShimLogEntry {
    std::uint32_t sequence;
    HRESULT result;
    const char* component;
    const char* function;
};

using FindUserById = HRESULT(WINAPI*)(std::uint64_t, XUserHandle*);
using CloseUserHandle = void(WINAPI*)(XUserHandle);
using InitializeProviderAsync = HRESULT(WINAPI*)(XUserHandle, const char*, bool, XAsyncBlock*);
using InitializeProviderResult = HRESULT(WINAPI*)(XAsyncBlock*, XGameSaveProviderHandle*);
using CreateContainer = HRESULT(WINAPI*)(XGameSaveProviderHandle, const char*,
                                        XGameSaveContainerHandle*);
using CloseProvider = void(WINAPI*)(XGameSaveProviderHandle);
using CloseContainer = void(WINAPI*)(XGameSaveContainerHandle);
using GetLog = std::size_t(WINAPI*)(XGameRuntimeShimLogEntry*, std::size_t);
using ClearLog = void(WINAPI*)();

struct CallbackProbe {
    int calls = 0;
    XAsyncBlock* seen = nullptr;
    XGameSaveProviderHandle result_inside = nullptr;
    HRESULT result_status = E_FAIL;
    InitializeProviderResult result_proc = nullptr;
};

void CALLBACK record_completion(XAsyncBlock* block)
{
    auto* probe = static_cast<CallbackProbe*>(block->context);
    ++probe->calls;
    probe->seen = block;
}

// Claims the result from inside the callback, as a title commonly does.
void CALLBACK claim_in_callback(XAsyncBlock* block)
{
    auto* probe = static_cast<CallbackProbe*>(block->context);
    ++probe->calls;
    probe->seen = block;
    probe->result_status = probe->result_proc(block, &probe->result_inside);
}

XAsyncBlock make_block(CallbackProbe* probe, CompletionRoutine* callback)
{
    XAsyncBlock block{};
    block.context = probe;
    block.callback = callback;
    std::memset(block.opaque, sentinel, sizeof(block.opaque));
    return block;
}

bool opaque_untouched(const XAsyncBlock& block)
{
    for (const std::uint8_t byte : block.opaque) {
        if (byte != sentinel) return false;
    }
    return true;
}

bool expect(bool condition, const char* what)
{
    if (!condition) std::printf("FAIL: %s\n", what);
    return condition;
}

bool expect_hr(HRESULT actual, HRESULT expected, const char* what)
{
    if (actual == expected) return true;
    std::printf("FAIL: %s returned 0x%08lx, expected 0x%08lx\n", what,
                static_cast<unsigned long>(actual), static_cast<unsigned long>(expected));
    return false;
}

fs::path normalized_path(const fs::path& path)
{
    std::error_code error;
    const auto canonical = fs::weakly_canonical(path, error);
    return error ? fs::absolute(path).lexically_normal() : canonical;
}

bool same_path(const fs::path& left, const fs::path& right)
{
    return _wcsicmp(normalized_path(left).c_str(), normalized_path(right).c_str()) == 0;
}

bool fixture_path_checks(const wchar_t* dll_path, const wchar_t* fixture_dir,
                         const wchar_t* normal_shim_dir)
{
    const fs::path dll = dll_path;
    if (dll.filename() != L"xgameruntime.dll" || !same_path(dll.parent_path(), fixture_dir)) {
        std::printf("FAIL: DLL is not at its fixture path %ls\n", fixture_dir);
        return false;
    }
    if (fs::exists(fs::path(normal_shim_dir) / L"xgameruntime.dll")) {
        std::printf("FAIL: normal shim directory contains xgameruntime.dll\n");
        return false;
    }
    return true;
}

template <typename Proc>
bool find_proc(HMODULE module, const char* name, Proc& proc)
{
    const FARPROC found = GetProcAddress(module, name);
    if (!found) {
        std::printf("FAIL: missing export %s\n", name);
        return false;
    }
    HMODULE owner = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(found), &owner) ||
        owner != module) {
        std::printf("FAIL: export %s is not owned by the fixture DLL\n", name);
        return false;
    }
    proc = reinterpret_cast<Proc>(found);
    return true;
}

struct Api {
    FindUserById find_user = nullptr;
    CloseUserHandle close_user = nullptr;
    InitializeProviderAsync initialize_async = nullptr;
    InitializeProviderResult initialize_result = nullptr;
    CreateContainer create_container = nullptr;
    CloseProvider close_provider = nullptr;
    CloseContainer close_container = nullptr;
};

bool load_api(HMODULE module, Api& api)
{
    return find_proc(module, "XUserFindUserById", api.find_user) &&
           find_proc(module, "XUserCloseHandle", api.close_user) &&
           find_proc(module, "XGameSaveInitializeProviderAsync", api.initialize_async) &&
           find_proc(module, "XGameSaveInitializeProviderResult", api.initialize_result) &&
           find_proc(module, "XGameSaveCreateContainer", api.create_container) &&
           find_proc(module, "XGameSaveCloseProvider", api.close_provider) &&
           find_proc(module, "XGameSaveCloseContainer", api.close_container);
}

constexpr char configuration_id[] = "synthetic-scid";

bool positive_completion(const Api& api, XUserHandle user)
{
    CallbackProbe probe;
    XAsyncBlock block = make_block(&probe, record_completion);
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                        "async initialize");
    ok = expect(probe.calls == 1 && probe.seen == &block,
                "callback ran once with the caller's block before the call returned") && ok;
    ok = expect(opaque_untouched(block) && block.context == &probe &&
                    block.callback == record_completion,
                "the caller's block fields are left as written") && ok;

    XGameSaveProviderHandle provider = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(&block, &provider), S_OK, "result") && ok;
    ok = expect(provider != nullptr, "result returns a provider") && ok;

    XGameSaveContainerHandle container = nullptr;
    ok = expect_hr(api.create_container(provider, "async-slot", &container), S_OK,
                   "provider from the async path creates a container") && ok;
    ok = expect(container != nullptr, "container handle is set") && ok;
    api.close_container(container);
    api.close_provider(provider);

    XGameSaveProviderHandle again = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(&block, &again), E_INVALIDARG,
                   "second result for the same operation") && ok;
    ok = expect(again == nullptr, "failed result clears the output") && ok;
    return ok;
}

bool null_callback_and_callback_claim(const Api& api, XUserHandle user)
{
    CallbackProbe silent;
    XAsyncBlock plain = make_block(&silent, nullptr);
    XGameSaveProviderHandle first = nullptr;
    bool ok = expect_hr(api.initialize_async(user, configuration_id, true, &plain), S_OK,
                        "async initialize without a callback");
    ok = expect_hr(api.initialize_result(&plain, &first), S_OK, "result without a callback") && ok;
    ok = expect(first != nullptr && silent.calls == 0, "no callback is invented") && ok;
    api.close_provider(first);

    CallbackProbe claimed;
    claimed.result_proc = api.initialize_result;
    XAsyncBlock block = make_block(&claimed, claim_in_callback);
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                   "async initialize claimed inside the callback") && ok;
    ok = expect(claimed.calls == 1 && claimed.result_status == S_OK &&
                    claimed.result_inside != nullptr,
                "the callback can retrieve the result") && ok;
    api.close_provider(claimed.result_inside);
    XGameSaveProviderHandle leftover = nullptr;
    ok = expect_hr(api.initialize_result(&block, &leftover), E_INVALIDARG,
                   "result already claimed in the callback") && ok;
    return ok;
}

bool argument_errors(const Api& api, XUserHandle user)
{
    CallbackProbe probe;
    XAsyncBlock block = make_block(&probe, record_completion);
    XGameSaveProviderHandle provider = nullptr;
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, nullptr), E_INVALIDARG,
                        "null block");
    ok = expect_hr(api.initialize_async(user, nullptr, false, &block), E_INVALIDARG,
                   "null configuration id") && ok;
    ok = expect_hr(api.initialize_async(user, "", false, &block), E_INVALIDARG,
                   "empty configuration id") && ok;
    ok = expect_hr(api.initialize_async(nullptr, configuration_id, false, &block), E_HANDLE,
                   "null user") && ok;

    XUserHandle closed = nullptr;
    ok = expect_hr(api.find_user(synthetic_user_id, &closed), S_OK, "second user handle") && ok;
    api.close_user(closed);
    ok = expect_hr(api.initialize_async(closed, configuration_id, false, &block), E_HANDLE,
                   "closed user handle") && ok;
    ok = expect(probe.calls == 0 && opaque_untouched(block),
                "rejected calls never complete or touch the block") && ok;
    ok = expect_hr(api.initialize_result(&block, &provider), E_INVALIDARG,
                   "rejected call leaves nothing to claim") && ok;

    ok = expect_hr(api.initialize_result(&block, nullptr), E_POINTER, "result null output") && ok;
    provider = reinterpret_cast<XGameSaveProviderHandle>(1);
    ok = expect_hr(api.initialize_result(nullptr, &provider), E_INVALIDARG, "result null block") && ok;
    ok = expect(provider == nullptr, "result null block clears the output") && ok;
    return ok;
}

bool reuse_and_capacity(const Api& api, XUserHandle user)
{
    CallbackProbe probe;
    XAsyncBlock block = make_block(&probe, record_completion);
    bool ok = expect_hr(api.initialize_async(user, configuration_id, false, &block), S_OK,
                        "first start");
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &block),
                   e_illegal_state_change, "reusing an unclaimed block") && ok;
    ok = expect(probe.calls == 1, "refused reuse does not complete again") && ok;
    XGameSaveProviderHandle held = nullptr;
    ok = expect_hr(api.initialize_result(&block, &held), S_OK, "original result survives") && ok;

    // The fixture holds four providers. Fill the rest, then show a refused start
    // consumes no async state and a freed provider makes room again.
    std::array<XAsyncBlock, 4> blocks{};
    std::array<XGameSaveProviderHandle, 4> providers{};
    std::size_t filled = 0;
    for (; filled < blocks.size(); ++filled) {
        blocks[filled] = make_block(&probe, nullptr);
        providers[filled] = nullptr;
        if (api.initialize_async(user, configuration_id, false, &blocks[filled]) != S_OK) break;
        if (FAILED(api.initialize_result(&blocks[filled], &providers[filled]))) {
            std::printf("FAIL: result for filled provider %zu\n", filled);
            return false;
        }
    }
    ok = expect(filled == 3, "three more providers fit beside the held one") && ok;
    XAsyncBlock refused = make_block(&probe, nullptr);
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &refused), E_OUTOFMEMORY,
                   "provider storage exhausted") && ok;
    XGameSaveProviderHandle none = nullptr;
    ok = expect_hr(api.initialize_result(&refused, &none), E_INVALIDARG,
                   "refused start left no async state") && ok;
    api.close_provider(held);
    ok = expect_hr(api.initialize_async(user, configuration_id, false, &refused), S_OK,
                   "start after a provider was freed") && ok;
    XGameSaveProviderHandle revived = nullptr;
    ok = expect_hr(api.initialize_result(&refused, &revived), S_OK, "result after freeing") && ok;
    api.close_provider(revived);
    for (std::size_t i = 0; i < filled; ++i) api.close_provider(providers[i]);
    return ok;
}

bool log_checks(HMODULE module)
{
    GetLog get_log = nullptr;
    if (!find_proc(module, "XGameRuntimeShimGetLog", get_log)) return false;
    std::array<XGameRuntimeShimLogEntry, 256> entries{};
    const std::size_t count = get_log(entries.data(), entries.size());
    bool async_ok = false, async_handle = false, result_ok = false, result_invalid = false;
    for (std::size_t i = 0; i < count && i < entries.size(); ++i) {
        const auto& entry = entries[i];
        if (!entry.function) return false;
        if (entry.result == not_implemented) {
            std::printf("FAIL: %s logged E_NOTIMPL\n", entry.function);
            return false;
        }
        if (std::strcmp(entry.function, "XGameSaveInitializeProviderAsync") == 0) {
            async_ok = async_ok || entry.result == S_OK;
            async_handle = async_handle || entry.result == E_HANDLE;
        }
        if (std::strcmp(entry.function, "XGameSaveInitializeProviderResult") == 0) {
            result_ok = result_ok || entry.result == S_OK;
            result_invalid = result_invalid || entry.result == E_INVALIDARG;
        }
    }
    return expect(async_ok && async_handle && result_ok && result_invalid,
                  "log carries the positive and negative async calls");
}

bool run(const wchar_t* dll_path, const wchar_t* fixture_dir, const wchar_t* normal_shim_dir)
{
    if (!fixture_path_checks(dll_path, fixture_dir, normal_shim_dir)) return false;
    HMODULE module = LoadLibraryW(dll_path);
    if (!module) {
        std::printf("FAIL: LoadLibraryW(%ls) error=%lu\n", dll_path, GetLastError());
        return false;
    }
    Api api;
    ClearLog clear_log = nullptr;
    bool passed = load_api(module, api) && find_proc(module, "XGameRuntimeShimClearLog", clear_log);
    XUserHandle user = nullptr;
    if (passed) {
        clear_log();
        passed = expect_hr(api.find_user(synthetic_user_id, &user), S_OK, "find user") &&
                 positive_completion(api, user) && null_callback_and_callback_claim(api, user) &&
                 argument_errors(api, user) && reuse_and_capacity(api, user) &&
                 log_checks(module);
        if (user) api.close_user(user);
    }
    FreeLibrary(module);
    return passed;
}
}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4) {
        std::printf("usage: xgameruntime_gamesave_async_check <dll> <fixture-dir> <normal-shim-dir>\n");
        return 2;
    }
    const bool passed = run(argv[1], argv[2], argv[3]);
    std::printf("xgameruntime game-save async: initialize+result positive, callback, argument, "
                "reuse and capacity controls %s\n",
                passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
