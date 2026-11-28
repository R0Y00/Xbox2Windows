// Exercise the game-save methods through their ordinary dispatcher. The source-built
// XAsync service can reject Begin, delay notification or enter the callback inline.
#include "../src/startup_stubs/runtime_proxy.cpp"
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>

static std::atomic<std::size_t> fail_allocation_size{0};
void* operator new(std::size_t size)
{
    std::size_t expected = size;
    if (size && fail_allocation_size.compare_exchange_strong(expected, 0)) {
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {
using Provider = decltype(&submit_provider);
struct Block {
    void* queue{};
    void* context{};
    void (WINAPI* callback)(Block*){};
    unsigned char internal[32]{};
};
struct Operations { void* methods[12]{}; } operations;
void** api = operations.methods;
bool refuse{}, inline_notify{};
struct Operation { Block* block{}; const void* id{}; HRESULT result{}; };
std::array<Operation,8> pending;
std::mutex pending_lock;
std::condition_variable progress;
Block* paused_block{};
bool entered{};
ULONG WINAPI drop(void*) { return 1; }
HRESULT WINAPI begin(void*, Block* block, void*, const void* id, const char*, Provider provider)
{
    std::unique_lock lock(pending_lock);
    if (refuse) return E_ACCESSDENIED;
    for (const auto& op : pending) if (op.block == block) return E_INVALIDARG;
    for (auto& op : pending) if (!op.block) {
        op = {block, id, E_PENDING};
        if (block == paused_block) {
            entered=true;
            progress.notify_one();
            progress.wait(lock, [] { return paused_block == nullptr; });
        }
        return provider(0,nullptr);
    }
    return E_OUTOFMEMORY;
}
void WINAPI complete(void*, Block* block, HRESULT status, size_t)
{
    std::unique_lock lock(pending_lock);
    for (auto& op : pending) if (op.block == block) {
        op.result = status;
        lock.unlock();
        if (inline_notify && block->callback) block->callback(block);
        return;
    }
    std::abort();
}
HRESULT WINAPI get_result(void*, Block* block, const void* id, size_t, void*, size_t*)
{
    const std::lock_guard lock(pending_lock);
    for (auto& op : pending) if (op.block == block && op.id == id) {
        const auto status = op.result;
        if (status != E_PENDING) op = {};
        return status;
    }
    return E_INVALIDARG;
}
HRESULT WINAPI service(const GUID* cls, const GUID* iid, void** out)
{
    if (*cls != submit_async_family || *iid != submit_async_family) return E_NOINTERFACE;
    *out = &api;
    return S_OK;
}
template<class T> u64 ptr(T* p) { return reinterpret_cast<u64>(p); }
u64 call(int slot, u64 a=0, u64 b=0, u64 c=0, u64 d=0)
{
    u64 args[]{a,b,c,d,0,0,0,0,0,0,0}, result{};
    if (!save_call(slot,args,&result)) std::abort();
    return result;
}
int value()
{
    std::vector<unsigned char> bytes;
    return save_store().read_blob("profile","preference",bytes) == x2w_save::Status::ok && bytes.size()==1 ? bytes[0] : -1;
}
struct Seen { unsigned callbacks{}; int stored{}; u64 status{}; };
void WINAPI on_complete(Block* block)
{
    auto& seen = *static_cast<Seen*>(block->context);
    seen.status = call(29,ptr(block));
    seen.stored = value();
    ++seen.callbacks;
}
int run(bool system)
{
    u64 provider_handle{}, container{}, good{}, bad{};
    if (call(3,local_user_handle,ptr("fixture"),0,ptr(&provider_handle)) ||
        call(16,provider_handle,ptr("profile"),ptr(&container)) ||
        call(23,container,0,ptr(&good)) || call(23,container,0,ptr(&bad))) return 1;
    unsigned char byte=0x11;
    const std::string long_name(80, 'c');
    const std::string name_copy(long_name);
    const std::size_t name_allocation = name_copy.capacity() + 1;
    fail_allocation_size = name_allocation;
    if (!x2w_save::valid_name(long_name) || fail_allocation_size != name_allocation) return 26;
    fail_allocation_size = 0;
    u64 long_container{}, long_update{};
    if (call(16,provider_handle,ptr(long_name.c_str()),ptr(&long_container)) ||
        call(23,long_container,0,ptr(&long_update))) return 22;
    fail_allocation_size = name_allocation;  // Fail the same string copy under save_lock.
    const u64 name_copy_result = call(25,long_update,ptr("preference"),ptr(&byte),1);
    if (fail_allocation_size || name_copy_result != e_outofmemory) return 23;
    if (!TryAcquireSRWLockExclusive(&save_lock)) return 24;
    ReleaseSRWLockExclusive(&save_lock);
    if (call(25,long_update,ptr("preference"),ptr(&byte),1) || call(27,long_update)) return 25;
    call(24,long_update);
    call(17,long_container);
    const std::array<unsigned char, 1025> large_blob{};
    for (const auto data_size : {std::size_t{1}, large_blob.size()}) {
        fail_allocation_size = data_size == 1 ? sizeof(x2w_save::Action) : data_size;
        u64 failed_stage{};
        try {
            failed_stage = call(25,good,ptr("preference"),ptr(large_blob.data()),data_size);
        } catch (const std::bad_alloc&) {
            return 20;
        }
        if (fail_allocation_size || failed_stage != e_outofmemory) {
            std::fprintf(stderr, "staging %zu bytes: pending allocation=%zu, HRESULT=0x%08llx\n",
                         data_size, fail_allocation_size.load(), static_cast<unsigned long long>(failed_stage));
            return 21;
        }
    }
    if (call(25,good,ptr("preference"),ptr(&byte),1) || call(27,good) || value()!=0x11) return 2;
    byte=0x77;
    if (call(25,good,ptr("preference"),ptr(&byte),1) || call(26,bad,ptr("absent"))) return 3;
    Seen seen;
    Block first{}, second{};
    if (system) {
        const auto dll = LoadLibraryExW(L"xgameruntime.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!dll) return 77;
        real_query = reinterpret_cast<QueryApi>(GetProcAddress(dll,"QueryApiImpl"));
        if (!real_query) return 77;
        void* runtime{};
        if (FAILED(load_real_query()(&submit_async_family,&submit_async_family,&runtime)) || !runtime) return 77;
        auto** methods = *static_cast<void***>(runtime);
        using CreateQueue = HRESULT(WINAPI*)(void*,unsigned,unsigned,void**);
        // XTaskQueueDispatchMode::Immediate for work and completion.
        if (FAILED(reinterpret_cast<CreateQueue>(methods[12])(runtime,3,3,&first.queue))) return 77;
        second.queue = first.queue;
        first.context=&seen; first.callback=on_complete;
        if (call(28,good,ptr(&first)) || call(28,bad,ptr(&second))) return 4;
        const auto bad_result=call(29,ptr(&second));
        std::printf("system inline callbacks=%u saved=%x status=%08llx other=%08llx\n",seen.callbacks,seen.stored,seen.status,bad_result);
        reinterpret_cast<void(WINAPI*)(void*,void*)>(methods[17])(runtime,first.queue);
        reinterpret_cast<ULONG(WINAPI*)(void*)>(methods[2])(runtime);
        return seen.callbacks==1 && seen.stored==0x77 && seen.status==0 && bad_result==0x80830008 ? 0 : 5;
    }
    operations.methods[2]=reinterpret_cast<void*>(&drop);
    operations.methods[7]=reinterpret_cast<void*>(&begin);
    operations.methods[10]=reinterpret_cast<void*>(&complete);
    operations.methods[11]=reinterpret_cast<void*>(&get_result);
    real_query=service;
    refuse=true;
    if (call(28,good,ptr(&first))!=static_cast<unsigned>(E_ACCESSDENIED) || call(28,good,0)!=e_fail || value()!=0x11) return 6;
    refuse=false;
    paused_block=&first;
    u64 first_start{};
    std::thread worker([&]{ first_start=call(28,good,ptr(&first)); });
    {
        std::unique_lock lock(pending_lock);
        progress.wait(lock,[]{ return entered; });
    }
    const bool independent = call(28,bad,ptr(&second)) == 0 &&
        call(29,ptr(&first)) == static_cast<unsigned>(E_PENDING) &&
        call(29,ptr(&second)) == 0x80830008 && value() == 0x11;
    {
        const std::lock_guard lock(pending_lock);
        paused_block=nullptr;
    }
    progress.notify_one();
    worker.join();
    if (!independent || first_start || value()!=0x77 || call(29,ptr(&first))) return 7;
    byte=0x88;
    if (call(25,good,ptr("preference"),ptr(&byte),1)) return 8;
    first.context=&seen; first.callback=on_complete; inline_notify=true;
    if (call(28,good,ptr(&first)) || seen.callbacks!=1 || seen.stored!=0x88 || seen.status) return 9;
    std::puts("refused Begin preserved 0x11; retry committed 0x77; two block results read in reverse; inline callback read 0x88");
    return 0;
}
}
int main(int argc, char** argv) { return run(argc==2 && std::strcmp(argv[1],"system")==0); }
