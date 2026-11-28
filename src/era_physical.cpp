#include "era_physical.h"

#include <mutex>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace era_physical {
namespace {
struct Registry {
    std::mutex lock;
    // grant[page] is the grant that owns the page, 0 when free. Page number 0 is never issued, so a
    // zero-filled array never names a live page.
    std::vector<std::uint32_t> grant;
    std::unordered_map<std::uint32_t, std::uint64_t> sizes;
    std::uint32_t next_grant = 1;
    std::uint64_t granted = 0;
    std::uint64_t search_from = 1;
};

Registry& registry() noexcept
{
    static Registry value;
    return value;
}

bool ready(Registry& r) noexcept
{
    if (!r.grant.empty()) return true;
    try {
        r.grant.assign(capacity_pages + 1, 0);
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    }
}
}

Status allocate(std::uint64_t flags, std::uint64_t* count, std::uint64_t* pages) noexcept
{
    if (!count) return Status::invalid;
    const auto requested = *count;
    *count = 0;
    if (flags != mem_64k_pages || !pages || requested == 0 || requested > capacity_pages) {
        return Status::invalid;
    }
    auto& r = registry();
    std::lock_guard guard(r.lock);
    if (!ready(r) || requested > capacity_pages - r.granted) return Status::exhausted;
    std::uint32_t id = r.next_grant;
    while (id == 0 || r.sizes.count(id)) ++id;
    try {
        r.sizes.emplace(id, requested);
    } catch (const std::bad_alloc&) {
        return Status::exhausted;
    }
    r.next_grant = id + 1;
    // Lowest free page numbers first, starting where the last search ended.
    std::uint64_t written = 0;
    for (std::uint64_t step = 0, page = r.search_from; written < requested && step < capacity_pages; ++step) {
        if (!r.grant[page]) {
            r.grant[page] = id;
            pages[written++] = page;
        }
        page = page == capacity_pages ? 1 : page + 1;
        r.search_from = page;
    }
    r.granted += requested;
    *count = requested;
    return Status::ok;
}

Status release(std::uint64_t count, const std::uint64_t* pages) noexcept
{
#ifdef ERA_PHYSICAL_LENIENT_RELEASE
    // Test-only negative control (era_physical_check): clears every listed page that is in range
    // without checking ownership, grant boundaries, duplicates or the count, like SlimEra's
    // FreeTitlePhysicalPages. Never built into xmem.dll.
    if (count && !pages) return Status::invalid;
    auto& r = registry();
    std::lock_guard guard(r.lock);
    if (r.grant.empty()) return Status::ok;
    for (std::uint64_t i = 0; i < count && i < capacity_pages; ++i) {
        const auto page = pages[i];
        if (page && page <= capacity_pages && r.grant[page]) {
            r.grant[page] = 0;
            --r.granted;
        }
    }
    return Status::ok;
#else
    if (!pages || count == 0 || count > capacity_pages) return Status::invalid;
    auto& r = registry();
    std::lock_guard guard(r.lock);
    if (r.grant.empty()) return Status::invalid;
    try {
        // Validate the whole array before changing anything.
        std::unordered_set<std::uint64_t> seen;
        std::unordered_map<std::uint32_t, std::uint64_t> named;
        seen.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0; i < count; ++i) {
            const auto page = pages[i];
            if (page == 0 || page > capacity_pages || !r.grant[page] || !seen.insert(page).second) {
                return Status::invalid;
            }
            ++named[r.grant[page]];
        }
        for (const auto& [id, pages_named] : named) {
            const auto size = r.sizes.find(id);
            if (size == r.sizes.end() || size->second != pages_named) return Status::invalid;
        }
        for (std::uint64_t i = 0; i < count; ++i) r.grant[pages[i]] = 0;
        for (const auto& entry : named) r.sizes.erase(entry.first);
    } catch (const std::bad_alloc&) {
        return Status::exhausted;
    }
    r.granted -= count;
    return Status::ok;
#endif
}

bool owns(std::uint64_t page) noexcept
{
    auto& r = registry();
    std::lock_guard guard(r.lock);
    return page && page <= capacity_pages && !r.grant.empty() && r.grant[page] != 0;
}

std::uint64_t granted_pages() noexcept
{
    auto& r = registry();
    std::lock_guard guard(r.lock);
    return r.granted;
}
}
