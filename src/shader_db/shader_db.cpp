// See shader_db.h.
#include "shader_db.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "startup_stubs/blake3/blake3.h"

namespace shader_db {
namespace {

struct Db {
    std::mutex lock;  // open/close and the miss set; lookups after open read only
    std::wstring path;
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
    const unsigned char* base = nullptr;
    uint64_t bytes = 0;
    const VxIndex* index = nullptr;
    uint32_t count = 0;
    std::vector<GpuShader> shaders;  // in index order
    Manifest manifest{};
    MissLog log = nullptr;
    std::unordered_set<uint64_t> missed;
};

Db& db()
{
    static Db instance;
    return instance;
}

void say(MissLog log, const char* format, ...)
{
    char line[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    if (log) log(line);
}

bool in_file(uint64_t offset, uint64_t bytes, uint64_t total) { return offset <= total && bytes <= total - offset; }

void unmap(Db& d)
{
    if (d.base) UnmapViewOfFile(d.base);
    if (d.mapping) CloseHandle(d.mapping);
    if (d.file != INVALID_HANDLE_VALUE) CloseHandle(d.file);
    d.base = nullptr;
    d.mapping = nullptr;
    d.file = INVALID_HANDLE_VALUE;
    d.index = nullptr;
    d.count = 0;
    d.shaders.clear();
    d.missed.clear();
    d.path.clear();
}

uint32_t number(const std::string& text, const char* name)
{
    const std::string key = std::string("\n") + name + " ";
    const std::size_t at = ("\n" + text).find(key);
    if (at == std::string::npos) return 0;
    return static_cast<uint32_t>(std::strtoul(text.c_str() + at + key.size() - 1, nullptr, 0));
}

void word(const std::string& text, const char* name, char* out, std::size_t capacity)
{
    out[0] = 0;
    const std::string key = std::string("\n") + name + " ";
    const std::size_t at = ("\n" + text).find(key);
    if (at == std::string::npos) return;
    std::size_t from = at + key.size() - 1, to = text.find('\n', from);
    if (to == std::string::npos) to = text.size();
    const std::size_t n = std::min(to - from, capacity - 1);
    std::memcpy(out, text.data() + from, n);
    out[n] = 0;
}

// Resolves one record; false if any array lies outside the file.
bool resolve(const unsigned char* base, uint64_t total, uint64_t offset, GpuShader& s)
{
    if (!in_file(offset, sizeof(VxRecord), total)) return false;
    VxRecord r;
    std::memcpy(&r, base + offset, sizeof(r));
    const auto array = [&](uint64_t at, uint64_t count, uint64_t size) { return in_file(at, count * size, total); };
    if (!array(r.dxil_offset, r.dxil_bytes, 1) || !array(r.user_data_offset, r.user_data_count, sizeof(VxUserData)) ||
        !array(r.samplers_offset, r.sampler_count, 16) || !array(r.static_samplers_offset, r.static_sampler_count, sizeof(VxStaticSampler)) ||
        !array(r.root_cbvs_offset, r.root_cbv_count, 1) || !array(r.root_srvs_offset, r.root_srv_count, 1) ||
        !array(r.root_uavs_offset, r.root_uav_count, 1) ||
        !array(r.root_indexed_views_offset, r.root_indexed_view_count, sizeof(VxRootIndexedView)) ||
        !array(r.used_offset, r.used_count, sizeof(VxUsed)) || r.stage > static_cast<uint32_t>(Stage::cs))
        return false;
    s.key = r.key;
    s.fnv64 = r.fnv64;
    s.stage = static_cast<Stage>(r.stage);
    s.container_bytes = r.container_bytes;
    s.dxil = base + r.dxil_offset;
    s.dxil_bytes = r.dxil_bytes;
    s.user_data_dwords = r.user_data_dwords;
    s.user_data = reinterpret_cast<const VxUserData*>(base + r.user_data_offset);
    s.user_data_count = r.user_data_count;
    s.samplers = reinterpret_cast<const uint32_t(*)[4]>(base + r.samplers_offset);
    s.sampler_count = r.sampler_count;
    s.static_samplers = reinterpret_cast<const VxStaticSampler*>(base + r.static_samplers_offset);
    s.static_sampler_count = r.static_sampler_count;
    s.root_cbvs = base + r.root_cbvs_offset;
    s.root_cbv_count = r.root_cbv_count;
    s.root_srvs = base + r.root_srvs_offset;
    s.root_srv_count = r.root_srv_count;
    s.root_uavs = base + r.root_uavs_offset;
    s.root_uav_count = r.root_uav_count;
    s.root_indexed_views = reinterpret_cast<const VxRootIndexedView*>(base + r.root_indexed_views_offset);
    s.root_indexed_view_count = r.root_indexed_view_count;
    s.used = reinterpret_cast<const VxUsed*>(base + r.used_offset);
    s.used_count = r.used_count;
    s.has_threads = (r.flags & flag_threads) != 0;
    std::memcpy(s.threads, r.threads, sizeof(s.threads));
    s.stores = (r.flags & flag_stores) != 0;
    s.stores_by_address = (r.flags & flag_stores_by_address) != 0;
    s.builder = (r.flags & flag_builder) != 0;
    s.wave_size = r.wave_size;
    s.vgprs = r.vgprs;
    return true;
}

}  // namespace

bool gpu_db_open(const wchar_t* path, uint32_t expected_view_heap_limit, uint32_t expected_raw_views, MissLog log)
{
    Db& d = db();
    std::lock_guard<std::mutex> hold(d.lock);
    if (d.base && d.path == path) return true;
    unmap(d);
    d.log = log;
    d.file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    LARGE_INTEGER size{};
    if (d.file == INVALID_HANDLE_VALUE || !GetFileSizeEx(d.file, &size) || size.QuadPart < static_cast<LONGLONG>(sizeof(VxHeader))) {
        say(log, "gpu-db: cannot open the database (error %lu)", GetLastError());
        unmap(d);
        return false;
    }
    d.mapping = CreateFileMappingW(d.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    d.base = d.mapping ? static_cast<const unsigned char*>(MapViewOfFile(d.mapping, FILE_MAP_READ, 0, 0, 0)) : nullptr;
    if (!d.base) {
        say(log, "gpu-db: cannot map the database (error %lu)", GetLastError());
        unmap(d);
        return false;
    }
    d.bytes = static_cast<uint64_t>(size.QuadPart);
    VxHeader h;
    std::memcpy(&h, d.base, sizeof(h));
    if (std::memcmp(h.magic, "VXDB0001", 8) || h.version != 1 || h.file_bytes != d.bytes ||
        !in_file(h.index_offset, uint64_t(h.record_count) * sizeof(VxIndex), d.bytes) ||
        !in_file(h.manifest_offset, h.manifest_bytes, d.bytes)) {
        say(log, "gpu-db: not a database of format 1, or truncated");
        unmap(d);
        return false;
    }
    const std::string manifest(reinterpret_cast<const char*>(d.base + h.manifest_offset), h.manifest_bytes);
    Manifest& m = d.manifest;
    m.max_wave = number(manifest, "max_wave");
    m.view_heap_limit = number(manifest, "view_heap_limit");
    m.raw_views = number(manifest, "raw_views");
    m.host_rdna = number(manifest, "host_rdna");
    m.loop_limit = number(manifest, "loop_limit");
    m.group_sync = number(manifest, "group_sync");
    m.shader_opt = number(manifest, "shader_opt");
    word(manifest, "build", m.build, sizeof(m.build));
    word(manifest, "commit", m.commit, sizeof(m.commit));
    word(manifest, "producer", m.producer, sizeof(m.producer));
    word(manifest, "X2W_SHADERCFG_HELPER_OWN", m.helper_own, sizeof(m.helper_own));
    word(manifest, "X2W_SHADERCFG_BVH_FENCES", m.bvh_fences, sizeof(m.bvh_fences));
    word(manifest, "X2W_SHADERCFG_BVH_BOUNDS", m.bvh_bounds, sizeof(m.bvh_bounds));
    if ((expected_view_heap_limit && m.view_heap_limit != expected_view_heap_limit) ||
        (expected_raw_views && m.raw_views != expected_raw_views)) {
        say(log, "gpu-db: refused: made for view_heap_limit %u raw_views %u, this heap has %u and %u", m.view_heap_limit,
            m.raw_views, expected_view_heap_limit, expected_raw_views);
        unmap(d);
        return false;
    }
    d.index = reinterpret_cast<const VxIndex*>(d.base + h.index_offset);
    d.count = h.record_count;
    d.shaders.resize(d.count);
    for (uint32_t i = 0; i < d.count; ++i) {
        if ((i && d.index[i - 1].key >= d.index[i].key) || !resolve(d.base, d.bytes, d.index[i].record_offset, d.shaders[i]) ||
            d.shaders[i].key != d.index[i].key) {
            say(log, "gpu-db: record %u is damaged or out of order", i);
            unmap(d);
            return false;
        }
    }
    d.path = path;
    say(log, "gpu-db: %u shaders, build %.16s, commit %.12s, view_heap_limit %u raw_views %u max_wave %u", d.count, m.build,
        m.commit, m.view_heap_limit, m.raw_views, m.max_wave);
    return true;
}

void gpu_db_close()
{
    Db& d = db();
    std::lock_guard<std::mutex> hold(d.lock);
    unmap(d);
}

const Manifest* gpu_db_manifest() { return db().base ? &db().manifest : nullptr; }
std::size_t gpu_db_count() { return db().count; }
const GpuShader* gpu_db_at(std::size_t i) { return i < db().count ? &db().shaders[i] : nullptr; }

uint64_t gpu_key(const void* bytes, std::size_t n)
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    blake3_hasher_update(&hasher, bytes, n);
    uint8_t out[8];
    blake3_hasher_finalize(&hasher, out, sizeof(out));
    uint64_t key = 0;
    for (uint8_t b : out) key = key << 8 | b;
    return key;
}

const GpuShader* gpu_shader_by_key(uint64_t key)
{
    const Db& d = db();
    if (!d.base) return nullptr;
    const VxIndex* end = d.index + d.count;
    const VxIndex* at = std::lower_bound(d.index, end, key, [](const VxIndex& e, uint64_t k) { return e.key < k; });
    return at != end && at->key == key ? &d.shaders[static_cast<std::size_t>(at - d.index)] : nullptr;
}

const GpuShader* gpu_shader(const void* bytes, std::size_t n, Stage stage)
{
    if (!bytes || !n) return nullptr;
    const uint64_t key = gpu_key(bytes, n);
    const GpuShader* found = gpu_shader_by_key(key);
    if (found && (stage == Stage::any || found->stage == stage)) return found;
    Db& d = db();
    std::lock_guard<std::mutex> hold(d.lock);
    if (d.missed.insert(key).second)
        say(d.log, "gpu-db: miss key=%016llx bytes=%zu stage=%u%s", static_cast<unsigned long long>(key), n,
            static_cast<unsigned>(stage), found ? " (recorded with another stage)" : "");
    return nullptr;
}

}  // namespace shader_db
