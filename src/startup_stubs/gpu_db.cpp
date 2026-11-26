// Shader database reader (WP1, vexport): contract C1-C6 over X2W's shader database (src/shader_db). The database is
// the file x2w_shaders writes (x2w_shaders.vxdb), from a shader export (--payload) or from the C++ translator port.
// It is found at %X2W_GPU_DB%, else %X2W_SHADER_CACHE%\x2w_shaders.vxdb, mapped once at the first lookup and
// kept for the process. Its manifest must agree with abi::view_heap_limit and, with twins on, abi::raw_views (0
// with X2W_GPU_TWINS=0); otherwise nothing is loaded and every lookup misses. Lookups after the load take no
// lock; a miss is logged once per key (C6).
#include "startup_stubs/gpu_db.h"
#include "startup_stubs/gpu_material_fields.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "shader_db/shader_db.h"
#include "startup_stubs/blake3/blake3.h"
#include "startup_stubs/local_log.h"

namespace x2w_gpu {

void gpu_key(const void* bytes, std::size_t size, char out[17]) noexcept
{
    blake3_hasher hasher;
    blake3_hasher_init(&hasher);
    if (size) blake3_hasher_update(&hasher, bytes, size);
    std::uint8_t hash[8];
    blake3_hasher_finalize(&hasher, hash, sizeof(hash));
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        out[2 * i] = digits[hash[i] >> 4];
        out[2 * i + 1] = digits[hash[i] & 15];
    }
    out[16] = 0;
}

namespace {

static_assert(sizeof(GpuSamplerWords) == 16, "sampler words as stored");
static_assert(sizeof(GpuStaticSampler) == sizeof(shader_db::VxStaticSampler), "static sampler as stored");
static_assert(sizeof(GpuRootIndexedView) == sizeof(shader_db::VxRootIndexedView), "root indexed view as stored");

struct Loaded {
    bool ok = false;
    std::vector<std::uint64_t> keys;  // sorted, parallel to shaders
    std::vector<GpuShader> shaders;
    std::vector<GpuUserData> user_data;
    std::vector<GpuUsedResource> used;
    GpuManifest manifest{};
    std::mutex miss_lock;
    std::unordered_set<std::uint64_t> missed;
};

void log_line(const char* line) { x2w_local::logf("x2w-gpu %s", line); }

GpuRootType root_of(std::uint32_t code, std::uint32_t& other)
{
    other = 0;
    if (code < shader_db::root_other_base) return static_cast<GpuRootType>(code);
    other = code - shader_db::root_other_base;
    return GpuRootType::other;
}

GpuStage title_stage(shader_db::Stage s)
{
    switch (s) {
    case shader_db::Stage::vs: return GpuStage::vertex;
    case shader_db::Stage::ps: return GpuStage::pixel;
    case shader_db::Stage::ms: return GpuStage::mesh;
    case shader_db::Stage::cs: return GpuStage::compute;
    default: return GpuStage::count;
    }
}

bool switch_set(const char* value) { return value[0] && std::strcmp(value, "0") != 0; }

std::wstring database_path()
{
    wchar_t path[MAX_PATH];
    DWORD n = GetEnvironmentVariableW(L"X2W_GPU_DB", path, MAX_PATH);
    if (n && n < MAX_PATH) return path;
    n = GetEnvironmentVariableW(L"X2W_SHADER_CACHE", path, MAX_PATH);
    if (n && n < MAX_PATH) return std::wstring(path) + L"\\x2w_shaders.vxdb";
    return {};
}

void load(Loaded& l)
{
    const DWORD saved = GetLastError();
    const std::wstring path = database_path();
    if (path.empty()) {
        x2w_local::logf("x2w-gpu db: no database (set X2W_GPU_DB or X2W_SHADER_CACHE); every shader misses");
        SetLastError(saved);
        return;
    }
    const std::uint32_t raw_views = gpu_twins_on() ? abi::raw_views : 0;
    if (!shader_db::gpu_db_open(path.c_str(), abi::view_heap_limit, 0, log_line)) {
        SetLastError(saved);
        return;
    }
    const shader_db::Manifest& m = *shader_db::gpu_db_manifest();
    if (m.raw_views != raw_views) {
        x2w_local::logf("x2w-gpu db: refused: made with raw_views %u, this run's heap has %u (X2W_GPU_TWINS)", m.raw_views, raw_views);
        shader_db::gpu_db_close();
        SetLastError(saved);
        return;
    }
    GpuManifest& v = l.manifest;
    v.view_heap_limit = m.view_heap_limit;
    v.raw_views = m.raw_views;
    v.host_rdna = m.host_rdna != 0;
    v.max_wave = m.max_wave;
    v.opt_bits = m.shader_opt;
    v.group_sync = m.group_sync != 0;
    v.helper_own = switch_set(m.helper_own);
    v.bvh_bits = (switch_set(m.bvh_fences) ? 1u : 0u) | (switch_set(m.bvh_bounds) ? 2u : 0u);
    std::snprintf(v.commit, sizeof(v.commit), "%s", m.commit);

    const std::size_t count = shader_db::gpu_db_count();
    std::size_t user_data = 0, used = 0;
    for (std::size_t i = 0; i < count; ++i) {
        user_data += shader_db::gpu_db_at(i)->user_data_count;
        used += shader_db::gpu_db_at(i)->used_count;
    }
    l.user_data.reserve(user_data);  // the spans below point into these: no reallocation after reserve
    l.used.reserve(used);
    l.keys.reserve(count);
    l.shaders.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const shader_db::GpuShader& s = *shader_db::gpu_db_at(i);
        GpuShader& o = l.shaders[i];
        std::snprintf(o.key, sizeof(o.key), "%016llx", static_cast<unsigned long long>(s.key));
        o.stage = title_stage(s.stage);
        o.hw_stage = static_cast<GpuHwStage>(s.stage);
        o.dxil = s.dxil;
        o.dxil_bytes = s.dxil_bytes;
        o.material_fields = o.stage == GpuStage::pixel ? x2w_material::lookup(s.key, s.dxil, s.dxil_bytes) : 0;
        const std::size_t ud_first = l.user_data.size();
        for (std::uint32_t k = 0; k < s.user_data_count; ++k) {
            const shader_db::VxUserData& u = s.user_data[k];
            GpuUserData d{};
            d.index = u.index;
            d.slot = u.slot;
            d.root = root_of(u.root, d.other);
            d.dword = u.dword;
            l.user_data.push_back(d);
        }
        o.user_data = {l.user_data.data() + ud_first, s.user_data_count};
        o.samplers = {reinterpret_cast<const GpuSamplerWords*>(s.samplers), s.sampler_count};
        o.static_samplers = {reinterpret_cast<const GpuStaticSampler*>(s.static_samplers), s.static_sampler_count};
        o.root_cbvs = {s.root_cbvs, s.root_cbv_count};
        o.root_srvs = {s.root_srvs, s.root_srv_count};
        o.root_uavs = {s.root_uavs, s.root_uav_count};
        o.root_indexed_views = {reinterpret_cast<const GpuRootIndexedView*>(s.root_indexed_views), s.root_indexed_view_count};
        o.has_threads = s.has_threads;
        std::memcpy(o.threads, s.threads, sizeof(o.threads));
        const std::size_t used_first = l.used.size();
        for (std::uint32_t k = 0; k < s.used_count; ++k) {
            const shader_db::VxUsed& u = s.used[k];
            GpuUsedResource d{};
            if (u.kind < shader_db::resource_other_base) {
                d.kind = static_cast<GpuResourceType>(u.kind);
            } else {
                d.kind = GpuResourceType::other;
                d.other_kind = static_cast<std::uint8_t>(u.kind - shader_db::resource_other_base);
            }
            d.root = root_of(u.root, d.root_other);
            d.table_offset_dwords = u.table_offset_dwords;
            d.lower_bound = u.lower_bound;
            d.stride = u.stride;
            d.dwords = static_cast<std::uint8_t>(u.dwords);
            d.user_data = static_cast<std::uint8_t>(u.user_data);
            d.slot = static_cast<std::uint8_t>(u.slot);
            l.used.push_back(d);
        }
        o.used = {l.used.data() + used_first, s.used_count};
        o.stores = s.stores;
        o.stores_by_address = s.stores_by_address;
        o.builder = s.builder;
        l.keys.push_back(s.key);
    }
    l.ok = true;
    SetLastError(saved);
}

Loaded& loaded()
{
    static Loaded l;
    static std::once_flag once;
    std::call_once(once, [] { load(l); });
    return l;
}

std::uint64_t parse_key(const char key[17])
{
    std::uint64_t v = 0;
    for (int i = 0; i < 16; ++i) {
        const char c = key[i];
        const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
        if (d < 0) return 0;
        v = v << 4 | static_cast<std::uint64_t>(d);
    }
    return v;
}

const GpuShader* find(std::uint64_t key, GpuStage stage, std::size_t bytes)
{
    Loaded& l = loaded();
    const GpuShader* found = nullptr;
    if (l.ok) {
        const auto at = std::lower_bound(l.keys.begin(), l.keys.end(), key);
        if (at != l.keys.end() && *at == key) found = &l.shaders[static_cast<std::size_t>(at - l.keys.begin())];
    }
    if (found && found->stage == stage) return found;
    std::lock_guard<std::mutex> hold(l.miss_lock);
    if (l.missed.insert(key).second)
        x2w_local::logf("x2w-gpu db miss key=%016llx stage=%s bytes=%zu%s%s", static_cast<unsigned long long>(key), gpu_stage_name(stage), bytes,
                        found ? " recorded as " : "", found ? gpu_stage_name(found->stage) : "");
    return nullptr;
}

}  // namespace

const GpuShader* gpu_shader(const void* bytes, std::size_t size, GpuStage stage) noexcept
{
    if (!bytes || !size) return nullptr;
    char key[17];
    gpu_key(bytes, size, key);
    return find(parse_key(key), stage, size);
}

const GpuShader* gpu_shader_by_key(const char key[17], GpuStage stage) noexcept { return find(parse_key(key), stage, 0); }

const GpuManifest* gpu_manifest() noexcept
{
    Loaded& l = loaded();
    return l.ok ? &l.manifest : nullptr;
}

}  // namespace x2w_gpu
