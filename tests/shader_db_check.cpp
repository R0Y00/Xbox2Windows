// shader_db_check: X2W's shader database (src/shader_db) without a GPU.
//   shader_db_check            BLAKE3 vectors, a synthetic record written and read back, a refused manifest,
//                              the archive scan of a synthetic SGD5 record and a container stored as is.
//   shader_db_check --mutate   the same, but the refused manifest is expected to open: must fail (WILL_FAIL).
//   shader_db_check --export <gpu export dir> [N]
//                              N records (default 5) of a real export written into a database and read back.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "shader_db/archive_scan.h"
#include "shader_db/shader_db_writer.h"
#include "shader_db/shader_db.h"
#include "shader_db/gpu_export.h"

using namespace shader_db;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

void quiet(const char*) {}

std::wstring temp_file(const wchar_t* name)
{
    wchar_t dir[MAX_PATH];
    GetTempPathW(MAX_PATH, dir);
    return std::wstring(dir) + name;
}

bool same(const ShaderRecord& a, const ShaderRecord& b)
{
    const auto eq = [](const auto& x, const auto& y) {
        return x.size() == y.size() && (x.empty() || !std::memcmp(x.data(), y.data(), x.size() * sizeof(x[0])));
    };
    return a.key == b.key && a.fnv64 == b.fnv64 && a.stage == b.stage && a.container_bytes == b.container_bytes && a.dxil == b.dxil &&
           a.user_data_dwords == b.user_data_dwords && eq(a.user_data, b.user_data) && eq(a.samplers, b.samplers) &&
           eq(a.static_samplers, b.static_samplers) && a.root_cbvs == b.root_cbvs && a.root_srvs == b.root_srvs &&
           a.root_uavs == b.root_uavs && eq(a.root_indexed_views, b.root_indexed_views) && eq(a.used, b.used) &&
           a.has_threads == b.has_threads && !std::memcmp(a.threads, b.threads, sizeof(a.threads)) && a.stores == b.stores &&
           a.stores_by_address == b.stores_by_address && a.builder == b.builder && a.wave_size == b.wave_size && a.vgprs == b.vgprs;
}

std::string hex(uint64_t v)
{
    char t[17];
    std::snprintf(t, sizeof(t), "%016llx", static_cast<unsigned long long>(v));
    return t;
}

// A minimal well-formed DXBC container with one part.
std::vector<uint8_t> container(const char* fourcc, const std::vector<uint8_t>& body)
{
    std::vector<uint8_t> c(32 + 4 + 8);
    std::memcpy(c.data(), "DXBC", 4);
    c.insert(c.end(), body.begin(), body.end());
    const uint32_t total = static_cast<uint32_t>(c.size()), count = 1, offset = 36, size = static_cast<uint32_t>(body.size());
    std::memcpy(c.data() + 24, &total, 4);
    std::memcpy(c.data() + 28, &count, 4);
    std::memcpy(c.data() + 32, &offset, 4);
    std::memcpy(c.data() + 36, fourcc, 4);
    std::memcpy(c.data() + 40, &size, 4);
    return c;
}

int synthetic(bool mutate)
{
    // BLAKE3 test vectors (the reference test_vectors.json: input of length 0, and of length 1 = byte 0).
    check(gpu_key("", 0) == 0xaf1349b9f5f9a1a6ull, "BLAKE3 of the empty input");
    const uint8_t zero = 0;
    check(gpu_key(&zero, 1) == 0x2d3adedff11b61f1ull, "BLAKE3 of one zero byte");
    check(fnv64("a", 1) == 0x44bd8ad473cd9906ull, "fnv_bytes of 'a' (X2W's basis 1469598103934665603)");

    const std::vector<uint8_t> title = container("DXIL", {1, 2, 3, 4, 5, 6, 7, 8});
    ShaderRecord r;
    r.key = gpu_key(title.data(), title.size());
    r.fnv64 = fnv64(title.data(), title.size());
    r.stage = Stage::ps;
    r.container_bytes = static_cast<uint32_t>(title.size());
    r.dxil = container("DXIL", {9, 9, 9});
    r.user_data_dwords = 8;
    r.user_data = {{0, 1, root_table, 0}, {5, 3, root_other_base + 7, 1}};
    r.samplers = {{0x400, 0xc00000, 0x18f00000, 0}};
    r.static_samplers = {{0x15, {1, 2, 3}, -0.5f, 16, 0, 0, 0.0f, 1000.0f}};
    r.root_cbvs = {2};
    r.root_srvs = {0, 3};
    r.root_uavs = {1};
    r.root_indexed_views = {{1, 2, 3, 32, 5}};
    r.used = {{resource_uav, root_table, 4, 0, 16, 8, 2, 1}};
    r.has_threads = true;
    r.threads[0] = 64;
    r.threads[1] = r.threads[2] = 1;
    r.stores = r.builder = true;
    r.wave_size = 64;
    r.vgprs = 40;
    ShaderRecord other = r;
    other.key ^= 1;
    other.stage = Stage::cs;
    other.dxil = {7};

    const std::wstring path = temp_file(mutate ? L"shader_db_check_mutate.vxdb" : L"shader_db_check.vxdb");
    const std::string manifest = "format gpu-x2w-1\nview_heap_limit 2097151\nraw_views 2097152\nmax_wave 64\nbuild abc\ncommit def\n";
    std::string error;
    check(write_db(path, {other, r}, manifest, error), "write");
    check(!write_db(path + L".dup", {r, r}, manifest, error), "a repeated key is refused");
    check(gpu_db_open(path.c_str(), 2097151, 2097152, quiet), "open");
    check(gpu_db_count() == 2, "count");
    const GpuShader* s = gpu_shader(title.data(), title.size(), Stage::ps);
    check(s && same(copy_record(*s), r), "record round trip");
    check(!gpu_shader(title.data(), title.size(), Stage::vs), "a stage mismatch is a miss");
    check(gpu_shader(title.data(), title.size(), Stage::any) == s, "Stage::any");
    check(!gpu_shader(title.data(), title.size() - 1, Stage::ps), "other bytes are a miss");
    const Manifest* m = gpu_db_manifest();
    check(m && m->view_heap_limit == 2097151 && m->raw_views == 2097152 && m->max_wave == 64 && !std::strcmp(m->build, "abc"), "manifest");
    gpu_db_close();

    // A database made for another heap layout is refused (the negative control expects it to open).
    const bool refused = !gpu_db_open(path.c_str(), mutate ? 2097151 : 1000000, 2097152, quiet);
    check(refused, "a manifest whose view_heap_limit differs is refused");
    gpu_db_close();
    std::vector<uint8_t> damaged;
    {
        FILE* f = _wfopen(path.c_str(), L"rb");
        damaged.resize(1 << 16);
        damaged.resize(std::fread(damaged.data(), 1, damaged.size(), f));
        std::fclose(f);
    }
    damaged.resize(damaged.size() - 1);
    const std::wstring cut = path + L".cut";
    FILE* f = _wfopen(cut.c_str(), L"wb");
    std::fwrite(damaged.data(), 1, damaged.size(), f);
    std::fclose(f);
    check(!gpu_db_open(cut.c_str(), 0, 0, quiet), "a truncated database is refused");
    gpu_db_close();
    DeleteFileW(cut.c_str());
    DeleteFileW(path.c_str());

    // Archive scan: an SGD5 record (raw deflate, one stored block) and a container stored as is with bytes after it.
    const std::vector<uint8_t> packed_container = container("DXIL", {4, 4, 4, 4});
    std::vector<uint8_t> file = {'S', 'G', 'D', '5', 0, 0, 0, 0};
    const std::string name = "PS_Test_0011aabb_Wrapped";
    file.push_back(0xf4);
    file.push_back(0x01);
    file.push_back(0);
    file.push_back(0);
    file.push_back(static_cast<uint8_t>(name.size() + 1));
    file.insert(file.end(), name.begin(), name.end());
    file.push_back(0);
    file.push_back(0x32);
    std::vector<uint8_t> deflate = {0x01};
    const uint16_t len = static_cast<uint16_t>(packed_container.size()), nlen = static_cast<uint16_t>(~len);
    deflate.push_back(len & 0xff);
    deflate.push_back(len >> 8);
    deflate.push_back(nlen & 0xff);
    deflate.push_back(nlen >> 8);
    deflate.insert(deflate.end(), packed_container.begin(), packed_container.end());
    const uint32_t n = static_cast<uint32_t>(deflate.size() + 12), marker = 0x000ddce7, unpacked = len;
    for (uint32_t v : {n, marker, n, unpacked}) file.insert(file.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + 4);
    file.insert(file.end(), deflate.begin(), deflate.end());
    const std::size_t raw_at = file.size();
    file.insert(file.end(), title.begin(), title.end());
    file.insert(file.end(), {0xde, 0xad, 0xbe, 0xef});
    std::vector<std::string> labels;
    std::vector<std::vector<uint8_t>> found;
    scan_bytes(file.data(), file.size(), [&](const uint8_t* b, std::size_t k, const std::string& label) {
        labels.push_back(label);
        found.emplace_back(b, b + k);
    });
    // A stored deflate block keeps its bytes as they are, so the scan finds three containers:
    // the record, its stored copy, and the following container.
    check(found.size() == 3, "three containers found");
    if (found.size() == 3) {
        check(found[0] == packed_container && labels[0].find(name) != std::string::npos, "the packed record, unpacked and named");
        check(found[1] == packed_container, "its stored bytes, as is");
        check(found[2] == title, "the container as is, cut to its size");
        char at[32];
        std::snprintf(at, sizeof(at), "@0x%zx", raw_at);
        check(labels[2] == at, "its label is its offset");
    }
    std::printf("%s\n", failures ? "shader_db_check: FAILED" : "shader_db_check: ok");
    return failures ? 1 : 0;
}

int real(const wchar_t* dir, std::size_t n)
{
    GpuExport ex;
    check(read_gpu_export(dir, ex, n), "read the export");
    check(ex.errors.empty(), "every record read");
    std::vector<ShaderRecord> records;
    for (const auto& [key, row] : ex.rows) records.push_back(row.record);
    const std::wstring path = temp_file(L"shader_db_check_real.vxdb");
    std::string error;
    check(write_db(path, records, ex.manifest, error), "write");
    check(gpu_db_open(path.c_str(), 0, 0, quiet), "open");
    const Manifest* m = gpu_db_manifest();
    if (m) std::printf("manifest: view_heap_limit %u raw_views %u max_wave %u host_rdna %u build %.16s\n", m->view_heap_limit, m->raw_views, m->max_wave,
                       m->host_rdna, m->build);
    for (const auto& r : records) {
        const GpuShader* s = gpu_shader_by_key(r.key);
        check(s && same(copy_record(*s), r), "record round trip");
        if (s)
            std::printf("%s %u dxil=%zu user_data_dwords=%u user_data=%u samplers=%u cbv=%u srv=%u uav=%u riv=%u used=%u threads=%s stores=%d/%d wave=%u\n",
                        hex(s->key).c_str(), static_cast<unsigned>(s->stage), s->dxil_bytes, s->user_data_dwords, s->user_data_count, s->sampler_count,
                        s->root_cbv_count, s->root_srv_count, s->root_uav_count, s->root_indexed_view_count, s->used_count,
                        s->has_threads ? std::to_string(s->threads[0]).c_str() : "-", s->stores, s->stores_by_address, s->wave_size);
    }
    gpu_db_close();
    DeleteFileW(path.c_str());
    std::printf("%s (%zu records)\n", failures ? "shader_db_check --export: FAILED" : "shader_db_check --export: ok", records.size());
    return failures ? 1 : 0;
}

}  // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc >= 3 && !std::wcscmp(argv[1], L"--export")) return real(argv[2], argc >= 4 ? static_cast<std::size_t>(_wtoi(argv[3])) : 5);
    return synthetic(argc >= 2 && !std::wcscmp(argv[1], L"--mutate"));
}
