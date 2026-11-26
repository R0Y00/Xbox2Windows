// See shader_db_writer.h.
#include "shader_db_writer.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace shader_db {

uint64_t fnv64(const void* bytes, std::size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ static_cast<const unsigned char*>(bytes)[i]) * 1099511628211ull;
    return h;
}

ShaderRecord copy_record(const GpuShader& s)
{
    ShaderRecord r;
    r.key = s.key;
    r.fnv64 = s.fnv64;
    r.stage = s.stage;
    r.container_bytes = s.container_bytes;
    const auto* dxil = static_cast<const uint8_t*>(s.dxil);
    r.dxil.assign(dxil, dxil + s.dxil_bytes);
    r.user_data_dwords = s.user_data_dwords;
    r.user_data.assign(s.user_data, s.user_data + s.user_data_count);
    for (uint32_t i = 0; i < s.sampler_count; ++i) r.samplers.push_back({s.samplers[i][0], s.samplers[i][1], s.samplers[i][2], s.samplers[i][3]});
    r.static_samplers.assign(s.static_samplers, s.static_samplers + s.static_sampler_count);
    r.root_cbvs.assign(s.root_cbvs, s.root_cbvs + s.root_cbv_count);
    r.root_srvs.assign(s.root_srvs, s.root_srvs + s.root_srv_count);
    r.root_uavs.assign(s.root_uavs, s.root_uavs + s.root_uav_count);
    r.root_indexed_views.assign(s.root_indexed_views, s.root_indexed_views + s.root_indexed_view_count);
    r.used.assign(s.used, s.used + s.used_count);
    r.has_threads = s.has_threads;
    std::memcpy(r.threads, s.threads, sizeof(r.threads));
    r.stores = s.stores;
    r.stores_by_address = s.stores_by_address;
    r.builder = s.builder;
    r.wave_size = s.wave_size;
    r.vgprs = s.vgprs;
    return r;
}

namespace {

struct Out {
    std::vector<uint8_t> bytes;
    void align(std::size_t a)
    {
        while (bytes.size() % a) bytes.push_back(0);
    }
    uint64_t put(const void* data, std::size_t n, std::size_t a)
    {
        align(a);
        const uint64_t at = bytes.size();
        const auto* p = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), p, p + n);
        return at;
    }
    template <class T>
    uint64_t put(const std::vector<T>& v, std::size_t a = 8)
    {
        return put(v.data(), v.size() * sizeof(T), a);
    }
};

}  // namespace

bool write_db(const std::wstring& path, std::vector<ShaderRecord> records, const std::string& manifest, std::string& error)
{
    std::sort(records.begin(), records.end(), [](const ShaderRecord& a, const ShaderRecord& b) { return a.key < b.key; });
    for (std::size_t i = 1; i < records.size(); ++i)
        if (records[i - 1].key == records[i].key) {
            char text[64];
            std::snprintf(text, sizeof(text), "key %016llx twice", static_cast<unsigned long long>(records[i].key));
            error = text;
            return false;
        }
    Out out;
    out.bytes.resize(sizeof(VxHeader) + records.size() * sizeof(VxIndex));
    std::vector<VxIndex> index(records.size());
    for (std::size_t i = 0; i < records.size(); ++i) {
        const ShaderRecord& s = records[i];
        if (s.dxil.size() > 0xffffffffu) {
            error = "a DXIL container over 4 GiB";
            return false;
        }
        out.align(8);
        const uint64_t at = out.bytes.size();
        out.bytes.resize(at + sizeof(VxRecord));
        VxRecord r{};
        r.key = s.key;
        r.fnv64 = s.fnv64;
        r.stage = static_cast<uint32_t>(s.stage);
        r.container_bytes = s.container_bytes;
        r.dxil_offset = out.put(s.dxil, 4);
        r.dxil_bytes = static_cast<uint32_t>(s.dxil.size());
        r.user_data_dwords = s.user_data_dwords;
        r.user_data_count = static_cast<uint32_t>(s.user_data.size());
        r.sampler_count = static_cast<uint32_t>(s.samplers.size());
        r.static_sampler_count = static_cast<uint32_t>(s.static_samplers.size());
        r.root_cbv_count = static_cast<uint32_t>(s.root_cbvs.size());
        r.root_srv_count = static_cast<uint32_t>(s.root_srvs.size());
        r.root_uav_count = static_cast<uint32_t>(s.root_uavs.size());
        r.root_indexed_view_count = static_cast<uint32_t>(s.root_indexed_views.size());
        r.used_count = static_cast<uint32_t>(s.used.size());
        r.user_data_offset = out.put(s.user_data);
        r.samplers_offset = out.put(s.samplers);
        r.static_samplers_offset = out.put(s.static_samplers);
        r.root_cbvs_offset = out.put(s.root_cbvs, 1);
        r.root_srvs_offset = out.put(s.root_srvs, 1);
        r.root_uavs_offset = out.put(s.root_uavs, 1);
        r.root_indexed_views_offset = out.put(s.root_indexed_views);
        r.used_offset = out.put(s.used);
        std::memcpy(r.threads, s.threads, sizeof(r.threads));
        r.flags = (s.stores ? flag_stores : 0) | (s.stores_by_address ? flag_stores_by_address : 0) | (s.builder ? flag_builder : 0) |
                  (s.has_threads ? flag_threads : 0);
        r.wave_size = s.wave_size;
        r.vgprs = s.vgprs;
        std::memcpy(out.bytes.data() + at, &r, sizeof(r));
        index[i] = {s.key, at};
    }
    VxHeader h{};
    std::memcpy(h.magic, "VXDB0001", 8);
    h.version = 1;
    h.record_count = static_cast<uint32_t>(records.size());
    h.index_offset = sizeof(VxHeader);
    h.manifest_offset = out.put(manifest.data(), manifest.size(), 1);
    h.manifest_bytes = static_cast<uint32_t>(manifest.size());
    h.file_bytes = out.bytes.size();
    std::memcpy(out.bytes.data(), &h, sizeof(h));
    if (!index.empty()) std::memcpy(out.bytes.data() + sizeof(VxHeader), index.data(), index.size() * sizeof(VxIndex));

    const std::wstring temporary = path + L".partial";
    HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "cannot create the database file";
        return false;
    }
    bool ok = true;
    for (std::size_t done = 0; ok && done < out.bytes.size();) {
        const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(out.bytes.size() - done, 1u << 30));
        DWORD written = 0;
        ok = WriteFile(file, out.bytes.data() + done, chunk, &written, nullptr) && written == chunk;
        done += chunk;
    }
    ok = CloseHandle(file) && ok;
    if (!ok || !MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileW(temporary.c_str());
        error = "cannot write the database file";
        return false;
    }
    return true;
}

}  // namespace shader_db
