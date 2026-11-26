// See gpu_export.h. A small JSON reader for the export's own files (objects, arrays, strings without
// escapes beyond \" \\ \uXXXX, numbers, true, false, null).
#include "gpu_export.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>

namespace shader_db {
namespace {

struct Value {
    enum Kind { null, boolean, number, string, array, object } kind = null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Value> items;
    std::vector<std::pair<std::string, Value>> fields;
    const Value* get(const char* name) const
    {
        for (const auto& f : fields)
            if (f.first == name) return &f.second;
        return nullptr;
    }
};

struct Parser {
    const char* p;
    const char* end;
    bool ok = true;
    void space()
    {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    bool eat(char c)
    {
        space();
        if (p < end && *p == c) {
            ++p;
            return true;
        }
        return false;
    }
    std::string text()
    {
        std::string out;
        if (!eat('"')) {
            ok = false;
            return out;
        }
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                ++p;
                if (*p == 'u' && end - p >= 5) {
                    out.push_back(static_cast<char>(std::strtoul(std::string(p + 1, p + 5).c_str(), nullptr, 16)));
                    p += 5;
                    continue;
                }
            }
            out.push_back(*p++);
        }
        ok = ok && p < end;
        ++p;
        return out;
    }
    Value value(int depth = 0)
    {
        Value v;
        space();
        if (p >= end || depth > 32) {
            ok = false;
            return v;
        }
        if (*p == '{') {
            ++p;
            v.kind = Value::object;
            if (eat('}')) return v;
            do {
                std::string name = text();
                if (!eat(':')) ok = false;
                v.fields.emplace_back(std::move(name), value(depth + 1));
            } while (ok && eat(','));
            if (!eat('}')) ok = false;
        } else if (*p == '[') {
            ++p;
            v.kind = Value::array;
            if (eat(']')) return v;
            do v.items.push_back(value(depth + 1));
            while (ok && eat(','));
            if (!eat(']')) ok = false;
        } else if (*p == '"') {
            v.kind = Value::string;
            v.s = text();
        } else if (!std::strncmp(p, "true", 4) || !std::strncmp(p, "false", 5)) {
            v.kind = Value::boolean;
            v.b = *p == 't';
            p += v.b ? 4 : 5;
        } else if (!std::strncmp(p, "null", 4)) {
            p += 4;
        } else {
            char* stop = nullptr;
            v.kind = Value::number;
            v.n = std::strtod(p, &stop);
            if (stop == p) ok = false;
            p = stop;
        }
        return v;
    }
};

struct Reader {
    const Value& root;
    std::string why;
    const Value* field(const Value& v, const char* name, Value::Kind kind)
    {
        const Value* f = v.get(name);
        if (!f || f->kind != kind) {
            if (why.empty()) why = std::string("field ") + name;
            return nullptr;
        }
        return f;
    }
    uint32_t u(const Value& v, const char* name)
    {
        const Value* f = field(v, name, Value::number);
        return f ? static_cast<uint32_t>(f->n) : 0;
    }
    float f(const Value& v, const char* name)
    {
        const Value* x = field(v, name, Value::number);
        return x ? static_cast<float>(x->n) : 0.0f;
    }
    bool b(const Value& v, const char* name)
    {
        const Value* x = field(v, name, Value::boolean);
        return x && x->b;
    }
    const std::vector<Value>& list(const Value& v, const char* name)
    {
        static const std::vector<Value> none;
        const Value* x = field(v, name, Value::array);
        return x ? x->items : none;
    }
};

uint32_t kind_code(const std::string& name, const char* const* names, uint32_t count, std::string& why)
{
    for (uint32_t i = 0; i < count; ++i)
        if (name == names[i]) return i;
    if (name.rfind("Other(", 0) == 0) return count + static_cast<uint32_t>(std::strtoul(name.c_str() + 6, nullptr, 10));
    why = "kind " + name;
    return 0;
}

const char* const root_names[] = {"Constants", "Descriptor", "Table", "ImmediateSampler", "ResourceHeap", "SamplerHeap"};
const char* const resource_names[] = {"Srv", "Sampler", "Cbv", "Uav"};
const char* const stage_names[] = {"Ls", "Hs", "Es", "Gs", "Vs", "Ps", "Ms", "Cs"};

bool read_file(const std::wstring& path, std::string& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream s;
    s << f.rdbuf();
    out = s.str();
    return true;
}

}  // namespace

bool record_from_json(const std::string& json, std::vector<uint8_t> dxil, ShaderRecord& out, std::string& why)
{
    Parser parser{json.data(), json.data() + json.size()};
    const Value root = parser.value();
    if (!parser.ok || root.kind != Value::object) {
        why = "not JSON";
        return false;
    }
    Reader r{root, {}};
    ShaderRecord s;
    const Value* key = r.field(root, "key", Value::string);
    const Value* fnv = r.field(root, "container_fnv64", Value::string);
    const Value* stage = r.field(root, "stage", Value::string);
    if (key) s.key = std::strtoull(key->s.c_str(), nullptr, 16);
    if (fnv) s.fnv64 = std::strtoull(fnv->s.c_str(), nullptr, 16);
    if (stage) s.stage = static_cast<Stage>(kind_code(stage->s, stage_names, 8, r.why));
    if (stage && static_cast<uint32_t>(s.stage) > 7 && r.why.empty()) r.why = "stage";
    s.container_bytes = r.u(root, "container_bytes");
    if (r.u(root, "dxil_bytes") != dxil.size() && r.why.empty()) r.why = "dxil_bytes disagrees with the .dxil file";
    s.dxil = std::move(dxil);
    s.user_data_dwords = r.u(root, "user_data_dwords");
    for (const Value& u : r.list(root, "user_data")) {
        const Value* root_kind = r.field(u, "root", Value::string);
        s.user_data.push_back({r.u(u, "index"), r.u(u, "slot"), root_kind ? kind_code(root_kind->s, root_names, 6, r.why) : 0, r.u(u, "dword")});
    }
    for (const Value& w : r.list(root, "samplers")) {
        std::array<uint32_t, 4> words{};
        if (w.items.size() != 4 && r.why.empty()) r.why = "sampler words";
        for (std::size_t i = 0; i < 4 && i < w.items.size(); ++i) words[i] = static_cast<uint32_t>(w.items[i].n);
        s.samplers.push_back(words);
    }
    for (const Value& t : r.list(root, "static_samplers")) {
        VxStaticSampler x{};
        x.filter = r.u(t, "filter");
        const auto& address = r.list(t, "address");
        for (std::size_t i = 0; i < 3 && i < address.size(); ++i) x.address[i] = static_cast<uint32_t>(address[i].n);
        x.mip_lod_bias = r.f(t, "mip_lod_bias");
        x.max_anisotropy = r.u(t, "max_anisotropy");
        x.comparison = r.u(t, "comparison");
        x.border = r.u(t, "border");
        x.min_lod = r.f(t, "min_lod");
        x.max_lod = r.f(t, "max_lod");
        s.static_samplers.push_back(x);
    }
    for (const Value& v : r.list(root, "root_cbvs")) s.root_cbvs.push_back(static_cast<uint8_t>(v.n));
    for (const Value& v : r.list(root, "root_srvs")) s.root_srvs.push_back(static_cast<uint8_t>(v.n));
    for (const Value& v : r.list(root, "root_uavs")) s.root_uavs.push_back(static_cast<uint8_t>(v.n));
    for (const Value& v : r.list(root, "root_indexed_views"))
        s.root_indexed_views.push_back({r.u(v, "table_slot"), r.u(v, "constant_slot"), r.u(v, "constant_dword"), r.u(v, "byte_offset"), r.u(v, "shift")});
    for (const Value& u : r.list(root, "used")) {
        const Value* kind = r.field(u, "kind", Value::string);
        const Value* root_kind = r.field(u, "root", Value::string);
        s.used.push_back({kind ? kind_code(kind->s, resource_names, 4, r.why) : 0, root_kind ? kind_code(root_kind->s, root_names, 6, r.why) : 0,
                          r.u(u, "table_offset_dwords"), r.u(u, "lower_bound"), r.u(u, "stride"), r.u(u, "dwords"), r.u(u, "user_data"),
                          r.u(u, "slot")});
    }
    const Value* threads = root.get("threads");
    if (threads && threads->kind == Value::array && threads->items.size() == 3) {
        s.has_threads = true;
        for (int i = 0; i < 3; ++i) s.threads[i] = static_cast<uint32_t>(threads->items[i].n);
    }
    s.stores = r.b(root, "stores");
    s.stores_by_address = r.b(root, "stores_by_address");
    s.builder = r.b(root, "builder");
    if (const Value* native = root.get("derived_native"); native && native->kind == Value::object) {
        s.wave_size = r.u(*native, "wave_size");
        s.vgprs = r.u(*native, "vgprs");
    }
    if (!r.why.empty()) {
        why = r.why;
        return false;
    }
    out = std::move(s);
    return true;
}

bool read_gpu_export(const std::wstring& dir, GpuExport& out, std::size_t limit)
{
    std::string index;
    if (!read_file(dir + L"\\index.tsv", index) || !read_file(dir + L"\\manifest.txt", out.manifest)) return false;
    std::istringstream lines(index);
    std::string line;
    std::getline(lines, line);  // header
    while (std::getline(lines, line) && (!limit || out.rows.size() < limit)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::vector<std::string> cols;
        for (std::size_t at = 0;;) {
            const std::size_t tab = line.find('\t', at);
            cols.push_back(line.substr(at, tab == std::string::npos ? std::string::npos : tab - at));
            if (tab == std::string::npos) break;
            at = tab + 1;
        }
        if (cols.size() < 2) continue;
        const std::wstring stem = dir + L"\\" + std::wstring(cols[0].begin(), cols[0].end());
        std::string json, dxil;
        std::string why;
        ExportRow row;
        if (!read_file(stem + L".json", json) || !read_file(stem + L".dxil", dxil)) why = "missing file";
        else if (record_from_json(json, std::vector<uint8_t>(dxil.begin(), dxil.end()), row.record, why)) {
            row.first_source = cols.back();
            const uint64_t key = row.record.key;
            out.rows.emplace(key, std::move(row));
            continue;
        }
        out.errors.push_back(cols[0] + ": " + why);
    }
    return true;
}

}  // namespace shader_db
